import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable

from deepzero.models.transformer import GPT
from deepzero.training.metrics import compute_metrics
from deepzero.models.checkpoints import save_checkpoint


class Trainer:
    def __init__(self, model: GPT, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 lr: float = 3e-4, weight_decay: float = 0.1, warmup_iters: int = 100,
                 max_iters: int = 5000, log_interval: int = 10, eval_interval: int = 100,
                 checkpoint_dir: str = "checkpoints", device: str = "cpu",
                 grad_clip: float = 1.0, compile_model: bool = False,
                 callbacks: Optional[list[Callable]] = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_iters = max_iters
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.callbacks = callbacks or []
        self.optimizer = self._build_optimizer(lr, weight_decay)
        self.scheduler = self._build_scheduler(lr, warmup_iters, max_iters)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        self.step = 0
        self.best_val_loss = float("inf")

    def _build_optimizer(self, lr: float, wd: float):
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        return torch.optim.AdamW([
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.95), fused=False)

    def _build_scheduler(self, lr: float, warmup: int, total: int):
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            return max(0.1, 1.0 - (step - warmup) / max(1, total - warmup))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        start = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                _, loss = self.model(x, y)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            total_loss += loss.item()
            self.step += 1
            if self.step % self.log_interval == 0:
                elapsed = time.time() - start
                tok_per_sec = self.log_interval * x.size(1) * x.size(0) / max(elapsed, 1e-6)
                metrics = compute_metrics(self.model, self.val_loader, self.device) if self.val_loader and self.step % self.eval_interval == 0 else None
                lr_now = self.scheduler.get_last_lr()[0]
                msg = f"step {self.step}/{self.max_iters} | loss {loss.item():.4f} | lr {lr_now:.2e} | tok/s {tok_per_sec:.0f}"
                if metrics:
                    msg += f" | val_loss {metrics['val_loss']:.4f}"
                    if metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = metrics["val_loss"]
                        save_checkpoint(self.model, os.path.join(self.checkpoint_dir, "best.pt"), self.step, metrics["val_loss"])
                print(msg)
                start = time.time()
                for cb in self.callbacks:
                    cb(self)
            if self.step >= self.max_iters:
                break
        return total_loss / max(1, self.step)

    def train(self):
        print(f"Training {self.model.n_params:,} param model for {self.max_iters} steps")
        self.train_epoch()
        save_checkpoint(self.model, os.path.join(self.checkpoint_dir, "final.pt"), self.step)
        print("Training complete")
