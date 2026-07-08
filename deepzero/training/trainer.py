import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable

from deepzero.models.transformer import GPT
from deepzero.models.checkpoints import save_checkpoint, try_resume, CHECKPOINT_NAMES
from deepzero.training.metrics import compute_metrics
from deepzero.training.optimizer import build_optimizer
from deepzero.training.dashboard import render_dashboard


def _memory_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def _cpu_percent() -> float:
    try:
        with open("/proc/stat") as f:
            line = f.readline().strip().split()
        if len(line) >= 5:
            total = sum(int(v) for v in line[1:])
            return min(100.0, total / 1e6)
    except Exception:
        pass
    return 0.0


class Trainer:
    def __init__(self, model: GPT, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 lr: float = 3e-4, weight_decay: float = 0.1, warmup_iters: int = 100,
                 max_iters: int = 5000, log_interval: int = 10, eval_interval: int = 100,
                 checkpoint_dir: str = "checkpoints", device: str = "cpu",
                 grad_clip: float = 1.0, compile_model: bool = False,
                 gradient_accumulation: int = 1,
                 checkpoint_time_minutes: float = 10.0,
                 callbacks: Optional[list[Callable]] = None,
                 optimizer_name: str = "adamw",
                 optimizer_kwargs: Optional[dict] = None,
                 metrics_tracker: Optional["MetricsTracker"] = None,
                 early_stop_patience: Optional[int] = None,
                 tokenizer_name: str = "",
                 eval_after_training: bool = False):
        self.device = device
        self.max_iters = max_iters
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_time_minutes = checkpoint_time_minutes
        self.callbacks = callbacks or []
        self.optimizer_name = optimizer_name
        self.tokenizer_name = tokenizer_name
        self.early_stop_patience = early_stop_patience
        self.eval_after_training = eval_after_training

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = self._build_optimizer(lr, weight_decay)
        self.scheduler = self._build_scheduler(lr, warmup_iters, max_iters)

        self.step = 0
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")
        self.best_tok_speed = 0.0
        self._last_log_time = time.time()
        self._last_ckpt_time = time.time()
        self._profile_data = []
        self._last_loss = float("inf")
        self._last_tok_speed = 0.0
        self._last_ram = 0.0
        self._last_lr = 0.0
        self._last_grad_norm = 0.0
        self._patience_counter = 0
        self._start_time = time.time()
        self.metrics_tracker = metrics_tracker
        if self.metrics_tracker is not None:
            self.metrics_tracker.set_max_steps(max_iters)

        ckpt_dir = os.path.dirname(self.checkpoint_dir) if self.checkpoint_dir.endswith(".pt") else self.checkpoint_dir
        resumed_step = try_resume(ckpt_dir, model, device, self.optimizer, self.scheduler, restore_rng=True)
        if resumed_step:
            self.step = resumed_step

        self.model = self._init_model(model, compile_model)

    def _init_model(self, model, compile_model):
        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)
        return model.to(self.device)

    def _build_optimizer(self, lr: float, wd: float):
        kw: dict = {"lr": lr, "weight_decay": wd}
        kw.update(self.optimizer_kwargs)
        return build_optimizer(self.model, self.optimizer_name, **kw)

    def _build_scheduler(self, lr: float, warmup: int, total: int):
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            return max(0.1, 1.0 - (step - warmup) / max(1, total - warmup))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _compute_grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
        return total ** 0.5

    def _profile(self, loss: float, samples: int, tokens: int, elapsed: float) -> dict:
        info = {
            "loss": loss,
            "samples_per_sec": samples / max(elapsed, 1e-8),
            "tokens_per_sec": tokens / max(elapsed, 1e-8),
            "ram_mb": _memory_mb(),
            "cpu_util": _cpu_percent(),
        }
        if self.device == "cuda" and torch.cuda.is_available():
            info["gpu_util"] = torch.cuda.utilization()
            info["vram_mb"] = torch.cuda.memory_allocated() / 1e6
        self._profile_data.append(info)
        return info

    def _save_auto_checkpoints(self, val_loss: Optional[float] = None):
        ckpt_dir = self.checkpoint_dir
        extra = {"optimizer": self.optimizer.state_dict()}

        latest_path = os.path.join(ckpt_dir, "latest.pt")
        save_checkpoint(self.model, latest_path, self.step, self._last_loss, tag="latest", extra=extra)

        if self._last_loss < self.best_train_loss:
            self.best_train_loss = self._last_loss
            save_checkpoint(self.model, os.path.join(ckpt_dir, "best_loss.pt"),
                            self.step, self._last_loss, tag="best_loss", extra=extra)

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_checkpoint(self.model, os.path.join(ckpt_dir, "best_val.pt"),
                            self.step, val_loss, tag="best_val", extra=extra)

        if self._last_tok_speed > self.best_tok_speed:
            self.best_tok_speed = self._last_tok_speed
            save_checkpoint(self.model, os.path.join(ckpt_dir, "fastest.pt"),
                            self.step, self._last_loss, tag="fastest", extra=extra)

    def _maybe_time_checkpoint(self):
        now = time.time()
        if (now - self._last_ckpt_time) >= self.checkpoint_time_minutes * 60:
            self._save_auto_checkpoints()
            self._last_ckpt_time = now

    def train(self):
        if self.step >= self.max_iters:
            print(f"Already at step {self.step}/{self.max_iters}, nothing to do.")
            return
        print(f"Training {sum(p.numel() for p in self.model.parameters()):,} param model "
              f"for {self.max_iters} steps (resuming from step {self.step})", flush=True)

        self.model.train()
        total_loss = 0.0
        start = time.time()
        self._start_time = time.time()
        accumulation_steps = self.gradient_accumulation
        self.optimizer.zero_grad()
        data_iter = iter(self.train_loader)
        _last_logged = -1
        _dashboard_shown = False

        while self.step < self.max_iters:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            loss = loss / accumulation_steps
            loss.backward()

            self._batch_count = getattr(self, "_batch_count", 0) + 1

            if self._batch_count % accumulation_steps == 0:
                self._last_grad_norm = self._compute_grad_norm()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.step += 1

            total_loss += loss.item() * accumulation_steps

            log_now = self.step > 0 and self.step % self.log_interval == 0 and self.step != _last_logged
            if log_now:
                _last_logged = self.step
                elapsed = time.time() - start
                samples = self.log_interval * x.size(0) * accumulation_steps
                tokens = self.log_interval * x.size(1) * x.size(0) * accumulation_steps
                prof = self._profile(loss.item() * accumulation_steps, samples, tokens, elapsed)

                lr_now = self.scheduler.get_last_lr()[0]
                self._last_loss = loss.item() * accumulation_steps
                self._last_tok_speed = prof['tokens_per_sec']
                self._last_ram = prof['ram_mb']
                self._last_lr = lr_now

                val_loss, val_ppl = None, None
                if self.val_loader and self.step % self.eval_interval == 0:
                    vstart = time.time()
                    metrics = compute_metrics(self.model, self.val_loader, self.device)
                    val_loss = metrics["val_loss"]
                    val_ppl = math.exp(val_loss) if val_loss is not None and val_loss < 100 else None
                    self._save_auto_checkpoints(val_loss)

                    # Early stopping
                    if self.early_stop_patience is not None:
                        if val_loss >= self.best_val_loss:
                            self._patience_counter += 1
                        else:
                            self._patience_counter = 0
                else:
                    self._save_auto_checkpoints()

                if self.metrics_tracker is not None:
                    self.metrics_tracker.record(
                        step=self.step, loss=self._last_loss,
                        validation_loss=val_loss, perplexity=val_ppl,
                        learning_rate=lr_now, gradient_norm=self._last_grad_norm,
                        tokens_per_second=prof['tokens_per_sec'],
                        samples_per_second=prof['samples_per_sec'],
                        peak_ram=prof['ram_mb'], current_ram=prof['ram_mb'],
                    )

                remaining_patience = None
                if self.early_stop_patience is not None:
                    remaining_patience = max(0, self.early_stop_patience - self._patience_counter)

                dashboard = render_dashboard(
                    step=self.step, max_steps=self.max_iters,
                    loss=self._last_loss, val_loss=val_loss, perplexity=val_ppl,
                    grad_norm=self._last_grad_norm, lr=lr_now,
                    opt_name=self.optimizer_name, tok_name=self.tokenizer_name,
                    tok_speed=prof['tokens_per_sec'], samp_speed=prof['samples_per_sec'],
                    ram_mb=prof['ram_mb'], elapsed=time.time() - self._start_time,
                    best_val_loss=self.best_val_loss if self.best_val_loss != float("inf") else None,
                    patience_remaining=remaining_patience,
                )
                print(f"\033[2J\033[H{dashboard}")  # clear screen + render
                _dashboard_shown = True
                start = time.time()
                self._maybe_time_checkpoint()
                for cb in self.callbacks:
                    cb(self)

            # Early stopping check
            if (self.early_stop_patience is not None
                    and self._patience_counter >= self.early_stop_patience):
                print(f"\nEarly stopping triggered at step {self.step} "
                      f"(val_loss didn't improve for {self.early_stop_patience} evals)")
                break

        if _dashboard_shown:
            print(f"\033[2J\033[H", end="")  # final clear

        self._save_auto_checkpoints()
        final_path = os.path.join(self.checkpoint_dir, "final.pt")
        save_checkpoint(self.model, final_path, self.step, self._last_loss,
                        tag="final", extra={"optimizer": self.optimizer.state_dict()})
        print(f"\nTraining complete. Final checkpoint: {final_path}")
