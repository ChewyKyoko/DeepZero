import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from config import Config
from tokenizer import BPETokenizer
from model import GPT
from dataset import TextDataset


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def _get_lr(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return c.learning_rate * (step + 1) / c.warmup_steps
        progress = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        return c.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self, model: GPT, dataset: TextDataset, tokenizer: BPETokenizer, run_dir: str = "runs/default"):
        os.makedirs(run_dir, exist_ok=True)
        import json, dataclasses
        keys = [k for k in dir(self.cfg) if not k.startswith("_")]
        cfg_dict = {k: getattr(self.cfg, k) for k in keys if isinstance(getattr(self.cfg, k), (int, float, str, bool, list, tuple))}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)
        model = model.to(self.device)

        pin = self.device.type == "cuda"
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True,
            pin_memory=pin, drop_last=True,
        )
        it = iter(loader)

        optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
        )

        best_loss = float("inf")
        tokens_seen = 0
        start_time = time.time()

        for step in range(1, self.cfg.max_steps + 1):
            lr = self._get_lr(step - 1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)

            x, y = x.to(self.device), y.to(self.device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimizer.step()

            tokens_seen += x.numel()

            if step % self.cfg.log_interval == 0:
                elapsed = time.time() - start_time
                tok_sec = tokens_seen / max(1, elapsed)
                ppl = math.exp(min(loss.item(), 20))
                print(f"step {step:>6d} | loss {loss.item():.4f} | ppl {ppl:.2f} | "
                      f"lr {lr:.2e} | tok/s {tok_sec:.0f} | elapsed {elapsed:.0f}s")

            if step % self.cfg.eval_interval == 0:
                val_loss = self._evaluate(model, dataset)
                print(f"  eval  | loss {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.2f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    ckpt = {
                        "step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "loss": val_loss,
                    }
                    torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                    print(f"  saved best checkpoint (loss {val_loss:.4f})")

        ckpt = {
            "step": self.cfg.max_steps, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "loss": loss.item(),
        }
        torch.save(ckpt, os.path.join(run_dir, "final.pt"))
        print(f"\nDone. Best loss: {best_loss:.4f}")
        return model

    def _evaluate(self, model: GPT, dataset: TextDataset) -> float:
        model.eval()
        pin = self.device.type == "cuda"
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True,
            pin_memory=pin, drop_last=True,
        )
        losses = []
        count = 0
        with torch.no_grad():
            for x, y in loader:
                if count >= self.cfg.eval_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                _, loss = model(x, y)
                losses.append(loss.item())
                count += 1
        model.train()
        return sum(losses) / max(1, len(losses))
