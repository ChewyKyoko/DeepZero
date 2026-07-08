import csv
import json
import os
import time
from pathlib import Path
from typing import Optional


class MetricsTracker:
    """Records per-step training metrics and exports to CSV + JSON.

    Tracks loss, validation loss, perplexity, learning rate, gradient norm,
    throughput, RAM, and time-to-target for configurable loss thresholds.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self._records: list[dict] = []
        self._start_time = time.time()
        self._peak_ram = 0.0
        self._last_step_time: Optional[float] = None
        self.targets: dict[float, dict | None] = {
            4.0: None, 3.5: None, 3.0: None, 2.5: None, 2.0: None,
        }

    @property
    def records(self) -> list[dict]:
        return list(self._records)

    def record(self, *, step: int, epoch: float = 0.0,
               loss: Optional[float] = None,
               validation_loss: Optional[float] = None,
               perplexity: Optional[float] = None,
               learning_rate: Optional[float] = None,
               gradient_norm: Optional[float] = None,
               tokens_per_second: Optional[float] = None,
               samples_per_second: Optional[float] = None,
               peak_ram: Optional[float] = None,
               current_ram: Optional[float] = None):
        now = time.time()
        elapsed = now - self._start_time

        if peak_ram is not None:
            self._peak_ram = max(self._peak_ram, peak_ram)

        remaining = None
        if step > 0 and self._last_step_time is not None:
            avg_step_time = elapsed / step
            remaining = avg_step_time * (self._max_steps - step) if hasattr(self, '_max_steps') else None

        rec = {
            "step": step,
            "epoch": round(epoch, 2),
            "loss": loss,
            "validation_loss": validation_loss,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "tokens_per_second": tokens_per_second,
            "samples_per_second": samples_per_second,
            "elapsed_time": round(elapsed, 1),
            "remaining_time": round(remaining, 1) if remaining else None,
            "peak_ram": self._peak_ram,
            "current_ram": current_ram,
        }
        self._records.append(rec)
        self._last_step_time = now

        self._check_targets(loss, step, elapsed)

    def set_max_steps(self, n: int):
        self._max_steps = n

    def _check_targets(self, loss: Optional[float], step: int, elapsed: float):
        if loss is None:
            return
        for target in sorted(self.targets, reverse=True):
            if loss <= target and self.targets[target] is None:
                self.targets[target] = {"step": step, "elapsed": round(elapsed, 1)}

    def time_to_target(self) -> dict:
        return {str(k): v for k, v in self.targets.items() if v is not None}

    def to_csv(self, path: str | Path):
        fieldnames = ["step", "epoch", "loss", "validation_loss", "perplexity",
                       "learning_rate", "gradient_norm", "tokens_per_second",
                       "samples_per_second", "elapsed_time", "remaining_time",
                       "peak_ram", "current_ram"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            if self._records:
                w.writerows(self._records)

    def to_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2)

    def summary(self) -> dict:
        losses = [r["loss"] for r in self._records if r["loss"] is not None]
        val_losses = [r["validation_loss"] for r in self._records if r["validation_loss"] is not None]
        tok_speeds = [r["tokens_per_second"] for r in self._records if r["tokens_per_second"] is not None]

        return {
            "total_steps": len(self._records),
            "final_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "avg_tokens_per_second": sum(tok_speeds) / len(tok_speeds) if tok_speeds else 0,
            "total_time_sec": round(time.time() - self._start_time, 1),
            "peak_ram_gb": round(self._peak_ram / 1024, 2),
            "time_to_target": self.time_to_target(),
        }
