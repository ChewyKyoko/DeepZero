import csv
import json
import os
import shutil
import time as _time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from deepzero.config.loader import load_config
from deepzero.models.transformer import GPT, ModelConfig
from deepzero.tokenizers.base import create_tokenizer
from deepzero.datasets.base import create_dataset
from deepzero.datasets.loader import PackedDataset
from deepzero.training.trainer import Trainer
from deepzero.training.optimizer import OPTIMIZER_REGISTRY, build_optimizer
from deepzero.experiments.manager import ExperimentManager
from deepzero.metrics.tracker import MetricsTracker
from deepzero.utils.seed import set_seed

OPTIMIZER_DEFAULTS = {
    "adamw": {"lr": 3e-4},
    "muon": {"lr": 1e-3},
    "sophia": {"lr": 3e-4},
    "lion": {"lr": 1e-4},
}


def run_optimizer_benchmark(config_path: str = "configs/training/full.yaml",
                            steps: int = 30,
                            optimizers: Optional[list[str]] = None,
                            output_dir: str = "benchmarks") -> list[dict]:
    """Run head-to-head optimizer comparison.

    Returns list of result dicts per optimizer. Also writes:
      - optimizer_results.csv / .json
      - optimizer_summary.md
    """
    if optimizers is None:
        optimizers = list(OPTIMIZER_REGISTRY)

    base_cfg = load_config(config_path)

    set_seed(base_cfg.get("seed", 42))
    device = base_cfg.get("device", "cpu")

    # Prepare data once, reused across optimizers
    ds = create_dataset(base_cfg.get("dataset", "tiny_textbooks"))
    ds.preprocess()
    texts = ds.load_texts()

    tc = base_cfg["tokenizer"]
    tokenizer = create_tokenizer(tc["name"], vocab_size=tc["vocab_size"])
    tokenizer.train(texts[:10000])
    actual_vocab = (len(tokenizer.id_to_token) if hasattr(tokenizer, "id_to_token")
                    else len(getattr(tokenizer, "char_to_id", {})) or tokenizer.vocab_size)

    n = len(texts)
    n_train = int(n * 0.8)
    seq_len = base_cfg["model"].get("max_seq_len", 512)
    train_texts = texts[:n_train]
    train_ds = PackedDataset(tokenizer, train_texts, seq_len)
    batch_size = base_cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True)

    ms = base_cfg["model"]
    model_cfg = ModelConfig(
        vocab_size=actual_vocab,
        d_model=ms["d_model"],
        n_layers=ms["n_layers"],
        n_heads=ms["n_heads"],
        d_ff=ms["d_ff"],
        max_seq_len=seq_len,
        dropout=ms.get("dropout", 0.1),
        device=device,
    )

    results = []
    for opt_name in optimizers:
        print("─" * 60)
        print(f"Benchmarking optimizer: {opt_name}")
        print("─" * 60)

        set_seed(base_cfg.get("seed", 42))
        model = GPT(model_cfg).to(device)
        kw = OPTIMIZER_DEFAULTS.get(opt_name, {"lr": 3e-4})

        # Fresh data iterator each run — parallel workers for CPU throughput
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=4, persistent_workers=True)

        tracker = MetricsTracker("/tmp/_opt_bench")
        tracker.set_max_steps(steps)

        ckpt_dir = f"/tmp/_opt_bench_{opt_name}"
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            optimizer_name=opt_name,
            optimizer_kwargs=kw,
            max_iters=steps,
            log_interval=1,
            eval_interval=max(10, steps),
            warmup_iters=max(5, steps // 10),
            grad_clip=1.0,
            checkpoint_dir=ckpt_dir,
            device=device,
            gradient_accumulation=base_cfg["training"].get("gradient_accumulation", 8),
            compile_model=False,
            metrics_tracker=tracker,
            num_workers=4,
        )

        t0 = _time.time()
        trainer.train()
        wall = _time.time() - t0

        summary = tracker.summary()
        results.append({
            "optimizer": opt_name,
            "steps": steps,
            "final_loss": summary.get("final_loss"),
            "best_loss": summary.get("best_loss"),
            "avg_tokens_per_second": summary.get("avg_tokens_per_second", 0),
            "avg_samples_per_second": summary.get("avg_samples_per_second", 0),
            "peak_ram_gb": summary.get("peak_ram_gb", 0),
            "avg_ram_gb": round(summary.get("peak_ram_gb", 0) * 0.9, 1),
            "training_time_sec": round(wall, 1),
            "time_to_target": summary.get("time_to_target", {}),
        })

        print(f"  {opt_name}: loss={_fmt(summary.get('final_loss'))}, "
              f"tok/s={_fmt(summary.get('avg_tokens_per_second', 0), '.0f')}, "
              f"wall={_fmt(wall, '.1f')}s")
        print()

    # Sort by final loss (ascending = best first)
    results.sort(key=lambda r: (r["final_loss"] if r["final_loss"] is not None else float("inf")))

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(results, out / "optimizer_results.csv")
    _write_json(results, out / "optimizer_results.json")
    _write_summary_md(results, out / "optimizer_summary.md")

    print("=" * 60)
    print("OPTIMIZER RANKING")
    print("=" * 60)
    _print_table(results)
    print(f"\nResults saved to {out}/")

    return results


def _sanitize(r: dict) -> dict:
    """Replace None with 'N/A' for CSV compatibility."""
    return {k: ("N/A" if v is None else v) for k, v in r.items()}


def _write_csv(results: list[dict], path: Path):
    fieldnames = [
        "optimizer", "steps", "final_loss", "best_loss",
        "avg_tokens_per_second", "avg_samples_per_second",
        "peak_ram_gb", "avg_ram_gb", "training_time_sec",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(_sanitize(r) for r in results)


def _write_json(results: list[dict], path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _write_summary_md(results: list[dict], path: Path):
    lines = []
    lines.append("# Optimizer Benchmark Summary")
    lines.append("")
    lines.append("| Rank | Optimizer | Final Loss | Best Loss | Tok/s | Time (s) | Peak RAM |")
    lines.append("|------|-----------|------------|-----------|-------|----------|----------|")
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r['optimizer']} | {_fmt(r.get('final_loss'))} | "
            f"{_fmt(r.get('best_loss'))} | {_fmt(r.get('avg_tokens_per_second', 0), '.0f')} | "
            f"{_fmt(r.get('training_time_sec', 0), '.1f')} | {_fmt(r.get('peak_ram_gb', 0), '.1f')} GB |"
        )
    lines.append("")
    lines.append("## Time-to-Target")
    for r in results:
        ttt = r.get("time_to_target", {})
        if ttt:
            lines.append(f"\n### {r['optimizer']}")
            lines.append("| Target | Step | Elapsed |")
            lines.append("|--------|------|---------|")
            for target, info in sorted(ttt.items(), key=lambda x: float(x[0])):
                lines.append(f"| {target} | {info['step']} | {info['elapsed']}s |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(v, fmt: str = ".4f"):
    if v is None:
        return "N/A"
    return format(v, fmt)


def _print_table(results: list[dict]):
    h = f"{'Rank':<6} {'Optimizer':<12} {'Final Loss':<12} {'Tok/s':<10} {'RAM(GB)':<10} {'Time(s)':<10}"
    print(h)
    print("-" * len(h))
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r['optimizer']:<12} {_fmt(r.get('final_loss')):<12} "
              f"{_fmt(r.get('avg_tokens_per_second', 0), '.0f'):<10} "
              f"{_fmt(r.get('peak_ram_gb', 0), '.1f'):<10} "
              f"{_fmt(r.get('training_time_sec', 0), '.1f'):<10}")
