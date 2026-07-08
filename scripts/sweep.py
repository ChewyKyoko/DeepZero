import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from deepzero.config.loader import load_config
from deepzero.models.transformer import GPT, ModelConfig
from deepzero.tokenizers.base import create_tokenizer
from deepzero.datasets.base import create_dataset
from deepzero.datasets.loader import PackedDataset
from deepzero.training.trainer import Trainer
from deepzero.utils.seed import set_seed
from deepzero.experiments.manager import ExperimentManager
from deepzero.visualization.plots import generate_all_plots
from deepzero.reports.generator import generate_experiment_report
from deepzero.sweeps.grid import run_grid_search


def _run_trial(cfg: dict, trial_dir: str) -> dict:
    """Run one training trial and return summary."""
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cpu")

    ds_name = cfg.get("dataset", "tiny_textbooks")
    ds = create_dataset(ds_name)
    ds.preprocess()
    texts = ds.load_texts()

    tc = cfg["tokenizer"]
    tokenizer = create_tokenizer(tc["name"], vocab_size=tc["vocab_size"])
    tokenizer.train(texts[:5000])

    for attr in ("id_to_token", "char_to_id"):
        if hasattr(tokenizer, attr):
            actual_vocab = len(getattr(tokenizer, attr))
            break
    else:
        actual_vocab = tokenizer.vocab_size

    limit = cfg.get("data_limit", 0)
    if limit:
        texts = texts[:limit]
    n = len(texts)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val] if n_train + n_val < n else texts[-max(1, n//10):]

    ms = cfg["model"]
    seq_len = ms.get("max_seq_len", 512)
    train_ds = PackedDataset(tokenizer, train_texts, seq_len)
    val_ds = PackedDataset(tokenizer, val_texts, seq_len)
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model_cfg = ModelConfig(
        vocab_size=actual_vocab,
        d_model=ms["d_model"], n_layers=ms["n_layers"], n_heads=ms["n_heads"],
        d_ff=ms["d_ff"], max_seq_len=seq_len, dropout=ms.get("dropout", 0.1), device=device,
    )
    model = GPT(model_cfg)

    tr = cfg["training"]
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        lr=tr["lr"], weight_decay=tr.get("weight_decay", 0.1),
        warmup_iters=tr.get("warmup_iters", 10), max_iters=tr["max_iters"],
        checkpoint_dir=checkpoint_dir, device=device,
        log_interval=tr.get("log_interval", 1),
        eval_interval=tr.get("eval_interval", max(5, tr["max_iters"] // 2)),
        grad_clip=tr.get("grad_clip", 1.0),
        gradient_accumulation=tr.get("gradient_accumulation", 2),
        optimizer_name=tr.get("optimizer", "adamw"),
        tokenizer_name=tc["name"],
        early_stop_patience=tr.get("early_stop_patience"),
    )
    trainer.train()

    return {
        "final_loss": trainer._last_loss,
        "best_loss": trainer.best_train_loss,
        "best_val_loss": trainer.best_val_loss if trainer.best_val_loss != float("inf") else None,
        "steps": trainer.step,
        "tok_speed": trainer._last_tok_speed,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training/sweep.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sweeps"

    cfg = load_config(config_path)
    if "search" not in cfg:
        print(f"Config {config_path} has no 'search:' section. Nothing to sweep.")
        return

    print(f"Grid search from {config_path}")
    print(f"Output: {output_dir}")
    run_grid_search(config_path, output_dir, run_fn=_run_trial)


if __name__ == "__main__":
    main()
