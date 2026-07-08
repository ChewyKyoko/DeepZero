import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from deepzero.models.transformer import GPT, ModelConfig
from deepzero.tokenizers.base import create_tokenizer
from deepzero.datasets.base import create_dataset
from deepzero.datasets.loader import PackedDataset
from deepzero.training.trainer import Trainer
from deepzero.utils.seed import set_seed
from deepzero.config.loader import load_config
from deepzero.experiments.manager import ExperimentManager
from deepzero.visualization.plots import generate_all_plots
from deepzero.reports.generator import generate_experiment_report
from deepzero.evaluation.suite import run_full_evaluation


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training/full.yaml"
    cfg = load_config(config_path)

    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cpu")

    exp_cfg = cfg.get("experiment", {})
    use_tracking = exp_cfg.get("enabled", False)
    experiment = None
    if use_tracking:
        experiment = ExperimentManager(cfg)
        experiment.logger.info("Experiment tracking enabled")

    ds_name = cfg.get("dataset", "tiny_codes")
    ds = create_dataset(ds_name)
    ds.preprocess()
    texts = ds.load_texts()

    # Dataset versioning
    if experiment and hasattr(ds, 'statistics'):
        ds_stats = ds.statistics()
        n_samples = len(texts)
        content_hash = hashlib.md5(
            f"{n_samples}{texts[0][:100] if texts else ''}{texts[-1][:100] if texts else ''}".encode()
        ).hexdigest()[:16]
        dataset_info = {
            "name": ds_name,
            "n_samples": n_samples,
            "hash": content_hash,
            "statistics": ds_stats,
        }
        experiment.save_dataset_info(dataset_info)

    tc = cfg["tokenizer"]
    tokenizer = create_tokenizer(tc["name"], vocab_size=tc["vocab_size"])
    tokenizer.train(texts[:10000])

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

    checkpoint_dir = experiment.get_checkpoint_dir() if experiment else cfg.get("checkpoint", {}).get("dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    import time as _t
    print(f"[{_t.time():.0f}] Creating model...", flush=True)
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
    model = GPT(model_cfg)
    print(f"[{_t.time():.0f}] Model created ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)

    tr = cfg["training"]
    print(f"[{_t.time():.0f}] Creating Trainer...", flush=True)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=tr["lr"],
        weight_decay=tr.get("weight_decay", 0.1),
        warmup_iters=tr.get("warmup_iters", 50),
        max_iters=tr["max_iters"],
        checkpoint_dir=checkpoint_dir,
        device=device,
        log_interval=tr.get("log_interval", 10),
        eval_interval=tr.get("eval_interval", 100),
        grad_clip=tr.get("grad_clip", 1.0),
        compile_model=tr.get("compile_model", False),
        gradient_accumulation=tr.get("gradient_accumulation", 1),
        checkpoint_time_minutes=tr.get("checkpoint_time_minutes", 10.0),
        optimizer_name=tr.get("optimizer", "adamw"),
        optimizer_kwargs=tr.get("optimizer_kwargs", None),
        metrics_tracker=experiment.metrics if experiment else None,
        early_stop_patience=tr.get("early_stop_patience"),
        tokenizer_name=tc["name"],
    )

    print(f"[{_t.time():.0f}] Starting training...", flush=True)
    trainer.train()

    # Evaluation after training
    eval_results = None
    if val_loader is not None:
        print(f"[{_t.time():.0f}] Running evaluation...", flush=True)
        eval_results = run_full_evaluation(
            model, tokenizer, val_loader, device,
            save_dir=str(experiment.run_dir) if experiment else None,
        )
        if experiment:
            experiment.save_evaluation(eval_results)

    if experiment:
        experiment.finalize()
        generate_all_plots(experiment.metrics, experiment.get_plots_dir())

        report = generate_experiment_report(
            cfg, experiment.metrics,
            experiment.get_checkpoint_dir(),
            experiment.get_plots_dir(),
            experiment.run_dir,
            extra={"evaluation": eval_results} if eval_results else None,
        )
        report_path = experiment.run_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(report)
        experiment.logger.info("Report generated: %s", report_path)
        print(f"Experiment complete. Results in {experiment.run_dir}")
    else:
        print("Training complete (no experiment tracking)")


if __name__ == "__main__":
    main()
