import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader

from deepzero.models.transformer import GPT, ModelConfig
from deepzero.training.trainer import Trainer
from deepzero.training.metrics import compute_metrics
from deepzero.tokenizers.base import create_tokenizer
from deepzero.datasets.pipeline import _normalize_whitespace
from deepzero.datasets.quality import analyze_dataset
from deepzero.evaluation.coding import evaluate_code_quality, repetition_rate
from deepzero.utils.seed import set_seed


@dataclass
class DatasetBenchConfig:
    dataset_names: list[str] = field(default_factory=lambda: ["tiny_codes", "humaneval", "mbpp"])
    tokenizer_name: str = "bpe"
    tokenizer_vocab_size: int = 5000
    model_d_model: int = 384
    model_n_layers: int = 8
    model_n_heads: int = 8
    model_d_ff: int = 1536
    model_max_seq_len: int = 512
    model_dropout: float = 0.1
    training_lr: float = 3e-4
    training_weight_decay: float = 0.1
    training_warmup_iters: int = 100
    training_max_iters: int = 3000
    training_batch_size: int = 8
    training_grad_clip: float = 1.0
    generation_prompts: list[str] = field(default_factory=lambda: [
        "def fibonacci(n):", "def factorial(n):", "def sort_list(arr):"
    ])
    results_dir: str = "results"
    seed: int = 42
    device: str = "cpu"


@dataclass
class DatasetExperimentResult:
    experiment_id: str
    dataset_name: str
    config: dict
    quality_analysis: dict = field(default_factory=dict)
    training_stats: dict = field(default_factory=dict)
    generation_stats: dict = field(default_factory=dict)
    coding_quality: dict = field(default_factory=dict)
    git_commit_hash: str = ""
    timestamp: str = ""
    status: str = "incomplete"


def _get_git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _build_tensor_dataset(texts: list[str], tokenizer, seq_len: int):
    all_tokens = []
    for t in texts:
        norm = _normalize_whitespace(t)
        ids = tokenizer.encode(norm)
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        all_tokens.extend(ids)

    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, sl):
            self.tokens = tokens
            self.seq_len = sl
        def __len__(self):
            return max(0, len(self.tokens) - self.seq_len)
        def __getitem__(self, idx):
            x = torch.tensor(self.tokens[idx: idx + self.seq_len], dtype=torch.long)
            y = torch.tensor(self.tokens[idx + 1: idx + 1 + self.seq_len], dtype=torch.long)
            return x, y
    return TokenizedDataset(all_tokens, seq_len)


def run_dataset_benchmark(cfg: DatasetBenchConfig) -> list[DatasetExperimentResult]:
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    git_hash = _get_git_hash()
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print(f"Dataset Benchmark — R0.3")
    print(f"Tokenizer: {cfg.tokenizer_name} (vocab_size={cfg.tokenizer_vocab_size})")
    print(f"Datasets: {cfg.dataset_names}")
    print(f"Git hash: {git_hash}")
    print("=" * 60)

    tokenizer = create_tokenizer(cfg.tokenizer_name, vocab_size=cfg.tokenizer_vocab_size)

    all_results = []

    for dataset_name in cfg.dataset_names:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        experiment_id = f"ds_{dataset_name}_{int(time.time())}_{git_hash}"

        result = DatasetExperimentResult(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            config=asdict(cfg),
            git_commit_hash=git_hash,
            timestamp=timestamp,
        )

        try:
            from deepzero.datasets.base import create_dataset
            ds = create_dataset(dataset_name)
            ds.preprocess()
            ds.deduplicate()
            ds.normalize()
            texts = ds.load_texts()
            result.quality_analysis = analyze_dataset(texts, dataset_name)
            print(f"  Samples: {len(texts):,}, "
                  f"Avg length: {result.quality_analysis['avg_length']:.0f}")

            print(f"  Training tokenizer on dataset...")
            tokenizer.train(texts)
            actual_vocab = len(tokenizer.id_to_token) if hasattr(tokenizer, 'id_to_token') else tokenizer.vocab_size
            print(f"  Vocab: {actual_vocab}")

            # Build train/val from first 80/10% split
            n = len(texts)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            train_texts = texts[:n_train]
            val_texts = texts[n_train:n_train + n_val] if n_train + n_val < n else texts[-max(1, n//10):]

            train_ds = _build_tensor_dataset(train_texts, tokenizer, cfg.model_max_seq_len)
            val_ds = _build_tensor_dataset(val_texts, tokenizer, cfg.model_max_seq_len)
            train_loader = DataLoader(train_ds, batch_size=cfg.training_batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.training_batch_size, shuffle=False)

            model_cfg = ModelConfig(
                vocab_size=actual_vocab,
                d_model=cfg.model_d_model,
                n_layers=cfg.model_n_layers,
                n_heads=cfg.model_n_heads,
                d_ff=cfg.model_d_ff,
                max_seq_len=cfg.model_max_seq_len,
                dropout=cfg.model_dropout,
                device=cfg.device,
            )
            model = GPT(model_cfg)
            print(f"  Model: {model.n_params:,} params")

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=cfg.training_lr,
                weight_decay=cfg.training_weight_decay,
                warmup_iters=cfg.training_warmup_iters,
                max_iters=cfg.training_max_iters,
                checkpoint_dir=os.path.join(cfg.results_dir, "checkpoints", experiment_id),
                device=cfg.device,
                log_interval=50,
                eval_interval=200,
                grad_clip=cfg.training_grad_clip,
            )

            train_start = time.time()
            trainer.train()
            total_train_time = time.time() - train_start

            final_metrics = compute_metrics(model, val_loader, cfg.device)
            result.training_stats = {
                "final_loss": final_metrics.get("val_loss", 0),
                "final_perplexity": final_metrics.get("perplexity", 0),
                "training_time_seconds": total_train_time,
                "n_steps": trainer.step,
                "tokens_per_second": (trainer.step * cfg.training_batch_size * cfg.model_max_seq_len) / max(1, total_train_time),
                "best_val_loss": trainer.best_val_loss,
            }

            # Generation eval
            gen_outputs = []
            for prompt in cfg.generation_prompts:
                try:
                    output = model.generate(tokenizer, prompt, max_len=80, temperature=0.8, top_k=40)
                    gen_outputs.append({"prompt": prompt, "output": output})
                except Exception as e:
                    gen_outputs.append({"prompt": prompt, "output": "", "error": str(e)})

            n_syn = sum(1 for g in gen_outputs if g.get("output") and evaluate_code_quality(g["output"])["syntax_valid"])
            n_comp = sum(1 for g in gen_outputs if g.get("output") and evaluate_code_quality(g["output"])["exec_success"])
            gen_lengths = [len(g.get("output", "")) for g in gen_outputs]
            rep_rates = [repetition_rate(g.get("output", ""), 3) for g in gen_outputs if g.get("output")]

            result.generation_stats = {
                "n_prompts": len(gen_outputs),
                "syntax_valid_rate": n_syn / max(1, len(gen_outputs)),
                "compile_success_rate": n_comp / max(1, len(gen_outputs)),
                "avg_output_length": sum(gen_lengths) / max(1, len(gen_lengths)),
                "avg_repetition_rate_3gram": sum(rep_rates) / max(1, len(rep_rates)),
            }

            # Coding quality on generated code
            qualities = [evaluate_code_quality(g.get("output", "")) for g in gen_outputs if g.get("output")]
            if qualities:
                result.coding_quality = {
                    "avg_n_functions": sum(q["n_functions"] for q in qualities) / len(qualities),
                    "avg_n_classes": sum(q["n_classes"] for q in qualities) / len(qualities),
                    "avg_n_lines": sum(q["n_lines"] for q in qualities) / len(qualities),
                    "syntax_valid_rate": sum(1 for q in qualities if q["syntax_valid"]) / len(qualities),
                    "exec_success_rate": sum(1 for q in qualities if q["exec_success"]) / len(qualities),
                }

            result.status = "completed"
            print(f"  ✓ Loss={result.training_stats['final_loss']:.4f}, "
                  f"PPL={result.training_stats['final_perplexity']:.2f}")

        except Exception as e:
            result.status = "failed"
            result.training_stats = {"error": str(e)}
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

        # Save individual
        exp_dir = os.path.join(cfg.results_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "result.json"), "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        all_results.append(result)

    combined_path = os.path.join(cfg.results_dir, "all_dataset_results.json")
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nResults saved to {combined_path}")
    return all_results
