import json
import os
import subprocess
import time
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from deepzero.models.transformer import GPT, ModelConfig
from deepzero.datasets.loader import TextDataset
from deepzero.training.trainer import Trainer
from deepzero.training.metrics import compute_metrics
from deepzero.tokenizers.base import BaseTokenizer, create_tokenizer
from deepzero.datasets.pipeline import build_dataset, load_dataset
from deepzero.evaluation.coding import evaluate_code_quality, compression_ratio, repetition_rate
from deepzero.evaluation.sandbox import CodeSandbox
from deepzero.utils.seed import set_seed


@dataclass
class BenchmarkConfig:
    tokenizer_names: list[str] = field(default_factory=lambda: ["bpe", "byte_bpe", "unigram", "character"])
    vocab_sizes: list[int] = field(default_factory=lambda: [500, 1000, 2000, 5000])
    model_d_model: int = 384
    model_n_layers: int = 8
    model_n_heads: int = 8
    model_d_ff: int = 1536
    model_max_seq_len: int = 512
    model_dropout: float = 0.1
    training_lr: float = 3e-4
    training_weight_decay: float = 0.1
    training_warmup_iters: int = 100
    training_max_iters: int = 5000
    training_batch_size: int = 8
    training_grad_clip: float = 1.0
    dataset_cache_dir: str = "data/tiny-codes"
    dataset_language: str = "python"
    dataset_min_length: int = 50
    dataset_max_length: int = 100000
    results_dir: str = "results"
    seed: int = 42
    device: str = "cpu"


@dataclass
class ExperimentResult:
    experiment_id: str
    tokenizer_name: str
    vocab_size: int
    config: dict
    dataset_stats: dict = field(default_factory=dict)
    tokenizer_stats: dict = field(default_factory=dict)
    training_stats: dict = field(default_factory=dict)
    generation_stats: dict = field(default_factory=dict)
    coding_quality: dict = field(default_factory=dict)
    git_commit_hash: str = ""
    timestamp: str = ""
    status: str = "incomplete"


def _get_git_hash() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _tokenize_dataset(tokenizer: BaseTokenizer, texts: list[str],
                      max_seq_len: int) -> list[list[int]]:
    tokenized = []
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        tokenized.append(ids)
    return tokenized


def _build_tensor_dataset(tokenized: list[list[int]], seq_len: int):
    all_tokens = []
    for ids in tokenized:
        all_tokens.extend(ids)
    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, tokens, seq_len):
            self.tokens = tokens
            self.seq_len = seq_len
        def __len__(self):
            return max(0, len(self.tokens) - self.seq_len)
        def __getitem__(self, idx):
            x = torch.tensor(self.tokens[idx: idx + self.seq_len], dtype=torch.long)
            y = torch.tensor(self.tokens[idx + 1: idx + 1 + self.seq_len], dtype=torch.long)
            return x, y
    return TokenizedDataset(all_tokens, seq_len)


def run_benchmark(cfg: BenchmarkConfig) -> list[ExperimentResult]:
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    git_hash = _get_git_hash()
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print(f"Tokenizer Benchmark — {timestamp}")
    print(f"Git hash: {git_hash}")
    print(f"Dataset: {cfg.dataset_cache_dir} ({cfg.dataset_language})")
    print("=" * 60)

    dataset_meta = build_dataset(
        cache_dir=cfg.dataset_cache_dir,
        language=cfg.dataset_language,
        min_length=cfg.dataset_min_length,
        max_length=cfg.dataset_max_length,
    )
    train_texts = load_dataset(cfg.dataset_cache_dir, "train")
    val_texts = load_dataset(cfg.dataset_cache_dir, "validation")
    test_texts = load_dataset(cfg.dataset_cache_dir, "test")
    print(f"Dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")

    dataset_stats = {
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "language": cfg.dataset_language,
        "mean_train_length": sum(len(t) for t in train_texts) / max(1, len(train_texts)),
    }

    all_results = []

    for tokenizer_name in cfg.tokenizer_names:
        for vocab_size in cfg.vocab_sizes:
            print(f"\n{'=' * 60}")
            print(f"Tokenizsr: {tokenizer_name} (vocab_size={vocab_size})")
            print(f"{'=' * 60}")

            experiment_id = f"{tokenizer_name}_v{vocab_size}_{int(time.time())}_{git_hash}"

            result = ExperimentResult(
                experiment_id=experiment_id,
                tokenizer_name=tokenizer_name,
                vocab_size=vocab_size,
                config=asdict(cfg),
                dataset_stats=dataset_stats,
                git_commit_hash=git_hash,
                timestamp=timestamp,
            )

            try:
                tokenizer = create_tokenizer(tokenizer_name, vocab_size=vocab_size)
                print(f"Training tokenizer...")
                start = time.time()
                tokenizer.train(train_texts + val_texts + test_texts)
                train_time = time.time() - start
                result.tokenizer_stats = tokenizer.statistics()
                result.tokenizer_stats["train_time_seconds"] = train_time

                tok_save_path = os.path.join(cfg.results_dir, f"{experiment_id}_tokenizer.json")
                tokenizer.save(tok_save_path)

                tokenizer_test_texts = test_texts[:100] if test_texts else train_texts[:100]
                tokens_per_sample = []
                for t in tokenizer_test_texts:
                    tokens_per_sample.append(len(tokenizer.encode(t)))
                result.tokenizer_stats["avg_tokens_per_sample"] = (
                    sum(tokens_per_sample) / max(1, len(tokens_per_sample))
                )
                result.tokenizer_stats["compression_ratio"] = compression_ratio(
                    tokenizer, tokenizer_test_texts
                )

                # Tokenize dataset and build loaders
                train_tokenized = _tokenize_dataset(tokenizer, train_texts, cfg.model_max_seq_len)
                val_tokenized = _tokenize_dataset(tokenizer, val_texts[:500], cfg.model_max_seq_len)
                n_train_tokens = sum(len(t) for t in train_tokenized)
                result.tokenizer_stats["n_train_tokens"] = n_train_tokens

                train_ds = _build_tensor_dataset(train_tokenized, cfg.model_max_seq_len)
                val_ds = _build_tensor_dataset(val_tokenized, cfg.model_max_seq_len)
                train_loader = DataLoader(train_ds, batch_size=cfg.training_batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=cfg.training_batch_size, shuffle=False)

                # Build model
                model_cfg = ModelConfig(
                    vocab_size=tokenizer.vocab_size if vocab_size == tokenizer.vocab_size else len(tokenizer.id_to_token),
                    d_model=cfg.model_d_model,
                    n_layers=cfg.model_n_layers,
                    n_heads=cfg.model_n_heads,
                    d_ff=cfg.model_d_ff,
                    max_seq_len=cfg.model_max_seq_len,
                    dropout=cfg.model_dropout,
                    device=cfg.device,
                )
                model = GPT(model_cfg)
                print(f"Model: {model.n_params:,} params, vocab={model_cfg.vocab_size}")

                # Collect pre-training info
                # Determine actual vocab from the model
                actual_vocab = model_cfg.vocab_size

                # Train
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

                # Training metrics
                final_metrics = compute_metrics(model, val_loader, cfg.device)
                result.training_stats = {
                    "final_loss": final_metrics.get("val_loss", 0),
                    "final_perplexity": final_metrics.get("perplexity", 0),
                    "training_time_seconds": total_train_time,
                    "n_steps": trainer.step,
                    "tokens_per_second": (trainer.step * cfg.training_batch_size * cfg.model_max_seq_len) / max(1, total_train_time),
                    "best_val_loss": trainer.best_val_loss,
                }

                # Generation evaluation
                generation_prompts = [
                    "def fibonacci(n):",
                    "def factorial(n):",
                    "def sort_list(arr):",
                    "class Node:",
                    "import os",
                    "def hello():",
                ]
                gen_outputs = []
                for prompt in generation_prompts[:3]:
                    try:
                        output = model.generate(
                            tokenizer, prompt,
                            max_len=100, temperature=0.8, top_k=40
                        )
                        gen_outputs.append({"prompt": prompt, "output": output})
                    except Exception as e:
                        gen_outputs.append({"prompt": prompt, "output": "", "error": str(e)})

                n_syntax_valid = 0
                n_compile = 0
                total_gen_length = 0
                total_rep_rate = 0.0
                code_qualities = []

                for gen in gen_outputs:
                    output_text = gen.get("output", "")
                    if not output_text:
                        continue
                    total_gen_length += len(output_text)
                    total_rep_rate += repetition_rate(output_text, 3)
                    quality = evaluate_code_quality(output_text)
                    code_qualities.append(quality)
                    if quality["syntax_valid"]:
                        n_syntax_valid += 1
                    if quality["exec_success"]:
                        n_compile += 1

                n_gen = len(gen_outputs)
                result.generation_stats = {
                    "n_prompts": n_gen,
                    "n_syntax_valid": n_syntax_valid,
                    "syntax_valid_rate": n_syntax_valid / max(1, n_gen),
                    "n_compile_success": n_compile,
                    "compile_success_rate": n_compile / max(1, n_gen),
                    "avg_output_length": total_gen_length / max(1, n_gen),
                    "avg_repetition_rate_3gram": total_rep_rate / max(1, n_gen),
                }

                # Coding quality (averaged)
                if code_qualities:
                    result.coding_quality = {
                        "avg_n_functions": sum(q["n_functions"] for q in code_qualities) / len(code_qualities),
                        "avg_n_classes": sum(q["n_classes"] for q in code_qualities) / len(code_qualities),
                        "avg_n_lines": sum(q["n_lines"] for q in code_qualities) / len(code_qualities),
                        "avg_repetition_rate": sum(q.get("repetition_rate_3gram", 0) for q in code_qualities) / len(code_qualities),
                        "syntax_valid_rate": sum(1 for q in code_qualities if q["syntax_valid"]) / len(code_qualities),
                        "exec_success_rate": sum(1 for q in code_qualities if q["exec_success"]) / len(code_qualities),
                    }

                result.status = "completed"
                print(f"  ✓ Completed — loss={result.training_stats['final_loss']:.4f}, "
                      f"ppl={result.training_stats['final_perplexity']:.2f}")

            except Exception as e:
                result.status = "failed"
                result.training_stats = {"error": str(e)}
                print(f"  ✗ Failed — {e}")
                import traceback
                traceback.print_exc()

            # Save individual result
            result_dir = os.path.join(cfg.results_dir, experiment_id)
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, "result.json"), "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)

            all_results.append(result)

    # Save combined results
    combined_path = os.path.join(cfg.results_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nAll results saved to {combined_path}")

    return all_results
