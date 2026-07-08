# DeepZero

[![Stage](https://img.shields.io/badge/Stage-R0.3--evaluation--sweep-blueviolet)]()
[![Python](https://img.shields.io/badge/Python-3.13+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-orange)]()

Reproducible AI research platform for studying how small language models become capable coding assistants — on resource-constrained hardware (CPU/laptop).

Runs on a single CPU core (Intel Core Ultra 7 155U, 16GB RAM). No GPU required.

---

## Research Phases

| Phase | What | Status |
|-------|------|--------|
| **R0.1** | Foundation — GPT-style transformer, character tokenizer, data pipeline, basic training | ✅ |
| **R0.2** | Experimentation — optimizer benchmarks (AdamW/Muon/Sophia/Lion), experiment tracking, metrics, live dashboard, plots, reports, 5-mode checkpoints, early stopping, config inheritance | ✅ |
| **R0.3** | Evaluation & Search — perplexity + 5-category benchmarks, hyperparameter grid search, dataset versioning | ✅ |
| **R0.4** | Scale & Ship — model scaling to 100M+, multi-GPU, ONNX/GGUF export, inference serving | ⏳ |

---

## Project Structure

```
├── deepzero/                   # Main package
│   ├── models/                 # GPT decoder-only transformer, checkpoints
│   ├── tokenizers/             # 4 tokenizer implementations (BPE, Byte BPE, Unigram, Character)
│   ├── datasets/               # Dataset loaders + PackedDataset chunking
│   ├── training/               # Trainer, optimizer (AdamW/Muon/Sophia/Lion), dashboard, metrics
│   ├── config/                 # YAML config loader with base: inheritance
│   ├── evaluation/             # Benchmark suite (perplexity + 5-category generation evals)
│   ├── experiments/            # ExperimentManager (run tracking, metadata, finalization)
│   ├── metrics/                # MetricsTracker (per-step recording, time-to-target)
│   ├── logging/                # Logging setup (file + console)
│   ├── visualization/          # 6 plot types (loss, val, lr, grad norm, RAM, tok/s)
│   ├── reports/                # Auto-generated experiment report.md
│   ├── benchmarks/             # Optimizer + tokenizer benchmark runners
│   ├── sweeps/                 # Hyperparameter grid search
│   └── inference/              # Top-k/top-p generation
├── configs/                    # YAML configs with base: inheritance
│   └── training/
│       ├── base.yaml           # Shared base config
│       └── sweep.yaml          # Grid search over search: section
├── scripts/                    # CLI entry points
├── tests/                      # Test suite
├── docs/                       # Architecture, roadmap, dependencies, optimizer comparison
├── runs/                       # Per-experiment run directories (auto-created)
├── benchmarks/                 # Benchmark output files (auto-created)
├── shell.nix                   # Nix shell with all dependencies
└── pyproject.toml              # Package metadata
```

---

## Quickstart

```bash
# Nix shell (recommended)
nix-shell

# Install
pip install -e ".[dev,test]"

# Train with experiment tracking
python scripts/train_pipeline.py configs/training/full.yaml

# Optimizer benchmark (AdamW, Muon, Sophia, Lion)
python scripts/benchmark_optimizers.py

# Tokenizer benchmark (BPE, Byte BPE, Unigram, Character)
python scripts/benchmark_tokenizers.py

# Hyperparameter sweep
python scripts/sweep.py configs/training/sweep.yaml
```

---

## Transformer Architecture

| Spec | Value |
|------|-------|
| Params | 19.1M (default) |
| Layers | 8 |
| Heads | 8 |
| d_model | 384 |
| d_ff | 1536 |
| Max seq | 512 |
| Vocab | configurable (128–5000) |
| Optimizer | AdamW / Muon / Sophia / Lion (configurable) |
| Schedule | Cosine + linear warmup |
| Norm | RMSNorm (instead of LayerNorm) |
| FF | SwiGLU (instead of ReLU) |
| Attention | Scaled dot-product (FlashAttention-compatible) |
| Weight tying | Token embed ↔ lm_head |

---

## Experiment Tracking

Every training run auto-creates `runs/YYYY-MM-DD_HH-MM-SS/` with:

```
run_001/
├── config.yaml           # Exact config used
├── hardware.json         # CPU, RAM, device, cores
├── git.json              # Commit hash, branch, dirty state
├── environment.json      # Python + PyTorch versions
├── dataset.json          # Dataset metadata + hash
├── evaluation.json       # Eval results (perplexity + benchmarks)
├── metrics.csv           # Per-step metrics (CSV)
├── metrics.json          # Per-step metrics (JSON)
├── summary.json          # Best loss, tokens/s, time-to-target
├── training.log          # Structured log file
├── report.md             # Auto-generated experiment report
├── checkpoints/
│   ├── best_loss.pt      # Lowest training loss
│   ├── best_val.pt       # Lowest validation loss
│   ├── fastest.pt        # Highest throughput
│   ├── latest.pt         # Most recent (default resume point)
│   └── final.pt          # End of training
└── plots/
    ├── loss_curve.png
    ├── validation_loss.png
    ├── learning_rate.png
    ├── gradient_norm.png
    ├── ram_usage.png
    └── tokens_per_second.png
```

Enable in config: `experiment.enabled: true`
Run: `python scripts/train_pipeline.py configs/training/full.yaml`

### Live Dashboard

Training displays a live terminal dashboard with progress bar, loss, validation loss, perplexity, learning rate, gradient norm, tokens/sec, RAM, and ETA — updated every `log_interval` steps.

### Config Inheritance

Configs support a `base:` key for inheritance:

```yaml
# sweep.yaml
base: configs/training/full.yaml
training:
  optimizer: adamw
  lr: 3e-4
search:
  training.lr: [1e-4, 3e-4, 1e-3]
```

## Optimizer Benchmark Results

Ranked by final loss (15 steps, 19.4M param model, CPU):

| Rank | Optimizer | Final Loss | Tok/s | RAM (GB) | Time (s) |
|------|-----------|-----------|-------|----------|----------|
| 1 | **Sophia** | **4.1593** | 728 | 4.9 | 345 |
| 2 | AdamW | 4.3122 | 491 | 5.5 | 510 |
| 3 | Lion | 4.7626 | **760** | **4.6** | **333** |
| 4 | Muon | 5.5519 | 723 | 5.5 | 349 |

**Sophia wins** on convergence — Hessian diagonal guidance delivers lower loss at comparable speed. Lion is fastest but converges ~10% worse. AdamW is reliable but 55% slower.

See [docs/optimizers.md](docs/optimizers.md) for full comparison.

## Tokenizers

| Tokenizer | Type | Train Time (10K texts) | Tok/text |
|-----------|------|----------------------|----------|
| BPE | Word-level subword | slowest | lowest |
| Byte-level BPE | Byte-level subword | moderate | medium |
| Unigram | Probabilistic | moderate | medium |
| Character | Per-character | instant | highest |

Benchmark: `python scripts/benchmark_tokenizers.py`

## Evaluation Suite

5 benchmark categories with 19 prompts total:

| Category | Prompts | Examples |
|----------|---------|---------|
| Code Completion | 5 | fibonacci, factorial, bubble_sort, reverse_string, fizzbuzz |
| Python Syntax | 5 | if_else, for_loop, list_comprehension, try_except, class_def |
| Math | 4 | sum_1_to_n, is_prime, gcd, fibonacci_nth |
| Reasoning | 2 | fizzbuzz, binary_search |
| Trivia | 3 | capital_of_france, meaning_of_life, hello_world |

Run: `python scripts/train_pipeline.py configs/training/full.yaml` (runs eval after training)

## Hyperparameter Sweep

Grid search over any config parameter:

```yaml
search:
  training.optimizer: [adamw, sophia]
  training.lr: [1e-4, 3e-4]
```

Run: `python scripts/sweep.py configs/training/sweep.yaml`
Generates: `leaderboard.csv` sorted by `best_loss`.

## Checkpoints

5 auto-selected checkpoints saved per run:

| File | Trigger | Use |
|------|---------|-----|
| `best_loss.pt` | Training loss improves | Track convergence |
| `best_val.pt` | Validation loss improves | Best generalizer |
| `fastest.pt` | Throughput improves | Performance tuning |
| `latest.pt` | Every save | Resume training |
| `final.pt` | Training ends | Eval / inference |

All checkpoints include optimizer state, scheduler state, and RNG state (torch + random + numpy) for exact resume.

---

## License

MIT
