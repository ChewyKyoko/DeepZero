# DeepZero

[![Stage](https://img.shields.io/badge/Stage-R0.5--intelligence-blueviolet)]()
[![Python](https://img.shields.io/badge/Python-3.13+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-orange)]()

Reproducible AI research platform for studying how small language models become capable coding assistants — on resource-constrained hardware (CPU/laptop).

Runs on a single CPU core (Intel Core Ultra 7 155U, 16GB RAM). No GPU required.

---

## Research Phases

| Phase | What | Status |
|-------|------|--------|
| **R0.0** | Package restructure (`deepzero/` submodules, `pyproject.toml`, configs, tests) | ✅ |
| **R0.2** | Tokenizer study — BPE, Byte BPE, SentencePiece Unigram, Character-level | ✅ |
| **R0.3** | Dataset study — 6 sources + weighted mixtures + quality analysis | ✅ |
| **R0.4** | Evaluation study — 20 coding tasks, sandbox executor, unified scoring | ✅ |
| **R0.5** | Intelligence layer — experiment index, ranking, suggestions, dead-end detection, dashboard | ✅ |

Next on deck: **R1.0** — architecture research with best-known tokenizer + dataset + eval.

---

## Project Structure

```
├── deepzero/               # Main package
│   ├── models/             # GPT decoder-only transformer
│   ├── tokenizers/         # 4 tokenizer implementations
│   ├── datasets/           # 6 dataset loaders + mixtures
│   ├── training/           # Trainer, optimizer, metrics
│   ├── evaluation/         # Sandbox executor, scoring, weakness analysis
│   ├── experiments/        # Benchmarks + intelligence pipeline
│   └── agents/             # RL self-improvement loop
├── configs/                # YAML configs for experiments
├── scripts/                # CLI entry points
├── tests/                  # 79 passing tests
├── docs/                   # Architecture & API docs
├── results/                # Experiment outputs (gitignored)
├── data/                   # Datasets & caches (gitignored)
├── rl/                     # Original RL self-improvement layer
├── config.py               # Legacy config (backward compat)
├── tokenizer.py            # Legacy tokenizer
├── model.py                # Old flat module
├── run.py                  # Legacy entry point
└── train.py                # Legacy training script
```

---

## Quickstart

```bash
# Install
pip install -e ".[dev,test]"

# Run tests (79 pass, 1 skips HF auth)
pytest tests/ -q

# Tokenizer benchmark
python scripts/benchmark.py

# Dataset benchmark
python scripts/run_dataset_bench.py

# Generate research dashboard
python -c "
from deepzero.experiments.intelligence.index import ExperimentIndex
from deepzero.experiments.intelligence.dashboard import build_dashboard, generate_dashboard_report

idx = ExperimentIndex()
idx.rebuild_from_disk('results')
build_dashboard(idx, output_dir='results')
generate_dashboard_report(idx, output_path='results/research_dashboard.md')
"
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
| Optimizer | AdamW (3e-4, 0.1 wd) |
| Schedule | Cosine + warmup |

Fixed across all experiments — only tokenizer, dataset, and evaluation variables change per phase.

---

## Results

After running benchmarks, view the dashboard:

```
results/research_dashboard.json   # Machine-readable
results/research_dashboard.md     # Human-readable
```

The intelligence layer automatically:
- Indexes all experiments from disk
- Ranks by composite score per category
- Suggests next experiments to fill coverage gaps
- Flags dead ends (diminishing returns, repeated configs)
- Computes convergence trends

---

## License

MIT
