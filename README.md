# DeepZero

[![Stage](https://img.shields.io/badge/Stage-RN-blueviolet)]()

**DeepZero** is a lightweight experimental AI learning system for exploring self-improving coding intelligence on resource-constrained hardware (CPU/laptop environments).

It is not a large-scale LLM training project, but a modular research framework that combines:
- small transformer language models
- structured coding datasets
- automated evaluation systems
- failure-based learning loops
- reinforcement-style self-improvement pipelines

The goal: study how coding ability can emerge and improve through iterative training, evaluation, and feedback — without relying on large-scale compute.

---

## Core Idea

```
┌──────────────────────────────────────────┐
│          Self-Improvement Cycle           │
│                                          │
│  Task Generator ──→ Model ──→ Evaluate   │
│       ↑                      │           │
│       │                      ▼           │
│  Fine-tune ←── Replay Buffer ←── Logger  │
│                                          │
└──────────────────────────────────────────┘
```

- **Task Generation** — creates coding problems and bug-fix tasks
- **Sandboxed Evaluation** — runs generated code safely, scores correctness
- **Failure Learning Loop** — stores incorrect outputs as training data
- **Self-Improvement** — generate → evaluate → log → retrain → repeat

---

## Project Structure

```
├── config.py          # Model hyperparameters
├── tokenizer.py       # BPE tokenizer (train/encode/decode)
├── model.py           # GPT-style decoder-only transformer
├── dataset.py         # Text dataset loader
├── train.py           # Training loop (AdamW, LR schedule, checkpointing)
├── generate.py        # Inference (temperature, top-k, top-p)
├── sample_data.txt    # Built-in training data (Python + prose)
│
├── rl/                # Self-improvement layer
│   ├── config.py      # Task definitions + RL hyperparams
│   ├── tasks.py       # Coding problem generator with test cases
│   ├── evaluate.py    # Sandboxed code execution + scoring
│   ├── logger.py      # Failure storage (JSONL)
│   ├── buffer.py      # Replay buffer curator
│   ├── loop.py        # Self-improvement loop + fine-tuning
│   └── run.py         # CLI entry point
│
├── models/gguf/       # Optional GGUF models for inference
├── runs/              # Checkpoints (gitignored)
└── data/              # Generated logs + buffers (gitignored)
```

---

## Quickstart

```bash
# 1. Install PyTorch
# Requires: torch>=2.0.0 (CPU or CUDA)

# 2. Train the base model
python3 run.py --steps 1000

# 3. Generate text from the trained model
python3 run.py --just-generate --prompt "def fibonacci"

# 4. Run self-improvement loop
python3 rl/run.py --iterations 3 --tasks 10

# Mini mode (faster CPU training)
python3 run.py --mini --steps 2000
python3 run.py --mini --just-generate
```

---

## Architecture

| Spec | Mini | Full |
|------|------|------|
| Parameters | 2.7M | 19.6M |
| Layers | 4 | 8 |
| Heads | 4 | 8 |
| d_model | 192 | 384 |
| Max seq | 256 | 512 |
| Vocab | BPE (1300) | BPE (1300) |

---

## Current State (R0 + RL Prototype)

- Small transformer (~2M–20M parameters)
- BPE tokenizer
- CPU-based training pipeline (~900–3300 tok/s)
- Working end-to-end RL loop
- Task → Code → Evaluate → Replay system

**Observed behavior:**
- Learns function patterns (factorial, fibonacci, binary search)
- Improves format understanding (task → code) after RL fine-tuning
- Currently limited by tokenizer quality and dataset scale, not pipeline design

**Key limitation:** 19.6M parameters and 10KB of training data are insufficient for reliable code generation. The learning loop is functional; reliability requires larger models and datasets.

---

## Research Direction

Future improvements may include:
- Pretrained code models as base (0.5B–3B range)
- Byte-level or code-optimized tokenization
- Larger structured code datasets (10MB+)
- Curriculum-based task difficulty scaling
- Improved evaluation-driven training signals

---

## Philosophy

> *"Models do not improve from training alone — they improve from structured experience and feedback loops."*

---

## License

MIT
