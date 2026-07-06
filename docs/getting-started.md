# Getting Started with DeepZero

## Installation

```bash
git clone https://github.com/ChewyKyoko/DeepZero.git
cd DeepZero
python -m venv .venv
source .venv/bin/activate
pip install torch>=2.0.0 pyyaml
```

## Training a Model

```bash
python scripts/run.py
```

Or with a config file:

```bash
python scripts/train.py configs/default.yaml
```

## Generating Text

```bash
python scripts/generate.py checkpoints/best.pt "Your prompt here"
```

## Interactive REPL

```bash
python scripts/repl.py checkpoints/best.pt
```

## Project Structure

```
deepzero/          # Main package
├── models/        # Transformer architecture
├── tokenizers/    # BPE tokenization
├── datasets/      # Data loading
├── training/      # Training loop
├── inference/     # Generation & sampling
├── evaluation/    # Code sandbox & scoring
├── replay/        # Failure logging & replay buffer
├── tasks/         # Coding task generator
├── experiments/   # Experiment registry & comparison
├── agents/        # Agent framework
└── utils/         # I/O, logging, seeding
configs/           # YAML/JSON config files
scripts/           # CLI entry points
tests/             # Unit tests
docs/              # Documentation
```
