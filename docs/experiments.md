# Experiments

DeepZero uses an experiment registry to track self-improvement rounds.

## Registry

Experiments are stored in `outputs/experiments/<run_id>/` with:

- `manifest.json` — run metadata, config, round results
- `round_<N>.json` — per-round results

## Running an Experiment

```python
from deepzero.models.transformer import GPT, ModelConfig
from deepzero.tokenizers.bpe import BPETokenizer
from deepzero.experiments.runner import SelfImprovementLoop

model = GPT(ModelConfig(vocab_size=1300))
tokenizer = BPETokenizer.from_pretrained("data/bpe_tokenizer.json")
loop = SelfImprovementLoop(model, tokenizer)
run_id = loop.run_experiment("my-test", n_rounds=3, tasks_per_round=5)
```

## Comparing Runs

```python
from deepzero.experiments.compare import compare_runs

comparison = compare_runs(["run_001", "run_002"])
```

## Failure Logging

Failed responses are logged to `outputs/failures.jsonl` for later replay training.
