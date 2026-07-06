# API Reference

## `deepzero.models`

### `GPT`
```python
GPT(config: ModelConfig) -> GPT
```
Decoder-only transformer model.

- `forward(x, targets=None) -> (logits, loss)`
- `generate(tokenizer, prompt, max_len, temperature, top_k) -> str`
- `n_params -> int`

### `ModelConfig`
```python
ModelConfig(vocab_size=5000, d_model=384, n_layers=8, n_heads=8, d_ff=1536, max_seq_len=512, dropout=0.1, device="cpu")
```

## `deepzero.tokenizers`

### `BPETokenizer`
```python
BPETokenizer(vocab_size=1300)
```
- `train(text)` — learn BPE merges
- `encode(text) -> list[int]`
- `decode(ids) -> str`
- `save(path)` / `from_pretrained(path)`

## `deepzero.training`

### `Trainer`
```python
Trainer(model, train_loader, val_loader=None, lr=3e-4, max_iters=5000, ...)
```
- `train()` — run full training
- `train_epoch() -> float` — single epoch, returns average loss

## `deepzero.experiments`

### `ExperimentRegistry`
```python
ExperimentRegistry(base_dir="outputs/experiments")
```
- `create_run(name, description, config) -> run_id`
- `log_round(run_id, round_num, results)`
- `finalize_run(run_id)`
- `list_runs() -> list[dict]`

### `SelfImprovementLoop`
```python
SelfImprovementLoop(model, tokenizer, ...)
```
- `run_round(n_tasks, temperature, top_k) -> dict`
- `run_experiment(name, n_rounds, tasks_per_round, ...) -> run_id`

## `deepzero.evaluation`

### `CodeSandbox`
```python
CodeSandbox(timeout=5)
```
- `check_syntax(code) -> (bool, error)`
- `run(code, timeout) -> dict`
