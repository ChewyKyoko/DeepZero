from typing import Optional

from deepzero.experiments.intelligence.index import ExperimentRecord, ExperimentIndex


DEFAULT_VOCAB_SIZES = [500, 1000, 2000, 5000]
DEFAULT_TOKENIZERS = ["bpe", "byte_bpe", "unigram", "character"]
DEFAULT_DATASETS = ["tiny_codes", "humaneval", "mbpp"]


def _get_completed(records: list[ExperimentRecord], exp_type: Optional[str] = None,
                   phase: Optional[str] = None) -> list[ExperimentRecord]:
    filtered = [r for r in records if r.status == "completed"]
    if exp_type:
        filtered = [r for r in filtered if r.exp_type == exp_type]
    if phase:
        filtered = [r for r in filtered if r.phase == phase]
    return filtered


def suggest_next_experiment(index: ExperimentIndex) -> list[dict]:
    records = index.records
    suggestions = []

    # What tokenizer variants haven't been tested?
    tokenizer_records = _get_completed(records, exp_type="tokenizer")
    tested_tokenizers = set(r.name for r in tokenizer_records)
    tested_vocab_sizes = set()
    for r in tokenizer_records:
        vs = r.config.get("vocab_sizes", [])
        if isinstance(vs, list):
            tested_vocab_sizes.update(vs)

    for tok in DEFAULT_TOKENIZERS:
        for vs in DEFAULT_VOCAB_SIZES:
            if f"{tok}_v{vs}" not in tested_tokenizers and vs not in tested_vocab_sizes:
                suggestions.append({
                    "priority": "high" if tok in ("bpe", "byte_bpe") else "medium",
                    "type": "tokenizer",
                    "suggestion": f"Test {tok} with vocab_size={vs}",
                    "rationale": f"Missing tokenizer variant: {tok} at vocab_size={vs}",
                    "config": {"tokenizer_name": tok, "vocab_sizes": [vs]},
                })

    # What datasets haven't been tested?
    dataset_records = _get_completed(records, exp_type="dataset")
    tested_datasets = set(r.name for r in dataset_records)
    for ds in DEFAULT_DATASETS:
        if ds not in tested_datasets:
            suggestions.append({
                "priority": "high",
                "type": "dataset",
                "suggestion": f"Train on {ds} dataset",
                "rationale": f"Dataset {ds} has not been benchmarked yet",
                "config": {"dataset_names": [ds]},
            })

    # Suggest mixture experiments
    if len(tested_datasets) >= 2:
        suggestions.append({
            "priority": "medium",
            "type": "mixture",
            "suggestion": "Try dataset mixture: Tiny Codes 70% + MBPP 20% + HumanEval 10%",
            "rationale": "Combining datasets may improve generalization",
            "config": {
                "dataset_names": list(tested_datasets),
                "mixture": {"tiny_codes": 0.7, "mbpp": 0.2, "humaneval": 0.1},
            },
        })

    # Suggest eval on best model
    best_tokenizer = None
    best_loss = float("inf")
    for r in tokenizer_records:
        l = r.metrics.get("loss", 999)
        if l < best_loss:
            best_loss = l
            best_tokenizer = r

    best_dataset = None
    best_ds_loss = float("inf")
    for r in dataset_records:
        l = r.metrics.get("loss", 999)
        if l < best_ds_loss:
            best_ds_loss = l
            best_dataset = r

    if best_tokenizer and best_dataset:
        suggestions.append({
            "priority": "high",
            "type": "eval",
            "suggestion": "Run evaluation on best tokenizer + best dataset",
            "rationale": f"Best tokenizer ({best_tokenizer.name}, loss={best_loss:.4f}) "
                        f"+ best dataset ({best_dataset.name}, loss={best_ds_loss:.4f})",
            "config": {
                "tokenizer_name": best_tokenizer.name.split("_v")[0] if "_v" in best_tokenizer.name else "bpe",
                "dataset_name": best_dataset.name,
            },
        })

    # Suggest trying more training steps
    tokenizer_steps = [r.metrics.get("tokens_per_second", 0) for r in tokenizer_records]
    if tokenizer_steps and max(tokenizer_steps) > 0:
        suggestions.append({
            "priority": "low",
            "type": "hyperparameter",
            "suggestion": "Increase training steps (max_iters=10000)",
            "rationale": f"Current max iters may be too low for convergence; "
                        f"best token/sec = {max(tokenizer_steps):.0f}",
            "config": {"training_max_iters": 10000},
        })

    # Rank by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: priority_order.get(s["priority"], 99))
    return suggestions


def find_gaps(index: ExperimentIndex) -> dict:
    records = index.records
    gaps = {}

    # Tokenizer coverage
    tokenizer_records = _get_completed(records, exp_type="tokenizer")
    tested = set()
    for r in tokenizer_records:
        tested.add(r.name)
    missing_tokenizers = []
    for tok in DEFAULT_TOKENIZERS:
        for vs in DEFAULT_VOCAB_SIZES:
            key = f"{tok}_v{vs}"
            if key not in tested:
                missing_tokenizers.append(key)
    gaps["missing_tokenizer_variants"] = {
        "count": len(missing_tokenizers),
        "items": missing_tokenizers[:10],
    }

    # Dataset coverage
    dataset_records = _get_completed(records, exp_type="dataset")
    tested_ds = set(r.name for r in dataset_records)
    missing_datasets = [d for d in DEFAULT_DATASETS if d not in tested_ds]
    gaps["missing_datasets"] = {
        "count": len(missing_datasets),
        "items": missing_datasets,
    }

    # Eval baseline
    eval_records = _get_completed(records, exp_type="eval")
    gaps["missing_baseline_eval"] = len(eval_records) == 0

    # Mixtures not tested
    gaps["mixtures_not_tested"] = len(tested_ds) >= 2

    return gaps
