from typing import Optional

from deepzero.experiments.intelligence.index import ExperimentRecord


def compute_score(record: ExperimentRecord, weights: Optional[dict] = None) -> float:
    if weights is None:
        weights = {"loss": -0.3, "perplexity": -0.2, "syntax_valid_rate": 0.2,
                   "compile_success_rate": 0.2, "tokens_per_second": 0.1}
    score = 0.0
    for metric, weight in weights.items():
        val = record.metrics.get(metric, None)
        if val is not None:
            score += val * weight
    return score


def rank_experiments(records: list[ExperimentRecord],
                     metric: Optional[str] = None,
                     phase: Optional[str] = None,
                     exp_type: Optional[str] = None,
                     top_n: int = 10) -> list[dict]:
    filtered = [r for r in records if r.status == "completed"]
    if phase:
        filtered = [r for r in filtered if r.phase == phase]
    if exp_type:
        filtered = [r for r in filtered if r.exp_type == exp_type]

    if metric and metric in ("loss", "perplexity"):
        filtered.sort(key=lambda r: r.metrics.get(metric, 999))
        lower_better = True
    elif metric:
        filtered.sort(key=lambda r: r.metrics.get(metric, 0), reverse=True)
        lower_better = False
    else:
        filtered.sort(key=lambda r: compute_score(r), reverse=True)
        lower_better = False
        metric = "composite_score"

    ranked = []
    for i, r in enumerate(filtered, 1):
        entry = {
            "rank": i,
            "id": r.id,
            "name": r.name,
            "phase": r.phase,
            "type": r.exp_type,
            "score": r.metrics.get(metric, r.key_metric) if metric != "composite_score"
                     else compute_score(r),
            "metrics": r.metrics,
            "timestamp": r.timestamp,
        }
        ranked.append(entry)
        if len(ranked) >= top_n:
            break
    return ranked


def best_per_category(records: list[ExperimentRecord]) -> dict:
    categories = {}
    for r in records:
        if r.status != "completed":
            continue
        key = f"{r.phase}/{r.exp_type}"
        if key not in categories:
            categories[key] = {"best": None, "count": 0, "scores": []}
        categories[key]["count"] += 1
        categories[key]["scores"].append(r.key_metric)
        cs = compute_score(r)
        if categories[key]["best"] is None or cs > categories[key]["best"]["score"]:
            categories[key]["best"] = {"id": r.id, "name": r.name, "score": cs, "key_metric": r.key_metric}
    for cat in categories.values():
        scores = cat["scores"]
        cat["avg_score"] = sum(scores) / max(1, len(scores))
        cat["max_score"] = max(scores) if scores else 0
        cat["min_score"] = min(scores) if scores else 0
    return categories
