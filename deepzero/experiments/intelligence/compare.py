import math
from typing import Optional

from deepzero.experiments.intelligence.index import ExperimentRecord


def safe_get(d: dict, *keys, default=0.0):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
        else:
            return default
    if isinstance(d, (int, float)):
        return d
    if isinstance(d, str):
        try:
            return float(d)
        except (ValueError, TypeError):
            return default
    return default


def compare_pair(a: ExperimentRecord, b: ExperimentRecord) -> dict:
    improvement_pct = 0.0
    if b.key_metric != 0:
        improvement_pct = ((a.key_metric - b.key_metric) / abs(b.key_metric)) * 100

    metric_diffs = {}
    all_keys = set(a.metrics.keys()) | set(b.metrics.keys())
    for k in sorted(all_keys):
        va = a.metrics.get(k, 0)
        vb = b.metrics.get(k, 0)
        diff = va - vb
        pct = ((va - vb) / abs(vb) * 100) if vb != 0 else 0
        metric_diffs[k] = {"a": va, "b": vb, "diff": diff, "pct_change": pct}

    return {
        "a": {"id": a.id, "name": a.name, "phase": a.phase, "key_metric": a.key_metric},
        "b": {"id": b.id, "name": b.name, "phase": b.phase, "key_metric": b.key_metric},
        "improvement_pct": improvement_pct,
        "better": a.key_metric < b.key_metric if "loss" in str(a.metrics.get("loss", 0)) else a.key_metric > b.key_metric,
        "metric_diffs": metric_diffs,
    }


def compare_multiple(records: list[ExperimentRecord]) -> list[dict]:
    comparisons = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            comparisons.append(compare_pair(records[i], records[j]))
    comparisons.sort(key=lambda c: abs(c["improvement_pct"]), reverse=True)
    return comparisons


def find_best_in_phase(records: list[ExperimentRecord], phase: str,
                       metric: str = "loss") -> Optional[ExperimentRecord]:
    filtered = [r for r in records if r.phase == phase and r.status == "completed"]
    if not filtered:
        return None
    lower_better = metric in ("loss", "perplexity")
    filtered.sort(key=lambda r: r.metrics.get(metric, 999) if lower_better else -r.metrics.get(metric, 0))
    return filtered[0]


def improvement_trend(records: list[ExperimentRecord], metric: str = "loss",
                      phase: Optional[str] = None) -> list[dict]:
    filtered = [r for r in records if r.status == "completed"]
    if phase:
        filtered = [r for r in filtered if r.phase == phase]
    filtered.sort(key=lambda r: r.timestamp)
    trend = []
    for r in filtered:
        val = r.metrics.get(metric, None)
        if val is not None:
            trend.append({"id": r.id, "name": r.name, metric: val, "timestamp": r.timestamp})
    return trend
