from typing import Optional

from deepzero.experiments.intelligence.index import ExperimentRecord, ExperimentIndex


def detect_dead_ends(index: ExperimentIndex) -> list[dict]:
    records = index.records
    dead_ends = []

    # Group experiments by type
    by_type = {}
    for r in records:
        if r.status != "completed":
            continue
        key = f"{r.phase}/{r.exp_type}"
        if key not in by_type:
            by_type[key] = []
        by_type[key].append(r)

    for key, group in by_type.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r.timestamp)
        improvements = []
        for i in range(1, len(group)):
            prev = group[i - 1].key_metric
            curr = group[i].key_metric
            if prev != 0:
                pct = ((curr - prev) / abs(prev)) * 100
                improvements.append(pct)

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            max_improvement = max(improvements)

            # Diminishing returns: last N improvements were small
            recent = improvements[-3:] if len(improvements) >= 3 else improvements
            recent_avg = sum(recent) / len(recent)

            # Flag if recent improvements are tiny and losses are converging
            if len(group) >= 3:
                losses = [r.metrics.get("loss", 999) for r in group]
                valid_losses = [l for l in losses if l < 999]
                if len(valid_losses) >= 3:
                    loss_range = max(valid_losses) - min(valid_losses)
                    if loss_range < 0.1 and abs(recent_avg) < 2.0:
                        dead_ends.append({
                            "type": key,
                            "severity": "warning",
                            "message": f"Diminishing returns in {key}: last {len(recent)} "
                                      f"experiments improved only {recent_avg:.1f}% avg",
                            "details": {
                                "n_experiments": len(group),
                                "avg_improvement": f"{avg_improvement:.1f}%",
                                "recent_avg_improvement": f"{recent_avg:.1f}%",
                                "loss_range": f"{loss_range:.4f}" if len(valid_losses) >= 3 else "N/A",
                                "experiments": [r.id for r in group],
                            },
                        })
            # Negative improvement = getting worse
            neg_improvements = [i for i in improvements if i < -5]
            if len(neg_improvements) >= 2:
                dead_ends.append({
                    "type": key,
                    "severity": "critical",
                    "message": f"Performance degradation in {key}: "
                              f"{len(neg_improvements)} experiments got worse by >5%",
                    "details": {
                        "n_regressions": len(neg_improvements),
                        "experiments": [r.id for r in group],
                    },
                })

    # Check for repeated identical configs
    config_seen = {}
    for r in records:
        cfg_str = str(sorted(r.config.items()) if isinstance(r.config, dict) else r.config)
        if cfg_str not in config_seen:
            config_seen[cfg_str] = []
        config_seen[cfg_str].append(r.id)
    for cfg_str, ids in config_seen.items():
        if len(ids) > 1:
            metrics = []
            for rid in ids:
                r = index.get(rid)
                if r:
                    metrics.append(r.key_metric)
            if len(set(metrics)) == 1 and len(ids) >= 2:
                dead_ends.append({
                    "type": "redundant",
                    "severity": "info",
                    "message": f"Redundant experiments: {len(ids)} runs with identical config",
                    "details": {"experiment_ids": ids, "same_metric": metrics[0]},
                })

    return dead_ends


def compute_convergence(records: list[ExperimentRecord],
                        metric: str = "loss") -> dict:
    completed = [r for r in records if r.status == "completed" and r.metrics.get(metric) is not None]
    if len(completed) < 3:
        return {"status": "insufficient_data", "n_experiments": len(completed)}

    completed.sort(key=lambda r: r.timestamp)
    values = [r.metrics[metric] for r in completed]

    first_third = values[:len(values) // 3]
    last_third = values[-len(values) // 3:]

    early_avg = sum(first_third) / max(1, len(first_third))
    late_avg = sum(last_third) / max(1, len(last_third))
    pct_change = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0

    is_converged = abs(pct_change) < 5 and len(values) >= 5
    is_improving = pct_change < -5
    is_degrading = pct_change > 5

    return {
        "status": "converged" if is_converged else ("improving" if is_improving else "degrading" if is_degrading else "unstable"),
        "n_experiments": len(values),
        "early_avg": early_avg,
        "late_avg": late_avg,
        "pct_change": pct_change,
        "metric": metric,
        "values": values,
    }
