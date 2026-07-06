import json
import os
from datetime import datetime, timezone
from typing import Optional

from deepzero.experiments.intelligence.index import ExperimentIndex
from deepzero.experiments.intelligence.ranking import best_per_category, rank_experiments
from deepzero.experiments.intelligence.suggestions import suggest_next_experiment, find_gaps
from deepzero.experiments.intelligence.deadend import detect_dead_ends, compute_convergence
from deepzero.experiments.intelligence.compare import improvement_trend


def build_dashboard(index: ExperimentIndex, output_dir: str = "results") -> str:
    summary = index.summary()
    best_cats = best_per_category(index.records)
    gaps = find_gaps(index)
    dead_ends = detect_dead_ends(index)
    suggestions = suggest_next_experiment(index)
    top_ranked = rank_experiments(index.records, top_n=5)
    loss_trend = improvement_trend(index.records, metric="loss")
    eval_ranking = rank_experiments(index.records, exp_type="eval", top_n=10)

    dashboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "best_per_category": {},
        "gaps": gaps,
        "dead_ends": dead_ends,
        "suggestions": suggestions,
        "top_ranked": top_ranked,
        "convergence": {},
    }

    for key, cat_data in best_cats.items():
        if cat_data.get("best"):
            dashboard["best_per_category"][key] = {
                "best": cat_data["best"],
                "count": cat_data["count"],
                "avg_score": f"{cat_data['avg_score']:.4f}",
            }

    for phase_data in summary.get("phases", {}):
        phase_records = index.query(phase=phase_data)
        if len(phase_records) >= 3:
            conv = compute_convergence(phase_records, metric="loss")
            dashboard["convergence"][phase_data] = conv

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "research_dashboard.json")
    with open(path, "w") as f:
        json.dump(dashboard, f, indent=2)

    return path


def generate_dashboard_report(index: ExperimentIndex, output_path: Optional[str] = None) -> str:
    if output_path is None:
        output_path = "results/research_dashboard.md"

    summary = index.summary()
    best_cats = best_per_category(index.records)
    gaps = find_gaps(index)
    dead_ends = detect_dead_ends(index)
    suggestions = suggest_next_experiment(index)
    top_ranked = rank_experiments(index.records, top_n=5)
    eval_ranking = rank_experiments(index.records, exp_type="eval", top_n=10)

    lines = [
        "# DeepZero Research Dashboard",
        f"\n*Generated: {datetime.now(timezone.utc).isoformat()}*",
        "",
        "## Experiment Summary",
        f"\n- Total experiments: {summary['total_experiments']}",
        "",
        "### By Phase",
        "| Phase | Total | Completed | Failed |",
        "|-------|-------|-----------|--------|",
    ]
    for phase, data in sorted(summary.get("phases", {}).items()):
        lines.append(f"| {phase} | {data['total']} | {data['completed']} | {data['failed']} |")

    lines.extend([
        "",
        "## Best Results by Category",
        "",
        "| Category | Best Run | Score | Count |",
        "|----------|----------|-------|-------|",
    ])
    for key, data in sorted(best_cats.items()):
        if data.get("best"):
            lines.append(f"| {key} | {data['best']['name']} | {data['best']['key_metric']:.4f} | {data['count']} |")

    if suggestions:
        lines.extend([
            "",
            "## Suggested Next Experiments",
            "",
            "| Priority | Type | Suggestion | Rationale |",
            "|----------|------|------------|-----------|",
        ])
        for s in suggestions[:5]:
            lines.append(f"| {s['priority']} | {s['type']} | {s['suggestion']} | {s['rationale']} |")

    if dead_ends:
        lines.extend([
            "",
            "## Dead-End Detection",
            "",
            "| Severity | Type | Message |",
            "|----------|------|---------|",
        ])
        for d in dead_ends:
            lines.append(f"| {d['severity']} | {d['type']} | {d['message']} |")

    lines.extend([
        "",
        "## Gaps in Coverage",
        "",
    ])
    for gap_key, gap_data in gaps.items():
        if isinstance(gap_data, dict):
            lines.append(f"- {gap_key}: {gap_data.get('count', gap_data.get('items', '?'))}")
        elif gap_data:
            lines.append(f"- {gap_key}: needs attention")

    if top_ranked:
        lines.extend([
            "",
            "## Top 5 Experiments (Composite Score)",
            "",
            "| Rank | Name | Phase | Score |",
            "|------|------|-------|-------|",
        ])
        for r in top_ranked:
            lines.append(f"| #{r['rank']} | {r['name']} | {r['phase']} | {r['score']:.4f} |")

    if eval_ranking:
        lines.extend([
            "",
            "## Evaluation Ranking",
            "",
            "| Rank | Experiment | Aggregate Score |",
            "|------|-----------|-----------------|",
        ])
        for r in eval_ranking:
            lines.append(f"| #{r['rank']} | {r['name'][:30]} | {r['score']:.1f} |")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    return output_path
