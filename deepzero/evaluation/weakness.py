from collections import Counter
from typing import Optional

from deepzero.evaluation.scoring_v2 import TaskScore


FAILURE_CATEGORIES = [
    "syntax_error",
    "logic_error",
    "incomplete_solution",
    "hallucinated_api",
    "infinite_loop",
    "formatting_issues",
    "unknown",
]

CATEGORY_LABELS = {
    "syntax_error": "Syntax Errors",
    "logic_error": "Logic Errors",
    "incomplete_solution": "Incomplete Solutions",
    "hallucinated_api": "Hallucinated APIs",
    "infinite_loop": "Infinite Loops",
    "formatting_issues": "Formatting Issues",
    "unknown": "Unknown",
}


def analyze_weaknesses(scores: list[TaskScore]) -> dict:
    categories = Counter()
    category_examples: dict[str, list[dict]] = {c: [] for c in FAILURE_CATEGORIES}
    per_task_type: dict[str, dict] = {}

    for s in scores:
        if s.final_score < 80.0:
            cat = s.failure_category if s.failure_category in FAILURE_CATEGORIES else "unknown"
            categories[cat] += 1
            if len(category_examples[cat]) < 3:
                category_examples[cat].append({
                    "task_id": s.task_id,
                    "prompt": s.prompt[:100],
                    "error": s.exec_error[:200] if s.exec_error else s.syntax_error[:200],
                    "score": s.final_score,
                })

        key = s.category or "unknown"
        if key not in per_task_type:
            per_task_type[key] = {"total": 0, "passed": 0, "scores": []}
        per_task_type[key]["total"] += 1
        per_task_type[key]["scores"].append(s.final_score)
        if s.exec_success and s.n_passed == s.n_tests:
            per_task_type[key]["passed"] += 1

    total_failures = sum(categories.values())
    top_5 = categories.most_common(5)

    failure_distribution = {}
    for cat, count in categories.most_common():
        failure_distribution[cat] = {
            "count": count,
            "percentage": count / max(1, total_failures) * 100,
            "label": CATEGORY_LABELS.get(cat, cat),
            "examples": category_examples.get(cat, []),
        }

    per_category_scores = {}
    for key, data in per_task_type.items():
        per_category_scores[key] = {
            "total": data["total"],
            "passed": data["passed"],
            "pass_rate": data["passed"] / max(1, data["total"]),
            "avg_score": sum(data["scores"]) / max(1, len(data["scores"])),
        }

    return {
        "total_scores": len(scores),
        "total_failures": total_failures,
        "failure_rate": total_failures / max(1, len(scores)),
        "top_5_failure_categories": [{"category": c, "count": n, "label": CATEGORY_LABELS.get(c, c)}
                                      for c, n in top_5],
        "failure_distribution": failure_distribution,
        "per_category_scores": per_category_scores,
    }


def generate_weakness_report(scores: list[TaskScore], output_path: Optional[str] = None) -> str:
    analysis = analyze_weaknesses(scores)
    lines = [
        "# Weakness Analysis Report",
        "",
        "## Overview",
        f"- Total evaluations: {analysis['total_scores']}",
        f"- Total failures: {analysis['total_failures']}",
        f"- Failure rate: {analysis['failure_rate']:.1%}",
        "",
        "## Top 5 Failure Categories",
        "",
        "| Rank | Category | Count | Percentage |",
        "|------|----------|-------|------------|",
    ]
    for i, fc in enumerate(analysis["top_5_failure_categories"], 1):
        pct = fc["count"] / max(1, analysis["total_failures"]) * 100
        lines.append(f"| #{i} | {fc['label']} | {fc['count']} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Failure Distribution",
        "",
        "| Category | Count | % of Failures |",
        "|----------|-------|---------------|",
    ])
    for cat, data in analysis["failure_distribution"].items():
        lines.append(f"| {data['label']} | {data['count']} | {data['percentage']:.1f}% |")

    lines.extend([
        "",
        "## Example Failure Cases",
        "",
    ])
    for cat, data in analysis["failure_distribution"].items():
        if data["examples"]:
            lines.append(f"### {data['label']}")
            for ex in data["examples"]:
                lines.append(f"- **{ex['task_id']}** (score: {ex['score']:.0f}): {ex['error']}")

    lines.extend([
        "",
        "## Per-Category Performance",
        "",
        "| Category | Total | Passed | Pass Rate | Avg Score |",
        "|----------|-------|--------|-----------|-----------|",
    ])
    for cat, data in analysis["per_category_scores"].items():
        lines.append(f"| {cat} | {data['total']} | {data['passed']} | {data['pass_rate']:.0%} | {data['avg_score']:.1f} |")

    report = "\n".join(lines)

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
