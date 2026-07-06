import csv
import json
import os
from datetime import datetime, timezone
from typing import Optional


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def generate_benchmark_csv(eval_results: list[dict], output_path: Optional[str] = None) -> str:
    if output_path is None:
        output_path = os.path.join("results", "benchmark_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fields = [
        "experiment_id", "model_params", "n_tasks",
        "aggregate_score", "algorithms_score", "data_structures_score",
        "programming_score", "debugging_score", "projects_score",
        "status",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in eval_results:
            pcs = r.get("per_category_scores", {})
            row = {
                "experiment_id": r.get("experiment_id", ""),
                "model_params": r.get("model_params", 0),
                "n_tasks": r.get("n_tasks", 0),
                "aggregate_score": f"{r.get('aggregate_score', 0):.1f}",
                "algorithms_score": f"{pcs.get('algorithms', {}).get('avg_score', 0):.1f}",
                "data_structures_score": f"{pcs.get('data_structures', {}).get('avg_score', 0):.1f}",
                "programming_score": f"{pcs.get('programming', {}).get('avg_score', 0):.1f}",
                "debugging_score": f"{pcs.get('debugging', {}).get('avg_score', 0):.1f}",
                "projects_score": f"{pcs.get('projects', {}).get('avg_score', 0):.1f}",
                "status": r.get("status", ""),
            }
            w.writerow(row)
    print(f"CSV: {output_path}")
    return output_path


def generate_benchmark_report(eval_results: list[dict], output_path: Optional[str] = None) -> str:
    if output_path is None:
        output_path = os.path.join("results", "benchmark_report.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# DeepZero Benchmark Report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Aggregate Results",
        "",
        "| Experiment | Model Params | Tasks | Aggregate Score | Status |",
        "|------------|-------------|-------|-----------------|--------|",
    ]
    for r in eval_results:
        eid = r.get("experiment_id", "")[:20]
        mp = f"{r.get('model_params', 0):,}"
        nt = r.get("n_tasks", 0)
        ag = f"{r.get('aggregate_score', 0):.1f}"
        st = r.get("status", "")
        lines.append(f"| {eid} | {mp} | {nt} | {ag} | {st} |")

    if eval_results:
        lines.extend([
            "",
            "## Per-Category Scores",
            "",
            "| Category | Tasks | Avg Score |",
            "|----------|-------|-----------|",
        ])
        r = eval_results[-1]
        for cat, data in r.get("per_category_scores", {}).items():
            lines.append(f"| {cat} | {data.get('count', 0)} | {data.get('avg_score', 0):.1f} |")

        lines.extend([
            "",
            "## Weakness Analysis",
            "",
        ])
        wa = r.get("weakness_analysis", {})
        lines.append(f"- Failure rate: {wa.get('failure_rate', 0):.1%}")
        lines.append("")
        lines.append("| Rank | Category | Count |")
        lines.append("|------|----------|-------|")
        for fc in wa.get("top_5_failure_categories", []):
            lines.append(f"| #{wa['top_5_failure_categories'].index(fc)+1} | {fc.get('label', '')} | {fc.get('count', 0)} |")

        if r.get("task_scores"):
            lines.extend([
                "",
                "## Per-Task Results",
                "",
                "| Task | Category | Score | Syntax | Exec | Tests Passed |",
                "|------|----------|-------|--------|------|--------------|",
            ])
            for ts in r["task_scores"]:
                tid = ts.get("task_id", "?")
                cat = ts.get("category", "?")
                sc = f"{ts.get('final_score', 0):.0f}"
                syn = "✓" if ts.get("syntax_valid") else "✗"
                exe = "✓" if ts.get("exec_success") else "✗"
                tp = f"{ts.get('n_passed', 0)}/{ts.get('n_tests', 0)}"
                lines.append(f"| {tid} | {cat} | {sc} | {syn} | {exe} | {tp} |")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report: {output_path}")
    return output_path


def generate_benchmark_plots(eval_results: list[dict], output_dir: Optional[str] = None) -> dict:
    if output_dir is None:
        output_dir = os.path.join("results", "benchmark_plots")
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r.get("experiment_id", "?")[:15] for r in eval_results]
        agg_scores = [r.get("aggregate_score", 0) for r in eval_results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("DeepZero Evaluation Benchmark", fontsize=14)

        if labels:
            axes[0].bar(range(len(labels)), agg_scores, color="steelblue")
            axes[0].set_xticks(range(len(labels)))
            axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            axes[0].set_title("Aggregate Scores")
            axes[0].set_ylabel("Score (0-100)")
            axes[0].set_ylim(0, 100)

        if eval_results:
            r = eval_results[-1]
            pcs = r.get("per_category_scores", {})
            cats = list(pcs.keys())
            c_scores = [pcs[c]["avg_score"] for c in cats]
            axes[1].bar(range(len(cats)), c_scores, color="coral")
            axes[1].set_xticks(range(len(cats)))
            axes[1].set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
            axes[1].set_title("Per-Category Scores")
            axes[1].set_ylabel("Score (0-100)")
            axes[1].set_ylim(0, 100)

        plt.tight_layout()
        p = os.path.join(output_dir, "benchmark_overview.png")
        plt.savefig(p, dpi=150)
        plt.close()
        generated["overview"] = p

        # Task score distribution
        if eval_results and eval_results[-1].get("task_scores"):
            r = eval_results[-1]
            task_scores = [ts.get("final_score", 0) for ts in r["task_scores"]]
            task_ids = [ts.get("task_id", "?") for ts in r["task_scores"]]
            colors = ["forestgreen" if s >= 80 else "gold" if s >= 40 else "coral" for s in task_scores]

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.bar(range(len(task_scores)), task_scores, color=colors)
            ax2.set_xticks(range(len(task_ids)))
            ax2.set_xticklabels(task_ids, rotation=45, ha="right", fontsize=8)
            ax2.set_title("Per-Task Scores")
            ax2.set_ylabel("Score (0-100)")
            ax2.set_ylim(0, 100)
            plt.tight_layout()
            p2 = os.path.join(output_dir, "per_task_scores.png")
            plt.savefig(p2, dpi=150)
            plt.close()
            generated["per_task"] = p2

        print(f"Plots: {output_dir}")

    except ImportError:
        print("matplotlib not available")
        generated["error"] = "matplotlib not installed"

    return generated


def full_benchmark_report(eval_results: list[dict], output_dir: str = "results") -> dict:
    csv_p = generate_benchmark_csv(eval_results)
    md_p = generate_benchmark_report(eval_results)
    plots = generate_benchmark_plots(eval_results, os.path.join(output_dir, "benchmark_plots"))
    return {"csv": csv_p, "report": md_p, "plots": plots}
