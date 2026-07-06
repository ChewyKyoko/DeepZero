import csv
import json
import os
from datetime import datetime, timezone
from typing import Optional


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _sf(r: dict, *keys, default=0.0):
    d = r
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


def generate_dataset_csv(results_path: str, output_path: Optional[str] = None) -> str:
    results = load_results(results_path)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(results_path), "dataset_comparison.csv")

    fields = [
        "experiment_id", "dataset_name", "status",
        "final_loss", "final_perplexity", "tokens_per_second",
        "training_time_seconds", "best_val_loss",
        "syntax_valid_rate", "compile_success_rate", "avg_output_length",
        "avg_repetition_rate_3gram",
        "avg_n_functions", "avg_n_classes",
        "n_samples", "avg_length", "total_chars",
        "syntax_valid_rate_data", "estimated_tokens",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            qa = r.get("quality_analysis", {})
            row = {
                "experiment_id": r.get("experiment_id", ""),
                "dataset_name": r.get("dataset_name", ""),
                "status": r.get("status", ""),
                "final_loss": _sf(r, "training_stats", "final_loss"),
                "final_perplexity": _sf(r, "training_stats", "final_perplexity"),
                "tokens_per_second": _sf(r, "training_stats", "tokens_per_second"),
                "training_time_seconds": _sf(r, "training_stats", "training_time_seconds"),
                "best_val_loss": _sf(r, "training_stats", "best_val_loss"),
                "syntax_valid_rate": _sf(r, "generation_stats", "syntax_valid_rate"),
                "compile_success_rate": _sf(r, "generation_stats", "compile_success_rate"),
                "avg_output_length": _sf(r, "generation_stats", "avg_output_length"),
                "avg_repetition_rate_3gram": _sf(r, "generation_stats", "avg_repetition_rate_3gram"),
                "avg_n_functions": _sf(r, "coding_quality", "avg_n_functions"),
                "avg_n_classes": _sf(r, "coding_quality", "avg_n_classes"),
                "n_samples": qa.get("n_samples", 0),
                "avg_length": qa.get("avg_length", 0),
                "total_chars": qa.get("total_chars", 0),
                "syntax_valid_rate_data": qa.get("syntax_valid_rate", 0),
                "estimated_tokens": qa.get("estimated_tokens", 0),
            }
            w.writerow(row)
    print(f"CSV: {output_path}")
    return output_path


def generate_dataset_report(results_path: str, output_path: Optional[str] = None) -> str:
    results = load_results(results_path)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(results_path), "dataset_report.md")

    completed = [r for r in results if r.get("status") == "completed"]
    lines = [
        "# Dataset Benchmark Report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        f"- Total experiments: {len(results)}",
        f"- Completed: {len(completed)}",
        f"- Failed: {len(results) - len(completed)}",
        "",
    ]

    if completed:
        lines.append("## Results Table")
        lines.append("\n| Dataset | Loss | PPL | Tok/s | Syntax | Compile | Samples | Avg Len | Rank |")
        lines.append("|---------|------|-----|-------|--------|---------|---------|---------|------|")

        def rank_key(r):
            return _sf(r, "training_stats", "final_loss")
        ranked = sorted(completed, key=rank_key)
        for i, r in enumerate(ranked, 1):
            name = r.get("dataset_name", "?")
            loss = f"{_sf(r, 'training_stats', 'final_loss'):.4f}"
            ppl = f"{_sf(r, 'training_stats', 'final_perplexity'):.2f}"
            tps = f"{_sf(r, 'training_stats', 'tokens_per_second'):.0f}"
            syn = f"{_sf(r, 'generation_stats', 'syntax_valid_rate'):.0%}"
            cmp = f"{_sf(r, 'generation_stats', 'compile_success_rate'):.0%}"
            ns = r.get("quality_analysis", {}).get("n_samples", 0)
            al = f"{r.get('quality_analysis', {}).get('avg_length', 0):.0f}"
            lines.append(f"| {name} | {loss} | {ppl} | {tps} | {syn} | {cmp} | {ns:,} | {al} | #{i} |")

        lines.append("\n## Quality Analysis Comparison")
        lines.append("\n| Dataset | Syntax Valid % | Functions/Sample | Classes/Sample | Comment % | Est. Tokens |")
        lines.append("|---------|---------------|------------------|----------------|-----------|-------------|")
        for r in ranked:
            qa = r.get("quality_analysis", {})
            name = r.get("dataset_name", "?")
            sv = f"{qa.get('syntax_valid_rate', 0):.1%}"
            fs = f"{qa.get('avg_functions_per_sample', 0):.2f}"
            cs = f"{qa.get('avg_classes_per_sample', 0):.2f}"
            cr = f"{qa.get('avg_comment_ratio', 0):.1%}"
            et = f"{qa.get('estimated_tokens', 0):,}"
            lines.append(f"| {name} | {sv} | {fs} | {cs} | {cr} | {et} |")

    lines.append("\n## Configuration")
    if completed:
        cfg = completed[0].get("config", {})
        lines.append(f"\n```json\n{json.dumps(cfg, indent=2)}\n```")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report: {output_path}")
    return output_path


def generate_dataset_plots(results_path: str, output_dir: Optional[str] = None) -> dict:
    results = load_results(results_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_path), "dataset_plots")
    os.makedirs(output_dir, exist_ok=True)
    completed = [r for r in results if r.get("status") == "completed"]
    generated = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r.get("dataset_name", "?") for r in completed]
        losses = [_sf(r, "training_stats", "final_loss") for r in completed]
        ppls = [_sf(r, "training_stats", "final_perplexity") for r in completed]
        tps = [_sf(r, "training_stats", "tokens_per_second") for r in completed]
        syn = [_sf(r, "generation_stats", "syntax_valid_rate") for r in completed]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Dataset Benchmark Results", fontsize=14)

        def bar(ax, x, v, t, yl, c="steelblue"):
            ax.bar(range(len(x)), v, color=c)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, rotation=30, ha="right", fontsize=9)
            ax.set_title(t)
            ax.set_ylabel(yl)

        bar(axes[0, 0], names, losses, "Final Loss", "Loss", "coral")
        bar(axes[0, 1], names, ppls, "Perplexity", "PPL", "coral")
        bar(axes[1, 0], names, tps, "Tokens/sec", "Speed", "forestgreen")
        bar(axes[1, 1], names, syn, "Syntax Validity Rate", "Rate", "steelblue")

        plt.tight_layout()
        p = os.path.join(output_dir, "dataset_benchmark.png")
        plt.savefig(p, dpi=150)
        plt.close()
        generated["overview"] = p

        # Quality comparison
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        qa = [r.get("quality_analysis", {}) for r in completed]
        sample_counts = [q.get("n_samples", 0) for q in qa]
        avg_lens = [q.get("avg_length", 0) for q in qa]
        x = range(len(names))
        ax2.bar([xi - 0.2 for xi in x], [s / max(1, max(sample_counts)) for s in sample_counts],
                0.4, label="Samples (norm)", color="steelblue")
        ax2.bar([xi + 0.2 for xi in x], [l / max(1, max(avg_lens)) for l in avg_lens],
                0.4, label="Avg Length (norm)", color="coral")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(names, rotation=30, ha="right")
        ax2.set_title("Dataset Size vs Average Length")
        ax2.legend()

        plt.tight_layout()
        p2 = os.path.join(output_dir, "dataset_quality.png")
        plt.savefig(p2, dpi=150)
        plt.close()
        generated["quality"] = p2

        print(f"Plots: {output_dir}")
    except ImportError:
        print("matplotlib not available")
        generated["error"] = "matplotlib not installed"

    return generated


def full_dataset_report(results_path: str, output_dir: Optional[str] = None) -> dict:
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    csv_p = generate_dataset_csv(results_path)
    md_p = generate_dataset_report(results_path)
    plots = generate_dataset_plots(results_path, os.path.join(output_dir, "dataset_plots"))
    return {"csv": csv_p, "report": md_p, "plots": plots}
