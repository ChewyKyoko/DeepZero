import csv
import json
import os
from datetime import datetime, timezone
from typing import Optional


def load_results(results_path: str) -> list[dict]:
    with open(results_path) as f:
        return json.load(f)


def _safe_get(d: dict, *keys, default=0.0):
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


def generate_csv(results_path: str, output_path: Optional[str] = None) -> str:
    results = load_results(results_path)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(results_path), "tokenizer_comparison.csv")

    fieldnames = [
        "experiment_id", "tokenizer_name", "vocab_size", "status",
        "final_loss", "final_perplexity", "tokens_per_second",
        "training_time_seconds", "best_val_loss",
        "syntax_valid_rate", "compile_success_rate", "avg_output_length",
        "avg_repetition_rate_3gram",
        "avg_n_functions", "avg_n_classes",
        "avg_tokens_per_sample", "compression_ratio",
        "vocab_size_actual", "n_train_tokens",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "experiment_id": r.get("experiment_id", ""),
                "tokenizer_name": r.get("tokenizer_name", ""),
                "vocab_size": r.get("vocab_size", ""),
                "status": r.get("status", ""),
                "final_loss": _safe_get(r, "training_stats", "final_loss"),
                "final_perplexity": _safe_get(r, "training_stats", "final_perplexity"),
                "tokens_per_second": _safe_get(r, "training_stats", "tokens_per_second"),
                "training_time_seconds": _safe_get(r, "training_stats", "training_time_seconds"),
                "best_val_loss": _safe_get(r, "training_stats", "best_val_loss"),
                "syntax_valid_rate": _safe_get(r, "generation_stats", "syntax_valid_rate"),
                "compile_success_rate": _safe_get(r, "generation_stats", "compile_success_rate"),
                "avg_output_length": _safe_get(r, "generation_stats", "avg_output_length"),
                "avg_repetition_rate_3gram": _safe_get(r, "generation_stats", "avg_repetition_rate_3gram"),
                "avg_n_functions": _safe_get(r, "coding_quality", "avg_n_functions"),
                "avg_n_classes": _safe_get(r, "coding_quality", "avg_n_classes"),
                "avg_tokens_per_sample": _safe_get(r, "tokenizer_stats", "avg_tokens_per_sample"),
                "compression_ratio": _safe_get(r, "tokenizer_stats", "compression_ratio"),
                "vocab_size_actual": _safe_get(r, "tokenizer_stats", "vocab_size_actual"),
                "n_train_tokens": _safe_get(r, "tokenizer_stats", "n_train_tokens"),
            }
            writer.writerow(row)
    print(f"CSV saved to {output_path}")
    return output_path


def _rank_column(rows: list[dict], key: str, lower_is_better: bool = True) -> list[int]:
    values = [(r.get(key, 0), i) for i, r in enumerate(rows)]
    values.sort(key=lambda x: x[0], reverse=not lower_is_better)
    ranks = [0] * len(rows)
    for rank, (_, idx) in enumerate(values, 1):
        ranks[idx] = rank
    return ranks


def generate_report(results_path: str, output_path: Optional[str] = None) -> str:
    results = load_results(results_path)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(results_path), "tokenizer_report.md")

    completed = [r for r in results if r.get("status") == "completed"]

    lines = []
    lines.append("# Tokenizer Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"\n## Summary")
    lines.append(f"\n- Total experiments: {len(results)}")
    lines.append(f"- Completed: {len(completed)}")
    lines.append(f"- Failed: {len(results) - len(completed)}")

    if completed:
        _loss = [_safe_get(r, "training_stats", "final_loss") for r in completed]
        _ppl = [_safe_get(r, "training_stats", "final_perplexity") for r in completed]
        _tok_s = [_safe_get(r, "training_stats", "tokens_per_second") for r in completed]
        _syn = [_safe_get(r, "generation_stats", "syntax_valid_rate") for r in completed]
        _cmp = [_safe_get(r, "generation_stats", "compile_success_rate") for r in completed]
        _comp = [_safe_get(r, "tokenizer_stats", "compression_ratio") for r in completed]

        if _loss:
            best_idx = min(range(len(_loss)), key=lambda i: _loss[i])
            best = completed[best_idx]
            lines.append(f"\n## Best Tokenizer")
            lines.append(f"\n**{best.get('tokenizer_name', '?')}** (vocab_size={best.get('vocab_size', '?')})")
            lines.append(f"- Final loss: {_loss[best_idx]:.4f}")
            lines.append(f"- Perplexity: {_ppl[best_idx]:.2f}")
            lines.append(f"- Tokens/sec: {_tok_s[best_idx]:.0f}")
            lines.append(f"- Syntax valid rate: {_syn[best_idx]:.2%}")
            lines.append(f"- Compile success rate: {_cmp[best_idx]:.2%}")
            lines.append(f"- Compression ratio: {_comp[best_idx]:.2f}")

        lines.append("\n## Results Table")
        lines.append("\n| Tokenizer | Vocab | Loss | PPL | Tok/s | Syntax | Compile | Compression | Rank |")
        lines.append("|-----------|-------|------|-----|-------|--------|---------|-------------|------|")

        ranks = _rank_column(completed, "final_loss", lower_is_better=True)
        rows = sorted(zip(ranks, completed), key=lambda x: x[0])
        for rank, r in rows:
            name = r.get("tokenizer_name", "?")
            vs = r.get("vocab_size", "?")
            loss = f"{_safe_get(r, 'training_stats', 'final_loss'):.4f}"
            ppl = f"{_safe_get(r, 'training_stats', 'final_perplexity'):.2f}"
            tps = f"{_safe_get(r, 'training_stats', 'tokens_per_second'):.0f}"
            syn = f"{_safe_get(r, 'generation_stats', 'syntax_valid_rate'):.0%}"
            cmp = f"{_safe_get(r, 'generation_stats', 'compile_success_rate'):.0%}"
            comp = f"{_safe_get(r, 'tokenizer_stats', 'compression_ratio'):.2f}"
            lines.append(f"| {name} | {vs} | {loss} | {ppl} | {tps} | {syn} | {cmp} | {comp} | #{rank} |")

        lines.append("\n## Detailed Metrics")

        for metric_name, field_path in [
            ("Training Loss", ["training_stats", "final_loss"]),
            ("Perplexity", ["training_stats", "final_perplexity"]),
            ("Tokens/Second", ["training_stats", "tokens_per_second"]),
            ("Syntax Validity Rate", ["generation_stats", "syntax_valid_rate"]),
            ("Compile Success Rate", ["generation_stats", "compile_success_rate"]),
            ("Compression Ratio", ["tokenizer_stats", "compression_ratio"]),
            ("Avg Tokens/Sample", ["tokenizer_stats", "avg_tokens_per_sample"]),
        ]:
            values = []
            for r in completed:
                v = _safe_get(r, *field_path)
                values.append(v)
            if values:
                lines.append(f"\n### {metric_name}")
                lines.append(f"- Min: {min(values):.4f}")
                lines.append(f"- Max: {max(values):.4f}")
                lines.append(f"- Mean: {sum(values)/len(values):.4f}")
                lines.append(f"- Median: {sorted(values)[len(values)//2]:.4f}")

    lines.append("\n## Configuration")
    if completed:
        cfg = completed[0].get("config", {})
        lines.append(f"\n```json")
        lines.append(json.dumps(cfg, indent=2))
        lines.append(f"```")

    lines.append("\n## Reproducibility")
    lines.append(f"\nEach experiment has a unique `experiment_id`.")
    lines.append(f"Results include git commit hash and timestamp.")
    lines.append(f"Tokenizer configs are saved in `results/<experiment_id>/`.")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")
    return output_path


def generate_plots(results_path: str, output_dir: Optional[str] = None) -> dict:
    results = load_results(results_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_path), "tokenizer_plots")
    os.makedirs(output_dir, exist_ok=True)
    completed = [r for r in results if r.get("status") == "completed"]

    generated = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tokenizer_names = [f"{r['tokenizer_name']}_v{r['vocab_size']}" for r in completed]
        losses = [_safe_get(r, "training_stats", "final_loss") for r in completed]
        ppls = [_safe_get(r, "training_stats", "final_perplexity") for r in completed]
        tok_s = [_safe_get(r, "training_stats", "tokens_per_second") for r in completed]
        syns = [_safe_get(r, "generation_stats", "syntax_valid_rate") for r in completed]
        comps = [_safe_get(r, "tokenizer_stats", "compression_ratio") for r in completed]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Tokenizer Benchmark Results", fontsize=16)

        def bar(ax, names, values, title, ylabel, color="steelblue"):
            ax.bar(range(len(names)), values, color=color)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_title(title)
            ax.set_ylabel(ylabel)

        bar(axes[0, 0], tokenizer_names, losses, "Final Loss", "Loss", "coral")
        bar(axes[0, 1], tokenizer_names, ppls, "Perplexity", "PPL", "coral")
        bar(axes[0, 2], tokenizer_names, tok_s, "Tokens/sec", "Speed", "forestgreen")
        bar(axes[1, 0], tokenizer_names, syns, "Syntax Validity Rate", "Rate", "steelblue")
        bar(axes[1, 1], tokenizer_names, comps, "Compression Ratio", "Ratio", "steelblue")

        axes[1, 2].axis("off")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "benchmark_overview.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        generated["overview"] = plot_path

        # Per-metric bar chart
        metrics_data = {
            "Loss": losses,
            "Perplexity": ppls,
            "Tokens/sec": tok_s,
            "Syntax Rate": syns,
            "Compression": comps,
        }
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        x = range(len(tokenizer_names))
        width = 0.15
        for i, (mname, mvals) in enumerate(metrics_data.items()):
            normalized = [v / max(1, max(mvals)) for v in mvals] if max(mvals) > 0 else mvals
            ax2.bar([xi + i * width for xi in x], normalized, width, label=mname)
        ax2.set_xticks([xi + width * 2 for xi in x])
        ax2.set_xticklabels(tokenizer_names, rotation=45, ha="right", fontsize=8)
        ax2.set_title("Normalized Metrics Comparison")
        ax2.set_ylabel("Normalized Score")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        norm_plot_path = os.path.join(output_dir, "normalized_comparison.png")
        plt.savefig(norm_plot_path, dpi=150)
        plt.close()
        generated["normalized"] = norm_plot_path

        print(f"Plots saved to {output_dir}")

    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        generated["error"] = "matplotlib not installed"

    return generated


def full_report(results_path: str, output_dir: Optional[str] = None) -> dict:
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    csv_path = generate_csv(results_path)
    md_path = generate_report(results_path)
    plots = generate_plots(results_path, os.path.join(output_dir, "tokenizer_plots"))
    return {"csv": csv_path, "report": md_path, "plots": plots}
