import json
import os
from datetime import datetime, timezone
from typing import Optional


def generate_research_doc(results_path: str, output_path: Optional[str] = None) -> str:
    with open(results_path) as f:
        results = json.load(f)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    completed = [r for r in results if r.get("status") == "completed"]

    if output_path is None:
        base = os.path.join(os.path.dirname(os.path.dirname(results_path)), "research")
        os.makedirs(base, exist_ok=True)
        output_path = os.path.join(base, "R0.3_dataset_study.md")

    def _sf(r, *keys, default=0.0):
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

    # Rank by loss
    ranked = sorted(completed, key=lambda r: _sf(r, "training_stats", "final_loss"))
    best = ranked[0] if ranked else None
    worst = ranked[-1] if ranked else None

    lines = [
        "# R0.3 Dataset Study — DeepZero Research Report",
        "",
        f"**Date**: {timestamp}",
        f"**Experiment Count**: {len(results)} ({len(completed)} completed)",
        "",
        "---",
        "",
        "## 1. Objective",
        "",
        "To scientifically determine which training dataset produces the best coding language model",
        "on consumer CPU hardware. The model architecture, optimizer, learning rate schedule, and",
        "training loop remain fixed — only the dataset is varied.",
        "",
        "## 2. Dataset Description",
        "",
        "| Dataset | Description |",
        "|---------|-------------|",
    ]

    for r in completed:
        name = r.get("dataset_name", "?")
        qa = r.get("quality_analysis", {})
        lines.append(f"| **{name}** | {qa.get('n_samples', 0):,} samples, "
                     f"avg {qa.get('avg_length', 0):.0f} chars/sample, "
                     f"~{qa.get('estimated_tokens', 0):,} estimated tokens |")

    lines.extend([
        "",
        "## 3. Configuration",
        "",
        "```json",
    ])
    if completed:
        lines.append(json.dumps(completed[0].get("config", {}), indent=2))
    lines.extend([
        "```",
        "",
        "## 4. Metrics",
        "",
        "### Training",
        "",
        "| Dataset | Loss | Perplexity | Tokens/sec | Training Time (s) |",
        "|---------|------|------------|------------|-------------------|",
    ])

    for r in ranked:
        name = r.get("dataset_name", "?")
        loss = f"{_sf(r, 'training_stats', 'final_loss'):.4f}"
        ppl = f"{_sf(r, 'training_stats', 'final_perplexity'):.2f}"
        tps = f"{_sf(r, 'training_stats', 'tokens_per_second'):.0f}"
        tt = f"{_sf(r, 'training_stats', 'training_time_seconds'):.1f}"
        lines.append(f"| {name} | {loss} | {ppl} | {tps} | {tt} |")

    lines.extend([
        "",
        "### Generation",
        "",
        "| Dataset | Syntax Valid | Compile Success | Avg Output Len | Repetition Rate |",
        "|---------|-------------|-----------------|----------------|-----------------|",
    ])
    for r in ranked:
        name = r.get("dataset_name", "?")
        syn = f"{_sf(r, 'generation_stats', 'syntax_valid_rate'):.0%}"
        cmp = f"{_sf(r, 'generation_stats', 'compile_success_rate'):.0%}"
        aol = f"{_sf(r, 'generation_stats', 'avg_output_length'):.1f}"
        rep = f"{_sf(r, 'generation_stats', 'avg_repetition_rate_3gram'):.2f}"
        lines.append(f"| {name} | {syn} | {cmp} | {aol} | {rep} |")

    lines.extend([
        "",
        "### Coding Quality",
        "",
        "| Dataset | Avg Functions | Avg Lines | Exec Success |",
        "|---------|--------------|-----------|--------------|",
    ])
    for r in ranked:
        name = r.get("dataset_name", "?")
        af = f"{_sf(r, 'coding_quality', 'avg_n_functions'):.2f}"
        al = f"{_sf(r, 'coding_quality', 'avg_n_lines'):.1f}"
        es = f"{_sf(r, 'coding_quality', 'exec_success_rate'):.0%}"
        lines.append(f"| {name} | {af} | {al} | {es} |")

    if best:
        lines.extend([
            "",
            "## 5. Graphs",
            "",
            "See `results/dataset_plots/` for visual comparisons.",
            "",
        ])

    lines.extend([
        "",
        "## 6. Strengths",
        "",
    ])
    if best:
        bname = best.get("dataset_name", "?")
        lines.append(f"- **{bname}** achieved the lowest loss ({_sf(best, 'training_stats', 'final_loss'):.4f})")
        lines.append(f"- Highest training speed: {max(_sf(r, 'training_stats', 'tokens_per_second') for r in completed):.0f} tok/s")

    lines.extend([
        "",
        "## 7. Weaknesses",
        "",
    ])
    if worst:
        wname = worst.get("dataset_name", "?")
        wloss = _sf(worst, "training_stats", "final_loss")
        gen_syn = _sf(worst, "generation_stats", "syntax_valid_rate")
        lines.append(f"- **{wname}** had the highest loss ({wloss:.4f})")
        if gen_syn < 0.5:
            lines.append(f"- Generation syntax quality was low ({gen_syn:.0%}) for some datasets")
        lines.append("- Small datasets may lead to overfitting or poor generalization")

    lines.extend([
        "",
        "## 8. Conclusion",
        "",
    ])
    if best:
        bname = best.get("dataset_name", "?")
        bl = _sf(best, "training_stats", "final_loss")
        bp = _sf(best, "training_stats", "final_perplexity")
        lines.append(f"The **{bname}** dataset produced the best overall results with "
                     f"loss={bl:.4f} and perplexity={bp:.2f}.")
    lines.extend([
        "Larger, diverse datasets with high syntax validity tend to produce better coding models.",
        "Dataset quality (deduplication, syntax filtering) matters as much as quantity.",
        "",
        "## 9. Recommended Next Experiment",
        "",
        "- **R0.4**: Dataset mixing — combine top datasets with weighted sampling",
        "- **R1.0**: Architecture improvements using the optimal dataset from R0.3",
        "- **RN1**: Reinforcement learning from replay buffer on the best dataset",
        "",
        "---",
        "",
        "*Generated automatically by DeepZero R0.3 Dataset Research Framework*",
    ])

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Research doc: {output_path}")
    return output_path
