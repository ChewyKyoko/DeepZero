from pathlib import Path

from deepzero.metrics.tracker import MetricsTracker


def generate_experiment_report(
    config: dict,
    tracker: MetricsTracker,
    checkpoint_dir: str | Path,
    plot_dir: str | Path,
    run_dir: str | Path,
    extra: dict | None = None,
) -> str:
    """Generate a comprehensive experiment report as markdown."""
    summary = tracker.summary()
    records = tracker.records

    lines = []
    _h = lambda *a: lines.append(" ".join(a))
    _p = lambda *a: lines.append(" ".join(a))
    _nl = lambda: lines.append("")

    # ── Title ──
    _h("# Experiment Report")
    _nl()

    # ── Hardware ──
    _h("## Hardware")
    _nl()
    try:
        import json as _json
        with open(Path(run_dir) / "hardware.json") as f:
            hw = _json.load(f)
        for k, v in hw.items():
            _p(f"- **{k}:** {v}")
        _nl()
    except Exception:
        pass

    # ── Configuration ──
    _h("## Configuration")
    _nl()
    _config_flat(config, _p)
    _nl()

    # ── Training Statistics ──
    _h("## Training Statistics")
    _nl()
    _p(f"- **Total steps:** {summary.get('total_steps', 'N/A')}")
    _p(f"- **Final loss:** {summary.get('final_loss', 'N/A'):.4f}" if summary.get("final_loss") else "- **Final loss:** N/A")
    _p(f"- **Best loss:** {summary.get('best_loss', 'N/A'):.4f}" if summary.get("best_loss") else "- **Best loss:** N/A")
    _p(f"- **Avg tokens/sec:** {summary.get('avg_tokens_per_second', 0):.0f}")
    _p(f"- **Total time:** {_fmt_time(summary.get('total_time_sec', 0))}")
    _p(f"- **Peak RAM:** {summary.get('peak_ram_gb', 'N/A')} GB")
    _nl()

    # ── Validation Statistics ──
    val_losses = [r["validation_loss"] for r in records if r.get("validation_loss") is not None]
    if val_losses:
        _h("## Validation Statistics")
        _nl()
        _p(f"- **Best validation loss:** {min(val_losses):.4f}")
        _p(f"- **Final validation loss:** {val_losses[-1]:.4f}")
        val_perps = [r.get("perplexity") for r in records if r.get("perplexity") is not None]
        if val_perps:
            _p(f"- **Best perplexity:** {min(val_perps):.2f}")
        _nl()

    # ── Time-to-Target ──
    ttt = summary.get("time_to_target", {})
    if ttt:
        _h("## Time-to-Target Results")
        _nl()
        _p("| Target Loss | Step | Wall Clock |")
        _p("|------------|------|------------|")
        for target_str, info in sorted(ttt.items(), key=lambda x: float(x[0])):
            _p(f"| {target_str} | {info['step']} | {_fmt_time(info['elapsed'])} |")
        _nl()

    # ── Best Checkpoint ──
    _h("## Best Checkpoint")
    _nl()
    best_path = Path(checkpoint_dir) / "best.pt"
    if best_path.exists():
        size_mb = best_path.stat().st_size / (1024 * 1024)
        _p(f"- **Path:** `{best_path}`")
        _p(f"- **Size:** {size_mb:.0f} MB")
    _nl()

    # ── Plots ──
    _h("## Plots")
    _nl()
    _p("The following plots are available in the `plots/` directory:")
    _p()
    for p in sorted(Path(plot_dir).glob("*.png")):
        _p(f"- `{p.name}`")
    _nl()

    # ── Final Metrics ──
    _h("## Final Metrics")
    _nl()
    _p(f"```json")
    _p(_json_dumps(summary))
    _p(f"```")
    _nl()

    # ── Conclusions ──
    _h("## Conclusions")
    _nl()
    if summary.get("best_loss") is not None:
        _p(f"The model reached a best loss of **{summary['best_loss']:.4f}** "
           f"after **{summary['total_steps']}** steps over "
           f"**{_fmt_time(summary['total_time_sec'])}**.")
    if ttt:
        earliest = min(ttt.items(), key=lambda x: x[1]["step"])
        _p(f"First time-to-target milestone: loss ≤ {earliest[0]} at step {earliest[1]['step']} "
           f"({_fmt_time(earliest[1]['elapsed'])}).")
    _nl()

    return "\n".join(lines)


def _config_flat(d: dict, write, prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _config_flat(v, write, key)
        else:
            write(f"- **{key}:** {v}")


def _fmt_time(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _json_dumps(d: dict) -> str:
    import json
    return json.dumps(d, indent=2, default=str)
