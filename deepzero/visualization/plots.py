from pathlib import Path
from typing import Optional

from deepzero.metrics.tracker import MetricsTracker


def _import_plt():
    """Import matplotlib with Agg backend. Returns None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _extract(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for r in records:
        v = r.get(key)
        if v is not None:
            steps.append(r["step"])
            vals.append(v)
    return steps, vals


def plot_loss_curve(records: list[dict], path: str | Path, title: str = "Training Loss"):
    plt = _import_plt()
    if plt is None:
        return
    steps, vals = _extract(records, "loss")
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals, label="train")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_validation_loss(records: list[dict], path: str | Path):
    plt = _import_plt()
    if plt is None:
        return
    steps, vals = _extract(records, "validation_loss")
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals, marker="o", label="validation")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_learning_rate(records: list[dict], path: str | Path):
    plt = _import_plt()
    if plt is None:
        return
    steps, vals = _extract(records, "learning_rate")
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_gradient_norm(records: list[dict], path: str | Path):
    plt = _import_plt()
    if plt is None:
        return
    steps, vals = _extract(records, "gradient_norm")
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_ram_usage(records: list[dict], path: str | Path):
    plt = _import_plt()
    if plt is None:
        return
    steps, peak = _extract(records, "peak_ram")
    _, curr = _extract(records, "current_ram")
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, peak, label="peak")
    if curr:
        ax.plot(steps, curr, label="current")
    ax.set_xlabel("Step")
    ax.set_ylabel("RAM (MB)")
    ax.set_title("RAM Usage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_tokens_per_second(records: list[dict], path: str | Path):
    plt = _import_plt()
    if plt is None:
        return
    steps, vals = _extract(records, "tokens_per_second")
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def generate_all_plots(tracker: MetricsTracker, output_dir: str | Path):
    """Generate all six standard plots from tracker data."""
    records = tracker.records
    if not records:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plot_funcs = [
        (plot_loss_curve, "loss_curve.png"),
        (plot_validation_loss, "validation_loss.png"),
        (plot_learning_rate, "learning_rate.png"),
        (plot_gradient_norm, "gradient_norm.png"),
        (plot_ram_usage, "ram_usage.png"),
        (plot_tokens_per_second, "tokens_per_second.png"),
    ]
    for func, name in plot_funcs:
        func(records, output / name)
