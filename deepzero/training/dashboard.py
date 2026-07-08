import shutil
import time


def render_dashboard(
    step: int,
    max_steps: int,
    loss: float,
    val_loss: float | None,
    perplexity: float | None,
    grad_norm: float,
    lr: float,
    opt_name: str,
    tok_name: str,
    tok_speed: float,
    samp_speed: float,
    ram_mb: float,
    elapsed: float,
    best_val_loss: float | None = None,
    patience_remaining: int | None = None,
    extra: str = "",
) -> str:
    """Return a formatted live-training dashboard string."""
    cols, _ = shutil.get_terminal_size((100, 20))

    bar_width = max(20, cols - 30)
    frac = step / max(1, max_steps)
    filled = int(frac * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    eta = ""
    if step > 0 and elapsed > 0:
        per_step = elapsed / step
        remaining = (max_steps - step) * per_step
        eta = _fmt_time(remaining)

    lines = []
    def ln(s=""):
        lines.append(s)

    ln("─" * cols)
    ln(f"  DeepZero Training  ({step}/{max_steps} steps)")
    ln(f"  {bar}  {frac:.0%}")
    ln()

    ln(f"  Loss        {loss:<8.4f}  LR           {lr:.2e}")
    val_str = f"{val_loss:.4f}" if val_loss is not None else "─"
    ln(f"  Val Loss    {val_str:<8}  Grad Norm    {grad_norm:.2f}")
    ppl_str = f"{perplexity:.1f}" if perplexity is not None else "─"
    ln(f"  PPL         {ppl_str:<8}  Best Val     {best_val_loss:.4f}" if best_val_loss else f"  PPL         {ppl_str}")
    ln()
    ln(f"  Optimizer   {opt_name:<12}  Tokenizer    {tok_name}")
    ln(f"  tok/s       {tok_speed:<8.0f}  samp/s       {samp_speed:.1f}")
    ln(f"  RAM         {ram_mb/1024:<8.1f} GB  ETA          {eta}")
    if patience_remaining is not None:
        ln(f"  Early stop  {patience_remaining} evals remaining")
    if extra:
        ln(f"  {extra}")
    ln("─" * cols)

    return "\n".join(lines)


def _fmt_time(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
