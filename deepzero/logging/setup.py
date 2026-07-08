import logging
import sys
from pathlib import Path


def setup_logging(run_dir: str | Path | None = None,
                  name: str = "deepzero",
                  level: int = logging.INFO,
                  console: bool = True) -> logging.Logger:
    """Configure logger with file + optional console handler.

    Args:
        run_dir: If set, logs are written to ``run_dir/training.log``.
        name: Logger name.
        level: Logging level.
        console: Whether to emit to stderr.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if run_dir is not None:
        path = Path(run_dir) / "training.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(path))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
