import os

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve(path: str, relative_to: str) -> str:
    """Resolve path relative to ``relative_to``, then fall back to CWD."""
    if os.path.isabs(path):
        return path
    resolved = os.path.join(os.path.dirname(os.path.abspath(relative_to)), path)
    if os.path.exists(resolved):
        return resolved
    return os.path.abspath(path)


def load_config(path: str) -> dict:
    """Load YAML config, resolving ``base:`` inheritance chains."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if "base" in cfg:
        base_path = _resolve(cfg["base"], path)
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
        cfg.pop("base", None)

    return cfg
