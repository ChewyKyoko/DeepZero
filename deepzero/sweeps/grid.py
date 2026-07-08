import csv
import itertools
import os
import time
from pathlib import Path
from typing import Optional

import yaml

from deepzero.config.loader import load_config


def _expand_search(cfg: dict, prefix: str = "") -> list[tuple[str, list]]:
    """Extract ``search:`` keys into a flat list of (keypath, values) pairs."""
    params = []
    search = cfg.get("search", {})
    for key, values in search.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(values, dict) and not any(isinstance(v, (list, tuple)) for v in values.values()):
            params.extend(_expand_search(values, full_key))
        elif isinstance(values, (list, tuple)):
            params.append((full_key, values))
    return params


def _set_nested(cfg: dict, keypath: str, value):
    parts = keypath.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _make_configs(base_cfg: dict) -> list[dict]:
    """Generate all config combinations from ``search:`` section."""
    params = _expand_search(base_cfg)
    keys = [p[0] for p in params]
    values = [p[1] for p in params]

    configs = []
    for combo in itertools.product(*values):
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg.pop("search", None)
        for k, v in zip(keys, combo):
            _set_nested(cfg, k, v)
        configs.append(cfg)
    return configs


def run_grid_search(config_path: str, output_dir: str = "sweeps",
                    run_fn=None, label: Optional[str] = None) -> list[dict]:
    """Run a grid search over all ``search:`` parameter combinations.

    Args:
        config_path: YAML file with ``search:`` section.
        output_dir: Where sweep results are written.
        run_fn: Callable ``fn(config, run_dir)`` that runs one trial.
        label: Optional sweep label (defaults to timestamp).

    Returns:
        List of result dicts, one per trial.
    """
    base_cfg = load_config(config_path)
    configs = _make_configs(base_cfg)

    label = label or time.strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for i, cfg in enumerate(configs):
        trial_label = f"{label}_trial_{i:03d}"
        print("═" * 60)
        print(f"Trial {i + 1}/{len(configs)}: {trial_label}")
        print("═" * 60)
        params = {k: _get_nested(cfg, k) for k, _ in _expand_search(base_cfg)}
        for k, v in params.items():
            print(f"  {k} = {v}")

        trial_dir = out / trial_label
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        start = time.time()
        try:
            result = run_fn(cfg, str(trial_dir)) if run_fn else {}
            result["status"] = "completed"
        except Exception as e:
            result = {"status": "failed", "error": str(e)}
            print(f"  FAILED: {e}")
        wall = time.time() - start
        result["trial"] = trial_label
        result["wall_sec"] = round(wall, 1)
        result.update(params)
        results.append(result)

        with open(trial_dir / "result.json", "w") as f:
            import json
            json.dump(result, f, indent=2)

    # Leaderboard
    results.sort(key=lambda r: (r.get("best_loss") if r.get("best_loss") is not None else float("inf")))
    leaderboard_path = out / f"{label}_leaderboard.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(leaderboard_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

    _print_leaderboard(results, label)
    return results


def _get_nested(cfg: dict, keypath: str):
    d = cfg
    for p in keypath.split("."):
        d = d.get(p, {})
    return d if not isinstance(d, dict) else d


def _print_leaderboard(results: list[dict], label: str):
    print()
    print("=" * 60)
    print(f"LEADERBOARD: {label}")
    print("=" * 60)
    h = f"{'Rank':<6} {'Trial':<24} {'Final Loss':<12} {'Best Loss':<12} {'Steps':<8} {'Time(s)':<10}"
    print(h)
    print("-" * len(h))
    for i, r in enumerate(results, 1):
        fl = r.get("final_loss", "N/A")
        bl = r.get("best_loss", "N/A")
        fl_s = f"{fl:.4f}" if isinstance(fl, float) else str(fl)
        bl_s = f"{bl:.4f}" if isinstance(bl, float) else str(bl)
        print(f"{i:<6} {r.get('trial', ''):<24} {fl_s:<12} {bl_s:<12} "
              f"{r.get('steps', '?'):<8} {r.get('wall_sec', 0):<10.1f}")
