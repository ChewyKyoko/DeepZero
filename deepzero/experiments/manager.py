import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from deepzero.metrics.tracker import MetricsTracker
from deepzero.logging.setup import setup_logging


def _get_git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--stat"], capture_output=True, text=True, check=False
        ).stdout.strip() != ""
    except Exception:
        commit, branch, dirty = "unknown", "unknown", False
    return {"commit_hash": commit, "branch": branch, "dirty": dirty}


def _get_hardware_info() -> dict:
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    total_ram = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_ram = int(line.split()[1]) // (1024 * 1024)
                    break
    except Exception:
        pass

    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": cpu_model,
        "cpu_cores": os.cpu_count() or 0,
        "total_ram_gb": total_ram,
        "device": "cpu",
    }


def _get_env_info() -> dict:
    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "not installed"

    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch_ver,
        "hostname": platform.node(),
    }


class ExperimentManager:
    """Creates and manages a single experiment run directory.

    Responsible for:
      - Creating ``runs/YYYY-MM-DD_HH-MM-SS/``
      - Saving config, hardware, git, environment metadata as JSON
      - Providing checkpoint / plot subdirectories
      - Coordinating MetricsTracker lifecycle
      - Setting up logging to file + console
    """

    def __init__(self, config: dict, run_dir: Optional[str] = None):
        self.config = config
        self._start_time = datetime.now()
        self.run_dir = Path(run_dir or self._default_run_dir())
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsTracker(self.run_dir)
        self.logger = setup_logging(self.run_dir, name="deepzero")

        self._save_config()
        self._save_hardware()
        self._save_git()
        self._save_environment()
        self._create_subdirs()

        self.logger.info("Experiment started: %s", self.run_dir.name)

    def _default_run_dir(self) -> str:
        base = self.config.get("experiment", {}).get("base_dir", "runs")
        ts = self._start_time.strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(base, ts)

    def _create_subdirs(self):
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

    def _save_config(self):
        path = self.run_dir / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        self.logger.info("Config saved to %s", path)

    def _save_hardware(self):
        info = _get_hardware_info()
        info["seed"] = self.config.get("seed")
        path = self.run_dir / "hardware.json"
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

    def _save_git(self):
        info = _get_git_info()
        path = self.run_dir / "git.json"
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

    def _save_environment(self):
        info = _get_env_info()
        path = self.run_dir / "environment.json"
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

    def save_dataset_info(self, dataset_info: dict):
        """Save dataset versioning metadata (hash, counts, stats)."""
        path = self.run_dir / "dataset.json"
        with open(path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        self.logger.info("Dataset info saved to %s", path)

    def save_evaluation(self, eval_results: dict):
        path = self.run_dir / "evaluation.json"
        with open(path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        self.logger.info("Evaluation saved to %s", path)

    def get_checkpoint_dir(self) -> str:
        return str(self.run_dir / "checkpoints")

    def get_plots_dir(self) -> str:
        return str(self.run_dir / "plots")

    def save_summary(self, extra: Optional[dict] = None):
        summary = self.metrics.summary()
        summary["status"] = "completed"
        summary["run_dir"] = str(self.run_dir)
        summary["started_at"] = self._start_time.isoformat()
        if extra:
            summary.update(extra)
        path = self.run_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info("Summary saved to %s", path)

    def finalize(self):
        """Export metrics and save summary. Call after training completes."""
        self.metrics.to_csv(str(self.run_dir / "metrics.csv"))
        self.metrics.to_json(str(self.run_dir / "metrics.json"))
        self.save_summary()
        # Save run metadata for resume detection
        meta = {"run_dir": str(self.run_dir), "started_at": self._start_time.isoformat()}
        with open(self.run_dir / ".meta.json", "w") as f:
            json.dump(meta, f)
        self.logger.info("Experiment finalized: %s", self.run_dir.name)
