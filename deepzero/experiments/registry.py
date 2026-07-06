import json
import os
import time
from typing import Optional


class ExperimentRegistry:
    def __init__(self, base_dir: str = "outputs/experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_run(self, name: str, description: str = "", config: dict = None) -> str:
        run_id = f"{name}_{int(time.time())}"
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "name": name,
            "description": description,
            "config": config or {},
            "created_at": time.time(),
            "status": "running",
            "rounds": [],
        }
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return run_id

    def log_round(self, run_id: str, round_num: int, results: dict):
        run_dir = os.path.join(self.base_dir, run_id)
        manifest_path = os.path.join(run_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["rounds"].append({"round": round_num, "results": results})
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(run_dir, f"round_{round_num}.json"), "w") as f:
            json.dump(results, f, indent=2)

    def finalize_run(self, run_id: str):
        run_dir = os.path.join(self.base_dir, run_id)
        manifest_path = os.path.join(run_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["status"] = "completed"
        manifest["completed_at"] = time.time()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def list_runs(self) -> list[dict]:
        runs = []
        for d in os.listdir(self.base_dir):
            manifest_path = os.path.join(self.base_dir, d, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    runs.append(json.load(f))
        return sorted(runs, key=lambda r: r.get("created_at", 0), reverse=True)
