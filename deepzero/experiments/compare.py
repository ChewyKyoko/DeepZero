import json
import os
from typing import Optional


def compare_runs(run_ids: list[str], registry_dir: str = "outputs/experiments") -> dict:
    comparison = {}
    for rid in run_ids:
        manifest_path = os.path.join(registry_dir, rid, "manifest.json")
        if not os.path.exists(manifest_path):
            comparison[rid] = {"error": "not found"}
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        rounds = manifest.get("rounds", [])
        pass_rates = []
        scores = []
        for r in rounds:
            res = r.get("results", {})
            total = res.get("total", 0)
            passed = res.get("passed", 0)
            pass_rates.append(passed / max(1, total))
            scores.append(res.get("score", 0.0))
        comparison[rid] = {
            "name": manifest.get("name"),
            "status": manifest.get("status"),
            "rounds": len(rounds),
            "avg_pass_rate": sum(pass_rates) / max(1, len(pass_rates)),
            "avg_score": sum(scores) / max(1, len(scores)),
            "config": manifest.get("config"),
        }
    return comparison
