import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ExperimentRecord:
    id: str
    phase: str = ""
    exp_type: str = ""
    name: str = ""
    key_metric: float = 0.0
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    git_hash: str = ""
    timestamp: str = ""
    status: str = ""
    path: str = ""
    tags: list = field(default_factory=list)

    @property
    def timestamp_dt(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.timestamp)
        except (ValueError, TypeError):
            return None


class ExperimentIndex:
    def __init__(self, index_path: str = "results/experiment_index.jsonl"):
        self.index_path = index_path
        self.records: list[ExperimentRecord] = []
        self._dirty = False
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self.records.append(ExperimentRecord(**data))
        self.rebuild_from_disk()

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r.__dict__) + "\n")
        self._dirty = False

    def add(self, record: ExperimentRecord):
        existing = [r for r in self.records if r.id == record.id]
        if not existing:
            self.records.append(record)
            self._dirty = True
            self._save()

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        for r in self.records:
            if r.id == experiment_id:
                return r
        return None

    def query(self, phase: Optional[str] = None, exp_type: Optional[str] = None,
              status: Optional[str] = None, name: Optional[str] = None,
              min_score: Optional[float] = None, max_score: Optional[float] = None,
              limit: Optional[int] = None) -> list[ExperimentRecord]:
        results = list(self.records)
        if phase:
            results = [r for r in results if r.phase == phase]
        if exp_type:
            results = [r for r in results if r.exp_type == exp_type]
        if status:
            results = [r for r in results if r.status == status]
        if name:
            results = [r for r in results if name.lower() in r.name.lower()]
        if min_score is not None:
            results = [r for r in results if r.key_metric >= min_score]
        if max_score is not None:
            results = [r for r in results if r.key_metric <= max_score]
        results.sort(key=lambda r: r.key_metric, reverse=True)
        if limit:
            results = results[:limit]
        return results

    def rebuild_from_disk(self, results_dir: str = "results"):
        found_ids = {r.id for r in self.records}
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                if f == "result.json":
                    path = os.path.join(root, f)
                    if path.endswith("experiment_index.jsonl"):
                        continue
                    try:
                        with open(path) as fh:
                            data = json.load(fh)
                        eid = data.get("experiment_id", "")
                        if not eid or eid in found_ids:
                            continue
                        record = self._parse_result(eid, data, path)
                        self.records.append(record)
                        found_ids.add(eid)
                    except (json.JSONDecodeError, KeyError):
                        continue
        combined_path = os.path.join(results_dir, "all_results.json")
        if os.path.exists(combined_path):
            try:
                with open(combined_path) as fh:
                    all_data = json.load(fh)
                if isinstance(all_data, list):
                    for entry in all_data:
                        eid = entry.get("experiment_id", "")
                        if eid and eid not in found_ids:
                            record = self._parse_result(eid, entry, combined_path)
                            self.records.append(record)
                            found_ids.add(eid)
            except (json.JSONDecodeError, KeyError):
                pass
        self._save()

    @staticmethod
    def _parse_result(eid: str, data: dict, path: str) -> ExperimentRecord:
        name = data.get("tokenizer_name") or data.get("dataset_name") or data.get("experiment_id", "")
        phase = "R0.2"
        exp_type = "tokenizer"
        key_metric = 0.0
        metrics = {}

        if "tokenizer_name" in data:
            phase = "R0.2"
            exp_type = "tokenizer"
            key_metric = _sf(data, "training_stats", "final_loss", default=999)
            metrics = {
                "loss": key_metric,
                "perplexity": _sf(data, "training_stats", "final_perplexity"),
                "tokens_per_second": _sf(data, "training_stats", "tokens_per_second"),
                "vocab_size": data.get("vocab_size", 0),
                "syntax_valid_rate": _sf(data, "generation_stats", "syntax_valid_rate"),
                "compile_success_rate": _sf(data, "generation_stats", "compile_success_rate"),
                "compression_ratio": _sf(data, "tokenizer_stats", "compression_ratio"),
            }
        elif "dataset_name" in data:
            phase = "R0.3"
            exp_type = "dataset"
            key_metric = _sf(data, "training_stats", "final_loss", default=999)
            metrics = {
                "loss": key_metric,
                "perplexity": _sf(data, "training_stats", "final_perplexity"),
                "tokens_per_second": _sf(data, "training_stats", "tokens_per_second"),
                "n_samples": _sf(data, "quality_analysis", "n_samples"),
                "avg_length": _sf(data, "quality_analysis", "avg_length"),
                "syntax_valid_rate": _sf(data, "generation_stats", "syntax_valid_rate"),
                "compile_success_rate": _sf(data, "generation_stats", "compile_success_rate"),
            }
        elif "aggregate_score" in data:
            phase = "R0.4"
            exp_type = "eval"
            key_metric = data.get("aggregate_score", 0)
            metrics = {
                "aggregate_score": key_metric,
                "n_tasks": data.get("n_tasks", 0),
                "model_params": data.get("model_params", 0),
            }

        tags = [phase, exp_type, name]
        return ExperimentRecord(
            id=eid, phase=phase, exp_type=exp_type, name=name,
            key_metric=key_metric, metrics=metrics, config=data.get("config", {}),
            git_hash=data.get("git_commit_hash", ""),
            timestamp=data.get("timestamp", ""),
            status=data.get("status", ""),
            path=path, tags=tags,
        )

    def summary(self) -> dict:
        phases = {}
        for r in self.records:
            p = r.phase or "unknown"
            if p not in phases:
                phases[p] = {"total": 0, "completed": 0, "failed": 0}
            phases[p]["total"] += 1
            if r.status == "completed":
                phases[p]["completed"] += 1
            elif r.status == "failed":
                phases[p]["failed"] += 1
        return {
            "total_experiments": len(self.records),
            "phases": phases,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }


def _sf(d: dict, *keys, default=0.0):
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
