"""Failure logging system — stores wrong outputs with metadata."""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LogEntry:
    task_id: str
    task_prompt: str
    solution_code: str
    correct_solution: str
    score: float
    passed: int
    total: int
    errors: list[str]
    category: str
    difficulty: float
    timestamp: float
    iteration: int

    def is_failure(self) -> bool:
        return self.score < 0.5

    def is_near_miss(self) -> bool:
        return 0.5 <= self.score < 1.0


class FailureLogger:
    def __init__(self, path: str = "data/rl_logs.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, entry: LogEntry):
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def query(self, min_score: float = 0.0, max_score: float = 1.0,
              category: Optional[str] = None, limit: int = 1000,
              iteration: Optional[int] = None) -> list[LogEntry]:
        results = []
        if not os.path.exists(self.path):
            return results
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entry = LogEntry(**d)
                if not (min_score <= entry.score <= max_score):
                    continue
                if category and entry.category != category:
                    continue
                if iteration is not None and entry.iteration != iteration:
                    continue
                results.append(entry)
        return results[-limit:]

    def failures(self, limit: int = 500, iteration: Optional[int] = None) -> list[LogEntry]:
        return self.query(max_score=0.5, limit=limit, iteration=iteration)

    def near_misses(self, limit: int = 500, iteration: Optional[int] = None) -> list[LogEntry]:
        return self.query(min_score=0.5, max_score=1.0, limit=limit, iteration=iteration)

    def stats(self) -> dict:
        entries = self.query(limit=100000)
        if not entries:
            return {"total": 0, "failures": 0, "near_misses": 0, "passed": 0}
        failures = sum(1 for e in entries if e.is_failure())
        near_misses = sum(1 for e in entries if e.is_near_miss())
        passed = sum(1 for e in entries if e.score >= 1.0)
        return {
            "total": len(entries),
            "failures": failures,
            "near_misses": near_misses,
            "passed": passed,
            "avg_score": sum(e.score for e in entries) / len(entries),
        }

    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
