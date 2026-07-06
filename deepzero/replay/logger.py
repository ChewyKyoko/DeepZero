import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LogEntry:
    timestamp: float = 0.0
    task: str = ""
    prompt: str = ""
    response: str = ""
    passed: bool = False
    score: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)


class FailureLogger:
    def __init__(self, path: str = "outputs/failures.jsonl"):
        self.path = path

    def log(self, entry: LogEntry):
        with open(self.path, "a") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp or time.time(),
                "task": entry.task,
                "prompt": entry.prompt,
                "response": entry.response,
                "passed": entry.passed,
                "score": entry.score,
                "error": entry.error,
                "metadata": entry.metadata,
            }) + "\n")

    def load(self) -> list[LogEntry]:
        import os
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    entries.append(LogEntry(**d))
        return entries
