"""Replay buffer — curates training data from failures and near-misses."""

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional

from rl.logger import LogEntry


@dataclass
class TrainingExample:
    text: str
    priority: float
    source: str
    task_id: str


class ReplayBuffer:
    def __init__(self, path: str = "data/rl_buffer.jsonl", max_size: int = 1000):
        self.path = path
        self.max_size = max_size
        self._examples: list[TrainingExample] = []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._load()

    def add_from_log(self, entries: list[LogEntry], priority: float = 1.0):
        for e in entries:
            text = f"{e.task_prompt}\n```python\n{e.correct_solution}\n```"
            ex = TrainingExample(text=text, priority=priority, source=e.category, task_id=e.task_id)
            self._examples.append(ex)
        self._trim()
        self._save()

    def add_corrected(self, task_prompt: str, wrong_code: str, correct_code: str, priority: float = 1.0):
        text = f"{task_prompt}\nWrong:\n{wrong_code}\nCorrect:\n```python\n{correct_code}\n```"
        ex = TrainingExample(text=text, priority=priority, source="corrected", task_id="manual")
        self._examples.append(ex)
        self._trim()
        self._save()

    def sample(self, n: int) -> list[TrainingExample]:
        weights = [e.priority for e in self._examples]
        if not weights:
            return []
        total = sum(weights)
        probs = [w / total for w in weights]
        sampled = random.choices(self._examples, weights=probs, k=min(n, len(self._examples)))
        return sampled

    def build_texts(self, n: Optional[int] = None) -> list[str]:
        if n is None:
            return [e.text for e in self._examples]
        return [e.text for e in self.sample(n)]

    def size(self) -> int:
        return len(self._examples)

    def stats(self) -> dict:
        if not self._examples:
            return {"total": 0}
        sources = {}
        for e in self._examples:
            sources[e.source] = sources.get(e.source, 0) + 1
        return {"total": len(self._examples), "by_source": sources}

    def clear(self):
        self._examples = []
        if os.path.exists(self.path):
            os.remove(self.path)

    def _trim(self):
        if len(self._examples) > self.max_size:
            self._examples.sort(key=lambda x: x.priority, reverse=True)
            self._examples = self._examples[:self.max_size]

    def _save(self):
        with open(self.path, "w") as f:
            for ex in self._examples:
                f.write(json.dumps(asdict(ex)) + "\n")

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        self._examples.append(TrainingExample(**d))
