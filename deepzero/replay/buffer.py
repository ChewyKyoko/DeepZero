import json
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingExample:
    input_ids: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    task: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self, capacity: int = 1000, path: str = "outputs/replay_buffer.jsonl"):
        self.capacity = capacity
        self.path = path
        self.buffer: list[TrainingExample] = []

    def add(self, example: TrainingExample):
        self.buffer.append(example)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def save(self):
        with open(self.path, "w") as f:
            for ex in self.buffer:
                f.write(json.dumps({
                    "input_ids": ex.input_ids,
                    "labels": ex.labels,
                    "task": ex.task,
                    "score": ex.score,
                    "metadata": ex.metadata,
                }) + "\n")

    def load(self):
        import os
        if not os.path.exists(self.path):
            return
        self.buffer = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.buffer.append(TrainingExample(**d))

    @property
    def size(self) -> int:
        return len(self.buffer)
