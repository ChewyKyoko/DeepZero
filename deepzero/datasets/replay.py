import json
import os
import random
from typing import Optional

from deepzero.datasets.base import BaseDataset


class ReplayDataset(BaseDataset):
    def __init__(self, cache_dir: str = "outputs", replay_path: str = "replay_buffer.jsonl"):
        super().__init__("replay", cache_dir)
        self.replay_path = replay_path
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        pass

    def preprocess(self) -> None:
        path = os.path.join(self.cache_dir, self.replay_path)
        if not os.path.exists(path):
            self._texts = []
            return
        texts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    text = entry.get("prompt", "")
                    response = entry.get("response", "")
                    combined = f"{text}\n{response}".strip()
                    if combined:
                        texts.append(combined)
        self._texts = texts

    def deduplicate(self) -> None:
        pass

    def split(self, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
              seed: int = 42) -> dict[str, list[str]]:
        rng = random.Random(seed)
        indices = list(range(len(self._texts)))
        rng.shuffle(indices)
        n = len(self._texts)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits = {
            "train": [self._texts[i] for i in indices[:n_train]],
            "validation": [self._texts[i] for i in indices[n_train:n_train + n_val]],
            "test": [self._texts[i] for i in indices[n_train + n_val:]],
        }
        self.splits = splits
        return splits

    def statistics(self) -> dict:
        return {
            **super().statistics(),
            "source": "DeepZero Replay Buffer",
            "replay_path": self.replay_path,
        }
