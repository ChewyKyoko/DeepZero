import os
import random
from pathlib import Path
from typing import Optional

from deepzero.datasets.base import BaseDataset


class LocalDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/local", path: str = "", ext: str = ".py"):
        super().__init__("local", cache_dir)
        self.path = path
        self.ext = ext
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        pass

    def preprocess(self) -> None:
        base = Path(self.path)
        if not base.exists():
            raise FileNotFoundError(f"Local dataset path not found: {self.path}")
        texts = []
        if base.is_file():
            with open(base) as f:
                texts.append(f.read())
        else:
            for fpath in sorted(base.rglob(f"*{self.ext}")):
                try:
                    with open(fpath) as f:
                        texts.append(f.read())
                except Exception:
                    continue
        self._texts = texts

    def deduplicate(self) -> None:
        seen: set[str] = set()
        deduped = []
        for t in self._texts:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        self._texts = deduped

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
            "source_path": self.path,
            "file_extension": self.ext,
        }
