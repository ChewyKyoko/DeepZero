import hashlib
import json
import os
import random

from deepzero.datasets.base import BaseDataset
from deepzero.datasets.pipeline import (
    _download_parquet, _load_raw_jsonl, _deduplicate,
    _filter_samples, _train_val_test_split, MIN_CODE_LENGTH, MAX_CODE_LENGTH,
)


class TinyCodesDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/tiny-codes", language: str = "python",
                 min_length: int = MIN_CODE_LENGTH, max_length: int = MAX_CODE_LENGTH):
        super().__init__("tiny_codes", cache_dir)
        self.language = language
        self.min_length = min_length
        self.max_length = max_length
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        _download_parquet(self.cache_dir)

    def preprocess(self) -> None:
        raw_path = os.path.join(self.cache_dir, "tiny-codes.jsonl")
        if not os.path.exists(raw_path):
            self.download()
        if not os.path.exists(raw_path):
            self._texts = []
            self._meta = {"n_raw": 0, "language": self.language, "error": "download_failed"}
            return
        samples = _load_raw_jsonl(raw_path)
        samples = _deduplicate(samples)
        samples = _filter_samples(samples, min_length=self.min_length,
                                  max_length=self.max_length, language=self.language,
                                  skip_compile_check=True)
        self._texts = [s["text"] for s in samples]
        self._meta = {"n_raw": len(samples), "language": self.language}

    def deduplicate(self) -> None:
        seen: set[str] = set()
        deduped = []
        for t in self._texts:
            h = hashlib.md5(t.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
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
            "language": self.language,
            "min_length": self.min_length,
            "max_length": self.max_length,
        }
