import json
import os
import random
from typing import Optional

from deepzero.datasets.base import BaseDataset
from deepzero.datasets.pipeline import _deduplicate, _filter_samples


STACK_SAMPLE_URL = "https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/main/data/train-00000-of-00001.parquet"


class TheStackDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/the_stack", max_samples: int = 100000,
                 language: str = "python", min_length: int = 100, max_length: int = 50000):
        super().__init__("the_stack", cache_dir)
        self.max_samples = max_samples
        self.language = language
        self.min_length = min_length
        self.max_length = max_length
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "stack.jsonl")
        if os.path.exists(jsonl_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            import requests
            parquet_path = os.path.join(self.cache_dir, "train-00000-of-00001.parquet")
            if not os.path.exists(parquet_path):
                resp = requests.get(STACK_SAMPLE_URL, stream=True, timeout=300)
                resp.raise_for_status()
                with open(parquet_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                df = df.head(self.max_samples)
                with open(jsonl_path, "w") as f:
                    for _, row in df.iterrows():
                        content = row.get("content", "")
                        f.write(json.dumps({"text": content}) + "\n")
            except ImportError:
                raise ImportError("pandas/pyarrow required. pip install pandas pyarrow")
        except ImportError:
            raise ImportError("requests required. pip install requests")

    def preprocess(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "stack.jsonl")
        if not os.path.exists(jsonl_path):
            self.download()
        samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        samples = _deduplicate(samples)
        samples = _filter_samples(samples, min_length=self.min_length,
                                  max_length=self.max_length, language=self.language)
        self._texts = [s["text"] for s in samples]

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
            "source": "BigCode The Stack (dedup)",
            "language": self.language,
            "max_samples": self.max_samples,
        }
