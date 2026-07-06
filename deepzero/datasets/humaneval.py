import json
import os
import random
from typing import Optional

from deepzero.datasets.base import BaseDataset


HUMANEVAL_URL = "https://huggingface.co/datasets/openai_humaneval/resolve/main/data/humaneval-py.jsonl.gz"


class HumanEvalDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/humaneval"):
        super().__init__("humaneval", cache_dir)
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "humaneval-py.jsonl")
        if os.path.exists(jsonl_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            import requests
            import gzip
            resp = requests.get(HUMANEVAL_URL, stream=True, timeout=120)
            resp.raise_for_status()
            out_path = os.path.join(self.cache_dir, "humaneval-py.jsonl.gz")
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            with gzip.open(out_path, "rt") as gz:
                with open(jsonl_path, "w") as out:
                    out.write(gz.read())
            os.unlink(out_path)
        except ImportError:
            raise ImportError("requests required. pip install requests")

    def preprocess(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "humaneval-py.jsonl")
        if not os.path.exists(jsonl_path):
            self.download()
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                prompt = entry.get("prompt", "")
                canonical = entry.get("canonical_solution", "")
                test = entry.get("test", "")
                combined = f"{prompt}\n{canonical}\n{test}"
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
            "source": "OpenAI HumanEval",
            "n_problems": len(self._texts),
        }
