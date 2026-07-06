import json
import os
import random
from typing import Optional

from deepzero.datasets.base import BaseDataset


MBPP_URL = "https://huggingface.co/datasets/mbpp/resolve/main/mbpp.jsonl"


class MBPPDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/mbpp"):
        super().__init__("mbpp", cache_dir)
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "mbpp.jsonl")
        if os.path.exists(jsonl_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            import requests
            resp = requests.get(MBPP_URL, stream=True, timeout=120)
            resp.raise_for_status()
            with open(jsonl_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
        except ImportError:
            raise ImportError("requests required. pip install requests")

    def preprocess(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "mbpp.jsonl")
        if not os.path.exists(jsonl_path):
            self.download()
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                prompt = entry.get("prompt", "")
                code = entry.get("code", "")
                test_list = entry.get("test_list", [])
                combined = f"{prompt}\n{code}\n" + "\n".join(test_list)
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
            "source": "Google MBPP",
            "n_problems": len(self._texts),
        }
