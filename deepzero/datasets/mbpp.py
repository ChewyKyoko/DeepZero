import json
import os
import random

from deepzero.datasets.base import BaseDataset


MBPP_REPO = "google-research-datasets/mbpp"
MBPP_PARQUET = "sanitized/train-00000-of-00001.parquet"


class MBPPDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/mbpp"):
        super().__init__("mbpp", cache_dir)
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "mbpp.jsonl")
        if os.path.exists(jsonl_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)

        import urllib.request
        parquet_name = MBPP_PARQUET.replace("/", "_")
        parquet_path = os.path.join(self.cache_dir, parquet_name)
        if not os.path.exists(parquet_path):
            try:
                urllib.request.urlretrieve(
                    f"https://huggingface.co/datasets/{MBPP_REPO}/resolve/main/{MBPP_PARQUET}",
                    parquet_path)
            except Exception:
                return

        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            with open(jsonl_path, "w") as f:
                for _, row in df.iterrows():
                    text = row.get("text", "")
                    if not text:
                        prompt = row.get("prompt", "")
                        code = row.get("code", "")
                        test_list = row.get("test_list", [])
                        test_setup = row.get("test_setup", "")
                        parts = [prompt, code]
                        if test_setup:
                            parts.append(test_setup)
                        parts.extend(str(t) for t in (test_list or []))
                        text = "\n".join(parts)
                    f.write(json.dumps({"text": text}) + "\n")
        except ImportError:
            raise ImportError("pandas/pyarrow required. pip install pandas pyarrow")

    def preprocess(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "mbpp.jsonl")
        if not os.path.exists(jsonl_path):
            self.download()
        if not os.path.exists(jsonl_path):
            self._texts = []
            return
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get("text", "")
                if not text:
                    continue
                texts.append(text)
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
