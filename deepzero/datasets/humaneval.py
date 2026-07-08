import json
import os
import random

from deepzero.datasets.base import BaseDataset


HUMANEVAL_REPO = "openai/openai_humaneval"
HUMANEVAL_PARQUET = "openai_humaneval/test-00000-of-00001.parquet"


class HumanEvalDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/humaneval"):
        super().__init__("humaneval", cache_dir)
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "humaneval-py.jsonl")
        if os.path.exists(jsonl_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)

        import urllib.request
        parquet_name = HUMANEVAL_PARQUET.split("/")[-1]
        parquet_path = os.path.join(self.cache_dir, parquet_name)
        if not os.path.exists(parquet_path):
            try:
                urllib.request.urlretrieve(
                    f"https://huggingface.co/datasets/{HUMANEVAL_REPO}/resolve/main/{HUMANEVAL_PARQUET}",
                    parquet_path)
            except Exception:
                return

        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            with open(jsonl_path, "w") as f:
                for _, row in df.iterrows():
                    prompt = row.get("prompt", "")
                    canonical = row.get("canonical_solution", "")
                    test = row.get("test", "")
                    f.write(json.dumps({"prompt": prompt, "canonical_solution": canonical, "test": test,
                                        "text": f"{prompt}\n{canonical}\n{test}"}) + "\n")
        except ImportError:
            raise ImportError("pandas/pyarrow required. pip install pandas pyarrow")

    def preprocess(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "humaneval-py.jsonl")
        if not os.path.exists(jsonl_path):
            self.download()
        if not os.path.exists(jsonl_path):
            self._texts = []
            return
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                combined = entry.get("text", "")
                if not combined:
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
