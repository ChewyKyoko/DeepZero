import hashlib
import json
import os
import random

from deepzero.datasets.base import BaseDataset
from deepzero.datasets.pipeline import MIN_CODE_LENGTH


TINY_TEXTBOOKS_REPO = "nampdn-ai/tiny-textbooks"
PARQUET_FILES = [
    "tiny-textbooks/train-00000-of-00001.parquet",
    "tiny-textbooks/test-00000-of-00001.parquet",
]


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path) as f:
                token = f.read().strip()
    return token


class TinyTextbooksDataset(BaseDataset):
    def __init__(self, cache_dir: str = "data/tiny-textbooks",
                 min_length: int = MIN_CODE_LENGTH):
        super().__init__("tiny_textbooks", cache_dir)
        self.min_length = min_length
        self.splits: dict[str, list[str]] = {}

    def download(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "tiny-textbooks.jsonl")
        if os.path.exists(jsonl_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)

        token = _hf_token()
        import urllib.request
        base_url = f"https://huggingface.co/datasets/{TINY_TEXTBOOKS_REPO}/resolve/main"
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        all_dfs = []
        for fname in PARQUET_FILES:
            dest = os.path.join(self.cache_dir, os.path.basename(fname))
            if not os.path.exists(dest):
                try:
                    req = urllib.request.Request(f"{base_url}/{fname}", headers=headers)
                    with urllib.request.urlopen(req) as resp:
                        length = int(resp.headers.get("Content-Length", 0))
                        print(f"  downloading {fname} ({length/1e6:.0f}MB)...", flush=True)
                        with open(dest, "wb") as f:
                            while True:
                                chunk = resp.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                except Exception as e:
                    print(f"  download failed for {fname}: {e}")
                    continue
            try:
                import pandas as pd
                all_dfs.append(pd.read_parquet(dest))
            except ImportError:
                raise ImportError("pandas/pyarrow required. pip install pandas pyarrow")

        if all_dfs:
            import pandas as pd
            df = pd.concat(all_dfs, ignore_index=True)
            with open(jsonl_path, "w") as f:
                for _, row in df.iterrows():
                    text = row.get("text", row.get("content", ""))
                    if text:
                        f.write(json.dumps({"text": text}) + "\n")
            print(f"  downloaded {len(df)} samples to {jsonl_path}")
        else:
            print("  no files downloaded, will use fallback")

    def preprocess(self) -> None:
        jsonl_path = os.path.join(self.cache_dir, "tiny-textbooks.jsonl")
        if not os.path.exists(jsonl_path):
            self.download()
        if not os.path.exists(jsonl_path):
            self._texts = []
            self._meta = {"n_raw": 0, "error": "download_failed"}
            return
        samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    if len(text) >= self.min_length:
                        samples.append(text)
        self._texts = samples
        self._meta = {"n_raw": len(samples)}

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
            "min_length": self.min_length,
        }
