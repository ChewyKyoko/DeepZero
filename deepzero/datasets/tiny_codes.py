import json
import os
import sys
from typing import Optional


def download_tiny_codes(cache_dir: Optional[str] = None) -> str:
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tiny-codes")
    jsonl_path = os.path.join(cache_dir, "tiny-codes.jsonl")
    if os.path.exists(jsonl_path):
        print(f"tiny-codes already cached at {jsonl_path}")
        return jsonl_path
    os.makedirs(cache_dir, exist_ok=True)
    base_url = "https://huggingface.co/datasets/nampdn-ai/tiny-codes/resolve/main"
    files = ["train-00000-of-00001-3a94d4fe3e9c3220.parquet"]
    try:
        import requests
        for fname in files:
            url = f"{base_url}/{fname}"
            dest = os.path.join(cache_dir, fname)
            print(f"Downloading {url}...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        try:
            import pandas as pd
            df = pd.read_parquet(os.path.join(cache_dir, "train-00000-of-00001-3a94d4fe3e9c3220.parquet"))
            with open(jsonl_path, "w") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps({"text": row.get("text", "")}) + "\n")
            return jsonl_path
        except ImportError:
            print("pandas not available; try: pip install pandas pyarrow")
            print(f"Parquet files downloaded to {cache_dir}")
            return cache_dir
    except ImportError:
        print("requests not available; try: pip install requests pandas pyarrow")
        print(f"Manual download: {base_url}")
        return base_url


def load_tiny_codes(split: str = "train", max_samples: Optional[int] = None) -> list[str]:
    jsonl_path = download_tiny_codes()
    texts = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            entry = json.loads(line)
            texts.append(entry.get("text", ""))
    return texts
