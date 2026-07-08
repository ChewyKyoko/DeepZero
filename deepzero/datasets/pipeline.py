import hashlib
import json
import os
import random
from typing import Optional


LANGUAGES = {
    "python": {".py"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp"},
    "rust": {".rs"},
    "javascript": {".js", ".jsx", ".ts", ".tsx"},
    "go": {".go"},
}

LANGS_BY_EXT: dict[str, str] = {}
for lang, exts in LANGUAGES.items():
    for ext in exts:
        LANGS_BY_EXT[ext] = lang

MIN_CODE_LENGTH = 50
MAX_CODE_LENGTH = 100000


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _is_valid_python(code: str) -> bool:
    try:
        compile(code, "<test>", "exec")
        return True
    except SyntaxError:
        return False


def _normalize_whitespace(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = code.split("\n")
    lines = [l.rstrip() for l in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def _infer_language(entry: dict) -> str:
    for key in ("language", "lang", "extension"):
        val = entry.get(key, "")
        if val:
            val = val.lower()
            for lang, exts in LANGUAGES.items():
                if val == lang or any(val == ext or val == ext.lstrip(".") for ext in exts):
                    return lang
    path = (entry.get("path") or entry.get("filename") or "")
    ext = os.path.splitext(path)[1].lower()
    return LANGS_BY_EXT.get(ext, "unknown")


def _download_parquet(cache_dir: str, hf_repo: str = "nampdn-ai/tiny-codes",
                      jsonl_name: str = "tiny-codes.jsonl",
                      parquet_files: Optional[list[str]] = None) -> str:
    jsonl_path = os.path.join(cache_dir, jsonl_name)
    if os.path.exists(jsonl_path):
        return jsonl_path
    os.makedirs(cache_dir, exist_ok=True)

    if parquet_files is None:
        parquet_files = [
            "part_1_200000.parquet", "part_2_400000.parquet",
            "part_3_600000.parquet", "part_4_800000.parquet",
            "part_5_1000000.parquet", "part_6_1200000.parquet",
            "part_7_1400000.parquet", "part_8_1600000.parquet",
            "part_9_1632520.parquet",
        ]

    import urllib.request
    base_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
    all_dfs = []
    for fname in parquet_files:
        dest = os.path.join(cache_dir, fname)
        if not os.path.exists(dest):
            try:
                urllib.request.urlretrieve(f"{base_url}/{fname}", dest)
            except Exception:
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
                text = row.get("text", row.get("content", row.get("code", "")))
                lang = row.get("language", row.get("lang", ""))
                entry = {"text": text}
                if lang:
                    entry["language"] = lang
                f.write(json.dumps(entry) + "\n")
    return jsonl_path


def _load_raw_jsonl(jsonl_path: str) -> list[dict]:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _deduplicate(samples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped = []
    for s in samples:
        text = s.get("text", "")
        h = _content_hash(text)
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    return deduped


def _filter_samples(samples: list[dict], min_length: int = MIN_CODE_LENGTH,
                    max_length: int = MAX_CODE_LENGTH, language: str = "python",
                    skip_compile_check: bool = False) -> list[dict]:
    filtered = []
    for s in samples:
        text = s.get("text", "")
        if len(text) < min_length or len(text) > max_length:
            continue
        text = _normalize_whitespace(text)
        s["text"] = text
        inferred = _infer_language(s)
        s["language"] = inferred
        if language and inferred != "unknown" and inferred != language:
            continue
        if language == "python" and not skip_compile_check and not _is_valid_python(text):
            continue
        filtered.append(s)
    return filtered


def _train_val_test_split(samples: list[dict], train_ratio: float = 0.80,
                          val_ratio: float = 0.10, seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = [samples[i] for i in indices[:n_train]]
    val = [samples[i] for i in indices[n_train:n_train + n_val]]
    test = [samples[i] for i in indices[n_train + n_val:]]
    return train, val, test


def _save_split(split: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in split:
            f.write(json.dumps(s) + "\n")


def build_dataset(cache_dir: str = "data/tiny-codes", force_rebuild: bool = False,
                  language: str = "python", min_length: int = MIN_CODE_LENGTH,
                  max_length: int = MAX_CODE_LENGTH, train_ratio: float = 0.80,
                  val_ratio: float = 0.10, test_ratio: float = 0.10,
                  skip_compile_check: bool = False) -> dict:
    meta_path = os.path.join(cache_dir, "dataset_meta.json")
    if not force_rebuild and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("language") == language and meta.get("status") == "built":
            return meta

    jsonl_path = _download_parquet(cache_dir)
    samples = _load_raw_jsonl(jsonl_path)
    samples = _deduplicate(samples)
    samples = _filter_samples(samples, min_length=min_length, max_length=max_length,
                              language=language, skip_compile_check=skip_compile_check)

    train, val, test = _train_val_test_split(samples, train_ratio, val_ratio)

    splits = {"train": train, "validation": val, "test": test}
    split_paths = {}
    for name, split_data in splits.items():
        path = os.path.join(cache_dir, f"{name}.jsonl")
        _save_split(split_data, path)
        split_paths[name] = path

    meta = {
        "cache_dir": cache_dir,
        "language": language,
        "status": "built",
        "n_samples": len(samples),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "min_length": min_length,
        "max_length": max_length,
        "splits": split_paths,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_dataset(cache_dir: str = "data/tiny-codes", split: str = "train") -> list[str]:
    path = os.path.join(cache_dir, f"{split}.jsonl")
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line).get("text", ""))
    return texts


def dataset_statistics(cache_dir: str = "data/tiny-codes") -> dict:
    stats = {}
    for split in ("train", "validation", "test"):
        texts = load_dataset(cache_dir, split)
        lengths = [len(t) for t in texts]
        lines = [t.count("\n") for t in texts]
        stats[split] = {
            "n_samples": len(texts),
            "mean_length": sum(lengths) / max(1, len(lengths)),
            "median_length": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "mean_lines": sum(lines) / max(1, len(lines)),
        }
    return stats
