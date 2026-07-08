import csv
import json
import os
import time as _time
from pathlib import Path
from typing import Optional

import yaml

from deepzero.tokenizers.base import create_tokenizer, TOKENIZER_REGISTRY
from deepzero.datasets.base import create_dataset

SAMPLE_PROMPT = (
    "def fibonacci(n: int) -> list[int]:\n"
    '    """Return the first n Fibonacci numbers."""\n'
    "    if n <= 0:\n"
    "        return []\n"
    "    elif n == 1:\n"
    "        return [0]\n"
    "    fib = [0, 1]\n"
    "    for i in range(2, n):\n"
    "        fib.append(fib[i-1] + fib[i-2])\n"
    "    return fib\n"
)


def run_tokenizer_benchmark(config_path: str = "configs/training/full.yaml",
                            tokenizers: Optional[list[str]] = None,
                            output_dir: str = "benchmarks",
                            n_train: int = 10000,
                            n_test: int = 1000) -> list[dict]:
    """Benchmark all tokenizers on encode/decode speed and compression.

    Returns list of result dicts. Also writes:
      - tokenizer_results.csv
      - tokenizer_results.json
      - tokenizer_summary.md
    """
    if tokenizers is None:
        tokenizers = list(TOKENIZER_REGISTRY)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds = create_dataset(cfg.get("dataset", "tiny_textbooks"))
    ds.preprocess()
    texts = ds.load_texts()
    train_texts = texts[:n_train]

    results = []
    for name in tokenizers:
        print("─" * 60)
        print(f"Benchmarking tokenizer: {name}")
        print("─" * 60)

        tok = create_tokenizer(name, vocab_size=5000)

        # Training
        t0 = _time.time()
        tok.train(train_texts)
        train_time = _time.time() - t0

        vocab_actual = (len(tok.id_to_token) if hasattr(tok, "id_to_token")
                        else len(getattr(tok, "char_to_id", {})) or tok.vocab_size)
        print(f"  Train: {train_time:.1f}s | Vocab: {vocab_actual}")

        # Encode / decode speed
        test_texts = train_texts[:n_test]
        encode_times, decode_times, encode_lens = [], [], []
        for text in test_texts:
            t0 = _time.time()
            ids = tok.encode(text)
            encode_times.append(_time.time() - t0)
            encode_lens.append(len(ids))

            t0 = _time.time()
            _ = tok.decode(ids)
            decode_times.append(_time.time() - t0)

        avg_encode = sum(encode_times) / len(encode_times)
        avg_decode = sum(decode_times) / len(decode_times)
        avg_len = sum(encode_lens) / len(encode_lens)
        avg_text_len = sum(len(t) for t in test_texts) / len(test_texts)

        # Compression ratio (chars per token)
        comp = avg_text_len / max(1, avg_len)

        # Sample prompt
        t0 = _time.time()
        prompt_ids = tok.encode(SAMPLE_PROMPT)
        prompt_encode = _time.time() - t0
        prompt_tok = len(prompt_ids)

        stats = tok.statistics() if hasattr(tok, "statistics") else {}

        results.append({
            "tokenizer": name,
            "train_time_sec": round(train_time, 1),
            "vocab_size": vocab_actual,
            "avg_encode_us": round(avg_encode * 1e6, 1),
            "avg_decode_us": round(avg_decode * 1e6, 1),
            "encode_speed_tps": round(1.0 / avg_encode, 0) if avg_encode > 0 else 0,
            "decode_speed_tps": round(1.0 / avg_decode, 0) if avg_decode > 0 else 0,
            "avg_tokens_per_text": round(avg_len, 1),
            "compression_ratio": round(comp, 1),
            "prompt_tokens": prompt_tok,
            "prompt_encode_us": round(prompt_encode * 1e6, 1),
        })

        print(f"  Encode: {avg_encode*1e6:.0f}µs | Decode: {avg_decode*1e6:.0f}µs | "
              f"Compression: {comp:.1f} chars/tok | {avg_len:.0f} tok/text")

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(results, out / "tokenizer_results.csv")
    _write_json(results, out / "tokenizer_results.json")
    _write_summary_md(results, out / "tokenizer_summary.md")

    print("=" * 60)
    print("TOKENIZER COMPARISON")
    print("=" * 60)
    h = f"{'Tokenizer':<14} {'Train(s)':<10} {'Vocab':<8} {'Enc(µs)':<10} {'Dec(µs)':<10} {'Compress':<10}"
    print(h)
    print("-" * len(h))
    for r in results:
        print(f"{r['tokenizer']:<14} {r['train_time_sec']:<10.1f} {r['vocab_size']:<8} "
              f"{r['avg_encode_us']:<10.0f} {r['avg_decode_us']:<10.0f} {r['compression_ratio']:<10.1f}")
    print(f"\nResults saved to {out}/")

    return results


def _write_csv(results: list[dict], path: Path):
    fieldnames = [
        "tokenizer", "train_time_sec", "vocab_size", "avg_encode_us", "avg_decode_us",
        "encode_speed_tps", "decode_speed_tps", "avg_tokens_per_text",
        "compression_ratio", "prompt_tokens", "prompt_encode_us",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


def _write_json(results: list[dict], path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _write_summary_md(results: list[dict], path: Path):
    lines = []
    lines.append("# Tokenizer Benchmark Summary")
    lines.append("")
    lines.append("| Tokenizer | Train (s) | Vocab | Encode (µs) | Decode (µs) | Compress | Tok/text |")
    lines.append("|-----------|-----------|-------|-------------|-------------|----------|----------|")
    for r in results:
        lines.append(
            f"| {r['tokenizer']} | {r['train_time_sec']} | {r['vocab_size']} | "
            f"{r['avg_encode_us']:.0f} | {r['avg_decode_us']:.0f} | "
            f"{r['compression_ratio']} | {r['avg_tokens_per_text']:.0f} |"
        )
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
