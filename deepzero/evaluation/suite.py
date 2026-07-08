import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from deepzero.models.transformer import GPT

# ── Evaluation benchmarks ──────────────────────────────────────────

CODE_COMPLETION_PROMPTS = [
    ("fibonacci", "def fibonacci(n: int) -> list[int]:\n"),
    ("factorial", "def factorial(n: int) -> int:\n"),
    ("bubble_sort", "def bubble_sort(arr: list[int]) -> list[int]:\n"),
    ("is_palindrome", "def is_palindrome(s: str) -> bool:\n"),
    ("reverse_string", "def reverse_string(s: str) -> str:\n"),
]

PYTHON_SYNTAX_PROMPTS = [
    ("if_else", "x = 5\nif x > 0:\n    print('positive')\nelse:\n"),
    ("for_loop", "for i in range(10):\n    print(i)\n\n# After loop\n"),
    ("try_except", "try:\n    result = 1 / 0\nexcept ZeroDivisionError:\n"),
    ("class_def", "class Animal:\n    def __init__(self, name: str):\n        self.name = name\n\n    def speak(self) -> str:\n"),
    ("list_comp", "squares = [x**2 for x in range(10) if x % 2 == 0]\n# squares is\n"),
]

MATH_PROMPTS = [
    ("sum_1_to_n", "def sum_1_to_n(n: int) -> int:\n    # Return the sum of numbers from 1 to n\n"),
    ("is_prime", "def is_prime(n: int) -> bool:\n    # Return True if n is prime\n"),
    ("gcd", "def gcd(a: int, b: int) -> int:\n    # Return the greatest common divisor\n"),
    ("count_vowels", "def count_vowels(s: str) -> int:\n    # Count vowels in a string\n"),
]

REASONING_PROMPTS = [
    ("fizzbuzz", "def fizzbuzz(n: int) -> list[str]:\n    # Return fizzbuzz sequence up to n\n"),
    ("binary_search", "def binary_search(arr: list[int], target: int) -> int:\n"
                      "    # Return index of target or -1\n"),
]

TRIVIA_PROMPTS = [
    ("capital_of_france", "Q: What is the capital of France?\nA:"),
    ("meaning_of_life", "Q: What is the meaning of life, the universe, and everything?\nA:"),
    ("python_creator", "Q: Who created the Python programming language?\nA:"),
]


BENCHMARKS = {
    "code_completion": CODE_COMPLETION_PROMPTS,
    "python_syntax": PYTHON_SYNTAX_PROMPTS,
    "math": MATH_PROMPTS,
    "reasoning": REASONING_PROMPTS,
    "trivia": TRIVIA_PROMPTS,
}


def evaluate_model(model: GPT, tokenizer, device: str = "cpu",
                   max_len: int = 50, temperature: float = 0.7,
                   benchmarks: Optional[list[str]] = None) -> dict:
    """Run all evaluation benchmarks on a trained model.

    Returns dict of benchmark_name -> list of per-prompt results.
    """
    model.eval()
    results = {}

    names = benchmarks or list(BENCHMARKS)
    for bench_name in names:
        prompts = BENCHMARKS.get(bench_name, [])
        bench_results = []
        for label, prompt in prompts:
            start = time.time()
            generated_text = model.generate(
                tokenizer, prompt,
                max_len=max_len,
                temperature=temperature,
            )
            latency = time.time() - start
            # Remove the input prompt to get only new tokens
            completion = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text

            bench_results.append({
                "prompt": label,
                "generated": completion[:200],
                "latency_sec": round(latency, 3),
            })
        results[bench_name] = bench_results

    return results


def compute_perplexity(model: GPT, val_loader: DataLoader, device: str = "cpu") -> dict:
    """Compute validation loss and perplexity over a validation loader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    latencies = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            start = time.time()
            _, loss = model(x, y)
            latencies.append(time.time() - start)
            total_loss += loss.item() * x.size(0)
            total_tokens += x.size(0) * x.size(1)

    avg_loss = total_loss / max(1, len(val_loader))
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "val_loss": avg_loss,
        "perplexity": ppl,
        "avg_inference_latency": sum(latencies) / len(latencies) if latencies else 0,
        "tokens_per_sec_inference": total_tokens / max(1, sum(latencies)),
    }


def run_full_evaluation(model: GPT, tokenizer, val_loader: Optional[DataLoader] = None,
                        device: str = "cpu", save_dir: Optional[str] = None) -> dict:
    """Run perplexity eval + all benchmark categories. Save results."""
    results = {}

    if val_loader is not None:
        ppl_results = compute_perplexity(model, val_loader, device)
        results["perplexity"] = ppl_results
        print(f"  Perplexity: {ppl_results['perplexity']:.2f} (val_loss: {ppl_results['val_loss']:.4f})")

    bench_results = evaluate_model(model, tokenizer, device)
    results["benchmarks"] = bench_results
    for bench_name, prompts in bench_results.items():
        print(f"  {bench_name}: {len(prompts)} prompts evaluated")

    if save_dir:
        path = Path(save_dir) / "evaluation.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Evaluation saved to {path}")

    return results
