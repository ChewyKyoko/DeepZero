#!/usr/bin/env python3
"""Run the full tokenizer benchmark suite."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepzero.experiments.tokenizer_bench import BenchmarkConfig, run_benchmark
from deepzero.experiments.reporting import full_report


def main():
    cfg = BenchmarkConfig(
        tokenizer_names=["bpe", "byte_bpe", "character"],
        vocab_sizes=[500, 1000],
        training_max_iters=2000,
        training_batch_size=8,
        results_dir="results",
    )
    print(f"Running benchmark with {len(cfg.tokenizer_names)} tokenizers × {len(cfg.vocab_sizes)} vocab sizes")
    results = run_benchmark(cfg)
    print(f"\nBenchmark complete: {len(results)} experiments")

    results_path = os.path.join(cfg.results_dir, "all_results.json")
    if os.path.exists(results_path):
        report = full_report(results_path)
        print(f"\nReports generated:")
        print(f"  CSV: {report['csv']}")
        print(f"  MD:  {report['report']}")
        print(f"  Plots: {report['plots']}")


if __name__ == "__main__":
    main()
