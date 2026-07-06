#!/usr/bin/env python3
"""Run the R0.3 Dataset Benchmark."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepzero.experiments.dataset_bench import DatasetBenchConfig, run_dataset_benchmark
from deepzero.experiments.dataset_reporting import full_dataset_report
from deepzero.experiments.research_doc import generate_research_doc


def main():
    cfg = DatasetBenchConfig(
        dataset_names=["tiny_codes", "humaneval", "mbpp"],
        tokenizer_name="bpe",
        tokenizer_vocab_size=5000,
        training_max_iters=2000,
        results_dir="results",
    )
    print(f"Running dataset benchmark: {cfg.dataset_names}")
    results = run_dataset_benchmark(cfg)

    results_path = os.path.join(cfg.results_dir, "all_dataset_results.json")
    if os.path.exists(results_path):
        report = full_dataset_report(results_path)
        print(f"\nReports:")
        print(f"  CSV: {report['csv']}")
        print(f"  MD:  {report['report']}")
        print(f"  Plots: {report['plots']}")

        doc = generate_research_doc(results_path)
        print(f"  Research doc: {doc}")


if __name__ == "__main__":
    main()
