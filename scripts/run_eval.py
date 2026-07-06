#!/usr/bin/env python3
"""Run the R0.4 Evaluation Benchmark."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from deepzero.experiments.eval_bench import EvalConfig, run_evaluation
from deepzero.experiments.eval_reporting import generate_benchmark_csv, generate_benchmark_report, generate_benchmark_plots


def main():
    cfg = EvalConfig(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_name="bpe",
        tokenizer_path="data/bpe_tokenizer.json",
        tokenizer_vocab_size=5000,
        results_dir="results",
    )

    print("Running R0.4 evaluation benchmark...")
    result = run_evaluation(cfg)

    # Save combined results for reporting
    combined_path = os.path.join(cfg.results_dir, "eval_combined.json")
    with open(combined_path, "w") as f:
        import dataclasses
        from deepzero.experiments.eval_bench import EvalExperimentResult
        if isinstance(result, EvalExperimentResult):
            json.dump([result.__dict__], f, indent=2, default=str)
        else:
            json.dump([result], f, indent=2, default=str)

    print(f"\nGenerating reports...")
    csv_p = generate_benchmark_csv([result.__dict__])
    md_p = generate_benchmark_report([result.__dict__])
    plots = generate_benchmark_plots([result.__dict__])

    print(f"\nResults:")
    print(f"  Aggregate Score: {result.aggregate_score:.1f}/100")
    print(f"  CSV: {csv_p}")
    print(f"  Report: {md_p}")
    print(f"  Plots: {plots}")


if __name__ == "__main__":
    main()
