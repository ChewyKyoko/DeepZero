import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepzero.benchmarks.tokenizer import run_tokenizer_benchmark
from deepzero.tokenizers.base import TOKENIZER_REGISTRY


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training/full.yaml"
    tokenizers = sys.argv[2:] if len(sys.argv) > 2 else list(TOKENIZER_REGISTRY)

    run_tokenizer_benchmark(
        config_path=config_path,
        tokenizers=tokenizers,
        output_dir="benchmarks",
    )


if __name__ == "__main__":
    main()
