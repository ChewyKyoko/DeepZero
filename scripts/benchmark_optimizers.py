import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepzero.benchmarks.optimizer import run_optimizer_benchmark
from deepzero.training.optimizer import OPTIMIZER_REGISTRY


def main():
    # Use tiny config by default for fast iteration; pass a config path to override
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/training/tiny.yaml"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    optimizers = sys.argv[3:] if len(sys.argv) > 3 else list(OPTIMIZER_REGISTRY)

    run_optimizer_benchmark(
        config_path=config_path,
        steps=steps,
        optimizers=optimizers,
        output_dir="benchmarks",
    )


if __name__ == "__main__":
    main()
