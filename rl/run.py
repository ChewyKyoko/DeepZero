#!/usr/bin/env python3
"""DeepZero RL — Self-Improvement Loop CLI.

Usage:
    ./run python3 rl/run.py --iterations 3 --tasks 5
    ./run python3 rl/run.py --iterations 5 --tasks 10 --model runs/default/best.pt
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config as ModelConfig
from tokenizer import BPETokenizer
from model import GPT
from rl.loop import SelfImprovementLoop


def main():
    parser = argparse.ArgumentParser(description="DeepZero RL Self-Improvement Loop")
    parser.add_argument("--iterations", type=int, default=3, help="Number of improvement iterations")
    parser.add_argument("--tasks", type=int, default=5, help="Tasks per iteration")
    parser.add_argument("--model", default="runs/default/best.pt", help="Base model checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    # Load tokenizer
    tok_path = "tokenizer.json"
    if not os.path.exists(tok_path):
        print("Tokenizer not found. Train one first with: python3 run.py --just-tokenize")
        sys.exit(1)
    tokenizer = BPETokenizer.load(tok_path)

    # Load model config
    cfg = ModelConfig()
    cfg.vocab_size = tokenizer.n_vocab
    cfg.device = args.device

    # Create model
    model = GPT(cfg)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=False)["model"])
        print(f"Loaded base model from {args.model}")
    else:
        print(f"No checkpoint found at {args.model}, using untrained model")

    # Run self-improvement loop
    loop = SelfImprovementLoop(model, tokenizer, cfg)
    results = loop.run(iterations=args.iterations, tasks_per_iter=args.tasks)

    # Summary
    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT SUMMARY")
    print("=" * 60)
    for r in results:
        s = r["stats"]
        print(f"  Iter {r['iteration']}: avg_score={s['avg_score']:.3f}, "
              f"failures={s['failures']}, buffer={r['buffer_size']}, "
              f"delta={r['score_delta']:+.3f}")

    # Save final model
    final_path = "runs/rl/rl_final.pt"
    os.makedirs("runs/rl", exist_ok=True)
    torch.save({"model": model.state_dict()}, final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()
