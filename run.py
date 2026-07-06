#!/usr/bin/env python3
"""DeepZero V2 — Full Pipeline Orchestrator

Trains a GPT-style transformer from scratch on sample data, then generates text.

Usage:
    python run.py                          # Full pipeline (default)
    python run.py --steps 5000             # Custom training steps
    python run.py --just-tokenize          # Only train tokenizer
    python run.py --just-generate          # Generate from existing checkpoint
    python run.py --prompt "def fib"       # Custom generation prompt
"""
import argparse
import os
import sys
import torch

from config import Config
from tokenizer import BPETokenizer
from model import GPT
from dataset import TextDataset
from train import Trainer
from generate import generate

SAMPLE_DATA = "sample_data.txt"
TOKENIZER_PATH = "tokenizer.json"
RUN_DIR = "runs/default"


def main():
    parser = argparse.ArgumentParser(description="DeepZero V2 — Train a GPT from scratch")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--device", default="auto", help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--prompt", default=None, help="Generation prompt")
    parser.add_argument("--just-tokenize", action="store_true", help="Only train tokenizer")
    parser.add_argument("--just-generate", action="store_true", help="Generate from existing checkpoint")
    parser.add_argument("--mini", action="store_true", help="Tiny model for quick CPU training")
    args = parser.parse_args()

    cfg = Config()
    if args.mini:
        cfg.d_model = 192
        cfg.n_layers = 4
        cfg.n_heads = 4
        cfg.max_seq_len = 256
        cfg.batch_size = 8
        cfg.max_steps = 2000
    if args.steps:
        cfg.max_steps = args.steps
    if args.device != "auto":
        cfg.device = args.device
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.d_ff = cfg.d_model * 4
    torch.manual_seed(42)
    import random
    random.seed(42)

    # Step 1: Train tokenizer
    if not os.path.exists(TOKENIZER_PATH) or args.just_tokenize:
        print("=" * 60)
        print("Training BPE tokenizer...")
        with open(SAMPLE_DATA) as f:
            text = f.read()
        tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
        tokenizer.train([text])
        tokenizer.save(TOKENIZER_PATH)
        print(f"  Vocab size: {tokenizer.n_vocab}")
        print(f"  Saved to:   {TOKENIZER_PATH}")
        if args.just_tokenize:
            return

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    cfg.vocab_size = tokenizer.n_vocab

    if args.just_generate:
        _do_generate(cfg, tokenizer, args.prompt)
        return

    # Step 2: Load dataset
    print("=" * 60)
    print("Loading dataset...")
    dataset = TextDataset(SAMPLE_DATA, tokenizer, cfg.max_seq_len)
    print(f"  Total tokens: {dataset.n_tokens:,}")
    print(f"  Sequences:    {len(dataset):,}")

    # Step 3: Create model
    print("=" * 60)
    print(f"Creating model ({sum(p.numel() for p in GPT(cfg).parameters()):,} params)...")
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:   {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Device:       {cfg.device}")
    print(f"  Max steps:    {cfg.max_steps}")
    print(f"  Batch size:   {cfg.batch_size}")

    # Step 4: Train
    print("=" * 60)
    print("Training...")
    trainer = Trainer(cfg)
    model = trainer.train(model, dataset, tokenizer, run_dir=RUN_DIR)

    # Step 5: Generate
    _do_generate(cfg, tokenizer, args.prompt)


def _cfg_from_dict(d: dict) -> Config:
    cfg = Config()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _load_cfg(run_dir: str) -> Config:
    import json
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        saved = json.load(open(cfg_path))
        cfg = _cfg_from_dict(saved)
    else:
        cfg = Config()
    cfg.d_ff = cfg.d_model * 4
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def _do_generate(cfg, tokenizer, prompt=None):
    print("=" * 60)
    print("Loading best checkpoint for generation...")
    ckpt_path = os.path.join(RUN_DIR, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt_cfg = _load_cfg(RUN_DIR)
        model = GPT(ckpt_cfg).to(ckpt_cfg.device)
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=ckpt_cfg.device, weights_only=False)["model"])
            print(f"  Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"  Checkpoint load failed ({e}), using untrained model")
            model = GPT(cfg).to(cfg.device)
    else:
        print("  No checkpoint found. Using untrained model.")
        model = GPT(cfg).to(cfg.device)

    prompts = [prompt] if prompt else [
        "def fibonacci",
        "The binary search algorithm",
        "A linked list is",
        "Machine learning is",
        "Python is",
    ]

    for p in prompts:
        print(f"\nPrompt: {p}")
        print("-" * 40)
        out = generate(
            model, tokenizer, p,
            max_len=cfg.max_gen_len,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
        )
        print(out)
        print("-" * 40)


if __name__ == "__main__":
    main()
