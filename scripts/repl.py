"""Interactive REPL for the model."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deepzero.models.checkpoints import load_checkpoint
from deepzero.tokenizers.bpe import BPETokenizer


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    tokenizer_path = "data/bpe_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("No tokenizer found at", tokenizer_path)
        return
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found at", checkpoint_path)
        return

    tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    model, state = load_checkpoint(checkpoint_path)
    model.eval()
    print(f"Loaded model from step {state.get('step', '?')}")
    print("Type 'quit' to exit.")

    while True:
        prompt = input("> ")
        if prompt.lower() in ("quit", "exit", "q"):
            break
        result = model.generate(tokenizer, prompt, max_len=256, temperature=0.8, top_k=40)
        print(result)


if __name__ == "__main__":
    main()
