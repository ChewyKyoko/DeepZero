"""Generate text from a trained model."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deepzero.models.checkpoints import load_checkpoint
from deepzero.tokenizers.bpe import BPETokenizer


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"

    tokenizer_path = "data/bpe_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("No tokenizer found at", tokenizer_path)
        return

    tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    model, _ = load_checkpoint(checkpoint_path)
    model.eval()

    result = model.generate(tokenizer, prompt, max_len=200, temperature=0.8, top_k=40)
    print(prompt)
    print(result)


if __name__ == "__main__":
    main()
