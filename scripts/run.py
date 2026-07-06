"""Run training pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split

from deepzero.models.transformer import GPT, ModelConfig
from deepzero.tokenizers.bpe import BPETokenizer
from deepzero.datasets.loader import TextDataset
from deepzero.training.trainer import Trainer
from deepzero.utils.seed import set_seed


def main():
    set_seed(42)
    device = "cpu"
    print(f"Device: {device}")

    tokenizer_path = "data/bpe_tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
        print(f"Loaded tokenizer with vocab_size={tokenizer.vocab_size}")
    else:
        print("Training tokenizer...")
        with open("data/sample_data.txt") as f:
            text = f.read()
        tokenizer = BPETokenizer(vocab_size=1300)
        tokenizer.train(text)
        tokenizer.save(tokenizer_path)

    config = ModelConfig(vocab_size=tokenizer.vocab_size, device=device)
    model = GPT(config)
    n_params = model.n_params
    print(f"Model has {n_params:,} parameters ({n_params/1e6:.1f}M)")

    dataset = TextDataset("data/sample_data.txt", seq_len=config.max_seq_len)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        max_iters=5000,
        checkpoint_dir="checkpoints",
        device=device,
        log_interval=10,
        eval_interval=100,
    )
    trainer.train()


if __name__ == "__main__":
    main()
