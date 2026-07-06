"""Train a model with a config file."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from deepzero.models.transformer import GPT, ModelConfig
from deepzero.tokenizers.bpe import BPETokenizer
from deepzero.datasets.loader import TextDataset
from deepzero.training.trainer import Trainer
from deepzero.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    cfg = load_config(config_path)
    set_seed(42)
    device = cfg.get("model", {}).get("device", "cpu")

    tokenizer_path = cfg.get("tokenizer", {}).get("path", "data/bpe_tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    else:
        with open(cfg["data"]["path"]) as f:
            text = f.read()
        vs = cfg.get("tokenizer", {}).get("vocab_size", 1300)
        tokenizer = BPETokenizer(vocab_size=vs)
        tokenizer.train(text)
        tokenizer.save(tokenizer_path)

    mc = cfg["model"]
    config = ModelConfig(vocab_size=tokenizer.vocab_size, device=device, **{k: v for k, v in mc.items() if k != "device"})
    model = GPT(config)
    print(f"Model: {model.n_params:,} params")

    dataset = TextDataset(cfg["data"]["path"], seq_len=cfg["data"].get("seq_len", 512))
    val_split = cfg["data"].get("val_split", 0.1)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"].get("batch_size", 8), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"].get("batch_size", 8), shuffle=False)

    tc = cfg["training"]
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                     lr=tc.get("lr", 3e-4), weight_decay=tc.get("weight_decay", 0.1),
                     warmup_iters=tc.get("warmup_iters", 100), max_iters=tc.get("max_iters", 5000),
                     checkpoint_dir=cfg.get("checkpoint", {}).get("dir", "checkpoints"),
                     device=device, log_interval=tc.get("log_interval", 10),
                     eval_interval=tc.get("eval_interval", 100), grad_clip=tc.get("grad_clip", 1.0))
    trainer.train()


if __name__ == "__main__":
    main()
