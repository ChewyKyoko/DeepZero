import os
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int):
        self.seq_len = seq_len
        if os.path.isdir(path):
            texts = []
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath) and fname.endswith(".txt"):
                    with open(fpath) as f:
                        texts.append(f.read())
            text = "\n".join(texts)
        else:
            with open(path) as f:
                text = f.read()
        self.tokens = tokenizer.encode(text)
        self.n_tokens = len(self.tokens)

    def __len__(self):
        return max(0, self.n_tokens - self.seq_len)

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
