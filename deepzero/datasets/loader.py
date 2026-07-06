import json
import os
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 512):
        super().__init__()
        self.seq_len = seq_len
        self.tokens = self._load_tokens(data_path)

    def _load_tokens(self, data_path: str) -> list[int]:
        if data_path.endswith(".txt") or data_path.endswith(".md"):
            with open(data_path) as f:
                text = f.read()
            return [ord(c) for c in text]
        elif data_path.endswith(".jsonl"):
            return self._load_jsonl_tokens(data_path)
        elif data_path.endswith(".pt"):
            return torch.load(data_path, weights_only=True).tolist()
        else:
            raise ValueError(f"Unknown data format: {data_path}")

    def _load_jsonl_tokens(self, path: str) -> list[int]:
        tokens = []
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                tokens.extend(entry.get("tokens", entry.get("input_ids", [])))
        return tokens

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.tokens[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + 1 + self.seq_len], dtype=torch.long)
        return x, y


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
