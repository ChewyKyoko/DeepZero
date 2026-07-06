import json
import os
from typing import Optional

from deepzero.tokenizers.base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 5000):
        super().__init__(vocab_size)
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 1

    @property
    def bos_token_id(self) -> int:
        return 2

    @property
    def eos_token_id(self) -> int:
        return 3

    def train(self, texts: list[str]) -> None:
        self.char_to_id = {}
        self.id_to_char = {}
        specials = [self._pad_token, self._unk_token, self._bos_token, self._eos_token]
        for i, t in enumerate(specials):
            self.char_to_id[t] = i
            self.id_to_char[i] = t
        chars: set[str] = set()
        for text in texts:
            chars.update(text)
        sorted_chars = sorted(chars)
        for c in sorted_chars:
            if c not in self.char_to_id and len(self.char_to_id) < self.vocab_size:
                tid = len(self.char_to_id)
                self.char_to_id[c] = tid
                self.id_to_char[tid] = c

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for c in text:
            ids.append(self.char_to_id.get(c, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            chars.append(self.id_to_char.get(idx, self._unk_token))
        return "".join(chars)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "type": "character",
                "vocab_size": self.vocab_size,
                "char_to_id": self.char_to_id,
                "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharacterTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        return tok

    def statistics(self) -> dict:
        return {
            **super().statistics(),
            "vocab_size_actual": len(self.id_to_char),
            "type": "Character-level",
            "n_chars": len(self.char_to_id) - 4,
        }
