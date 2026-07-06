import json
import os
import re
from collections import Counter, OrderedDict
from typing import Optional

from deepzero.tokenizers.base import BaseTokenizer


_SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\w\s]+|\s+(?!\S)|\s+""")


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 5000):
        super().__init__(vocab_size)
        self.merges: OrderedDict[tuple[str, str], int] = OrderedDict()
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self._vocab_size_actual = 0

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id.get(self._pad_token, 0)

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id.get(self._unk_token, 1)

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id.get(self._bos_token, 2)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id.get(self._eos_token, 3)

    @property
    def n_vocab(self) -> int:
        return len(self.id_to_token)

    def _init_base_vocab(self):
        specials = [self._pad_token, self._unk_token, self._bos_token, self._eos_token]
        for i, t in enumerate(specials):
            self.token_to_id[t] = i
            self.id_to_token[i] = t

    def _get_word_freqs(self, texts: list[str]) -> Counter:
        word_freqs: Counter = Counter()
        for text in texts:
            words = _SPLIT.findall(text)
            for w in words:
                word_freqs[" ".join(list(w)) + " </w>"] += 1
        return word_freqs

    def train(self, texts: list[str]) -> None:
        self.token_to_id = {}
        self.id_to_token = {}
        self._init_base_vocab()
        word_freqs = self._get_word_freqs(texts)
        chars: set[str] = set()
        for w in word_freqs:
            for c in w.split():
                chars.add(c)
        sorted_chars = sorted(chars)
        for i, c in enumerate(sorted_chars):
            tid = len(self.token_to_id)
            self.token_to_id[c] = tid
            self.id_to_token[tid] = c
        num_merges = self.vocab_size - len(self.token_to_id)
        for _ in range(num_merges):
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i + 1])] += freq
            if not pair_freqs:
                break
            best_pair = pair_freqs.most_common(1)[0][0]
            merged = best_pair[0] + best_pair[1]
            new_id = len(self.token_to_id)
            self.merges[best_pair] = new_id
            self.token_to_id[merged] = new_id
            self.id_to_token[new_id] = merged
            new_word_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                out = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(symbols[i])
                        i += 1
                new_word_freqs[" ".join(out)] += freq
            word_freqs = new_word_freqs
        self._vocab_size_actual = len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        words = _SPLIT.findall(text)
        ids = [self.bos_token_id]
        for word in words:
            symbols = list(word) + ["</w>"]
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                best = None
                best_rank = float("inf")
                for pair in pairs:
                    rank = self.merges.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best = pair
                if best is None:
                    break
                merged = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                        merged.append(best[0] + best[1])
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                symbols = merged
            for s in symbols:
                ids.append(self.token_to_id.get(s, self.unk_token_id))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for idx in ids:
            if idx in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            token = self.id_to_token.get(idx, self._unk_token)
            if token == "</w>" or token == self._unk_token:
                out.append(" " if token == "</w>" else "?")
            elif token.endswith("</w>"):
                out.append(token[:-4])
                out.append(" ")
            else:
                out.append(token)
        result = "".join(out)
        result = result.replace("  ", " ").strip()
        return result

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "type": "bpe",
                "vocab_size": self.vocab_size,
                "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()},
                "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
                "token_to_id": self.token_to_id,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = OrderedDict()
        for k, v in data["merges"].items():
            tok.merges[tuple(k.split(" ", 1))] = v
        tok.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tok.token_to_id = data["token_to_id"]
        tok._vocab_size_actual = len(tok.id_to_token)
        return tok

    # Backward-compat aliases
    from_pretrained = load

    @property
    def PAD(self) -> str:
        return self._pad_token

    @property
    def UNK(self) -> str:
        return self._unk_token

    @property
    def BOS(self) -> str:
        return self._bos_token

    @property
    def EOS(self) -> str:
        return self._eos_token

    def statistics(self) -> dict:
        return {
            **super().statistics(),
            "n_merges": len(self.merges),
            "vocab_size_actual": self._vocab_size_actual,
            "type": "BPE (word-level)",
        }
