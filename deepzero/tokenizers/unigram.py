import json
import math
import os
import re
from collections import Counter
from typing import Optional

from deepzero.tokenizers.base import BaseTokenizer


_SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\w\s]+|\s+(?!\S)|\s+""")


class UnigramTokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 5000):
        super().__init__(vocab_size)
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.scores: dict[int, float] = {}
        self._init_vocab()

    def _init_vocab(self):
        specials = [self._pad_token, self._unk_token, self._bos_token, self._eos_token]
        for i, t in enumerate(specials):
            self.token_to_id[t] = i
            self.id_to_token[i] = t
            self.scores[i] = 0.0

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

    def _get_seed_vocab(self, texts: list[str]) -> Counter:
        freq: Counter = Counter()
        for text in texts:
            words = _SPLIT.findall(text)
            for w in words:
                freq[w] += 1
                for i in range(len(w)):
                    freq[w[i]] += 1
                for i in range(len(w) - 1):
                    freq[w[i:i + 2]] += 1
        return freq

    def _compute_loss(self, texts: list[str]) -> float:
        total_log_prob = 0.0
        total_tokens = 0
        for text in texts:
            ids = self.encode(text)
            for idx in ids:
                if idx in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
                    continue
                score = self.scores.get(idx, float("-inf"))
                total_log_prob += score
                total_tokens += 1
        return -total_log_prob / max(1, total_tokens)

    def _viterbi_segment(self, text: str) -> list[int]:
        chars = list(text)
        n = len(chars)
        best_score = [float("-inf")] * (n + 1)
        best_path = [-1] * (n + 1)
        best_score[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - 20), i):
                substr = "".join(chars[j:i])
                idx = self.token_to_id.get(substr)
                if idx is not None:
                    score = best_score[j] + self.scores.get(idx, float("-inf"))
                    if score > best_score[i]:
                        best_score[i] = score
                        best_path[i] = j
        ids = []
        i = n
        while i > 0:
            j = best_path[i]
            if j < 0:
                ids.append(self.unk_token_id)
                i -= 1
            else:
                substr = "".join(chars[j:i])
                idx = self.token_to_id.get(substr, self.unk_token_id)
                ids.append(idx)
                i = j
        ids.reverse()
        return ids

    def train(self, texts: list[str]) -> None:
        self.token_to_id = {}
        self.id_to_token = {}
        self.scores = {}
        self._init_vocab()
        freq = self._get_seed_vocab(texts)
        total_freq = sum(freq.values())
        for token, f in freq.most_common(self.vocab_size - 4):
            if f < 1:
                break
            tid = len(self.token_to_id)
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token
            self.scores[tid] = math.log(f / total_freq) if f > 0 else float("-inf")
        for _ in range(5):
            for tid in self.scores:
                idx_tokens = sum(1 for t in texts for _ in self.encode(t))
                count = max(1, idx_tokens)
                token = self.id_to_token[tid]
                freq_est = sum(1 for t in texts if token in t)
                self.scores[tid] = math.log(max(1, freq_est) / count)

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        ids.extend(self._viterbi_segment(text))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for idx in ids:
            if idx in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            token = self.id_to_token.get(idx, self._unk_token)
            out.append(token)
        return "".join(out)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "type": "unigram",
                "vocab_size": self.vocab_size,
                "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
                "token_to_id": self.token_to_id,
                "scores": {str(k): v for k, v in self.scores.items()},
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UnigramTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tok.token_to_id = data["token_to_id"]
        tok.scores = {int(k): v for k, v in data["scores"].items()}
        return tok

    def statistics(self) -> dict:
        return {
            **super().statistics(),
            "n_scores": len(self.scores),
            "vocab_size_actual": len(self.id_to_token),
            "type": "Unigram (SentencePiece-style)",
        }
