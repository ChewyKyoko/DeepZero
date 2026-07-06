import json
import os
from collections import Counter, OrderedDict
from typing import Optional

from deepzero.tokenizers.base import BaseTokenizer


def _to_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _from_bytes(bytes_list: list[int]) -> str:
    return bytes(bytes_list).decode("utf-8", errors="replace")


class ByteLevelBPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 5000):
        super().__init__(vocab_size)
        self.merges: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.id_to_token: dict[int, bytes] = {}
        self.token_to_id: dict[bytes, int] = {}

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

    def _init_base_vocab(self):
        special_ids = {
            0: self._pad_token.encode(),
            1: self._unk_token.encode(),
            2: self._bos_token.encode(),
            3: self._eos_token.encode(),
        }
        for i, b in special_ids.items():
            self.token_to_id[b] = i
            self.id_to_token[i] = b
        for byte_val in range(256):
            b = bytes([byte_val])
            if b not in self.token_to_id:
                tid = len(self.token_to_id)
                self.token_to_id[b] = tid
                self.id_to_token[tid] = b

    def _get_pair_stats(self, ids_list: list[list[int]]) -> Counter:
        stats: Counter = Counter()
        for ids in ids_list:
            for i in range(len(ids) - 1):
                stats[(ids[i], ids[i + 1])] += 1
        return stats

    def _merge_ids(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def train(self, texts: list[str]) -> None:
        self.token_to_id = {}
        self.id_to_token = {}
        self._init_base_vocab()
        encoded = [_to_bytes(t) for t in texts]
        ids_list = [[b for b in e] for e in encoded]
        num_merges = self.vocab_size - len(self.token_to_id)
        for _ in range(num_merges):
            stats = self._get_pair_stats(ids_list)
            if not stats:
                break
            pair = stats.most_common(1)[0][0]
            new_id = len(self.token_to_id)
            merged_bytes = self.id_to_token[pair[0]] + self.id_to_token[pair[1]]
            self.merges[pair] = new_id
            self.token_to_id[merged_bytes] = new_id
            self.id_to_token[new_id] = merged_bytes
            ids_list = [self._merge_ids(ids, pair, new_id) for ids in ids_list]

    def encode(self, text: str) -> list[int]:
        raw = _to_bytes(text)
        ids = [self.bos_token_id]
        ids.extend(raw)
        ids.append(self.eos_token_id)
        changed = True
        while changed:
            changed = False
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
            best = None
            best_rank = float("inf")
            for p in pairs:
                rank = self.merges.get(p)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best = p
            if best is None:
                break
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    new_ids.append(best_rank)
                    i += 2
                    changed = True
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b""
        for idx in ids:
            if idx in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            token = self.id_to_token.get(idx, self._unk_token.encode())
            if token == self._unk_token.encode():
                raw += b"\xef\xbf\xbd"
            else:
                raw += token
        return _from_bytes(list(raw))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "type": "byte_bpe",
                "vocab_size": self.vocab_size,
                "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
                "id_to_token": {str(k): v.hex() for k, v in self.id_to_token.items()},
                "token_to_id": {k.hex(): v for k, v in self.token_to_id.items()},
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = OrderedDict()
        for k, v in data["merges"].items():
            parts = k.split(",")
            tok.merges[(int(parts[0]), int(parts[1]))] = v
        tok.id_to_token = {int(k): bytes.fromhex(v) for k, v in data["id_to_token"].items()}
        tok.token_to_id = {bytes.fromhex(k): v for k, v in data["token_to_id"].items()}
        return tok

    def statistics(self) -> dict:
        return {
            **super().statistics(),
            "n_merges": len(self.merges),
            "vocab_size_actual": len(self.id_to_token),
            "type": "Byte-Level BPE",
            "byte_coverage": min(256, self.vocab_size),
        }
