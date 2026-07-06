import re
from collections import defaultdict, Counter


_SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\w\s]+|\s+(?!\S)|\s+""")


class BPETokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: dict[tuple[str, str], int] = {}
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}

    @property
    def n_special(self) -> int:
        return 4

    def train(self, texts: list[str]):
        words = []
        for t in texts:
            words.extend(_SPLIT.findall(t))

        word_freqs: defaultdict[str, int] = defaultdict(int)
        for w in words:
            word_freqs[" ".join(list(w)) + " </w>"] += 1

        chars: set[str] = set()
        for w in word_freqs:
            for c in w.split():
                chars.add(c)

        sorted_chars = sorted(chars)
        base_vocab = {i + self.n_special: c for i, c in enumerate(sorted_chars)}
        base_vocab[self.PAD] = "<pad>"
        base_vocab[self.BOS] = "<bos>"
        base_vocab[self.EOS] = "<eos>"
        base_vocab[self.UNK] = "<unk>"

        self.id_to_token = dict(base_vocab)
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

        num_merges = self.vocab_size - len(base_vocab)
        for _ in range(num_merges):
            pair_freqs: defaultdict[tuple[str, str], int] = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged = best_pair[0] + best_pair[1]
            new_id = len(self.id_to_token)
            self.merges[best_pair] = new_id
            self.id_to_token[new_id] = merged
            self.token_to_id[merged] = new_id

            new_word_freqs: defaultdict[str, int] = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                out: list[str] = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(symbols[i])
                        i += 1
                new_word_freqs[" ".join(out)] = freq
            word_freqs = new_word_freqs

    def encode(self, text: str) -> list[int]:
        words = _SPLIT.findall(text)
        ids: list[int] = [self.BOS]
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
                merged: list[str] = []
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
                ids.append(self.token_to_id.get(s, self.UNK))
        ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for idx in ids:
            if idx in (self.PAD, self.BOS, self.EOS):
                continue
            token = self.id_to_token.get(idx, "<unk>")
            if token == "</w>" or token == "<unk>":
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
        import json
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()},
                "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        import json
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = {tuple(k.split(" ", 1)): v for k, v in data["merges"].items()}
        tok.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tok.token_to_id = {v: k for k, v in tok.id_to_token.items()}
        return tok

    @property
    def n_vocab(self) -> int:
        return len(self.id_to_token)
