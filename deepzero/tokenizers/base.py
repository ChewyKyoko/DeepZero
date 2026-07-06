from abc import ABC, abstractmethod
from typing import Optional


class BaseTokenizer(ABC):
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self._pad_token = "<PAD>"
        self._unk_token = "<UNK>"
        self._bos_token = "<BOS>"
        self._eos_token = "<EOS>"

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        ...

    @property
    @abstractmethod
    def unk_token_id(self) -> int:
        ...

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        ...

    @abstractmethod
    def train(self, texts: list[str]) -> None:
        ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTokenizer":
        ...

    def statistics(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "pad_token": self._pad_token,
            "unk_token": self._unk_token,
            "bos_token": self._bos_token,
            "eos_token": self._eos_token,
        }


def create_tokenizer(name: str, vocab_size: int = 5000, **kwargs) -> BaseTokenizer:
    if name == "bpe":
        from deepzero.tokenizers.bpe import BPETokenizer
        return BPETokenizer(vocab_size=vocab_size, **kwargs)
    elif name == "byte_bpe":
        from deepzero.tokenizers.byte_bpe import ByteLevelBPETokenizer
        return ByteLevelBPETokenizer(vocab_size=vocab_size, **kwargs)
    elif name == "unigram":
        from deepzero.tokenizers.unigram import UnigramTokenizer
        return UnigramTokenizer(vocab_size=vocab_size, **kwargs)
    elif name == "character":
        from deepzero.tokenizers.character import CharacterTokenizer
        return CharacterTokenizer(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer: {name}. Choose from: bpe, byte_bpe, unigram, character")


TOKENIZER_REGISTRY = {
    "bpe": "deepzero.tokenizers.bpe:BPETokenizer",
    "byte_bpe": "deepzero.tokenizers.byte_bpe:ByteLevelBPETokenizer",
    "unigram": "deepzero.tokenizers.unigram:UnigramTokenizer",
    "character": "deepzero.tokenizers.character:CharacterTokenizer",
}
