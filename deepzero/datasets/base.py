from abc import ABC, abstractmethod
from typing import Optional


class BaseDataset(ABC):
    def __init__(self, name: str, cache_dir: str = "data"):
        self.name = name
        self.cache_dir = cache_dir
        self._texts: list[str] = []
        self._meta: dict = {}

    @abstractmethod
    def download(self) -> None:
        ...

    def verify(self) -> bool:
        return len(self._texts) > 0

    @abstractmethod
    def preprocess(self) -> None:
        ...

    @abstractmethod
    def deduplicate(self) -> None:
        ...

    def normalize(self) -> None:
        from deepzero.datasets.pipeline import _normalize_whitespace
        self._texts = [_normalize_whitespace(t) for t in self._texts]

    @abstractmethod
    def split(self, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
              seed: int = 42) -> dict[str, list[str]]:
        ...

    def statistics(self) -> dict:
        total_chars = sum(len(t) for t in self._texts)
        return {
            "name": self.name,
            "n_samples": len(self._texts),
            "total_chars": total_chars,
            "avg_length": total_chars / max(1, len(self._texts)),
        }

    def load_texts(self) -> list[str]:
        return self._texts


def create_dataset(name: str, **kwargs) -> BaseDataset:
    if name == "tiny_codes":
        from deepzero.datasets.tiny_codes_v2 import TinyCodesDataset
        return TinyCodesDataset(**kwargs)
    elif name == "humaneval":
        from deepzero.datasets.humaneval import HumanEvalDataset
        return HumanEvalDataset(**kwargs)
    elif name == "mbpp":
        from deepzero.datasets.mbpp import MBPPDataset
        return MBPPDataset(**kwargs)
    elif name == "the_stack":
        from deepzero.datasets.the_stack import TheStackDataset
        return TheStackDataset(**kwargs)
    elif name == "local":
        from deepzero.datasets.local import LocalDataset
        return LocalDataset(**kwargs)
    elif name == "replay":
        from deepzero.datasets.replay import ReplayDataset
        return ReplayDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


DATASET_REGISTRY = {
    "tiny_codes": "deepzero.datasets.tiny_codes_v2:TinyCodesDataset",
    "humaneval": "deepzero.datasets.humaneval:HumanEvalDataset",
    "mbpp": "deepzero.datasets.mbpp:MBPPDataset",
    "the_stack": "deepzero.datasets.the_stack:TheStackDataset",
    "local": "deepzero.datasets.local:LocalDataset",
    "replay": "deepzero.datasets.replay:ReplayDataset",
}
