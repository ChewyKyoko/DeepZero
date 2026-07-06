import random
from typing import Optional

from deepzero.datasets.base import BaseDataset, create_dataset


class DatasetMixture:
    def __init__(self, name: str = "default"):
        self.name = name
        self.components: list[tuple[BaseDataset, float]] = []

    def add(self, dataset: BaseDataset, weight: float = 1.0):
        self.components.append((dataset, weight))

    def load_all(self):
        for ds, _ in self.components:
            ds.preprocess()
            ds.deduplicate()
            ds.normalize()

    def get_train_texts(self, seed: int = 42) -> list[str]:
        rng = random.Random(seed)
        all_texts = []
        for ds, weight in self.components:
            texts = ds.load_texts()
            n = max(1, int(len(texts) * weight))
            sampled = rng.sample(texts, min(n, len(texts)))
            all_texts.extend(sampled)
        rng.shuffle(all_texts)
        return all_texts

    def get_split_texts(self, split: str = "train", seed: int = 42) -> list[str]:
        rng = random.Random(seed)
        all_texts = []
        for ds, weight in self.components:
            splits = ds.split(seed=seed)
            pool = splits.get(split, [])
            n = max(1, int(len(pool) * weight))
            sampled = rng.sample(pool, min(n, len(pool)))
            all_texts.extend(sampled)
        rng.shuffle(all_texts)
        return all_texts

    def statistics(self) -> dict:
        stats = {"name": self.name, "components": []}
        for ds, weight in self.components:
            ds_stats = ds.statistics()
            ds_stats["mixture_weight"] = weight
            stats["components"].append(ds_stats)
        return stats

    @classmethod
    def from_config(cls, config: dict) -> "DatasetMixture":
        mix = cls(name=config.get("name", "mixture"))
        for comp in config.get("components", []):
            ds = create_dataset(comp["name"], **comp.get("kwargs", {}))
            mix.add(ds, weight=comp.get("weight", 1.0))
        return mix


def load_mixture_from_yaml(path: str) -> DatasetMixture:
    try:
        import yaml
    except ImportError:
        raise ImportError("yaml (PyYAML) required. pip install pyyaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return DatasetMixture.from_config(cfg)
