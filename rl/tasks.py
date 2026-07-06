"""Coding task generator — produces problems with test cases."""

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from rl.config import TASKS, BUGGY_TASKS


@dataclass
class CodingTask:
    id: str
    prompt: str
    solution: str
    tests: str
    difficulty: float
    category: str = "write"
    buggy_code: Optional[str] = None
    bug_explanation: Optional[str] = None

    @property
    def full_prompt(self) -> str:
        if self.buggy_code:
            return f"{self.prompt}\n\nBuggy code:\n{self.buggy_code}"
        return self.prompt

    def to_training_text(self) -> str:
        return f"{self.prompt}\n```python\n{self.solution}\n```"


class TaskGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, n: int, difficulties: Optional[list[float]] = None) -> list[CodingTask]:
        all_tasks = self._build_all()
        if difficulties:
            all_tasks = [t for t in all_tasks if t.difficulty in difficulties]
        self.rng.shuffle(all_tasks)
        result = []
        for i in range(n):
            t = all_tasks[i % len(all_tasks)]
            result.append(copy.deepcopy(t))
            if (i // len(all_tasks)) > 0:
                result[-1].id = f"{t.id}_v{i}"
        return result

    def generate_with_variants(self, n: int, variants_per_base: int = 3) -> list[CodingTask]:
        bases = self._build_all()
        variants = []
        prompts = [
            "Implement a function that",
            "Write code for",
            "Create a function called",
        ]
        for base in bases:
            v = copy.deepcopy(base)
            variant_prompt = self.rng.choice(prompts) + " " + base.prompt.lower().replace("write a function that ", "").replace("write a function that", "")
            v.prompt = variant_prompt
            v.id = f"{base.id}_v0"
            variants.append(v)
            for vi in range(1, variants_per_base):
                v = copy.deepcopy(base)
                alt_prompt = self.rng.choice(prompts) + " " + base.prompt.lower().replace("write a function that ", "").replace("write a function that", "")
                v.prompt = alt_prompt
                v.id = f"{base.id}_v{vi}"
                variants.append(v)
        self.rng.shuffle(variants)
        return variants[:n]

    def generate_by_difficulty(self, n: int, max_difficulty: float) -> list[CodingTask]:
        all_tasks = self._build_all()
        filtered = [t for t in all_tasks if t.difficulty <= max_difficulty]
        if not filtered:
            filtered = all_tasks
        self.rng.shuffle(filtered)
        result = []
        for i in range(n):
            t = filtered[i % len(filtered)]
            result.append(copy.deepcopy(t))
        return result

    def _build_all(self) -> list[CodingTask]:
        tasks = []
        for t in TASKS:
            tasks.append(CodingTask(
                id=t["id"], prompt=t["prompt"], solution=t["solution"],
                tests=t["tests"], difficulty=t["difficulty"], category="write",
            ))
        for t in BUGGY_TASKS:
            tasks.append(CodingTask(
                id=t["id"], prompt=t["prompt"], solution=t["solution"],
                tests=t["tests"], difficulty=t["difficulty"], category="fix",
                buggy_code=t["buggy_code"], bug_explanation=t["bug_explanation"],
            ))
        return tasks
