import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodingTask:
    name: str = ""
    instruction: str = ""
    test_code: str = ""
    difficulty: str = "easy"
    metadata: dict = field(default_factory=dict)


EASY_TASKS = [
    CodingTask(name="hello_world", instruction="Write a function that prints 'Hello, World!'",
               test_code="print('Hello, World!')", difficulty="easy"),
    CodingTask(name="add_numbers", instruction="Write a function that adds two numbers and prints the result",
               test_code="print(2 + 3)", difficulty="easy"),
    CodingTask(name="reverse_string", instruction="Write a function that reverses a string",
               test_code="print('hello'[::-1])", difficulty="easy"),
]

MEDIUM_TASKS = [
    CodingTask(name="fibonacci", instruction="Write a function that computes the nth Fibonacci number",
               test_code="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)\nprint(fib(10))", difficulty="medium"),
    CodingTask(name="is_palindrome", instruction="Check if a string is a palindrome",
               test_code="def is_pal(s): return s == s[::-1]\nprint(is_pal('racecar'))", difficulty="medium"),
]

HARD_TASKS = [
    CodingTask(name="quicksort", instruction="Implement quicksort",
               test_code="def qs(a): return a if len(a)<=1 else qs([x for x in a[1:] if x<=a[0]])+[a[0]]+qs([x for x in a[1:] if x>a[0]])\nprint(qs([3,1,4,1,5,9]))", difficulty="hard"),
]


class TaskGenerator:
    def __init__(self):
        self.tasks = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS

    def sample(self, difficulty: Optional[str] = None) -> CodingTask:
        pool = [t for t in self.tasks if difficulty is None or t.difficulty == difficulty]
        return random.choice(pool)

    def get(self, name: str) -> Optional[CodingTask]:
        for t in self.tasks:
            if t.name == name:
                return t
        return None
