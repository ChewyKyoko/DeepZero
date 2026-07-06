from dataclasses import dataclass, field
from typing import Optional
from deepzero.evaluation.sandbox import CodeSandbox


@dataclass
class ScoreResult:
    passed: bool = False
    output: str = ""
    error: str = ""
    n_tests: int = 0
    n_passed: int = 0
    score: float = 0.0


def score_solution(solution: str, test_code: str = "") -> ScoreResult:
    sandbox = CodeSandbox()
    if not test_code:
        test_code = solution
    result = sandbox.run(test_code)
    sr = ScoreResult(
        passed=result["success"],
        output=result["output"],
        error=result["error"] or result["exception"],
    )
    if result["success"]:
        sr.score = 1.0
        sr.n_tests = 1
        sr.n_passed = 1
    return sr
