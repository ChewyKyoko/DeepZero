"""Sandboxed code execution + correctness scoring."""

import subprocess
import sys
import tempfile
import os
import traceback
from dataclasses import dataclass


@dataclass
class EvalResult:
    score: float           # 0.0 to 1.0
    passed: int
    total: int
    errors: list[str]
    stdout: str
    stderr: str
    timed_out: bool = False


class CodeEvaluator:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def evaluate(self, code: str, tests: str) -> EvalResult:
        full_code = f"{code}\n\n{tests}"
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return EvalResult(0.0, 0, self._count_tests(tests), ["Timed out"], "", "", timed_out=True)
        except Exception as e:
            return EvalResult(0.0, 0, self._count_tests(tests), [str(e)], "", "")

        errors = self._parse_errors(result.stderr)
        total = self._count_tests(tests)
        has_error = bool(result.stderr.strip())
        failed = total if has_error else self._count_failed(result.stderr)
        passed = total - failed
        score = passed / max(total, 1)
        return EvalResult(score, passed, total, errors, result.stdout, result.stderr)

    def _count_tests(self, tests: str) -> int:
        return sum(1 for line in tests.strip().split("\n") if line.startswith("assert"))

    def _count_failed(self, stderr: str) -> int:
        if not stderr:
            return 0
        return len([l for l in stderr.split("\n") if "AssertionError" in l])

    def _parse_errors(self, stderr: str) -> list[str]:
        if not stderr:
            return []
        lines = stderr.strip().split("\n")
        errors = []
        for line in lines:
            if "AssertionError" in line or "Error" in line or "Exception" in line:
                errors.append(line.strip())
        if not errors and stderr:
            errors.append(stderr.strip()[:200])
        return errors

    def score_near_miss(self, result: EvalResult) -> float:
        if result.total == 0:
            return 0.0
        return result.passed / result.total
