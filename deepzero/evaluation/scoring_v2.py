import difflib
from dataclasses import dataclass, field
from typing import Optional

from deepzero.evaluation.executor import SandboxExecutor
from deepzero.evaluation.coding import check_syntax, count_functions, count_classes, count_lines


@dataclass
class TaskScore:
    task_id: str
    category: str
    prompt: str
    generated_code: str = ""
    syntax_valid: bool = False
    syntax_error: str = ""
    exec_success: bool = False
    exec_output: str = ""
    exec_error: str = ""
    timed_out: bool = False
    n_tests: int = 0
    n_passed: int = 0
    test_results: list[bool] = field(default_factory=list)
    output_similarity: float = 0.0
    code_quality_score: float = 0.0
    final_score: float = 0.0
    failure_category: str = ""


def _output_similarity(generated: str, expected: str) -> float:
    if not expected:
        return 0.5
    gen = generated.strip().split()
    exp = expected.strip().split()
    if not gen and not exp:
        return 1.0
    if not gen or not exp:
        return 0.0
    matcher = difflib.SequenceMatcher(None, gen, exp)
    return matcher.ratio()


def _code_quality_heuristic(code: str) -> float:
    if not code.strip():
        return 0.0
    score = 0.5
    n_funcs = count_functions(code)
    n_classes = count_classes(code)
    n_lines = count_lines(code)
    if n_funcs > 0:
        score += 0.15
    if n_classes > 0:
        score += 0.1
    if 5 <= n_lines <= 100:
        score += 0.1
    if code.count("def ") > 0:
        score += 0.05
    if code.count('"""') > 0 or code.count("'''") > 0:
        score += 0.05
    if code.count("#") > 0:
        score += 0.05
    return min(1.0, score)


def _categorize_failure(task_score: TaskScore) -> str:
    if task_score.timed_out:
        return "infinite_loop"
    if not task_score.syntax_valid:
        return "syntax_error"
    if not task_score.exec_success:
        if task_score.exec_error and "NameError" in task_score.exec_error:
            return "hallucinated_api"
        return "logic_error"
    if task_score.n_tests > 0 and task_score.n_passed < task_score.n_tests:
        return "logic_error"
    if not task_score.generated_code.strip():
        return "incomplete_solution"
    if len(task_score.generated_code) < 10:
        return "incomplete_solution"
    return "unknown"


def score_task(generated_code: str, task_prompt: str, expected_output: str = "",
               test_code: str = "") -> TaskScore:
    score = TaskScore(
        task_id="",
        category="",
        prompt=task_prompt,
        generated_code=generated_code,
    )

    stripped = generated_code.strip()
    if not stripped or len(stripped) < 5:
        score.failure_category = "incomplete_solution"
        score.final_score = 5.0
        return score

    syntax_ok, syntax_err = check_syntax(generated_code)
    score.syntax_valid = syntax_ok
    score.syntax_error = syntax_err or ""

    if not syntax_ok:
        score.final_score = 10.0
        score.failure_category = _categorize_failure(score)
        return score

    executor = SandboxExecutor(timeout=5)
    if test_code:
        combined = f"{generated_code}\n\n{test_code}"
        result = executor.run_code(combined)
    else:
        result = executor.run_code(generated_code)

    score.exec_success = result["success"]
    score.exec_output = result["output"]
    score.exec_error = result["error"] or result.get("exception", "")
    score.timed_out = result.get("timed_out", False)

    if test_code and result["output"]:
        score.test_results = [line == expected_output.strip() for line in result["output"].strip().split("\n") if line]
        score.n_tests = len(score.test_results) if score.test_results else 1
        score.n_passed = sum(score.test_results)
    elif result["success"]:
        score.n_tests = 1
        score.n_passed = 1
        score.test_results = [True]

    score.output_similarity = _output_similarity(result["output"], expected_output) if expected_output else 0.5
    score.code_quality_score = _code_quality_heuristic(generated_code)
    score.failure_category = _categorize_failure(score)

    test_pass_rate = score.n_passed / max(1, score.n_tests)

    score.final_score = (
        0.30 * test_pass_rate +
        0.25 * (1.0 if score.syntax_valid else 0.0) +
        0.20 * (1.0 if score.exec_success else 0.0) +
        0.15 * score.output_similarity +
        0.10 * score.code_quality_score
    ) * 100.0

    return score


def score_tasks(generated_codes: list[str], tasks: list) -> list[TaskScore]:
    scores = []
    for gen_code, task in zip(generated_codes, tasks):
        ts = score_task(gen_code, task.prompt, task.expected_output, task.test_code)
        ts.task_id = task.id
        ts.category = task.category
        scores.append(ts)
    return scores
