import ast
import re
from typing import Optional

from deepzero.evaluation.sandbox import CodeSandbox


def check_syntax(code: str) -> tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def count_functions(code: str) -> int:
    try:
        tree = ast.parse(code)
        return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except SyntaxError:
        return 0


def count_classes(code: str) -> int:
    try:
        tree = ast.parse(code)
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    except SyntaxError:
        return 0


def has_docstring(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)):
                    return True
        return False
    except SyntaxError:
        return False


def count_lines(code: str) -> int:
    return len([l for l in code.split("\n") if l.strip()])


def repetition_rate(text: str, ngram: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < ngram * 2:
        return 0.0
    seen = set()
    total = 0
    for i in range(len(tokens) - ngram + 1):
        ngram_tuple = tuple(tokens[i:i + ngram])
        total += 1
        seen.add(ngram_tuple)
    return 1.0 - len(seen) / max(1, total)


def evaluate_code_quality(code: str) -> dict:
    syntax_ok, syntax_err = check_syntax(code)
    result = {
        "syntax_valid": syntax_ok,
        "syntax_error": syntax_err or "",
        "n_functions": count_functions(code),
        "n_classes": count_classes(code),
        "has_docstring": has_docstring(code),
        "n_lines": count_lines(code),
        "n_chars": len(code),
        "repetition_rate_3gram": repetition_rate(code, 3),
        "repetition_rate_4gram": repetition_rate(code, 4),
    }
    if syntax_ok:
        sandbox = CodeSandbox()
        run_result = sandbox.run(code)
        result["exec_success"] = run_result["success"]
        result["exec_output"] = run_result["output"]
        result["exec_error"] = run_result.get("error", "") or run_result.get("exception", "")
    else:
        result["exec_success"] = False
        result["exec_output"] = ""
        result["exec_error"] = syntax_err or ""
    return result


def compression_ratio(tokenizer, texts: list[str]) -> float:
    raw_bytes = sum(len(t.encode("utf-8")) for t in texts)
    token_ids = 0
    for t in texts:
        token_ids += len(tokenizer.encode(t))
    return raw_bytes / max(1, token_ids)
