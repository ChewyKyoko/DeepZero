import ast
import os
from typing import Optional

from deepzero.evaluation.coding import (
    check_syntax, count_functions, count_classes, has_docstring, count_lines
)


def _comment_ratio(text: str) -> float:
    lines = text.split("\n")
    n_comments = sum(1 for l in lines if l.strip().startswith("#"))
    return n_comments / max(1, len(lines))


def _docstring_ratio(text: str) -> float:
    count = 0
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)):
                    count += 1
    except SyntaxError:
        pass
    return count / max(1, len(text.split("\n")))


def _vocabulary_stats(texts: list[str]) -> dict:
    all_words: list[str] = []
    unique_chars: set[str] = set()
    for t in texts:
        unique_chars.update(t)
        all_words.extend(t.split())
    return {
        "unique_chars": len(unique_chars),
        "unique_words": len(set(all_words)),
        "total_words": len(all_words),
        "lexical_diversity": len(set(all_words)) / max(1, len(all_words)),
    }


def _language_distribution(texts: list[str]) -> dict:
    from deepzero.datasets.pipeline import LANGUAGES
    counts: dict[str, int] = {}
    for lang in LANGUAGES:
        counts[lang] = 0
    counts["unknown"] = 0
    keywords = {
        "python": {"def ", "import ", "class ", "if __name__"},
        "cpp": {"#include", "int main", "std::"},
        "rust": {"fn ", "let mut", "use std"},
        "javascript": {"function", "const ", "let ", "=>"},
        "go": {"func ", "package ", "import ("},
    }
    for t in texts:
        detected = False
        for lang, kws in keywords.items():
            if any(kw in t for kw in kws):
                counts[lang] = counts.get(lang, 0) + 1
                detected = True
                break
        if not detected:
            counts["unknown"] = counts.get("unknown", 0) + 1
    return counts


def _estimated_tokens(texts: list[str]) -> int:
    return sum(len(t.split()) for t in texts)


def analyze_dataset(texts: list[str], name: str = "dataset") -> dict:
    lengths = [len(t) for t in texts]
    syntax_valid = sum(1 for t in texts if check_syntax(t)[0])

    total_functions = sum(count_functions(t) for t in texts)
    total_classes = sum(count_classes(t) for t in texts)

    return {
        "name": name,
        "n_samples": len(texts),
        "total_chars": sum(lengths),
        "total_lines": sum(t.count("\n") for t in texts),
        "avg_length": sum(lengths) / max(1, len(lengths)),
        "median_length": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "std_length": (sum((l - sum(lengths)/max(1, len(lengths)))**2 for l in lengths) / max(1, len(lengths)))**0.5 if lengths else 0,
        "syntax_valid_rate": syntax_valid / max(1, len(texts)),
        "total_functions": total_functions,
        "total_classes": total_classes,
        "avg_functions_per_sample": total_functions / max(1, len(texts)),
        "avg_classes_per_sample": total_classes / max(1, len(texts)),
        "avg_comment_ratio": sum(_comment_ratio(t) for t in texts) / max(1, len(texts)),
        "avg_docstring_ratio": sum(_docstring_ratio(t) for t in texts) / max(1, len(texts)),
        "estimated_tokens": _estimated_tokens(texts),
        "vocabulary": _vocabulary_stats(texts),
        "language_distribution": _language_distribution(texts),
    }


def generate_quality_report(texts: list[str], name: str = "dataset", output_dir: str = "results") -> str:
    import json
    os.makedirs(output_dir, exist_ok=True)
    analysis = analyze_dataset(texts, name)
    report_path = os.path.join(output_dir, f"{name}_quality_report.json")
    with open(report_path, "w") as f:
        json.dump(analysis, f, indent=2)
    md_path = os.path.join(output_dir, f"{name}_quality_report.md")
    lines = [
        f"# Dataset Quality Report: {name}",
        "",
        "## Overview",
        f"- Samples: {analysis['n_samples']:,}",
        f"- Total chars: {analysis['total_chars']:,}",
        f"- Total lines: {analysis['total_lines']:,}",
        f"- Estimated tokens: {analysis['estimated_tokens']:,}",
        "",
        "## Length Statistics",
        f"- Mean: {analysis['avg_length']:.1f}",
        f"- Median: {analysis['median_length']:.0f}",
        f"- Min: {analysis['min_length']}",
        f"- Max: {analysis['max_length']}",
        f"- Std Dev: {analysis['std_length']:.1f}",
        "",
        "## Code Quality",
        f"- Syntax valid rate: {analysis['syntax_valid_rate']:.1%}",
        f"- Total functions: {analysis['total_functions']:,}",
        f"- Total classes: {analysis['total_classes']:,}",
        f"- Avg functions/sample: {analysis['avg_functions_per_sample']:.2f}",
        f"- Avg classes/sample: {analysis['avg_classes_per_sample']:.2f}",
        f"- Avg comment ratio: {analysis['avg_comment_ratio']:.2%}",
        f"- Avg docstring ratio: {analysis['avg_docstring_ratio']:.2%}",
        "",
        "## Vocabulary",
        f"- Unique chars: {analysis['vocabulary']['unique_chars']:,}",
        f"- Unique words: {analysis['vocabulary']['unique_words']:,}",
        f"- Lexical diversity: {analysis['vocabulary']['lexical_diversity']:.4f}",
        "",
        "## Language Distribution",
    ]
    for lang, count in sorted(analysis["language_distribution"].items()):
        lines.append(f"- {lang}: {count:,} ({count/max(1, analysis['n_samples']):.1%})")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return md_path
