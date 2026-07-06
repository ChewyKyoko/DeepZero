import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from deepzero.evaluation.benchmark import ALL_TASKS, TASK_IDS
from deepzero.evaluation.executor import SandboxExecutor
from deepzero.evaluation.scoring_v2 import score_task, TaskScore
from deepzero.evaluation.weakness import analyze_weaknesses, generate_weakness_report
from deepzero.models.transformer import GPT
from deepzero.models.checkpoints import load_checkpoint
from deepzero.tokenizers.base import create_tokenizer


@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/best.pt"
    tokenizer_name: str = "bpe"
    tokenizer_path: str = "data/bpe_tokenizer.json"
    tokenizer_vocab_size: int = 5000
    task_ids: list[str] = field(default_factory=lambda: [t.id for t in ALL_TASKS])
    categories: list[str] = field(default_factory=list)
    temperature: float = 0.7
    top_k: int = 40
    max_gen_len: int = 256
    results_dir: str = "results"
    device: str = "cpu"


@dataclass
class EvalExperimentResult:
    experiment_id: str
    config: dict
    model_params: int = 0
    n_tasks: int = 0
    task_scores: list = field(default_factory=list)
    aggregate_score: float = 0.0
    per_category_scores: dict = field(default_factory=dict)
    weakness_analysis: dict = field(default_factory=dict)
    git_commit_hash: str = ""
    timestamp: str = ""
    status: str = "incomplete"


def _get_git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _generate_code(model: GPT, tokenizer, prompt: str, temperature: float = 0.7,
                   top_k: int = 40, max_len: int = 256) -> str:
    try:
        return model.generate(tokenizer, prompt, max_len=max_len,
                             temperature=temperature, top_k=top_k)
    except Exception as e:
        return f"# Generation error: {e}"


def _extract_code(text: str) -> str:
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
        elif not line.strip().startswith("#") and not line.strip().startswith("//"):
            code_lines.append(line)
    return "\n".join(code_lines).strip()


def run_evaluation(cfg: EvalConfig) -> EvalExperimentResult:
    os.makedirs(cfg.results_dir, exist_ok=True)
    git_hash = _get_git_hash()
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = f"eval_{int(time.time())}_{git_hash}"

    result = EvalExperimentResult(
        experiment_id=experiment_id,
        config=asdict(cfg),
        git_commit_hash=git_hash,
        timestamp=timestamp,
    )

    print("=" * 60)
    print(f"DeepZero Evaluation — R0.4")
    print(f"Experiment: {experiment_id}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print("=" * 60)

    print(f"Loading tokenizer ({cfg.tokenizer_name})...")
    tokenizer = create_tokenizer(cfg.tokenizer_name, vocab_size=cfg.tokenizer_vocab_size)
    if os.path.exists(cfg.tokenizer_path):
        tokenizer = tokenizer.load(cfg.tokenizer_path)
        print(f"  Loaded from {cfg.tokenizer_path}")

    print(f"Loading model from {cfg.checkpoint_path}...")
    if not os.path.exists(cfg.checkpoint_path):
        print(f"  Checkpoint not found. Creating untrained model for testing.")
        from deepzero.models.transformer import ModelConfig
        model_cfg = ModelConfig(vocab_size=cfg.tokenizer_vocab_size, device=cfg.device)
        model = GPT(model_cfg)
    else:
        model, state = load_checkpoint(cfg.checkpoint_path, cfg.device)
        print(f"  Model: {model.n_params:,} params")
    result.model_params = model.n_params

    tasks_to_run = [TASK_IDS[tid] for tid in cfg.task_ids if tid in TASK_IDS]
    if cfg.categories:
        tasks_to_run = [t for t in tasks_to_run if t.category in cfg.categories]
    result.n_tasks = len(tasks_to_run)
    print(f"\nRunning {result.n_tasks} tasks...")

    task_scores = []
    for i, task in enumerate(tasks_to_run, 1):
        print(f"  [{i}/{result.n_tasks}] {task.id} ({task.category})...", end=" ", flush=True)
        try:
            generated = _generate_code(model, tokenizer, task.prompt,
                                      temperature=cfg.temperature,
                                      top_k=cfg.top_k,
                                      max_len=cfg.max_gen_len)
            code = _extract_code(generated)
            if not code:
                code = generated
            ts = score_task(code, task.prompt, task.expected_output, task.test_code)
            ts.task_id = task.id
            ts.category = task.category
            task_scores.append(ts)
            print(f"score={ts.final_score:.0f}")
        except Exception as e:
            print(f"error={e}")
            ts = TaskScore(task_id=task.id, category=task.category, prompt=task.prompt,
                          generated_code="", final_score=0.0)
            task_scores.append(ts)

    result.task_scores = [asdict(ts) for ts in task_scores]

    if task_scores:
        result.aggregate_score = sum(ts.final_score for ts in task_scores) / len(task_scores)

        cat_scores = {}
        for ts in task_scores:
            cat = ts.category or "unknown"
            if cat not in cat_scores:
                cat_scores[cat] = {"total": 0, "sum": 0.0, "count": 0}
            cat_scores[cat]["total"] += 1
            cat_scores[cat]["sum"] += ts.final_score
        result.per_category_scores = {
            cat: {"count": d["total"], "avg_score": d["sum"] / max(1, d["total"])}
            for cat, d in cat_scores.items()
        }

        result.weakness_analysis = analyze_weaknesses(task_scores)

    result.status = "completed"
    print(f"\n{'=' * 60}")
    print(f"Aggregate Score: {result.aggregate_score:.1f}/100")
    print(f"Tasks: {result.n_tasks}")
    print(f"{'=' * 60}")

    exp_dir = os.path.join(cfg.results_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    weakness_path = os.path.join(exp_dir, "weakness_report.md")
    generate_weakness_report(task_scores, weakness_path)
    print(f"Weakness report: {weakness_path}")

    return result
