import json
import time
import os
from typing import Optional

from deepzero.models.transformer import GPT
from deepzero.inference.generate import generate
from deepzero.evaluation.scoring import score_solution, ScoreResult
from deepzero.replay.logger import LogEntry, FailureLogger
from deepzero.replay.buffer import TrainingExample, ReplayBuffer
from deepzero.tasks.generator import TaskGenerator, CodingTask
from deepzero.experiments.registry import ExperimentRegistry


class SelfImprovementLoop:
    def __init__(self, model: GPT, tokenizer, registry_dir: str = "outputs/experiments",
                 task_gen: Optional[TaskGenerator] = None,
                 failure_logger: Optional[FailureLogger] = None,
                 replay_buffer: Optional[ReplayBuffer] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.task_gen = task_gen or TaskGenerator()
        self.failure_logger = failure_logger or FailureLogger()
        self.replay_buffer = replay_buffer or ReplayBuffer()
        self.registry = ExperimentRegistry(registry_dir)

    def run_round(self, n_tasks: int = 5, temperature: float = 0.8, top_k: int = 40) -> dict:
        results = {"total": n_tasks, "passed": 0, "failed": 0, "score": 0.0}
        for i in range(n_tasks):
            task = self.task_gen.sample()
            prompt = f"Write Python code for: {task.instruction}\n```python\n"
            response = generate(self.model, self.tokenizer, prompt,
                              temperature=temperature, top_k=top_k)
            result = score_solution(response, task.test_code)
            entry = LogEntry(
                timestamp=time.time(),
                task=task.name,
                prompt=prompt,
                response=response,
                passed=result.passed,
                score=result.score,
                error=result.error,
            )
            if result.passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                self.failure_logger.log(entry)
            results["score"] += result.score
        results["score"] /= max(1, n_tasks)
        return results

    def run_experiment(self, name: str, n_rounds: int = 3, tasks_per_round: int = 5,
                       temperature: float = 0.8, top_k: int = 40,
                       description: str = "") -> str:
        run_id = self.registry.create_run(name, description, {
            "n_rounds": n_rounds,
            "tasks_per_round": tasks_per_round,
            "temperature": temperature,
            "top_k": top_k,
        })
        for r in range(n_rounds):
            print(f"Round {r+1}/{n_rounds}")
            results = self.run_round(n_tasks=tasks_per_round, temperature=temperature, top_k=top_k)
            self.registry.log_round(run_id, r, results)
        self.registry.finalize_run(run_id)
        return run_id
