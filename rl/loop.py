"""Self-improvement loop — generate → evaluate → log → buffer → retrain."""

import math
import os
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from rl.config import N_SOLUTIONS_PER_TASK, FAILED_PRIORITY, NEAR_MISS_PRIORITY, FT_LR, FT_BATCH_SIZE, FT_MAX_STEPS, EVAL_TIMEOUT
from rl.tasks import TaskGenerator
from rl.evaluate import CodeEvaluator
from rl.logger import FailureLogger, LogEntry
from rl.buffer import ReplayBuffer


class RLDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, seq_len: int):
        self.tokens = []
        for t in texts:
            ids = tokenizer.encode(t)
            self.tokens.extend(ids)

    def __len__(self):
        return max(0, len(self.tokens) - 512)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + 513]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def fine_tune(model, tokenizer, texts: list[str], cfg, run_dir: str = "runs/rl"):
    if not texts:
        print("  No training data — skipping fine-tune")
        return

    os.makedirs(run_dir, exist_ok=True)
    device = torch.device(cfg.device)
    dataset = RLDataset(texts, tokenizer, cfg.max_seq_len)

    if len(dataset) == 0:
        print(f"  Dataset too small ({len(dataset)} samples)")
        return

    loader = DataLoader(dataset, batch_size=FT_BATCH_SIZE, shuffle=True, drop_last=True)
    it = iter(loader)

    optimizer = AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
    model.train()
    best_loss = float("inf")
    steps = min(FT_MAX_STEPS, max(100, len(dataset) // FT_BATCH_SIZE * 3))

    print(f"  Fine-tuning on {len(dataset)} sequences, {steps} steps")
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == steps:
            print(f"    step {step}/{steps} | loss {loss.item():.4f} | ppl {math.exp(min(loss.item(), 10)):.2f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({"step": step, "model": model.state_dict(), "loss": loss.item()},
                       os.path.join(run_dir, "rl_best.pt"))

    return model


class SelfImprovementLoop:
    def __init__(self, model, tokenizer, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.task_gen = TaskGenerator()
        self.evaluator = CodeEvaluator(timeout=EVAL_TIMEOUT)
        self.logger = FailureLogger()
        self.buffer = ReplayBuffer(max_size=cfg.max_seq_len)

    def run(self, iterations: int = 3, tasks_per_iter: int = 5):
        device = torch.device(self.cfg.device)
        self.model.to(device)

        all_results = []

        for it in range(1, iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {it}/{iterations}")
            print(f"{'='*60}")

            # Step 1: Generate tasks
            tasks = self.task_gen.generate(tasks_per_iter)
            print(f"  Generated {len(tasks)} tasks")

            # Step 2: Solve + evaluate each task
            it_entries = []
            for task in tasks:
                prompt = task.full_prompt
                solutions = self._sample_solutions(prompt, N_SOLUTIONS_PER_TASK)
                for sol in solutions:
                    result = self.evaluator.evaluate(sol, task.tests)
                    entry = LogEntry(
                        task_id=task.id, task_prompt=prompt,
                        solution_code=sol, correct_solution=task.solution,
                        score=result.score, passed=result.passed, total=result.total,
                        errors=result.errors, category=task.category,
                        difficulty=task.difficulty, timestamp=time.time(),
                        iteration=it,
                    )
                    self.logger.log(entry)
                    it_entries.append(entry)

            # Step 3: Log outcomes
            stats = self.logger.stats()
            print(f"  [{it}] Total: {stats['total']}, Failures: {stats['failures']}, "
                  f"Near-misses: {stats['near_misses']}, Avg score: {stats['avg_score']:.3f}")

            # Step 4: Build replay buffer from failures + near-misses
            failures = self.logger.failures(limit=100, iteration=it)
            near_misses = self.logger.near_misses(limit=100, iteration=it)

            added = 0
            if failures:
                self.buffer.add_from_log(failures, priority=FAILED_PRIORITY)
                added += len(failures)
            if near_misses:
                self.buffer.add_from_log(near_misses, priority=NEAR_MISS_PRIORITY)
                added += len(near_misses)
            print(f"  Added {added} examples to replay buffer (total: {self.buffer.size()})")

            # Step 5: Fine-tune on buffer
            texts = self.buffer.build_texts()
            if texts:
                fine_tune(self.model, self.tokenizer, texts, self.cfg)
                # Load best checkpoint
                ckpt_path = os.path.join("runs/rl", "rl_best.pt")
                if os.path.exists(ckpt_path):
                    self.model.load_state_dict(
                        torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
                    )
                    print(f"  Loaded best RL checkpoint")

            # Step 6: Evaluate improvement
            score_delta = self._eval_improvement(tasks)
            print(f"  Improvement delta: {score_delta:+.3f}")

            all_results.append({
                "iteration": it,
                "stats": stats,
                "buffer_size": self.buffer.size(),
                "score_delta": score_delta,
            })

        return all_results

    def _sample_solutions(self, prompt: str, n: int) -> list[str]:
        self.model.eval()
        solutions = []
        for _ in range(n):
            out = self.model.generate(
                self.tokenizer, prompt,
                max_len=256, temperature=0.8, top_k=40,
            )
            solutions.append(out)
        return solutions

    def _eval_improvement(self, tasks) -> float:
        scores = []
        for task in tasks[:3]:
            prompt = task.full_prompt
            sol = self._sample_solutions(prompt, 1)[0]
            result = self.evaluator.evaluate(sol, task.tests)
            scores.append(result.score)
        return sum(scores) / max(len(scores), 1)
