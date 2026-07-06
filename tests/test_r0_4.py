import unittest
import json
import tempfile
import os

from deepzero.evaluation.benchmark import (
    CodingTask, ALL_TASKS, ALL_TASKS_BY_CATEGORY, get_task, get_tasks_by_category, get_categories
)
from deepzero.evaluation.executor import SandboxExecutor
from deepzero.evaluation.scoring_v2 import score_task, TaskScore
from deepzero.evaluation.weakness import analyze_weaknesses, generate_weakness_report


class TestBenchmarkTasks(unittest.TestCase):
    def test_all_tasks_nonempty(self):
        self.assertGreater(len(ALL_TASKS), 0)

    def test_categories(self):
        cats = get_categories()
        self.assertIn("algorithms", cats)
        self.assertIn("data_structures", cats)
        self.assertIn("programming", cats)
        self.assertIn("debugging", cats)
        self.assertIn("projects", cats)

    def test_get_task_by_id(self):
        task = get_task("fibonacci")
        self.assertIsNotNone(task)
        self.assertEqual(task.category, "algorithms")

    def test_get_task_invalid(self):
        self.assertIsNone(get_task("nonexistent"))

    def test_get_tasks_by_category(self):
        algos = get_tasks_by_category("algorithms")
        self.assertGreater(len(algos), 0)
        for t in algos:
            self.assertEqual(t.category, "algorithms")

    def test_each_task_has_required_fields(self):
        for t in ALL_TASKS:
            self.assertTrue(t.id)
            self.assertTrue(t.category)
            self.assertTrue(t.prompt)
            self.assertTrue(t.test_code)
            self.assertTrue(t.expected_output)

    def test_missing_test_code_has_expected_output(self):
        for t in ALL_TASKS:
            if t.test_code:
                self.assertTrue(t.expected_output, f"Task {t.id} has test_code but no expected_output")

    def test_task_ids_unique(self):
        ids = [t.id for t in ALL_TASKS]
        self.assertEqual(len(ids), len(set(ids)))


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.exec = SandboxExecutor(timeout=5)

    def test_run_success(self):
        result = self.exec.run_code("print('hello')")
        self.assertTrue(result["success"])
        self.assertEqual(result["output"].strip(), "hello")

    def test_run_syntax_error(self):
        result = self.exec.run_code("x = ")
        self.assertFalse(result["success"])
        self.assertIn("SyntaxError", result["error"])

    def test_run_exception(self):
        result = self.exec.run_code("raise ValueError('test')")
        self.assertFalse(result["success"])
        self.assertIn("ValueError", result["exception"])

    def test_run_with_stdin(self):
        result = self.exec.run_code("print(input().upper())", stdin="hello")
        self.assertTrue(result["success"])
        self.assertEqual(result["output"].strip(), "HELLO")

    def test_check_syntax(self):
        ok, err = self.exec.check_syntax("x = 1")
        self.assertTrue(ok)
        ok, err = self.exec.check_syntax("x = ")
        self.assertFalse(ok)

    def test_generate_and_test(self):
        gen = "def add(a, b): return a + b"
        test = "assert add(2, 3) == 5\nprint('PASS')"
        result = self.exec.generate_and_test(gen, test)
        self.assertTrue(result["success"])

    def test_restricted_builtins(self):
        result = self.exec.run_code("print(open('/etc/passwd').read())")
        self.assertFalse(result["success"])
        self.assertTrue("Error" in result.get("exception", "") or "Error" in result.get("error", ""))


class TestScoringV2(unittest.TestCase):
    def test_score_correct_code(self):
        ts = score_task(
            "def add(a, b): return a + b",
            "Add two numbers",
            "5\n",
            "assert add(2, 3) == 5\nprint('')",
        )
        self.assertGreater(ts.final_score, 50)
        self.assertTrue(ts.syntax_valid)
        self.assertTrue(ts.exec_success)

    def test_score_syntax_error(self):
        ts = score_task("def add(a, b:", "Add two numbers")
        self.assertFalse(ts.syntax_valid)
        self.assertLess(ts.final_score, 50)
        self.assertEqual(ts.failure_category, "syntax_error")

    def test_score_empty_code(self):
        ts = score_task("", "Do something")
        self.assertEqual(ts.final_score, 5.0)
        self.assertEqual(ts.failure_category, "incomplete_solution")

    def test_output_similarity(self):
        from deepzero.evaluation.scoring_v2 import _output_similarity
        sim = _output_similarity("hello world", "hello world")
        self.assertGreater(sim, 0.9)
        sim = _output_similarity("hello", "goodbye")
        self.assertLess(sim, 0.5)


class TestWeaknessAnalysis(unittest.TestCase):
    def test_analyze_weaknesses(self):
        scores = [
            TaskScore(task_id="a", category="algorithms", prompt="", final_score=90.0,
                     syntax_valid=True, exec_success=True, n_tests=1, n_passed=1),
            TaskScore(task_id="b", category="algorithms", prompt="", final_score=20.0,
                     syntax_valid=False, syntax_error="bad syntax", failure_category="syntax_error"),
            TaskScore(task_id="c", category="debugging", prompt="", final_score=10.0,
                     timed_out=True, failure_category="infinite_loop"),
        ]
        analysis = analyze_weaknesses(scores)
        self.assertIn("top_5_failure_categories", analysis)
        self.assertIn("failure_distribution", analysis)
        self.assertIn("per_category_scores", analysis)

    def test_weakness_report(self):
        scores = [
            TaskScore(task_id="test", category="algorithms", prompt="Write code", final_score=50.0,
                     syntax_valid=True, exec_success=False, exec_error="ValueError: x",
                     failure_category="logic_error"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weakness.md")
            report = generate_weakness_report(scores, path)
            self.assertTrue(os.path.exists(path))
            self.assertIn("Weakness Analysis", report)


class TestEvalConfig(unittest.TestCase):
    def test_config_defaults(self):
        from deepzero.experiments.eval_bench import EvalConfig
        cfg = EvalConfig()
        self.assertEqual(cfg.tokenizer_name, "bpe")
        self.assertGreater(len(cfg.task_ids), 0)

    def test_experiment_result(self):
        from deepzero.experiments.eval_bench import EvalExperimentResult
        r = EvalExperimentResult(experiment_id="test", config={})
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.experiment_id, "test")

    def test_eval_reporting_csv(self):
        from deepzero.experiments.eval_reporting import generate_benchmark_csv
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [{
                "experiment_id": "eval_test",
                "model_params": 1000,
                "n_tasks": 5,
                "aggregate_score": 75.0,
                "per_category_scores": {"algorithms": {"count": 3, "avg_score": 80.0}},
                "status": "completed",
            }]
            csv_p = generate_benchmark_csv(results, os.path.join(tmpdir, "out.csv"))
            self.assertTrue(os.path.exists(csv_p))

    def test_eval_reporting_md(self):
        from deepzero.experiments.eval_reporting import generate_benchmark_report
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [{
                "experiment_id": "eval_test",
                "model_params": 1000,
                "n_tasks": 5,
                "aggregate_score": 75.0,
                "per_category_scores": {"algorithms": {"count": 3, "avg_score": 80.0}},
                "weakness_analysis": {
                    "failure_rate": 0.2,
                    "top_5_failure_categories": [{"category": "syntax", "count": 2, "label": "Syntax Errors"}],
                },
                "task_scores": [{"task_id": "fib", "category": "algorithms", "final_score": 80,
                                "syntax_valid": True, "exec_success": True, "n_passed": 1, "n_tests": 1}],
                "status": "completed",
            }]
            md_p = generate_benchmark_report(results, os.path.join(tmpdir, "out.md"))
            self.assertTrue(os.path.exists(md_p))


if __name__ == "__main__":
    unittest.main()
