import unittest
import tempfile
import os
import json


class TestDatasetBase(unittest.TestCase):
    def test_create_dataset(self):
        from deepzero.datasets.base import create_dataset, DATASET_REGISTRY
        for name in ("tiny_codes", "humaneval", "mbpp", "the_stack", "local", "replay"):
            ds = create_dataset(name)
            self.assertIsNotNone(ds)
            self.assertEqual(ds.name, name)
        self.assertIn("tiny_codes", DATASET_REGISTRY)

    def test_dataset_registry(self):
        from deepzero.datasets.base import DATASET_REGISTRY
        expected = {"tiny_codes", "humaneval", "mbpp", "the_stack", "local", "replay"}
        self.assertTrue(expected.issubset(set(DATASET_REGISTRY.keys())))

    def test_local_dataset(self):
        from deepzero.datasets.local import LocalDataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\nprint(x)\n")
            path = f.name
        try:
            ds = LocalDataset(path=path)
            ds.preprocess()
            self.assertGreater(len(ds.load_texts()), 0)
            stats = ds.statistics()
            self.assertIn("n_samples", stats)
        finally:
            os.unlink(path)

    def test_replay_dataset(self):
        from deepzero.datasets.replay import ReplayDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_path = os.path.join(tmpdir, "replay.jsonl")
            with open(replay_path, "w") as f:
                f.write(json.dumps({"prompt": "def foo():", "response": "    pass"}) + "\n")
            ds = ReplayDataset(cache_dir=tmpdir, replay_path="replay.jsonl")
            ds.preprocess()
            self.assertGreater(len(ds.load_texts()), 0)

    def test_humaneval_init(self):
        from deepzero.datasets.humaneval import HumanEvalDataset
        ds = HumanEvalDataset(cache_dir="/tmp/test_humaneval")
        self.assertEqual(ds.name, "humaneval")
        stats = ds.statistics()
        self.assertIn("source", stats)

    def test_mbpp_init(self):
        from deepzero.datasets.mbpp import MBPPDataset
        ds = MBPPDataset(cache_dir="/tmp/test_mbpp")
        self.assertEqual(ds.name, "mbpp")
        stats = ds.statistics()
        self.assertIn("source", stats)

    def test_tiny_codes_v2_init(self):
        from deepzero.datasets.tiny_codes_v2 import TinyCodesDataset
        ds = TinyCodesDataset(cache_dir="/tmp/test_tiny_codes_v2")
        self.assertEqual(ds.name, "tiny_codes")
        stats = ds.statistics()
        self.assertIn("language", stats)

    def test_the_stack_init(self):
        from deepzero.datasets.the_stack import TheStackDataset
        ds = TheStackDataset(cache_dir="/tmp/test_stack", max_samples=100)
        self.assertEqual(ds.name, "the_stack")
        stats = ds.statistics()
        self.assertIn("max_samples", stats)

    def test_local_dataset_missing_path(self):
        from deepzero.datasets.local import LocalDataset
        ds = LocalDataset(path="/nonexistent/path")
        with self.assertRaises(FileNotFoundError):
            ds.preprocess()


class TestDatasetMixing(unittest.TestCase):
    def test_mixture_basic(self):
        from deepzero.datasets.mixture import DatasetMixture
        from deepzero.datasets.local import LocalDataset
        mix = DatasetMixture(name="test_mix")
        ds = LocalDataset(path="/tmp")
        mix.add(ds, weight=1.0)
        self.assertEqual(len(mix.components), 1)
        stats = mix.statistics()
        self.assertEqual(stats["name"], "test_mix")

    def test_mixture_from_config(self):
        from deepzero.datasets.mixture import DatasetMixture
        config = {
            "name": "test",
            "components": [
                {"name": "local", "weight": 0.5, "kwargs": {"path": "/tmp"}},
                {"name": "local", "weight": 0.5, "kwargs": {"path": "/tmp"}},
            ]
        }
        mix = DatasetMixture.from_config(config)
        self.assertEqual(len(mix.components), 2)

    def test_load_mixture_from_yaml(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("yaml not available")
        from deepzero.datasets.mixture import load_mixture_from_yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test\ncomponents:\n  - name: local\n    weight: 1.0\n    kwargs:\n      path: /tmp\n")
            path = f.name
        try:
            mix = load_mixture_from_yaml(path)
            self.assertEqual(mix.name, "test")
        finally:
            os.unlink(path)


class TestQualityAnalysis(unittest.TestCase):
    def test_analyze_dataset(self):
        from deepzero.datasets.quality import analyze_dataset
        texts = ["def foo(): pass\n", "x = 1\nprint(x)\n"]
        analysis = analyze_dataset(texts, "test")
        self.assertIn("n_samples", analysis)
        self.assertEqual(analysis["n_samples"], 2)
        self.assertIn("avg_length", analysis)
        self.assertIn("syntax_valid_rate", analysis)
        self.assertIn("vocabulary", analysis)
        self.assertIn("language_distribution", analysis)

    def test_quality_report(self):
        from deepzero.datasets.quality import generate_quality_report
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_quality_report(["def foo(): pass\n"], "test_ds", tmpdir)
            self.assertTrue(os.path.exists(path))

    def test_comment_ratio(self):
        from deepzero.datasets.quality import _comment_ratio
        text = "# comment\ncode\n# another\n"
        ratio = _comment_ratio(text)
        self.assertAlmostEqual(ratio, 0.5, places=1)

    def test_vocabulary_stats(self):
        from deepzero.datasets.quality import _vocabulary_stats
        stats = _vocabulary_stats(["hello world hello"])
        self.assertIn("unique_chars", stats)
        self.assertIn("lexical_diversity", stats)


class TestDatasetBenchConfig(unittest.TestCase):
    def test_config_defaults(self):
        from deepzero.experiments.dataset_bench import DatasetBenchConfig
        cfg = DatasetBenchConfig()
        self.assertIn("tiny_codes", cfg.dataset_names)
        self.assertEqual(cfg.tokenizer_name, "bpe")
        self.assertEqual(cfg.training_max_iters, 3000)

    def test_experiment_result(self):
        from deepzero.experiments.dataset_bench import DatasetExperimentResult
        r = DatasetExperimentResult(experiment_id="test", dataset_name="test", config={})
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.dataset_name, "test")


class TestResearchDoc(unittest.TestCase):
    def test_generate_research_doc(self):
        from deepzero.experiments.research_doc import generate_research_doc
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            with open(results_path, "w") as f:
                json.dump([{
                    "experiment_id": "test_1",
                    "dataset_name": "test_ds",
                    "config": {"seed": 42},
                    "quality_analysis": {"n_samples": 100, "avg_length": 50, "estimated_tokens": 500},
                    "training_stats": {"final_loss": 1.5, "final_perplexity": 4.5, "tokens_per_second": 100,
                                       "training_time_seconds": 10, "best_val_loss": 1.5, "n_steps": 100},
                    "generation_stats": {"syntax_valid_rate": 0.8, "compile_success_rate": 0.6,
                                         "avg_output_length": 30, "avg_repetition_rate_3gram": 0.1},
                    "coding_quality": {"avg_n_functions": 2.0, "avg_n_classes": 0.5, "avg_n_lines": 15,
                                       "syntax_valid_rate": 0.8, "exec_success_rate": 0.6},
                    "git_commit_hash": "abc123",
                    "timestamp": "2024-01-01T00:00:00",
                    "status": "completed",
                }], f)
            doc_path = generate_research_doc(results_path, os.path.join(tmpdir, "R0.3_test.md"))
            self.assertTrue(os.path.exists(doc_path))
            with open(doc_path) as f:
                content = f.read()
            self.assertIn("R0.3 Dataset Study", content)
            self.assertIn("Objective", content)


class TestDatasetReporting(unittest.TestCase):
    def setUp(self):
        self.sample_results = [{
            "experiment_id": "ds_tiny_codes_123_abc",
            "dataset_name": "tiny_codes",
            "config": {"seed": 42},
            "quality_analysis": {"n_samples": 1000, "avg_length": 200, "total_chars": 200000,
                                 "syntax_valid_rate": 0.95, "estimated_tokens": 50000,
                                 "avg_functions_per_sample": 3.0, "avg_classes_per_sample": 0.5,
                                 "avg_comment_ratio": 0.1, "avg_docstring_ratio": 0.05},
            "training_stats": {"final_loss": 1.2, "final_perplexity": 3.3, "tokens_per_second": 500,
                               "training_time_seconds": 30, "best_val_loss": 1.2, "n_steps": 100},
            "generation_stats": {"syntax_valid_rate": 0.8, "compile_success_rate": 0.6,
                                 "avg_output_length": 40, "avg_repetition_rate_3gram": 0.1},
            "coding_quality": {"avg_n_functions": 2.0, "avg_n_classes": 0.5, "avg_n_lines": 20,
                               "syntax_valid_rate": 0.8, "exec_success_rate": 0.6},
            "git_commit_hash": "abc123",
            "timestamp": "2024-01-01T00:00:00",
            "status": "completed",
        }]

    def test_generate_dataset_csv(self):
        from deepzero.experiments.dataset_reporting import generate_dataset_csv
        with tempfile.TemporaryDirectory() as tmpdir:
            rp = os.path.join(tmpdir, "res.json")
            with open(rp, "w") as f:
                json.dump(self.sample_results, f)
            csv_p = generate_dataset_csv(rp, os.path.join(tmpdir, "out.csv"))
            self.assertTrue(os.path.exists(csv_p))

    def test_generate_dataset_report(self):
        from deepzero.experiments.dataset_reporting import generate_dataset_report
        with tempfile.TemporaryDirectory() as tmpdir:
            rp = os.path.join(tmpdir, "res.json")
            with open(rp, "w") as f:
                json.dump(self.sample_results, f)
            md_p = generate_dataset_report(rp, os.path.join(tmpdir, "out.md"))
            self.assertTrue(os.path.exists(md_p))


if __name__ == "__main__":
    unittest.main()
