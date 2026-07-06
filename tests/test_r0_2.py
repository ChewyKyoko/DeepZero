import unittest
import tempfile
import os
import json
from deepzero.tokenizers.base import create_tokenizer


class TestDatasetPipeline(unittest.TestCase):
    def test_build_and_load(self):
        from deepzero.datasets.pipeline import build_dataset, load_dataset, dataset_statistics
        try:
            import requests
        except ImportError:
            self.skipTest("requests not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = build_dataset(
                cache_dir=tmpdir,
                language="python",
                min_length=10,
                max_length=100000,
            )
            self.assertIn("splits", meta)
            train = load_dataset(tmpdir, "train")
            self.assertIsInstance(train, list)
            stats = dataset_statistics(tmpdir)
            self.assertIn("train", stats)

    def test_language_filtering(self):
        from deepzero.datasets.pipeline import _infer_language
        self.assertEqual(_infer_language({"language": "python"}), "python")
        self.assertEqual(_infer_language({"extension": ".rs"}), "rust")
        self.assertEqual(_infer_language({"path": "main.go"}), "go")
        self.assertEqual(_infer_language({"path": "unknown.xyz"}), "unknown")

    def test_deduplication(self):
        from deepzero.datasets.pipeline import _deduplicate, _content_hash
        samples = [{"text": "hello"}, {"text": "hello"}, {"text": "world"}]
        deduped = _deduplicate(samples)
        self.assertEqual(len(deduped), 2)

    def test_normalize_whitespace(self):
        from deepzero.datasets.pipeline import _normalize_whitespace
        result = _normalize_whitespace("  hello\n\nworld\n  ")
        self.assertEqual(result, "  hello\n\nworld\n")

    def test_filter_samples(self):
        from deepzero.datasets.pipeline import _filter_samples
        samples = [
            {"text": "x = 1\nprint(x)\n", "language": "python"},
            {"text": "a", "language": "python"},
        ]
        filtered = _filter_samples(samples, min_length=5, language="python")
        self.assertEqual(len(filtered), 1)


class TestTokenizerInterface(unittest.TestCase):
    def test_create_tokenizer(self):
        for name in ("bpe", "byte_bpe", "character"):
            tok = create_tokenizer(name, vocab_size=100)
            self.assertIsNotNone(tok)
            self.assertEqual(tok.vocab_size, 100)

    def test_all_tokenizers_train_encode_decode(self):
        texts = ["def hello(): pass", "print('world')", "x = 1 + 2"]
        for name in ("bpe", "byte_bpe", "character"):
            tok = create_tokenizer(name, vocab_size=100)
            tok.train(texts)
            ids = tok.encode("hello world")
            self.assertGreater(len(ids), 0)
            decoded = tok.decode(ids)
            self.assertIsInstance(decoded, str)
            stats = tok.statistics()
            self.assertIn("vocab_size", stats)
            self.assertIn("name", stats)

    def test_save_load_roundtrip(self):
        texts = ["def foo(): return 42"]
        for name in ("bpe", "byte_bpe", "character"):
            tok = create_tokenizer(name, vocab_size=100)
            tok.train(texts)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                path = f.name
            try:
                tok.save(path)
                loaded = tok.load(path)
                self.assertEqual(loaded.vocab_size, tok.vocab_size)
                ids = loaded.encode("foo bar")
                self.assertGreater(len(ids), 0)
            finally:
                os.unlink(path)

    def test_unigram_tokenizer(self):
        try:
            tok = create_tokenizer("unigram", vocab_size=100)
            tok.train(["def hello(): pass"])
            ids = tok.encode("hello")
            self.assertGreater(len(ids), 0)
            decoded = tok.decode(ids)
            self.assertIsInstance(decoded, str)
        except Exception as e:
            self.fail(f"Unigram tokenizer failed: {e}")


class TestCodeQuality(unittest.TestCase):
    def test_check_syntax(self):
        from deepzero.evaluation.coding import check_syntax
        ok, err = check_syntax("x = 1")
        self.assertTrue(ok)
        ok, err = check_syntax("x = ")
        self.assertFalse(ok)

    def test_count_functions(self):
        from deepzero.evaluation.coding import count_functions
        code = "def a(): pass\ndef b(): pass\nclass C: pass"
        self.assertEqual(count_functions(code), 2)

    def test_repetition_rate(self):
        from deepzero.evaluation.coding import repetition_rate
        text = "a b a b a b"
        rate = repetition_rate(text, 2)
        self.assertGreater(rate, 0)

    def test_compression_ratio(self):
        from deepzero.evaluation.coding import compression_ratio
        from deepzero.tokenizers.base import create_tokenizer
        tok = create_tokenizer("character", vocab_size=100)
        texts = ["hello world"]
        tok.train(texts)
        ratio = compression_ratio(tok, texts)
        self.assertGreater(ratio, 0)

    def test_evaluate_code_quality(self):
        from deepzero.evaluation.coding import evaluate_code_quality
        q = evaluate_code_quality("print('hello')\n")
        self.assertIn("syntax_valid", q)
        self.assertIn("exec_success", q)
        self.assertIn("n_functions", q)


if __name__ == "__main__":
    unittest.main()
