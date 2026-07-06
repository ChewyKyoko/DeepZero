import unittest
import torch
from deepzero.models.transformer import GPT, ModelConfig
from deepzero.inference.generate import generate
from deepzero.inference.sampling import sample_top_k, sample_top_p


class TestInference(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64)
        self.model = GPT(self.config)

    def test_sample_top_k(self):
        logits = torch.randn(1, 100)
        probs = sample_top_k(logits, 10)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_sample_top_p(self):
        logits = torch.randn(1, 100)
        probs = sample_top_p(logits, 0.9)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_generate_via_model(self):
        class MockTokenizer:
            def __init__(self):
                self.EOS = 1
            def encode(self, text):
                return [5, 10, 15]
            def decode(self, ids):
                return "decoded"
        tok = MockTokenizer()
        result = self.model.generate(tok, "test", max_len=10, temperature=1.0, top_k=0)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
