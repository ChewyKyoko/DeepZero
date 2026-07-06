import unittest
import torch
from deepzero.models.transformer import GPT, ModelConfig
from deepzero.models.layers import RMSNorm, CausalSelfAttention, MLP, TransformerBlock


class TestModel(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64)
        self.model = GPT(self.config)

    def test_forward_shape(self):
        x = torch.randint(0, 100, (2, 16))
        logits, loss = self.model(x, targets=x)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertIsNotNone(loss)

    def test_forward_no_targets(self):
        x = torch.randint(0, 100, (2, 16))
        logits, loss = self.model(x)
        self.assertIsNone(loss)

    def test_n_params(self):
        self.assertGreater(self.model.n_params, 0)

    def test_rmsnorm(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 16, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_attention(self):
        attn = CausalSelfAttention(32, 4)
        x = torch.randn(2, 16, 32)
        y = attn(x)
        self.assertEqual(y.shape, x.shape)

    def test_mlp(self):
        mlp = MLP(32, 128)
        x = torch.randn(2, 16, 32)
        y = mlp(x)
        self.assertEqual(y.shape, x.shape)

    def test_transformer_block(self):
        block = TransformerBlock(32, 4, 128)
        x = torch.randn(2, 16, 32)
        y = block(x)
        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
