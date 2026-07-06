import unittest
import os
import tempfile
from deepzero.tokenizers.bpe import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer(vocab_size=50)

    def test_train_and_encode_decode(self):
        text = "hello world hello"
        self.tokenizer.train(text)
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertGreater(len(ids), 0)
        self.assertTrue(len(decoded) > 0)

    def test_save_load(self):
        text = "test data for tokenizer"
        self.tokenizer.train(text)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.tokenizer.save(path)
            loaded = BPETokenizer.from_pretrained(path)
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            ids = loaded.encode(text)
            self.assertGreater(len(ids), 0)
        finally:
            os.unlink(path)

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.PAD, "<PAD>")
        self.assertEqual(self.tokenizer.UNK, "<UNK>")
        self.assertEqual(self.tokenizer.EOS, "<EOS>")
        self.assertEqual(self.tokenizer.BOS, "<BOS>")


if __name__ == "__main__":
    unittest.main()
