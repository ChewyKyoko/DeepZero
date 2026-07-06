import unittest
import tempfile
import os
from deepzero.datasets.loader import TextDataset, load_jsonl


class TestDataset(unittest.TestCase):
    def test_text_dataset(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world " * 200)
            path = f.name
        try:
            ds = TextDataset(path, seq_len=16)
            self.assertGreater(len(ds), 0)
            x, y = ds[0]
            self.assertEqual(len(x), 16)
            self.assertEqual(len(y), 16)
        finally:
            os.unlink(path)

    def test_load_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n{"b": 2}\n')
            path = f.name
        try:
            data = load_jsonl(path)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["a"], 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
