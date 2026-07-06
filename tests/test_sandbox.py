import unittest
from deepzero.evaluation.sandbox import CodeSandbox


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.sandbox = CodeSandbox()

    def test_check_syntax_valid(self):
        ok, err = self.sandbox.check_syntax("x = 1")
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_check_syntax_invalid(self):
        ok, err = self.sandbox.check_syntax("x = ")
        self.assertFalse(ok)
        self.assertIsNotNone(err)

    def test_run_success(self):
        result = self.sandbox.run("print('hello')")
        self.assertTrue(result["success"])
        self.assertEqual(result["output"].strip(), "hello")

    def test_run_exception(self):
        result = self.sandbox.run("raise ValueError('test')")
        self.assertFalse(result["success"])
        self.assertIn("ValueError", result["exception"])

    def test_run_syntax_error(self):
        result = self.sandbox.run("x = ")
        self.assertFalse(result["success"])
        self.assertIn("SyntaxError", result["error"])


if __name__ == "__main__":
    unittest.main()
