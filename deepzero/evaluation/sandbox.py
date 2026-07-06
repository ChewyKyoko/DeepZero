import ast
import sys
import io
import traceback
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr


class CodeSandbox:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    @staticmethod
    def check_syntax(code: str) -> tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def run(code: str, timeout: int = 5) -> dict:
        result = {"success": False, "output": "", "error": "", "exception": ""}
        ok, err = CodeSandbox.check_syntax(code)
        if not ok:
            result["error"] = f"SyntaxError: {err}"
            return result
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, {"__builtins__": __builtins__})
            result["success"] = True
            result["output"] = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                result["error"] = stderr_capture.getvalue()
        except Exception as e:
            result["success"] = False
            result["exception"] = f"{type(e).__name__}: {e}"
            result["output"] = stdout_capture.getvalue()
            result["error"] = traceback.format_exc()
        return result
