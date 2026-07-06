import ast
import io
import signal
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

from deepzero.evaluation.coding import check_syntax


class TimeoutError(RuntimeError):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


class SandboxExecutor:
    def __init__(self, timeout: int = 5, restricted: bool = True):
        self.timeout = timeout
        self.restricted = restricted

    def _restricted_builtins(self) -> dict:
        allowed = {
            "abs": abs, "all": all, "any": any, "ascii": ascii,
            "bin": bin, "bool": bool, "bytearray": bytearray, "bytes": bytes,
            "callable": callable, "chr": chr, "complex": complex,
            "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
            "id": id, "input": input, "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len, "list": list,
            "map": map, "max": max, "min": min, "next": next, "object": object,
            "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round,
            "set": set, "slice": slice, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "type": type, "zip": zip,
            "True": True, "False": False, "None": None,
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "StopIteration": StopIteration,
            "RuntimeError": RuntimeError, "AssertionError": AssertionError,
            "__import__": __import__,
        }
        return allowed

    def run_code(self, code: str, stdin: str = "") -> dict:
        result = {"success": False, "output": "", "error": "", "exception": "", "timed_out": False}
        syntax_ok, syntax_err = check_syntax(code)
        if not syntax_ok:
            result["error"] = f"SyntaxError: {syntax_err}"
            return result

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        stdin_buffer = io.StringIO(stdin)

        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.timeout)

        try:
            try:
                globals_dict = {"__builtins__": self._restricted_builtins() if self.restricted else __builtins__}
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    sys.stdin = stdin_buffer
                    compiled = compile(code, "<sandbox>", "exec")
                    exec(compiled, globals_dict)
                result["success"] = True
                result["output"] = stdout_capture.getvalue()
                if stderr_capture.getvalue():
                    result["error"] = stderr_capture.getvalue()
            except TimeoutError:
                result["timed_out"] = True
                result["error"] = "Execution timed out"
                result["exception"] = "TimeoutError"
            except Exception as e:
                result["exception"] = f"{type(e).__name__}: {e}"
                result["output"] = stdout_capture.getvalue()
                result["error"] = traceback.format_exc()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            sys.stdin = sys.__stdin__

        return result

    def run_test(self, test_code: str) -> dict:
        return self.run_code(test_code)

    def generate_and_test(self, generated_code: str, test_code: str) -> dict:
        combined = f"{generated_code}\n\n{test_code}"
        return self.run_code(combined, stdin="")

    def check_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        return check_syntax(code)
