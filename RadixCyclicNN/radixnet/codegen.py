"""Code generation with a sandbox, an Ollama teacher / judge and 2NRL rewards.

    problem -> Python program -> sandbox run -> judge -> 2NRL (punish / reward)

Two semi-supervised phases over the same list of problems:

* ``teacher`` — an Ollama model (default ``gemma4``) writes a solution, the
  sandbox runs it, the teacher fixes what failed, and the judge (the same LLM
  plus an objective PEP 8 / naming check) confirms it.  The network then
  learns the concatenated question + answer: every wrong attempt is 2NRL
  garbage (negative phase), the correct one the fine-tune pass (positive
  phase) — wrong answers *before* the correct answer.
* ``model`` — the network itself continues each problem prompt into code.
  Every attempt is run and judged the same way; a program that errors or is
  judged wrong is punished (negative phase) and the loop tries again, a
  correct one is rewarded (positive phase).  When the network never succeeds
  the teacher supplies the correct answer for the reward (``fallback_teacher``).

Correctness = the program runs in the sandbox (exit 0, no timeout), matches
the expected output / passes the appended tests when the problem has them,
is judged to accomplish the task, and (``strictness="strict"``) passes the
PEP 8 formatting and naming checks of both the objective checker and the LLM.

Security: the sandbox runs *generated* code with ``python -I`` in a scratch
directory with memory / CPU / file-size limits and, where ``unshare`` allows
it, without network access.  That is isolation from accidents, not from a
hostile program - run untrusted generations inside the Docker image.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import Any

from .ollama import OllamaClient, OllamaError, _loads_lenient

__all__ = [
    "DEFAULT_TEACHER_MODEL",
    "PHASES",
    "STRICTNESS",
    "Attempt",
    "CodeGenConfig",
    "CodeGenTrainer",
    "Problem",
    "RunResult",
    "Sandbox",
    "StyleReport",
    "Verdict",
    "check_style",
    "decide",
    "extract_code",
    "judge_with_ollama",
    "load_problems",
    "model_prefix",
    "parse_problem_file",
    "parse_problems",
    "solution_text",
    "teacher_fix",
    "teacher_generate",
]

DEFAULT_TEACHER_MODEL = os.environ.get("RADIXNET_CODEGEN_MODEL", "").strip() or "gemma4"
"""Ollama model that writes, fixes and judges solutions (``RADIXNET_CODEGEN_MODEL`` overrides)."""

PHASES = ("teacher", "model")
STRICTNESS = ("strict", "lenient")
MAX_OUTPUT_CHARS = 4000
MAX_LINE_LENGTH = 79


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Problem:
    """A task to solve: a prompt, optional test code (appended to the program) and optional exact stdout."""

    id: str
    prompt: str
    tests: str | None = None
    expected_output: str | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "prompt": self.prompt, "tests": self.tests, "expected_output": self.expected_output}

    @classmethod
    def from_any(cls, item: Any, index: int) -> "Problem":
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"problem {index}: empty prompt")
            return cls(f"p{index}", item.strip())
        if isinstance(item, dict):
            prompt = None
            for key in ("prompt", "problem", "task", "question"):
                if isinstance(item.get(key), str) and item[key].strip():
                    prompt = item[key].strip()
                    break
            if prompt is None:
                raise ValueError(f"problem {index}: missing 'prompt'")
            tests = item.get("tests")
            expected = item.get("expected_output", item.get("expected"))
            if tests is not None and not isinstance(tests, str):
                raise ValueError(f"problem {index}: 'tests' must be a string of Python code")
            if expected is not None and not isinstance(expected, str):
                raise ValueError(f"problem {index}: 'expected_output' must be a string")
            raw_id = item.get("id")
            pid = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else f"p{index}"
            return cls(pid, prompt, tests, expected)
        raise ValueError(f"problem {index}: expected a string or an object, got {type(item).__name__}")


def parse_problems(items: Iterable[Any]) -> list[Problem]:
    """Problems from strings / dicts; duplicate ids get a numeric suffix."""
    problems: list[Problem] = []
    seen: set[str] = set()
    for index, item in enumerate(items, 1):
        problem = Problem.from_any(item, index)
        pid, n = problem.id, 2
        while pid in seen:
            pid = f"{problem.id}-{n}"
            n += 1
        problem.id = pid
        seen.add(pid)
        problems.append(problem)
    if not problems:
        raise ValueError("no problems given")
    return problems


def parse_problem_file(text: str, ext: str = ".txt") -> list[Problem]:
    """Problem file content: ``.jsonl`` (one object per line), ``.json`` (list or ``{"problems": [...]}``),
    anything else one prompt per non-blank line (``#`` lines are comments)."""
    ext = (ext or "").lower()
    if ext == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif ext == ".json":
        data = json.loads(text)
        items = data.get("problems", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("a JSON problem file must hold a list or {\"problems\": [...]}")
    else:
        items = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return parse_problems(items)


def load_problems(path: str) -> list[Problem]:
    with open(path, encoding="utf-8") as fh:
        return parse_problem_file(fh.read(), os.path.splitext(path)[1])


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class RunResult:
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None
    timed_out: bool
    seconds: float
    expected_ok: bool | None = None
    network_isolated: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_BOOTSTRAP = """
import runpy, sys
try:
    import resource
except ImportError:
    resource = None
mem, cpu, script = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if resource is not None:
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    if cpu:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
""".strip()


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters)"


def _clean_traceback(stderr: str) -> str:
    """Drop the bootstrap's own frames from a traceback."""
    lines = stderr.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        if line.startswith('  File "<string>"') or "runpy.py" in line or "<frozen runpy>" in line:
            skip = True
            continue
        if skip and line.startswith("    "):
            continue
        skip = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()


class Sandbox:
    """Runs a generated program in a scratch directory with resource limits.

    ``python -I -B`` (isolated: no environment, no user site, no ``.pyc``),
    a fresh temporary working directory, an empty environment, memory / CPU
    / file-size limits set inside the child before the program starts, a
    wall-clock ``timeout``, and - when ``unshare`` can create a network
    namespace for an unprivileged user - no network.
    """

    def __init__(
        self,
        python: str | None = None,
        timeout: float = 10.0,
        memory_mb: int = 256,
        cpu_seconds: int | None = None,
        isolate_network: bool = True,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if memory_mb < 0:
            raise ValueError("memory_mb must be >= 0 (0 = unlimited)")
        self.python = python or sys.executable
        self.timeout = float(timeout)
        self.memory_mb = int(memory_mb)
        self.cpu_seconds = int(cpu_seconds) if cpu_seconds else int(timeout) + 1
        self._unshare = self._probe_unshare() if isolate_network else None

    @property
    def network_isolated(self) -> bool:
        return self._unshare is not None

    @staticmethod
    def _probe_unshare() -> list[str] | None:
        if os.name != "posix" or shutil.which("unshare") is None:
            return None
        try:
            probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return ["unshare", "-rn"] if probe.returncode == 0 else None

    def run(
        self, code: str, tests: str | None = None, expected_output: str | None = None, stdin: str = ""
    ) -> RunResult:
        """Run ``code`` (+ ``tests`` appended) and report what happened; never raises for a failing program."""
        source = code if code.endswith("\n") else code + "\n"
        if tests and tests.strip():
            source += "\n\n# --- tests ---\n" + tests.strip("\n") + "\n"
        with tempfile.TemporaryDirectory(prefix="radixnet-sandbox-") as workdir:
            script = os.path.join(workdir, "solution.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(source)
            command = list(self._unshare or []) + [
                self.python, "-I", "-B", "-c", _BOOTSTRAP,
                str(self.memory_mb << 20), str(self.cpu_seconds), script,
            ]
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": workdir,
                "TMPDIR": workdir,
                "LANG": "C.UTF-8",
                "PYTHONIOENCODING": "utf-8",
            }
            t0 = time.perf_counter()
            timed_out = False
            try:
                proc = subprocess.run(
                    command, cwd=workdir, env=env, input=stdin, capture_output=True, text=True,
                    errors="replace", timeout=self.timeout,
                )
                exit_code: int | None = proc.returncode
                stdout, stderr = proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = None
                stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                stderr = (stderr + f"\nTimeoutError: the program did not finish within {self.timeout:g} seconds").strip()
            seconds = time.perf_counter() - t0
        stderr = _clean_traceback(stderr)
        ok = exit_code == 0 and not timed_out
        error = None
        if not ok:
            tail = [line for line in stderr.splitlines() if line.strip()]
            error = tail[-1].strip() if tail else f"exit code {exit_code}"
            if exit_code == -9 and not timed_out:
                error = f"killed (memory or CPU limit exceeded): {error}"
        expected_ok = None
        if expected_output is not None:
            expected_ok = ok and stdout.strip() == expected_output.strip()
        return RunResult(
            ok=ok, exit_code=exit_code, stdout=_truncate(stdout), stderr=_truncate(stderr), error=error,
            timed_out=timed_out, seconds=seconds, expected_ok=expected_ok, network_isolated=self.network_isolated,
        )


# ---------------------------------------------------------------------------
# objective style check (PEP 8 formatting and naming, standard library only)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class StyleReport:
    ok: bool
    syntax_ok: bool
    pep8_ok: bool
    naming_ok: bool
    issues: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_SNAKE = re.compile(r"^_*[a-z][a-z0-9_]*$|^_+$")
_CAPWORDS = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")
_CONSTANT = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
_DUNDER = re.compile(r"^__[a-z0-9_]+__$")


def check_style(code: str) -> StyleReport:
    """PEP 8 formatting (indentation, line length, whitespace, blank lines) and naming, without external tools."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StyleReport(False, False, False, False, [f"syntax error: {exc.msg} (line {exc.lineno})"])
    formatting: list[str] = []
    lines = code.split("\n")
    if code and not code.endswith("\n"):
        formatting.append("W292 no newline at end of file")
    for number, line in enumerate(lines, 1):
        stripped = line.lstrip(" \t")
        if not stripped:
            if line:
                formatting.append(f"W293 line {number}: whitespace on a blank line")
            continue
        indent = line[: len(line) - len(stripped)]
        if "\t" in indent:
            formatting.append(f"W191 line {number}: indentation contains tabs")
        elif len(indent) % 4:
            formatting.append(f"E111 line {number}: indentation is not a multiple of four")
        if len(line) > MAX_LINE_LENGTH:
            formatting.append(f"E501 line {number}: line too long ({len(line)} > {MAX_LINE_LENGTH})")
        if line.rstrip() != line:
            formatting.append(f"W291 line {number}: trailing whitespace")
    # two blank lines before top-level definitions (E302)
    body = tree.body
    for previous, node in zip(body, body[1:]):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = getattr(previous, "end_lineno", previous.lineno)
            blank = [not lines[i - 1].strip() for i in range(end + 1, first) if 1 <= i - 1 < len(lines)]
            comment_only = all(lines[i - 1].lstrip().startswith("#") or not lines[i - 1].strip() for i in range(end + 1, first))
            if comment_only and sum(blank) < 2:
                formatting.append(f"E302 line {first}: expected 2 blank lines before a top-level definition")
    naming: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not (_SNAKE.match(node.name) or _DUNDER.match(node.name)):
                naming.append(f"N802 line {node.lineno}: function name '{node.name}' should be snake_case")
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [a for a in (args.vararg, args.kwarg) if a]:
                if not _SNAKE.match(arg.arg):
                    naming.append(f"N803 line {node.lineno}: argument '{arg.arg}' should be snake_case")
        elif isinstance(node, ast.ClassDef):
            if not _CAPWORDS.match(node.name):
                naming.append(f"N801 line {node.lineno}: class name '{node.name}' should use CapWords")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
            if not (_SNAKE.match(name) or _CONSTANT.match(name) or _CAPWORDS.match(name)):
                naming.append(f"N806 line {node.lineno}: variable '{name}' should be snake_case (or UPPER_CASE for a constant)")
    seen: set[str] = set()
    naming = [n for n in naming if not (n in seen or seen.add(n))]
    return StyleReport(
        ok=not formatting and not naming, syntax_ok=True, pep8_ok=not formatting, naming_ok=not naming,
        issues=formatting + naming,
    )


# ---------------------------------------------------------------------------
# texts: code blocks, training texts, model prefixes
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> str:
    """The Python program inside an LLM answer: the longest fenced block, else the whole text."""
    blocks = [b for b in _FENCE.findall(text or "") if b.strip()]
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n"
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("python"):
            stripped = stripped[6:]
        stripped = stripped.strip()
    return stripped + "\n" if stripped else ""


def solution_text(problem: Problem, code: str) -> str:
    """The text the network trains on: the question, a newline, the answer."""
    return f"{problem.prompt.strip()}\n{code.strip()}\n"


def model_prefix(template: str, problem: Problem) -> str:
    """The prefix the network continues into code (``{problem}`` is the prompt)."""
    if "{problem}" not in template:
        raise ValueError("model_prompt must contain {problem}")
    return template.replace("{problem}", problem.prompt.strip())


# ---------------------------------------------------------------------------
# the teacher (writes and fixes programs) and the judge
# ---------------------------------------------------------------------------

TEACHER_SYSTEM = (
    "You are an expert Python programmer writing small, self-contained programs. Answer with exactly one "
    "```python code block and nothing else: no explanations before or after it. The program must run with "
    "`python3 solution.py` on a plain Python 3 installation (standard library only), must not read input unless "
    "the task says so, must finish on its own, and must print its result when the task asks for output. Follow "
    "PEP 8: four-space indentation, lines of at most 79 characters, snake_case function and variable names, "
    "CapWords class names, UPPER_CASE constants, two blank lines before top-level definitions, a newline at the "
    "end of the file."
)

JUDGE_SYSTEM = (
    "You are a strict reviewer of small Python programs written for a stated task. Decide whether the program "
    "genuinely accomplishes the task (not merely runs), whether its formatting follows PEP 8, and whether its "
    "naming follows PEP 8 (snake_case functions and variables, CapWords classes, UPPER_CASE constants, "
    "descriptive names). Be adversarial: look for wrong results, missing requirements, unhandled cases and "
    "sloppy names. Reply with JSON only, exactly of the form {\"task_accomplished\": true or false, "
    "\"pep8\": true or false, \"naming\": true or false, \"score\": <0-10>, \"issues\": [\"...\"], "
    "\"critique\": \"one sentence\"}."
)


def _problem_block(problem: Problem) -> str:
    text = f"Task:\n{problem.prompt.strip()}\n"
    if problem.expected_output is not None:
        text += f"\nThe program's standard output must be exactly:\n{problem.expected_output.strip()}\n"
    if problem.tests:
        text += f"\nThis test code is appended to the program and must pass:\n{problem.tests.strip()}\n"
    return text


def teacher_generate(client: OllamaClient, problem: Problem, extra: str | None = None, model: str | None = None) -> str:
    """Ask the teacher for a first program."""
    user = _problem_block(problem)
    if extra and extra.strip():
        user += f"\nAdditional instructions:\n{extra.strip()}\n"
    user += "\nWrite the program now."
    return extract_code(client.generate(user, system=TEACHER_SYSTEM, model=model, options={"temperature": 0.3}))


def teacher_fix(
    client: OllamaClient, problem: Problem, attempt: "Attempt", extra: str | None = None, model: str | None = None
) -> str:
    """Ask the teacher to correct a program that failed or was judged wrong."""
    user = _problem_block(problem)
    user += f"\nThis program is not acceptable yet:\n```python\n{attempt.code.rstrip()}\n```\n\nWhat went wrong:\n{attempt.feedback()}\n"
    if extra and extra.strip():
        user += f"\nAdditional instructions:\n{extra.strip()}\n"
    user += "\nReturn the complete corrected program as one ```python block and nothing else."
    return extract_code(client.generate(user, system=TEACHER_SYSTEM, model=model, options={"temperature": 0.3}))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "y", "1", "pass", "passed", "ok"):
            return True
        if text in ("false", "no", "n", "0", "fail", "failed"):
            return False
    return default


def judge_with_ollama(
    client: OllamaClient, problem: Problem, code: str, run: RunResult, style: StyleReport, model: str | None = None
) -> dict:
    """The LLM's opinion: ``{"task", "pep8", "naming", "score", "issues", "critique"}``."""
    user = _problem_block(problem)
    user += (
        f"\nProgram:\n```python\n{code.rstrip()}\n```\n\nExecution: exit code {run.exit_code}, {run.seconds:.2f}s\n"
        f"stdout:\n{run.stdout[:1500] or '(empty)'}\nstderr:\n{run.stderr[:800] or '(empty)'}\n"
    )
    if run.expected_ok is not None:
        user += f"\nThe stdout {'matches' if run.expected_ok else 'does NOT match'} the expected output.\n"
    if problem.tests:
        user += f"\nThe appended tests {'passed' if run.ok else 'FAILED'}.\n"
    user += f"\nAutomated style check: {'no issues' if style.ok else '; '.join(style.issues[:8])}\n\nReturn the JSON now."
    raw = client.generate(user, system=JUDGE_SYSTEM, model=model, json_mode=True, options={"temperature": 0.1})
    data = _loads_lenient(raw)
    if not isinstance(data, dict):
        raise OllamaError("the judge did not answer with a JSON object")
    task = _as_bool(data.get("task_accomplished", data.get("correct", data.get("task"))), False)
    score: float | None
    try:
        score = max(0.0, min(10.0, float(data.get("score"))))
    except (TypeError, ValueError):
        score = None
    issues = data.get("issues")
    if isinstance(issues, str):
        issues = [issues]
    issues = [str(i).strip() for i in issues if str(i).strip()] if isinstance(issues, list) else []
    return {
        "task": task,
        "pep8": _as_bool(data.get("pep8", data.get("formatting")), True),
        "naming": _as_bool(data.get("naming"), True),
        "score": score,
        "issues": issues[:10],
        "critique": str(data.get("critique") or data.get("reason") or "").strip(),
    }


@dataclasses.dataclass(slots=True)
class Verdict:
    correct: bool
    runs: bool
    task: bool | None
    pep8: bool
    naming: bool
    score: float | None
    issues: list[str]
    critique: str
    judged_by: str  # sandbox | tests | ollama | none

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def decide(run: RunResult, style: StyleReport, llm: dict | None, strictness: str = "strict") -> Verdict:
    """Combine the sandbox result, the objective style report and the LLM's opinion into one verdict."""
    if strictness not in STRICTNESS:
        raise ValueError(f"strictness must be one of {', '.join(STRICTNESS)}")
    runs = run.ok
    issues: list[str] = []
    if not runs:
        issues.append(run.error or "the program did not run")
    if run.expected_ok is False:
        issues.append("stdout differs from the expected output")
    if not runs or run.expected_ok is False:
        task: bool | None = False
        judged_by = "sandbox"
    elif llm is not None:
        task = llm["task"]
        judged_by = "ollama"
    elif run.expected_ok is True:
        task = True
        judged_by = "tests"
    else:
        task = None  # nothing to judge with: running counts
        judged_by = "none"
    pep8 = style.pep8_ok and (llm["pep8"] if llm else True)
    naming = style.naming_ok and (llm["naming"] if llm else True)
    issues += style.issues[:6]
    if llm:
        issues += llm["issues"][:6]
    strict = strictness == "strict"
    correct = runs and run.expected_ok is not False and task is not False and (not strict or (pep8 and naming and style.ok))
    return Verdict(
        correct=correct, runs=runs, task=task, pep8=pep8, naming=naming,
        score=llm["score"] if llm else None, issues=issues, critique=(llm["critique"] if llm else ""), judged_by=judged_by,
    )


@dataclasses.dataclass(slots=True)
class Attempt:
    index: int
    source: str  # "ollama" | "model"
    code: str
    text: str
    run: RunResult
    style: StyleReport
    verdict: Verdict
    seconds: float

    def feedback(self) -> str:
        """Human-readable reasons a program was rejected (used by the teacher's fix prompt)."""
        parts: list[str] = []
        if not self.run.ok:
            parts.append(f"It did not run: {self.run.error}")
            if self.run.stderr:
                parts.append("stderr:\n" + self.run.stderr[-1200:])
        if self.run.expected_ok is False:
            parts.append("Its output differed from the expected output. Actual stdout:\n" + (self.run.stdout[-800:] or "(empty)"))
        if self.style.issues:
            parts.append("Style checker: " + "; ".join(self.style.issues[:8]))
        if self.verdict.judged_by == "ollama" and self.verdict.task is False:
            parts.append("Reviewer: the task is not accomplished. " + self.verdict.critique)
        elif self.verdict.critique:
            parts.append("Reviewer: " + self.verdict.critique)
        if self.verdict.issues:
            parts.append("Issues: " + "; ".join(dict.fromkeys(self.verdict.issues))[:1500])
        return "\n".join(parts) or "It was judged incorrect."

    def to_dict(self) -> dict:
        return {
            "index": self.index, "source": self.source, "code": self.code, "text_chars": len(self.text),
            "run": self.run.to_dict(), "style": self.style.to_dict(), "verdict": self.verdict.to_dict(),
            "correct": self.verdict.correct, "seconds": self.seconds,
        }


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CodeGenConfig:
    teacher_model: str = DEFAULT_TEACHER_MODEL
    judge_model: str | None = None
    phases: tuple[str, ...] = PHASES
    rounds: int = 1
    teacher_attempts: int = 3
    model_attempts: int = 4
    first_attempt_dijkstra: bool = True
    temperature: float = 1.0
    max_length: int = 800
    strictness: str = "strict"
    use_judge: bool = True
    fallback_teacher: bool = True
    twonrl_per: str = "problem"  # "problem" | "round"
    replay: bool = True
    replay_limit: int = 64
    teacher_prompt: str | None = None
    model_prompt: str = "{problem}\n"
    neg_epochs: int = 2
    pos_epochs: int = 3
    neg_lr: float = 0.5
    pos_lr: float = 0.1
    batch_size: int = 4
    checkpoint_every: int = 0  # problems

    def validate(self) -> None:
        if not self.phases or any(p not in PHASES for p in self.phases):
            raise ValueError(f"phases must be a non-empty subset of {', '.join(PHASES)}")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.teacher_attempts < 1 or self.model_attempts < 1:
            raise ValueError("teacher_attempts and model_attempts must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.strictness not in STRICTNESS:
            raise ValueError(f"strictness must be one of {', '.join(STRICTNESS)}")
        if self.twonrl_per not in ("problem", "round"):
            raise ValueError("twonrl_per must be 'problem' or 'round'")
        if self.replay_limit < 0 or self.neg_epochs < 0 or self.pos_epochs < 0 or self.checkpoint_every < 0:
            raise ValueError("replay_limit, epochs and checkpoint_every must be >= 0")
        if self.neg_lr < 0 or self.pos_lr < 0:
            raise ValueError("learning rates must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        model_prefix(self.model_prompt, Problem("check", "x"))

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["phases"] = list(self.phases)
        return data


ProgressFn = Callable[[dict], None]


class CodeGenTrainer:
    """Runs the teacher / model phases over problems and applies 2NRL rewards to the network.

    ``external`` is an optional zero-argument callable returning a context
    manager that is entered around slow external work (sandbox runs, LLM
    calls); the API uses it to release the model lock so the server stays
    responsive while a program runs or the teacher thinks.
    """

    def __init__(
        self,
        model: Any,
        client: OllamaClient,
        sandbox: Sandbox | None = None,
        config: CodeGenConfig | None = None,
        external: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        self.client = client
        self.sandbox = sandbox or Sandbox()
        self.config = config or CodeGenConfig()
        self.config.validate()
        self._external = external or nullcontext
        self.history: list[dict] = []
        self.replay_buffer: list[str] = []
        self.solved: dict[str, str] = {}
        self._stop = threading.Event()

    # -- evaluation ------------------------------------------------------------

    def _stopped(self) -> bool:
        return self._stop.is_set()

    def evaluate(self, problem: Problem, code: str, source: str, index: int) -> Attempt:
        """Sandbox + style check + (for programs that run) the LLM judge."""
        t0 = time.perf_counter()
        code = code if code.endswith("\n") else code + "\n"
        if not code.strip():
            run = RunResult(False, None, "", "", "empty program", False, 0.0, None, self.sandbox.network_isolated)
        else:
            with self._external():
                run = self.sandbox.run(code, tests=problem.tests, expected_output=problem.expected_output)
        style = check_style(code)
        llm = None
        if run.ok and self.config.use_judge:
            with self._external():
                llm = judge_with_ollama(
                    self.client, problem, code, run, style, model=self.config.judge_model or self.config.teacher_model,
                )
        verdict = decide(run, style, llm, self.config.strictness)
        return Attempt(
            index=index, source=source, code=code, text=solution_text(problem, code), run=run, style=style,
            verdict=verdict, seconds=time.perf_counter() - t0,
        )

    # -- the two ways of producing a program -----------------------------------

    def solve_with_teacher(
        self, problem: Problem, progress: ProgressFn | None, phase: str, round_no: int, start_index: int = 0
    ) -> list[Attempt]:
        attempts: list[Attempt] = []
        cfg = self.config
        with self._external():
            code = teacher_generate(self.client, problem, cfg.teacher_prompt, cfg.teacher_model)
        for i in range(cfg.teacher_attempts):
            attempt = self.evaluate(problem, code, "ollama", start_index + i)
            attempts.append(attempt)
            self._emit_attempt(progress, phase, round_no, problem, attempt)
            if attempt.verdict.correct or i == cfg.teacher_attempts - 1 or self._stopped():
                break
            with self._external():
                code = teacher_fix(self.client, problem, attempt, cfg.teacher_prompt, cfg.teacher_model)
        return attempts

    def generate_with_model(self, problem: Problem, index: int) -> str:
        """The network's ``index``-th program for ``problem`` (cheapest path first, then samples)."""
        cfg = self.config
        prefix = model_prefix(cfg.model_prompt, problem)
        mode = "dijkstra" if index == 0 and cfg.first_attempt_dijkstra else "sample"
        result = self.model.predict(
            prefix, length=1, mode=mode, temperature=cfg.temperature, max_length=cfg.max_length, to_end=(mode == "dijkstra"),
        )
        return result.text

    def solve_with_model(self, problem: Problem, progress: ProgressFn | None, phase: str, round_no: int) -> list[Attempt]:
        attempts: list[Attempt] = []
        cfg = self.config
        for i in range(cfg.model_attempts):
            code = self.generate_with_model(problem, i)
            attempt = self.evaluate(problem, code, "model", i)
            attempts.append(attempt)
            self._emit_attempt(progress, phase, round_no, problem, attempt)
            if attempt.verdict.correct or self._stopped():
                break
        if not any(a.verdict.correct for a in attempts) and cfg.fallback_teacher and not self._stopped():
            attempts += self.solve_with_teacher(problem, progress, phase, round_no, start_index=len(attempts))
        return attempts

    # -- learning ----------------------------------------------------------------

    def learn(self, bad: list[str], good: list[str]) -> dict:
        """2NRL on the wrong (``bad``) and correct (``good``) solution texts; replay keeps earlier successes."""
        cfg = self.config
        good_all = list(dict.fromkeys(good))
        if cfg.replay:
            extra = [t for t in self.replay_buffer if t not in good_all]
            good_all += extra[-cfg.replay_limit:] if cfg.replay_limit else []
        result: dict[str, Any] = {"bad": len(bad), "good": len(good_all), "action": None, "neg_loss": None, "pos_loss": None}
        if not bad and not good_all:
            return result
        if bad and good_all:
            outcome = self.model.two_nrl(
                bad, good_all, neg_epochs=cfg.neg_epochs, pos_epochs=cfg.pos_epochs, neg_lr=cfg.neg_lr, pos_lr=cfg.pos_lr,
                batch_size=cfg.batch_size, stop_event=self._stop,
            )
            result.update(action="2nrl", neg_loss=_last_loss(outcome["negative"]), pos_loss=_last_loss(outcome["positive"]))
        elif good_all:
            records = self.model.reward(
                good_all, epochs=cfg.pos_epochs, lr=cfg.pos_lr, batch_size=cfg.batch_size, stop_event=self._stop,
            )
            result.update(action="reward", pos_loss=_last_loss(records))
        else:  # punish only: negative phase, then (RadixNet) invert so the wrong programs become unlikely
            records = self.model.punish(bad, epochs=cfg.neg_epochs, lr=cfg.neg_lr, batch_size=cfg.batch_size, stop_event=self._stop)
            result.update(action="punish", neg_loss=_last_loss(records))
        for text in good:
            if text not in self.replay_buffer:
                self.replay_buffer.append(text)
        if cfg.replay_limit and len(self.replay_buffer) > cfg.replay_limit:
            del self.replay_buffer[: len(self.replay_buffer) - cfg.replay_limit]
        return result

    # -- driving -------------------------------------------------------------------

    def _emit(self, progress: ProgressFn | None, record: dict) -> None:
        self.history.append(record)
        if progress is not None:
            progress(record)

    def _emit_attempt(self, progress: ProgressFn | None, phase: str, round_no: int, problem: Problem, attempt: Attempt) -> None:
        verdict = attempt.verdict
        self._emit(progress, {
            "kind": "attempt", "phase": phase, "round": round_no, "problem": problem.id, "attempt": attempt.index + 1,
            "source": attempt.source, "runs": verdict.runs, "correct": verdict.correct, "score": verdict.score,
            "judged_by": verdict.judged_by, "error": attempt.run.error, "issues": verdict.issues[:4],
            "code": attempt.code[:2000], "stdout": attempt.run.stdout[:500], "seconds": attempt.seconds,
        })

    def run_problem(self, problem: Problem, phase: str, round_no: int, progress: ProgressFn | None) -> tuple[dict, list[str], list[str]]:
        t0 = time.perf_counter()
        if phase == "teacher":
            attempts = self.solve_with_teacher(problem, progress, phase, round_no)
        else:
            attempts = self.solve_with_model(problem, progress, phase, round_no)
        correct = next((a for a in attempts if a.verdict.correct), None)
        good = [a.text for a in attempts if a.verdict.correct]
        bad = [a.text for a in attempts if not a.verdict.correct]
        if correct is not None:
            self.solved[problem.id] = correct.text
        shown = correct if correct is not None else attempts[-1]
        record = {
            "kind": "problem", "phase": phase, "round": round_no, "problem": problem.id,
            "prompt": problem.prompt[:200], "attempts": len(attempts), "correct": correct is not None,
            "solved_by": correct.source if correct else None,
            "model_solved": any(a.source == "model" and a.verdict.correct for a in attempts),
            "bad": len(bad), "good": len(good), "score": shown.verdict.score, "code": shown.code[:2000],
            "issues": shown.verdict.issues[:4], "seconds": time.perf_counter() - t0,
        }
        return record, bad, good

    def run(
        self,
        problems: list[Problem],
        progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
        checkpoint_manager: Any = None,
    ) -> list[dict]:
        """All rounds and phases over ``problems``; returns the problem / round records (attempt records go to ``progress`` too)."""
        if not problems:
            raise ValueError("no problems to solve")
        cfg = self.config
        self._stop = stop_event if stop_event is not None else threading.Event()
        records: list[dict] = []
        step = 0
        for round_no in range(1, cfg.rounds + 1):
            for phase in cfg.phases:
                t0 = time.perf_counter()
                round_bad: list[str] = []
                round_good: list[str] = []
                solved = model_solved = done = 0
                for problem in problems:
                    if self._stopped():
                        break
                    t_problem = time.perf_counter()
                    record, bad, good = self.run_problem(problem, phase, round_no, progress)
                    if cfg.twonrl_per == "problem":
                        record.update(self.learn(bad, good))
                    else:
                        round_bad += bad
                        round_good += good
                    done += 1
                    solved += int(record["correct"])
                    model_solved += int(record["model_solved"])
                    record["seconds"] = time.perf_counter() - t_problem
                    self._emit(progress, record)
                    records.append(record)
                    step += 1
                    if checkpoint_manager is not None and cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                        checkpoint_manager.save(self.model, step, "codegen", {k: record[k] for k in ("phase", "round", "problem", "correct")})
                learned: dict = {}
                if cfg.twonrl_per == "round" and (round_bad or round_good) and not self._stopped():
                    learned = self.learn(round_bad, round_good)
                summary = {
                    "kind": "round", "phase": phase, "round": round_no, "problems": done, "solved": solved,
                    "model_solved": model_solved, "seconds": time.perf_counter() - t0, **learned,
                }
                self._emit(progress, summary)
                records.append(summary)
                if self._stopped():
                    return records
        return records


def _last_loss(records: list[dict]) -> float | None:
    return records[-1].get("loss") if records else None
