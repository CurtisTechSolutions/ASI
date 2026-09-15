"""Grading, and the discipline of only grading what can be graded.

Two sources, and they are not interchangeable.

**Self-grading** happens where a verifier exists. For Python that verifier is
real and it is free: does it parse, does it survive the safety screen, does it
run, does it pass its own contract tests, does it do the same thing twice. Five
stages, each pass/fail, weighted into `[-1, 1]`. This is the strongest signal in
the system because nothing about it is an opinion -- the interpreter either
raised or it did not.

**User grading** is everything else.

The temptation is to close the gap by having a model grade the output. This
package does not, and the reason is worth stating: an LLM judging an LLM's prose
produces a number with the *shape* of evidence and the *content* of a second
opinion, and once it enters the store, recall cannot tell it apart from a passing
test. So `gradeable()` returns False for prose and the trace stays `ungraded`,
which is an honest state that recall already handles -- ungraded traces sit at
the credibility prior, neither promoted nor buried.

The stage weights say what the system believes about correctness: passing tests
is worth more than merely running (0.40 vs 0.20), and running the same way twice
is worth 10% on its own, because a tool that is right intermittently is worse
than one that is wrong consistently -- the second gets fixed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .memory import Source
from .sandbox import Execution, run_source, screen

#: stage -> weight. Sums to 1.0; the grade is 2*(weighted pass rate) - 1.
WEIGHTS = {
    "parse": 0.15,        # it is Python
    "screen": 0.10,       # it is not obviously destructive
    "run": 0.20,          # it executes to completion
    "test": 0.40,         # it does what it claims
    "determinism": 0.15,  # it does so repeatably
}


@dataclass
class Stage:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'} {self.name}: {self.detail}"


@dataclass
class Grade:
    score: float                      # [-1, 1]
    source: str
    stages: list[Stage] = field(default_factory=list)
    diagnostic: str = ""

    @property
    def passed(self) -> bool:
        return self.score > 0.0

    @property
    def clean(self) -> bool:
        return all(s.passed for s in self.stages) and bool(self.stages)

    def summary(self) -> str:
        marks = "".join("+" if s.passed else "-" for s in self.stages)
        return f"{self.score:+.2f} [{marks}] {self.diagnostic}"

    def report(self) -> str:
        return "\n".join([f"grade {self.score:+.3f} ({self.source})"] + [f"  {s}" for s in self.stages])


def _score(stages: list[Stage]) -> float:
    earned = sum(WEIGHTS.get(s.name, 0.0) for s in stages if s.passed)
    total = sum(WEIGHTS.get(s.name, 0.0) for s in stages) or 1.0
    return round(2.0 * (earned / total) - 1.0, 4)


def gradeable(payload: dict) -> bool:
    """Is there anything here a machine can check?

    Deliberately narrow. Code is gradeable. A claim with an attached check is
    gradeable. An essay is not, and saying so is the feature.
    """
    if payload.get("language") == "python" and payload.get("source"):
        return True
    return bool(payload.get("check"))


def grade_python(source: str, tests: str = "", timeout: float = 10.0,
                 check_determinism: bool = True) -> Grade:
    """Run the five stages. Later stages are skipped -- and recorded as failed --
    once an earlier one fails, because a module that will not parse cannot be
    said to have failed its tests; it never reached them, and scoring it as
    though it did would let a syntax error and a wrong answer grade the same."""
    stages: list[Stage] = []

    try:
        ast.parse(source)
        stages.append(Stage("parse", True, "parses"))
    except SyntaxError as exc:
        stages.append(Stage("parse", False, f"line {exc.lineno}: {exc.msg}"))
        return _abort(stages, "parse", f"SyntaxError: {exc.msg}")

    scr = screen(source)
    stages.append(Stage("screen", scr.ok, "clean" if scr.ok else "; ".join(scr.findings)))
    if not scr.ok:
        return _abort(stages, "screen", scr.findings[0])

    run = run_source(source, timeout=timeout)
    stages.append(Stage("run", run.ok, run.diagnostic))
    if not run.ok:
        return _abort(stages, "run", run.diagnostic)

    if tests.strip():
        harness = f"{source}\n\n# --- contract tests ---\n{tests}\n"
        tested = run_source(harness, timeout=timeout)
        stages.append(Stage("test", tested.ok, tested.diagnostic))
        if not tested.ok:
            return _abort(stages, "test", tested.diagnostic)
    else:
        # No tests is not a pass. A tool nobody specified a contract for is
        # unverified, and unverified must not score the same as verified.
        stages.append(Stage("test", False, "no contract tests supplied"))

    if check_determinism:
        again = run_source(source, timeout=timeout)
        same = again.ok and again.stdout == run.stdout
        stages.append(Stage("determinism", same,
                            "stable across two runs" if same else "output differs between runs"))

    g = Grade(_score(stages), Source.SELF, stages)
    g.diagnostic = "verified" if g.clean else next((s.detail for s in stages if not s.passed), "")
    return g


def _abort(stages: list[Stage], failed_at: str, diagnostic: str) -> Grade:
    order = list(WEIGHTS)
    for name in order[order.index(failed_at) + 1:]:
        stages.append(Stage(name, False, f"not reached ({failed_at} failed)"))
    return Grade(_score(stages), Source.SELF, stages, diagnostic)


def grade_check(check_source: str, timeout: float = 5.0) -> Grade:
    """Grade a claim by running the check attached to it.

    The check is a self-contained snippet that exits non-zero when the claim is
    false -- a bare `assert` is the usual form. This is what makes a *fact*
    gradeable: "sorted() is stable" is prose until someone writes the assert.
    """
    run = run_source(check_source, timeout=timeout)
    stages = [Stage("run", run.ok, run.diagnostic)]
    return Grade(1.0 if run.ok else -1.0, Source.SELF, stages, run.diagnostic)


def grade_user(score: float, comment: str = "") -> Grade:
    score = max(-1.0, min(1.0, float(score)))
    return Grade(score, Source.USER, [Stage("user", score > 0, comment or "user judgement")],
                 comment or "user grade")


def ungraded(reason: str = "nothing checkable") -> Grade:
    """The honest default. Scores 0, which recall reads as the prior -- neither
    promoted for being right nor buried for being wrong."""
    return Grade(0.0, Source.NONE, [], reason)
