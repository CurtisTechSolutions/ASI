"""Editing its own source, without the obvious ways that ends badly.

Self-modification is asked for and it is implemented. The design question is not
whether to allow it but what has to be true for an edit to land, and the answer
here is four things: it is written to a **shadow copy**, it passes the
**invariants**, it passes the **existing test suite**, and the previous state is
**snapshotted** so the edit is reversible. An edit failing any of those is
recorded with its diagnostic and discarded.

The invariant list is the part worth reading, because it guards the failure that
is specific to this kind of system rather than to software in general:

> **An optimiser that can edit its own grader will edit its own grader.**

Every stage of this package's self-improvement is scored by `grade.py`. Nothing
stops a well-meaning "improve the pass rate" instruction from rewriting the
weights so that everything passes, and the result would be a system reporting
monotone improvement while getting worse -- with no external signal to contradict
it, because the signal *is* the thing that was edited. So the grader's weights,
the test suite's size, and this invariant list itself are fixed points: an edit
touching them is refused regardless of how well it tests, because it would be
tested by the thing it changed.

That is not a limit on capability. The system can still write any tool, tune any
policy parameter, and rewrite any of its own reasoning, selection, retrieval,
exploration or synthesis code. It cannot mark its own homework.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .memory import Kind, Source
from .provider import Message
from .sandbox import run_file, screen

EDIT_SYSTEM = (
    "You rewrite exactly one Python module. Reply with one fenced python block "
    "containing the COMPLETE new file, standard library only, preserving every "
    "public name the rest of the package imports. Change the least that achieves "
    "the instruction."
)

#: Modules an edit may never touch: the grader, the invariant checker, and the
#: test suite. See the module docstring.
PROTECTED = {"grade.py", "selfedit.py", "sandbox.py"}


@dataclass
class Edit:
    target: str                    # module filename, e.g. "reason.py"
    before: str
    after: str
    rationale: str
    why_chain: str = ""            # the interrogation that motivated it

    @property
    def diff(self) -> str:
        return "".join(difflib.unified_diff(
            self.before.splitlines(keepends=True), self.after.splitlines(keepends=True),
            fromfile=f"a/{self.target}", tofile=f"b/{self.target}"))

    @property
    def churn(self) -> int:
        return sum(1 for l in self.diff.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))


@dataclass
class Report:
    ok: bool
    reason: str
    violations: list[str] = field(default_factory=list)
    tests_before: int = 0
    tests_after: int = 0
    stdout: str = ""

    def summary(self) -> str:
        return (f"{'ACCEPT' if self.ok else 'REJECT'} {self.reason}"
                + (f" | violations: {'; '.join(self.violations)}" if self.violations else ""))


@dataclass
class Generation:
    id: str
    parent: str | None
    target: str
    rationale: str
    churn: int
    accepted: bool
    reason: str
    at: float
    snapshot: str | None = None      # directory holding the pre-edit package

    def to_json(self) -> dict:
        return {"id": self.id, "parent": self.parent, "target": self.target,
                "rationale": self.rationale, "churn": self.churn, "accepted": self.accepted,
                "reason": self.reason, "at": self.at, "snapshot": self.snapshot}


def count_tests(tests_dir: Path) -> int:
    """Test functions defined under `tests/`. Counted by AST rather than by
    grepping `def test_`, because a rename or a decorator would fool the grep and
    this number is load-bearing: it is what stops an edit from deleting the
    checks that would have caught it."""
    total = 0
    for path in sorted(tests_dir.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        total += sum(1 for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith("test_"))
    return total


def digest(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest() if path.exists() else ""


class SelfEditor:
    def __init__(self, memory, provider, root: Path, home: Path) -> None:
        self.memory = memory
        self.provider = provider
        self.root = Path(root)                  # the DISTIL/ directory
        self.package = self.root / "distil"
        self.tests = self.root / "tests"
        self.home = Path(home)
        self.generations_dir = self.home / "generations"
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_path = self.home / "lineage.json"

    # -- lineage ------------------------------------------------------------

    def lineage(self) -> list[Generation]:
        if not self.lineage_path.exists():
            return []
        return [Generation(**g) for g in json.loads(self.lineage_path.read_text())]

    def _append(self, gen: Generation) -> None:
        log = [g.to_json() for g in self.lineage()] + [gen.to_json()]
        self.lineage_path.write_text(json.dumps(log, indent=2))

    # -- proposing ----------------------------------------------------------

    def propose(self, target: str, instruction: str, rationale: str = "") -> Edit | None:
        path = self.package / target
        if not path.exists():
            return None
        before = path.read_text()
        try:
            reply = self.provider.complete(
                [Message("system", EDIT_SYSTEM),
                 Message("user", f"EDIT:: module {target}\ninstruction: {instruction}\n\n"
                                 f"```python\n{before}\n```")],
                temperature=0.2, max_tokens=4000)
        except Exception:
            return None
        from .toolsmith import parse_reply
        after, _ = parse_reply(reply)
        if not after.strip() or after.strip() == before.strip():
            return None
        return Edit(target=target, before=before, after=after,
                    rationale=rationale or instruction)

    # -- the invariants -----------------------------------------------------

    def invariants(self, shadow: Path, edit: Edit) -> list[str]:
        """Everything that must still hold after the edit. Checked on the shadow
        tree, before any of it runs."""
        bad: list[str] = []

        if edit.target in PROTECTED:
            bad.append(f"{edit.target} is protected: an edit to the grader, the sandbox "
                       f"or this checker would be validated by the thing it changes")
            return bad                     # no point checking the rest

        try:
            ast.parse(edit.after)
        except SyntaxError as exc:
            bad.append(f"does not parse: line {exc.lineno}: {exc.msg}")
            return bad

        scr = screen(edit.after)
        if not scr.ok:
            bad.append("failed the safety screen: " + "; ".join(scr.findings))

        # The public surface the rest of the package imports must survive. A
        # rename that "cleans up" a name every other module calls is not an
        # improvement, it is a broken build with a good rationale.
        missing = _lost_names(edit.before, edit.after)
        if missing:
            bad.append(f"removes public name(s) other modules import: {', '.join(sorted(missing))}")

        # The grader must be untouched even indirectly.
        for name in PROTECTED:
            if digest(self.package / name) != digest(shadow / "distil" / name):
                bad.append(f"protected module changed on the shadow tree: {name}")

        before_n = count_tests(self.tests)
        after_n = count_tests(shadow / "tests")
        if after_n < before_n:
            bad.append(f"test count fell {before_n} -> {after_n}")
        return bad

    # -- evaluating ---------------------------------------------------------

    def evaluate(self, edit: Edit, timeout: float = 420.0) -> Report:
        """Apply to a throwaway copy, check invariants, run the real test suite.

        The suite that runs is the one on disk, not one the edit supplied. That
        distinction is the entire value of the step.

        The timeout is generous because a suite that times out looks exactly like
        a suite that failed, and the consequence is a correct edit rejected for a
        reason nobody can see. The suite runs in about a minute unsandboxed and
        the sandbox adds rlimits on top; 420s leaves room for a slow machine
        without letting a genuinely hung edit sit forever.
        """
        shadow = Path(self.home) / "shadow" / uuid.uuid4().hex[:8]
        if shadow.exists():
            shutil.rmtree(shadow)
        shadow.mkdir(parents=True)
        shutil.copytree(self.package, shadow / "distil",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(self.tests, shadow / "tests",
                        ignore=shutil.ignore_patterns("__pycache__"))
        try:
            (shadow / "distil" / edit.target).write_text(edit.after)
            before_n = count_tests(self.tests)
            violations = self.invariants(shadow, edit)
            if violations:
                return Report(False, "invariant violation", violations, before_n, before_n)
            runner = shadow / "_run_tests.py"
            runner.write_text(
                "import sys, pathlib\n"
                "sys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
                "from tests.test_distil import main\n"
                "raise SystemExit(0 if main() else 1)\n")
            run = run_file(runner, timeout=timeout, cwd=shadow)
            after_n = count_tests(shadow / "tests")
            if not run.ok:
                return Report(False, f"test suite failed: {run.diagnostic}", [], before_n,
                              after_n, stdout=(run.stdout + run.stderr)[-2000:])
            return Report(True, "invariants held and the suite passed", [], before_n, after_n,
                          stdout=run.stdout[-2000:])
        finally:
            shutil.rmtree(shadow.parent, ignore_errors=True)

    # -- promoting ----------------------------------------------------------

    def promote(self, edit: Edit, report: Report) -> Generation | None:
        """Snapshot, then write. In that order, always."""
        if not report.ok:
            self.memory.remember(
                Kind.FAILURE,
                f"self-edit to {edit.target} rejected: {report.reason}",
                meta={"target": edit.target, "violations": report.violations,
                      "churn": edit.churn, "rationale": edit.rationale},
                grade=-1.0, source=Source.SELF)
            gen = Generation(uuid.uuid4().hex[:8], self._head(), edit.target, edit.rationale,
                             edit.churn, False, report.reason, time.time())
            self._append(gen)
            return None
        gen_id = uuid.uuid4().hex[:8]
        snapshot = self.generations_dir / gen_id
        shutil.copytree(self.package, snapshot, ignore=shutil.ignore_patterns("__pycache__"))
        (self.package / edit.target).write_text(edit.after)
        gen = Generation(gen_id, self._head(), edit.target, edit.rationale, edit.churn,
                         True, report.reason, time.time(), snapshot=str(snapshot))
        self._append(gen)
        self.memory.remember(
            Kind.FACT,
            f"self-edit accepted: {edit.target} -- {edit.rationale} (+/-{edit.churn} lines)",
            meta={"generation": gen_id, "target": edit.target, "churn": edit.churn,
                  "diff": edit.diff[:4000]},
            grade=1.0, source=Source.SELF)
        return gen

    def _head(self) -> str | None:
        accepted = [g for g in self.lineage() if g.accepted]
        return accepted[-1].id if accepted else None

    def rollback(self, generation_id: str | None = None) -> str:
        """Restore the package as it was before a generation landed.

        Default is the most recent accepted one. Rollback is itself recorded as a
        generation, so undoing an undo is a normal operation rather than a
        special case -- the lineage is a log, not a stack.
        """
        accepted = [g for g in self.lineage() if g.accepted and g.snapshot]
        if not accepted:
            return "nothing to roll back"
        gen = next((g for g in accepted if g.id == generation_id), accepted[-1])
        snapshot = Path(gen.snapshot)
        if not snapshot.exists():
            return f"snapshot for {gen.id} is gone; cannot roll back"
        for src in snapshot.glob("*.py"):
            shutil.copy2(src, self.package / src.name)
        self._append(Generation(uuid.uuid4().hex[:8], self._head(), gen.target,
                                f"rollback of {gen.id}", gen.churn, True,
                                f"restored snapshot {gen.id}", time.time()))
        self.memory.remember(Kind.FACT, f"rolled back generation {gen.id} ({gen.target})",
                             meta={"generation": gen.id}, grade=0.0, source=Source.SELF)
        return f"rolled back {gen.id}: {gen.target} restored"

    # -- the whole cycle ----------------------------------------------------

    def attempt(self, target: str, instruction: str, rationale: str = "") -> dict:
        edit = self.propose(target, instruction, rationale)
        if edit is None:
            return {"ok": False, "reason": "no edit proposed", "target": target}
        report = self.evaluate(edit)
        gen = self.promote(edit, report)
        return {"ok": report.ok, "reason": report.reason, "target": target,
                "churn": edit.churn, "generation": gen.id if gen else None,
                "violations": report.violations, "diff": edit.diff}


def _lost_names(before: str, after: str) -> set[str]:
    """Public module-level names present before and absent after."""
    def public(src: str) -> set[str]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return set()
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and not t.id.startswith("_"):
                        names.add(t.id)
        return names
    return public(before) - public(after)
