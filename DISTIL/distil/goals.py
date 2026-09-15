"""Tasks become goals; goals become subgoals; the recursion has to stop somewhere.

The stopping rule is the design decision in this file, and it is not "when the
goal looks small enough":

> **Distillation stops when a goal is verifiable.**

A goal is atomic when a check can be written that decides whether it has been
met. Not when it feels primitive, not at a fixed depth -- when someone could
write the assert. That rule does three things at once. It terminates the
recursion on an objective criterion rather than a vibe. It guarantees every leaf
is gradeable, so the whole tree is gradeable bottom-up through `game.shapley`.
And it makes "I do not know how to decompose this" and "I do not know how to
check this" the same complaint, which they always were.

The consequence is that an under-specified task cannot be distilled to atoms --
it bottoms out in goals nobody can write a check for. That is the correct
outcome and it is *information*: it names the exact clause that needs
interrogating, which is where `challenge.py` is pointed.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


class Status:
    OPEN = "open"
    ACTIVE = "active"
    MET = "met"
    FAILED = "failed"
    BLOCKED = "blocked"        # refused, and the refusal has been mapped
    ABANDONED = "abandoned"    # not worth the cost, decided explicitly


@dataclass
class Verifier:
    """How you would know. `kind` is 'python' (a snippet that exits non-zero when
    unmet), 'predicate' (a callable), or 'manual' (a human decides -- which makes
    the goal checkable but not *self*-checkable, and the distinction is kept)."""
    kind: str
    body: str = ""
    fn: object = None

    @property
    def automatic(self) -> bool:
        """Can this be checked without a person?

        A 'predicate' verifier is a callable, and callables do not serialise.
        After a round trip the kind survives and `fn` is None, so this reported
        `automatic=True` for a check that no longer existed -- a goal that looked
        machine-verifiable and had nothing to run.
        """
        if self.kind == "predicate":
            return self.fn is not None
        return self.kind == "python"


@dataclass
class Goal:
    id: str
    text: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    status: str = Status.OPEN
    verifier: Verifier | None = None
    p_success: float = 0.5           # prior, revised from memory at selection time
    cost: float = 1.0                # relative effort; used per-bit in ranking
    notes: list[str] = field(default_factory=list)
    trace_id: str | None = None      # the memory trace this goal was written to

    @property
    def atomic(self) -> bool:
        return self.verifier is not None

    @property
    def leaf(self) -> bool:
        return not self.children

    def to_json(self) -> dict:
        return {"id": self.id, "text": self.text, "parent": self.parent,
                "children": self.children, "depth": self.depth, "status": self.status,
                "verifier": {"kind": self.verifier.kind, "body": self.verifier.body}
                if self.verifier else None,
                "p_success": self.p_success, "cost": self.cost, "notes": self.notes,
                "trace_id": self.trace_id}

    @classmethod
    def from_json(cls, d: dict) -> "Goal":
        v = d.get("verifier")
        return cls(id=d["id"], text=d["text"], parent=d.get("parent"),
                   children=d.get("children", []), depth=d.get("depth", 0),
                   status=d.get("status", Status.OPEN),
                   verifier=Verifier(v["kind"], v.get("body", "")) if v else None,
                   p_success=d.get("p_success", 0.5), cost=d.get("cost", 1.0),
                   notes=d.get("notes", []), trace_id=d.get("trace_id"))


class GoalTree:
    def __init__(self, task: str) -> None:
        self.task = task
        self.root = Goal(id="root", text=task, depth=0)
        self.goals: dict[str, Goal] = {"root": self.root}

    def add(self, text: str, parent: str = "root", verifier: Verifier | None = None,
            cost: float = 1.0, p_success: float = 0.5) -> Goal:
        p = self.goals[parent]
        g = Goal(id=uuid.uuid4().hex[:8], text=text, parent=parent, depth=p.depth + 1,
                 verifier=verifier, cost=cost, p_success=p_success)
        self.goals[g.id] = g
        p.children.append(g.id)
        return g

    def frontier(self) -> list[Goal]:
        """Leaves still open: what could be worked on right now."""
        return [g for g in self.goals.values()
                if g.leaf and g.status in (Status.OPEN, Status.ACTIVE) and g.id != "root"]

    def atoms(self) -> list[Goal]:
        return [g for g in self.goals.values() if g.atomic and g.id != "root"]

    def unverifiable(self) -> list[Goal]:
        """Leaves with no verifier: the parts of the task nobody can check.

        The most useful list the tree produces. Every entry is either a goal that
        needs further distillation or a clause of the task that was never
        properly specified, and telling those two apart is exactly what the
        interrogation layer is for.
        """
        return [g for g in self.frontier() if not g.atomic]

    def path(self, goal_id: str) -> list[Goal]:
        out, cur = [], self.goals.get(goal_id)
        while cur is not None:
            out.append(cur)
            cur = self.goals.get(cur.parent) if cur.parent else None
        return list(reversed(out))

    def mark(self, goal_id: str, status: str, note: str | None = None) -> Goal:
        g = self.goals[goal_id]
        g.status = status
        if note:
            g.notes.append(note)
        self._settle(g)
        return g

    def _settle(self, goal: Goal) -> None:
        """Propagate completion upward.

        A parent is met when every child is met; it fails only when a child fails
        *and* no sibling can still carry it. Children are treated as conjunctive
        because that is what decomposition means -- if any subgoal were optional
        it should not have been distilled out as a subgoal.
        """
        parent_id = goal.parent
        while parent_id:
            parent = self.goals[parent_id]
            kids = [self.goals[c] for c in parent.children]
            if kids and all(k.status == Status.MET for k in kids):
                parent.status = Status.MET
            elif any(k.status == Status.FAILED for k in kids):
                parent.status = Status.FAILED
            parent_id = parent.parent

    @property
    def solved(self) -> bool:
        return self.root.status == Status.MET

    def render(self) -> str:
        lines: list[str] = []

        def walk(gid: str) -> None:
            g = self.goals[gid]
            mark = {Status.MET: "[x]", Status.FAILED: "[!]", Status.ACTIVE: "[>]",
                    Status.BLOCKED: "[#]", Status.ABANDONED: "[-]"}.get(g.status, "[ ]")
            atom = "verifiable" if g.atomic else "needs a check"
            lines.append(f"{'  ' * g.depth}{mark} {g.text}" + ("" if g.depth == 0 else f"   ({atom})"))
            for c in g.children:
                walk(c)

        walk("root")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"task": self.task, "goals": {k: v.to_json() for k, v in self.goals.items()}}


# --------------------------------------------------------------------------- #
# guessing whether a goal is checkable at all
# --------------------------------------------------------------------------- #

#: Verbs whose outcome is observable. A goal starting with one of these can
#: usually be checked by running something.
_CHECKABLE = re.compile(
    r"\b(compute|parse|sort|count|convert|return|produce|write|generate|read|"
    r"validate|check|verify|filter|merge|reverse|find|extract|format|encode|"
    r"decode|round|sum|build|implement|raise|handle)\b", re.I)
#: Verbs whose outcome is a judgement. These need a human or a rubric, and
#: pretending otherwise is how a system marks "improve the design" as done.
_UNCHECKABLE = re.compile(
    r"\b(improve|optimi[sz]e|better|clean|nice|elegant|understand|consider|"
    r"explore|think|design|refactor|modernise|modernize|simplify|reasonable)\b", re.I)


def checkability(text: str) -> float:
    """A prior in [0,1] on whether this goal admits an automatic check.

    Lexical, therefore crude, and it is only a prior -- the real answer is
    whether a verifier could actually be written, which is settled by trying.
    Its job is to order the frontier so the distiller attacks the vague parts
    first, and for that it only has to be right on average.
    """
    if _UNCHECKABLE.search(text):
        return 0.15
    if _CHECKABLE.search(text):
        return 0.85
    has_object = len(text.split()) >= 3
    return 0.5 if has_object else 0.3
