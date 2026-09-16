"""The assembled system.

Nine modules with one object in front of them. The composition is the design, so
it is worth stating in one place what happens on a call to `solve`:

    understand the game      frame.Framer -- players, actions, payoff, referee
    ask if it is unclear     clarify.Clarifier -- questions, not guesses
    map it to what we can do frame.capabilities + frame.agenda
    check for precedent      casebook.adapt -- and what did NOT work
    question the task        challenge.interrogate + challenge.challenge
    recall what is known     memory.recall, ranked by grade
    distil to goals          reason.distill, stopping at verifiability
    estimate the world       reason.beliefs, moved by the interrogation
    choose a goal            game.choose over a payoff matrix against Nature
    reach for a tool         toolbox.find, or toolsmith.forge one
    verify                   grade.grade_python in the sandbox
    refuse to stop           challenge.Persistence, reframing on every no
    assign credit            game.shapley over the subgoals
    write it all down        memory.remember, every step, graded

`solve` is the one to read. Everything it learns goes back into the same store
it started from, which is the only structural claim this package makes: there is
no separate training phase, because the loop that answers and the loop that
learns are the same loop.
"""
from __future__ import annotations

import random
from pathlib import Path

from .casebook import Casebook
from .challenge import Persistence
from .clarify import Clarifier
from .compress import Compressor
from .embed import HashEmbedder, ProviderEmbedder
from .explore import Explorer
from .frame import Framer, agenda, capabilities
from .goals import Status
from .grade import grade_user
from .mcp import McpRegistry
from .memory import Kind, Memory, Source
from .policy import Policy
from .provider import Provider, auto
from .reason import Reasoner, Step
from .selfedit import SelfEditor
from .toolsmith import Toolbox, Toolsmith
from .workspace import Workspace


class Distil:
    def __init__(self, provider: Provider | None = None, home: Path | str | None = None,
                 root: Path | str | None = None, seed: int | None = None,
                 use_provider_embeddings: bool | None = None, store=None) -> None:
        self.workspace = Workspace(home)
        self.provider = provider or auto()
        self.policy = Policy.load(self.workspace.policy)
        # Provider embeddings when the provider has them and nothing says otherwise.
        # The lexical embedder is kept underneath either way: it is the fallback on
        # a failed call and it keeps working when the key expires mid-run.
        want = (self.provider.can_embed if use_provider_embeddings is None
                else use_provider_embeddings)
        base = HashEmbedder()
        self.embedder = ProviderEmbedder(self.provider, base) if want and self.provider.can_embed else base
        self.memory = Memory(self.embedder, self.policy, store=store)
        self.memory.load(self.workspace.memory)
        self.mcp = McpRegistry(self.memory, self.workspace)
        self.mcp.load_config(self.workspace.home / "mcp.json")
        self.toolbox = Toolbox(self.memory, self.workspace.workshop, self.mcp)
        self.toolsmith = Toolsmith(self.memory, self.provider, self.workspace.workshop,
                                   toolbox=self.toolbox)
        self.reasoner = Reasoner(self.memory, self.provider, self.policy, self.toolbox)
        self.framer = Framer(self.memory, self.provider)
        self.clarifier = Clarifier(self.memory, self.framer, self.toolbox)
        self.casebook = Casebook(self.memory)
        self.compressor = Compressor(self.memory, self.provider,
                                     archive=self.workspace.home / "archive.jsonl")
        self.rng = random.Random(seed)
        self.explorer = Explorer(self.memory, self.provider, self.policy, self.workspace,
                                 self.toolsmith, self.toolbox, self.rng)
        self.root = Path(root or Path(__file__).resolve().parent.parent)
        self.editor = SelfEditor(self.memory, self.provider, self.root, self.workspace.home)

    # -- the main loop ---------------------------------------------------------

    def understand(self, task: str) -> tuple:
        """Step zero: what game is this, and which of its moves can we make?

        Runs before any reasoning about the task, because the answer decides
        which solution concept is valid and what has to be built before play is
        even possible. Returns the frame and the agenda; the caller decides
        whether the confidence is sufficient to continue.
        """
        frame = self.framer.frame(task)
        caps = capabilities(frame, self.memory, self.toolbox)
        plan = agenda(frame, caps)
        self.framer.remember(frame)
        return frame, plan

    def solve(self, task: str, persist: bool = True, interrogate: bool = True,
              understand_first: bool = True, ask=None) -> dict:
        """Understand the game, then reason about it, then refuse to stop at the
        first refusal.

        If the game is not understood well enough for the first step to be
        actionable, this **stops and asks** rather than proceeding on a guess.
        With an `ask` callback it runs the clarification loop; without one it
        returns `needs_clarification` and the questions, for the caller to put to
        whoever knows. Distilling a task whose objective nobody can state
        produces a well-organised plan for the wrong problem, and that is the
        most expensive thing this system can do (DESIGN 7.2: BUILD pays -0.80 in
        a wrong frame).
        """
        if understand_first:
            clarified = self.clarifier.clarify(task, ask=ask)
            frame, plan = clarified.frame, clarified.plan
            if not clarified.actionable:
                self.save()
                return {"task": task, "solved": False, "session": None,
                        "needs_clarification": True, "questions": clarified.questions,
                        "clarification": clarified, "frame": frame, "agenda": plan,
                        "reason": clarified.reason}
        else:
            frame, plan = (None, None)
        objective = frame.objective if frame is not None else None
        lookup = objective if (objective and objective.strip() != task.strip()) else task
        precedent = self.casebook.adapt(lookup)
        session = self.reasoner.run(task, interrogate_first=interrogate,
                                    objective=objective)
        if frame is not None:
            session.chain.steps.insert(0, _thought(
                Step.FRAME,
                f"{frame.players}, {frame.payoff}, {frame.horizon}; referee: "
                f"{frame.referee or 'none'}; solution concept: {frame.solution} "
                f"(confidence {frame.confidence:.2f})",
                frame=frame.to_json()))
            session.chain.steps.insert(1, _thought(
                Step.AGENDA,
                f"{len(plan.gaps)} capability gap(s) of {len(plan.capabilities)}; "
                f"first item: {plan.items[0][:70] if plan.items else 'none'}",
                items=plan.items, gaps=[c.action for c in plan.gaps]))
        if precedent.get("precedent") or precedent.get("avoid"):
            best = precedent.get("precedent")
            session.chain.steps.insert(2 if frame is not None else 0, _thought(
                Step.PRECEDENT,
                (f"closest working case: {best.case.solution[:60]!r}" if best
                 else "no case on record worked")
                + f"; {len(precedent['avoid'])} known-bad approach(es) to avoid",
                note=precedent["note"],
                avoid=[p.case.solution for p in precedent["avoid"]]))
        session.frame = frame
        session.agenda = plan
        session.precedent = precedent
        if session.chosen is None:
            return {"task": task, "session": session, "solved": False,
                    "frame": frame, "agenda": plan, "precedent": precedent,
                    "reason": "nothing on the frontier to act on"}

        goal = session.chosen
        session.tree.mark(goal.id, Status.ACTIVE)
        attempts: list[dict] = []

        def attempt(goal_text: str, reframe: str | None):
            """One try at a goal: reuse a verified tool if one fits, else write one.

            Reuse does not call the tool. The goal text says what is wanted, not
            what to pass -- "compute the median of a column" carries no column --
            and an earlier version invoked the match with an empty argument list,
            which raised `TypeError` for every tool that takes a parameter and
            made reuse silently impossible. Inventing arguments to get a green
            result would have been the worse fix: it is exactly the fabrication
            the grading layer exists to prevent. A verified tool matching the
            goal means the capability is already present and checked; running it
            on real data is the caller's business, through `Toolbox.invoke`.
            """
            found = self.toolbox.find(goal_text, k=1)
            if found and found[0][2] >= 0.35:            # threshold on similarity
                spec, _, similarity = found[0]
                if _reusable(spec, self.memory) and self._runnable(spec):
                    # NOT record_use. That records a Source.SELF success, and
                    # this branch deliberately does not run the tool -- so it was
                    # manufacturing verifier evidence for something that never
                    # executed, and each reuse pushed a +1 that could outvote the
                    # real negative grades a failing tool had earned. The reuse
                    # is noted as a fact; only an actual run may grade.
                    self.memory.remember(
                        Kind.FACT,
                        f"{spec.name} already covers {goal_text!r} (similarity {similarity:.2f})",
                        meta={"tool": spec.name, "goal": goal_text, "ran": False},
                        links=[goal.trace_id] if goal.trace_id else None)
                    attempts.append({"goal": goal_text, "reframe": reframe,
                                     "via": f"existing tool {spec.name}", "ok": True,
                                     "similarity": round(similarity, 3)})
                    return True, (f"already covered by the verified tool {spec.name} "
                                  f"(similarity {similarity:.2f})")
            spec = self.toolsmith.forge(goal_text, goal.trace_id)
            if spec is None:
                attempts.append({"goal": goal_text, "reframe": reframe, "via": "none", "ok": False})
                return False, "no tool could be written for this goal"
            grade = self.toolsmith.validate(spec)
            kept = self.toolsmith.register(spec, goal.trace_id)
            attempts.append({"goal": goal_text, "reframe": reframe, "via": f"new tool {spec.name}",
                             "ok": bool(kept), "grade": grade.score})
            if kept:
                return True, f"wrote and verified {spec.name} ({grade.summary()})"
            return False, f"{spec.name} failed verification: {grade.diagnostic}"

        if persist:
            runner = Persistence(self.memory.embedder, max_attempts=6)
            outcome = runner.pursue(goal.text, attempt)
        else:
            ok, detail = attempt(goal.text, None)
            outcome = {"solved": ok, "reason": detail, "refusals": [] if ok else [detail],
                       "attempts": []}

        session.chain.add(Step.ACT,
                          f"{len(attempts)} attempt(s) on {goal.text[:60]!r}",
                          attempts=attempts)
        session.chain.add(Step.VERIFY,
                          ("met: " if outcome["solved"] else "not met: ") + outcome["reason"],
                          refusals=outcome.get("refusals", []))
        session.tree.mark(goal.id, Status.MET if outcome["solved"] else Status.BLOCKED,
                          outcome["reason"])

        # Everything learned goes back to the same store, including the boundary
        # mapped by failing -- which is the output a system that stops at the
        # first refusal can never produce.
        for refusal in outcome.get("refusals", []):
            self.memory.remember(Kind.FAILURE, f"refused while pursuing {goal.text!r}: {refusal}",
                                 links=[goal.trace_id] if goal.trace_id else None,
                                 meta={"task": task, "goal": goal.text},
                                 grade=-0.5, source=Source.SELF)
        # File it as a case: the problem, what was tried, and whether it worked.
        # Failures are filed with the same care as successes -- "that has been
        # tried and it does not work" is the more valuable of the two records.
        solution_text = (attempts[-1]["via"] if attempts else "nothing was attempted")
        self.casebook.record(
            problem=goal.text,
            solution=f"{solution_text} :: {outcome['reason']}",
            grade=1.0 if outcome["solved"] else -1.0,
            via=solution_text, evidence=outcome["reason"],
            cost=float(len(attempts) or 1),
            links=[goal.trace_id] if goal.trace_id else None,
            tags=[frame.players, frame.payoff] if frame is not None else [])
        shares = self.reasoner.credit(session.tree, {"source": Source.SELF}, session.chain)
        if session.trace_id:
            self.memory.grade(session.trace_id, 1.0 if outcome["solved"] else -0.5, Source.SELF)
        self.save()
        return {"task": task, "session": session, "solved": outcome["solved"],
                "reason": outcome["reason"], "goal": goal.text, "attempts": attempts,
                "refusals": outcome.get("refusals", []), "credit": shares,
                "boundary": outcome.get("boundary"), "frame": frame,
                "agenda": plan, "precedent": precedent}

    # -- the other entry points --------------------------------------------------

    def ask_user_grade(self, trace_id: str, score: float, comment: str = "") -> dict:
        g = grade_user(score, comment)
        trace = self.memory.grade(trace_id, g.score, Source.USER)
        self.save()
        return {"trace": trace.id, "kind": trace.kind, "grade": g.score,
                "credibility": round(trace.credibility(self.policy), 4),
                "text": trace.text[:120]}

    def explore(self, steps: int = 5, seed: str | None = None) -> list:
        out = self.explorer.loop(steps, seed)
        self.save()
        return out

    def upgrade(self, tasks: list[str] | None = None, trials: int = 8) -> dict:
        tasks = tasks or [t.text for t in self.memory.of_kind(Kind.QUERY)[-8:]] or ["a task"]
        result = self.explorer.upgrade(tasks, trials)
        # Adopt the winner. `Explorer.upgrade` rebinds its own policy and saves
        # it; `self.save()` below then wrote THIS object back over the file, so
        # every tuning run was undone by the save that was supposed to persist
        # it -- and the process was left holding three divergent Policy objects.
        self.policy = self.explorer.policy
        self.memory.policy = self.policy
        self.reasoner.policy = self.policy
        self.save()
        return result

    def self_edit(self, target: str, instruction: str) -> dict:
        result = self.editor.attempt(target, instruction)
        self.save()
        return result

    def _runnable(self, spec) -> bool:
        """Could this tool actually be executed right now?

        Grades say a tool worked once; they say nothing about whether its server
        is still configured. MCP traces outlive the registry that discovered
        them, so a goal was being reported solved -- and a +1 case filed -- by a
        tool nothing could have called.
        """
        if spec.transport != "mcp":
            return True
        server = spec.name.split(".")[0]
        return self.mcp is not None and server in self.mcp.servers

    def compress(self, dry_run: bool = False) -> dict:
        """Consolidate cold, redundant memory into digests. Reversible."""
        report = self.compressor.compress(dry_run=dry_run)
        if not dry_run:
            self.save()
        return report

    def save(self) -> None:
        self.memory.save(self.workspace.memory)
        self.policy.save(self.workspace.policy)

    def stats(self) -> dict:
        return {**self.memory.stats(), "cases": len(self.casebook.all()),
                "provider": self.provider.name,
                "embedder": getattr(self.embedder, "name", "?"),
                "tools": self.toolbox.names(), "home": str(self.workspace),
                "generations": len(self.editor.lineage())}


def _thought(kind: str, text: str, **payload):
    from .reason import Thought
    return Thought(kind, text, payload)


def _live_grade(spec, memory):
    """What the STORE currently thinks of this tool, not what it thought once.

    `spec.grade` for a local tool is the score frozen into its JSON at
    registration. Everything that happened since -- every `record_use`, every
    failure -- lands on the memory trace instead, so gating on the frozen value
    reused a tool the store had already graded negative and reported it as
    verified.
    """
    for trace in memory.of_kind(Kind.TOOL):
        if trace.meta.get("tool") == spec.name:
            return trace.mean_grade
    return spec.grade.score if spec.grade else None


def _proven(spec, memory) -> bool:
    """Has this tool actually carried a problem before?

    For a locally forged tool the evidence is `solved`, written when its contract
    tests passed. An MCP tool has no contract tests and `solved` starts empty, so
    gating on `solved` alone made every MCP tool permanently unreachable through
    `solve` -- the condition could not be satisfied by any sequence of events.

    So an MCP tool counts as proven once it has been used successfully at least
    once and carries a positive grade. Nothing is reused on faith, and nothing
    is unreachable forever.
    """
    if spec.solved:
        return True
    if spec.transport != "mcp" or not spec.trace_id:
        return False
    trace = memory.get(spec.trace_id)
    return bool(trace and trace.verified and (trace.mean_grade or 0) > 0)


def _reusable(spec, memory) -> bool:
    """Proven, and not since disgraced."""
    grade = _live_grade(spec, memory)
    if grade is not None and grade <= 0:
        return False
    return _proven(spec, memory)
