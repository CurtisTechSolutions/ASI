"""Writing tools, proving they work, and remembering what they solved.

A tool enters the toolbox only by passing `grade.grade_python` -- parse, screen,
run, contract tests, determinism. There is no "probably fine" path, because the
whole value of a tool over re-deriving the answer is that the tool has *already
been checked*, and an unchecked tool is strictly worse than no tool: it is a
confident wrong answer with a name and a docstring.

Contract tests are required and are written at the same time as the tool, by the
same author. That is not the ideal -- an independent test author is better -- but
it is the honest trade, and the failure mode it leaves (a test that asserts the
bug) is visible in the stored source rather than hidden in a score.

Tools are registered into the embedding layer as `Kind.TOOL`, linked to the goal
that motivated them, and the text that gets embedded is **purpose plus signature
plus every problem the tool has solved**. That last part is what makes retrieval
work: a tool is found by the shape of the problem, not by its name. A tool called
`normalise_rows` is unreachable by anyone who did not already know it exists;
the same tool carrying "stripped whitespace from a ragged CSV" in its embedded
text is found by whoever has that problem next.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .grade import Grade, grade_python
from .memory import Kind, Source
from .provider import Message
from .sandbox import run_source

TOOL_SYSTEM = (
    "You write one small, self-contained Python tool. Standard library only. "
    "Reply with exactly two fenced blocks: first the tool, then contract tests "
    "that assert its behaviour and print nothing on success. Define one primary "
    "function. No I/O outside the function. No network."
)


@dataclass
class ToolSpec:
    name: str
    purpose: str
    source: str
    tests: str = ""
    signature: str = ""
    grade: Grade | None = None
    solved: list[str] = field(default_factory=list)   # problems this tool has handled
    trace_id: str | None = None
    built_by: str = "provider"      # "template" or "provider" -- see Idea.claim
    transport: str = "python"       # "python" (runs in the sandbox) or "mcp" (another process)
    deps: list[str] = field(default_factory=list)   # other registered tools this one calls

    def embed_text(self) -> str:
        """What the embedding layer actually indexes. Deliberately not the source:
        code embeds by its syntax, and nobody searches by syntax."""
        parts = [f"tool {self.name}: {self.purpose}", self.signature]
        parts += [f"solved: {p}" for p in self.solved]
        return "\n".join(p for p in parts if p)

    def to_json(self) -> dict:
        # Every field `from_json` reads must be written here. An earlier version
        # dropped `built_by`, `transport` and `deps`, and nothing caught it: the
        # tests all inspected freshly forged specs, where the in-memory value was
        # still correct. The bug only appeared once a composite tool was loaded
        # back from disk and its dependencies had silently become `[]`.
        return {"name": self.name, "purpose": self.purpose, "source": self.source,
                "tests": self.tests, "signature": self.signature, "solved": self.solved,
                "trace_id": self.trace_id, "built_by": self.built_by,
                "transport": self.transport, "deps": self.deps,
                "grade": self.grade.score if self.grade else None}

    @classmethod
    def from_json(cls, d: dict) -> "ToolSpec":
        # `grade` is written by to_json and used to be dropped here, so a tool
        # loaded from disk looked ungraded and every caller gating on
        # `spec.grade` fell through to its "unknown" branch. Only the score is
        # persisted, so it returns as a score-only Grade -- exactly what the
        # stored data supports, and no more.
        spec = cls(name=d["name"], purpose=d["purpose"], source=d["source"],
                   tests=d.get("tests", ""), signature=d.get("signature", ""),
                   solved=d.get("solved", []), trace_id=d.get("trace_id"),
                   built_by=d.get("built_by", "provider"),
                   transport=d.get("transport", "python"), deps=d.get("deps", []))
        score = d.get("grade")
        if score is not None:
            from .grade import Stage
            spec.grade = Grade(float(score), Source.SELF,
                               [Stage("recorded", float(score) > 0, "score restored from disk")],
                               "restored from disk")
        return spec


_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_DEF = re.compile(r"^def\s+([a-zA-Z_]\w*)\s*\((.*?)\)", re.M)


def parse_reply(text: str) -> tuple[str, str]:
    """Pull the tool and its tests out of a model reply.

    Two fenced blocks is the contract; one block means the tests were forgotten,
    and that is returned as an empty test string rather than guessed at, so the
    grader can dock it for exactly what is missing.
    """
    blocks = _FENCE.findall(text or "")
    if len(blocks) >= 2:
        return blocks[0].strip(), blocks[1].strip()
    if len(blocks) == 1:
        return blocks[0].strip(), ""
    return (text or "").strip(), ""


def signature_of(source: str) -> tuple[str, str]:
    m = _DEF.search(source)
    if not m:
        return "", ""
    return m.group(1), f"{m.group(1)}({m.group(2)})"


# --------------------------------------------------------------------------- #
# the offline synthesiser
# --------------------------------------------------------------------------- #

#: Goal shape -> (function name, source, tests). A rule-based synthesiser over a
#: handful of shapes, so the loop closes with no model attached. It generalises
#: to nothing outside this table and returns None rather than guessing -- an
#: unrecognised goal should reach a real provider, not a plausible-looking stub
#: that was never going to run.
TEMPLATES: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"\bmedian\b", re.I), "median",
     '''def median(xs):
    """Middle value of a sequence; mean of the two middle values when even."""
    s = sorted(xs)
    if not s:
        raise ValueError("median of an empty sequence")
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
''',
     '''assert median([3, 1, 2]) == 2
assert median([1, 2, 3, 4]) == 2.5
assert median([7]) == 7
try:
    median([]); raise SystemExit("should have raised")
except ValueError:
    pass
'''),
    (re.compile(r"\b(dedupe|deduplicat|unique|distinct)\w*\b", re.I), "dedupe",
     '''def dedupe(xs):
    """Unique items, first occurrence order preserved."""
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''',
     '''assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]
assert dedupe([]) == []
assert dedupe(["a", "a"]) == ["a"]
'''),
    (re.compile(r"\bflatten\b", re.I), "flatten",
     '''def flatten(xs):
    """One level of nesting removed; non-iterables pass through."""
    out = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out.extend(x)
        else:
            out.append(x)
    return out
''',
     '''assert flatten([[1, 2], [3], 4]) == [1, 2, 3, 4]
assert flatten([]) == []
assert flatten([[[1]]]) == [[1]]
'''),
    (re.compile(r"\b(csv|comma.separated)\b", re.I), "parse_csv",
     '''def parse_csv(text, sep=","):
    """Split text into rows of stripped fields. Blank lines are dropped."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append([cell.strip() for cell in line.split(sep)])
    return rows
''',
     '''assert parse_csv("a, b\\nc,d") == [["a", "b"], ["c", "d"]]
assert parse_csv("") == []
assert parse_csv("x\\n\\ny") == [["x"], ["y"]]
'''),
    (re.compile(r"\b(word|token)\s*count|\bcount\b.*\bword", re.I), "word_count",
     '''def word_count(text):
    """Case-folded word frequencies, in descending count order."""
    import re as _re
    from collections import Counter
    words = _re.findall(r"[a-z0-9']+", (text or "").lower())
    return dict(Counter(words).most_common())
''',
     '''assert word_count("a b A") == {"a": 2, "b": 1}
assert word_count("") == {}
assert list(word_count("x y y").keys())[0] == "y"
'''),
    (re.compile(r"\b(reverse|invert)\b", re.I), "reverse_words",
     '''def reverse_words(text):
    """Word order reversed; internal whitespace collapsed to single spaces."""
    return " ".join(reversed((text or "").split()))
''',
     '''assert reverse_words("a b c") == "c b a"
assert reverse_words("") == ""
assert reverse_words("  x   y ") == "y x"
'''),
]


def synthesise(goal: str) -> tuple[str, str, str] | None:
    for pattern, name, source, tests in TEMPLATES:
        if pattern.search(goal):
            return name, source, tests
    return None


# --------------------------------------------------------------------------- #

class Toolsmith:
    def __init__(self, memory, provider, workshop: Path, toolbox=None) -> None:
        self.memory = memory
        self.provider = provider
        self.workshop = Path(workshop)
        self.workshop.mkdir(parents=True, exist_ok=True)
        # Held so `register` can bundle dependencies on its fallback validate.
        # Without it a composite registered without an explicit validate call was
        # graded against its own source alone, failed with NameError, was
        # rejected, and a false FAILURE about a working tool went into memory.
        self.toolbox = toolbox

    def forge(self, goal: str, goal_trace_id: str | None = None) -> ToolSpec | None:
        """Write a tool for a goal. The rule synthesiser first -- it is free,
        instant and correct where it applies -- then the provider."""
        built = synthesise(goal)
        if built:
            name, source, tests = built
            origin = "template"
        else:
            origin = "provider"
            try:
                reply = self.provider.complete(
                    [Message("system", TOOL_SYSTEM), Message("user", f"TOOL:: {goal}")],
                    temperature=0.2, max_tokens=900)
            except Exception:
                return None
            source, tests = parse_reply(reply)
            if not source.strip():
                return None
            name, _ = signature_of(source)
            if not name:
                return None
        fname, sig = signature_of(source)
        return ToolSpec(name=fname or name, purpose=goal, source=source, tests=tests,
                        signature=sig, built_by=origin)

    def validate(self, spec: ToolSpec, toolbox=None) -> Grade:
        """Grade the tool as it will actually run -- dependencies included.

        Validating `spec.source` alone would pass a composite tool whose
        dependency is missing at call time, which is the one failure the grading
        layer exists to catch before it reaches the toolbox.
        """
        toolbox = toolbox or self.toolbox
        try:
            source = toolbox.bundle(spec) if (toolbox and spec.deps) else spec.source
        except ValueError as exc:
            from .grade import Stage
            spec.grade = Grade(-1.0, Source.SELF, [Stage("bundle", False, str(exc))], str(exc))
            return spec.grade
        spec.grade = grade_python(source, spec.tests)
        return spec.grade

    def register(self, spec: ToolSpec, goal_trace_id: str | None = None,
                 threshold: float = 0.5) -> bool:
        """Keep it only if it earned its place; remember the failure either way.

        A rejected tool is written to memory as `Kind.FAILURE` with its
        diagnostic. That is the more valuable record of the two: it stops the
        system re-deriving the same broken approach, and `explore.py` mutates
        those failures into the next round of experiments.
        """
        grade = spec.grade or self.validate(spec, self.toolbox)
        incumbent = self.toolbox.load(spec.name) if self.toolbox else None
        if (incumbent is not None and incumbent.transport == "python"
                and incumbent.source.strip() != spec.source.strip()
                and incumbent.grade is not None and incumbent.grade.score >= grade.score):
            # A tool is keyed by its function name, so forging "median" writes
            # over whatever `median` was already there -- including a seed tool
            # with a far stronger contract test suite, and (via the identity
            # merge) its single trace and grade history with it. Every composite
            # depending on it silently inherited the weaker implementation.
            # A replacement has to be strictly better to land.
            self.memory.remember(
                Kind.FAILURE,
                f"kept the existing {spec.name!r} ({incumbent.grade.score:+.2f}) over a "
                f"replacement grading {grade.score:+.2f}",
                meta={"tool": spec.name, "kept": True, "grade": grade.score})
            return False
        if grade.score < threshold:
            self.memory.remember(
                Kind.FAILURE,
                f"tool {spec.name!r} for {spec.purpose!r} rejected: {grade.diagnostic}",
                meta={"tool": spec.name, "grade": grade.score, "stages": [s.name for s in grade.stages if not s.passed]},
                links=[goal_trace_id] if goal_trace_id else None,
                grade=grade.score, source=Source.SELF)
            return False
        spec.solved.append(spec.purpose)
        path = self.workshop / f"{spec.name}.py"
        path.write_text(spec.source if spec.source.endswith("\n") else spec.source + "\n")
        (self.workshop / f"{spec.name}.tests.py").write_text(spec.tests + "\n")
        trace = self.memory.remember(
            Kind.TOOL, spec.embed_text(),
            meta={"tool": spec.name, "path": str(path), "signature": spec.signature,
                  "grade": grade.score, "purpose": spec.purpose,
                  "transport": spec.transport, "solved": list(spec.solved)},
            links=[goal_trace_id] if goal_trace_id else None,
            grade=grade.score, source=Source.SELF,
            # One trace per tool name, for the life of the store. Re-forging a
            # tool used to create a SECOND Kind.TOOL trace, and every consumer
            # that looks a tool up by name takes the first match -- so the old,
            # negatively graded trace kept answering for the new tool and it
            # could never be reused again.
            identity=f"tool:{spec.name}")
        spec.trace_id = trace.id
        (self.workshop / f"{spec.name}.json").write_text(json.dumps(spec.to_json(), indent=2))
        return True


def _schema_problems(spec, kwargs: dict, memory) -> list[str]:
    """Check MCP arguments against the schema recorded at discovery.

    Only what is unambiguous from a JSON Schema and cheap to check locally:
    required keys, and keys the server never declared. Deeper validation belongs
    to the server, which owns the contract.
    """
    schema = next((t.meta.get("schema") for t in memory.of_kind(Kind.TOOL)
                   if t.meta.get("tool") == spec.name), None)
    if not isinstance(schema, dict):
        return []
    problems = [f"{spec.name}: required argument {k!r} is missing"
                for k in (schema.get("required") or []) if k not in kwargs]
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        problems += [f"{spec.name}: {k!r} is not an argument this tool declares"
                     for k in kwargs if k not in props]
    return problems


class Toolbox:
    """Every capability, whatever process it runs in.

    `find` is a memory query restricted to `Kind.TOOL`, so it inherits graded
    recall: a tool that has failed since being registered sinks, without anyone
    having to remember to delete it. Locally forged Python tools and tools
    exposed by MCP servers are both `Kind.TOOL` traces in that one query, which
    is the point -- at the moment of recall "what can act on this?" does not care
    which process the answer lives in, and a second registry would just be one
    more place every caller has to remember to look.

    Dispatch happens at `invoke`, on `ToolSpec.transport`.
    """

    def __init__(self, memory, workshop: Path, mcp=None) -> None:
        self.memory = memory
        self.workshop = Path(workshop)
        self.workshop.mkdir(parents=True, exist_ok=True)
        self.mcp = mcp                    # an McpRegistry, or None

    def names(self) -> list[str]:
        """Local tools on disk plus every MCP tool in memory, as one sorted list."""
        local = {p.stem for p in self.workshop.glob("*.json")}
        remote = {t.meta.get("tool") for t in self.memory.of_kind(Kind.TOOL)
                  if t.meta.get("transport") == "mcp" and t.meta.get("tool")}
        return sorted(local | remote)

    def load(self, name: str) -> ToolSpec | None:
        """A spec for either transport.

        MCP tools have no file on disk -- the source lives in someone else's
        process -- so their spec is reconstructed from the memory trace that
        registered them. Returning one `ToolSpec` for both keeps every caller
        downstream (`find`, `invoke`, `record_use`, `agent.solve`) on a single
        type instead of branching on transport in five places.
        """
        # Memory first for MCP tools. A server's catalogue is whatever the
        # server says today; a JSON file from an earlier discovery is a snapshot
        # that can disagree with it. Checking disk first also let a local tool
        # whose name contains a dot shadow a real MCP tool of that qualified name.
        #
        # Only qualified names take that path. MCP tools are always
        # "server.tool", so a bare name cannot be one, and scanning every trace
        # in the store on every lookup made the common case -- a local tool one
        # file read away -- cost a full scan.
        if "." in name:
            for trace in self.memory.of_kind(Kind.TOOL):
                if trace.meta.get("tool") == name and trace.meta.get("transport") == "mcp":
                    return ToolSpec(
                        name=name, purpose=trace.meta.get("purpose", ""), source="",
                        signature=trace.meta.get("signature", ""), transport="mcp",
                        built_by="mcp", solved=list(trace.meta.get("solved", [])),
                        trace_id=trace.id)
        path = self.workshop / f"{name}.json"
        if path.exists():
            return ToolSpec.from_json(json.loads(path.read_text()))
        return None

    def all(self) -> list[ToolSpec]:
        return [s for s in (self.load(n) for n in self.names()) if s]

    def find(self, problem: str, k: int = 3) -> list[tuple[ToolSpec, float, float]]:
        """Tools for a problem, as (spec, composite score, raw similarity).

        Both numbers are returned because they answer different questions.
        *Similarity* decides whether a tool is even about this problem, and is
        what a caller should threshold on. The *composite score* -- which folds
        in how well the tool has been graded -- decides which of several
        applicable tools to reach for first. Collapsing them into one number
        makes a highly-trusted tool look applicable to a problem it has nothing
        to do with.
        """
        out = []
        for hit in self.memory.recall(problem, k=k, kinds=(Kind.TOOL,)):
            spec = self.load(hit.trace.meta.get("tool", ""))
            if spec:
                out.append((spec, hit.score, hit.similarity))
        return out

    def bundle(self, spec: ToolSpec, seen: set | None = None) -> str:
        """A tool's source with its dependencies prepended, depth-first.

        This is how a forged tool builds on tools already proven to work, which
        is the difference between a growing toolbox and a pile of one-offs. The
        sandbox runs one file with no import path back into the workshop, so
        composition is by concatenation -- crude, and correct: the dependency
        source that runs is exactly the source that was verified.

        `seen` breaks cycles and de-duplicates a diamond, so a tool depending on
        two tools that share a dependency gets one copy of it rather than a
        redefinition.
        """
        seen = seen if seen is not None else set()
        if spec.name in seen:
            return ""
        seen.add(spec.name)
        parts = []
        for dep_name in spec.deps:
            dep = self.load(dep_name)
            if dep is None:
                raise ValueError(f"{spec.name} depends on {dep_name!r}, which is not registered")
            if dep.transport != "python":
                # Skipping it silently graded the tool against source that is not
                # what runs: validation passed on a bundle missing the dependency
                # and the call then died with NameError. An out-of-process
                # dependency cannot be inlined, so refuse rather than pretend.
                raise ValueError(
                    f"{spec.name} depends on {dep_name!r}, which runs over "
                    f"{dep.transport} and cannot be inlined into the sandbox")
            parts.append(self.bundle(dep, seen))
        parts.append(spec.source)
        return "\n\n".join(p for p in parts if p.strip())

    def invoke(self, name: str, args: list | None = None, kwargs: dict | None = None,
               timeout: float = 10.0):
        """Call a registered tool in the sandbox and bring back the result.

        The result crosses the process boundary as JSON, so a tool returning
        something unserialisable comes back as its `repr` with `json_ok` false
        rather than as a crash -- the tool ran, and that is a different fact from
        the tool failing.
        """
        spec = self.load(name)
        if spec is None:
            return {"ok": False, "error": f"no such tool: {name}"}
        if spec.transport == "mcp":
            # MCP tools are keyword-only by schema: there is no positional form,
            # so positional arguments cannot be delivered. They used to be
            # dropped while the call still reported ok=True -- a silent wrong
            # answer, the worst outcome a tool call has.
            if args:
                return {"ok": False, "error": (
                    f"{name} is an mcp tool and takes named arguments only; "
                    f"pass kwargs, not {len(args)} positional value(s)")}
            if self.mcp is None:
                return {"ok": False, "error": f"{name} is an mcp tool but no registry is attached"}
            problems = _schema_problems(spec, kwargs or {}, self.memory)
            if problems:
                # The schema was recorded at discovery and never consulted.
                # Checking it turns a confusing server-side error into a precise
                # local one, without a round trip.
                return {"ok": False, "error": "; ".join(problems)}
            mcp_name = next((t.meta.get("mcp_name") for t in self.memory.of_kind(Kind.TOOL)
                             if t.meta.get("tool") == name), None)
            out = self.mcp.call(name, kwargs or {}, tool_name=mcp_name)
            return {"ok": out.ok, "value": out.content, "json_ok": False,
                    **({} if out.ok else {"error": out.error})}
        try:
            bundled = self.bundle(spec)
        except ValueError as exc:
            # bundle() raises for a missing or out-of-process dependency. Every
            # other caller handles it; this one let the exception out of a
            # function whose whole contract is to return a result dict.
            return {"ok": False, "error": str(exc)}
        driver = (
            f"{bundled}\n\n"
            "import json as _json\n"
            f"_args = _json.loads({json.dumps(json.dumps(args or []))})\n"
            f"_kwargs = _json.loads({json.dumps(json.dumps(kwargs or {}))})\n"
            f"_out = {spec.name}(*_args, **_kwargs)\n"
            "try:\n"
            "    print('__RESULT__' + _json.dumps({'json_ok': True, 'value': _out}))\n"
            "except TypeError:\n"
            "    print('__RESULT__' + _json.dumps({'json_ok': False, 'value': repr(_out)}))\n")
        run = run_source(driver, timeout=timeout)
        if not run.ok:
            return {"ok": False, "error": run.diagnostic, "stderr": run.stderr[-500:]}
        for line in run.stdout.splitlines():
            if line.startswith("__RESULT__"):
                payload = json.loads(line[len("__RESULT__"):])
                return {"ok": True, **payload}
        return {"ok": False, "error": "tool produced no result", "stdout": run.stdout[-300:]}

    def record_use(self, name: str, problem: str, worked: bool) -> None:
        """Attach a solved problem to the tool and grade it.

        This is the line that makes the toolbox improve with use: the problem
        text joins the tool's embedded description, so the next query shaped like
        this one finds it, and the grade moves its credibility in recall.
        """
        spec = self.load(name)
        if spec is None:
            return
        if worked and problem not in spec.solved:
            spec.solved.append(problem)
            if spec.transport == "python":
                (self.workshop / f"{name}.json").write_text(json.dumps(spec.to_json(), indent=2))

        # Update the trace this tool already has. `remember` with fresh meta
        # created a SECOND trace for MCP tools -- their embed_text differs from
        # what discovery wrote -- so grades piled up on a duplicate while the
        # trace `load` and `names` actually read never moved. The same tool got
        # better and worse at once, in two places.
        # Prefer the trace this spec actually points at. Picking the first
        # match by name grades a stale duplicate whenever two traces share a
        # meta["tool"] -- which is exactly the situation record_use was fixed to
        # stop creating, so it must not depend on that never happening.
        existing = self.memory.get(spec.trace_id) if spec.trace_id else None
        # A trace_id read off disk can be stale, and after compression or a
        # rebuild the id may now belong to something else entirely. Following it
        # blindly graded and re-embedded a different tool's trace.
        if existing is not None and (existing.kind != Kind.TOOL
                                     or existing.meta.get("tool") != name):
            existing = None
        if existing is None:
            existing = next((t for t in self.memory.of_kind(Kind.TOOL)
                             if t.meta.get("tool") == name), None)
        if existing is None:
            existing = self.memory.remember(
                Kind.TOOL, spec.embed_text(),
                meta={"tool": name, "signature": spec.signature, "purpose": spec.purpose,
                      "transport": spec.transport, "solved": list(spec.solved)})
        else:
            existing.meta["solved"] = list(spec.solved)
            # Re-embed by EXTENDING the description, not replacing it. An MCP
            # tool's text is written at discovery and carries its server,
            # argument names and required list; overwriting it with the generic
            # ToolSpec text destroyed that provenance, so the tool that had
            # proved itself became the hardest one to find.
            # Re-forging replaces the trace text, so a cached base_text from the
            # previous version would quietly restore the old description on the
            # next use. Refresh it whenever the trace no longer starts with it.
            base = existing.meta.get("base_text")
            if not base or not existing.text.startswith(base):
                base = existing.text.split("\nsolved: ")[0]
                existing.meta["base_text"] = base
            solved = [f"solved: {p}" for p in spec.solved]
            existing.text = "\n".join([base] + solved)
            existing.vector = self.memory.embedder.embed(existing.text)
            self.memory.store.touch(existing)
        self.memory.grade(existing.id, 1.0 if worked else -1.0, Source.SELF)
