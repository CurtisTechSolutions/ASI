"""The starter toolkit: what is worth building before anything has been asked.

Three criteria decide what belongs here, and they come from the rest of the
design rather than from taste:

1. **It must be decidable.** Distillation stops at verifiability (DESIGN 8) and
   nothing enters the toolbox without passing its contract tests (DESIGN 11). A
   tool nobody can write an assert for is not a candidate.
2. **It must be routed through often.** A tool earns its place by how many later
   problems reach it, so the seeds are primitives other tools take as
   dependencies, not finished applications.
3. **It should make more things gradeable.** This is the one that matters most
   and is easiest to miss.

On that third point. The system's real ceiling is stated plainly in DESIGN 16:
prose is never self-graded, so it improves fastest at things a computer can
check. A tool that *creates a referee* therefore buys more than a tool that does
work -- it moves a whole class of goals from `UNCHECKABLE` to `CHECKABLE` in
`goals.checkability`, which is what decides where distillation is allowed to
stop. `assert_schema` turns "the output is well-formed" from an opinion into an
assert. `extract_identifiers` and `normalise_error` turn a wall of diagnostic
text into a stable class label, which is what `GREN/DESIGN.md` 15 means when it
says a structured code *is* the class.

These are planted through the ordinary `Toolsmith.validate` / `register` path,
not trusted because they ship with the package. If a seed fails its own contract
tests it is rejected exactly like anything else -- which is the point of having a
grader at all.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Seed:
    name: str
    purpose: str
    source: str
    tests: str
    deps: tuple[str, ...] = ()


# --- tier 0: the two things later seeds depend on ----------------------------
#
# These duplicate entries in `toolsmith.TEMPLATES`, and that is deliberate: the
# starter toolkit must stand on its own. Depending on the synthesiser having been
# asked the right question first made `median_of_column` fail to register on a
# fresh workspace, which is the wrong failure for a *starter* kit. `plant` skips
# either one if it is already present, so nothing is duplicated in practice.

MEDIAN = Seed("median", "the middle value of a sequence",
'''def median(xs):
    """Middle value; mean of the two middle values when the count is even."""
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
''')

PARSE_CSV = Seed("parse_csv", "split delimited text into rows of stripped fields",
'''def parse_csv(text, sep=","):
    """Rows of stripped fields. Blank lines are dropped."""
    rows = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        rows.append([cell.strip() for cell in line.split(sep)])
    return rows
''',
'''assert parse_csv("a, b\\nc,d") == [["a", "b"], ["c", "d"]]
assert parse_csv("") == []
assert parse_csv("x\\n\\ny") == [["x"], ["y"]]
''')


# --- tier 1: data substrate -- what almost every later tool stands on ---------

CHUNK = Seed("chunk", "split a sequence into batches of at most n",
'''def chunk(xs, n):
    """Consecutive batches of at most n items. The last batch may be short."""
    if n <= 0:
        raise ValueError("chunk size must be positive")
    return [list(xs[i:i + n]) for i in range(0, len(xs), n)]
''',
'''assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
assert chunk([], 3) == []
assert chunk([1], 5) == [[1]]
try:
    chunk([1], 0); raise SystemExit("should have raised")
except ValueError:
    pass
''')

PERCENTILE = Seed("percentile", "the p-th percentile of a sequence by linear interpolation",
'''def percentile(xs, p):
    """Linear-interpolation percentile, p in [0, 100].

    Interpolating rather than picking the nearest rank so that p50 of an
    even-length sequence agrees with the median, which is the property anyone
    comparing the two will assume holds.
    """
    if not xs:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (p / 100.0)
    low = int(pos)
    high = min(low + 1, len(s) - 1)
    return float(s[low] + (s[high] - s[low]) * (pos - low))
''',
'''assert percentile([1,2,3,4], 50) == 2.5
assert percentile([1,2,3,4], 0) == 1.0
assert percentile([1,2,3,4], 100) == 4.0
assert percentile([5], 99) == 5.0
try:
    percentile([], 50); raise SystemExit("should have raised")
except ValueError:
    pass
''')

JACCARD = Seed("jaccard", "token-overlap similarity between two texts, no embedder needed",
'''def jaccard(a, b):
    """|intersection| / |union| over lowercased word tokens, in [0, 1].

    A cheap second opinion on similarity that does not depend on the embedding
    layer -- useful precisely when the question is whether the embedder is
    behaving, since an agreement between two independent measures is worth more
    than either alone.
    """
    import re as _re
    ta = set(_re.findall(r"[a-z0-9']+", (a or "").lower()))
    tb = set(_re.findall(r"[a-z0-9']+", (b or "").lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
''',
'''assert jaccard("a b c", "a b c") == 1.0
assert jaccard("a b", "c d") == 0.0
assert jaccard("", "") == 1.0
assert jaccard("a", "") == 0.0
assert 0.3 < jaccard("the quick brown fox", "the quick red fox") < 0.7
''')

DIFF_LINES = Seed("diff_lines", "which lines were added and removed between two texts",
'''def diff_lines(before, after):
    """Added and removed lines, order preserved, duplicates respected."""
    import difflib as _difflib
    a = (before or "").splitlines()
    b = (after or "").splitlines()
    added, removed = [], []
    for line in _difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return {"added": added, "removed": removed}
''',
'''d = diff_lines("a\\nb", "a\\nc")
assert d["added"] == ["c"] and d["removed"] == ["b"]
assert diff_lines("x", "x") == {"added": [], "removed": []}
assert diff_lines("", "new")["added"] == ["new"]
''')

TOPO_SORT = Seed("topological_sort", "order items so every dependency comes first",
'''def topological_sort(graph):
    """Kahn's algorithm over {node: [dependencies]}. Raises on a cycle.

    Raising rather than returning a partial order: a dependency cycle is a bug in
    whatever built the graph, and silently emitting a plausible-looking order
    hides it until something downstream runs in the wrong sequence.
    """
    nodes = set(graph) | {d for deps in graph.values() for d in deps}
    remaining = {n: set(graph.get(n, [])) & nodes for n in nodes}
    out = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError(f"dependency cycle among {sorted(remaining)}")
        for n in ready:
            out.append(n)
            del remaining[n]
        for deps in remaining.values():
            deps.difference_update(ready)
    return out
''',
'''order = topological_sort({"app": ["lib"], "lib": ["core"], "core": []})
assert order.index("core") < order.index("lib") < order.index("app")
assert topological_sort({}) == []
assert topological_sort({"a": []}) == ["a"]
try:
    topological_sort({"a": ["b"], "b": ["a"]}); raise SystemExit("should have raised")
except ValueError:
    pass
''')

# --- tier 2: turning diagnostics into class labels ---------------------------

NORMALISE_ERROR = Seed("normalise_error", "collapse a traceback to a stable class label",
'''def normalise_error(text):
    """Reduce a traceback to {kind, message, signature}.

    The last line of a traceback is the exception; everything above is the path
    taken to reach it. Run-specific detail -- addresses, line numbers, temp paths
    -- is replaced with placeholders so the same *rule* violation produces the
    same signature across runs. That signature is what makes two failures
    comparable, and comparing failures is how a refusal boundary gets mapped.
    """
    import re as _re
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        return {"kind": "", "message": "", "signature": ""}
    last = lines[-1]
    match = _re.match(r"^([A-Za-z_][\\w.]*(?:Error|Exception|Warning|Exit))\\s*:?\\s*(.*)$", last)
    kind = match.group(1) if match else ""
    message = (match.group(2) if match else last).strip()
    sig = message
    sig = _re.sub(r"0x[0-9a-fA-F]+", "<addr>", sig)
    sig = _re.sub(r"(?:/[\\w.\\-]+)+", "<path>", sig)
    sig = _re.sub(r"\\bline \\d+\\b", "line <n>", sig)
    sig = _re.sub(r"(?<![\\w])\\d+(?![\\w])", "<n>", sig)
    sig = _re.sub(r"\\s+", " ", sig).strip()
    return {"kind": kind, "message": message,
            "signature": f"{kind}: {sig}" if kind else sig}
''',
'''r = normalise_error("Traceback...\\nValueError: invalid literal for int() with base 10: '7'")
assert r["kind"] == "ValueError"
assert normalise_error("KeyError: 'x'")["kind"] == "KeyError"
a = normalise_error("IndexError: index 5 is out of range")["signature"]
b = normalise_error("IndexError: index 9 is out of range")["signature"]
assert a == b, "the same rule violation must share a signature"
assert normalise_error("")["kind"] == ""
''')

EXTRACT_IDENTIFIERS = Seed("extract_identifiers", "pull the specifics out of a wall of text",
'''def extract_identifiers(text):
    """Identifiers, paths, numbers, quoted strings and error codes, deduplicated.

    These are the tokens that survive compression (DESIGN 13.1): the parts of a
    message that distinguish it from every other message. Ordinary prose words
    are exactly what a summary may safely lose; `E0308`, `5432` and
    `/etc/hosts` are not.
    """
    import re as _re
    text = text or ""
    found = {
        "codes": _re.findall(r"\\b[A-Z]{1,5}\\d{2,5}\\b", text),
        "paths": _re.findall(r"(?:/[\\w.\\-]+){2,}", text),
        "numbers": _re.findall(r"(?<![\\w.])\\d+(?:\\.\\d+)?(?![\\w.])", text),
        "quoted": _re.findall(r"['\\"]([^'\\"\\n]{1,60})['\\"]", text),
        "dotted": _re.findall(r"\\b[a-zA-Z_]\\w*(?:\\.\\w+){1,}\\b", text),
    }
    return {k: sorted(set(v)) for k, v in found.items()}
''',
'''r = extract_identifiers("connection to /var/run/pg.sock on port 5432 failed with E1234")
assert "5432" in r["numbers"]
assert "E1234" in r["codes"]
assert "/var/run/pg.sock" in r["paths"]
assert extract_identifiers("")["numbers"] == []
assert "it" in extract_identifiers("he said 'it' loudly")["quoted"]
''')

# --- tier 3: tools that create referees -- the highest-leverage kind ---------

ASSERT_SCHEMA = Seed("assert_schema", "decide whether data is well-formed, so output becomes checkable",
'''def assert_schema(value, schema, path="$"):
    """Validate against a small JSON-Schema subset. Returns a list of problems.

    Supports type, required, properties, items, enum, minimum, maximum. Returns
    problems rather than raising, because the caller usually wants all of them at
    once -- one assert per run turns a ten-field mismatch into ten runs.

    This is the tool that buys the most: "the output is well-formed" is an
    opinion until something can decide it, and a goal with no decision procedure
    is one distillation is not allowed to stop at (DESIGN 8).
    """
    problems = []
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "number": (int, float),
             "integer": int, "boolean": bool, "null": type(None)}
    if expected:
        want = types.get(expected)
        ok = isinstance(value, want) if want else True
        if expected in ("number", "integer") and isinstance(value, bool):
            ok = False                      # bool is an int in Python; not here
        if not ok:
            problems.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return problems                 # further checks assume the type held
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: {value} is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: {value} is above maximum {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}.{key}: required but missing")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                problems.extend(assert_schema(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and schema.get("items"):
        for i, item in enumerate(value):
            problems.extend(assert_schema(item, schema["items"], f"{path}[{i}]"))
    return problems
''',
'''S = {"type": "object", "required": ["name", "age"],
     "properties": {"name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string"}}}}
assert assert_schema({"name": "a", "age": 3}, S) == []
assert assert_schema({"name": "a"}, S) == ["$.age: required but missing"]
assert assert_schema({"name": 1, "age": 3}, S) == ["$.name: expected string, got int"]
assert assert_schema({"name": "a", "age": -1}, S) == ["$.age: -1 is below minimum 0"]
assert len(assert_schema({"name": "a", "age": 1, "tags": ["x", 2]}, S)) == 1
assert assert_schema(True, {"type": "integer"}) != [], "a bool is not an integer here"
''')

SUMMARISE_COUNTS = Seed("summarise_counts", "the shape of a column: counts, distinct, missing",
'''def summarise_counts(values):
    """A profile of one column -- total, missing, distinct, and the top values.

    What anyone actually asks first of an unfamiliar dataset, and the answer is
    fully decidable, which is why it is a tool rather than a judgement.
    """
    from collections import Counter as _Counter
    total = len(values)
    present = [v for v in values if v is not None and v != ""]
    counts = _Counter(present)
    return {"total": total, "missing": total - len(present),
            "distinct": len(counts), "top": counts.most_common(5)}
''',
'''r = summarise_counts(["a", "b", "a", None, ""])
assert r["total"] == 5 and r["missing"] == 2 and r["distinct"] == 2
assert r["top"][0] == ("a", 2)
assert summarise_counts([])["distinct"] == 0
''')

MEDIAN_OF_COLUMN = Seed("median_of_column", "median of one column of parsed CSV text",
'''def median_of_column(text, col=0):
    """Median of a CSV column, by composing two tools that are already verified.

    The dependencies are not imported -- the sandbox runs one file -- they are
    inlined by `Toolbox.bundle`. The source that runs is exactly the source that
    passed its own contract tests, which is what makes composition safe here.
    """
    rows = parse_csv(text)
    return median([float(r[col]) for r in rows if len(r) > col and r[col] != ""])
''',
'''assert median_of_column("10\\n30\\n20") == 20.0
assert median_of_column("1,5\\n2,9", 1) == 7.0
''', deps=("median", "parse_csv"))


#: Order matters: dependencies are planted before the tools that need them.
SEEDS: tuple[Seed, ...] = (
    MEDIAN, PARSE_CSV,
    CHUNK, PERCENTILE, JACCARD, DIFF_LINES, TOPO_SORT,
    NORMALISE_ERROR, EXTRACT_IDENTIFIERS,
    ASSERT_SCHEMA, SUMMARISE_COUNTS,
    MEDIAN_OF_COLUMN,
)


def plant(toolsmith, toolbox, seeds=SEEDS) -> dict:
    """Validate and register the starter toolkit through the ordinary path.

    Nothing is trusted for shipping with the package. A seed that fails its own
    contract tests is rejected and filed as a failure, exactly like a tool the
    system wrote for itself -- which is the only way the grade on the others
    means anything.
    """
    from .toolsmith import ToolSpec, signature_of
    report = {"planted": [], "rejected": [], "already": [], "skipped": []}
    for seed in seeds:
        if seed.name in toolbox.names():
            report["already"].append(seed.name)
            continue
        missing = [d for d in seed.deps if toolbox.load(d) is None]
        if missing:
            # Report it rather than registering a tool that cannot run. A
            # composite whose dependency is absent passes nothing and fails at
            # call time, which is exactly the class of failure the grader exists
            # to catch before the toolbox believes in it.
            report["skipped"].append((seed.name, f"missing dependencies: {', '.join(missing)}"))
            continue
        name, sig = signature_of(seed.source)
        spec = ToolSpec(name=name or seed.name, purpose=seed.purpose, source=seed.source,
                        tests=seed.tests, signature=sig, built_by="seed",
                        deps=list(seed.deps))
        grade = toolsmith.validate(spec, toolbox)
        if toolsmith.register(spec):
            report["planted"].append((spec.name, grade.score))
        else:
            report["rejected"].append((spec.name, grade.diagnostic))
    return report
