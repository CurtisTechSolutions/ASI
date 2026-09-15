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
    if n % 2:
        return s[n // 2]
    return _midpoint(s[n // 2 - 1], s[n // 2])


def _midpoint(lo, hi):
    """The value halfway between two numbers, exactly where that is possible.

    Two separate traps, and the obvious one-liner falls into one or the other.
    `(lo + hi) / 2` overflows to inf for two large floats whose midpoint is
    perfectly representable. `lo + (hi - lo) / 2` fixes that but promotes large
    integers to float and silently loses the unit -- the midpoint of 10**17 and
    10**17 + 2 came back as 1e17 rather than 10**17 + 1.

    So integers are handled with integer arithmetic and floats with the
    overflow-safe form.
    """
    if isinstance(lo, int) and isinstance(hi, int):
        total = lo + hi
        return total // 2 if total % 2 == 0 else total / 2
    return lo + (hi - lo) / 2
''',
'''assert median([3, 1, 2]) == 2
assert median([1, 2, 3, 4]) == 2.5
assert median([7]) == 7
assert median(iter([3, 1, 2])) == 2, "an iterator must work too"
big = 1.7e308
assert median([big, big]) == big, "the midpoint of two large floats must not overflow"
assert median([10**17, 10**17 + 2]) == 10**17 + 1, "exact for large integers"
try:
    median([]); raise SystemExit("should have raised")
except ValueError:
    pass
try:
    median(iter([])); raise SystemExit("should have raised")
except ValueError:
    pass
''')

PARSE_CSV = Seed("parse_csv", "parse CSV text into rows, honouring quoting and embedded separators",
'''def parse_csv(text, sep=","):
    """Rows of stripped fields, using the real CSV grammar.

    This was a `line.split(sep)` and the name was a lie: a quoted field
    containing the separator was torn in half, and every tool built on it
    inherited the bug. The standard library already implements the grammar
    correctly, so using it is strictly better than a splitter that is right most
    of the time.
    """
    import csv as _csv
    import io as _io
    rows = []
    for row in _csv.reader(_io.StringIO(text or ""), delimiter=sep,
                           skipinitialspace=True):
        if not any(cell.strip() for cell in row):
            continue
        rows.append([cell.strip() for cell in row])
    return rows
''',
'''assert parse_csv("a, b\\nc,d") == [["a", "b"], ["c", "d"]]
assert parse_csv("") == []
assert parse_csv("x\\n\\ny") == [["x"], ["y"]]
assert parse_csv(\'"a,b",c\') == [["a,b", "c"]], "a quoted separator is one field"
assert parse_csv(\'"say ""hi""",x\') == [[\'say "hi"\', "x"]]
assert parse_csv("a\\tb", sep="\\t") == [["a", "b"]]
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
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    # Sort first, then test. `not xs` is False for a non-empty-looking iterator
    # and for any generator at all, so an empty iterator reached the indexing
    # below and raised IndexError instead of the documented ValueError.
    s = sorted(xs)
    if not s:
        raise ValueError("percentile of an empty sequence")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (p / 100.0)
    low = int(pos)
    frac = pos - low
    if frac == 0:
        return s[low]          # exact: no float cast, so large ints stay exact
    high = min(low + 1, len(s) - 1)
    if frac == 0.5 and isinstance(s[low], int) and isinstance(s[high], int):
        # The p50 case, which must agree with `median` exactly -- including on
        # integers large enough that a float round trip loses the unit.
        total = s[low] + s[high]
        return total // 2 if total % 2 == 0 else total / 2
    return s[low] + (s[high] - s[low]) * frac
''',
'''assert percentile([1,2,3,4], 50) == 2.5
assert percentile([1,2,3,4], 0) == 1
assert percentile([1,2,3,4], 100) == 4
assert percentile([5], 99) == 5
# the property the docstring promises, on the inputs that used to break it
def _mid(lo, hi):
    if isinstance(lo, int) and isinstance(hi, int):
        t = lo + hi
        return t // 2 if t % 2 == 0 else t / 2
    return lo + (hi - lo) / 2
for case in ([1,2,3], [1,2,3,4], [0.1,0.2,0.3], [10**17, 10**17+1, 10**17+2],
             [10**17, 10**17+2], [1.5, 2.5], [7], [1, 2]):
    srt = sorted(case)
    n = len(srt)
    want = srt[n//2] if n % 2 else _mid(srt[n//2-1], srt[n//2])
    assert percentile(case, 50) == want, (case, percentile(case, 50), want)
try:
    percentile([], 50); raise SystemExit("should have raised")
except ValueError:
    pass
try:
    percentile(iter([]), 50); raise SystemExit("should have raised")
except ValueError:
    pass
try:
    percentile([1], 101); raise SystemExit("should have raised")
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
    ta = set(_re.findall(r"\\w+", (a or "").lower(), _re.UNICODE))
    tb = set(_re.findall(r"\\w+", (b or "").lower(), _re.UNICODE))
    if not ta and not tb:
        # Two texts with no word characters are not automatically identical.
        # The old ASCII-only pattern produced no tokens for any non-Latin script,
        # so every pair of such texts scored a perfect 1.0 -- a similarity
        # measure that says "the same" about two unrelated documents.
        return 1.0 if (a or "").strip() == (b or "").strip() else 0.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
''',
'''assert jaccard("a b c", "a b c") == 1.0
assert jaccard("a b", "c d") == 0.0
assert jaccard("", "") == 1.0
assert jaccard("a", "") == 0.0
assert 0.3 < jaccard("the quick brown fox", "the quick red fox") < 0.7
assert jaccard("\\u65e5\\u672c\\u8a9e", "\\u4e2d\\u6587") == 0.0, "different scripts are not identical"
assert jaccard("\\u65e5\\u672c", "\\u65e5\\u672c") == 1.0
assert jaccard("!!!", "???") == 0.0, "punctuation-only texts are not identical"
assert jaccard("!!!", "!!!") == 1.0
''')

DIFF_LINES = Seed("diff_lines", "which lines were added and removed between two texts",
'''def diff_lines(before, after):
    """Added and removed lines, order preserved, duplicates respected."""
    import difflib as _difflib
    a = (before or "").splitlines()
    b = (after or "").splitlines()
    added, removed = [], []
    # Opcodes, not parsed diff text. Reading unified_diff output back meant
    # guessing whether a leading "+" was the marker or the content, so a line
    # that itself began with "+++" or "---" was silently dropped -- and a change
    # consisting only of such lines was reported as no change at all. Diffing C
    # preprocessor output or a diff-of-a-diff hit it immediately.
    for tag, i1, i2, j1, j2 in _difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(a[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(b[j1:j2])
    return {"added": added, "removed": removed}
''',
'''d = diff_lines("a\\nb", "a\\nc")
assert d["added"] == ["c"] and d["removed"] == ["b"]
assert diff_lines("x", "x") == {"added": [], "removed": []}
assert diff_lines("", "new")["added"] == ["new"]
# lines that look like diff markers are content, not syntax
d = diff_lines("--- old", "+++ new")
assert d["removed"] == ["--- old"] and d["added"] == ["+++ new"]
d = diff_lines("@@ -1 +1 @@", "@@ -2 +2 @@")
assert d["removed"] and d["added"], "a change must never report as no change"
assert diff_lines("a\\na", "a")["removed"] == ["a"], "duplicates are respected"
''')

TOPO_SORT = Seed("topological_sort", "order items so every dependency comes first",
'''def topological_sort(graph):
    """Kahn's algorithm over {node: [dependencies]}. Raises on a cycle.

    Raising rather than returning a partial order: a dependency cycle is a bug in
    whatever built the graph, and silently emitting a plausible-looking order
    hides it until something downstream runs in the wrong sequence.
    """
    for node, deps in graph.items():
        if isinstance(deps, (str, bytes)):
            # set("abc") is {"a","b","c"}. A string dependency list therefore
            # expanded into one phantom node per character and the sort silently
            # succeeded over a graph nobody wrote.
            raise TypeError(f"dependencies of {node!r} must be a list, not a string")
    nodes = set(graph) | {d for deps in graph.values() for d in deps}
    remaining = {n: set(graph.get(n, [])) & nodes for n in nodes}
    out = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError(f"dependency cycle: {\' -> \'.join(_find_cycle(remaining))}")
        for n in ready:
            out.append(n)
            del remaining[n]
        for deps in remaining.values():
            deps.difference_update(ready)
    return out


def _find_cycle(remaining):
    """One actual cycle, by walking edges until a node repeats.

    The message used to list every unresolved node, which on a large graph is
    hundreds of names of which two are the problem.
    """
    start = sorted(remaining)[0]
    seen, path, node = {}, [], start
    while node not in seen:
        seen[node] = len(path)
        path.append(node)
        nxt = sorted(remaining.get(node, ()))
        if not nxt:
            return path
        node = nxt[0]
    return path[seen[node]:] + [node]
''',
'''order = topological_sort({"app": ["lib"], "lib": ["core"], "core": []})
assert order.index("core") < order.index("lib") < order.index("app")
assert topological_sort({}) == []
assert topological_sort({"a": []}) == ["a"]
try:
    topological_sort({"a": ["b"], "b": ["a"]}); raise SystemExit("should have raised")
except ValueError as e:
    assert "a" in str(e) and "b" in str(e)
try:
    topological_sort({"a": "bc"}); raise SystemExit("a string is not a dependency list")
except TypeError:
    pass
# the cycle message names the cycle, not every unresolved node
g = {"x": ["y"], "y": ["x"]}
for i in range(20):
    g["n%d" % i] = ["x"]
try:
    topological_sort(g); raise SystemExit("should have raised")
except ValueError as e:
    assert "n5" not in str(e), "only the cycle belongs in the message"
''')

# --- tier 2: turning diagnostics into class labels ---------------------------

NORMALISE_ERROR = Seed("normalise_error", "collapse a traceback to a stable class label",
'''def normalise_error(text):
    """Reduce a traceback to {kind, message, signature}.

    The signature is the point: run-specific detail is replaced with
    placeholders so the same *rule* violation produces the same string across
    runs, and comparing failures is how a refusal boundary gets mapped.
    """
    import re as _re
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        return {"kind": "", "message": "", "signature": ""}

    # An exception name is a dotted identifier whose final component starts with
    # a capital. The old pattern was a suffix whitelist -- Error|Exception|
    # Warning|Exit -- and missed KeyboardInterrupt, StopIteration, socket.timeout
    # and every custom type not ending in "Error".
    # Either a module-qualified name of any case (socket.timeout is a real
    # exception type and is lowercase) or a single capitalised name.
    head = _re.compile(r"^((?:[a-zA-Z_]\w*\.)+[a-zA-Z_]\w*|[A-Z]\w*)\s*:\s*(.*)$")

    # Search backwards for the last line that looks like an exception. Taking the
    # last non-blank line unconditionally meant a multi-line message -- common
    # for SyntaxError and assertion output -- produced an empty kind and a
    # signature that moved with the body text.
    kind, message, idx = "", lines[-1], len(lines) - 1
    for i in range(len(lines) - 1, -1, -1):
        m = head.match(lines[i])
        if m:
            kind, message, idx = m.group(1), m.group(2).strip(), i
            break
    if kind and idx + 1 < len(lines):
        message = " ".join([message] + lines[idx + 1:]).strip()

    sig = message
    sig = _re.sub(r"0x[0-9a-fA-F]+", "<addr>", sig)
    # Two or more segments before this counts as a path. One segment matched any
    # slash-joined pair -- including the "/" in "unsupported operand type(s) for
    # /:" -- and collapsed unrelated error classes onto one signature.
    sig = _re.sub(r"(?:/[\w.\-]+){2,}", "<path>", sig)
    sig = _re.sub(r"\\bline \d+\\b", "line <n>", sig)
    sig = _re.sub(r"(?<![\w])-?\d+(?![\w])", "<n>", sig)
    sig = _re.sub(r"\s+", " ", sig).strip()
    return {"kind": kind, "message": message,
            "signature": f"{kind}: {sig}" if kind else sig}
''',
'''r = normalise_error("Traceback...\\nValueError: invalid literal for int()")
assert r["kind"] == "ValueError"
assert normalise_error("KeyError: 'x'")["kind"] == "KeyError"
a = normalise_error("IndexError: index 5 is out of range")["signature"]
b = normalise_error("IndexError: index 9 is out of range")["signature"]
assert a == b, "the same rule violation must share a signature"
assert normalise_error("")["kind"] == ""
assert normalise_error("KeyboardInterrupt: ")["kind"] == "KeyboardInterrupt"
assert normalise_error("StopIteration: x")["kind"] == "StopIteration"
assert normalise_error("socket.timeout: timed out")["kind"] == "socket.timeout"
multi = normalise_error("Traceback:\\n  File x\\nAssertionError: expected\\n  got 3 instead")
assert multi["kind"] == "AssertionError", multi
assert "got" in multi["message"], "continuation lines belong to the message"
x = normalise_error("TypeError: unsupported operand type(s) for /: int")["signature"]
y = normalise_error("TypeError: unsupported operand type(s) for +: int")["signature"]
assert x != y, "different operators are different rules"
p1 = normalise_error("OSError: cannot open /var/run/a.sock")["signature"]
p2 = normalise_error("OSError: cannot open /var/run/b.sock")["signature"]
assert p1 == p2 and "<path>" in p1
assert (normalise_error("IndexError: index -5 out of range")["signature"] ==
        normalise_error("IndexError: index -9 out of range")["signature"])
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
    # A path segment may contain dots, but a trailing dot is sentence
    # punctuation -- "/etc/hosts." and "/etc/hosts" are one path, not two.
    paths = [m.rstrip(".,;:)") for m in
             _re.findall(r"(?<![\w.])(?:/[\w.\-]+){2,}", text)]
    return {
        "codes": sorted(set(_re.findall(r"\\b[A-Z]{1,5}\d{2,5}\\b", text))),
        "paths": sorted(set(paths)),
        # The old lookahead (?![\w.]) rejected any number followed by a period,
        # which is every number ending a sentence -- "refused on port 5432."
        # lost the port. The sign is captured too: -5 and 5 are different.
        "numbers": sorted(set(
            _re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?!\w)", text))),
        # A backreference, so a quote must be closed by the SAME kind of quote.
        # The old character class let an apostrophe open a span that a double
        # quote closed, swallowing everything between them.
        "quoted": sorted({m[1] for m in
                          _re.findall(r"(['\\"])([^'\\"\\n]{1,60})\\1", text)}),
        "dotted": sorted(set(
            _re.findall(r"\\b[a-zA-Z_]\w*(?:\.\w+){1,}\\b", text))),
    }
''',
'''r = extract_identifiers("connection to /var/run/pg.sock on port 5432 failed with E1234")
assert "5432" in r["numbers"]
assert "E1234" in r["codes"]
assert "/var/run/pg.sock" in r["paths"]
assert extract_identifiers("")["numbers"] == []
assert "it" in extract_identifiers("he said 'it' loudly")["quoted"]
assert "5432" in extract_identifiers("refused on port 5432.")["numbers"]
assert "3" in extract_identifiers("retried 3 times. gave up")["numbers"]
assert "-5" in extract_identifiers("index -5 is out of range")["numbers"]
q = extract_identifiers("mixing 'single' and \\"double\\" quotes")["quoted"]
assert "single" in q and "double" in q, q
assert extract_identifiers("see /etc/nginx/nginx.conf.")["paths"] == ["/etc/nginx/nginx.conf"]
''')

# --- tier 3: tools that create referees -- the highest-leverage kind ---------

ASSERT_SCHEMA = Seed("assert_schema", "decide whether data is well-formed, so output becomes checkable",
'''#: Keywords this validator actually enforces.
_ENFORCED = {"type", "required", "properties", "items", "enum", "minimum", "maximum"}
#: Keywords that carry no constraint, so ignoring them loses nothing.
_ANNOTATIONS = {"title", "description", "$comment", "default", "examples",
                "$schema", "$id", "definitions", "$defs"}
_TYPES = {"object": dict, "array": list, "string": str, "number": (int, float),
          "integer": int, "boolean": bool, "null": type(None)}


def assert_schema(value, schema, path="$"):
    """Validate against a small JSON-Schema subset. Returns a list of problems.

    Returns problems rather than raising, because the caller usually wants all
    of them at once -- one assert per run turns a ten-field mismatch into ten
    runs.

    This tool is a referee, and a referee that fails open is worse than none:
    it reports success for constraints it never checked. So it fails CLOSED in
    three ways. An unrecognised `type` name is a problem, not a pass. A keyword
    outside the enforced set is reported as unenforced rather than ignored, so
    the caller knows what was not checked. And a malformed schema is reported as
    a schema problem instead of raising AttributeError from somewhere inside.
    """
    problems = []
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object, got {type(schema).__name__}"]

    unknown = set(schema) - _ENFORCED - _ANNOTATIONS
    for key in sorted(unknown):
        problems.append(f"{path}: keyword {key!r} is not enforced by this validator")

    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        bad = [n for n in names if n not in _TYPES]
        if bad:
            # Previously an unknown type name meant `want` was None and the
            # check was skipped, so a typo like "interger" accepted every value
            # silently.
            return problems + [f"{path}: unknown type {bad[0]!r} in schema"]
        if not any(_is_type(value, n) for n in names):
            problems.append(
                f"{path}: expected {expected}, got {type(value).__name__}")
            return problems          # later checks assume the type held

    if "enum" in schema:
        allowed = schema["enum"]
        if not isinstance(allowed, list):
            problems.append(f"{path}: enum must be a list")
        elif not any(_same(value, a) for a in allowed):
            problems.append(f"{path}: {value!r} is not one of {allowed}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, op, word in (("minimum", float.__lt__, "below"),
                              ("maximum", float.__gt__, "above")):
            bound = schema.get(key)
            if isinstance(bound, (int, float)) and not isinstance(bound, bool):
                if op(float(value), float(bound)):
                    problems.append(f"{path}: {value} is {word} {key} {bound}")

    if isinstance(value, dict):
        required = schema.get("required") or []
        if isinstance(required, (str, bytes)):
            problems.append(f"{path}: required must be a list of names")
            required = []
        for key in required:
            if key not in value:
                problems.append(f"{path}.{key}: required but missing")
        props = schema.get("properties") or {}
        if not isinstance(props, dict):
            problems.append(f"{path}: properties must be an object")
            props = {}
        for key, sub in props.items():
            if key in value:
                problems.extend(assert_schema(value[key], sub, f"{path}.{key}"))

    if isinstance(value, list) and schema.get("items") is not None:
        items = schema["items"]
        if isinstance(items, list):
            for i, (item, sub) in enumerate(zip(value, items)):
                problems.extend(assert_schema(item, sub, f"{path}[{i}]"))
        else:
            for i, item in enumerate(value):
                problems.extend(assert_schema(item, items, f"{path}[{i}]"))
    return problems


def _is_type(value, name):
    """JSON Schema typing, where a bool is NOT a number and 1.0 IS an integer."""
    if name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (
            isinstance(value, float) and value.is_integer())
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    return isinstance(value, _TYPES[name])


def _same(a, b):
    """Equality that does not let True satisfy an enum of 1.

    Python says True == 1, so a plain `in` check accepted a boolean wherever a
    numeric literal was allowed -- contradicting the bool-is-not-an-integer rule
    this same validator enforces two lines earlier.
    """
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b
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
assert assert_schema(True, {"type": "integer"}) != [], "a bool is not an integer"
assert assert_schema(True, {"type": "number"}) != [], "a bool is not a number"
assert assert_schema(True, {"type": "boolean"}) == []
# fail closed on a schema it does not understand
assert assert_schema("anything", {"type": "interger"}) != [], "a typo must not accept everything"
assert assert_schema(1, {"multipleOf": 2}) != [], "an unenforced keyword must be reported"
assert assert_schema(1, {"type": "integer", "title": "n"}) == [], "annotations are safe to ignore"
# enum must not let a bool satisfy a numeric option
assert assert_schema(True, {"enum": [1, 2]}) != []
assert assert_schema(1, {"enum": [1, 2]}) == []
# JSON Schema counts an integral float as an integer
assert assert_schema(1.0, {"type": "integer"}) == []
assert assert_schema(1.5, {"type": "integer"}) != []
# malformed schemas are reported, never raised
assert assert_schema({}, {"properties": "nope"}) != []
assert assert_schema({}, "not a schema") != []
assert assert_schema({"a": 1}, {"required": "a"}) != []
assert assert_schema([1, "x"], {"items": [{"type": "integer"}, {"type": "string"}]}) == []
assert assert_schema(5, {"type": ["integer", "string"]}) == []
''')

SUMMARISE_COUNTS = Seed("summarise_counts", "the shape of a column: counts, distinct, missing",
'''def summarise_counts(values):
    """A profile of one column -- total, missing, distinct, and the top values.

    What anyone actually asks first of an unfamiliar dataset, and the answer is
    fully decidable, which is why it is a tool rather than a judgement.
    """
    from collections import Counter as _Counter
    values = list(values)
    total = len(values)
    present = [v for v in values if v is not None and v != ""]
    # Key by (type, value). True == 1 and 1.0 == 1 in Python, so a plain Counter
    # merged booleans with integers and under-reported `distinct` on exactly the
    # mixed-type column a profile is being asked about.
    counts = _Counter((type(v).__name__, v) for v in present)
    return {"total": total, "missing": total - len(present),
            "distinct": len(counts),
            "top": [(v, n) for (_t, v), n in counts.most_common(5)]}
''',
'''r = summarise_counts(["a", "b", "a", None, ""])
assert r["total"] == 5 and r["missing"] == 2 and r["distinct"] == 2
assert r["top"][0] == ("a", 2)
assert summarise_counts([])["distinct"] == 0
assert summarise_counts(iter(["a", "a"]))["total"] == 2, "an iterator must work"
assert summarise_counts([True, 1, 1.0])["distinct"] == 3, "True is not 1 in a column"
assert summarise_counts([0, False])["distinct"] == 2
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
