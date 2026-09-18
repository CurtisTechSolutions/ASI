"""A local HTTP frontend for the system, in the standard library.

`http.server` rather than a framework, and hand-written SVG rather than a chart
library, for the same reason the rest of this package has no dependencies: an
agent whose interface needs a build step and a package manager is one you cannot
run on the machine where the problem is.

**This is a local tool, not a service.** It binds 127.0.0.1 by default and it
drives an agent that writes files and executes generated code in a subprocess --
so exposing it on a network hands anyone who can reach it that capability.
`--host` exists because someone will want it in a container; the warning it
prints exists because they should know what they are opening. There is no
authentication here and adding a token would suggest a level of hardening this
does not have.

Three defences that are cheap and worth having anyway. Static files are resolved
and confirmed to sit under `web/` before they are read, so `..` cannot escape it.
Requests whose `Host` header is not a loopback name are refused, which is what
stops a DNS-rebinding page in a browser from reaching this server on the user's
behalf. And cross-site requests are refused outright -- the `Host` check alone
does not stop them, because a page on any origin may address `127.0.0.1`
directly and the `Host` header it sends is then perfectly honest. Without that
third check a visited web page could POST to `/api/forge` and have this agent
write and execute code; the attacker could not read the reply, but the side
effect would already have happened.

The API is JSON over GET and POST. Every endpoint returns `{ok, ...}` or
`{ok: false, error}` -- the agent degrades rather than raising almost everywhere,
and the frontend should show what went wrong rather than a blank panel.
"""
from __future__ import annotations

import json
import mimetypes
import queue
import threading
import traceback
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .agent import Distil
from .grade import Grade
from .memory import Kind, Source, Trace
from .project import project
from .provider import catalogue

WEB = Path(__file__).resolve().parent.parent / "web"
LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


# --------------------------------------------------------------------------- #
# turning the system's objects into JSON
# --------------------------------------------------------------------------- #

def trace_json(t: Trace, policy) -> dict:
    return {
        "id": t.id, "kind": t.kind, "text": t.text, "meta": _plain(t.meta),
        "links": t.links, "hits": t.hits, "seen": t.seen,
        "grades": [{"score": g[0], "source": g[1]} for g in t.grades],
        "grade": t.mean_grade, "verified": t.verified,
        "credibility": round(t.credibility(policy), 4),
        "embedder": t.embedder,
    }


def grade_json(g: Grade | None) -> dict | None:
    if g is None:
        return None
    return {"score": g.score, "source": g.source, "passed": g.passed, "clean": g.clean,
            "diagnostic": g.diagnostic,
            "stages": [{"name": s.name, "passed": s.passed, "detail": s.detail}
                       for s in g.stages]}


def frame_json(frame, plan=None) -> dict | None:
    if frame is None:
        return None
    out = frame.to_json()
    out["solution"] = frame.solution
    out["understood"] = frame.understood
    if plan is not None:
        out["agenda"] = {
            "items": plan.items,
            "needs_person": plan.needs_person,
            "advisories": plan.advisories,
            "actionable": plan.actionable_items,
            "capabilities": [{"action": c.action, "covered_by": c.covered_by,
                              "confidence": c.confidence, "gap": c.gap}
                             for c in plan.capabilities],
        }
    return out


def chain_json(chain) -> list[dict]:
    return [{"kind": s.kind, "text": s.text, "payload": _plain(s.payload)}
            for s in chain.steps]


def tree_json(tree) -> dict:
    return {
        "task": tree.task,
        "goals": [{"id": g.id, "text": g.text, "parent": g.parent, "depth": g.depth,
                   "status": g.status, "atomic": g.atomic, "cost": g.cost,
                   "p_success": g.p_success, "children": g.children}
                  for g in tree.goals.values()],
        "solved": tree.solved,
    }


def _plain(value):
    """Anything the frontend receives has to survive `json.dumps`.

    Payloads carry whatever a reasoning step attached -- dataclasses, sets,
    tuples, occasionally a Grade -- and one unserialisable object would fail the
    whole response rather than the field. Unknown types degrade to their repr.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return _plain(asdict(value))
        except TypeError:
            return repr(value)
    return repr(value)


# --------------------------------------------------------------------------- #

class Api:
    """The agent, plus a lock.

    `ThreadingHTTPServer` handles requests concurrently and `Distil` holds
    mutable memory, a workshop directory and MCP subprocesses. One lock around
    every call is coarse and correct; the alternative is two requests forging
    the same tool into the same file at once.
    """

    def __init__(self, agent: Distil) -> None:
        self.agent = agent
        self.lock = threading.Lock()
        self.auto = None                 # the running autonomous loop, if any

    # -- reads ---------------------------------------------------------------

    def state(self, _q) -> dict:
        d = self.agent
        return {
            "stats": d.stats(),
            "policy": d.policy.to_json(),
            "providers": [{"name": p.name, "available": p.available(),
                           "embeds": p.can_embed, "why": p.why_unavailable()}
                          for p in catalogue()],
            "provider": d.provider.name,
            # How the attached provider is actually doing. A provider that is
            # selected but answering nothing looks identical to a working one
            # from the outside -- every caller degrades on failure, correctly --
            # so the counters are the only way to see it.
            "provider_health": d.provider.health(),
            "home": str(d.workspace),
            "mcp": d.mcp.names(),
            "kinds": list(Kind.ALL),
        }

    def memory(self, q) -> dict:
        """Every trace, with a 2D position.

        The projection runs over the STORED vectors rather than re-embedding the
        text: the lexical embedder learns its IDF online, so re-embedding an old
        trace gives a slightly different vector than the one recall actually
        compares against. Showing a picture of vectors the system does not use
        would be a picture of nothing.
        """
        kinds = [k for k in q.get("kind", []) if k in Kind.ALL]
        traces = [t for t in self.agent.memory.store.all()
                  if not kinds or t.kind in kinds]
        traces.sort(key=lambda t: t.created)
        by_backend: dict[str, list] = {}
        for t in traces:
            by_backend.setdefault(t.embedder, []).append(t)
        points: dict[str, tuple] = {}
        for backend, group in by_backend.items():
            # One projection per backend. Two geometries share no plane, and
            # laying them out together would put unrelated things side by side.
            for t, xy in zip(group, project([t.vector for t in group])):
                points[t.id] = xy
        policy = self.agent.policy
        return {"traces": [{**trace_json(t, policy),
                            "x": points.get(t.id, (0.0, 0.0))[0],
                            "y": points.get(t.id, (0.0, 0.0))[1]}
                           for t in traces],
                "backends": sorted(by_backend)}

    def recall(self, q) -> dict:
        query = (q.get("q") or [""])[0]
        k = int((q.get("k") or ["8"])[0])
        if not query.strip():
            return {"hits": [], "query": query}
        policy = self.agent.policy
        hits = self.agent.memory.recall(query, k=k)
        return {"query": query, "weights": {
            "similarity": policy.recall_similarity_weight,
            "credibility": policy.recall_credibility_weight,
            "recency": policy.recall_recency_weight},
            "hits": [{"trace": trace_json(h.trace, policy), "score": round(h.score, 5),
                      "similarity": round(h.similarity, 5),
                      "credibility": round(h.credibility, 5),
                      "recency": round(h.recency, 5)} for h in hits]}

    def frame(self, q) -> dict:
        task = (q.get("task") or [""])[0]
        if not task.strip():
            return {"error": "a task is required"}
        frame, plan = self.agent.understand(task)
        return {"frame": frame_json(frame, plan)}

    def tools(self, _q) -> dict:
        out = []
        for spec in self.agent.toolbox.all():
            out.append({"name": spec.name, "purpose": spec.purpose,
                        "signature": spec.signature, "transport": spec.transport,
                        "built_by": spec.built_by, "deps": spec.deps,
                        "solved": spec.solved, "source": spec.source,
                        "tests": spec.tests, "grade": grade_json(spec.grade)})
        return {"tools": out}

    def cases(self, q) -> dict:
        like = (q.get("like") or [""])[0]
        if like.strip():
            found = self.agent.casebook.adapt(like)
            best = found.get("precedent")
            return {"note": found["note"],
                    "precedent": _case(best) if best else None,
                    "avoid": [_case(p) for p in found["avoid"]]}
        return {"cases": [{"problem": c.problem, "solution": c.solution, "grade": c.grade,
                           "via": c.via, "evidence": c.evidence, "tags": c.tags}
                          for c in self.agent.casebook.all()]}

    def ideas(self, q) -> dict:
        seed = (q.get("seed") or [None])[0]
        found = self.agent.explorer.brainstorm(seed, n=int((q.get("n") or ["8"])[0]))
        policy = self.agent.policy
        from .game import entropy
        return {"target": policy.target_success,
                "ideas": [{"text": i.text, "origin": i.origin, "p": i.p_success,
                           "cost": i.cost, "novelty": round(i.novelty, 4),
                           "bits": round(entropy(i.p_success), 4),
                           "value": round(i.value(policy), 4)} for i in found]}

    def trace(self, q) -> dict:
        """One trace, with the graph around it resolved.

        `links` is stored on every trace and was never rendered, which meant
        credit propagation -- the mechanism that moves a grade from an answer back
        to what it was built on -- ran entirely out of sight. Backlinks are
        computed by scanning the store because links are one-directional: a tool
        does not know which solutions used it, and that is the direction a person
        reads in.
        """
        trace_id = (q.get("id") or [""])[0].strip()
        if not trace_id:
            return {"error": "a trace id is required"}
        trace = self.agent.memory.store.get(trace_id)
        if trace is None:
            return {"error": f"no such trace: {trace_id}"}
        policy = self.agent.policy
        linked = [self.agent.memory.store.get(i) for i in trace.links]
        backlinks = [t for t in self.agent.memory.store.all()
                     if trace_id in t.links and t.id != trace_id]
        return {"trace": trace_json(trace, policy),
                "links": [_edge(t, policy) for t in linked if t is not None],
                "missing": [i for i, t in zip(trace.links, linked) if t is None],
                "backlinks": [_edge(t, policy) for t in backlinks]}

    def speech(self, _q) -> dict:
        """What this machine can transcribe, and where the audio would go."""
        from .speech import available
        return available()

    def mcp(self, _q) -> dict:
        servers = []
        for name in self.agent.mcp.names():
            server = self.agent.mcp.servers[name]
            servers.append({"name": name, "command": list(server.command),
                            "alive": server.alive()})
        return {"servers": servers,
                "config": str(self.agent.mcp.config_path or ""),
                "tools": [t.name for t in self.agent.toolbox.all()
                          if t.transport == "mcp"]}

    def concepts(self, q) -> dict:
        """Cluster the store and name what forms, compressing each cluster.

        A read that writes, which is unusual enough to be worth stating: the
        point of thinking is that what it finds becomes memory. `dry=1` looks
        without writing, for a caller that wants to see the structure first.
        """
        from .think import Thinker
        thinker = Thinker(self.agent.memory, self.agent.policy,
                          compressor=self.agent.compressor)
        limit = max(1, min(12, int((q.get("limit") or ["4"])[0])))
        found = thinker.think(limit=limit)
        written = [] if (q.get("dry") or [""])[0] else thinker.absorb(found)
        if written:
            self.agent.save()
        return {"concepts": [{**d.to_json(), "text": d.text} for d in found],
                "written": [t.id for t in written],
                "skipped": thinker.skipped(),
                "pool": len(thinker.pool())}

    def journal(self, _q) -> dict:
        path = self.agent.workspace.journal
        if not path.exists():
            return {"entries": []}
        entries = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"entries": entries[-200:]}

    def lineage(self, _q) -> dict:
        return {"generations": [g.to_json() for g in self.agent.editor.lineage()]}

    # -- writes --------------------------------------------------------------

    def ask(self, body: dict) -> dict:
        task = (body.get("task") or "").strip()
        if not task:
            return {"error": "a task is required"}
        answers = body.get("answers") or {}
        ask_fn = body.get("_ask") or ((lambda questions: dict(answers)) if answers else None)
        result = self.agent.solve(task, ask=ask_fn,
                                  interrogate=bool(body.get("interrogate", True)),
                                  observer=body.get("_observer"))
        out = {"task": task, "solved": result.get("solved"),
               "reason": result.get("reason"),
               "frame": frame_json(result.get("frame"), result.get("agenda")),
               "needs_clarification": bool(result.get("needs_clarification")),
               "attempts": result.get("attempts", []),
               "goals_met": result.get("goals_met", 0),
               "goals_open": result.get("goals_open", 0),
               "refusals": result.get("refusals", []),
               "boundary": result.get("boundary"),
               # Every reply carries what the store already knew, gated or not.
               "recall": result.get("recall") or {"hits": []},
               "credit": result.get("credit", {})}
        if result.get("questions"):
            out["questions"] = [{"gap": q.gap, "text": q.text, "unblocks": q.unblocks,
                                 "value": q.value, "known": q.answered_by_memory}
                                for q in result["questions"]]
        session = result.get("session")
        if session is not None:
            out["chain"] = chain_json(session.chain)
            out["tree"] = tree_json(session.tree)
            out["chosen"] = session.chosen.text if session.chosen else None
            out["trace_id"] = session.trace_id
            select = [s for s in session.chain.steps if s.kind == "SELECT"]
            payoff = [s for s in session.chain.steps if s.kind == "PAYOFF"]
            if payoff:
                out["payoff"] = {"matrix": payoff[0].payload.get("matrix"),
                                 "states": payoff[0].payload.get("states"),
                                 "roles": payoff[0].payload.get("roles"),
                                 # From the step's own payload, not the frontier:
                                 # the chosen goal has been marked met by now and
                                 # is no longer on it, so the rows came back
                                 # labelled "goal 1".
                                 "goals": payoff[0].payload.get("goals") or []}
            if select:
                out["select"] = _plain(select[0].payload)
        precedent = result.get("precedent") or {}
        if precedent.get("precedent"):
            out["precedent"] = _case(precedent["precedent"])
        return out

    def clarify(self, body: dict) -> dict:
        """Questions until the first step is actionable -- without committing.

        `ask` runs this too, but only as a gate it then passes through. Exposing
        it alone means a task can be sharpened before anything is distilled, which
        is the cheap half of the loop.
        """
        task = (body.get("task") or "").strip()
        if not task:
            return {"error": "a task is required"}
        answers = body.get("answers") or {}
        ask_fn = (lambda questions: dict(answers)) if answers else None
        out = self.agent.clarifier.clarify(task, ask=ask_fn,
                                           max_rounds=int(body.get("rounds", 2)))
        self.agent.save()
        return {"task": task, "actionable": out.actionable, "reason": out.reason,
                "rounds": out.rounds, "frame": frame_json(out.frame, out.plan),
                "recall": self.agent.recall_for(task),
                "questions": [{"gap": q.gap, "text": q.text, "unblocks": q.unblocks,
                               "value": q.value, "known": q.answered_by_memory}
                              for q in out.questions]}

    def mcp_attach(self, body: dict) -> dict:
        """Attach or detach an MCP server.

        This starts a subprocess of the caller's choosing, which is the most
        powerful thing any endpoint here does -- so it is the clearest case for
        the cross-site refusal in `Handler`: a page you merely visited must never
        reach it.
        """
        action = (body.get("action") or "add").strip()
        name = (body.get("name") or "").strip()
        if not name:
            return {"error": "a server name is required"}
        if action == "remove":
            return {"removed": self.agent.mcp.detach(name), "servers": self.agent.mcp.names()}
        command = body.get("command") or []
        if isinstance(command, str):
            command = command.split()
        command = [str(part) for part in command if str(part).strip()]
        if not command:
            return {"error": "a command is required, e.g. [\"npx\", \"-y\", \"@mcp/fs\", \"/tmp\"]"}
        self.agent.mcp.attach(name, command)
        report = self.agent.mcp.discover(name)
        self.agent.save()
        failed = [f for f in report["failed"] if f["server"] == name]
        return {"attached": name, "tools": report["servers"].get(name, []),
                "failed": failed, "servers": self.agent.mcp.names()}

    def transcribe_audio(self, raw: bytes) -> dict:
        from .speech import SpeechError, transcribe
        try:
            return transcribe(raw)
        except SpeechError as exc:
            return {"error": str(exc)}

    def grade(self, body: dict) -> dict:
        try:
            return {"graded": self.agent.ask_user_grade(
                body["id"], float(body["score"]), body.get("comment", ""))}
        except (KeyError, ValueError) as exc:
            return {"error": str(exc)}

    def forge(self, body: dict) -> dict:
        goal = (body.get("goal") or "").strip()
        if not goal:
            return {"error": "a goal is required"}
        spec = self.agent.toolsmith.forge(goal)
        if spec is None:
            return {"error": "no tool could be written for that goal"}
        grade = self.agent.toolsmith.validate(spec, self.agent.toolbox)
        kept = self.agent.toolsmith.register(spec)
        self.agent.save()
        return {"name": spec.name, "registered": kept, "grade": grade_json(grade),
                "source": spec.source, "tests": spec.tests}

    def invoke(self, body: dict) -> dict:
        name = body.get("name") or ""
        return {"result": _plain(self.agent.toolbox.invoke(
            name, body.get("args") or [], body.get("kwargs") or {}))}

    def explore(self, body: dict) -> dict:
        steps = max(1, min(20, int(body.get("steps", 3))))
        runs = self.agent.explore(steps=steps, seed=body.get("seed"))
        strategy = dict(zip(self.agent.explorer.regret.actions,
                            [round(x, 4) for x in self.agent.explorer.regret.average_strategy()]))
        return {"experiments": [{"idea": e.idea.text, "origin": e.idea.origin,
                                 "ran": e.ran, "passed": e.passed, "grade": e.grade,
                                 "detail": e.detail, "seconds": round(e.seconds, 3)}
                                for e in runs],
                "strategy": strategy}

    def seed(self, _body: dict) -> dict:
        from .seed import plant
        report = plant(self.agent.toolsmith, self.agent.toolbox)
        self.agent.save()
        return {"planted": report["planted"], "rejected": report["rejected"],
                "already": report["already"], "skipped": report["skipped"]}

    def compress(self, body: dict) -> dict:
        report = self.agent.compress(dry_run=bool(body.get("dry_run", True)))
        return {"report": {k: v for k, v in report.items() if k != "digests"},
                "digests": report.get("digests", [])[:10]}

    def upgrade(self, body: dict) -> dict:
        return {"result": self.agent.upgrade(trials=int(body.get("trials", 6)))}


def _edge(t, policy) -> dict:
    """A neighbour in the link graph: enough to render and click, no more."""
    return {"id": t.id, "kind": t.kind, "text": t.text, "grade": t.mean_grade,
            "verified": t.verified, "credibility": round(t.credibility(policy), 4)}


def _case(p) -> dict:
    return {"problem": p.case.problem, "solution": p.case.solution, "grade": p.case.grade,
            "via": p.case.via, "similarity": round(p.similarity, 4),
            "score": round(p.score, 4), "worked": p.worked, "novel": p.novel_terms[:8]}


GET_ROUTES = {"state", "memory", "recall", "frame", "tools", "cases", "ideas",
              "journal", "lineage", "trace", "speech", "mcp", "concepts"}
POST_ROUTES = {"ask", "grade", "forge", "invoke", "explore", "seed", "compress",
               "upgrade", "clarify", "mcp_attach"}
#: Routes that stream rather than answering once. Handled apart from the JSON
#: table because the response shape, the content type and the lifetime all differ.
STREAM_ROUTES = {"ask/stream", "auto/stream"}
#: Content types a POST may carry, per route. Everything else must be JSON: a
#: type outside this set is not a CORS "simple request", so a cross-origin POST
#: has to be preflighted, and the preflight finds no CORS headers here.
POST_CONTENT = {"transcribe": ("audio/wav", "audio/wave", "audio/x-wav", "audio/webm")}


class Handler(BaseHTTPRequestHandler):
    api: Api = None                      # set by serve()
    server_version = "distil"

    def log_message(self, fmt, *args):   # one line per request, not three
        print(f"  {self.address_string()} {fmt % args}")

    # -- plumbing ------------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # The UI is same-origin and served from here; nothing external should be
        # able to frame it or load it into another page.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _local(self) -> bool:
        """Refuse a Host header that is not loopback.

        This is what stops a page in the user's browser from resolving an
        attacker-controlled name to 127.0.0.1 and driving this server on their
        behalf. It is not authentication and is not offered as any.
        """
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in LOOPBACK or host == ""

    def _same_site(self) -> bool:
        """Refuse a request a browser tells us came from another page.

        `Sec-Fetch-Site` is set by the browser itself and cannot be forged from
        script, so when it says `cross-site` the request came from a page the
        user merely visited. Non-browser clients (curl, the tests) send no such
        header and are unaffected -- this stops the browser attack, not everyone.
        """
        site = self.headers.get("Sec-Fetch-Site")
        return site is None or site in ("same-origin", "same-site", "none")

    def _unforgeable_post(self, route: str = "") -> str | None:
        """Why this POST should be refused, or None if it should not be.

        Two conditions, both about browsers. An `Origin` that is not loopback
        means the request came from another page even if `Host` is honest. And
        requiring `application/json` means a cross-origin POST can no longer be
        a "simple request": the browser has to preflight it, the preflight finds
        no CORS headers here, and the real request is never sent. A form or a
        `text/plain` fetch -- the shapes that skip the preflight -- are exactly
        what this rejects.
        """
        origin = self.headers.get("Origin")
        if origin and origin != "null":
            host = urlparse(origin).hostname
            if host not in LOOPBACK:
                return "cross-origin request refused"
        allowed = POST_CONTENT.get(route, ("application/json",))
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype not in allowed:
            return f"Content-Type must be one of: {', '.join(allowed)}"
        return None

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:
        if not self._local():
            return self._json({"ok": False, "error": "non-loopback Host refused"}, 403)
        if not self._same_site():
            return self._json({"ok": False, "error": "cross-site request refused"}, 403)
        url = urlparse(self.path)
        path = unquote(url.path)
        if path.startswith("/api/"):
            name = path[5:].strip("/")
            if name == "auto/stream":
                return self._stream_auto(parse_qs(url.query))
            if name in STREAM_ROUTES:
                return self._stream_ask(parse_qs(url.query))
            if name not in GET_ROUTES:
                return self._json({"ok": False, "error": f"no such endpoint: {name}"}, 404)
            return self._run(lambda: getattr(self.api, name)(parse_qs(url.query)))
        return self._static(path)

    def do_POST(self) -> None:
        if not self._local():
            return self._json({"ok": False, "error": "non-loopback Host refused"}, 403)
        if not self._same_site():
            return self._json({"ok": False, "error": "cross-site request refused"}, 403)
        path = unquote(urlparse(self.path).path)
        if not path.startswith("/api/"):
            return self._json({"ok": False, "error": "not found"}, 404)
        name = path[5:].strip("/")
        if name not in POST_ROUTES and name != "transcribe":
            return self._json({"ok": False, "error": f"no such endpoint: {name}"}, 404)
        refusal = self._unforgeable_post(name)
        if refusal:
            return self._json({"ok": False, "error": refusal}, 403)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
        except ValueError as exc:
            return self._json({"ok": False, "error": f"bad Content-Length: {exc}"}, 400)
        if name == "transcribe":
            # Audio, not JSON. Read whole -- speech.check_wav bounds the size
            # before anything is written to disk or handed to a subprocess.
            return self._run(lambda: self.api.transcribe_audio(raw))
        try:
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json({"ok": False, "error": f"bad request body: {exc}"}, 400)
        return self._run(lambda: getattr(self.api, name)(body))

    def _run(self, call) -> None:
        try:
            with self.api.lock:
                payload = call()
        except Exception as exc:                      # degrade, never 500 blankly
            traceback.print_exc()
            return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
        if isinstance(payload, dict) and payload.get("error"):
            return self._json({"ok": False, **payload}, 400)
        self._json({"ok": True, **payload})

    def _stream_ask(self, q) -> None:
        """`GET /api/ask/stream?task=...` -- the chain as it is thought.

        Reasoning that arrives all at once, a minute later, reads as a result.
        The whole point of a typed chain is that it can be watched, so each step
        is sent the moment it exists.

        The agent runs on a worker thread and the request thread drains a queue.
        That split is what lets a closed browser be noticed: writing to a dead
        socket raises here, the queue stops being drained, and `Chain.watch`
        drops the observer on its next failure rather than the run continuing to
        push into a queue nobody reads.

        The agent lock is held for the whole run, as everywhere else -- `Distil`
        holds mutable memory and a workshop directory, so exactly one call runs at
        a time. A long ask therefore blocks other endpoints; for a single-user
        local tool that is the right trade against making the agent re-entrant.
        """
        task = (q.get("task") or [""])[0].strip()
        if not task:
            return self._json({"ok": False, "error": "a task is required"}, 400)
        try:
            answers = json.loads((q.get("answers") or ["{}"])[0])
            if not isinstance(answers, dict):
                answers = {}
        except json.JSONDecodeError:
            answers = {}

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()

        steps: "queue.Queue" = queue.Queue()
        DONE = object()

        def work():
            try:
                with self.api.lock:
                    ask_fn = (lambda questions: dict(answers)) if answers else None
                    result = self.api.ask({"task": task, "answers": answers,
                                           "_observer": steps.put,
                                           "_ask": ask_fn})
                steps.put(("result", result))
            except Exception as exc:
                traceback.print_exc()
                steps.put(("error", {"error": f"{type(exc).__name__}: {exc}"}))
            finally:
                steps.put(DONE)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self._send_event("open", {"task": task})
        try:
            while True:
                try:
                    item = steps.get(timeout=15.0)
                except queue.Empty:
                    # A comment frame: proves the connection is alive through a
                    # long step without inventing a step that did not happen.
                    self.wfile.write(b": still thinking\n\n")
                    self.wfile.flush()
                    continue
                if item is DONE:
                    break
                if isinstance(item, tuple):
                    self._send_event(item[0], item[1])
                else:
                    self._send_event("step", {"kind": item.kind, "text": item.text,
                                              "payload": _plain(item.payload)})
        except (BrokenPipeError, ConnectionResetError):
            return                     # the browser left; the run finishes anyway
        try:
            self._send_event("done", {})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream_auto(self, q) -> None:
        """`GET /api/auto/stream?cycles=N` -- the system running itself.

        Same shape as the ask stream and for the same reason: an autonomous loop
        you cannot watch is one you have to trust. Every cycle is reported with
        what it chose, what it found and what that was worth in bits.

        Stopping is by disconnection. The browser closes the EventSource, the
        next write fails, and the stop flag is set -- which is the only signal
        that survives the server being a plain `http.server` with no session of
        its own. The loop finishes the cycle it is in rather than being killed
        mid-forge, so memory is never left half-written.
        """
        try:
            cycles = max(0, min(10000, int((q.get("cycles") or ["0"])[0])))
        except ValueError:
            cycles = 0

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()

        from .auto import Auto
        done: "queue.Queue" = queue.Queue()
        stopping = threading.Event()
        DONE = object()

        def work():
            try:
                with self.api.lock:
                    loop = Auto(self.api.agent, stop=stopping.is_set)
                    self.api.auto = loop
                    loop.run(cycles=cycles, on_cycle=lambda c: done.put((loop, c)))
            except Exception as exc:
                traceback.print_exc()
                done.put(("error", {"error": f"{type(exc).__name__}: {exc}"}))
            finally:
                done.put(DONE)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            self._send_event("open", {"cycles": cycles})
            while True:
                try:
                    item = done.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": still running\n\n")
                    self.wfile.flush()
                    continue
                if item is DONE:
                    break
                if isinstance(item, tuple) and item[0] == "error":
                    self._send_event("error", item[1])
                    continue
                loop, cycle = item
                self._send_event("cycle", {**cycle.to_json(),
                                           "strategy": loop.strategy(),
                                           "stats": self.api.agent.stats()})
        except (BrokenPipeError, ConnectionResetError):
            stopping.set()               # the tab closed: finish this cycle, then stop
            return
        try:
            self._send_event("done", {})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_event(self, name: str, data: dict) -> None:
        body = json.dumps(data)
        self.wfile.write(f"event: {name}\ndata: {body}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _static(self, path: str) -> None:
        target = (WEB / (path.lstrip("/") or "index.html")).resolve()
        try:
            # `resolve()` collapses "..", and this confirms the result is still
            # inside web/ -- without it, /../../etc/passwd is a file read.
            target.relative_to(WEB.resolve())
        except ValueError:
            return self._json({"ok": False, "error": "forbidden"}, 403)
        if not target.is_file():
            return self._json({"ok": False, "error": "not found"}, 404)
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), kind)


def serve(host: str = "127.0.0.1", port: int = 8765, agent: Distil | None = None) -> None:
    Handler.api = Api(agent or Distil())
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"  distil ui   http://{host}:{port}")
    print(f"  home        {Handler.api.agent.workspace}")
    print(f"  provider    {Handler.api.agent.provider.name}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("\n  WARNING: bound to a non-loopback address. This UI drives an agent that\n"
              "  writes files and executes generated code. Anyone who can reach this port\n"
              "  can do both. There is no authentication.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        Handler.api.agent.save()
        Handler.api.agent.mcp.close()
        httpd.server_close()
