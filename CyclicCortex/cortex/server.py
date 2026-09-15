"""The HTTP API, and the static files of the page in front of it.

``python3 -m cortex.cli serve`` puts the whole of V1 behind a browser: the
similarity graph and its regions drawn rather than printed, training you can
watch a curve of and stop, and a board you can sit down at against the cortex --
or watch two cortexes play each other, or a cortex play the engine.

Everything the page can do, the command line could already do. Nothing here
invents a capability: `/api/jobs` runs `Cortex.train`, `Cortex.selfplay` and
`Cortex.distill`, `/api/match` runs `cortex/arena.py`, `/api/model` reports
`Cortex.stats` and the graph. What the browser adds is the part that only works
live -- seeing `p_valid` and `grade` for every legal move while it is your turn.

`http.server` from the standard library, because the package has no
dependencies and this is not the place to acquire one.

Routes
------
``GET  /api/info``            what this build supports; the defaults the page builds its controls from
``GET  /api/state``           models, jobs and matches, in one poll
``GET  /api/model``           one model: regions, vocabularies, network shapes, the similarity matrix, the routing log
``POST /api/model/build``     a fresh cortex over a list of games
``POST /api/model/add``       admit a game to an existing cortex, optionally rehearsing the region it joins
``POST /api/model/copy``      fork a model under a new name -- which is how model-vs-model gets two of them
``POST /api/model/delete``    forget a model
``GET  /api/checkpoints``     what is in the checkpoint directory
``POST /api/model/save``      write a checkpoint
``POST /api/model/load``      read one back under a name
``GET  /api/jobs``            every job, newest first
``POST /api/jobs``            start one: train, selfplay, rehearse, distill, evaluate, benchmark, credit
``GET  /api/job``             one job, with the events after a cursor
``POST /api/job/stop``        ask a job to stop at the end of its chunk
``GET  /api/matches``         every match
``POST /api/match``           start one, with a player on each side
``GET  /api/match``           a match: board, legal moves, history, and what the model thinks of each move
``POST /api/match/move``      a human's move
``POST /api/match/step``      advance one ply by whoever is not human
``POST /api/match/run``       advance until a human is on the move or the game ends
``POST /api/match/delete``    forget a match

The socket defaults to 127.0.0.1. Training is unauthenticated CPU on demand, so
binding this to a public interface hands anyone who can reach it the machine;
``--host`` has to be given deliberately.
"""
import json, os, posixpath, socket, threading, time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from cortex import arena, trainer
from cortex.arena import Match, Player
from cortex.games import ALL
from cortex.trainer import Registry, Runner

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8099
MAX_BODY = 8 * 1024 * 1024
MAX_MATCHES = 60

_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".json": "application/json",
          ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
          ".map": "application/json", ".txt": "text/plain; charset=utf-8"}


class ServerError(Exception):
    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = int(status)


def frontend_dir(path=None):
    if path: return os.path.abspath(path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


# ------------------------------------------------------------------- the app

class App:
    """Everything the server owns: the models, the jobs, the matches.

    One object rather than module globals, so a test can make its own and two
    of them can exist in one process."""

    def __init__(self, checkpoint_dir=None, games=("chess", "checkers", "go", "sudoku"),
                 seed=0, nh=24, tau_new=0.60):
        self.reg = Registry()
        self.runner = Runner(self.reg)
        self.matches = {}
        self.order = []
        self.lock = threading.Lock()
        self.started = time.time()
        self.checkpoint_dir = os.path.abspath(
            checkpoint_dir or os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "checkpoints"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.reg.put("main", trainer.build(list(games), tau_new=tau_new, seed=seed, nh=nh),
                     note="built at startup")

    # -------------------------------------------------------------- helpers
    def checkpoint_path(self, name):
        """Inside the checkpoint directory or nowhere. A file name arrives from
        the browser, and `../` in one of those is how a page ends up reading
        /etc/passwd."""
        base = os.path.basename(str(name or "").strip())
        if not base or base.startswith("."):
            raise ServerError("a checkpoint needs a file name")
        if not base.endswith((".json", ".json.gz")): base += ".json.gz"
        full = os.path.abspath(os.path.join(self.checkpoint_dir, base))
        if os.path.dirname(full) != self.checkpoint_dir:
            raise ServerError("checkpoints stay in the checkpoint directory")
        return full

    def checkpoints(self):
        out = []
        for f in sorted(os.listdir(self.checkpoint_dir)):
            if not f.endswith((".json", ".json.gz")): continue
            p = os.path.join(self.checkpoint_dir, f)
            out.append({"file": f, "bytes": os.path.getsize(p),
                        "modified": os.path.getmtime(p)})
        return sorted(out, key=lambda d: -d["modified"])

    def match(self, mid):
        m = self.matches.get(mid)
        if m is None: raise ServerError(f"no match {mid!r}", HTTPStatus.NOT_FOUND)
        return m

    def add_match(self, m):
        with self.lock:
            self.matches[m.id] = m
            self.order.append(m.id)
            for old in self.order[:-MAX_MATCHES]: self.matches.pop(old, None)
            self.order = [i for i in self.order if i in self.matches]
        return m

    # ---------------------------------------------------------------- views
    def model_view(self, name):
        c = self.reg.require(name)
        with self.reg.lock:
            g = c.graph
            regions = []
            for r in c.regions:
                slots = [{"mechanic": m, "offset": r.vocab.slot[m][0],
                          "width": r.vocab.slot[m][1]} for m in r.vocab.order]
                net = r.net
                regions.append({
                    "id": r.id, "games": [x.name for x in r.games],
                    "vocab_width": r.vocab.width, "slots": slots,
                    "mechanics": sorted(r.mech), "rule": r.rule,
                    "reputation": round(r.reputation(), 4),
                    "plays": r.plays, "hits": r.hits,
                    "shape": r.shape(),
                    "seen": net.seen if net else 0,
                    "loss": round(net.loss, 6) if net else None,
                    "growth": (net.log or []) if net else [],
                    "trained": bool(net and net.seen > 0),
                })
            return {"name": name, "meta": self.reg.meta.get(name, {}),
                    "tau_new": c.tau_new, "seed": c.seed, "nh": c.nh,
                    "regions": regions, "log": c.log,
                    "route_of": dict(c.route_of),
                    "stats": c.stats(),
                    "graph": {"names": list(g.names), "matrix": g.matrix(),
                              "tau": g.tau, "violations": g.check_metric(),
                              "mechanics": {n: sorted(m) for n, m in
                                            zip(g.names, g.mech)}}}

    def info(self):
        from cortex import stockfish as sfmod
        return {
            "name": "CyclicCortex", "version": "V1",
            "games": [{"name": n, "mechanics": sorted(g.mechanics),
                       "spec": g.spec, "sides": list(arena.SIDES[n]),
                       "side_names": arena.SIDE_NAMES[n],
                       "cortex_side": arena.CORTEX_SIDE[n],
                       "two_player": n != "sudoku",
                       "ply_cap": arena.PLY_CAP[n]} for n, g in sorted(ALL.items())],
            "player_kinds": list(Player.KINDS),
            "job_kinds": list(trainer.KINDS),
            "rules": sorted(__import__("cortex.cortex", fromlist=["Cortex"]).Cortex.RULES),
            "stockfish": {"available": sfmod.available(), "path": sfmod.find()},
            "checkpoint_dir": self.checkpoint_dir,
            "started": self.started,
        }

    def state(self):
        return {"models": self.reg.summary(), "jobs": self.runner.list(),
                "matches": [self.matches[i].to_dict(analysis=False, history=1)
                            for i in reversed(self.order) if i in self.matches],
                "checkpoints": self.checkpoints()}


# ----------------------------------------------------------------- the routes

def _players(game, spec):
    """`{"w": {...}, "b": {...}}`, or the two shorthands the page uses most:
    `{"cortex": "w"}`-style side keys are already explicit, so the only shorthand
    is a missing side, which becomes a human."""
    sides = arena.SIDES[game]
    spec = spec or {}
    out = {}
    for s in sides:
        out[s] = Player.from_dict(spec.get(s) or {"kind": "human"})
    return out


class Routes:
    def __init__(self, app): self.app = app

    # ---- GET
    def get_info(self, q): return self.app.info()

    def get_state(self, q): return self.app.state()

    def get_model(self, q): return self.app.model_view(q.get("name", ["main"])[0])

    def get_checkpoints(self, q): return {"checkpoints": self.app.checkpoints(),
                                          "dir": self.app.checkpoint_dir}

    def get_jobs(self, q): return {"jobs": self.app.runner.list()}

    def get_job(self, q):
        jid = q.get("id", [""])[0]
        j = self.app.runner.get(jid)
        if j is None: raise ServerError(f"no job {jid!r}", HTTPStatus.NOT_FOUND)
        since = int(q.get("since", ["0"])[0])
        return {"job": j.to_dict(since=since)}

    def get_matches(self, q):
        return {"matches": [self.app.matches[i].to_dict(analysis=False, history=1)
                            for i in reversed(self.app.order) if i in self.app.matches]}

    def get_match(self, q):
        m = self.app.match(q.get("id", [""])[0])
        analysis = q.get("analysis", ["1"])[0] not in ("0", "false")
        with self.app.reg.lock:
            return {"match": m.to_dict(analysis=analysis)}

    # ---- POST
    def post_model_build(self, b):
        name = b.get("name", "main")
        games = b.get("games") or ["chess", "checkers", "go", "sudoku"]
        c = trainer.build(games, tau_new=b.get("tau_new", 0.60),
                          seed=b.get("seed", 0), nh=b.get("nh", 24))
        self.app.reg.put(name, c, note="built from the page")
        return {"model": self.app.model_view(name)}

    def post_model_add(self, b):
        name = b.get("name", "main")
        game = b.get("game")
        if game not in ALL: raise ServerError(f"unknown game {game!r}")
        c = self.app.reg.require(name)
        if game in c.route_of: raise ServerError(f"{game!r} is already in {name!r}")
        with self.app.reg.lock:
            if b.get("rehearse"):
                r, info = c.add_game_and_rehearse(ALL[game],
                                                  episodes=int(b.get("episodes", 200)))
            else:
                r, info = c.add_game(ALL[game]), {"rehearsed": [], "samples": 0}
        return {"model": self.app.model_view(name), "region": r.id, "rehearse": info}

    def post_model_copy(self, b):
        """Fork a model. Model-vs-model needs two cortexes, and the interesting
        pairing is a model against ITS OWN EARLIER SELF -- fork, train one side,
        then play them. A checkpoint round trip is the deep copy."""
        src, dst = b.get("from", "main"), b.get("to")
        if not dst: raise ServerError("copy needs a destination name")
        c = self.app.reg.require(src)
        tmp = self.app.checkpoint_path(f"_fork_{int(time.time()*1000)}.json.gz")
        try:
            with self.app.reg.lock: trainer.checkpoint.save(c, tmp)
            trainer.load(self.app.reg, dst, tmp, note=f"forked from {src!r}")
        finally:
            try: os.remove(tmp)
            except OSError: pass
        return {"models": self.app.reg.summary(), "model": self.app.model_view(dst)}

    def post_model_delete(self, b):
        name = b.get("name")
        if name == "main": raise ServerError("main cannot be deleted; rebuild it instead")
        self.app.reg.drop(name)
        return {"models": self.app.reg.summary()}

    def post_model_save(self, b):
        name = b.get("name", "main")
        path = self.app.checkpoint_path(b.get("file") or f"{name}.json.gz")
        n = trainer.save(self.app.reg, name, path)
        return {"file": os.path.basename(path), "bytes": n,
                "checkpoints": self.app.checkpoints()}

    def post_model_load(self, b):
        path = self.app.checkpoint_path(b.get("file"))
        if not os.path.exists(path):
            raise ServerError(f"no checkpoint {os.path.basename(path)!r}",
                              HTTPStatus.NOT_FOUND)
        name = b.get("name", "main")
        trainer.load(self.app.reg, name, path)
        return {"model": self.app.model_view(name), "models": self.app.reg.summary()}

    def post_jobs(self, b):
        kind = b.get("kind")
        try:
            j = self.app.runner.start(kind, b)
        except RuntimeError as e:
            raise ServerError(str(e), HTTPStatus.CONFLICT)
        return {"job": j.to_dict(since=0)}

    def post_job_stop(self, b):
        try:
            j = self.app.runner.stop(b.get("id"))
        except LookupError as e:
            raise ServerError(str(e), HTTPStatus.NOT_FOUND)
        return {"job": j.to_dict()}

    def post_match(self, b):
        game = b.get("game", "chess")
        if game not in ALL: raise ServerError(f"unknown game {game!r}")
        players = _players(game, b.get("players"))
        for s, p in players.items():
            if p.kind == "cortex": self.app.reg.require(p.model)
            if p.kind == "stockfish" and not arena.stockfish_player(p.skill, p.depth):
                raise ServerError("stockfish is not installed")
        m = Match(game, players, resolve=self.app.reg.resolve(),
                  seed=int(b.get("seed", int(time.time()) % 10000)),
                  max_plies=int(b.get("max_plies", 0)) or None,
                  spectator=b.get("spectator", "main"))
        self.app.add_match(m)
        if b.get("run"):
            with self.app.reg.lock: m.run()
        return {"match": m.to_dict()}

    def post_match_move(self, b):
        m = self.app.match(b.get("id"))
        try:
            m.play(b.get("key"))
        except ValueError as e:
            raise ServerError(str(e))
        with self.app.reg.lock:
            if b.get("reply", True): m.run()
            return {"match": m.to_dict()}

    def post_match_step(self, b):
        m = self.app.match(b.get("id"))
        n = max(1, min(400, int(b.get("n", 1))))
        out = []
        with self.app.reg.lock:
            for _ in range(n):
                rec = m.step()
                if rec is None: break
                out.append(rec)
            return {"match": m.to_dict(), "moves": out}

    def post_match_run(self, b):
        m = self.app.match(b.get("id"))
        with self.app.reg.lock:
            out = m.run(limit=int(b.get("limit", 400)))
            return {"match": m.to_dict(), "moves": out}

    def post_match_delete(self, b):
        mid = b.get("id")
        with self.app.lock:
            self.app.matches.pop(mid, None)
            self.app.order = [i for i in self.app.order if i in self.app.matches]
        return {"ok": True}


GET_ROUTES = {"/api/info": "get_info", "/api/state": "get_state",
              "/api/model": "get_model", "/api/checkpoints": "get_checkpoints",
              "/api/jobs": "get_jobs", "/api/job": "get_job",
              "/api/matches": "get_matches", "/api/match": "get_match"}

POST_ROUTES = {"/api/model/build": "post_model_build", "/api/model/add": "post_model_add",
               "/api/model/copy": "post_model_copy", "/api/model/delete": "post_model_delete",
               "/api/model/save": "post_model_save", "/api/model/load": "post_model_load",
               "/api/jobs": "post_jobs", "/api/job/stop": "post_job_stop",
               "/api/match": "post_match", "/api/match/move": "post_match_move",
               "/api/match/step": "post_match_step", "/api/match/run": "post_match_run",
               "/api/match/delete": "post_match_delete"}


# ---------------------------------------------------------------- the handler

class Handler(BaseHTTPRequestHandler):
    server_version = "CyclicCortex/1"
    protocol_version = "HTTP/1.1"

    @property
    def routes(self): return self.server.routes

    def log_message(self, fmt, *args):
        if getattr(self.server, "quiet", False): return
        super().log_message(fmt, *args)

    # ------------------------------------------------------------- replying
    def _send(self, body, status=HTTPStatus.OK, ctype="application/json"):
        if isinstance(body, str): body = body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD": self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(json.dumps(obj, default=_plain), status)

    def _fail(self, message, status=HTTPStatus.BAD_REQUEST):
        self._json({"error": str(message), "status": int(status)}, status)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY: raise ServerError("request body too large",
                                           HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if n <= 0: return {}
        raw = self.rfile.read(n)
        try:
            out = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ServerError(f"body is not JSON: {e}")
        if not isinstance(out, dict): raise ServerError("body must be a JSON object")
        return out

    # ------------------------------------------------------------- the verbs
    def do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        if path.startswith("/api/"):
            name = GET_ROUTES.get(path)
            if name is None: return self._fail(f"no route {path}", HTTPStatus.NOT_FOUND)
            try:
                return self._json(getattr(self.routes, name)(parse_qs(u.query)))
            except ServerError as e: return self._fail(e, e.status)
            except LookupError as e: return self._fail(e, HTTPStatus.NOT_FOUND)
            except Exception as e: return self._fail(f"{type(e).__name__}: {e}",
                                                     HTTPStatus.INTERNAL_SERVER_ERROR)
        return self._static(path)

    do_HEAD = do_GET

    def do_POST(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        name = POST_ROUTES.get(path)
        if name is None: return self._fail(f"no route {path}", HTTPStatus.NOT_FOUND)
        try:
            return self._json(getattr(self.routes, name)(self._body()))
        except ServerError as e: return self._fail(e, e.status)
        except LookupError as e: return self._fail(e, HTTPStatus.NOT_FOUND)
        except ValueError as e: return self._fail(e)
        except Exception as e: return self._fail(f"{type(e).__name__}: {e}",
                                                 HTTPStatus.INTERNAL_SERVER_ERROR)

    # ------------------------------------------------------- the static page
    def _static(self, path):
        """The page and its two files, and nothing outside their directory.

        The path is already percent-decoded, so `..` normalises rather than
        hiding, and the resolved file is checked against the root by
        `commonpath` -- a prefix test on strings would let a sibling directory
        whose name merely STARTS with the root's name through."""
        root = os.path.abspath(self.server.frontend)
        rel = posixpath.normpath(path.lstrip("/"))
        if rel in (".", "", "/"): rel = "index.html"
        full = os.path.abspath(os.path.join(root, *rel.split("/")))
        try:
            inside = os.path.commonpath([full, root]) == root
        except ValueError:                      # different drives, on Windows
            inside = False
        if not inside: return self._fail("outside the frontend directory",
                                         HTTPStatus.FORBIDDEN)
        if os.path.isdir(full): full = os.path.join(full, "index.html")
        if not os.path.isfile(full):
            return self._fail(f"no file {rel!r}", HTTPStatus.NOT_FOUND)
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as f: data = f.read()
        return self._send(data, ctype=_TYPES.get(ext, "application/octet-stream"))


def _plain(o):
    """Anything json does not know how to write is written as text rather than
    failing the whole response -- a frozenset of mechanics, most often."""
    if isinstance(o, (set, frozenset, tuple)): return sorted(o) if isinstance(o, (set, frozenset)) else list(o)
    return str(o)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(host=DEFAULT_HOST, port=DEFAULT_PORT, app=None, frontend=None,
                quiet=False):
    app = app or App()
    srv = Server((host, port), Handler)
    srv.app = app
    srv.routes = Routes(app)
    srv.frontend = frontend_dir(frontend)
    srv.quiet = quiet
    return srv


def run_server(host=DEFAULT_HOST, port=DEFAULT_PORT, app=None, frontend=None,
               quiet=False, open_browser=False):
    srv = make_server(host, port, app, frontend, quiet)
    shown = srv.server_address[0]
    if shown in ("0.0.0.0", "::"): shown = socket.gethostname()
    url = f"http://{shown}:{srv.server_address[1]}/"
    print(f"\n  CyclicCortex is at {url}")
    print(f"  checkpoints: {srv.app.checkpoint_dir}")
    print(f"  models: {', '.join(srv.app.reg.names())}")
    print("  ctrl-c to stop\n")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        srv.shutdown(); srv.server_close(); arena.close_engines()
    return srv
