"""Training as something you can watch.

`cli.play` trains and prints when it is done. A page cannot wait: it needs the
run broken into pieces, a number out of each piece, and a way to stop. So every
job here is the SAME call the command line makes, in a loop over chunks, with an
evaluation after each chunk.

That is not only a progress bar. The README reports that 400, 1200 and 3000
episodes of chess all land near 0.80 -- a claim about a CURVE, made from three
end points. Chunking produces the curve itself, and the page draws it.

  train      `Cortex.train` per chunk, `Cortex.evaluate` after each
  selfplay   `Cortex.selfplay` per chunk, with the win/loss split per chunk
  rehearse   `Cortex.rehearse` -- interleave a region's games after a join
  distill    `Cortex.distill` from Stockfish, when it is installed
  evaluate   one measurement, no training
  benchmark  N matches through `cortex/arena.py`, any pairing of players
  credit     coverage, exact Shapley over regions, and the auction

Concurrency is deliberately dull. One lock guards every model; a chunk takes it,
trains, and gives it back, so a match being drawn in the browser waits for one
chunk rather than for the whole run. Two jobs that would both WRITE to the same
model are refused -- there is no merge for two gradient streams into one network,
and pretending otherwise would corrupt the weights silently.
"""
import random, threading, time, traceback

from cortex import checkpoint, credit as credit_mod, routing
from cortex.arena import BOT, Match, Player, stockfish_player
from cortex.cortex import Cortex
from cortex.games import ALL

KINDS = ("train", "selfplay", "rehearse", "distill", "evaluate", "benchmark",
         "credit", "transfer")

# Jobs that change weights. Two of these on one model at the same time is refused.
WRITES = ("train", "selfplay", "rehearse", "distill")


def opponent_for(game, spec):
    """The opponent a training run plays against, from a small JSON spec.

    `None` is a real choice, not a missing one: `Cortex._states` then draws
    random positions instead of game positions, which is a different -- and
    measurably worse -- training distribution."""
    spec = spec or {}
    kind = spec.get("kind", "bot")
    if kind in (None, "none", ""): return None
    if kind == "stockfish":
        if game != "chess": raise ValueError("stockfish plays chess only")
        sf = stockfish_player(int(spec.get("skill", 5)), int(spec.get("depth", 6)))
        if sf is None: raise LookupError("stockfish is not installed")
        return lambda s, d, r: sf.best(s, depth=max(1, d))
    if game not in BOT: raise ValueError(f"no opponent for {game!r}")
    return BOT[game]


class Registry:
    """The named cortexes, and the lock that keeps a job from mutating one while
    a match reads it.

    TWO locks, deliberately. `lock` is the COMPUTE lock: a training chunk holds
    it, and a chunk can run for seconds. `_names` guards only the dictionary --
    which name maps to which object -- and is never held while anything thinks.

    Conflating them was a real defect and not a theoretical one: looking a model
    up took the compute lock, so STARTING a second job blocked until the first
    job's chunk finished, and the page appeared to hang on the button. The fix
    is that finding a model and using one are different questions.
    """

    def __init__(self):
        self.lock = threading.RLock()          # held while a model is read or written
        self._names = threading.RLock()        # held only to touch the dictionaries
        self.models = {}
        self.meta = {}

    def put(self, name, cortex, note=""):
        with self._names:
            self.models[name] = cortex
            self.meta[name] = {"name": name, "created": time.time(), "note": note}
        return cortex

    def get(self, name):
        with self._names: return self.models.get(name)

    def require(self, name):
        c = self.get(name)
        if c is None: raise LookupError(f"no model named {name!r}")
        return c

    def drop(self, name):
        with self._names:
            self.models.pop(name, None); self.meta.pop(name, None)

    def names(self):
        with self._names: return sorted(self.models)

    def resolve(self):
        """A callable for `arena.Match`, so the arena never imports the registry."""
        return self.get

    def summary(self):
        """Scalars only, under the name lock -- so the page's every-few-seconds
        poll never waits on a training chunk."""
        with self._names:
            out = []
            for n in sorted(self.models):
                c = self.models[n]
                st = c.stats()
                out.append({"name": n, **self.meta.get(n, {}),
                            "games": sorted(c.route_of),
                            "regions": st["regions"],
                            "trained": any(r.net is not None and r.net.seen > 0
                                           for r in c.regions)})
            return out


class Job:
    """One run. Events accumulate; the page asks for everything after a cursor."""

    def __init__(self, jid, kind, params):
        self.id, self.kind, self.params = jid, kind, params
        self.status = "queued"
        self.events = []
        self.result = None
        self.error = None
        self.created = time.time()
        self.started = self.finished = None
        self.progress = {"done": 0, "total": int(params.get("total") or 0)}
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def emit(self, type, **kw):
        with self.lock:
            e = {"seq": len(self.events), "at": round(time.time() - (self.started or self.created), 3),
                 "type": type, **kw}
            self.events.append(e)
            return e

    def log(self, text): return self.emit("log", text=text)

    def metric(self, step, values, **kw):
        return self.emit("metric", step=step, values=values, **kw)

    def since(self, n):
        with self.lock: return self.events[max(0, int(n)):]

    def to_dict(self, since=None):
        d = {"id": self.id, "kind": self.kind, "params": self.params,
             "status": self.status, "progress": dict(self.progress),
             "result": self.result, "error": self.error,
             "created": self.created, "started": self.started,
             "finished": self.finished,
             "events_total": len(self.events),
             "stopping": self.stop_flag.is_set()}
        if since is not None: d["events"] = self.since(since)
        return d


class Runner:
    """Starts jobs on threads and keeps them so the page can read them back."""

    def __init__(self, registry, keep=40):
        self.reg = registry
        self.jobs = {}
        self.order = []
        self.keep = keep
        # Re-entrant: `start` holds this while it mints an id, which also takes it.
        self.lock = threading.RLock()
        self.writing = {}          # model -> job id currently writing to it
        self._n = 0

    def _jid(self):
        with self.lock:
            self._n += 1
            return f"j{self._n}"

    def start(self, kind, params):
        if kind not in KINDS: raise ValueError(f"unknown job kind {kind!r}")
        model = params.get("model", "main")
        self.reg.require(model)
        with self.lock:
            if kind in WRITES:
                busy = self.writing.get(model)
                # "queued" counts: the thread may not have been scheduled yet, and
                # a second writer admitted in that window is the race this guard
                # exists to stop.
                if busy and self.jobs[busy].status in ("queued", "running"):
                    raise RuntimeError(f"{busy} is already training {model!r}")
            job = Job(self._jid(), kind, params)
            self.jobs[job.id] = job
            self.order.append(job.id)
            if kind in WRITES: self.writing[model] = job.id
            for old in self.order[:-self.keep]:
                j = self.jobs.get(old)
                if j and j.status in ("done", "error", "stopped"):
                    self.jobs.pop(old, None)
            self.order = [i for i in self.order if i in self.jobs]
        t = threading.Thread(target=self._run, args=(job,), daemon=True,
                             name=f"cortex-{job.id}")
        t.start()
        return job

    def get(self, jid): return self.jobs.get(jid)

    def stop(self, jid):
        j = self.jobs.get(jid)
        if j is None: raise LookupError(f"no job {jid!r}")
        j.stop_flag.set()
        return j

    def list(self):
        return [self.jobs[i].to_dict() for i in reversed(self.order) if i in self.jobs]

    # ------------------------------------------------------------------ run
    def _run(self, job):
        job.status = "running"; job.started = time.time()
        try:
            fn = getattr(self, f"_job_{job.kind}")
            job.result = fn(job)
            job.status = "stopped" if job.stop_flag.is_set() else "done"
            job.emit("done", result=job.result, status=job.status)
        except Exception as e:                       # a job must not kill the server
            job.error = f"{type(e).__name__}: {e}"
            job.status = "error"
            job.emit("error", error=job.error, trace=traceback.format_exc()[-1200:])
        finally:
            job.finished = time.time()

    # ------------------------------------------------------------- the jobs
    def _job_train(self, job):
        p = job.params
        model, name = p.get("model", "main"), p["game"]
        c = self.reg.require(model); game = ALL[name]
        episodes = int(p.get("episodes", 400))
        chunk = max(1, int(p.get("chunk", 50)))
        k = int(p.get("k", 12)); lr = float(p.get("lr", 0.05))
        eval_n = int(p.get("eval_n", 40))
        balance = bool(p.get("balance", True))
        selfplay = float(p.get("selfplay", 0.5))
        opp = opponent_for(name, p.get("opponent"))
        rng = random.Random(int(p.get("seed", 0)))
        job.progress["total"] = episodes
        done = 0; last = None
        job.log(f"train {name} on model {model!r}: {episodes} episodes in chunks of {chunk}")
        while done < episodes and not job.stop_flag.is_set():
            n = min(chunk, episodes - done)
            with self.reg.lock:
                tr = c.train(game, episodes=n, rng=rng, lr=lr, k=k, balance=balance,
                             opponent=opp, selfplay=selfplay)
                ev = c.evaluate(game, n=eval_n, rng=random.Random(7), k=k,
                                opponent=opp, selfplay=selfplay)
            done += n
            job.progress["done"] = done
            last = {"train": tr, "eval": ev}
            job.metric(done, {"legality": ev["legality_acc"],
                              "baseline": ev["majority_baseline"],
                              "top_choice_legal": ev["top_choice_legal"],
                              "random_pick": ev["random_pick_baseline"],
                              "train_acc": tr["train_acc"],
                              "hidden": tr["shape"]["nh"]},
                       rule=tr["rule"], grown=tr["hidden_grown"], region=tr["region"])
        return {"game": name, "model": model, "episodes": done, **(last or {})}

    def _job_selfplay(self, job):
        p = job.params
        model, name = p.get("model", "main"), p["game"]
        c = self.reg.require(model); game = ALL[name]
        rounds = int(p.get("rounds", 40))
        chunk = max(1, int(p.get("chunk", 5)))
        opp = opponent_for(name, p.get("opponent"))
        depth = int(p.get("opponent_depth", 1))
        plies = int(p.get("max_plies", 120))
        eps = float(p.get("eps", 0.25))
        rng = random.Random(int(p.get("seed", 0)))
        job.progress["total"] = rounds
        done = 0; tally = {"win": 0, "loss": 0, "draw": 0}; last = None
        job.log(f"selfplay {name} on {model!r}: {rounds} games against "
                f"{(p.get('opponent') or {}).get('kind', 'bot')}")
        while done < rounds and not job.stop_flag.is_set():
            n = min(chunk, rounds - done)
            with self.reg.lock:
                r = c.selfplay(game, rounds=n, rng=rng, opponent=opp,
                               opponent_depth=depth, max_plies=plies, eps=eps)
            done += n
            for key in tally: tally[key] += r[key]
            job.progress["done"] = done
            last = r
            job.metric(done, {"win_rate": round(tally["win"] / done, 4),
                              "loss_rate": round(tally["loss"] / done, 4),
                              "draw_rate": round(tally["draw"] / done, 4),
                              "hidden": r["shape"]["nh"]},
                       gamma=r["gamma"])
        return {"game": name, "model": model, "rounds": done, **tally,
                "last": last}

    def _job_rehearse(self, job):
        p = job.params
        model = p.get("model", "main")
        c = self.reg.require(model)
        rid = p.get("region")
        if rid is None and p.get("game"): rid = c.region_for(ALL[p["game"]]).id
        if rid is None: raise ValueError("rehearse needs a region or a game")
        region = c.regions[int(rid)]
        episodes = int(p.get("episodes", 200))
        chunk = max(1, int(p.get("chunk", 50)))
        rng = random.Random(int(p.get("seed", 0)))
        job.progress["total"] = episodes
        done = 0; samples = 0
        job.log(f"rehearse region {region.id} ({', '.join(g.name for g in region.games)})")
        while done < episodes and not job.stop_flag.is_set():
            n = min(chunk, episodes - done)
            with self.reg.lock:
                info = c.rehearse(region, rng=rng, episodes=n, k=int(p.get("k", 12)))
            done += n; samples += info["samples"]
            job.progress["done"] = done
            job.metric(done, {"samples": samples})
        with self.reg.lock:
            evals = {g.name: c.evaluate(g, n=int(p.get("eval_n", 30)),
                                        rng=random.Random(7), k=12)
                     for g in region.games}
        return {"region": region.id, "episodes": done, "samples": samples,
                "rehearsed": [g.name for g in region.games], "eval": evals}

    def _job_distill(self, job):
        p = job.params
        model, name = p.get("model", "main"), p.get("game", "chess")
        c = self.reg.require(model); game = ALL[name]
        from cortex import stockfish as sfmod
        if not sfmod.available():
            raise LookupError("stockfish is not installed; set STOCKFISH_PATH")
        teacher = stockfish_player(int(p.get("skill", 20)), int(p.get("depth", 8)))
        positions = int(p.get("positions", 150))
        chunk = max(1, int(p.get("chunk", 25)))
        opp = opponent_for(name, p.get("opponent"))
        rng = random.Random(int(p.get("seed", 0)))
        job.progress["total"] = positions
        done = 0; scored = 0
        job.log(f"distil {name} grade from stockfish over {positions} positions")
        while done < positions and not job.stop_flag.is_set():
            n = min(chunk, positions - done)
            with self.reg.lock:
                r = c.distill(game, teacher, positions=n, rng=rng,
                              depth=int(p.get("depth", 8)),
                              multipv=int(p.get("multipv", 24)), opponent=opp)
            done += n; scored += r["moves_scored"]
            job.progress["done"] = done
            job.metric(done, {"moves_scored": scored, "skipped": r["skipped"]})
        with self.reg.lock:
            ev = c.evaluate(game, n=40, rng=random.Random(7), k=12, opponent=opp)
        return {"game": name, "positions": done, "moves_scored": scored, "eval": ev}

    def _job_evaluate(self, job):
        p = job.params
        model = p.get("model", "main")
        c = self.reg.require(model)
        names = p.get("games") or sorted(c.route_of)
        out = {}
        job.progress["total"] = len(names)
        for i, name in enumerate(names):
            if job.stop_flag.is_set(): break
            opp = opponent_for(name, p.get("opponent")) if name != "sudoku" else None
            with self.reg.lock:
                out[name] = c.evaluate(ALL[name], n=int(p.get("n", 40)),
                                       rng=random.Random(int(p.get("seed", 7))),
                                       k=int(p.get("k", 12)), opponent=opp)
            job.progress["done"] = i + 1
            job.metric(i + 1, {"legality": out[name]["legality_acc"],
                               "top_choice_legal": out[name]["top_choice_legal"],
                               "baseline": out[name]["majority_baseline"]},
                       game=name)
        return {"model": model, "eval": out}

    def _job_benchmark(self, job):
        """N matches of one pairing. This is what turns a single game into a
        result: one win proves nothing, and the illegal rate over a hundred
        plies is the number that actually moves."""
        p = job.params
        name = p["game"]
        games_n = int(p.get("games", 6))
        a = Player.from_dict(p.get("a") or {"kind": "cortex"})
        b = Player.from_dict(p.get("b") or {"kind": "bot", "depth": 1})
        from cortex.arena import SIDES
        sides = SIDES[name]
        seed0 = int(p.get("seed", 100))
        job.progress["total"] = games_n
        tally = {"a": 0, "b": 0, "none": 0}
        illegal = {sides[0]: [0, 0], sides[1]: [0, 0]}
        rows = []
        swap = bool(p.get("swap_sides", False))
        for i in range(games_n):
            if job.stop_flag.is_set(): break
            pa, pb = (b, a) if (swap and i % 2) else (a, b)
            with self.reg.lock:
                m = Match(name, {sides[0]: pa, sides[1]: pb},
                          resolve=self.reg.resolve(), seed=seed0 + i,
                          max_plies=int(p.get("max_plies", 0)) or None)
                m.run()
            res = m.result or {}
            w = res.get("winner")
            who = "none"
            if w == sides[0]: who = "b" if (swap and i % 2) else "a"
            elif w == sides[1]: who = "a" if (swap and i % 2) else "b"
            tally[who] += 1
            st = m.stats()
            for s in sides:
                illegal[s][0] += st[s]["illegal"]; illegal[s][1] += st[s]["tried"]
            row = {"game": i, "winner": w, "who": who, "reason": res.get("reason"),
                   "natural": res.get("natural"), "plies": m.plies,
                   "detail": res.get("detail"), "stats": st,
                   "sides": {sides[0]: pa.name(), sides[1]: pb.name()}}
            rows.append(row)
            job.progress["done"] = i + 1
            job.metric(i + 1, {"a_wins": tally["a"], "b_wins": tally["b"],
                               "drawn": tally["none"]}, row=row)
        return {"game": name, "played": sum(tally.values()), "tally": tally,
                "a": a.to_dict(), "b": b.to_dict(), "rows": rows,
                "illegal": {s: {"illegal": v[0], "tried": v[1],
                                "rate": round(v[0] / v[1], 4) if v[1] else 0.0}
                            for s, v in illegal.items()}}

    def _job_transfer(self, job):
        """DESIGN.md 15.1, as a measurement rather than a hope.

        Regions share no weights, so an ensemble is the ONLY channel by which
        one region's learning can reach another's game. Each game is answered
        four ways -- by its own region, by the two regions that win the auction,
        by every region, and by the majority baseline -- and the question is
        whether any column beats the first. The README measures one case that
        does (checkers gains 0.025 from the go region) and three that are flat;
        this recomputes it on the model that is loaded.
        """
        from cortex import credit as credit_mod, routing
        p = job.params
        model = p.get("model", "main")
        c = self.reg.require(model)
        names = p.get("games") or sorted(c.route_of)
        n_states = int(p.get("states", 24)); k = int(p.get("k", 6))
        rows = []
        job.progress["total"] = len(names)
        for i, name in enumerate(names):
            if job.stop_flag.is_set(): break
            game = ALL[name]
            opp = opponent_for(name, p.get("opponent")) if name != "sudoku" else None
            with self.reg.lock:
                rr = random.Random(7); S = []
                for st in c._states(game, rr, n_states, 0.5 if opp else 0.0, opp):
                    for mv in game.candidates(st, rr, k):
                        S.append((st, mv, game.is_legal(st, mv)))
                if not S: continue
                def acc(regions):
                    return sum(1 for st, mv, l in S
                               if (credit_mod.ensemble(regions, c, game, st, mv)[0] >= 0.5) == l) / len(S)
                own = [c.region_for(game)]
                top2 = routing.allocate(c, game, m=2).seats
                alln = [r for r in c.regions if r.net is not None]
                legal_n = sum(1 for _, _, l in S if l)
                base = max(legal_n, len(S) - legal_n) / len(S)
                row = {"game": name, "own": round(acc(own), 4),
                       "auction_top2": round(acc(top2), 4), "all": round(acc(alln), 4),
                       "baseline": round(base, 4), "samples": len(S),
                       "own_region": own[0].id, "seats": [r.id for r in top2]}
            row["gain"] = round(max(row["auction_top2"], row["all"]) - row["own"], 4)
            rows.append(row)
            job.progress["done"] = i + 1
            job.metric(i + 1, {"own": row["own"], "auction_top2": row["auction_top2"],
                               "all": row["all"], "baseline": row["baseline"]}, game=name)
        return {"model": model, "rows": rows}

    def _job_credit(self, job):
        """Coverage, Shapley over regions, and the auction -- the numbers the
        README reports, computed live on the model that is loaded."""
        p = job.params
        model = p.get("model", "main")
        c = self.reg.require(model)
        names = p.get("games") or sorted(c.route_of)
        n_states = int(p.get("states", 16)); k = int(p.get("k", 4))
        # The same mixture `cli.cmd_credit` samples, so the page's numbers and
        # the README's numbers are answers to the same question.
        selfplay = float(p.get("selfplay", 0.5))
        cov = []
        with self.reg.lock:
            for r in c.regions:
                cov.append({"region": r.id, "games": [g.name for g in r.games],
                            "coverage": {n: round(credit_mod.coverage(r, ALL[n]), 3)
                                         for n in names}})
        out = {}
        job.progress["total"] = len(names)
        for i, name in enumerate(names):
            if job.stop_flag.is_set(): break
            game = ALL[name]
            rr = random.Random(3)
            opp = opponent_for(name, p.get("opponent")) if name != "sudoku" else None
            with self.reg.lock:
                S = []
                for st in c._states(game, rr, n_states, selfplay if opp else 0.0, opp):
                    for mv in game.candidates(st, rr, k):
                        S.append((st, mv, game.is_legal(st, mv)))
                sh = credit_mod.shapley(c, game, S)
                alloc = routing.allocate(c, game, m=int(p.get("seats", 2)))
            out[name] = {"phi": {str(kk): round(v, 4) for kk, v in sh["phi"].items()},
                         "total": round(sh["total"], 4),
                         "efficiency_error": sh["efficiency_error"],
                         "method": sh["method"], "samples": len(S),
                         "auction": alloc.to_dict()}
            job.progress["done"] = i + 1
            job.metric(i + 1, {"total": out[name]["total"]}, game=name,
                       phi=out[name]["phi"])
        return {"model": model, "coverage": cov, "shapley": out}


# -------------------------------------------------------------- checkpoints

def build(games, tau_new=0.60, seed=0, nh=24):
    """A fresh cortex over the named games, in the order given -- which matters:
    the graph routes each game against the regions that already exist."""
    c = Cortex(tau_new=float(tau_new), seed=int(seed), nh=int(nh))
    for n in games:
        if n not in ALL: raise ValueError(f"unknown game {n!r}")
        c.add_game(ALL[n])
    return c


def save(registry, name, path):
    with registry.lock:
        return checkpoint.save(registry.require(name), path)


def load(registry, name, path, note=""):
    c = checkpoint.load(path)
    registry.put(name, c, note=note or f"loaded from {path}")
    return c
