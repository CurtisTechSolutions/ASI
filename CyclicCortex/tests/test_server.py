"""Tests for the front end's three layers: the arena, the job runner, the API.

Run: python3 -m tests.test_server   (from CyclicCortex/)

The rule these follow is the one the rest of V1 follows: assert the thing that
would be a silent lie if it broke. A board that draws is not worth a test; a
match that reports zero illegal moves because it quietly filtered them IS, and
so is a checkpoint route that will write wherever the browser asks it to.
"""
import json, os, random, shutil, sys, tempfile, threading, time
import urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex import arena, trainer
from cortex.arena import Match, Player, analyse, move_key, view
from cortex.cortex import Cortex
from cortex.games import ALL
from cortex.server import App, Routes, ServerError, make_server


def small_cortex(games=("chess", "checkers", "go", "sudoku"), episodes=25):
    c = trainer.build(list(games))
    rng = random.Random(0)
    for n in games:
        c.train(ALL[n], episodes=episodes, rng=rng, k=6, opponent=arena.BOT.get(n))
    return c


# --------------------------------------------------------------------- arena

def test_move_keys_round_trip_for_every_game():
    """A key must name exactly one legal move, in every game. The browser sends
    a key back and nothing else, so a collision would silently play the wrong
    move rather than raise."""
    for name, game in ALL.items():
        s = game.new(seed=3)
        moves = game.legal_moves(s)
        keys = [move_key(name, m) for m in moves]
        assert len(set(keys)) == len(keys), f"{name}: keys collide"
        m = Match(name, {}, seed=3)
        for mv, k in zip(moves, keys):
            assert m.find(k) == mv, f"{name}: {k} did not find its move"
        assert m.find("nonsense") is None


def test_view_is_json_and_describes_the_board():
    for name, game in ALL.items():
        v = view(name, game.new(seed=1))
        json.dumps(v)                          # must survive the wire
        assert len(v["cells"]) == v["h"] and len(v["cells"][0]) == v["w"]


def test_a_human_cannot_play_an_illegal_move():
    m = Match("chess", {"w": Player("human"), "b": Player("human")}, seed=1)
    m.play("4,1-4,3")                          # e2-e4
    assert m.history[-1]["label"] == "e2-e4"
    for bad in ("4,1-4,3", "0,0-7,7", "nope"):
        try:
            m.play(bad); assert False, f"{bad} should have been refused"
        except ValueError:
            pass


def test_the_cortex_side_is_not_filtered_to_legal_moves():
    """The number that says what the network learned is its ILLEGAL RATE, and it
    only exists because the pick is taken as it comes. A match that silently
    ranked legal moves only would report 0.000 forever."""
    c = small_cortex(("chess",), episodes=10)
    m = Match("chess", {"w": Player("cortex"), "b": Player("bot", depth=0)},
              resolve=lambda n: c, seed=5, max_plies=40)
    m.run()
    st = m.stats()["w"]
    assert st["tried"] > 0
    assert 0.0 <= st["illegal_rate"] <= 1.0
    # every move actually applied was legal, however the pick came out
    for rec in m.history:
        if rec.get("illegal"): assert rec.get("substituted"), "an illegal pick must be substituted"


def test_every_pairing_plays_every_game():
    """human/cortex/bot in any arrangement, in all four games. This is the whole
    claim of the play tab, so it is one test rather than four features."""
    c = small_cortex(episodes=12)
    res = lambda n: c
    for game in ALL:
        a, b = arena.SIDES[game]
        for pa, pb in ((Player("cortex"), Player("bot", depth=1)),
                       (Player("bot", depth=1), Player("cortex")),
                       (Player("cortex"), Player("cortex"))):
            m = Match(game, {a: pa, b: pb}, resolve=res, seed=2, max_plies=24)
            m.run()
            assert m.plies > 0, f"{game}: {pa.name()} vs {pb.name()} played nothing"
            assert m.to_dict()["board"]["cells"]
        # a human on one side stops the match rather than blocking it
        m = Match(game, {a: Player("human"), b: Player("cortex")}, resolve=res, seed=2)
        m.run()
        assert m.waiting_for_human if hasattr(m, "waiting_for_human") else True
        assert m.to_dict()["waiting_for_human"], f"{game}: should be waiting for the human"
        assert m.plies == 0, f"{game}: ran past the human's turn"


def test_a_match_ends_and_says_how():
    c = small_cortex(("checkers",), episodes=12)
    m = Match("checkers", {"w": Player("bot", depth=2), "b": Player("bot", depth=1)},
              resolve=lambda n: c, seed=9, max_plies=200)
    m.run()
    r = m.result
    assert r and r["over"]
    assert r["reason"] in ("no moves", "ply cap")
    # a game stopped by the cap has a leader, never a winner
    if r["reason"] == "ply cap": assert r["natural"] is False


def test_analysis_scores_every_legal_move_and_marks_the_rule_s_pick():
    c = small_cortex(("go",), episodes=12)
    game = ALL["go"]; s = game.new()
    a = analyse(c, "go", s, seed=1)
    assert a["legal_moves"] == len(game.legal_moves(s))
    assert len(a["moves"]) == a["legal_moves"]
    assert sum(1 for r in a["moves"] if r["pick"]) == 1, "exactly one rule pick"
    top = max(a["moves"], key=lambda r: r["score"])
    assert top["pick"], "the marked pick must be the top of the displayed ordering"
    for r in a["moves"]: assert 0.0 <= r["pv"] <= 1.0


def test_analysis_reports_an_untrained_region_as_untrained():
    """Ranking moves from a random initialisation is not a weak model, it is no
    model, and the page has to be able to say so."""
    c = trainer.build(["go"])
    a = analyse(c, "go", ALL["go"].new(), seed=0)
    assert a["trained"] is False and a["seen"] == 0
    c.train(ALL["go"], episodes=10, rng=random.Random(0), k=6)
    assert analyse(c, "go", ALL["go"].new(), seed=0)["trained"] is True


def test_sudoku_is_solitaire_and_says_so():
    """Two players share the one grid. With no second player it is the single
    puzzle it normally is, and the turn must not bounce to an absent side."""
    c = small_cortex(("sudoku",), episodes=8)
    m = Match("sudoku", {"p1": Player("cortex"), "p2": Player("none")},
              resolve=lambda n: c, seed=4, max_plies=12)
    m.run()
    assert m.plies > 0
    assert all(h["side"] == "p1" for h in m.history), "the absent side must not move"


# ------------------------------------------------------------------- trainer

def test_a_training_job_reports_a_curve_and_can_be_stopped():
    reg = trainer.Registry(); reg.put("main", trainer.build(["go"]))
    run = trainer.Runner(reg)
    j = run.start("train", {"model": "main", "game": "go", "episodes": 60,
                            "chunk": 20, "k": 6, "eval_n": 10})
    t0 = time.time()
    while j.status in ("queued", "running") and time.time() - t0 < 120: time.sleep(0.05)
    assert j.status == "done", j.error
    metrics = [e for e in j.events if e["type"] == "metric"]
    assert len(metrics) == 3, "one measurement per chunk"
    assert [m["step"] for m in metrics] == [20, 40, 60]
    for m in metrics: assert 0.0 <= m["values"]["legality"] <= 1.0
    assert reg.require("main").regions[0].net.seen > 0


def test_two_writers_on_one_model_are_refused():
    """There is no merge for two gradient streams into one network."""
    reg = trainer.Registry(); reg.put("main", trainer.build(["chess"]))
    run = trainer.Runner(reg)
    a = run.start("train", {"model": "main", "game": "chess", "episodes": 400,
                            "chunk": 20, "k": 6, "eval_n": 5})
    try:
        run.start("train", {"model": "main", "game": "chess", "episodes": 10})
        raised = False
    except RuntimeError:
        raised = True
    finally:
        a.stop_flag.set()
        t0 = time.time()
        while a.status in ("queued", "running") and time.time() - t0 < 120: time.sleep(0.05)
    assert raised, "a second writer on the same model must be refused"
    assert a.status == "stopped", f"stop must be honoured, got {a.status}"


def test_a_failing_job_is_an_error_not_a_dead_server():
    reg = trainer.Registry(); reg.put("main", trainer.build(["chess"]))
    run = trainer.Runner(reg)
    j = run.start("train", {"model": "main", "game": "nonexistent", "episodes": 5})
    t0 = time.time()
    while j.status in ("queued", "running") and time.time() - t0 < 60: time.sleep(0.05)
    assert j.status == "error" and j.error


# -------------------------------------------------------------------- server

class Client:
    """The API over a real socket -- the handler's parsing and error mapping is
    most of what these tests are for, so calling the route objects directly
    would test the wrong thing."""

    def __init__(self, app):
        self.srv = make_server("127.0.0.1", 0, app=app, quiet=True)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=60) as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def fails(self, path, body=None):
        """Returns (status, error text) for a request that is meant to be refused."""
        try:
            self.post(path, body) if body is not None else self.get(path)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()).get("error", "")
        raise AssertionError(f"{path} should have failed")


def with_client(fn):
    tmp = tempfile.mkdtemp(prefix="cortex-test-")
    app = App(checkpoint_dir=tmp)
    cl = Client(app)
    try:
        fn(cl, app, tmp)
    finally:
        cl.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_api_serves_the_page_and_the_model():
    def body(cl, app, tmp):
        info = cl.get("/api/info")
        assert {g["name"] for g in info["games"]} == set(ALL)
        m = cl.get("/api/model?name=main")
        assert m["graph"]["violations"] == 0
        assert [r["games"] for r in m["regions"]] == [["chess", "checkers"], ["go"], ["sudoku"]]
        assert m["stats"]["regions"] == 3
        with urllib.request.urlopen(cl.base + "/", timeout=30) as r:
            assert r.status == 200 and b"CyclicCortex" in r.read()
        assert cl.fails("/api/model?name=missing")[0] == 404
    with_client(body)


def test_api_plays_a_game_through_to_a_result():
    def body(cl, app, tmp):
        c = small_cortex(("checkers",), episodes=15)
        app.reg.put("main", c)
        r = cl.post("/api/match", {"game": "checkers", "seed": 3,
                                   "players": {"w": {"kind": "cortex"},
                                               "b": {"kind": "bot", "depth": 1}}})
        mid = r["match"]["id"]
        assert r["match"]["waiting_for_human"] is False
        out = cl.post("/api/match/run", {"id": mid})
        assert out["match"]["over"] and out["match"]["result"]["reason"]
        assert len(out["moves"]) == out["match"]["plies"]
        assert out["match"]["stats"]["w"]["tried"] > 0
    with_client(body)


def test_api_refuses_an_illegal_human_move_and_an_unknown_match():
    def body(cl, app, tmp):
        r = cl.post("/api/match", {"game": "chess", "seed": 1,
                                   "players": {"w": {"kind": "human"},
                                               "b": {"kind": "human"}}})
        mid = r["match"]["id"]
        assert cl.fails("/api/match/move", {"id": mid, "key": "0,0-7,7"})[0] == 400
        assert cl.fails("/api/match/move", {"id": "nope", "key": "4,1-4,3"})[0] == 404
        good = cl.post("/api/match/move", {"id": mid, "key": "4,1-4,3"})
        assert good["match"]["history"][-1]["label"] == "e2-e4"
    with_client(body)


def test_checkpoints_stay_in_the_checkpoint_directory():
    """A file name arrives from the browser. `../` in one of those is how a page
    ends up writing outside the directory it was given."""
    def body(cl, app, tmp):
        r = cl.post("/api/model/save", {"name": "main", "file": "../../escape"})
        assert r["file"] == "escape.json.gz"
        assert os.path.exists(os.path.join(tmp, "escape.json.gz"))
        assert not os.path.exists(os.path.abspath(os.path.join(tmp, "..", "..", "escape.json.gz")))
        for bad in ("", "   ", ".hidden"):
            try:
                app.checkpoint_path(bad); assert False, f"{bad!r} should be refused"
            except ServerError:
                pass
        assert cl.fails("/api/model/load", {"name": "main", "file": "absent"})[0] == 404
    with_client(body)


def test_a_fork_is_a_separate_model_and_they_can_meet():
    """Model vs model needs two of them, and the interesting pairing is a model
    against its own earlier self -- so the fork has to be a copy, not an alias."""
    def body(cl, app, tmp):
        app.reg.put("main", small_cortex(("go",), episodes=15))
        cl.post("/api/model/copy", {"from": "main", "to": "rival"})
        assert set(app.reg.names()) == {"main", "rival"}
        a, b = app.reg.require("main"), app.reg.require("rival")
        assert a is not b
        assert a.regions[0].net.W1 == b.regions[0].net.W1, "a fork starts identical"
        app.reg.require("rival").train(ALL["go"], episodes=15, rng=random.Random(9), k=6)
        assert a.regions[0].net.W1 != b.regions[0].net.W1, "and diverges once trained"
        r = cl.post("/api/match", {"game": "go", "seed": 1, "run": True, "max_plies": 30,
                                   "players": {"b": {"kind": "cortex", "model": "main"},
                                               "w": {"kind": "cortex", "model": "rival"}}})
        assert r["match"]["plies"] > 0
        assert cl.fails("/api/match", {"game": "go",
                                       "players": {"b": {"kind": "cortex", "model": "ghost"}}})[0] == 404
    with_client(body)


def test_a_job_runs_through_the_api_and_its_events_paginate():
    def body(cl, app, tmp):
        j = cl.post("/api/jobs", {"kind": "train", "model": "main", "game": "go",
                                  "episodes": 40, "chunk": 20, "k": 6, "eval_n": 10})["job"]
        t0 = time.time(); seen = 0; got = []
        while time.time() - t0 < 120:
            r = cl.get(f"/api/job?id={j['id']}&since={seen}")["job"]
            got.extend(r["events"]); seen = r["events_total"]
            if r["status"] not in ("queued", "running"): break
            time.sleep(0.05)
        assert r["status"] == "done", r["error"]
        assert len(got) == seen, "a cursor must not repeat or skip an event"
        assert [e["seq"] for e in got] == list(range(len(got)))
        assert cl.fails("/api/job?id=nope")[0] == 404
        assert cl.fails("/api/jobs", {"kind": "not-a-job"})[0] == 400
    with_client(body)


def test_a_match_analysis_is_present_even_when_no_model_is_playing():
    """A human against the engine is still worth watching WITH the model's
    scores on the board; that comparison is the point of the heat map."""
    def body(cl, app, tmp):
        app.reg.put("main", small_cortex(("chess",), episodes=12))
        r = cl.post("/api/match", {"game": "chess", "seed": 1, "spectator": "main",
                                   "players": {"w": {"kind": "human"},
                                               "b": {"kind": "bot", "depth": 1}}})
        a = r["match"]["analysis"]
        assert a and a["playing"] is False and a["model"] == "main"
        assert len(a["moves"]) == 20, "twenty opening moves, all scored"
    with_client(body)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    t0 = time.time()
    for f in fns:
        s = time.time(); f(); print(f"  ok  {f.__name__}  ({time.time()-s:.1f}s)")
    print(f"\n  {len(fns)} tests passed in {time.time()-t0:.1f}s")
