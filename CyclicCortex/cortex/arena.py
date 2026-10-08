"""The arena: one match, any two players, any of the four games.

The cortex could already play -- `cli.play` puts it against the built-in engine
and prints a line. What it could not do is play against a HUMAN, against another
cortex, or against anything while somebody watches what it was thinking. That is
what this module is for, and it is deliberately separate from the HTTP layer:
everything here is ordinary Python objects, so the tests drive matches without a
socket.

Four kinds of player, and every pairing of them is legal:

  ``human``      supplies a move through `Match.play`
  ``cortex``     a region's own ranking picks the move
  ``bot``        the built-in opponents in `cortex/engine.py`, `depth=0` random
  ``stockfish``  a real engine, chess only, when it is installed

so "human vs model", "model vs model" and "bot vs model" are not three features
but three rows of the same table.

Two things are reported that a plain result cannot say:

*The pick is not filtered.* A cortex player ranks its own candidate set exactly
as `Cortex.choose` does, which includes moves that are not legal. An illegal pick
is COUNTED and the best legal alternative is substituted, so the game continues
and `illegal_rate` measures what the network learned rather than hiding it --
the same accounting `cli.play` does, kept because it is the honest number.

*The scores are kept.* `analyse` returns `p_valid` and `grade` for every legal
move in a position, which is what the board heat map draws. A result says the
cortex lost; the scores say it wanted to play a move that hangs a rook.

Sudoku is solitaire, so "two players" means something different there and is
labelled as what it is: both players place on the ONE grid, alternating, and the
score is how many legal placements each of them made. A second player of kind
``none`` leaves it as the single-handed puzzle it normally is.
"""
import random, time

from cortex import engine
from cortex.board_games import GN, PASS
from cortex.cortex import Cortex
from cortex.games import ALL

# ---------------------------------------------------------------- the games

SIDES = {"chess": ("w", "b"), "checkers": ("w", "b"),
         "go": ("b", "w"), "sudoku": ("p1", "p2")}

SIDE_NAMES = {
    "chess":    {"w": "White", "b": "Black"},
    "checkers": {"w": "White", "b": "Black"},
    "go":       {"b": "Black", "w": "White"},
    "sudoku":   {"p1": "First", "p2": "Second"},
}

# Which side the cortex takes when nothing says otherwise. Matches `cli.py`, so
# a match started from the page is the same experiment the command line runs.
CORTEX_SIDE = {"chess": "w", "checkers": "w", "go": "b", "sudoku": "p1"}

# A ply cap per game, because none of these three can be relied on to end: chess
# has no repetition rule here, go only ends on two passes, and a cortex that has
# not learned to pass will not produce them.
PLY_CAP = {"chess": 120, "checkers": 150, "go": 200, "sudoku": 162}

FILES = "abcdefgh"
GO_FILES = "ABCDEFGHJ"          # no I, as every go diagram omits it


# ------------------------------------------------------------- moves as text

def move_key(game, mv):
    """A stable string for one move, so the browser can hand a move back.

    The alternative is parsing coordinates out of a JSON body and rebuilding the
    tuple, which is the same work with more ways to be wrong. A key is looked up
    in the legal-move list and the tuple that produced it is used, so a key that
    does not name a legal move simply is not found."""
    if game == "sudoku":
        x, y, v = mv
        return f"{x},{y}={v}"
    if game == "go":
        return "pass" if mv == PASS else f"{mv[0]},{mv[1]}"
    (fx, fy), (tx, ty) = mv
    return f"{fx},{fy}-{tx},{ty}"


def move_label(game, state, mv):
    """What a person reads in the move list."""
    if game == "sudoku":
        x, y, v = mv
        return f"r{y+1}c{x+1}={v}"
    if game == "go":
        if mv == PASS: return "pass"
        x, y = mv
        return f"{GO_FILES[x]}{y+1}"
    (fx, fy), (tx, ty) = mv
    a, b = f"{FILES[fx]}{fy+1}", f"{FILES[tx]}{ty+1}"
    if game == "checkers":
        return f"{a}x{b}" if abs(tx - fx) == 2 else f"{a}-{b}"
    piece = state.at(fx, fy)
    kind = piece[1] if piece and piece[1] != "P" else ""
    return f"{kind}{a}{'x' if state.at(tx, ty) else '-'}{b}"


def move_json(game, state, mv):
    """The move as the page needs it: a key, a label, and the squares to draw."""
    out = {"key": move_key(game, mv), "label": move_label(game, state, mv)}
    if game == "sudoku":
        x, y, v = mv
        out.update(cell=[x, y], value=v)
    elif game == "go":
        if mv == PASS: out["pass"] = True
        else: out["point"] = list(mv)
    else:
        (fx, fy), (tx, ty) = mv
        out.update({"from": [fx, fy], "to": [tx, ty]})
    return out


# ------------------------------------------------------------- the board seen

def view(game, state):
    """The position as JSON. Only what is needed to draw it and to say who is
    winning -- the authoritative state stays a Python object in the match."""
    if game == "chess":
        return {"kind": "chess", "w": 8, "h": 8,
                "cells": [row[:] for row in state.b],
                "turn": state.turn, "in_check": state.in_check(),
                "material": {"w": engine.material(state, "w"),
                             "b": engine.material(state, "b")}}
    if game == "checkers":
        return {"kind": "checkers", "w": 8, "h": 8,
                "cells": [row[:] for row in state.b], "turn": state.turn,
                "counts": {"w": state.count("w"), "b": state.count("b")},
                "chain": list(state.chain) if state.chain else None,
                "forced": bool(state.jumps())}
    if game == "go":
        bs, ws = state.score()
        return {"kind": "go", "w": GN, "h": GN,
                "cells": [row[:] for row in state.b], "turn": state.turn,
                "score": {"b": bs, "w": ws}, "passes": state.passes}
    if game == "sudoku":
        return {"kind": "sudoku", "w": 9, "h": 9,
                "cells": [row[:] for row in state.g],
                "given": [[1 if (x, y) in state.fixed else 0 for x in range(9)]
                          for y in range(9)],
                "empties": state.empties(), "solved": state.solved()}
    raise ValueError(f"unknown game {game!r}")


# ------------------------------------------------------------------ the bots

def sudoku_choose(s, depth=1, rng=None):
    """A bot for the solitaire game: the most-constrained cell first, which is
    the standard human rule and the one `sudoku._solve` backtracks with.
    ``depth=0`` is the random-legal control, same as everywhere else."""
    rng = rng or random
    moves = s.legal_moves()
    if not moves: return None
    if depth <= 0: return rng.choice(moves)
    best = None
    for y in range(9):
        for x in range(9):
            if s.g[y][x]: continue
            c = s.candidates(x, y)
            if not c: continue
            if best is None or len(c) < len(best[1]): best = ((x, y), c)
    if best is None: return rng.choice(moves)
    (x, y), c = best
    return (x, y, rng.choice(c))


BOT = {"chess":    lambda s, d, r: engine.choose(s, depth=d, rng=r),
       "checkers": lambda s, d, r: engine.checkers_choose(s, d, r),
       "go":       lambda s, d, r: engine.go_choose(s, d, r),
       "sudoku":   lambda s, d, r: sudoku_choose(s, d, r)}

_SF = {}

def stockfish_player(skill=5, depth=6):
    """One process per (skill, depth), kept alive across matches -- starting an
    engine costs more than the search does at these depths."""
    from cortex import stockfish as sfmod
    if not sfmod.available(): return None
    key = (skill, depth)
    if key not in _SF:
        sf = sfmod.Stockfish(depth=depth)
        sf.set_skill(skill)
        _SF[key] = sf
    return _SF[key]


def close_engines():
    for sf in list(_SF.values()):
        try: sf.close()
        except Exception: pass
    _SF.clear()


# ---------------------------------------------------------- what the net sees

def rule_score(rule, pv, grade):
    """The pointwise number a region's rule ranks by, for display.

    `gated` is not pointwise -- it takes the best grade among moves it believes
    legal, and only falls back to the best `p_valid` when it believes none of
    them. Subtracting a large constant from the disbelieved moves reproduces that
    argmax exactly while staying a single number a column can be sorted on."""
    if rule == "pv": return pv
    if rule == "penalised": return pv * grade - 2.0 * (1.0 - pv)
    return grade if pv >= 0.5 else pv - 100.0


def analyse(cortex, game_name, state, seed=0, k=24):
    """Every legal move in this position, scored by the region that owns the game.

    Returns the ranking the region would actually use (`pick`), and separately
    the pick its own candidate sampling produces (`honest`), which is what
    `Cortex.choose` does in play and CAN be illegal. The difference between the
    two is the whole measurement: `honest_legal` false means the network's
    preferred move was not a move."""
    game = ALL[game_name]
    region = cortex.region_for(game)
    if region.net is None: return None
    legal = game.legal_moves(state)
    rows, scored = [], []
    for mv in legal:
        pv, gr = region.net.predict(cortex.encode(region, game, state, mv))
        scored.append((pv, gr, mv))
        rows.append({**move_json(game_name, state, mv),
                     "pv": round(pv, 4), "grade": round(gr, 4),
                     "score": round(rule_score(region.rule, pv, gr), 4),
                     "legal": True})
    pick = Cortex.rank(scored, region.rule) if scored else None
    pick_key = move_key(game_name, pick) if pick is not None else None
    for r in rows: r["pick"] = (r["key"] == pick_key)
    rows.sort(key=lambda r: -r["score"])

    rng = random.Random(seed)
    honest = cortex.choose(game, state, rng, k=k)
    honest_legal = honest is not None and game.is_legal(state, honest)
    return {
        "game": game_name, "region": region.id, "rule": region.rule,
        "vocab_width": region.vocab.width, "shape": region.shape(),
        # A region that has seen nothing still ranks moves -- from its random
        # initialisation. Saying so is the difference between "the model is bad"
        # and "there is no model yet".
        "seen": region.net.seen, "trained": region.net.seen > 0,
        "moves": rows,
        "pick": pick_key,
        "honest": move_json(game_name, state, honest) if honest is not None else None,
        "honest_legal": honest_legal,
        "legal_moves": len(legal),
    }


# ----------------------------------------------------------------- a player

class Player:
    """Who is on one side. `model` names a cortex in the server's registry;
    `depth` throttles a bot (0 is random-legal); `skill` throttles Stockfish."""

    KINDS = ("human", "cortex", "bot", "stockfish", "none")

    def __init__(self, kind="human", model="main", depth=1, skill=5, label=None):
        if kind not in self.KINDS: raise ValueError(f"unknown player kind {kind!r}")
        self.kind, self.model = kind, model
        self.depth, self.skill = int(depth), int(skill)
        self.label = label

    def name(self):
        if self.label: return self.label
        if self.kind == "cortex": return f"cortex:{self.model}"
        if self.kind == "bot": return f"bot depth {self.depth}" if self.depth else "random"
        if self.kind == "stockfish": return f"stockfish skill {self.skill}"
        return self.kind

    def to_dict(self):
        return {"kind": self.kind, "model": self.model, "depth": self.depth,
                "skill": self.skill, "name": self.name()}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(kind=d.get("kind", "human"), model=d.get("model", "main"),
                   depth=int(d.get("depth", 1)), skill=int(d.get("skill", 5)),
                   label=d.get("label"))


# ------------------------------------------------------------------ a match

class Match:
    """One game between two players, played a ply at a time.

    Nothing happens on its own. `step` advances exactly one ply and only if the
    side to move is not human, `play` takes a human's move, and `run` repeats
    `step` until a human is on the move or the game is over. A match therefore
    never blocks on a human and never runs away from one."""

    def __init__(self, game, players, resolve=None, seed=0, max_plies=None,
                 mid=None, spectator="main"):
        if game not in ALL: raise ValueError(f"unknown game {game!r}")
        self.id = mid or f"m{int(time.time()*1000)%10**9}"
        self.game_name = game
        self.game = ALL[game]
        self.sides = SIDES[game]
        self.players = {s: players.get(s) or Player("none") for s in self.sides}
        self.resolve = resolve or (lambda name: None)
        # Whose opinion is reported when neither player is a cortex. A human
        # against the engine is still worth watching WITH the model's scores on
        # the board -- that is the comparison, and it costs one forward pass a
        # move to have it.
        self.spectator = spectator
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.max_plies = int(max_plies or PLY_CAP[game])
        self.created = time.time()
        self.state = self.game.new(seed=self.seed)
        self.turn = self.sides[0]
        self.history = []
        self.plies = 0
        self.illegal = {s: 0 for s in self.sides}
        self.tried = {s: 0 for s in self.sides}
        self.placed = {s: 0 for s in self.sides}      # sudoku: legal placements each
        self.result = None
        self._refresh_turn()

    # ------------------------------------------------------------- the turn
    def _refresh_turn(self):
        """Who is on the move. Three of the games carry it in the state -- and
        checkers carries a MID-CHAIN turn, where the same side moves again, which
        is a rule and not bookkeeping. Sudoku has no turn of its own, so the
        match alternates, skipping a side that is not present."""
        if self.game_name == "sudoku":
            if self.players[self.sides[1]].kind == "none": self.turn = self.sides[0]
            elif self.players[self.sides[0]].kind == "none": self.turn = self.sides[1]
            return
        self.turn = self.state.turn

    def current(self): return self.players[self.turn]

    def _other(self, side):
        return self.sides[1] if side == self.sides[0] else self.sides[0]

    def _advance_turn(self):
        if self.game_name == "sudoku":
            other = self._other(self.turn)
            if self.players[other].kind != "none": self.turn = other
        else:
            self._refresh_turn()

    # ------------------------------------------------------------- the moves
    def legal_moves(self):
        return self.game.legal_moves(self.state)

    def legal_json(self):
        return [move_json(self.game_name, self.state, m) for m in self.legal_moves()]

    def find(self, key):
        for m in self.legal_moves():
            if move_key(self.game_name, m) == key: return m
        return None

    # ------------------------------------------------------------ the result
    def finished(self):
        if self.result: return True
        g, s = self.game, self.state
        if self.game_name == "sudoku":
            if s.solved():
                self.result = self._end("solved", None, "the grid is complete")
                return True
            if not s.legal_moves():
                self.result = self._end("stuck", self._most_placed(),
                                        "no legal placement remains")
                return True
        elif self.game_name == "go":
            if s.over():
                bs, ws = s.score()
                w = "b" if bs > ws else ("w" if ws > bs else None)
                self.result = self._end("two passes", w, f"area {bs}-{ws}")
                return True
        elif self.game_name == "checkers":
            w = s.winner()
            if w is not None:
                self.result = self._end("no moves", w,
                                        f"{SIDE_NAMES['checkers'][self._other(w)]} has nothing to play")
                return True
        elif self.game_name == "chess":
            if not engine.legal_moves(s):
                if s.in_check():
                    self.result = self._end("checkmate", self._other(s.turn), "mate")
                else:
                    self.result = self._end("stalemate", None, "no legal move, not in check")
                return True
        if self.plies >= self.max_plies:
            self.result = self._end("ply cap", self._ahead(),
                                    f"stopped at {self.plies} plies")
            return True
        return False

    def _most_placed(self):
        a, b = self.sides
        if self.placed[a] == self.placed[b]: return None
        return a if self.placed[a] > self.placed[b] else b

    def _ahead(self):
        """Who is ahead when a game is stopped rather than finished. Reported as
        a lead, never as a win -- a capped game has no winner."""
        if self.game_name == "sudoku": return self._most_placed()
        if self.game_name == "go":
            bs, ws = self.state.score()
            return "b" if bs > ws else ("w" if ws > bs else None)
        if self.game_name == "checkers":
            n, m = self.state.count("w"), self.state.count("b")
            return "w" if n > m else ("b" if m > n else None)
        mat = engine.material(self.state, "w")
        return "w" if mat > 0 else ("b" if mat < 0 else None)

    def _end(self, reason, winner, detail):
        return {"over": True, "reason": reason, "winner": winner,
                "winner_name": SIDE_NAMES[self.game_name].get(winner) if winner else None,
                # `natural` separates a game that ended by its own rules from one
                # this arena stopped. A capped game has a leader, never a winner.
                "natural": reason not in ("ply cap", "no move"),
                "detail": detail, "plies": self.plies}

    # ------------------------------------------------------------ the players
    def _cortex_move(self, side, player):
        """The cortex picks from its own candidate set, exactly as it does in
        `cli.play`: the pick is NOT filtered to legal moves, an illegal one is
        counted, and the best legal alternative by grade is substituted so the
        game goes on. Hiding the substitution would make every cortex look like
        it never plays an illegal move."""
        c = self.resolve(player.model)
        if c is None: raise LookupError(f"no model named {player.model!r}")
        region = c.region_for(self.game)
        if region.net is None: raise LookupError(f"model {player.model!r} is untrained")
        legal = self.legal_moves()
        mv = c.choose(self.game, self.state, self.rng, k=24)
        self.tried[side] += 1
        info = {"model": player.model, "region": region.id, "rule": region.rule}
        if mv is None or not self.game.is_legal(self.state, mv):
            self.illegal[side] += 1
            info["illegal"] = True
            if mv is not None:
                info["intended"] = move_json(self.game_name, self.state, mv)
            mv = max(legal, key=lambda m:
                     region.net.predict(c.encode(region, self.game, self.state, m))[1])
            info["substituted"] = True
        scored = []
        for m in legal:
            pv, gr = region.net.predict(c.encode(region, self.game, self.state, m))
            scored.append({**move_json(self.game_name, self.state, m),
                           "pv": round(pv, 4), "grade": round(gr, 4),
                           "score": round(rule_score(region.rule, pv, gr), 4)})
        scored.sort(key=lambda r: -r["score"])
        info["top"] = scored[:6]
        return mv, info

    def _bot_move(self, side, player):
        mv = BOT[self.game_name](self.state, player.depth, self.rng)
        self.tried[side] += 1
        return mv, {"depth": player.depth}

    def _stockfish_move(self, side, player):
        if self.game_name != "chess":
            raise ValueError("stockfish plays chess only")
        sf = stockfish_player(player.skill, max(1, player.depth))
        if sf is None: raise LookupError("stockfish is not installed")
        mv = sf.best(self.state, depth=max(1, player.depth))
        self.tried[side] += 1
        return mv, {"skill": player.skill, "depth": player.depth}

    # --------------------------------------------------------------- playing
    def _apply(self, side, mv, kind, info, seconds):
        label = move_label(self.game_name, self.state, mv)
        rec = {"ply": self.plies + 1, "side": side,
               "side_name": SIDE_NAMES[self.game_name][side],
               "by": kind, "player": self.players[side].name(),
               "key": move_key(self.game_name, mv), "label": label,
               "move": move_json(self.game_name, self.state, mv),
               "seconds": round(seconds, 4), **info}
        self.state = self.game.apply(self.state, mv)
        self.plies += 1
        self.placed[side] += 1
        self.history.append(rec)
        self._advance_turn()
        self.finished()
        return rec

    def play(self, key):
        """A human's move, named by key. Anything not in the legal-move list is
        refused -- a person is not measured on illegal picks, only the cortex is,
        and only because that number means something."""
        if self.finished(): raise ValueError("the game is over")
        side = self.turn
        if self.players[side].kind != "human":
            raise ValueError(f"{SIDE_NAMES[self.game_name][side]} is not human")
        mv = self.find(key)
        if mv is None: raise ValueError(f"{key!r} is not a legal move")
        self.tried[side] += 1
        return self._apply(side, mv, "human", {}, 0.0)

    def step(self):
        """Advance one ply, if the side to move is not a human."""
        if self.finished(): return None
        side = self.turn
        p = self.players[side]
        if p.kind == "human": return None
        if p.kind == "none": raise ValueError(f"no player on {side}")
        t0 = time.time()
        if p.kind == "cortex": mv, info = self._cortex_move(side, p)
        elif p.kind == "bot": mv, info = self._bot_move(side, p)
        else: mv, info = self._stockfish_move(side, p)
        if mv is None:
            self.result = self._end("no move", self._other(side), f"{p.name()} had nothing to play")
            return None
        return self._apply(side, mv, p.kind, info, time.time() - t0)

    def run(self, limit=400):
        """Step until a human is on the move or the game ends. Model vs model
        and bot vs model both run to the end in one call; a human game stops the
        moment it is the human's turn."""
        out = []
        for _ in range(limit):
            if self.finished(): break
            if self.current().kind == "human": break
            rec = self.step()
            if rec is None: break
            out.append(rec)
        return out

    # ----------------------------------------------------------------- state
    def stats(self):
        out = {}
        for s in self.sides:
            out[s] = {"illegal": self.illegal[s], "tried": self.tried[s],
                      "illegal_rate": round(self.illegal[s] / self.tried[s], 4)
                                      if self.tried[s] else 0.0,
                      "moves": self.placed[s]}
        return out

    def to_dict(self, analysis=True, history=40):
        self.finished()
        d = {"id": self.id, "game": self.game_name, "seed": self.seed,
             "players": {s: self.players[s].to_dict() for s in self.sides},
             "sides": list(self.sides),
             "side_names": SIDE_NAMES[self.game_name],
             "board": view(self.game_name, self.state),
             "turn": self.turn, "turn_name": SIDE_NAMES[self.game_name][self.turn],
             "to_move": self.players[self.turn].to_dict(),
             "plies": self.plies, "max_plies": self.max_plies,
             "result": self.result, "over": bool(self.result),
             "stats": self.stats(),
             "history": self.history[-history:] if history else self.history,
             "legal": [] if self.result else self.legal_json(),
             "waiting_for_human": (not self.result
                                   and self.players[self.turn].kind == "human")}
        if analysis and not self.result:
            model, playing = None, False
            p = self.players[self.turn]
            if p.kind == "cortex": model, playing = p.model, True
            else:
                for s in self.sides:
                    if self.players[s].kind == "cortex":
                        model, playing = self.players[s].model, True
                        break
            if model is None: model = self.spectator
            if model:
                c = self.resolve(model)
                if c is not None:
                    try:
                        a = analyse(c, self.game_name, self.state, seed=self.seed)
                        if a: a.update(model=model, playing=playing)
                        d["analysis"] = a
                    except (LookupError, KeyError):
                        d["analysis"] = None
        return d
