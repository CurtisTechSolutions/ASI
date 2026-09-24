"""Adding a game: Hexapawn, end to end, in one runnable file.

Run:  python3 examples/hexapawn.py

Hexapawn is 3x3 chess with pawns only - Martin Gardner's 1962 toy.  It is here
because it is the smallest game that is not *trivially* small: unlike Nim or
tic-tac-toe it has real move structure (a pawn pushes forward or captures
diagonally, and the two are different), it has three separate ways to end, and
its action space is a ``from x to`` product that wants a structured token.  It
is also not one of the six in :mod:`uge.games`, so nothing here is special-cased
for it anywhere in the package.

The file is the tutorial in ``USING.md`` made executable.  Everything below runs
against the same encoder, decoder, corpus generator, metrics and network the six
built-in games use, and the only thing written for Hexapawn is the class - about
seventy lines of rules and three encoding hints.

What it prints, in order:

1.  the rules, exercised - a board, its legal moves, and their tokens;
2.  one encoded game and its decode;
3.  the decoder refusing a tampered tape, and saying exactly where;
4.  a trained model's numbers against the guessing baseline;
5.  the model playing a whole game against a random opponent.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uge.codec import decode, encode, encode_prefix
from uge.corpus import positions, self_play, split
from uge.game import Game, register
from uge.metrics import evaluate, propose, rank_tokens
from uge.net import TRAIN_DEFAULTS, new_net
from uge.tape import TapeSpec, digits_token, token_digits
from uge.teachers import make_policy

# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------

W, H = 3, 3
EMPTY, WHITE, BLACK = ".", "P", "p"


class Hexapawn(Game):
    """3x3, pawns only.  White moves up the board, black moves down.

    Three ways to end, which is what makes it worth using as an example: reach
    the far rank, take the last enemy pawn, or leave the opponent with no legal
    move.  A game whose ``winner`` has one clause teaches nothing about writing
    one that has three.

    State is ``(cells, mover)`` with ``cells`` a 9-character string indexed
    ``rank * 3 + file``, rank 0 being white's home rank.
    """

    name = "hexapawn"
    players = 2

    # -- the transition system ------------------------------------------------

    def initial(self) -> tuple[str, int]:
        return (WHITE * 3 + EMPTY * 3 + BLACK * 3, 0)

    def legal(self, state: tuple[str, int]) -> list[tuple[int, int]]:
        if self.winner(state) is not None:
            return []
        cells, mover = state
        mine, theirs = (WHITE, BLACK) if mover == 0 else (BLACK, WHITE)
        forward = 1 if mover == 0 else -1
        moves = []
        for src, piece in enumerate(cells):
            if piece != mine:
                continue
            rank, file = divmod(src, W)
            to_rank = rank + forward
            if not 0 <= to_rank < H:
                continue
            ahead = to_rank * W + file
            if cells[ahead] == EMPTY:
                moves.append((src, ahead))
            for to_file in (file - 1, file + 1):
                if 0 <= to_file < W and cells[to_rank * W + to_file] == theirs:
                    moves.append((src, to_rank * W + to_file))
        return sorted(moves)

    def apply(self, state: tuple[str, int], action: tuple[int, int]) -> tuple[str, int]:
        cells, mover = state
        src, dst = action
        out = list(cells)
        out[dst] = out[src]
        out[src] = EMPTY
        return ("".join(out), 1 - mover)

    def winner(self, state: tuple[str, int]) -> int | None:
        cells, mover = state
        if WHITE in cells[(H - 1) * W :]:
            return 0  # a white pawn reached the top rank
        if BLACK in cells[:W]:
            return 1
        if WHITE not in cells:
            return 1  # no pawns left to move
        if BLACK not in cells:
            return 0
        return 1 - mover if not self._has_move(state) else None

    def _has_move(self, state: tuple[str, int]) -> bool:
        """``legal`` calls ``winner`` calls ``legal`` - so the stalemate test needs its own."""
        cells, mover = state
        mine, theirs = (WHITE, BLACK) if mover == 0 else (BLACK, WHITE)
        forward = 1 if mover == 0 else -1
        for src, piece in enumerate(cells):
            if piece != mine:
                continue
            rank, file = divmod(src, W)
            to_rank = rank + forward
            if not 0 <= to_rank < H:
                continue
            if cells[to_rank * W + file] == EMPTY:
                return True
            for to_file in (file - 1, file + 1):
                if 0 <= to_file < W and cells[to_rank * W + to_file] == theirs:
                    return True
        return False

    def to_move(self, state: tuple[str, int]) -> int:
        return state[1]

    # -- the encoding hooks ---------------------------------------------------

    @property
    def action_count(self) -> int:
        return 81  # every (from, to) pair, legal or not

    def action_index(self, action: tuple[int, int]) -> int:
        return action[0] * 9 + action[1]

    def index_action(self, index: int) -> tuple[int, int]:
        return divmod(index, 9)

    def struct_token(self, action: tuple[int, int]) -> str:
        """``(from, to, 0)`` - one digit each, so the seams read "it landed on 4; what starts next?"."""
        return digits_token((action[0], action[1], 0))

    def struct_action(self, token: str) -> tuple[int, int] | None:
        src, dst, pad = token_digits(token)
        return None if pad or src > 8 or dst > 8 else (src, dst)

    def state_key(self, state: tuple[str, int]) -> bytes:
        return f"{state[0]}{state[1]}".encode()

    def state_code(self, state: tuple[str, int]) -> int:
        """``3 ** 9 * 2 = 39_366`` - small enough that ``phi_exact`` is an option here."""
        code = 0
        for ch in state[0]:
            code = code * 3 + ".Pp".index(ch)
        return code * 2 + state[1]

    def action_summary(self, action: tuple[int, int]) -> int:
        """The square it landed on - what ``phi_prev_bits`` writes down."""
        return action[1]

    action_summary_bound = 9

    # -- the optional extras --------------------------------------------------

    def heuristic(self, state: tuple[str, int]) -> float:
        """Material, plus how far the most advanced pawn has got.  Enough for a teacher."""
        cells = state[0]
        score = float(cells.count(WHITE) - cells.count(BLACK))
        for i, ch in enumerate(cells):
            if ch == WHITE:
                score += 0.3 * (i // W)
            elif ch == BLACK:
                score -= 0.3 * (H - 1 - i // W)
        return score

    def render(self, state: tuple[str, int]) -> str:
        cells = state[0]
        rows = [f"{r} " + " ".join(cells[r * W : (r + 1) * W]) for r in range(H - 1, -1, -1)]
        return "\n".join(rows) + f"\n  0 1 2   to move: {'white' if state[1] == 0 else 'black'}"

    def action_str(self, action: tuple[int, int]) -> str:
        return f"{action[0]}->{action[1]}"


HEXAPAWN = register(Hexapawn())


# ---------------------------------------------------------------------------
# the tutorial
# ---------------------------------------------------------------------------


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    game = HEXAPAWN
    game.validate()  # the action space fits every codebook and round-trips

    # the spec: one token of hashed state before every move, the disjoint
    # codebook (no phase ambiguity), the game's own digit layout
    spec = TapeSpec("hexapawn", game.index, phi_bits=15, layout="struct", book="plain")

    rule("1. the rules, exercised")
    state = game.initial()
    print(game.render(state))
    print("\nlegal moves and the tokens they encode to:")
    for action in game.legal(state):
        print(f"  {game.action_str(action):8s} index {game.action_index(action):3d}"
              f"  token {game.action_token(action, spec)}")

    rule("2. one game, encoded and decoded")
    corpus = self_play(game, spec, 1, make_policy("search3"), seed=4)
    play, tape = corpus.plays[0], corpus.tapes[0]
    print(f"spec:  {spec.describe()}")
    print(f"tape:  {tape}")
    print(f"       {len(tape)} characters = {len(tape) // 3} tokens"
          f" = header + {play.plies} x (phi + move) + outcome")
    back = decode(game, spec, tape, assume_winner=play.winner)
    print(f"decode: ok={back.ok} complete={back.complete} plies={back.plies} "
          f"outcome={back.outcome} claimed={back.claimed_outcome}")
    print(f"moves:  {[game.action_str(a) for a in back.actions]}")
    assert [game.action_token(a, spec) for a in back.actions] == \
           [game.action_token(a, spec) for a in play.actions], "round trip failed"

    rule("3. the decoder is the referee")
    # replace the first move with a pawn teleporting across the board
    at = (1 + spec.phi_tokens_for(game.token_bound(spec))) * 3
    tampered = tape[:at] + game.action_token((0, 8), spec) + tape[at + 3 :]
    broken = decode(game, spec, tampered, assume_winner=play.winner)
    print(f"put 0->8 at ply 0:  break_kind={broken.break_kind!r} at ply {broken.plies}"
          f"  token={broken.break_token!r}")
    junk = tape[:at] + "###" + tape[at + 3 :]
    print(f"put '###' there:    break_kind={decode(game, spec, junk).break_kind!r}")
    cut = tape[: at + 4]
    short = decode(game, spec, cut)
    print(f"cut mid-token:      trimmed={short.trimmed} char(s), "
          f"break_kind={short.break_kind!r}, plies={short.plies}")

    rule("4. train, and measure against guessing")
    corpus = self_play(game, spec, 400, make_policy("search3"), seed=1)
    train, test = split(corpus, 0.2)
    print(f"corpus: {len(corpus)} games, {corpus.stats()['plies_mean']:.1f} plies each, "
          f"{corpus.stats()['chars']} characters")
    model = new_net(seed=0)
    model.train(train.tapes, **dict(TRAIN_DEFAULTS, epochs=8))
    r = evaluate(model, game, spec, positions(game, test, limit=200), test.tapes, rollouts=3)
    print(f"graph:  {r['nodes']:.0f} nodes, {r['edges']:.0f} edges, "
          f"{r['trigrams']:.0f} trigrams, compression {r['compression_ratio']:.2f}x")
    print(f"coverage           {r['coverage']:.3f}   (positions it had seen)")
    print(f"legal@1 covered    {r['legal_at_1_covered']:.3f}   against guessing's "
          f"{r['legal_at_1_baseline']:.3f}")
    print(f"refusals/move      {r['refusals']:.2f}     against guessing's "
          f"{r['refusals_baseline']:.2f}")
    print(f"teacher match      {r['teacher_match']:.3f}")
    print(f"unaided plies      {r['greedy_plies']:.1f}  (no second chance; "
          f"ended by {r['greedy_break']!r})")

    rule("5. the model plays a game")
    rng = random.Random(7)
    state, states, actions = game.initial(), [game.initial()], []
    refusals = 0
    while game.winner(state) is None and len(actions) < 30:
        if game.to_move(state) == 0:
            prefix = encode_prefix(game, spec, states, actions, assume_winner=0)
            answer = propose(model, game, spec, prefix, state)
            action = answer["action"]
            refusals += answer["refusals"]
            how = f"model (rank {answer['refusals']:.0f}, {answer['ranked']} ranked)"
            if action is None:
                action, how = rng.choice(game.legal(state)), "fallback - model was silent"
        else:
            action, how = rng.choice(game.legal(state)), "random opponent"
        print(f"  ply {len(actions):2d}  {game.action_str(action):8s}  {how}")
        state = game.apply(state, action)
        states.append(state)
        actions.append(action)
    print(f"\n{game.render(state)}")
    result = {0: "white (the model) wins", 1: "black (random) wins", -1: "draw"}
    print(f"\nresult: {result.get(game.winner(state), 'unfinished')} after {len(actions)} plies; "
          f"{refusals:.0f} refusals in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
