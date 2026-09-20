"""Everything this directory claims, checked.

Run:  python3 tests.py            the fast set (a few seconds)
      python3 tests.py --full     adds perft to depth 4 and every spec combination
      python3 tests.py perft      one group by name

No pytest, no dependencies: a failing assertion prints what it expected and the
exit code is non-zero.  The groups are in the order they matter - the chess
rules first, because every legality number in ``README.md`` is measured against
them, and a movegen with one wrong en-passant case passes a hundred hand-written
tests and fails ``perft(3)``.
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uge
from uge import chess_rules as C
from uge.codec import decode, encode_prefix, phi_region, state_token
from uge.corpus import self_play, split
from uge.tape import (
    DISJOINT,
    LAYOUTS,
    PLAIN,
    RESERVED,
    W,
    Codebook,
    TapeError,
    TapeSpec,
    phase_ambiguity,
    trim_to_tokens,
)
from uge.teachers import make_policy, search_value

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL  {message}", file=sys.stderr)


def eq(got, want, message: str) -> None:
    check(got == want, f"{message}: got {got!r}, want {want!r}")


# ---------------------------------------------------------------------------


def test_perft(full: bool) -> None:
    """The only proof that :mod:`uge.chess_rules` is chess."""
    for name, fen, counts in C.PERFT_POSITIONS:
        pos = C.from_fen(fen)
        eq(C.to_fen(pos), fen, f"{name}: FEN does not round trip")
        depths = counts if full else counts[:3]
        for depth, want in enumerate(depths, start=1):
            t0 = time.perf_counter()
            got = C.perft(pos, depth)
            eq(got, want, f"{name}: perft({depth})")
            print(f"  {name:10s} perft({depth}) = {got:8d}  ({time.perf_counter() - t0:5.2f}s)")


def test_chess_details() -> None:
    """The four rules a movegen usually gets wrong, stated one at a time."""
    ep = C.from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    moves = C.legal_moves(ep)
    capture = (C.square_index("e5"), C.square_index("f6"), 0)
    check(capture in moves, "en passant capture is not generated")
    after = C.make_move(ep, capture)
    eq(after.board[C.square_index("f5")], ".", "en passant did not remove the captured pawn")

    castle = C.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    kingside = (C.square_index("e1"), C.square_index("g1"), 0)
    check(kingside in C.legal_moves(castle), "kingside castling is not generated")
    moved = C.make_move(castle, kingside)
    eq(moved.board[C.square_index("f1")], "R", "the rook did not move with the king")
    eq(moved.castling, "kq", "castling rights survived the king moving")

    through = C.from_fen("r3k2r/8/8/8/8/8/6q1/R3K2R w KQkq - 0 1")
    check(kingside not in C.legal_moves(through), "castled through an attacked square")

    promo = C.from_fen("8/P7/8/8/8/8/8/K6k w - - 0 1")
    promos = [m for m in C.legal_moves(promo) if m[2]]
    eq(len(promos), 4, "a promoting pawn should have four choices")
    eq(C.make_move(promo, (C.square_index("a7"), C.square_index("a8"), 4)).board[C.square_index("a8")],
       "Q", "promotion to queen did not put a queen there")

    mate = C.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    eq(C.outcome(mate), 1, "fool's mate is not recognised as a win for black")
    stale = C.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    eq(C.outcome(stale), -1, "stalemate is not recognised as a draw")
    eq(C.outcome(C.from_fen("8/8/4k3/8/8/4K3/8/8 w - - 0 1")), -1, "K vs K is not a draw")
    eq(C.outcome(C.from_fen("8/8/4k3/8/8/4K3/4B3/8 w - - 0 1")), -1, "K+B vs K is not a draw")
    check(C.outcome(C.from_fen("8/8/4k3/8/8/4K3/4R3/8 w - - 0 1")) is None, "K+R vs K is not a draw")


def test_codebooks() -> None:
    """The two bijections, and the property the disjoint one exists for."""
    for book in (PLAIN, DISJOINT):
        eq(book.capacity, book.radices[0] * book.radices[1] * book.radices[2], f"{book.name}: capacity")
        rng = random.Random(4)
        for _ in range(2000):
            code = rng.randrange(book.capacity)
            for order in ("lsb", "msb"):
                token = book.encode(code, order)
                eq(len(token), W, f"{book.name}: token width")
                eq(book.decode(token, order), code, f"{book.name}/{order}: {code} does not round trip")
        eq(book.meta_index(book.meta(3, 8), 8), 3, f"{book.name}: meta index does not round trip")
        check(book.meta_index(book.encode(0, "lsb"), 8) is None, f"{book.name}: code 0 read as meta")

    check(DISJOINT.disjoint, "the disjoint codebook is not disjoint")
    check(not PLAIN.disjoint, "the plain codebook claims to be disjoint")
    check(DISJOINT.capacity - RESERVED > 20480, "chess does not fit the disjoint codebook")

    # the property, stated directly: every trigram of a disjoint tape has one phase
    rng = random.Random(9)
    tape = "".join(DISJOINT.encode(rng.randrange(DISJOINT.capacity), "lsb") for _ in range(400))
    eq(phase_ambiguity([tape])["ambiguous"], 0, "a disjoint tape has an ambiguous trigram")


def test_partition() -> None:
    """Actions, ``phi``, header and outcome must not share a token - the bug that cost the most."""
    for name in uge.game_names():
        game = uge.get_game(name)
        for book, layout in (("disjoint", "lsb"), ("plain", "struct"), ("plain", "lsb")):
            spec = TapeSpec(name, game.index, phi_bits=10, layout=layout, book=book)
            base, room = phi_region(game, spec)
            check(room > 1024, f"{name}/{book}: only {room} codes left for phi")

            actions = {
                game.action_token(game.index_action(i), spec)
                for i in range(min(game.action_count, 4096))
            }
            metas = {spec.header()} | {spec.outcome_token(o) for o in ("draw", "first", "second", "ongoing")}
            check(not (actions & metas), f"{name}/{book}/{layout}: an action token is also a meta token")

            corpus = self_play(game, spec, 4, make_policy("random"), seed=3)
            phis = {
                state_token(game, spec, state, play.winner)
                for play in corpus.plays
                for state in play.states[:-1]
            }
            check(not (phis & actions), f"{name}/{book}/{layout}: a phi token is also an action token")
            check(not (phis & metas), f"{name}/{book}/{layout}: a phi token is also a meta token")


def test_games() -> None:
    """Every game's own interface, and the two that answer ``state_code``."""
    for name in uge.game_names():
        game = uge.get_game(name)
        game.validate()
        state = game.initial()
        check(game.winner(state) is None, f"{name}: the initial state is terminal")
        check(len(game.legal(state)) > 0, f"{name}: no legal action at the start")
        before = game.initial()
        game.apply(state, game.legal(state)[0])
        eq(game.state_key(state), game.state_key(before), f"{name}: apply() mutated the state")

    ttt = uge.get_game("tictactoe")
    eq(ttt.state_code(ttt.initial()), 0, "the empty tic-tac-toe board should be code 0")
    codes = set()
    seen = [ttt.initial()]
    for _ in range(200):
        s = seen[-1]
        if ttt.winner(s) is not None:
            seen.append(ttt.initial())
            continue
        codes.add(ttt.state_code(s))
        seen.append(ttt.apply(s, ttt.legal(s)[0]))
    check(len(codes) > 5, "tic-tac-toe state codes are not distinct")

    nim = uge.get_game("nim")
    eq(sorted(nim.optimal(((1, 2, 3), 0))), sorted(nim.legal(((1, 2, 3), 0))),
       "a lost Nim position should offer every move as 'optimal'")
    best = nim.optimal(((1, 2, 4), 0))
    eq(best, [(2, 1)], "the winning Nim move from (1,2,4) is to take one from the third heap")

    pig = uge.get_game("pig")
    rolled = pig.apply(pig.initial(), 1)
    check(pig.is_chance(rolled), "pig: rolling does not produce a chance node")
    eq(len(pig.legal(rolled)), 6, "pig: a die should have six faces")
    eq(pig.apply(rolled, 2)[2], 0, "pig: rolling a 1 should wipe the turn total")
    eq(pig.apply(rolled, 7)[2], 6, "pig: rolling a 6 should add six")


def test_roundtrip(full: bool) -> None:
    """Encode, decode, and get the same play back - across every spec that is legal."""
    configs = [
        {"book": "disjoint", "layout": "lsb"},
        {"book": "plain", "layout": "struct"},
        {"book": "plain", "layout": "lsb"},
        {"book": "plain", "layout": "msb"},
    ]
    bit_options = [0, 8, 13, 15, 17, 30] if full else [0, 10, 13]
    total = 0
    for name in uge.game_names():
        game = uge.get_game(name)
        for cfg in configs if full else configs[:2]:
            for bits in bit_options:
                for cond, prev in ((False, 0), (True, 0), (False, "half")):
                    prev_bits = bits // 2 if prev == "half" else 0
                    spec = TapeSpec(
                        name, game.index, phi_bits=bits, phi_prev_bits=prev_bits,
                        condition_result=cond, **cfg,
                    )
                    corpus = self_play(game, spec, 3, make_policy("random"), seed=2)
                    for play, tape in zip(corpus.plays, corpus.tapes):
                        eq(len(tape) % W, 0, f"{spec.describe()}: tape is not a whole number of tokens")
                        d = decode(game, spec, tape, assume_winner=play.winner)
                        check(d.ok, f"{spec.describe()}: the rules refused an encoded tape ({d.break_kind})")
                        eq(d.plies, play.plies, f"{spec.describe()}: ply count")
                        eq(d.claimed_outcome, play.outcome, f"{spec.describe()}: outcome token")
                        eq([game.action_token(a, spec) for a in d.actions],
                           [game.action_token(a, spec) for a in play.actions],
                           f"{spec.describe()}: the moves came back different")
                        total += 1
            if game.state_code(game.initial()) is not None:
                spec = TapeSpec(name, game.index, phi_exact=True, **cfg)
                corpus = self_play(game, spec, 3, make_policy("random"), seed=2)
                for play, tape in zip(corpus.plays, corpus.tapes):
                    d = decode(game, spec, tape)
                    check(d.ok, f"{spec.describe()}: exact phi did not decode ({d.break_kind})")
                    eq(d.plies, play.plies, f"{spec.describe()}: exact phi ply count")
                    total += 1
    print(f"  {total} tapes round-tripped")


def test_referee() -> None:
    """The decoder has to say *where* a tape broke and *why* - that is the training signal."""
    game = uge.get_game("connect4")
    spec = TapeSpec("connect4", game.index, phi_bits=12)
    corpus = self_play(game, spec, 1, make_policy("random"), seed=5)
    play, tape = corpus.plays[0], corpus.tapes[0]
    group = spec.group_for(game.token_bound(spec))
    n_phi = group - 1

    clean = decode(game, spec, tape)
    check(clean.ok, "the unmodified tape did not decode")

    at = (1 + n_phi) * W  # the first action token
    junk = tape[:at] + "___" + tape[at + W :]
    broken = decode(game, spec, junk)
    eq(broken.break_kind, "syntax", "a token that names no action should be a syntax break")
    eq(broken.plies, 0, "the syntax break is at ply 0")

    full_column = [0] * 6
    states, actions = [game.initial()], []
    for a in full_column:
        states.append(game.apply(states[-1], a))
        actions.append(a)
    prefix = encode_prefix(game, spec, states, actions)
    over = prefix + game.action_token(0, spec)
    illegal = decode(game, spec, over)
    eq(illegal.break_kind, "illegal", "a seventh disc in a full column should be an illegal break")
    eq(illegal.plies, 6, "the illegal break should come after six accepted plies")

    if n_phi:
        wrong_phi = tape[: (1 + group) * W] + spec.codebook.encode(
            phi_region(game, spec)[0] + 1234, "lsb"
        ) + tape[(1 + group + 1) * W :]
        desync = decode(game, spec, wrong_phi)
        eq(desync.break_kind, "phi", "a state token that does not match the replay should be a phi break")

    eq(decode(game, spec, "zzz" + tape[3:]).break_kind, "header", "a wrong header should be a header break")

    for cut in (1, 2, 4, 5):
        ragged = tape[:-cut] if cut < len(tape) else tape
        d = decode(game, spec, ragged)
        eq(d.trimmed, len(ragged) % W, f"a tape cut by {cut} should report its trimmed characters")
        check(d.break_kind in ("none", "truncated"),
              f"a tape cut by {cut} broke with {d.break_kind}, not a clean stop")
    eq(trim_to_tokens("ABCDEFGH"), "ABCDEF", "trim_to_tokens did not cut to the grid")


def test_mixed_phi() -> None:
    """``phi_prev_bits`` splits one token between the last move and the position."""
    game = uge.get_game("chess")
    probe = TapeSpec("chess", game.index, phi_bits=0)
    bits = probe.codebook.phi_bits_for(game.token_bound(probe))
    check(bits >= 10, f"chess should keep at least 10 bits of phi, got {bits}")

    state = game.initial()
    move = game.legal(state)[0]
    after = game.apply(state, move)
    spec = TapeSpec("chess", game.index, phi_bits=bits, phi_prev_bits=bits)
    a = state_token(game, spec, after, None, move)
    b = state_token(game, spec, after, None, game.legal(state)[1])
    check(a != b, "history-only phi ignores which move was played")

    # two different positions reached by moves that landed on the same square
    # share a bucket when all the bits are history - that is the locality it buys
    only_hash = TapeSpec("chess", game.index, phi_bits=bits, phi_prev_bits=0)
    eq(state_token(game, only_hash, after, None, move),
       state_token(game, only_hash, after, None, None),
       "a position-only phi should not depend on the previous move")

    eq(game.action_summary((12, 28, 0)), 28, "chess should summarise a move by where it landed")
    for name in uge.game_names():
        g = uge.get_game(name)
        for action in g.legal(g.initial()):
            v = g.action_summary(action)
            check(0 <= v < g.action_summary_bound,
                  f"{name}: action_summary {v} outside [0, {g.action_summary_bound})")

    try:
        TapeSpec("chess", game.index, phi_bits=4, phi_prev_bits=8)
    except TapeError:
        pass
    else:
        check(False, "phi_prev_bits larger than phi_bits should be refused")


def test_determinism() -> None:
    """Same seed, same corpus - or a sweep is a sweep over two variables."""
    game = uge.get_game("othello")
    spec = TapeSpec("othello", game.index, phi_bits=12)
    a = self_play(game, spec, 5, make_policy("search2"), seed=11)
    b = self_play(game, spec, 5, make_policy("search2"), seed=11)
    eq(a.tapes, b.tapes, "two corpora with the same seed differ")
    c = self_play(game, spec, 3, make_policy("search2"), seed=11)
    eq(c.tapes, a.tapes[:3], "a shorter corpus is not a prefix of a longer one")
    d = self_play(game, spec, 5, make_policy("search2"), seed=12)
    check(d.tapes != a.tapes, "two corpora with different seeds are identical")

    train, test = split(a, 0.2)
    eq(len(train) + len(test), len(a), "the split loses games")
    check(not (set(train.tapes) & set(test.tapes)), "a game is in both halves of the split")


def test_teachers() -> None:
    """The search has to actually find the win, or the corpus is noise wearing a label."""
    ttt = uge.get_game("tictactoe")
    # X to move, two in a row on the top line: taking the third wins at once
    state = ("XX.OO....", 0)
    values = [(search_value(ttt, ttt.apply(state, a), 6), a) for a in ttt.legal(state)]
    best = max(values)[1]
    eq(best, 2, "the winning tic-tac-toe move was not found")

    nim = uge.get_game("nim")
    policy = make_policy("search4:0")
    rng = random.Random(0)
    state = ((1, 2, 4), 0)
    eq(policy.act(nim, state, rng), (2, 1), "the search did not find Nim's winning move")

    chess = uge.get_game("chess")
    mate_in_one = C.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    values = [(search_value(chess, chess.apply(mate_in_one, m), 1), m) for m in chess.legal(mate_in_one)]
    check(max(v for v, _ in values) > 1000, "the chess search does not see a mate in one")


GROUPS = {
    "perft": lambda full: test_perft(full),
    "chess": lambda full: test_chess_details(),
    "codebooks": lambda full: test_codebooks(),
    "partition": lambda full: test_partition(),
    "games": lambda full: test_games(),
    "mixed": lambda full: test_mixed_phi(),
    "roundtrip": lambda full: test_roundtrip(full),
    "referee": lambda full: test_referee(),
    "determinism": lambda full: test_determinism(),
    "teachers": lambda full: test_teachers(),
}


def main(argv: list[str]) -> int:
    full = "--full" in argv
    names = [a for a in argv if not a.startswith("-")] or list(GROUPS)
    unknown = [n for n in names if n not in GROUPS]
    if unknown:
        print(f"unknown group(s) {unknown}; have {list(GROUPS)}", file=sys.stderr)
        return 2
    for name in names:
        print(f"[{name}]")
        t0 = time.perf_counter()
        GROUPS[name](full)
        print(f"  ({time.perf_counter() - t0:.2f}s)")
    print(f"\n{CHECKS} checks, {len(FAILURES)} failed")
    for f in FAILURES:
        print(f"  {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
