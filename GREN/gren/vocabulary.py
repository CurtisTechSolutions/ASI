"""Which features explain which refusal -- measured, not named, and aligned.

CyclicCortex groups a game's inputs into named blocks (GRID_MOVE, OCCUPANCY,
PIECE_TYPES...) and two games share input slots when their blocks share a name.
That is transfer by spelling. I wrote both names, so of course they matched.

This module earns the sharing. For every feature DIMENSION it asks: does this
number separate the moves the game refuses with code C from the moves it
accepts? A dimension that does is a description of C, whatever I called it, and
belongs in C's slots -- where every other game refusing C also writes. A
dimension that separates no refusal stays private to its game.

That split is the p_valid / grade split made structural: legality is a property
of the rule and transfers, quality is a property of the game and does not. A
dimension that predicts no refusal is a quality dimension.

Keying by code alone is NOT enough, and getting it wrong is silent. Chess's
GRID_MOVE and checkers' FORCED_CAPTURE both explain WRONG_PATTERN, so both would
write slot 0 of WRONG_PATTERN -- one a file offset, the other a boolean about
jumps. Sharing a name without sharing a meaning aliases unrelated quantities
onto one weight, which is worse than not sharing at all.

Ranking within a code does not fix it either; that was tried and measured. See
the note above `align`, which does: dimensions are matched ACROSS games by their
whole legality signature, so a slot holds one quantity rather than one rank.

Separation is AUC (Mann-Whitney), scale-free, compared against the same
statistic on SHUFFLED labels. Without that floor every dimension scores above
zero on finite samples and the assignment is noise with a threshold on it.
"""
import random

MARGIN = 0.10    # a dimension must beat its own shuffled floor by this much
TAU_ALIGN = 0.80 # and match another game's dimension this closely to share a slot


def _auc(pos, neg):
    """P(a random positive ranks above a random negative), ties at 0.5."""
    if not pos or not neg: return 0.5
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n = len(xs); ranks = [0.0] * n; i = 0
    while i < n:                              # average ranks within a tie group
        j = i
        while j + 1 < n and xs[j + 1][0] == xs[i][0]: j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1): ranks[t] = avg
        i = j + 1
    sp = sum(ranks[t] for t in range(n) if xs[t][1] == 1)
    np_, nn = len(pos), len(neg)
    return (sp - np_ * (np_ + 1) / 2.0) / (np_ * nn)


def _signed(pos, neg):
    """2*(AUC - 0.5): -1..1, positive when a LARGER value means MORE refused."""
    return (_auc(pos, neg) - 0.5) * 2.0


def sample(oracle, adapter, rng, states=60, k=12):
    """Probe the game and record, per move, its refusal code and its features.

    Returns (rows, blocks); a row is (code_or_None, {block: [values]}).

    Legality comes from `is_legal`, never from `why`. `why` is a DIAGNOSIS: it
    assumes refusal and falls through to a default code, so reading legality off
    it labels every move illegal. That mistake gave chess a zero-size legal class
    and every block a separation of exactly 0.000, and gave go three blocks that
    separated REPETITION at a flawless 1.000 -- off the handful of passes that
    were the only rows it called legal.
    """
    g = oracle.game
    rows, blocks = [], {}
    for _ in range(states):
        st = (g.random_state(rng) if hasattr(g, "random_state")
              else g.new(seed=rng.randrange(10 ** 6)))
        for mv in oracle.candidates(st, rng, k):
            try:
                feats = adapter.generalise(st, mv)
                legal = g.is_legal(st, mv)
                code = None if legal else oracle.why(st, mv)[0]
            except Exception:
                continue
            rows.append((code, feats))
            for b, v in feats.items(): blocks[b] = max(blocks.get(b, 0), len(v))
    return rows, blocks


def columns(rows, blocks):
    """Flatten to (block, index) -> the column of values over all rows."""
    cols = {}
    for b, w in sorted(blocks.items()):
        for i in range(w):
            cols[(b, i)] = [((f.get(b) or ())[i] if i < len(f.get(b) or ()) else 0.0)
                            for _, f in rows]
    return cols


def associate(rows, cols, trials=8, seed=0, min_class=10):
    """(block,i) -> {code: (signed_separation, noise_floor)}.

    The floor is the strongest separation the SAME column reaches against
    SHUFFLED labels over `trials` shuffles, so it absorbs the optimism of picking
    the best code afterwards.
    """
    rng = random.Random(seed)
    codes = sorted({c for c, _ in rows if c})
    n = len(rows)
    legal = [t for t in range(n) if rows[t][0] is None]
    hits = {c: [t for t in range(n) if rows[t][0] == c] for c in codes}
    out = {}
    for key, col in cols.items():
        per = {}
        for c in codes:
            hit = hits[c]
            if len(hit) < min_class or len(legal) < min_class: continue
            s = _signed([col[t] for t in hit], [col[t] for t in legal])
            floor = 0.0
            pool = hit + legal
            for _ in range(trials):
                sh = list(pool); rng.shuffle(sh)
                floor = max(floor, abs(_signed([col[t] for t in sh[:len(hit)]],
                                               [col[t] for t in sh[len(hit):]])))
            per[c] = (s, floor)
        out[key] = per
    return out


def explains(assoc, margin=MARGIN):
    """(block,index) -> (code_or_None, signed_separation, noise_floor).

    Which refusal each dimension best describes, for the report. A dimension
    that describes several is describing the board rather than the rule, and the
    strongest claim is the honest one to print. This does NOT decide slots --
    `align` does, and on the whole signature rather than on the winner.
    """
    detail = {}
    for key, per in assoc.items():
        c0, s0, f0 = None, 0.0, 0.0
        for c, (s, f) in per.items():
            if abs(s) - f > abs(s0) - f0: c0, s0, f0 = c, s, f
        detail[key] = (c0 if (c0 and abs(s0) - f0 >= margin) else None,
                       round(s0, 3), round(f0, 3))
    return detail


# --------------------------------------------------------------- alignment
# Ranking within a code is NOT enough, and the failure is silent. Measured:
# chess's strongest OCCUPIED_TARGET predictor is "target holds my own piece"
# and checkers' is "target is empty", so slot 0 of OCCUPIED_TARGET held two
# different quantities and chess's weight for it was applied to checkers'
# opposite. Chess-to-checkers transfer went from +0.286 under the hand-written
# blocks to -0.062: worse than not sharing at all.
#
# So dimensions are matched across games by their whole LEGALITY SIGNATURE --
# the signed separation against every code in the shared vocabulary. Two
# dimensions describing the same quantity refuse the same things in the same
# direction, whatever game they sit in and whatever I named them. A negative
# cosine means the same quantity inverted, which is a match with a sign.

def signature(assoc, codes):
    """(block,i) -> the signed-separation vector over `codes`."""
    return {k: [per.get(c, (0.0, 0.0))[0] for c in codes] for k, per in assoc.items()}


def _cos(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na < 1e-9 or nb < 1e-9: return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _strength(assoc_k):
    return max((abs(s) - f for s, f in assoc_k.values()), default=0.0)


def align(per_game, codes, tau=TAU_ALIGN, margin=MARGIN, min_games=2):
    """Group dimensions ACROSS games into shared slots.

    `per_game`: {game: assoc} from associate(). Returns
      (slots, private) where a slot is
        {"name": str, "code": str, "dims": {game: (block, index, sign)}}
      and `private` is {game: [(block, index)]} for everything unmatched.

    A slot holds at most one dimension per game -- two dimensions of the same
    game are by construction not the same quantity. Matching is greedy over
    similarity, which is the standard approximation; the alternative is an
    assignment problem whose exact answer is not worth the code for four games.
    """
    pool = []
    for g, assoc in per_game.items():
        sig = signature(assoc, codes)
        for k, v in sig.items():
            s = _strength(assoc[k])
            if s >= margin: pool.append((g, k, v, s))
    pairs = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if pool[i][0] == pool[j][0]: continue
            c = _cos(pool[i][2], pool[j][2])
            if abs(c) >= tau: pairs.append((abs(c), c, i, j))
    pairs.sort(key=lambda t: (-t[0], t[2], t[3]))

    slot_of = {}                     # pool index -> slot id
    slots = []                       # [{game: (key, sign), ...}]
    for _, c, i, j in pairs:
        si, sj = slot_of.get(i), slot_of.get(j)
        sgn = 1 if c >= 0 else -1
        if si is None and sj is None:
            slots.append({pool[i][0]: (pool[i][1], 1), pool[j][0]: (pool[j][1], sgn)})
            slot_of[i] = slot_of[j] = len(slots) - 1
        elif si is None or sj is None:
            have, add = (sj, i) if si is None else (si, j)
            g = pool[add][0]
            if g in slots[have]: continue          # that game already wrote here
            ref = i if si is not None else j
            slots[have][g] = (pool[add][1], slots[have][pool[ref][0]][1] * sgn)
            slot_of[add] = have
        # both already placed: leave them, merging two slots could collide games

    out, used = [], set()
    for sid, members in enumerate(slots):
        if len(members) < min_games: continue
        best, bs = None, -1.0
        for g, (k, _) in members.items():
            for c in codes:
                s, f = per_game[g][k].get(c, (0.0, 0.0))
                if abs(s) - f > bs: best, bs = c, abs(s) - f
        out.append({"code": best, "strength": round(bs, 3),
                    "dims": {g: (k[0], k[1], sg) for g, (k, sg) in members.items()}})
        for g, (k, _) in members.items(): used.add((g, k))
    out.sort(key=lambda s: (s["code"], -s["strength"]))
    seen = {}
    for s in out:
        n = seen.get(s["code"], 0); seen[s["code"]] = n + 1
        s["name"] = f"{s['code']}/{n}"

    private = {}
    for g, assoc in per_game.items():
        private[g] = [k for k in sorted(assoc) if (g, k) not in used]
    return out, private


def layouts(slots, private):
    """Turn alignment into the per-game {key: [(block, index, sign)]} the cortex
    consumes. A shared slot is ONE input, so its width is 1 -- the width comes
    from how many slots agree, not from padding."""
    lay = {}
    for s in slots:
        for g, (b, i, sg) in s["dims"].items():
            lay.setdefault(g, {})[s["name"]] = [(b, i, sg)]
    for g, keys in private.items():
        if keys: lay.setdefault(g, {})[f"_{g}"] = [(b, i, 1) for b, i in keys]
    return lay


def build_aligned(oracles, codes, adapters=None, seed=0, states=60, k=12,
                  margin=MARGIN, tau=TAU_ALIGN):
    """The whole derivation: probe, associate, align, lay out.

    Returns (layouts, slots, private, detail) where `detail` is the per-game
    explains() map, for the report.
    """
    per_game, detail = {}, {}
    for n, o in oracles.items():
        a = (adapters or {}).get(n, o.game)
        rng = random.Random(seed)
        rows, blocks = sample(o, a, rng, states=states, k=k)
        cols = columns(rows, blocks)
        per_game[n] = associate(rows, cols, seed=seed)
        detail[n] = explains(per_game[n], margin=margin)
    slots, private = align(per_game, codes, tau=tau, margin=margin)
    return layouts(slots, private), slots, private, detail
