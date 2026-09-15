"""Goal regression: work backwards from the end goal to subgoals.

Forward search asks "where can I get from here?". Regression asks "what would
have to be true for the goal to hold?", and keeps asking until the answer is
something that is already true. It is how chess endgame tablebases are built --
backwards from mate -- and it is what DESIGN.md 8.3's STRIPS operators were for.

The hard part of regression is normally writing the operators: every action needs
its preconditions declared by hand, and that hand is where the domain knowledge
smuggles itself in. GREN does not need them written, because it already
discovered them:

    A REFUSAL CODE IS A PRECONDITION VIOLATION.

`BLOCKED_PATH` means move(from,to) requires a clear path. `CONSTRAINT_ROW` means
place(x,y,v) requires v absent from row y. `OCCUPIED_TARGET` means the square must
be free. The oracle's `why()` answers "what blocks this action?" -- which is
exactly the question regression asks at every step. The preconditions are
measured, not authored.

One limitation, stated rather than hidden: `why()` short-circuits and returns the
FIRST code that fires, not every violated precondition. That is enough for
regression -- fix the blocker, ask again -- but it makes the regression path
order-dependent, and a move blocked two ways takes two rounds to reveal both.
`blockers()` below recovers the full set where a game can be probed for it.
"""
import random
from dataclasses import dataclass, field


@dataclass
class Subgoal:
    """Something that must become true. `code` is the refusal it would clear."""
    code: str
    action: object = None           # the action this blocks, when it is action-specific
    depth: int = 0
    why: str = ""

    def __repr__(self):
        return f"Subgoal({self.code}{'' if self.action is None else f' for {self.action}'})"


@dataclass
class Plan:
    """A regression, front to back: the goal, then what had to be true for it."""
    goal: str
    achieved: bool
    action: object = None
    chain: list = field(default_factory=list)      # Subgoal, outermost first
    expanded: int = 0                              # states looked at
    note: str = ""

    def to_dict(self):
        return {"goal": self.goal, "achieved": self.achieved,
                "action": str(self.action) if self.action is not None else None,
                "chain": [s.code for s in self.chain], "expanded": self.expanded,
                "note": self.note}


def blocker(oracle, state, action):
    """The first precondition this action violates, or None if it is applicable.

    This is the whole regression step, and it is one call into what GREN already
    knows."""
    try:
        if oracle.game.is_legal(state, action): return None
        code, why = oracle.why(state, action)
        return Subgoal(code=code, action=action, why=why or "")
    except Exception as e:
        return Subgoal(code="ERROR", action=action, why=repr(e))


def blockers(oracle, state, action, codes=None):
    """Every precondition the action violates, not just the first.

    `why()` short-circuits, so this re-asks under each single-code hypothesis
    where the oracle supports it. Falls back to the first blocker otherwise --
    honestly incomplete rather than silently so."""
    b = blocker(oracle, state, action)
    if b is None: return []
    out = [b]
    probe = getattr(oracle, "violations", None)
    if probe is not None:
        try:
            return [Subgoal(code=c, action=action) for c in probe(state, action)] or out
        except Exception:
            return out
    return out


# ------------------------------------------------------------------ regression
def regress(oracle, state, goal_test, actions_fn, max_depth=4, rng=None,
            order=None, budget=20000):
    """Work backwards from `goal_test` to the actions that would satisfy it.

    For each candidate action that WOULD achieve the goal, ask what currently
    blocks it. No blocker means the action is applicable now and the plan is one
    step. A blocker becomes a subgoal, and the search recurses on the actions
    that would clear it.

    Returns a Plan whose `chain` is the subgoals outermost-first: the goal, what
    had to be true for it, what had to be true for that."""
    rng = rng or random.Random(0)
    expanded = [0]

    def go(st, depth, seen):
        if expanded[0] >= budget:
            return Plan("budget", False, expanded=expanded[0], note="budget exhausted")
        cands = actions_fn(st)
        if order: cands = order(st, cands)
        blocked = []
        for a in cands:
            expanded[0] += 1
            if expanded[0] >= budget: break
            b = blocker(oracle, st, a)
            if b is None:
                nxt = oracle.game.apply(st, a)
                if goal_test(nxt):
                    return Plan("goal", True, action=a, expanded=expanded[0])
                if depth + 1 < max_depth:
                    key = _key(nxt)
                    if key not in seen:
                        seen.add(key)
                        sub = go(nxt, depth + 1, seen)
                        if sub.achieved:
                            sub.chain.insert(0, Subgoal(code="STEP", action=a, depth=depth))
                            sub.action = sub.action or a
                            sub.expanded = expanded[0]
                            return sub
            else:
                blocked.append(b)
        if blocked and depth == 0:
            # Nothing was applicable. The blockers ARE the subgoals: clear one of
            # these and the goal becomes reachable.
            counts = {}
            for b in blocked: counts[b.code] = counts.get(b.code, 0) + 1
            chain = [Subgoal(code=c, depth=0, why=f"blocks {n} candidate action(s)")
                     for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
            return Plan("goal", False, chain=chain, expanded=expanded[0],
                        note="no applicable action; these preconditions are what stand in the way")
        return Plan("goal", False, expanded=expanded[0])

    return go(state, 0, {_key(state)})


def _key(state):
    for attr in ("b", "g"):
        v = getattr(state, attr, None)
        if v is not None:
            return (tuple(tuple(r) for r in v), getattr(state, "turn", None))
    return repr(state)


# --------------------------------------------------- regression as a heuristic
def blocker_histogram(oracle, state, actions):
    """Which preconditions stand between here and every candidate action.

    This is regression used as a MEASUREMENT rather than a search: the codes that
    block the most actions are the ones worth clearing, and that ranking is a
    plan sketch derived from nothing but what the game refuses."""
    out = {}
    for a in actions:
        b = blocker(oracle, state, a)
        k = "APPLICABLE" if b is None else b.code
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def most_constrained(oracle, state, actions, key):
    """Order candidates by how few ways there are to satisfy them.

    Regression's standard move: attack the subgoal with the fewest achievers
    first, because it is the one most likely to fail and cheapest to refute."""
    groups = {}
    for a in actions:
        if blocker(oracle, state, a) is None:
            groups.setdefault(key(a), []).append(a)
    return sorted(groups.items(), key=lambda kv: len(kv[1]))


# ------------------------------------------------------- what regression buys
def selectivity(oracle, states, actions_fn, subgoal_test):
    """The fraction of candidate actions that CLEAR the subgoal.

    This is the number that predicts whether regressing is worth it. A subgoal
    prunes exactly what it excludes: one that 96% of actions fail removes 96% of
    the expensive work, and one that 62% of actions pass removes almost nothing.
    Measured on chess mate-in-1 -- `gives check` clears 4% of moves in positions
    from real play and 62% in positions with pieces scattered at random, and the
    speedups are 7.8x and 1.7x respectively."""
    tot = clear = 0
    for st in states:
        for a in actions_fn(st):
            tot += 1
            if subgoal_test(st, a): clear += 1
    return clear / tot if tot else 0.0


def conjunctive(goal_parts, cheap_first=True):
    """Order a conjunctive goal so the cheap, selective part is asked first.

    `goal_parts` is [(cost, selectivity, test)]. Regression over a conjunctive
    goal is EXACTLY what a short-circuiting `and` does when the cheap conjunct is
    written first -- which is why it is easy to have done it already without
    noticing. Chess's `is_mate` is `in_check() and not legal_moves()`, and the
    first attempt to measure regression against it scored 1.0x because the
    baseline had regressed all along.

    Expected work for an ordering is sum over parts of (cost_i * prod of the
    selectivities before it), so the cheapest-per-unit-pruning part goes first."""
    scored = sorted(goal_parts, key=lambda p: (p[0] * p[1], p[0]))
    return scored if cheap_first else list(reversed(scored))


def expected_work(order):
    """Expected cost of a conjunctive goal in this order: each part is paid for
    only when everything before it passed."""
    total, reach = 0.0, 1.0
    for cost, sel, _ in order:
        total += cost * reach
        reach *= sel
    return total
