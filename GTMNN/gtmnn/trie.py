"""A trie over game-theoretic structure, general at the root and specific at the
leaves.

The root does nothing. It asserts no property, discriminates on nothing, and
guarantees only what is true of every finite game: a mixed equilibrium exists
(Nash 1950). Each level down adds exactly ONE fact, and the facts are ordered so
that every step strictly narrows the class:

    depth 0   .                       every finite game
    depth 1   players                 how many decision makers
    depth 2   payoff                  common to all / player-specific / opposed
    depth 3   structure               congestion / matrix / arbitrary
    depth 4   monotone                non-increasing in load / not
    depth 5   potential               exact / ordinal / none

The point of the ordering is that A NODE'S GUARANTEES ARE A FUNCTION OF ITS PATH
AND NOTHING ELSE. Descending is accumulating premises; the theorems that fire at
a node are exactly those whose hypotheses the prefix satisfies. The path IS the
proof, which is why this is a trie and not a lookup table -- an internal node is
a real answer ("a two-player zero-sum game" already tells you fictitious play
converges) and you can act on it before reaching a leaf.

That is the trie half, the same one GREN/gren/radix.py takes: a terminal flag on
any node, internal ones included, so a general class can be a strict prefix of a
specific one. No path compression here, deliberately -- GREN compresses because
its signatures are long and sparse and only a few tokens discriminate. These
paths are five deep and every level discriminates, so compression would buy
nothing and would cost the property that DEPTH EQUALS NUMBER OF FACTS ASSERTED,
which is the whole readout.

What makes this more than a diagram: each node states which solver is
*guaranteed* to work there, and those are falsifiable. `gtmnn.cli trie` runs
every solver on games drawn from every class and checks the predictions.

SCOPE. A trie belongs to GAME THEORY AND GAME CLASSIFICATION, and to nothing
else in this system. There are exactly two in the repository and both stay on
that side of the line:

    GREN/gren/radix.py   which game IS this -- indexed by what the game refuses
    this module          what may I ASSUME about it -- indexed by its structure

Neither touches prediction, features, micro weights or credit. The only hook
into model code is `model.solver_for`, which classifies the payoff in hand to
pick a solver, and that is game classification. Nothing here should acquire a
second job: the moment a trie starts carrying activations or standing in for a
network, it stops being an index and the guarantees stop meaning anything.
"""
from dataclasses import dataclass, field

AXES = ("players", "payoff", "structure", "monotone", "potential")

# Theorems, and the prefix each one needs. A theorem fires at a node when its
# requirement is a subset of the facts the path has asserted.
THEOREMS = [
    ("Nash 1950", {},
     "a mixed-strategy equilibrium exists in every finite game"),
    ("von Neumann 1928", {"players": "2", "payoff": "opposed"},
     "a minimax value exists; fictitious play converges to it"),
    ("Rosenthal 1973", {"structure": "congestion", "payoff": "common",
                        "monotone": "non-increasing", "potential": "exact"},
     "exact potential: pure NE exists, finite improvement property, "
     "best response TERMINATES"),
    ("Milchtaich 1996", {"structure": "congestion", "payoff": "player-specific",
                         "monotone": "non-increasing"},
     "pure NE exists and an improving path reaches one, but an arbitrary "
     "best-response path MAY CYCLE"),
    ("Hart & Mas-Colell 2000", {},
     "regret matching drives the empirical play to the correlated equilibria"),
    ("Monderer & Shapley 1996", {"potential": "exact"},
     "an exact potential alone gives the finite improvement property -- "
     "monotone load is Rosenthal's hypothesis, not the FIP's"),
    ("Monderer & Shapley 1996", {"potential": "ordinal"},
     "an ordinal potential is enough for the finite improvement property"),
]

# The deepest guarantee wins. A solver is recommended only where something
# PROVES it terminates or converges; elsewhere the fallback is the solver that
# converges on everything.
SOLVER_RULES = [
    ({"potential": "exact"}, "best_response",
     "terminates at a pure NE by the finite improvement property"),
    ({"potential": "ordinal"}, "best_response",
     "the finite improvement property survives an ordinal potential"),
    ({"players": "2", "payoff": "opposed"}, "fictitious",
     "converges to the minimax value on a two-player zero-sum game"),
    ({"structure": "congestion", "payoff": "player-specific"}, "regret_plus",
     "best response may cycle here; regret matching converges regardless"),
]
FALLBACK = ("regret_plus", "nothing stronger is provable; regret matching "
            "converges on every finite game")


@dataclass
class Node:
    token: object = None          # the fact this edge asserted, None at the root
    depth: int = 0
    parent: int = -1
    children: dict = field(default_factory=dict)
    terminal: bool = False
    games: list = field(default_factory=list)


class GameTrie:
    """Games indexed by what is provable about them."""

    def __init__(self, axes=AXES):
        self.axes = tuple(axes)
        self.nodes = [Node()]

    # ------------------------------------------------------------------ build
    def _child(self, node, token):
        c = self.nodes[node].children.get(token)
        if c is None:
            c = len(self.nodes)
            self.nodes.append(Node(token=token, depth=self.nodes[node].depth + 1,
                                   parent=node))
            self.nodes[node].children[token] = c
        return c

    def insert(self, facts, name):
        """Walk the axes in order, creating what is missing. The game is recorded
        at the deepest node its facts reach; a fact it does not state stops the
        descent, so a partially-known game lands on an INTERNAL node and is still
        a real answer there."""
        node = 0
        for axis in self.axes:
            v = facts.get(axis)
            if v is None: break
            node = self._child(node, (axis, v))
        self.nodes[node].terminal = True
        if name not in self.nodes[node].games: self.nodes[node].games.append(name)
        return node

    # ------------------------------------------------------------------ query
    def path(self, node):
        out = []
        while node > 0:
            out.append(self.nodes[node].token)
            node = self.nodes[node].parent
        return list(reversed(out))

    def facts(self, node):
        return dict(self.path(node))

    def walk(self, facts):
        """Descend as far as the facts allow; return every node passed through.

        The whole list is the answer, not just the last element: each node is a
        class the game genuinely belongs to, and the guarantees accumulate."""
        node, seen = 0, [0]
        for axis in self.axes:
            v = facts.get(axis)
            if v is None: break
            c = self.nodes[node].children.get((axis, v))
            if c is None: break
            node = c; seen.append(node)
        return seen

    def guarantees(self, node):
        """Exactly the theorems whose hypotheses this node's PATH satisfies."""
        f = self.facts(node)
        return [(n, why) for n, req, why in THEOREMS
                if all(f.get(k) == v for k, v in req.items())]

    def solver(self, node):
        """(method, why). The deepest rule whose premises the path meets."""
        f = self.facts(node)
        best, depth = None, -1
        for req, method, why in SOLVER_RULES:
            if all(f.get(k) == v for k, v in req.items()) and len(req) > depth:
                best, depth = (method, why), len(req)
        return best or FALLBACK

    def terminals(self):
        return [i for i, n in enumerate(self.nodes) if n.terminal]

    def render(self, node=0, prefix="", out=None):
        out = out if out is not None else []
        n = self.nodes[node]
        if node == 0:
            out.append(". " + " " * 34 + "every finite game" +
                       ("   [" + ", ".join(n.games) + "]" if n.games else ""))
        else:
            axis, v = n.token
            label = f"{axis}={v}"
            games = ("   [" + ", ".join(n.games) + "]") if n.games else ""
            out.append(f"{prefix}{label}{games}")
        kids = list(n.children.values())
        for k, c in enumerate(kids):
            last = k == len(kids) - 1
            out.append_prefix = None
            self.render(c, prefix + ("    " if node == 0 else "  ") + ("+-- "), out)
        return out


# --------------------------------------------------------- classifying a game
def classify(payoff, ctx=None, seats=None):
    """The facts a payoff states about itself.

    Every one of these is asked of the payoff object rather than hard-coded per
    class, so a new game joins the trie by answering the same questions and
    nothing here has to change."""
    name = getattr(payoff, "name", type(payoff).__name__)
    summary = getattr(payoff, "summary", "loads")
    f = {}
    f["players"] = "2" if summary == "profile" else ("n" if (seats or 3) > 2 else "2")
    if getattr(payoff, "zero_sum", None) is True: f["payoff"] = "opposed"
    elif getattr(payoff, "player_specific", None) is True: f["payoff"] = "player-specific"
    else: f["payoff"] = "common"
    f["structure"] = "matrix" if summary == "profile" else "congestion"
    f["monotone"] = "non-increasing" if getattr(payoff, "monotone_in_load", False) else "n/a"
    if payoff.is_exact_potential(): f["potential"] = "exact"
    elif getattr(payoff, "ordinal_potential", False): f["potential"] = "ordinal"
    else: f["potential"] = "none"
    return name, f


def build(payoffs=None, seats=8):
    """The catalogue, indexed."""
    from gtmnn import payoff as pay
    payoffs = payoffs if payoffs is not None else [
        pay.CorrectnessCongestion(), pay.BeliefCongestion(),
        pay.Inverted(pay.CorrectnessCongestion()),
        pay.RockPaperScissors(), pay.StagHunt(), pay.PrisonersDilemma()]
    t = GameTrie()
    placed = {}
    for p in payoffs:
        name, f = classify(p, seats=seats)
        placed[name] = (t.insert(f, name), f)
    return t, placed


# ------------------------------------------------- a network at every node
class TrieNetworks:
    """Each node of a GameTrie may own a population. Reaching a node is how you
    find the network to query.

    The trie still only classifies -- it holds no activations and computes no
    prediction. It says WHICH population answers, and the population answers.

    The mechanism that makes this worth having is BACKOFF, and it is the reason
    a terminal flag sits on internal nodes as well as leaves. A leaf that has
    never been trained is not an answer; the general class above it is. So a
    query walks as deep as the facts allow, then climbs back up to the nearest
    ancestor with a population that has actually played, and asks that one. "A
    congestion game with a common payoff" is a usable answer while "...with an
    exact potential and non-increasing load" is still cold.

    That is the trie half doing real work rather than decorating a lookup: a
    general class is a strict prefix of a specific one, and the prefix is
    queryable on its own.
    """

    def __init__(self, trie, F, n=128, R=8, H=6, K=4, seed=0, warm=1.0):
        self.trie, self.F = trie, F
        self.n, self.R, self.H, self.K, self.seed = n, R, H, K, seed
        self.warm = warm            # plays a node needs before it counts as trained
        self.pools = {}             # node -> MicroPool
        self.plays = {}             # node -> stage games this node has played

    def attach(self, node, alphabet_size=2):
        from gtmnn.micro import MicroPool
        p = self.pools.get(node)
        if p is None:
            p = MicroPool(self.n, self.F, self.R, self.H, self.K,
                          seed=self.seed + node, alphabet_size=alphabet_size)
            self.pools[node] = p; self.plays[node] = 0.0
        return p

    def trained(self, node):
        return node in self.pools and self.plays.get(node, 0.0) >= self.warm

    def resolve(self, facts, require_trained=True):
        """(node, pool, backed_off). The deepest node the facts reach that has a
        population worth asking; None when nothing on the path has one."""
        seen = self.trie.walk(facts)
        deepest = seen[-1]
        for nd in reversed(seen):
            if (self.trained(nd) if require_trained else nd in self.pools):
                return nd, self.pools[nd], nd != deepest
        return None, None, False

    def note(self, node, games=1.0):
        self.plays[node] = self.plays.get(node, 0.0) + games

    def describe(self):
        out = []
        for nd in sorted(self.pools):
            f = self.trie.facts(nd)
            out.append({"node": nd, "depth": self.trie.nodes[nd].depth,
                        "path": "/".join(f"{k}={v}" for k, v in f.items()) or ".",
                        "plays": self.plays.get(nd, 0.0),
                        "trained": self.trained(nd),
                        "games": list(self.trie.nodes[nd].games)})
        return out
