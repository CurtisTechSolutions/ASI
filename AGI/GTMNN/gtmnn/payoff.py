"""The game catalogue. Nothing downstream knows which game it is holding.

`loads` is indexed by GLOBAL SYMBOL ID, with one final entry counting
abstentions -- two seats naming "e" from different repertoire slots must count
as the same load. This is the only place slot-space and symbol-space meet, and
getting it wrong silently turns the congestion game into a vote.
"""
import math

ABSTAIN_SLOT = None             # set per game: it is the last slot, index K


class Payoff:
    name = "payoff"
    # A congestion payoff summarises the profile as a LOAD HISTOGRAM over symbol
    # ids; a matrix fixture needs the opponent's actual action. `summary` says
    # which, and StageGame.loads() hands over the right one.
    summary = "loads"

    # What the game states about ITSELF. gtmnn/trie.py asks these rather than
    # hard-coding a class list, so a new game joins the index by answering the
    # same questions and nothing in the trie changes.
    zero_sum = False            # one player's gain is another's loss
    player_specific = False     # two players can value the same action differently
    monotone_in_load = False    # payoff non-increasing in how many chose it
    ordinal_potential = False   # weaker than exact, still gives the FIP

    def utility(self, seat, action_slot, loads, ctx): raise NotImplementedError
    def is_exact_potential(self): return False
    def potential(self, loads, ctx): return None


class CorrectnessCongestion(Payoff):
    """The TRAINING game. The label is known, so every seat faces the same
    function of (action, load):

        u(a) =  B / n_a   if a == y
             = -lam       if a is wrong and not ABSTAIN
             =  0         if a == ABSTAIN

    Identical across players and non-increasing in load, so by Rosenthal (1973)
    this is an exact potential game with

        Phi(loads) = B * H(n_y) - lam * W

    (H the harmonic number, W the wrong-count). Best-response dynamics therefore
    strictly increases Phi on every accepted deviation, Phi is bounded above by
    B*H(M), and the action space is finite: the solver TERMINATES, at a pure Nash
    equilibrium, always. Training never escalates to the meta-player.

    The split B/n_a is the entire specialisation mechanism. Being the ninth micro
    to say "e" after a space is worth B/9; being the only one that can call the
    "q" in "Iraq" is worth B. Micros are not told to specialise and are not
    regularised into it -- they are paid for it."""
    name = "correctness_congestion"
    monotone_in_load = True

    def utility(self, seat, action_slot, loads, ctx):
        sym = ctx.slot_symbol[seat][action_slot]
        if sym is None: return 0.0                       # ABSTAIN
        if sym == ctx.y:
            n = loads[sym]
            return ctx.B / n if n > 0 else ctx.B
        return -ctx.lam

    def is_exact_potential(self): return True

    def potential(self, loads, ctx):
        ny = loads[ctx.y] if ctx.y is not None and ctx.y < len(loads) - 1 else 0
        phi = sum(ctx.B / j for j in range(1, ny + 1))
        wrong = 0
        for s in range(len(loads) - 1):
            if s != ctx.y: wrong += loads[s]
        return phi - ctx.lam * wrong


class BeliefCongestion(Payoff):
    """The INFERENCE game. The label is unknown, so each seat scores an action by
    its own expectation:

        E_i[u(a)] = q_i(a) * B/n_a + (1 - q_i(a)) * (-lam)      a != ABSTAIN
        E_i[u(ABSTAIN)] = 0

    Still non-increasing in n_a, but now PLAYER-SPECIFIC: two seats on the same
    action can value it differently. By Milchtaich (1996) a pure Nash equilibrium
    still exists and an improvement path to one exists from any profile, but an
    arbitrary best-response path may cycle.

    This is the formal home of Research/CyclesAreAFeature.md. The cycle is not
    the solver failing to converge -- the solver is behaving correctly on a game
    whose improvement paths genuinely loop. Iterating longer does not help.
    Leaving does (meta.py)."""
    name = "belief_congestion"
    monotone_in_load = True
    player_specific = True      # seat i scores an action by its OWN q_i

    def utility(self, seat, action_slot, loads, ctx):
        sym = ctx.slot_symbol[seat][action_slot]
        if sym is None: return 0.0
        q = ctx.q[seat][action_slot]
        n = loads[sym]
        n = n if n > 0 else 1
        return q * (ctx.B / n) - (1.0 - q) * ctx.lam

    def is_exact_potential(self): return False
    def potential(self, loads, ctx): return None


class Inverted(Payoff):
    """The 2NRL anti-game. Negating an exact potential leaves it exact, so the
    training game stays convergent throughout the inverted pass."""
    def __init__(self, inner):
        self.inner = inner
        self.name = "inverted_" + inner.name
        self.summary = inner.summary
        self.zero_sum = inner.zero_sum
        self.player_specific = inner.player_specific
        self.ordinal_potential = inner.ordinal_potential
        # Negating a payoff flips the direction of congestion: -B/n RISES with n.
        # Saying otherwise would hand Milchtaich a hypothesis it does not have.
        self.monotone_in_load = False

    def utility(self, seat, action_slot, loads, ctx):
        return -self.inner.utility(seat, action_slot, loads, ctx)

    def is_exact_potential(self): return self.inner.is_exact_potential()

    def potential(self, loads, ctx):
        p = self.inner.potential(loads, ctx)
        return None if p is None else -p


# ------------------------------------------------------- the matrix fixtures
class MatrixGame(Payoff):
    """A two-player normal-form fixture. Its `loads` IS the action profile."""
    name = "matrix"
    summary = "profile"
    table = ()

    def utility(self, seat, action_slot, loads, ctx):
        oa = loads[1 - seat]
        row, col = (action_slot, oa) if seat == 0 else (oa, action_slot)
        return self.table[row][col][seat]


class PrisonersDilemma(MatrixGame):
    """Activation sharing. A micro may share its hidden vector with a partner
    (who then reads it as R extra inputs) or withhold while still reading the
    partner's. T > Rw > P > S and 2Rw > T + S.

    One-shot, defection dominates. Repeated inside a standing coalition with
    tit-for-tat memory, cooperation is sustainable -- and cooperation is the
    population's ONLY route to a wider view, from R=8 to 2R=16, without growing
    a single weight matrix. It has to be earned from a partner, not allocated."""
    name = "prisoners_dilemma"
    ordinal_potential = True    # dominant strategies give a trivial ordinal potential
    T, Rw, P, S = 1.6, 1.0, 0.2, -0.4
    table = (((1.0, 1.0), (-0.4, 1.6)),
             ((1.6, -0.4), (0.2, 0.2)))


class StagHunt(MatrixGame):
    """Coalition commitment. (stag, stag) is payoff-dominant, (hare, hare) is
    risk-dominant -- exactly the tension of committing to a minority answer."""
    name = "stag_hunt"
    ordinal_potential = True    # (4,4)/(3,3) coordination: an ordinal potential exists
    table = (((4.0, 4.0), (0.0, 3.0)),
             ((3.0, 0.0), (3.0, 3.0)))


class RockPaperScissors(MatrixGame):
    """The cycle fixture. Zero-sum, three actions, unique equilibrium at uniform,
    no pure Nash, so best-response dynamics is GUARANTEED to cycle -- with period
    3 in sweeps. Not part of the prediction pipeline; it is what proves the cycle
    detector and the meta-player work.

    Genuine RPS structure does arise in the population and it is worth naming
    why: intransitive dominance among specialists is the normal case, not a
    pathology. A beats B where A's receptive field is informative, B beats C
    where B's is, C beats A where C's is. Nothing forbids the loop from closing,
    and when it does there is no single best micro to defer to."""
    name = "rock_paper_scissors"
    zero_sum = True
    table = (((0.0, 0.0), (-1.0, 1.0), (1.0, -1.0)),
             ((1.0, -1.0), (0.0, 0.0), (-1.0, 1.0)),
             ((-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)))


CATALOGUE = {c.name: c for c in (CorrectnessCongestion, BeliefCongestion,
                                 PrisonersDilemma, StagHunt, RockPaperScissors)}
