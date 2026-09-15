"""The suite. Standard library only: `python3 -m tests.test_distil`.

No pytest, for the same reason the rest of the package has no dependencies, and
because `selfedit.py` runs this file inside the sandbox to decide whether an edit
to the system's own source is allowed to land. That makes the suite part of the
control loop rather than a development convenience: a self-edit is accepted
exactly when these assertions still hold.

Where a result has a known closed form -- matching pennies, Shapley on an
additive game, the glove game -- the test asserts the closed form rather than a
golden value recorded from a previous run. A golden value only proves the code
still does what it did.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil import game
from distil.agent import Distil
from distil.casebook import Case, Casebook
from distil.clarify import (Clarification, Clarifier, Gap, first_step_actionable,
                            gaps as frame_gaps)
from distil.challenge import (Attack, Ground, Persistence, challenge, classify,
                              interrogate, premises)
from distil.compress import PRESERVE, Compressor
from distil.embed import HashEmbedder, Vocabulary
from distil.explore import Explorer, Idea, Origin
from distil.frame import (Framer, GameFrame, Horizon, Information, Payoff, Players,
                          PRIMITIVES, Solution, agenda, capabilities, classify as classify_game)
from distil.goals import GoalTree, Status, Verifier, checkability
from distil.mcp import McpRegistry, McpServer, McpTool
from distil.grade import grade_check, grade_python, gradeable
from distil.memory import Kind, Memory, Source, Trace
from distil.policy import BOUNDS, Policy
from distil.provider import LocalProvider, Message, auto, catalogue
from distil.reason import Reasoner, Role, State, role_of
from distil.sandbox import run_source, screen
from distil.seed import SEEDS, plant
from distil.selfedit import SelfEditor, _lost_names, count_tests
from distil.store import InProcessStore, TieredStore
from distil.toolsmith import (Toolbox, ToolSpec, Toolsmith, parse_reply, signature_of,
                              synthesise)
from distil.vector import centroid, cosine, normalise, spread
from distil.workspace import Workspace

TMP: list[Path] = []


def tmpdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="distil-test-"))
    TMP.append(d)
    return d


def fresh(seed: int = 0) -> Distil:
    return Distil(LocalProvider(), home=tmpdir(), seed=seed)


# --------------------------------------------------------------------------- #
# vectors
# --------------------------------------------------------------------------- #

def test_normalise_gives_unit_length():
    v = normalise([3.0, 4.0])
    assert abs(sum(x * x for x in v) - 1.0) < 1e-12
    assert normalise([0.0, 0.0]) == [0.0, 0.0], "the zero vector has no direction"


def test_cosine_bounds_and_orthogonality():
    a, b = normalise([1, 0]), normalise([0, 1])
    assert abs(cosine(a, a) - 1.0) < 1e-12
    assert abs(cosine(a, b)) < 1e-12
    assert abs(cosine(a, normalise([-1, 0])) + 1.0) < 1e-12


def test_cosine_handles_unnormalised_input():
    assert abs(cosine([3, 0], [5, 0]) - 1.0) < 1e-12


def test_spread_is_zero_for_identical_vectors():
    v = normalise([1, 2, 3])
    assert spread([v, v, v]) < 1e-12
    assert spread([normalise([1, 0]), normalise([0, 1])]) > 0.1


def test_centroid_of_empty_is_empty():
    assert centroid([]) == []


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #

def test_embedding_is_stable_across_embedder_instances():
    # The point of blake2b over hash(): a vector written today must match the
    # same text embedded in a different process tomorrow.
    a = HashEmbedder().embed("TypeError: unhashable type", learn=False)
    b = HashEmbedder().embed("TypeError: unhashable type", learn=False)
    assert a == b


def test_embedding_separates_related_from_unrelated():
    e = HashEmbedder()
    a = e.embed("TypeError: unhashable type dict while sorting")
    b = e.embed("TypeError unhashable type: dict raised when sorting keys")
    c = e.embed("the weather in Reykjavik is cold and windy")
    assert cosine(a, b) > 0.35
    assert cosine(a, c) < 0.15
    assert cosine(a, b) > cosine(a, c)


def test_character_ngrams_survive_a_typo():
    e = HashEmbedder()
    assert cosine(e.embed("ValueErrpr: invalid literal"),
                  e.embed("ValueError: invalid literal")) > 0.6


def test_empty_text_embeds_to_zero():
    assert set(HashEmbedder().embed("")) == {0.0}


def test_idf_makes_a_rare_token_count_more_than_a_common_one():
    v = Vocabulary()
    for _ in range(50):
        v.observe(["common"])
    v.observe(["rare"])
    assert v.idf("rare") > v.idf("common")


def test_vocabulary_round_trips():
    v = Vocabulary()
    v.observe(["a", "b"])
    w = Vocabulary.from_json(v.to_json())
    assert w.docs == v.docs and w.df == v.df


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #

def test_graded_recall_outranks_a_closer_but_refuted_memory():
    """The claim the memory layer rests on: a plain vector store returns the
    wrong answer here, and grading fixes it without deleting anything."""
    def top(weight):
        m = Memory(HashEmbedder(), Policy(recall_credibility_weight=weight))
        right = m.remember(Kind.ANSWER, "to merge two dicts use z = {**a, **b} which copies both")
        wrong = m.remember(Kind.ANSWER, "to merge two dicts use a.merge(b) which copies both")
        for _ in range(3):
            m.grade(right.id, 1.0, Source.SELF)
            m.grade(wrong.id, -1.0, Source.SELF)
        hit = m.recall("merge two dicts", k=2)[0]
        return hit.trace.text, hit.similarity

    unweighted, sim_u = top(0.0)
    weighted, sim_w = top(0.6)
    assert "a.merge(b)" in unweighted, "similarity alone should prefer the wrong answer here"
    assert "{**a, **b}" in weighted, "credibility should promote the verified answer"
    assert sim_w < sim_u, "and it should do so despite being less similar"


def test_refuted_memory_is_demoted_not_deleted():
    m = Memory(HashEmbedder())
    bad = m.remember(Kind.ANSWER, "use a.merge(b) to merge dicts")
    m.grade(bad.id, -1.0, Source.SELF)
    assert m.get(bad.id) is not None
    assert any(h.trace.id == bad.id for h in m.recall("merge dicts", k=5))


def test_near_duplicates_merge_instead_of_accumulating():
    m = Memory(HashEmbedder())
    t = m.remember(Kind.FACT, "sorted() is stable in CPython")
    again = m.remember(Kind.FACT, "sorted() is stable in CPython")
    assert again.id == t.id and again.seen == 2
    assert len(m.store.all()) == 1


def test_different_kinds_do_not_merge():
    m = Memory(HashEmbedder())
    a = m.remember(Kind.FACT, "sorted() is stable")
    b = m.remember(Kind.GOAL, "sorted() is stable")
    assert a.id != b.id


def test_user_grades_weigh_more_than_self_grades():
    p = Policy()
    a = Trace("a", Kind.ANSWER, "x", [1.0], "hash-v1", 0.0)
    b = Trace("b", Kind.ANSWER, "x", [1.0], "hash-v1", 0.0)
    a.grades = [(1.0, Source.SELF, 0.0)]
    b.grades = [(1.0, Source.USER, 0.0)]
    assert b.credibility(p) > a.credibility(p)


def test_credibility_shrinks_toward_the_prior():
    p = Policy()
    once = Trace("a", Kind.ANSWER, "x", [1.0], "hash-v1", 0.0)
    once.grades = [(1.0, Source.SELF, 0.0)]
    many = Trace("b", Kind.ANSWER, "x", [1.0], "hash-v1", 0.0)
    many.grades = [(1.0, Source.SELF, 0.0)] * 20
    ungraded = Trace("c", Kind.ANSWER, "x", [1.0], "hash-v1", 0.0)
    assert ungraded.credibility(p) == p.credibility_prior
    assert once.credibility(p) < many.credibility(p)


def test_credit_propagates_along_links_and_decays():
    m = Memory(HashEmbedder(), Policy(credit_decay=0.6, dedupe_threshold=0.999))
    root = m.remember(Kind.GOAL, "the originating goal about parsing")
    mid = m.remember(Kind.CHAIN, "a reasoning chain about tokenising", links=[root.id])
    leaf = m.remember(Kind.TOOL, "a tool that splits on commas reliably", links=[mid.id])
    m.grade(leaf.id, 1.0, Source.SELF)
    assert m.get(mid.id).mean_grade is not None, "one hop should be credited"
    assert m.get(root.id).mean_grade is not None, "two hops should be credited"
    assert m.get(mid.id).mean_grade > m.get(root.id).mean_grade, "credit must decay per hop"


def test_credit_propagation_terminates_on_a_cycle():
    m = Memory(HashEmbedder(), Policy(dedupe_threshold=0.999))
    a = m.remember(Kind.GOAL, "alpha goal about widgets")
    b = m.remember(Kind.GOAL, "beta goal about sprockets", links=[a.id])
    m.get(a.id).links.append(b.id)                      # a -> b -> a
    m.grade(b.id, 1.0, Source.SELF)                     # must not hang


def test_cross_embedder_traces_are_never_compared():
    m = Memory(HashEmbedder())
    t = m.remember(Kind.FACT, "a fact embedded by another backend")
    t.embedder = "some-other-model"
    m.store.touch(t)
    assert m.recall("a fact embedded by another backend", k=3) == []


def test_memory_round_trips_through_disk():
    m = Memory(HashEmbedder())
    t = m.remember(Kind.FACT, "sorted() is stable in CPython", meta={"k": 1})
    m.grade(t.id, 1.0, Source.USER)
    path = tmpdir() / "memory.jsonl"
    m.save(path)
    n = Memory(HashEmbedder())
    assert n.load(path) == 1
    restored = n.get(t.id)
    assert restored.text == t.text and restored.meta == {"k": 1}
    assert restored.mean_grade == 1.0
    assert not restored.verified, "a user grade is an opinion, not a verification"


def test_gaps_surface_disagreement_rather_than_emptiness():
    m = Memory(HashEmbedder())
    for i in range(4):
        t = m.remember(Kind.ANSWER, f"contradictory claim number {i} about parsing csv quoting")
        m.grade(t.id, -0.8, Source.SELF)
    for i in range(4):
        t = m.remember(Kind.ANSWER, f"settled claim number {i} about integer arithmetic overflow")
        m.grade(t.id, 1.0, Source.SELF)
    gaps = m.gaps(k=2)
    assert gaps, "a store with contradictions should report a gap"
    assert gaps[0]["credibility"] < 0.5


def test_in_process_store_search_filters_by_kind():
    store = InProcessStore()
    m = Memory(HashEmbedder(), store=store)
    m.remember(Kind.TOOL, "a tool for splitting text on commas")
    m.remember(Kind.FACT, "a fact about splitting text on commas")
    hits = m.recall("splitting text on commas", k=5, kinds=(Kind.TOOL,))
    assert len(hits) == 1 and hits[0].trace.kind == Kind.TOOL


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #

def test_policy_clamps_to_its_bounds():
    p = Policy(recency_halflife_days=1e9, branch_factor=-5, dedupe_threshold=2.0).clamp()
    assert p.recency_halflife_days == BOUNDS["recency_halflife_days"][1]
    assert p.branch_factor == BOUNDS["branch_factor"][0]
    assert p.dedupe_threshold == BOUNDS["dedupe_threshold"][1]


def test_mutation_changes_exactly_one_field_and_stays_in_bounds():
    rng = random.Random(4)
    base = Policy()
    for _ in range(60):
        child = base.mutate(rng, sigma=5.0)          # deliberately violent
        assert len(base.diff(child)) <= 1
        for name, (lo, hi) in BOUNDS.items():
            assert lo <= getattr(child, name) <= hi, name


def test_integer_fields_stay_integers():
    child = Policy().mutate(random.Random(1), sigma=3.0)
    assert isinstance(child.branch_factor, int) and isinstance(child.max_depth, int)


def test_policy_round_trips():
    p = Policy(branch_factor=5, credit_decay=0.3)
    path = tmpdir() / "policy.json"
    p.save(path)
    assert Policy.load(path).to_json() == p.to_json()


def test_corrupt_policy_file_falls_back_to_defaults():
    path = tmpdir() / "policy.json"
    path.write_text("{not json")
    assert Policy.load(path).to_json() == Policy().to_json()


# --------------------------------------------------------------------------- #
# game theory -- asserted against closed forms
# --------------------------------------------------------------------------- #

def test_matching_pennies_has_a_uniform_equilibrium():
    p, q, v = game.fictitious_play([[1, -1], [-1, 1]], iterations=4000)
    assert abs(p[0] - 0.5) < 0.02 and abs(q[0] - 0.5) < 0.02
    assert abs(v) < 0.02, "the value of matching pennies is zero"


def test_rock_paper_scissors_converges_to_uniform():
    p, _, v = game.fictitious_play([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], iterations=6000)
    assert all(abs(x - 1 / 3) < 0.03 for x in p)
    assert abs(v) < 0.03


def test_dominant_strategy_is_played_purely():
    p, _, _ = game.fictitious_play([[2, 3], [0, 1]], iterations=500)
    assert p[0] > 0.99, "row 1 dominates row 2"


def test_maximin_and_minimax_regret_can_disagree():
    m = [[10, -100], [1, 1]]
    assert game.maximin(m) == 1, "row 2 has the better worst case"
    assert game.minimax_regret(m) == 1


def test_regret_matrix_is_non_negative_with_a_zero_per_column():
    r = game.regret_matrix([[3, -4], [1, 0]])
    assert all(x >= 0 for row in r for x in row)
    for j in range(2):
        assert min(r[i][j] for i in range(2)) == 0


def test_expected_utility_respects_the_belief():
    m = [[1, 0], [0, 1]]
    assert game.expected_utility(m, [1.0, 0.0]) == [1.0, 0.0]
    assert game.expected_utility(m, [0.0, 1.0]) == [0.0, 1.0]


def test_choose_shifts_with_aversion():
    m = [[0.9, -0.9], [0.2, 0.2]]           # gamble vs. safe
    assert game.choose(m, [0.9, 0.1], aversion=0.0).index == 0, "expected utility takes the gamble"
    assert game.choose(m, [0.5, 0.5], aversion=1.0).index == 1, "regret aversion takes the safe row"


def test_ragged_and_empty_matrices_are_rejected():
    for bad in ([], [[]], [[1, 2], [3]]):
        try:
            game.maximin(bad)
            raise AssertionError(f"should have rejected {bad}")
        except ValueError:
            pass


def test_shapley_of_an_additive_game_is_the_individual_values():
    values = {"a": 3.0, "b": 5.0, "c": 2.0}
    got = game.shapley(list(values), lambda s: sum(values[x] for x in s))
    for k, v in values.items():
        assert abs(got[k] - v) < 1e-9


def test_shapley_is_efficient():
    """Efficiency: the shares sum to the value of the grand coalition. This is
    what stops credit being conjured out of a decomposition."""
    values = {"a": 1.0, "b": 4.0, "c": 2.0}
    def v(s):
        return sum(values[x] for x in s) + (2.0 if len(s) == 3 else 0.0)
    got = game.shapley(list(values), v)
    assert abs(sum(got.values()) - v(list(values))) < 1e-9


def test_shapley_gives_a_null_player_nothing():
    got = game.shapley(["a", "b", "null"], lambda s: len([x for x in s if x != "null"]))
    assert abs(got["null"]) < 1e-9


def test_glove_game_matches_the_textbook():
    """One left glove, two right: the scarce side is worth 2/3."""
    got = game.shapley(["L", "R1", "R2"],
                       lambda s: min(sum(1 for x in s if x == "L"),
                                     sum(1 for x in s if x.startswith("R"))))
    assert abs(got["L"] - 2 / 3) < 1e-9
    assert abs(got["R1"] - 1 / 6) < 1e-9


def test_regret_matching_converges_on_rock_paper_scissors():
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    rm = game.RegretMatching(["rock", "paper", "scissors"])
    for _ in range(3000):
        s = rm.strategy()
        rm.observe([sum(rps[i][j] * s[j] for j in range(3)) for i in range(3)])
    assert all(abs(x - 1 / 3) < 0.05 for x in rm.average_strategy())


def test_regret_matching_round_trips():
    rm = game.RegretMatching(["a", "b"])
    rm.observe([1.0, 0.0])
    back = game.RegretMatching.from_json(rm.to_json())
    assert back.regret == rm.regret and back.rounds == rm.rounds


def test_entropy_peaks_at_one_half():
    assert abs(game.entropy(0.5) - 1.0) < 1e-12
    assert game.entropy(0.0) == 0.0 and game.entropy(1.0) == 0.0
    assert game.entropy(0.9) < game.entropy(0.6) < game.entropy(0.5)


def test_information_gain_prefers_the_cheaper_of_two_equal_experiments():
    assert game.information_gain(0.5, 1.0) > game.information_gain(0.5, 2.0)


# --------------------------------------------------------------------------- #
# interrogation
# --------------------------------------------------------------------------- #

def test_circular_justification_is_caught_by_vector_not_string():
    e = HashEmbedder()
    answers = iter(["because it is faster",
                    "because it does less work per call",
                    "because it is faster than it was before"])
    chain = interrogate("rewrite the parser", lambda q, c: next(answers, ""), e)
    assert chain.terminal == Ground.CIRCULAR
    assert chain.depth == 3, "it should stop the moment the chain loops"


def test_a_testable_justification_grounds_the_chain():
    e = HashEmbedder()
    answers = iter(["because the p99 latency regressed",
                    "because a benchmark measured 400ms on the new path"])
    chain = interrogate("rewrite the parser", lambda q, c: next(answers, ""), e)
    assert chain.terminal == Ground.GROUNDED


def test_an_appeal_to_convention_is_labelled_assumed():
    assert classify("everyone should always do it this way", [], HashEmbedder()) == Ground.ASSUMED


def test_premises_find_the_load_bearing_clauses():
    found = premises("we must rewrite the parser because Python is always too slow")
    assert any("must" in p or "always" in p for p in found)


def test_challenge_spreads_across_premises_and_attack_types():
    qs = challenge("we must rewrite the parser in Rust because Python is always too slow", limit=6)
    assert len(qs) == 6
    assert len({q.attack for q in qs}) >= 3, "six variants of one attack is one question"
    assert all(q.attack in Attack.ALL for q in qs)
    assert qs == sorted(qs, key=lambda q: q.value, reverse=True)


def test_questions_the_store_can_already_answer_are_worth_fewer_bits():
    m = Memory(HashEmbedder())
    text = "what exactly counts as the parser here, and what is the nearest thing that does not?"
    known = m.remember(Kind.ANSWER, text)
    m.grade(known.id, 1.0, Source.USER)
    with_memory = challenge("the parser must be rewritten", m, limit=6)
    without = challenge("the parser must be rewritten", None, limit=6)
    assert max(q.p_answer_known for q in with_memory) >= max(q.p_answer_known for q in without)


# --------------------------------------------------------------------------- #
# persistence -- never take no for an answer
# --------------------------------------------------------------------------- #

def test_persistence_does_not_stop_at_the_first_refusal():
    tried = []

    def attempt(goal, reframe):
        tried.append(reframe)
        return (True, "worked") if reframe == "substitute" else (False, f"refused: {goal[:20]}")

    out = Persistence(HashEmbedder()).pursue("do the thing", attempt)
    assert out["solved"] and out["reframe"] == "substitute"
    assert tried[0] is None and len(tried) > 1


def test_persistence_stops_when_refusals_stop_being_informative():
    calls = []

    def attempt(goal, reframe):
        calls.append(reframe)
        return False, "the same refusal every single time"

    out = Persistence(HashEmbedder(), max_attempts=10, patience=2).pursue("x", attempt)
    assert not out["solved"]
    assert "stopped being informative" in out["reason"]
    assert len(calls) == 3, "one direct attempt plus patience"
    assert len(out["refusals"]) == 1, "identical refusals are one refusal"


def test_persistence_keeps_going_while_refusals_are_novel():
    reasons = iter(["cannot open the file", "the schema is unknown",
                    "the units are ambiguous", "no clock is available",
                    "the key is missing", "nothing is mounted"])

    def attempt(goal, reframe):
        return False, next(reasons, "exhausted")

    out = Persistence(HashEmbedder(), max_attempts=6, patience=2).pursue("x", attempt)
    assert not out["solved"]
    assert len(out["refusals"]) >= 4, "distinct refusals should each buy another attempt"


def test_an_unsolved_run_still_returns_the_boundary():
    out = Persistence(HashEmbedder()).pursue("x", lambda g, r: (False, "always refused identically"))
    assert "boundary" in out and "refusal" in out["boundary"]


# --------------------------------------------------------------------------- #
# goals
# --------------------------------------------------------------------------- #

def test_a_goal_is_atomic_exactly_when_it_has_a_verifier():
    t = GoalTree("task")
    a = t.add("parse the input", verifier=Verifier("python", "assert True"))
    b = t.add("improve the design")
    assert a.atomic and not b.atomic
    assert [g.text for g in t.unverifiable()] == ["improve the design"]


def test_checkability_separates_observable_from_judgemental_verbs():
    assert checkability("compute the median") > 0.8
    assert checkability("improve the code quality") < 0.2


def test_completion_propagates_to_the_parent():
    t = GoalTree("task")
    a, b = t.add("first"), t.add("second")
    t.mark(a.id, Status.MET)
    assert t.root.status != Status.MET, "a conjunction is not met by one conjunct"
    t.mark(b.id, Status.MET)
    assert t.solved


def test_a_failed_child_fails_the_parent():
    t = GoalTree("task")
    a = t.add("first")
    t.mark(a.id, Status.FAILED)
    assert t.root.status == Status.FAILED


def test_goal_round_trips():
    g = GoalTree("t").add("do it", verifier=Verifier("python", "assert 1"))
    from distil.goals import Goal
    assert Goal.from_json(g.to_json()).verifier.kind == "python"


# --------------------------------------------------------------------------- #
# sandbox
# --------------------------------------------------------------------------- #

def test_clean_code_runs():
    assert run_source("print(sum(range(10)))").stdout.strip() == "45"


def test_an_exception_is_reported_as_its_last_line():
    e = run_source("x = 1/0")
    assert not e.ok and "ZeroDivisionError" in e.diagnostic


def test_an_infinite_loop_is_killed_and_diagnosed():
    e = run_source("while True: pass", timeout=1.5)
    assert e.timed_out and "timed out" in e.diagnostic


def test_destructive_source_is_refused_before_running():
    e = run_source("import shutil; shutil.rmtree('/')")
    assert e.screened_out and "rmtree" in e.diagnostic


def test_network_imports_are_refused():
    assert not screen("import socket").ok
    assert not screen("import urllib.request").ok


def test_eval_and_exec_are_refused():
    assert not screen("eval('1+1')").ok
    assert not screen("exec('x=1')").ok


def test_unparseable_source_is_a_finding_not_an_exception():
    s = screen("def f(:")
    assert not s.ok and "syntax error" in s.findings[0]


def test_credentials_do_not_reach_the_child():
    import os
    os.environ["DISTIL_TEST_SECRET"] = "hunter2"
    try:
        out = run_source("import os; print('DISTIL_TEST_SECRET' in os.environ)")
        assert out.stdout.strip() == "False"
    finally:
        os.environ.pop("DISTIL_TEST_SECRET", None)


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #

GOOD = '''
def median(xs):
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
'''
GOOD_TESTS = 'assert median([3,1,2]) == 2\nassert median([1,2,3,4]) == 2.5'


def test_a_verified_tool_scores_top_marks():
    g = grade_python(GOOD, GOOD_TESTS)
    assert g.score == 1.0 and g.clean and g.passed


def test_a_syntax_error_scores_the_floor():
    g = grade_python("def f(:", GOOD_TESTS)
    assert g.score == -1.0
    assert all(not s.passed for s in g.stages)


def test_later_stages_are_not_reached_after_an_early_failure():
    g = grade_python("def f(:", GOOD_TESTS)
    assert any("not reached" in s.detail for s in g.stages)


def test_failing_the_contract_tests_is_scored_worse_than_having_none():
    wrong = grade_python("def median(xs): return xs[0]", GOOD_TESTS)
    absent = grade_python(GOOD, "")
    assert wrong.score < absent.score < 1.0
    assert not absent.clean, "no tests must never count as verified"


def test_nondeterminism_is_detected():
    g = grade_python("import random; print(random.random())", "assert True")
    assert not g.clean
    assert any(s.name == "determinism" and not s.passed for s in g.stages)


def test_a_check_grades_a_claim_both_ways():
    assert grade_check("assert sorted([1,2]) == [1,2]").score == 1.0
    assert grade_check("assert False").score == -1.0


def test_prose_is_not_gradeable():
    assert not gradeable({"text": "a well-argued essay"})
    assert gradeable({"language": "python", "source": "x = 1"})
    assert gradeable({"check": "assert True"})


# --------------------------------------------------------------------------- #
# toolsmith
# --------------------------------------------------------------------------- #

def test_a_forged_tool_is_verified_registered_and_callable():
    d = fresh()
    spec = d.toolsmith.forge("compute the median of a list")
    assert spec is not None and spec.built_by == "template"
    assert d.toolsmith.validate(spec).score == 1.0
    assert d.toolsmith.register(spec)
    out = d.toolbox.invoke("median", [[5, 3, 1, 4]])
    assert out["ok"] and out["value"] == 3.5


def test_a_broken_tool_is_rejected_and_remembered_as_a_failure():
    d = fresh()
    spec = d.toolsmith.forge("compute the median of a list")
    spec.source = "def median(xs): return xs[0]"
    spec.grade = None
    assert not d.toolsmith.register(spec)
    assert d.memory.of_kind(Kind.FAILURE), "the rejection must be recorded"
    assert "median" not in d.toolbox.names()


def test_a_tool_is_found_by_the_shape_of_the_problem_not_its_name():
    d = fresh()
    spec = d.toolsmith.forge("compute the median of a list")
    d.toolsmith.register(spec)
    found = d.toolbox.find("what is the middle value of these numbers")
    assert found and found[0][0].name == "median"
    assert found[0][2] > 0.0, "find reports raw similarity alongside the score"


def test_using_a_tool_records_the_problem_it_solved():
    d = fresh()
    d.toolsmith.register(d.toolsmith.forge("compute the median of a list"))
    d.toolbox.record_use("median", "found the midpoint of a latency distribution", True)
    assert any("latency" in p for p in d.toolbox.load("median").solved)


def test_unknown_goals_produce_no_template_tool():
    assert synthesise("negotiate a lease renewal with the landlord") is None


def test_reply_parsing_separates_tool_from_tests():
    src, tests = parse_reply("```python\ndef f(): pass\n```\n```python\nassert True\n```")
    assert "def f" in src and "assert True" in tests
    lone, none = parse_reply("```python\ndef f(): pass\n```")
    assert "def f" in lone and none == ""


def test_signature_extraction():
    assert signature_of("def add(a, b=1):\n    return a + b") == ("add", "add(a, b=1)")


def test_invoking_a_missing_tool_is_an_error_not_a_crash():
    d = fresh()
    assert not d.toolbox.invoke("nonexistent")["ok"]


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #

def test_a_session_produces_every_step_of_the_spine():
    d = fresh()
    s = d.reasoner.run("build a csv cleaner and compute the median of each column")
    kinds = {t.kind for t in s.chain.steps}
    for expected in ("WHY", "CHALLENGE", "RETRIEVE", "DISTILL", "PAYOFF", "SELECT"):
        assert expected in kinds, expected


def test_distillation_stops_at_verifiable_goals():
    d = fresh()
    tree = d.reasoner.distill("build a csv cleaner and compute the median of each column")
    assert tree.atoms(), "something should have become checkable"
    for g in tree.atoms():
        assert not g.children, "a verifiable goal is a leaf"


def test_roles_are_recognised():
    assert role_of("verify the output matches") == Role.CHECK
    assert role_of("define what counts as a duplicate") == Role.CLARIFY
    assert role_of("write a reusable helper") == Role.TOOL
    assert role_of("write the importer") == Role.BUILD


def test_the_payoff_matrix_is_well_formed():
    d = fresh()
    tree = d.reasoner.distill("build a csv cleaner")
    m = d.reasoner.payoff_matrix(tree.frontier())
    assert m and all(len(row) == len(State.ALL) for row in m)
    assert all(-1.0 <= x <= 1.5 for row in m for x in row)


def test_an_unexamined_premise_raises_the_odds_of_a_wrong_frame():
    d = fresh()

    class Chain:
        terminal = Ground.CIRCULAR
    grounded = type("C", (), {"terminal": Ground.GROUNDED})()
    circular = d.reasoner.beliefs("a task", Chain())
    solid = d.reasoner.beliefs("a task", grounded)
    i = State.ALL.index(State.WRONG_FRAME)
    assert circular[i] > solid[i]


def test_a_restated_parent_is_not_accepted_as_a_subgoal():
    d = fresh()
    tree = d.reasoner.distill("define the schema")
    for g in tree.goals.values():
        assert not g.text.startswith("define define")


def test_shapley_credit_reaches_the_memory_traces():
    d = fresh()
    r = d.solve("compute the median of a column", interrogate=False)
    assert r["credit"], "a solved task should assign credit"
    assert abs(sum(r["credit"].values()) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# exploration
# --------------------------------------------------------------------------- #

def test_ideas_are_ranked_by_bits_per_unit_cost():
    p = Policy()
    coin = Idea("a genuine coin flip", Origin.CURIOSITY, p_success=0.5, cost=1.0)
    certain = Idea("a thing that obviously works", Origin.CURIOSITY, p_success=0.99, cost=1.0)
    doomed = Idea("a thing that obviously fails", Origin.CURIOSITY, p_success=0.01, cost=1.0)
    assert coin.value(p) > certain.value(p)
    assert coin.value(p) > doomed.value(p), "certain failure teaches as little as certain success"


def test_an_expensive_experiment_is_worth_less_than_a_cheap_one():
    p = Policy()
    cheap = Idea("x", Origin.GAP, p_success=0.5, cost=1.0)
    dear = Idea("x", Origin.GAP, p_success=0.5, cost=4.0)
    assert cheap.value(p) > dear.value(p)


def test_claims_and_proposals_are_distinguished():
    assert Idea("x", Origin.CURIOSITY).claim and Idea("x", Origin.INVERSION).claim
    assert not Idea("x", Origin.GAP).claim and not Idea("x", Origin.PROPOSAL).claim


def test_a_function_template_never_confirms_a_claim():
    """The soundness rule: a template that builds a working function must not be
    recorded as having established an unrelated assertion."""
    d = fresh(seed=5)
    for t in ("compute the median of a column", "dedupe the records"):
        d.solve(t, interrogate=False)
    idea = Idea("test the opposite: what if it is false that the median is well defined",
                Origin.INVERSION, p_success=0.3)
    result = d.explorer.test(idea)
    assert not result.passed
    assert not result.ran or "does not test" in result.detail


def test_analogy_transfers_a_working_approach_onto_an_open_problem():
    """Invention by finding similarity: what worked over there, tried over here."""
    d = fresh(seed=5)
    m = d.memory
    a = m.remember(Kind.FACT, "retrying a flaky network call with exponential backoff "
                              "fixed the intermittent timeout failures")
    m.grade(a.id, 1.0, Source.SELF)
    b = m.remember(Kind.FACT, "caching the parsed schema removed the repeated startup cost")
    m.grade(b.id, 1.0, Source.SELF)
    m.remember(Kind.FAILURE, "the integration test suite fails intermittently with a timeout")
    m.remember(Kind.FAILURE, "the importer re-parses the same schema on every single record")
    ideas = d.explorer.analogies(limit=4)
    assert ideas, "analogous pairs should be found"
    assert all(i.origin == Origin.ANALOGY for i in ideas)
    joined = " | ".join(i.text for i in ideas)
    assert "backoff" in joined and "timeout" in joined


def test_analogy_ignores_pairs_that_are_too_close_or_too_far():
    """The band is the mechanism: identical memories transfer nothing, unrelated
    ones have no structure to carry."""
    d = fresh(seed=5)
    m = d.memory
    a = m.remember(Kind.FACT, "the quick brown fox jumps over the lazy dog every morning")
    m.grade(a.id, 1.0, Source.SELF)
    m.remember(Kind.FAILURE, "the quick brown fox jumps over the lazy dog every morning too")
    assert d.explorer.analogies(limit=3) == [], "a restatement is not an analogy"


def test_a_refuted_experiment_is_stored_as_a_failure():
    d = fresh(seed=2)
    idea = Idea("build a median helper that returns the first element",
                Origin.GAP, p_success=0.5)
    before = len(d.memory.of_kind(Kind.FAILURE)) + len(d.memory.of_kind(Kind.FACT))
    d.explorer.test(idea)
    after = len(d.memory.of_kind(Kind.FAILURE)) + len(d.memory.of_kind(Kind.FACT)) \
        + len(d.memory.of_kind(Kind.IDEA))
    assert after > before, "an experiment must leave a record either way"


def _gameable_store():
    """Well-graded material that is irrelevant, and relevant material that is
    merely adequate. Any objective worth optimising must prefer the second."""
    m = Memory(HashEmbedder())
    for i in range(4):
        t = m.remember(Kind.ANSWER, f"parse an ini config file section {i} with a state machine")
        m.grade(t.id, 0.4, Source.SELF)
    for i in range(4):
        # Shares "parse" and "file" with the query, so it is retrievable -- which
        # is what makes it dangerous. A trace too dissimilar to be returned can
        # never game a ranking.
        t = m.remember(Kind.ANSWER, f"parse an audio file buffer {i} with a discrete cosine transform")
        for _ in range(4):
            m.grade(t.id, 1.0, Source.SELF)
    return m


def _grade_only_score(memory, tasks, k=3):
    """The objective as it was first written: mean grade of the top hits, with
    no reference to whether they had anything to do with the question."""
    total, n = 0.0, 0
    for task in tasks:
        for rank, h in enumerate(memory.recall(task, k=k)):
            g = h.trace.mean_grade
            if g is not None:
                total += g / (1.0 + rank)
                n += 1
    return total / n if n else 0.0


def test_the_first_upgrade_objective_was_gameable():
    """Documents the hole, so the fix has something to be a fix of.

    Self-upgrade drove `recall_similarity_weight` to its floor and scored better
    for it, because a recall that ignores the query returns the best-graded
    traces in the store for every task. Under the original objective that is a
    higher score and a useless memory.
    """
    m = _gameable_store()
    tasks = ["parse an ini config file"]
    m.policy = Policy(recall_similarity_weight=0.0)
    ignoring = _grade_only_score(m, tasks)
    m.policy = Policy(recall_similarity_weight=1.0)
    attending = _grade_only_score(m, tasks)
    assert ignoring > attending, "the original objective rewarded ignoring the query"


def test_the_fixed_objective_punishes_ignoring_the_query():
    """The regression test. Same store, same policies, similarity-weighted."""
    m = _gameable_store()
    d = fresh()
    d.explorer.memory = m
    tasks = ["parse an ini config file"]
    m.policy = Policy(recall_similarity_weight=0.0)
    ignoring = d.explorer.score(tasks)
    m.policy = Policy(recall_similarity_weight=1.0)
    attending = d.explorer.score(tasks)
    assert attending > ignoring, f"{attending} must beat {ignoring}"


def test_upgrade_never_leaves_the_bounds():
    d = fresh(seed=3)
    d.solve("compute the median of a column", interrogate=False)
    d.upgrade(["compute the median of a column"], trials=6)
    for name, (lo, hi) in BOUNDS.items():
        assert lo <= getattr(d.policy, name) <= hi, name


def test_exploration_records_a_journal():
    d = fresh(seed=11)
    d.solve("dedupe the records", interrogate=False)
    d.explore(steps=2)
    assert d.workspace.journal.exists()




FIXTURE = str(Path(__file__).resolve().parent / "fixtures" / "echo_mcp_server.py")


def echo_server(name: str = "echo") -> McpServer:
    return McpServer(name, [sys.executable, FIXTURE])


# --------------------------------------------------------------------------- #
# asking when the objective is not understood
# --------------------------------------------------------------------------- #

def _clarifier(d=None):
    d = d or fresh()
    return d, Clarifier(d.memory, d.framer, d.toolbox)


def test_a_vague_task_yields_questions_rather_than_a_guess():
    """With nobody to ask, the system returns the questions. It does not invent
    an objective and proceed, which is the failure this whole module exists to
    prevent."""
    _, c = _clarifier()
    out = c.clarify("make the thing better")
    assert not out.actionable
    assert out.questions, "an ununderstood task must produce questions"
    assert out.questions[0].gap == Gap.OBJECTIVE, "objective is asked first, always"


def test_questions_are_ordered_by_what_they_unblock():
    _, c = _clarifier()
    out = c.clarify("make the thing better")
    order = [q.gap for q in out.questions]
    assert order == sorted(order, key=Gap.ORDER.index)
    assert out.questions[0].value > out.questions[-1].value


def test_an_echoed_task_is_not_treated_as_an_objective():
    """The Framer falls back to the task text when it cannot read an objective.
    Accepting that as an answer is how a system convinces itself it understands
    a request it has only repeated back."""
    d, c = _clarifier()
    frame = d.framer.frame("make the thing better")
    assert frame.objective.strip() == frame.task.strip()
    assert Gap.OBJECTIVE in frame_gaps(frame, None)


def test_answers_make_the_first_step_actionable():
    _, c = _clarifier()
    answers = {"objective": "p99 latency under 200ms on the import path",
               "referee": "the benchmark suite",
               "actions": "profile, cache, rewrite the hot loop"}
    out = c.clarify("make the thing better", ask=lambda qs: answers)
    assert out.actionable, out.reason
    assert out.frame.objective.startswith("p99")
    assert out.frame.referee == "the benchmark suite"
    assert len(out.frame.actions) == 3


def test_a_gap_answered_in_a_later_round_leaves_the_contested_list():
    """Otherwise the report says a question is still open that the person
    already answered."""
    _, c = _clarifier()
    rounds = iter([{"referee": "the benchmark suite"},
                   {"objective": "p99 latency under 200ms on the import path"}])
    out = c.clarify("make the thing better", ask=lambda qs: next(rounds, {}))
    assert out.actionable, out.reason
    assert "objective" not in out.contested, "answered in round 2; must leave the list"
    assert "objective" in out.resolved


def test_incidental_questions_are_asked_once_but_blockers_come_back():
    """The refined rule. Re-asking "what are your inputs?" is pestering; letting
    the objective go unasked because it was raised once and ignored is how the
    loop gives up on the only thing preventing progress."""
    seen = []

    def ask(questions):
        seen.extend(q.gap for q in questions)
        return {}                       # answer nothing, forcing more rounds

    _, c = _clarifier()
    c.clarify("make the thing better", ask=ask, max_rounds=3)
    incidental = [g for g in seen if g not in Gap.BLOCKING]
    assert len(incidental) == len(set(incidental)), f"repeated incidentals: {incidental}"
    assert seen.count(Gap.OBJECTIVE) > 1, "a blocking gap must be asked again"


def test_a_blocking_gap_answered_late_still_unblocks():
    _, c = _clarifier()
    rounds = iter([{"referee": "the benchmark suite"},
                   {"objective": "p99 latency under 200ms on the import path"}])
    out = c.clarify("make the thing better", ask=lambda qs: next(rounds, {}))
    assert out.actionable, out.reason
    assert out.rounds >= 2


def test_clarification_terminates_when_nothing_is_ever_answered():
    _, c = _clarifier()
    out = c.clarify("improve the design", ask=lambda qs: {}, max_rounds=2)
    assert not out.actionable
    assert out.rounds <= 2


def test_an_answer_given_once_is_not_asked_for_again():
    """The realistic path: a previous round's answer was absorbed, which records
    it with its gap and task, and a later clarification finds it exactly."""
    d, c = _clarifier()
    frame = d.framer.frame("make the importer better")
    c.absorb(frame, {Gap.OBJECTIVE: "p99 under 200ms on the import path"})
    again = d.framer.frame("make the importer better")
    answered = [q for q in c.questions(again, None) if q.answered_by_memory]
    assert answered, "an answer already on record must pre-empt its question"
    assert answered[0].gap == Gap.OBJECTIVE


def test_pre_emption_requires_an_exact_gap_match():
    """A fact about one gap must never be used to skip a question about another.
    An earlier version matched by similarity and silently answered every question
    from whatever it recalled about the task."""
    d, c = _clarifier()
    frame = d.framer.frame("make the importer better")
    c.absorb(frame, {Gap.OBJECTIVE: "p99 under 200ms"})
    other = c.questions(d.framer.frame("make the importer better"), None)
    for q in other:
        if q.gap != Gap.OBJECTIVE:
            assert not q.answered_by_memory, f"{q.gap} was answered by an objective fact"


def test_a_capability_gap_does_not_block_the_first_step():
    """Forging is itself a primitive, so 'write the tool you are missing' is a
    step the system can take. Only an item needing a human blocks."""
    from distil.frame import GameFrame, agenda, capabilities
    d = fresh()
    f = GameFrame(task="t", objective="a specific finished state", actions=["negotiate"],
                  referee="a reviewer")
    plan = agenda(f, capabilities(f, d.memory, d.toolbox))
    ok, reason = first_step_actionable(f, plan)
    assert ok, reason


def test_a_missing_objective_blocks_the_first_step():
    from distil.frame import GameFrame, agenda, capabilities
    d = fresh()
    f = GameFrame(task="t", objective="", actions=["build"])
    plan = agenda(f, capabilities(f, d.memory, d.toolbox))
    ok, reason = first_step_actionable(f, plan)
    assert not ok and "objective" in reason


def test_agenda_marks_which_items_need_a_person():
    from distil.frame import GameFrame, agenda, capabilities
    d = fresh()
    f = GameFrame(task="t", objective="", actions=[])
    plan = agenda(f, capabilities(f, d.memory, d.toolbox))
    assert plan.needs_person, "items nobody can discharge must be marked"
    assert plan.items[0] in plan.needs_person
    assert all(i not in plan.needs_person for i in plan.actionable_items)


# --------------------------------------------------------------------------- #
# MCP: tools that live in another process
# --------------------------------------------------------------------------- #

def test_the_handshake_completes_and_lists_tools():
    with echo_server() as s:
        assert s.started
        names = [t.name for t in s.list_tools()]
        assert {"echo", "add", "explode"} <= set(names)


def test_a_tool_call_returns_its_text_content():
    with echo_server() as s:
        assert s.call("echo", {"text": "over jsonrpc"}).content == "over jsonrpc"
        assert s.call("add", {"a": 2, "b": 40}).content == "42"


def test_stray_output_and_notifications_do_not_break_the_client():
    """The fixture emits a non-JSON line and an unsolicited notification before
    replying to tools/list. A client that trips over either works against
    exactly one server."""
    with echo_server() as s:
        assert len(s.list_tools()) == 3


def test_listing_tools_is_not_flaky():
    """Regression: select() polled the file descriptor while readline() read from
    Python's text buffer, so a buffered-but-unread response looked like a
    timeout. tools/list came back empty about a third of the time."""
    with echo_server() as s:
        for _ in range(6):
            assert len(s.list_tools()) == 3


def test_a_tool_reporting_failure_is_not_a_successful_call():
    with echo_server() as s:
        out = s.call("explode")
        assert not out.ok and "deliberate failure" in out.error


def test_an_unknown_tool_surfaces_the_jsonrpc_error():
    with echo_server() as s:
        out = s.call("does_not_exist")
        assert not out.ok and "unknown tool" in out.error


def test_a_server_that_cannot_start_degrades_rather_than_raising():
    s = McpServer("broken", [sys.executable, "/nonexistent/server.py"])
    out = s.start()
    assert not out.ok and out.error
    s.close()


def test_mcp_tools_are_embedded_and_recalled_like_any_other_tool():
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    report = d.mcp.discover()
    assert report["tools"] == 3 and not report["failed"]
    hits = d.memory.recall("add two numbers together", k=3, kinds=(Kind.TOOL,))
    assert any(h.trace.meta.get("tool") == "echo.add" for h in hits)
    d.mcp.close()


def test_mcp_tools_enter_ungraded():
    """There are no contract tests to run against someone else's server, so
    registering them as verified would be manufacturing evidence."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    tools = [t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("transport") == "mcp"]
    assert tools and all(not t.graded for t in tools)
    d.mcp.close()


def test_one_toolbox_invokes_both_transports():
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    d.toolsmith.register(d.toolsmith.forge("compute the median of a list"))
    assert "median" in d.toolbox.names() and "echo.add" in d.toolbox.names()
    assert d.toolbox.invoke("median", [[5, 3, 1, 4]])["value"] == 3.5
    assert d.toolbox.invoke("echo.add", kwargs={"a": 2, "b": 40})["value"] == "42"
    d.mcp.close()


def test_mcp_tool_names_are_namespaced_by_server():
    a = McpTool("alpha", "search", "", {})
    b = McpTool("beta", "search", "", {})
    assert a.qualified != b.qualified


def test_an_mcp_config_file_is_loaded():
    d = fresh()
    path = tmpdir() / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"echo": {"command": sys.executable,
                                                        "args": [FIXTURE]}}}))
    assert d.mcp.load_config(path) == ["echo"]
    assert "echo" in d.mcp.names()


def test_a_missing_or_corrupt_config_is_not_an_error():
    d = fresh()
    bad = tmpdir() / "mcp.json"
    bad.write_text("{not json")
    assert d.mcp.load_config(bad) == []
    assert d.mcp.load_config(tmpdir() / "absent.json") == []


# --------------------------------------------------------------------------- #
# composition and serialisation
# --------------------------------------------------------------------------- #

def test_every_toolspec_field_survives_the_disk_round_trip():
    """Regression for a real bug: to_json() omitted built_by, transport and deps
    while from_json() read all three, so a composite tool loaded from disk had
    silently lost its dependencies. Every test passed, because they all inspected
    freshly built specs."""
    import dataclasses
    spec = ToolSpec(name="n", purpose="p", source="s", tests="t", signature="sig",
                    solved=["x"], trace_id="tid", built_by="template",
                    transport="mcp", deps=["a", "b"])
    back = ToolSpec.from_json(spec.to_json())
    lost = [f.name for f in dataclasses.fields(ToolSpec)
            if f.name != "grade" and getattr(back, f.name) != getattr(spec, f.name)]
    assert not lost, f"fields lost in serialisation: {lost}"


def test_a_composite_tool_runs_with_its_dependencies_inlined():
    d = fresh()
    d.toolsmith.register(d.toolsmith.forge("compute the median of a list"))
    d.toolsmith.register(d.toolsmith.forge("parse a csv file"))
    spec = ToolSpec(
        name="median_of_column", purpose="median of a csv column",
        source="def median_of_column(text, col=0):\n"
               "    return median([float(r[col]) for r in parse_csv(text)])\n",
        tests='assert median_of_column("1\\n3\\n2") == 2.0',
        signature="median_of_column(text, col=0)", deps=["median", "parse_csv"])
    assert d.toolsmith.validate(spec, d.toolbox).score == 1.0
    assert d.toolsmith.register(spec)
    assert d.toolbox.invoke("median_of_column", ["10\n30\n20"])["value"] == 20.0


def test_a_dependency_appears_before_its_dependent_in_the_bundle():
    d = fresh()
    d.toolsmith.register(d.toolsmith.forge("compute the median of a list"))
    spec = ToolSpec(name="wrapper", purpose="p",
                    source="def wrapper(xs):\n    return median(xs)\n",
                    tests="assert wrapper([1,2,3]) == 2", signature="wrapper(xs)",
                    deps=["median"])
    bundled = d.toolbox.bundle(spec)
    assert bundled.index("def median") < bundled.index("def wrapper")


def test_a_dependency_cycle_does_not_hang_the_bundler():
    d = fresh()
    for name, dep in (("a", "b"), ("b", "a")):
        spec = ToolSpec(name=name, purpose="p", source=f"def {name}():\n    return 1\n",
                        tests="assert True", signature=f"{name}()", deps=[dep])
        d.toolsmith.register(spec, threshold=-2.0)
    out = d.toolbox.bundle(d.toolbox.load("a"))
    assert out.count("def a(") == 1 and out.count("def b(") == 1


# --------------------------------------------------------------------------- #
# the starter toolkit
# --------------------------------------------------------------------------- #

def test_every_seed_tool_passes_its_own_contract_tests():
    """Seeds are not trusted for shipping with the package -- they go through the
    same grader as anything the system writes for itself."""
    d = fresh()
    report = plant(d.toolsmith, d.toolbox)
    assert not report["rejected"], report["rejected"]
    assert not report["skipped"], report["skipped"]
    assert len(report["planted"]) == len(SEEDS), "the kit must stand on its own"
    assert all(score == 1.0 for _, score in report["planted"])


def test_a_seed_with_an_unmet_dependency_is_skipped_not_registered():
    """A composite whose dependency is absent passes nothing and fails at call
    time. Reporting it beats registering a tool that cannot run."""
    from distil.seed import MEDIAN_OF_COLUMN
    d = fresh()
    report = plant(d.toolsmith, d.toolbox, seeds=(MEDIAN_OF_COLUMN,))
    assert report["skipped"] and "missing dependencies" in report["skipped"][0][1]
    assert "median_of_column" not in d.toolbox.names()


def test_planting_twice_does_not_duplicate():
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    second = plant(d.toolsmith, d.toolbox)
    assert not second["planted"] and len(second["already"]) == len(SEEDS)


def test_seed_tools_are_callable_through_the_toolbox():
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    assert d.toolbox.invoke("chunk", [[1, 2, 3, 4, 5], 2])["value"] == [[1, 2], [3, 4], [5]]
    assert d.toolbox.invoke("percentile", [[1, 2, 3, 4], 50])["value"] == 2.5
    assert d.toolbox.invoke("median_of_column", ["10,1\n30,2\n20,3"])["value"] == 20.0


def test_the_schema_tool_reports_every_problem_at_once():
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    schema = {"type": "object", "required": ["age"],
              "properties": {"name": {"type": "string"}}}
    out = d.toolbox.invoke("assert_schema", [{"name": 1}, schema])
    assert len(out["value"]) == 2, "one assert per run turns ten mismatches into ten runs"


def test_error_normalisation_gives_two_instances_one_signature():
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    a = d.toolbox.invoke("normalise_error", ["IndexError: index 5 is out of range"])["value"]
    b = d.toolbox.invoke("normalise_error", ["IndexError: index 9 is out of range"])["value"]
    assert a["kind"] == "IndexError"
    assert a["signature"] == b["signature"], "the same rule violation is one class"

# --------------------------------------------------------------------------- #
# understanding the game, first
# --------------------------------------------------------------------------- #

def test_an_adversarial_zero_sum_game_is_framed_as_minimax():
    f = Framer(Memory(HashEmbedder()), LocalProvider()).frame(
        "win a chess endgame against a stronger opponent")
    assert f.players == Players.VS_ADVERSARY
    assert f.payoff == Payoff.ZERO_SUM
    assert f.solution == Solution.MINIMAX


def test_a_repeated_negotiation_is_framed_as_a_repeated_game():
    f = Framer(Memory(HashEmbedder()), LocalProvider()).frame(
        "negotiate a lease renewal with the landlord every year")
    assert f.players == Players.N_PARTY
    assert f.horizon == Horizon.REPEATED
    assert f.solution == Solution.FOLK, "repeated play makes cooperation enforceable"


def test_a_solitaire_task_against_a_compiler_names_its_referee():
    f = Framer(Memory(HashEmbedder()), LocalProvider()).frame(
        "parse the config and make the test suite pass")
    assert f.players == Players.SOLITAIRE
    assert f.referee, "a game with no referee cannot be self-graded"
    assert f.solution == Solution.DECISION_THEORY


def test_hidden_information_is_detected():
    f = Framer(Memory(HashEmbedder()), LocalProvider()).frame(
        "play poker against players whose hands are hidden")
    assert f.information == Information.IMPERFECT
    assert f.stochastic


def test_incomplete_information_outranks_every_other_rule():
    f = GameFrame(task="x", information=Information.INCOMPLETE,
                  players=Players.VS_ADVERSARY, payoff=Payoff.ZERO_SUM)
    assert classify_game(f) == Solution.BAYESIAN, "unknown payoffs come first"


def test_a_frame_below_the_threshold_reports_that_it_is_not_understood():
    f = GameFrame(task="do the thing")
    assert not f.understood
    assert "below the play threshold" in f.render()


def test_capabilities_match_surface_forms_exactly():
    m = Memory(HashEmbedder())
    f = GameFrame(task="t", actions=["build", "test", "choose"])
    caps = {c.action: c.covered_by for c in capabilities(f, m)}
    assert caps["build"] == "primitive:forge"
    assert caps["test"] == "primitive:check"
    assert caps["choose"] == "primitive:select"


def test_an_uncovered_action_becomes_a_gap_and_heads_the_agenda():
    m = Memory(HashEmbedder())
    f = GameFrame(task="t", actions=["negotiate"], objective="settle the lease")
    caps = capabilities(f, m)
    plan = agenda(f, caps)
    assert caps[0].gap
    assert any("forge a capability" in item for item in plan.items)


def test_a_repeated_game_agenda_says_the_move_must_survive_repetition():
    m = Memory(HashEmbedder())
    f = GameFrame(task="t", actions=["build"], objective="x", horizon=Horizon.REPEATED)
    assert any("repeats" in i for i in agenda(f, capabilities(f, m)).items)


def test_a_frame_is_remembered_and_recalled_by_structure():
    m = Memory(HashEmbedder())
    framer = Framer(m, LocalProvider())
    first = framer.frame("win a chess endgame against a stronger opponent")
    framer.remember(first)
    assert m.of_kind(Kind.GAME), "frames must be retrievable"


def test_understand_runs_before_reasoning_in_a_solve():
    d = fresh(seed=1)
    r = d.solve("build a csv parser that passes the test suite")
    kinds = [t.kind for t in r["session"].chain.steps]
    assert kinds[0] == "FRAME", "the game is understood before anything else happens"
    assert "AGENDA" in kinds
    assert kinds.index("FRAME") < kinds.index("SELECT")


# --------------------------------------------------------------------------- #
# the casebook: problems and what solved them
# --------------------------------------------------------------------------- #

def test_two_solutions_to_one_problem_stay_two_cases():
    """The bug this guards: both cases embed almost identically because the
    problem text dominates, so cosine dedupe merged them and averaged a +1 and a
    -1 into nothing."""
    cb = Casebook(Memory(HashEmbedder()))
    cb.record("csv rows have ragged column counts", "pad short rows to the header width", 1.0)
    cb.record("csv rows have ragged column counts", "drop any row that is not full width", -1.0)
    assert len(cb.all()) == 2


def test_an_identical_case_recorded_twice_is_one_case():
    cb = Casebook(Memory(HashEmbedder()))
    cb.record("a problem", "a solution", 1.0)
    cb.record("a problem", "a solution", 1.0)
    assert len(cb.all()) == 1


def test_the_working_precedent_outranks_the_refuted_one():
    cb = Casebook(Memory(HashEmbedder()))
    cb.record("csv rows have ragged column counts",
              "pad short rows to the header width before parsing", 1.0, via="parse_csv")
    cb.record("csv rows have ragged column counts",
              "drop any row that is not the header width", -1.0, via="naive")
    cb.record("json keys arrive in inconsistent case", "normalise keys to snake_case", 1.0)
    out = cb.adapt("the csv export has ragged rows with inconsistent column counts")
    assert out["precedent"] is not None
    assert "pad short rows" in out["precedent"].case.solution
    assert any("drop any row" in p.case.solution for p in out["avoid"])


def test_a_precedent_reports_what_is_different_this_time():
    cb = Casebook(Memory(HashEmbedder()))
    cb.record("parse the csv export", "split on commas", 1.0)
    p = cb.precedents("parse the csv export over a unicode boundary")[0]
    assert "unicode" in p.novel_terms


def test_cases_are_retrievable_from_the_solution_side():
    cb = Casebook(Memory(HashEmbedder()))
    cb.record("the importer is slow", "cache the parsed schema between records", 1.0)
    cb.record("colours look wrong", "convert to linear space before blending", 1.0)
    found = cb.by_solution("caching the parsed schema")
    assert found and "cache the parsed schema" in found[0].case.solution


def test_a_solved_task_is_filed_as_a_case():
    d = fresh(seed=1)
    d.solve("compute the median of a column", interrogate=False)
    assert d.casebook.all(), "solving should leave a case behind"


def test_a_failed_task_is_also_filed_as_a_case():
    d = fresh(seed=3)
    d.solve("parse a quantum waveform capture file", interrogate=False)
    assert any(c.grade < 0 for c in d.casebook.all()), "failures are results too"


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #

def _cold_store():
    clock = [1_000_000.0]
    m = Memory(HashEmbedder(), clock=lambda: clock[0])
    for i in range(6):
        t = m.remember(Kind.FAILURE,
                       f"connection to postgres on port 5432 refused during migration "
                       f"attempt {i}: OperationalError")
        m.grade(t.id, -0.8, Source.SELF)
    hot = m.remember(Kind.FACT, "connection to postgres on port 5432 refused because "
                                "pg_hba.conf lacked a host rule")
    m.grade(hot.id, 1.0, Source.SELF)
    clock[0] += 86400 * 120
    for _ in range(9):
        # Read the *fact* specifically: this is the trace that stays warm while
        # the failure traces around it go cold.
        m.recall("pg_hba.conf lacked a host rule", k=1)
    return m, hot


def test_compression_is_driven_by_access_not_age():
    m, hot = _cold_store()
    cold_ids = {t.id for t in Compressor(m).cold()}
    assert hot.id not in cold_ids, "a trace still being read is not cold"
    assert cold_ids, "traces nobody reads go cold"


def test_a_digest_keeps_the_identifiers_verbatim():
    m, _ = _cold_store()
    c = Compressor(m)
    clusters = c.clusters()
    assert clusters, "near-identical cold traces should cluster"
    digest = c.summarise(clusters[0])
    assert "5432" in digest.text, "numbers that identify the problem must survive"
    assert "operationalerror" in " ".join(digest.kept_terms).lower()
    assert digest.exemplar in digest.text, "the best member survives verbatim"


def test_a_digest_keeps_the_member_count_and_grades():
    m, _ = _cold_store()
    c = Compressor(m)
    digest = c.summarise(c.clusters()[0])
    assert "grades: mean" in digest.text
    assert digest.mean_grade is not None and digest.mean_grade < 0


def test_compression_is_reversible():
    m, _ = _cold_store()
    archive = tmpdir() / "archive.jsonl"
    c = Compressor(m, archive=archive)
    report = c.compress()
    assert report["compressed"] >= 1 and report["freed"] >= 3
    assert archive.exists()
    assert c.restore() == report["freed"], "everything compressed must come back"


def test_tools_and_cases_are_never_compressed():
    assert Kind.TOOL in PRESERVE and Kind.SOLUTION in PRESERVE and Kind.GAME in PRESERVE


def test_a_dry_run_changes_nothing():
    m, _ = _cold_store()
    before = len(m.store.all())
    Compressor(m).compress(dry_run=True)
    assert len(m.store.all()) == before


# --------------------------------------------------------------------------- #
# tiered memory
# --------------------------------------------------------------------------- #

def test_tiered_promotion_keeps_only_what_earns_it():
    hot, cold = InProcessStore(), InProcessStore()
    m = Memory(HashEmbedder(), store=TieredStore(hot, cold))
    m.remember(Kind.QUERY, "a one-off question nobody ever asked again")
    m.remember(Kind.TOOL, "tool median: compute the middle value of a sequence")
    graded = m.remember(Kind.ANSWER, "an answer that was later checked")
    m.grade(graded.id, 1.0, Source.SELF)
    kinds = {t.kind for t in cold.all()}
    assert kinds == {Kind.TOOL, Kind.ANSWER}, "an unread, ungraded query is not worth keeping"
    assert hot.count() == 3, "short-term memory holds everything until it expires"


def test_being_recalled_promotes_a_trace_to_long_term():
    hot, cold = InProcessStore(), InProcessStore()
    m = Memory(HashEmbedder(), store=TieredStore(hot, cold))
    t = m.remember(Kind.QUERY, "a question that turned out to matter after all")
    assert cold.get(t.id) is None
    m.recall("a question that turned out to matter after all", k=1)
    assert cold.get(t.id) is not None, "a second read is the evidence that it is wanted"


def test_write_through_keeps_everything():
    hot, cold = InProcessStore(), InProcessStore()
    m = Memory(HashEmbedder(), store=TieredStore(hot, cold, write_through=True))
    m.remember(Kind.QUERY, "a one-off question nobody ever asked again")
    assert cold.count() == 1, "cache mode retains everything; that is the trade"


def test_tiered_search_merges_both_tiers():
    hot, cold = InProcessStore(), InProcessStore()
    tier = TieredStore(hot, cold)
    m = Memory(HashEmbedder(), store=tier)
    a = m.remember(Kind.TOOL, "tool for splitting text on commas")      # both tiers
    b = m.remember(Kind.QUERY, "splitting text on commas by hand")      # hot only
    hits = {h.trace.id for h in m.recall("splitting text on commas", k=5)}
    assert {a.id, b.id} <= hits


def test_consolidate_promotes_what_was_missed():
    hot, cold = InProcessStore(), InProcessStore()
    tier = TieredStore(hot, cold)
    m = Memory(HashEmbedder(), store=tier)
    t = m.remember(Kind.QUERY, "a question")
    t.hits = 5                                   # earned it without going through touch
    hot.put(t)
    assert tier.consolidate()["promoted"] == 1


# --------------------------------------------------------------------------- #
# self-editing
# --------------------------------------------------------------------------- #

def _editor() -> SelfEditor:
    d = fresh()
    return d.editor


def test_the_grader_is_protected_from_self_edits():
    from distil.selfedit import Edit
    ed = _editor()
    edit = Edit("grade.py", "WEIGHTS = {'test': 0.4}", "WEIGHTS = {'test': 0.99}", "raise pass rate")
    report = ed.evaluate(edit)
    assert not report.ok
    assert any("protected" in v for v in report.violations)


def test_the_sandbox_and_the_editor_are_protected_too():
    from distil.selfedit import PROTECTED
    assert {"grade.py", "sandbox.py", "selfedit.py"} <= PROTECTED


def test_an_edit_that_drops_a_public_name_is_refused():
    from distil.selfedit import Edit
    ed = _editor()
    before = (ed.package / "goals.py").read_text()
    edit = Edit("goals.py", before, "def _only_private():\n    pass\n", "tidy up")
    violations = ed.invariants(ed.package.parent, edit)
    assert any("removes public name" in v for v in violations)


def test_lost_names_detects_removals_but_ignores_privates():
    assert _lost_names("def a(): pass\ndef b(): pass", "def a(): pass") == {"b"}
    assert _lost_names("def _a(): pass", "") == set()


def test_an_unparseable_edit_is_refused():
    from distil.selfedit import Edit
    ed = _editor()
    violations = ed.invariants(ed.package.parent, Edit("goals.py", "x = 1", "def f(:", "oops"))
    assert any("does not parse" in v for v in violations)


def test_the_test_count_guard_can_see_this_file():
    assert count_tests(Path(__file__).parent) > 50, "the suite must be countable by AST"


def test_rollback_with_no_history_is_not_an_error():
    assert "nothing to roll back" in _editor().rollback()


# --------------------------------------------------------------------------- #
# the assembled agent
# --------------------------------------------------------------------------- #

def test_solve_refuses_to_guess_at_an_unclear_objective():
    """The requirement in one test: do not distil a task nobody can state the
    objective of. A well-organised plan for the wrong problem is the most
    expensive thing this system can produce."""
    d = fresh(seed=1)
    r = d.solve("make the thing better")
    assert r.get("needs_clarification")
    assert r["questions"] and r["questions"][0].gap == Gap.OBJECTIVE
    assert r["session"] is None, "nothing should have been reasoned about yet"


def test_a_clear_task_is_not_gated():
    d = fresh(seed=1)
    r = d.solve("build a csv parser that passes the test suite")
    assert not r.get("needs_clarification", False)
    assert r["solved"]


def test_answers_unblock_a_previously_gated_task():
    d = fresh(seed=1)
    answers = {"objective": "p99 latency under 200ms on the import path",
               "referee": "the benchmark suite",
               "actions": "parse a csv file, compute the median"}
    r = d.solve("make the thing better", ask=lambda qs: answers)
    assert not r.get("needs_clarification", False), r.get("reason")
    assert r["session"] is not None


def test_a_missing_referee_does_not_block_the_first_step():
    """It says the work cannot be self-graded, not that it cannot be started.
    Gating on it refused plain instructions like 'dedupe the records'."""
    d = fresh(seed=1)
    r = d.solve("dedupe the records")
    assert not r.get("needs_clarification", False)
    assert not r["frame"].referee


def test_an_unrecognised_verb_still_yields_a_move_set():
    d = fresh(seed=1)
    frame, _ = d.understand("dedupe the records")
    assert frame.actions == ["dedupe"], "the leading word is the action by construction"


def test_solve_closes_the_loop_offline():
    d = fresh(seed=1)
    r = d.solve("build a csv cleaner and compute the median of each column")
    assert r["solved"]
    assert d.toolbox.names(), "a solved task should leave a tool behind"
    assert d.memory.stats()["verified"] > 0


def test_solve_reuses_a_tool_on_the_second_encounter():
    d = fresh(seed=1)
    d.solve("compute the median of a column", interrogate=False)
    r = d.solve("compute the median of a column", interrogate=False)
    assert any("existing tool" in a["via"] for a in r["attempts"])


def test_an_impossible_task_returns_a_mapped_boundary_not_a_crash():
    d = fresh(seed=3)
    r = d.solve("parse a quantum waveform capture file", interrogate=False)
    assert not r["solved"]
    assert r["refusals"] and r["boundary"]


def test_state_survives_a_restart():
    home = tmpdir()
    a = Distil(LocalProvider(), home=home, seed=1)
    a.solve("compute the median of a column", interrogate=False)
    before = a.memory.stats()["traces"]
    b = Distil(LocalProvider(), home=home, seed=1)
    assert b.memory.stats()["traces"] == before
    assert b.toolbox.names() == a.toolbox.names()


def test_user_grading_moves_credibility():
    d = fresh()
    t = d.memory.remember(Kind.ANSWER, "a claim nobody has checked")
    before = t.credibility(d.policy)
    d.ask_user_grade(t.id, 1.0, "this is right")
    assert d.memory.get(t.id).credibility(d.policy) > before


def test_stats_reports_the_whole_system():
    s = fresh().stats()
    for key in ("traces", "provider", "embedder", "tools", "home", "generations"):
        assert key in s


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #

def test_the_local_provider_is_always_available():
    assert LocalProvider().available()
    assert auto("local").name == "local"


def test_every_provider_declares_whether_it_embeds():
    for p in catalogue():
        assert isinstance(p.can_embed, bool)
        assert isinstance(p.available(), bool)


def test_anthropic_declares_no_embedding_endpoint():
    from distil.provider import AnthropicProvider
    assert not AnthropicProvider().can_embed, "there is no first-party embeddings API"


def test_the_offline_decomposer_splits_on_conjunctions():
    out = LocalProvider().complete([Message("user", "DECOMPOSE:: build a parser and validate it")])
    assert len(out.splitlines()) >= 2


def test_an_unknown_provider_is_rejected_loudly():
    from distil.provider import ProviderError
    try:
        auto("gpt-9000")
        raise AssertionError("should have raised")
    except ProviderError:
        pass


def test_workspace_creates_its_directories():
    w = Workspace(tmpdir() / "nested" / "home")
    assert w.home.exists() and w.workshop.exists()


# --------------------------------------------------------------------------- #

def main() -> bool:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures.append((name, traceback.format_exc()))
    for d in TMP:
        shutil.rmtree(d, ignore_errors=True)
    TMP.clear()
    for name, tb in failures:
        print(f"\nFAIL {name}\n{tb}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return not failures


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
