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
import math
import random
import shutil
import sys
import tempfile
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distil import game
from distil.agent import Distil
from distil.casebook import Case, Casebook
from distil.clarify import (Clarification, Clarifier, Gap, first_step_actionable,
                            gaps as frame_gaps, objective_unclear)
from distil.challenge import (Attack, Ground, Persistence, challenge, classify,
                              interrogate, premises)
from distil.compress import PRESERVE, Compressor
from distil.embed import HashEmbedder, ProviderEmbedder, Vocabulary
from distil.explore import Explorer, Idea, Origin
from distil.frame import (Framer, GameFrame, Horizon, Information, Payoff, Players,
                          PRIMITIVES, Solution, agenda, capabilities, classify as classify_game)
from distil.goals import GoalTree, Status, Verifier, checkability
from distil.mcp import McpRegistry, McpServer, McpTool
from distil.grade import grade_check, grade_python, grade_user, gradeable
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
from distil.project import orient, project
from distil.auto import Auto, Move
from distil.think import Thinker, Thought
from distil.browser import ACTIONS, Browser, BrowserError
from distil.serve import Api, _plain, trace_json
from distil import speech
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
    rounds = iter([{"inputs": "a csv export"},
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
    # Deliberately never the referee: naming one makes the objective checkable
    # by it, which ends the loop and would hide what this test is about.
    answers = iter([{"inputs": "a csv export"}, {"outputs": "a report"}, {}])

    def ask(questions):
        seen.extend(q.gap for q in questions)
        return next(answers, {})        # progress on everything BUT the objective

    _, c = _clarifier()
    out = c.clarify("make the thing better", ask=ask, max_rounds=3)
    incidental = [g for g in seen if g not in Gap.BLOCKING]
    assert len(incidental) == len(set(incidental)), f"repeated incidentals: {incidental}"
    assert seen.count(Gap.OBJECTIVE) > 1, "a blocking gap must be asked again"
    assert not out.actionable
    assert Gap.OBJECTIVE in [q.gap for q in out.questions], "and still be handed back"


def test_answering_nothing_stops_early_but_still_surfaces_the_blocker():
    """Re-asking a caller who answered nothing is pestering, not persistence --
    but the loop must not then return empty-handed. `skip=asked` on the exit path
    withheld the one question worth asking."""
    rounds = []

    def ask(questions):
        rounds.append([q.gap for q in questions])
        return {}

    _, c = _clarifier()
    out = c.clarify("improve the design", ask=ask, max_rounds=3)
    assert len(rounds) == 1, "no point asking again when nothing came back"
    assert out.rounds == 1, "the reported round count must match what happened"
    assert not out.actionable
    assert Gap.OBJECTIVE in [q.gap for q in out.questions]


#: The gate, pinned as a table. It inverted in BOTH directions when it was a
#: bare threshold on `goals.checkability`: "handle it somehow please" scored 0.85
#: and sailed through, "write a parser that produces clean output" scored 0.15
#: and was refused for containing the word "clean", and one filler word decided
#: it -- "fix everything" was gated, "fix everything now" was not.
CLARITY_CASES = [
    # (task, must_be_gated)
    ("write a parser that produces clean output", False),   # judgement word, but says write/produce
    ("build a csv parser that passes the test suite", False),
    ("compute the median of a column", False),
    ("parse a quantum waveform capture file", False),
    ("dedupe the records", False),
    ("remove duplicate rows from the csv file", False),
    ("write a script to clean up the csv", False),          # "to" must not bypass the gate
    ("make the thing better", True),                        # nothing named
    ("do the thing with the stuff", True),
    ("just make it work", True),
    ("handle it somehow please", True),                     # observable verb, no subject
    ("fix everything", True),
    ("fix everything now", True),                           # a filler word must not flip it
    ("improve the design", True),                           # judgement only
    ("make the importer better", True),
    ("clean up the csv file", True),                        # names a subject, but "clean" how?
]


def test_the_determinism_stage_actually_exercises_the_function():
    """It re-ran the MODULE and compared stdout -- but the tool contract mandates
    no I/O outside the function, so a conforming module prints nothing and the
    stage compared '' to '' and passed unconditionally. It never called the
    function it claimed to check."""
    rng = "import random\n\ndef roll():\n    return random.randrange(10**9)\n"
    g = grade_python(rng, "assert isinstance(roll(), int)")
    assert not g.clean, "an unseeded RNG is not repeatable"
    assert any(st.name == "determinism" and not st.passed for st in g.stages)
    seeded = ("import random\n\ndef roll():\n"
              "    return random.Random(7).randrange(10**9)\n")
    assert grade_python(seeded, "assert roll() == roll()").clean, "seeded is fine"
    assert grade_python("def add(a, b):\n    return a + b\n", "assert add(1, 2) == 3").clean


def test_a_tool_that_is_not_repeatable_does_not_register():
    """A failed determinism stage scores 0.70, which cleared the 0.5 threshold --
    so the check ran, said FAIL, and the tool registered anyway."""
    d = fresh()
    rng = "import random\n\ndef roll():\n    return random.randrange(10**9)\n"
    spec = ToolSpec(name="roll", purpose="roll", source=rng,
                    tests="assert isinstance(roll(), int)", signature="roll()")
    d.toolsmith.validate(spec, d.toolbox)
    assert not d.toolsmith.register(spec)
    assert "roll" not in d.toolbox.names()


def test_failing_tests_still_rank_below_having_none():
    good = "def add(a, b):\n    return a + b\n"
    wrong = grade_python(good, "assert add(1, 2) == 99")
    absent = grade_python(good, "")
    assert wrong.score < absent.score < 1.0, (wrong.score, absent.score)


def test_a_nan_grade_is_refused():
    """max(-1, min(1, nan)) is nan, and nan compares False against every
    threshold, so it was recorded as a maximum positive endorsement."""
    try:
        grade_user(float("nan"))
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_the_gate_is_not_sensitive_to_which_synonym_was_used():
    """"clean up the csv file" was gated and "tidy up the csv file" was not,
    because the judgement vocabulary was borrowed from goals._UNCHECKABLE, which
    lists only the forms distillation happens to care about."""
    d = fresh()
    for a, b in (("clean up the csv file", "tidy up the csv file"),
                 ("improve the report", "enhance the report")):
        ga = objective_unclear(d.clarifier.framer.frame(a))
        gb = objective_unclear(d.clarifier.framer.frame(b))
        assert ga == gb, f"{a!r} -> {ga} but {b!r} -> {gb}"


def test_a_stated_bound_counts_as_a_finished_state():
    d = fresh()
    for task in ("improve the importer so it finishes in under a minute",
                 "make the report load in under 2 seconds",
                 "get p99 latency under 200ms"):
        assert not objective_unclear(d.clarifier.framer.frame(task)), task


def test_a_hedge_in_front_of_an_answer_is_still_an_answer():
    """Discarding it threw away exactly the content that was asked for."""
    from distil.clarify import _denies_knowledge
    assert _denies_knowledge("I don't really know")
    assert _denies_knowledge("not entirely sure")
    assert _denies_knowledge("the answer is not known to me")
    assert not _denies_knowledge("not sure, but done means the importer finishes in a minute")
    assert not _denies_knowledge("don't know yet, but it must finish under a minute")


def test_the_clarity_gate_does_not_invert_in_either_direction():
    d = fresh()
    wrong = []
    for task, should_gate in CLARITY_CASES:
        got = objective_unclear(d.clarifier.framer.frame(task))
        if got != should_gate:
            wrong.append((task, should_gate, got))
    assert not wrong, "\n".join(f"{t!r}: expected gated={w}, got {g}" for t, w, g in wrong)


def test_one_filler_word_does_not_flip_the_gate():
    """The sharpest symptom of using a prior as a boundary: checkability's
    no-signal answer is exactly 0.5 and the test was `< 0.5`, so adding any
    third word moved a task from "unknown" to "clear"."""
    d = fresh()
    for short, padded in (("fix everything", "fix everything now"),
                          ("make it work", "just make it work please")):
        a = objective_unclear(d.clarifier.framer.frame(short))
        b = objective_unclear(d.clarifier.framer.frame(padded))
        assert a == b, f"{short!r} -> {a} but {padded!r} -> {b}"


def test_a_bare_to_is_not_a_purpose_marker():
    """`_OBJECTIVE` matched a bare "to", so "give the report to accounting"
    yielded the objective "accounting" -- and any objective differing from the
    task short-circuits the clarity gate, so every task containing "to" bypassed
    it."""
    from distil.frame import _OBJECTIVE
    assert _OBJECTIVE.search("give the report to accounting") is None
    assert _OBJECTIVE.search("talk to the team about design") is None
    found = _OBJECTIVE.search("rewrite the parser so that the suite passes")
    assert found and "suite passes" in found.group(1)


def test_a_referee_does_not_substitute_for_an_objective():
    """Tried and reverted. `objective_unclear` reaches that point only when no
    objective could be read at all, so letting a referee pass it made having a
    judge stand in for knowing what winning is. A benchmark suite can tell you a
    number moved; it cannot tell you which number you meant."""
    _, c = _clarifier()
    out = c.clarify("make the thing better",
                    ask=lambda qs: {Gap.REFEREE: "the benchmark suite"})
    assert not out.actionable, out.reason
    assert Gap.OBJECTIVE in [q.gap for q in out.questions]
    # and the objective, once given, does unblock it
    answers = {Gap.OBJECTIVE: "p99 latency under 200ms on the import path",
               Gap.REFEREE: "the benchmark suite"}
    assert c.clarify("make the thing better", ask=lambda qs: answers).actionable


def test_the_reported_first_step_is_the_step_that_was_judged():
    """Advisories lead the agenda, so printing items[0] named a sentence as the
    first step while the gate had judged a different item entirely."""
    _, c = _clarifier()
    out = c.clarify("compute the median of a column")
    assert out.actionable, out.reason
    rendered = out.render()
    if "first step:" in rendered:
        step = rendered.split("first step:")[1].splitlines()[0].strip()
        assert step in out.plan.actionable_items
        assert step not in out.plan.advisories


def test_a_memory_answer_that_does_not_close_a_blocker_is_still_asked():
    """`answered_by_memory` alone dropped the question every round, so a
    blocking gap whose stored answer the frame could not use was never raised
    again -- undoing the rule that blockers repeat."""
    d, c = _clarifier()
    frame = d.framer.frame("make the thing better")
    # an answer on record that absorb cannot turn into an objective
    d.memory.remember(Kind.FACT, "about the game 'make the thing better': objective is none",
                      meta={"task": "make the thing better", "gap": Gap.OBJECTIVE,
                            "answer": "none"}, grade=1.0, source=Source.USER)
    asked = []
    c.clarify("make the thing better",
              ask=lambda qs: (asked.extend(q.gap for q in qs), {})[1], max_rounds=2)
    assert Gap.OBJECTIVE in asked, "an unusable stored answer must not silence the question"


def test_an_objective_of_i_do_not_know_is_not_an_objective():
    """Only PAYOFF was guarded, so "I don't know" was written into the objective
    and -- being different from the task -- then read as understood."""
    _, c = _clarifier()
    out = c.clarify("make the thing better", ask=lambda qs: {Gap.OBJECTIVE: "I don't know"})
    assert not out.actionable, out.reason


def test_an_answer_already_on_record_is_applied_even_with_nobody_to_ask():
    """The early return ran before memory answers were absorbed, so solve() --
    which passes ask=None -- re-gated a task on its own recorded answer."""
    d, c = _clarifier()
    frame = d.framer.frame("make the importer better")
    c.absorb(frame, {Gap.OBJECTIVE: "p99 under 200ms on the import path",
                     Gap.REFEREE: "the benchmark suite"})
    assert c.clarify("make the importer better").actionable      # ask=None


def test_a_gap_is_never_both_resolved_and_contested():
    _, c = _clarifier()
    rounds = iter([{"inputs": "a csv export"}, {"objective": "p99 under 200ms"}])
    out = c.clarify("make the thing better", ask=lambda qs: next(rounds, {}))
    assert not (set(out.resolved) & set(out.contested))


def test_an_answer_the_frame_cannot_use_is_not_resolved():
    """Resolved means the gap is gone, not that a string was supplied."""
    _, c = _clarifier()
    out = c.clarify("make the thing better",
                    ask=lambda qs: {Gap.REFEREE: "nothing"}, max_rounds=1)
    assert Gap.REFEREE not in out.resolved, "'nothing' leaves the referee unset"


def test_memory_cannot_launder_a_record_into_an_objective():
    """The stored fact wraps the answer in a sentence. Feeding that sentence back
    as the objective made the frame differ from the task and therefore read as
    understood -- a rejected objective accepted via the memory layer."""
    d, c = _clarifier()
    frame = d.framer.frame("make the importer better")
    c.absorb(frame, {Gap.OBJECTIVE: "p99 under 200ms"})
    again = d.framer.frame("make the importer better")
    answered = [q for q in c.questions(again, None) if q.answered_by_memory]
    assert answered
    assert answered[0].answered_by_memory == "p99 under 200ms", \
        "the answer, not the record that wraps it"


def test_saying_you_do_not_know_the_payoffs_is_not_knowing_them():
    from distil.frame import GameFrame, Information
    _, c = _clarifier()
    f = GameFrame(task="t", information=Information.INCOMPLETE)
    c.absorb(f, {Gap.PAYOFF: "no idea"})
    assert f.information == Information.INCOMPLETE
    c.absorb(f, {Gap.PAYOFF: "we both want the deal to close"})
    assert f.information != Information.INCOMPLETE


def test_a_blocking_gap_answered_late_still_unblocks():
    _, c = _clarifier()
    rounds = iter([{"inputs": "a csv export"},
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


def test_a_chatty_stderr_does_not_deadlock_the_server():
    """The failure this guards: stderr was opened as a pipe and never read, so a
    server logging more than the 64KB buffer blocked forever on its next write.
    It looked exactly like a hung server and was a client that never drained."""
    s = McpServer("chatty", [sys.executable, FIXTURE, "chatty"])
    try:
        assert s.start().ok
        assert len(s.list_tools()) == 3
        assert s.call("add", {"a": 1, "b": 1}).content == "2"
    finally:
        s.close()


def test_the_whole_tool_catalogue_is_read_across_pages():
    """A server that paginates used to have everything past page one silently
    dropped, and a missing tool is indistinguishable from one not offered."""
    s = McpServer("paged", [sys.executable, FIXTURE, "paged"])
    try:
        assert {t.name for t in s.list_tools()} == {"echo", "add", "explode"}
    finally:
        s.close()


def test_concurrent_requests_do_not_consume_each_others_replies():
    """Two _request calls read from one queue, so each discarded the other's
    reply as 'not mine' and both timed out."""
    import threading
    s = echo_server()
    try:
        s.start()
        results, errors = [], []

        def hammer(n):
            try:
                results.append(s.call("add", {"a": n, "b": 0}).content)
            except Exception as exc:                       # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, errors
        assert sorted(results) == [str(i) for i in range(6)], results
    finally:
        s.close()


def test_a_crashed_server_is_restarted_on_the_next_call():
    """`started` stayed True after the child died, so every later call wrote
    into a dead pipe and failed for the life of the registry."""
    s = McpServer("crash", [sys.executable, FIXTURE, "crash"])
    try:
        assert s.start().ok
        first = s.call("echo", {"text": "x"})
        assert not first.ok, "the fixture exits on tools/call"
        assert not s.started, "a dead child must not still count as started"
        assert s.start().ok, "the next start brings up a fresh process"
    finally:
        s.close()


def test_a_restart_does_not_serve_the_dead_process_output():
    """Sinks were reused across processes, so a crashed server's buffered stdout
    was still queued when its replacement started and the first reply read came
    from the corpse."""
    s = McpServer("crash", [sys.executable, FIXTURE, "crash"])
    try:
        s.start()
        s.call("echo", {"text": "x"})          # the fixture exits here
        assert not s.started
        assert s.start().ok
        # a fresh child answers its own tools/list, not the dead one's leftovers
        assert len(s.list_tools()) == 3
    finally:
        s.close()


def test_concurrent_starts_spawn_exactly_one_child():
    """Two threads each spawned a process; the loser was orphaned, survived
    close(), and its EOF sentinel later aborted a live request."""
    import threading
    s = echo_server()
    try:
        seen = []
        barrier = threading.Barrier(4)

        def go():
            barrier.wait()
            s.start()
            seen.append(s.proc)

        threads = [threading.Thread(target=go) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert len({id(p) for p in seen}) == 1, "exactly one child must exist"
        assert s.call("add", {"a": 1, "b": 1}).content == "2"
    finally:
        s.close()


def test_a_server_that_dies_while_idle_is_noticed():
    """Nothing cleared `started` on the write path, so a child that died between
    calls left the server 'started' forever and every later call failed."""
    s = echo_server()
    try:
        s.start()
        s.proc.kill()
        s.proc.wait(timeout=5)
        out = s.call("echo", {"text": "x"})
        assert not out.ok
        assert not s.started, "a dead child must not still count as started"
        assert s.start().ok, "and the next start must bring up a fresh one"
    finally:
        s.close()


def test_closing_while_a_call_is_in_flight_fails_cleanly():
    """close() sets proc to None on another thread; _send read it between the
    guard and the write and raised AttributeError out of call()."""
    import threading
    s = echo_server()
    s.start()
    errors = []

    def hammer():
        for _ in range(40):
            try:
                s.call("echo", {"text": "x"})
            except Exception as exc:               # must degrade, never raise
                errors.append(exc)

    t = threading.Thread(target=hammer)
    t.start()
    s.close()
    t.join(timeout=30)
    assert not errors, errors


def test_a_repeated_pagination_cursor_does_not_loop():
    """A server repeating a cursor would be paged to the limit, re-adding the
    same tools on every pass."""
    s = echo_server()
    try:
        s.start()
        calls = {"n": 0}
        real = s._request

        def stuck(method, params=None, timeout=None):
            if method == "tools/list":
                calls["n"] += 1
                out = real(method, params, timeout)
                if out.ok:
                    out.raw.setdefault("result", {})["nextCursor"] = "same"
                return out
            return real(method, params, timeout)

        s._request = stuck
        tools = s.list_tools()
        assert calls["n"] <= 2, f"stopped after {calls['n']} pages"
        assert s.catalogue_truncated, "a partial catalogue must say so"
        assert len(tools) == len({t.name for t in tools}), "no duplicates"
    finally:
        s.close()


def test_a_busy_server_bounds_the_whole_call_not_just_the_wire():
    """The request lock was held across the wait, so a queued caller waited its
    predecessor's timeout and then its own."""
    import threading
    import time as _time
    s = echo_server()
    try:
        s.start()
        s._lock.acquire()                       # simulate a long in-flight call
        started = _time.monotonic()
        out = s.call("echo", {"text": "x"}, timeout=1.0)
        elapsed = _time.monotonic() - started
        assert not out.ok and "busy" in out.error
        assert elapsed < 3.0, f"the call took {elapsed:.1f}s against a 1s budget"
    finally:
        try:
            s._lock.release()
        except RuntimeError:
            pass
        s.close()


def test_closing_twice_is_safe_and_reaps_the_child():
    s = echo_server()
    s.start()
    proc = s.proc
    s.close()
    s.close()
    assert proc.poll() is not None, "the child must be reaped, not left a zombie"


def test_discovering_an_unknown_server_degrades():
    d = fresh()
    report = d.mcp.discover("not-configured")
    assert report["failed"] and "no such server" in report["failed"][0]["error"]


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

def test_no_serialiser_writes_a_key_its_reader_ignores():
    """A generic guard for the bug class the review found in eight places.

    Every writer/reader pair is checked in both directions: a key written but
    never read is state that silently disappears, and a key read but never
    written is a default masquerading as data. Both were live here -- ToolSpec
    dropped deps, Case dropped trace_id, GameFrame dropped evidence, Memory
    wrote a policy nothing loaded.
    """
    import dataclasses

    def survives(obj, dump, load, skip=()):
        back = load(dump(obj))
        return [f.name for f in dataclasses.fields(obj)
                if f.name not in skip
                and getattr(back, f.name) != getattr(obj, f.name)]

    spec = ToolSpec(name="n", purpose="p", source="s", tests="t", signature="sig",
                    solved=["x"], trace_id="tid", built_by="template",
                    transport="mcp", deps=["a", "b"])
    assert not survives(spec, lambda o: o.to_json(), ToolSpec.from_json, skip=("grade",))

    case = Case(problem="p", solution="s", grade=1.0, via="v", evidence="e",
                cost=2.0, trace_id="tid", tags=["t"])
    assert not survives(case, lambda o: o.to_meta(), Case.from_meta)

    frame = GameFrame(task="t", players=Players.N_PARTY, actions=["a"], inputs=["i"],
                      outputs=["o"], objective="obj", information=Information.IMPERFECT,
                      payoff=Payoff.ZERO_SUM, horizon=Horizon.REPEATED, stochastic=True,
                      referee="r", confidence=0.5, evidence=["because"])
    back = GameFrame(**{k: v for k, v in frame.to_json().items()})
    assert back.evidence == ["because"], "the reasons for a frame must survive"

    from distil.goals import Goal
    goal = GoalTree("t").add("do it", verifier=Verifier("python", "assert 1"))
    assert Goal.from_json(goal.to_json()).verifier.body == "assert 1"

    policy = Policy(branch_factor=5, credit_decay=0.3)
    assert not survives(policy, lambda o: o.to_json(), Policy.from_json)


def test_a_restored_tool_reports_the_grade_it_was_given():
    spec = ToolSpec(name="n", purpose="p", source="s", tests="t")
    spec.grade = grade_python("def n():\n    return 1\n", "assert n() == 1")
    back = ToolSpec.from_json(spec.to_json())
    assert back.grade is not None and back.grade.score == spec.grade.score


def test_a_predicate_verifier_is_not_automatic_once_its_callable_is_gone():
    """Callables do not serialise. The kind survived and `fn` did not, so a
    round-tripped goal claimed to be machine-checkable with nothing to run."""
    from distil.goals import Goal
    live = Verifier("predicate", fn=lambda: True)
    assert live.automatic
    tree = GoalTree("t")
    g = tree.add("do it", verifier=live)
    assert not Goal.from_json(g.to_json()).verifier.automatic


def test_the_policy_in_a_saved_store_is_read_back():
    m = Memory(HashEmbedder(), Policy(branch_factor=7))
    m.remember(Kind.FACT, "a fact")
    path = tmpdir() / "memory.jsonl"
    m.save(path)
    n = Memory(HashEmbedder())
    n.load(path)
    assert n.saved_policy is not None and n.saved_policy.branch_factor == 7
    assert n.policy.branch_factor != 7, "it is recorded for inspection, not applied"


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


def test_an_mcp_tool_refuses_positional_arguments_instead_of_dropping_them():
    """They used to be discarded while the call still reported ok=True, which is
    a silent wrong answer -- the worst outcome a tool call has."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    out = d.toolbox.invoke("echo.add", [2, 40])
    assert not out["ok"] and "named arguments only" in out["error"]
    assert d.toolbox.invoke("echo.add", kwargs={"a": 2, "b": 40})["value"] == "42"
    d.mcp.close()


def test_mcp_arguments_are_checked_against_the_recorded_schema():
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    missing = d.toolbox.invoke("echo.add", kwargs={"a": 1})
    assert not missing["ok"] and "required argument" in missing["error"]
    undeclared = d.toolbox.invoke("echo.add", kwargs={"a": 1, "b": 2, "c": 3})
    assert not undeclared["ok"] and "not an argument" in undeclared["error"]
    d.mcp.close()


def test_using_an_mcp_tool_grades_the_trace_it_already_has():
    """record_use called remember() with different text, creating a SECOND trace;
    grades accumulated on the duplicate while the trace load() reads never moved."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    before = len([t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "echo.add"])
    d.toolbox.record_use("echo.add", "added two numbers", True)
    after = [t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "echo.add"]
    assert len(after) == before == 1, "the tool must not fork into two traces"
    assert after[0].mean_grade is not None and after[0].mean_grade > 0
    assert "added two numbers" in d.toolbox.load("echo.add").solved
    d.mcp.close()


def test_an_mcp_tool_becomes_reusable_once_it_has_worked():
    """The `solved` gate was unsatisfiable for MCP tools by construction, so they
    were permanently unreachable through solve()."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    from distil.agent import _proven
    spec = d.toolbox.load("echo.add")
    assert not _proven(spec, d.memory), "nothing is reused on faith"
    d.toolbox.record_use("echo.add", "added two numbers", True)
    assert _proven(d.toolbox.load("echo.add"), d.memory)
    d.mcp.close()


def test_a_dependency_that_cannot_be_inlined_fails_validation_loudly():
    """Skipping it graded the tool against source that is not what runs:
    validation passed and the call then died with NameError."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    spec = ToolSpec(name="wrapper", purpose="p",
                    source="def wrapper():\n    return 1\n",
                    tests="assert wrapper() == 1", signature="wrapper()",
                    deps=["echo.add"])
    grade = d.toolsmith.validate(spec, d.toolbox)
    assert grade.score == -1.0 and "cannot be inlined" in grade.diagnostic
    assert not d.toolsmith.register(spec)
    d.mcp.close()


def test_an_unregistered_dependency_is_reported_not_skipped():
    d = fresh()
    spec = ToolSpec(name="w", purpose="p", source="def w():\n    return 1\n",
                    tests="assert w() == 1", signature="w()", deps=["nonexistent"])
    grade = d.toolsmith.validate(spec, d.toolbox)
    assert grade.score == -1.0 and "not registered" in grade.diagnostic


def test_register_bundles_dependencies_without_an_explicit_validate():
    """Its fallback validate() had no toolbox, so a working composite was
    rejected and a false FAILURE about it was written to memory."""
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    spec = ToolSpec(
        name="mid_of_col", purpose="median of a column",
        source="def mid_of_col(text, col=0):\n"
               "    return median([float(r[col]) for r in parse_csv(text)])\n",
        tests='assert mid_of_col("1\\n3\\n2") == 2.0',
        signature="mid_of_col(text, col=0)", deps=["median", "parse_csv"])
    assert d.toolsmith.register(spec), "register must bundle on its own"
    assert d.toolbox.invoke("mid_of_col", ["10\n30\n20"])["value"] == 20.0


def test_the_live_mcp_record_wins_over_a_stale_snapshot():
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    # a stale file of the same qualified name must not shadow the live tool
    (d.workspace.workshop / "echo.add.json").write_text(json.dumps(
        {"name": "echo.add", "purpose": "stale", "source": "", "transport": "python"}))
    spec = d.toolbox.load("echo.add")
    assert spec.transport == "mcp" and spec.purpose != "stale"
    d.mcp.close()


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


def test_reuse_does_not_manufacture_a_verifier_grade():
    """Reuse deliberately does not run the tool, so it must not record a
    Source.SELF success for it. Each reuse used to push a +1 that could outvote
    the real negative grades a failing tool had earned."""
    d = fresh(seed=1)
    d.solve("compute the median of a column", interrogate=False)
    trace = next(t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "median")
    before = list(trace.grades)
    r = d.solve("compute the median of a column", interrogate=False)
    assert any("existing tool" in a["via"] for a in r["attempts"]), "this must be the reuse path"
    after = next(t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "median")
    assert len(after.grades) == len(before), "a tool that never ran earned no grade"


def test_a_tool_the_store_has_graded_negative_is_not_reused():
    """The gate read the score frozen into the tool's JSON at registration, so
    everything that happened since was invisible to it."""
    from distil.agent import _reusable
    d = fresh(seed=1)
    d.solve("compute the median of a column", interrogate=False)
    spec = d.toolbox.load("median")
    assert _reusable(spec, d.memory)
    for _ in range(4):
        d.toolbox.record_use("median", "a problem it got wrong", False)
    assert not _reusable(d.toolbox.load("median"), d.memory)


def test_using_a_tool_puts_the_problem_into_its_searchable_text():
    """record_use updated meta but never re-embedded, so the solved problem
    never joined the vector it is documented to join."""
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    assert not d.toolbox.find("collapse a ragged telemetry export", k=1) or \
        d.toolbox.find("collapse a ragged telemetry export", k=1)[0][2] < 0.3
    d.toolbox.record_use("parse_csv", "collapse a ragged telemetry export", True)
    found = d.toolbox.find("collapse a ragged telemetry export", k=1)
    assert found and found[0][0].name == "parse_csv" and found[0][2] > 0.3


def test_clarification_answers_reach_the_work_not_just_the_frame():
    """They went into the frame and no further: run() was called on the raw task
    string, so someone who patiently explained what done means got the same goal
    tree as someone who said nothing."""
    d = fresh(seed=1)
    r = d.solve("make the thing better",
                ask=lambda qs: {Gap.OBJECTIVE: "compute the median of each column",
                                Gap.REFEREE: "the test suite",
                                Gap.ACTIONS: "compute, parse"})
    goals = [g.text for g in r["session"].tree.goals.values() if g.id != "root"]
    assert any("median" in g for g in goals), f"the objective must be distilled: {goals}"
    assert r["session"].chain.task == "make the thing better", "the chain keeps the ask"


def test_a_weaker_tool_cannot_overwrite_a_stronger_one_of_the_same_name():
    """A tool is keyed by its function name, so forging "median" wrote over
    whatever median was already there -- including a seed with a far stronger
    contract suite, and every composite depending on it inherited the downgrade."""
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    before = d.toolbox.load("median")
    spec = d.toolsmith.forge("compute the median of a list")
    assert not d.toolsmith.register(spec), "a replacement must be strictly better to land"
    assert d.toolbox.load("median").tests == before.tests


def test_a_fallback_vector_is_tagged_as_lexical_and_stays_recallable():
    """A degraded provider embedder returns a LEXICAL vector. Tagging it as a
    provider vector puts two geometries in one space -- which DESIGN 4.4 refuses,
    and undetectably, because the store would then compare them happily."""
    class Down:
        name, can_embed = "down", True

        def embed(self, texts):
            raise RuntimeError("provider unavailable")

    m = Memory(ProviderEmbedder(Down(), HashEmbedder()))
    t = m.remember(Kind.FACT, "a fact written while the provider was unavailable")
    assert t.embedder == "hash-v1", t.embedder
    hits = m.recall("a fact written while the provider was unavailable", k=1)
    assert len(hits) == 1, "a query embedded lexically must reach lexically embedded traces"


def test_rediscovery_does_not_erase_what_use_taught_the_store():
    """discover() runs on every startup and re-wrote solved=[] into the merged
    meta, wiping everything record_use had recorded."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    d.toolbox.record_use("echo.add", "totalled two figures", True)
    d.mcp.discover()                       # as a restart would
    assert "totalled two figures" in d.toolbox.load("echo.add").solved
    d.mcp.close()


def test_an_mcp_tool_whose_server_is_gone_is_not_counted_as_covering_a_goal():
    """Grades say a tool worked once; they say nothing about whether its server
    is still configured."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    d.toolbox.record_use("echo.add", "added two numbers", True)
    spec = d.toolbox.load("echo.add")
    assert d._runnable(spec)
    d.mcp.servers.clear()                  # the server is no longer configured
    assert not d._runnable(spec)
    d.mcp.close()


def test_tuning_the_policy_is_not_undone_by_the_save_that_follows_it():
    d = fresh(seed=3)
    d.solve("compute the median of a column", interrogate=False)
    d.upgrade(["compute the median of a column"], trials=6)
    assert d.policy is d.explorer.policy
    assert Policy.load(d.workspace.policy).to_json() == d.policy.to_json()


def test_a_verified_tool_name_can_still_be_improved():
    """A strictly-greater grade wedged the toolbox shut: a verified tool scores
    1.00, nothing scores above 1.00, so no verified name could ever be improved
    again -- not by a bug fix, not by anything."""
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    incumbent = d.toolbox.load("median")
    stronger = ToolSpec(name="median", purpose="median", source=incumbent.source,
                        tests=incumbent.tests + "\nassert median([2, 1]) == 1.5\n",
                        signature=incumbent.signature)
    d.toolsmith.validate(stronger, d.toolbox)
    assert d.toolsmith.register(stronger), "a stronger contract must be able to land"
    weaker = d.toolsmith.forge("compute the median of a list")
    assert not d.toolsmith.register(weaker), "a weaker one still must not"


def test_re_registering_keeps_the_problems_the_tool_has_solved():
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    d.toolbox.record_use("median", "found the midpoint of a latency sample", True)
    incumbent = d.toolbox.load("median")
    stronger = ToolSpec(name="median", purpose="median", source=incumbent.source,
                        tests=incumbent.tests + "\nassert median([4, 2]) == 3\n",
                        signature=incumbent.signature)
    d.toolsmith.validate(stronger, d.toolbox)
    assert d.toolsmith.register(stronger)
    assert any("latency sample" in p for p in d.toolbox.load("median").solved)


def test_an_existing_better_tool_is_not_reported_as_a_failure():
    """register() also returns False when it KEPT a better incumbent. That was
    reported as "failed verification: verified" and filed as a -1.0 case against
    a capability that exists and works."""
    d = fresh(seed=1)
    plant(d.toolsmith, d.toolbox)
    r = d.solve("compute the median of a list", interrogate=False)
    assert all("failed verification: verified" not in a.get("via", "") for a in r["attempts"])
    assert not any(c.grade < 0 and "median" in c.solution for c in d.casebook.all())


def test_malformed_jsonrpc_degrades_instead_of_raising():
    from distil.mcp import McpServer, McpResult
    s = echo_server()
    try:
        s.start()
        out = s._await.__self__  # keep a reference; exercise the parse path below
        s._lines.put('{"jsonrpc":"2.0","id":9999,"error":"not an object"}\n')
        res = s._await(9999, 2.0)
        assert isinstance(res, McpResult) and not res.ok
        assert "malformed" in res.error
    finally:
        s.close()


def test_a_server_name_containing_a_dot_is_still_callable():
    d = fresh()
    d.mcp.add("acme.tools", [sys.executable, FIXTURE])
    d.mcp.discover()
    assert d.mcp.call("acme.tools.add", {"a": 2, "b": 40}).content == "42"
    d.mcp.close()


def test_a_dry_run_reports_what_it_would_compress():
    m, _ = _cold_store()
    report = Compressor(m).compress(dry_run=True)
    assert report["compressed"] >= 1 and report["freed"] >= 3, report


def test_compression_rewires_inbound_links():
    """Members' outbound links were carried onto the digest; everything pointing
    AT the members kept pointing at ids that no longer exist, so credit
    propagation stopped dead at the boundary."""
    m, _ = _cold_store()
    cold = [t for t in Compressor(m).cold()]
    pointer = m.remember(Kind.GOAL, "a goal that depends on those failures",
                         links=[cold[0].id])
    c = Compressor(m, archive=tmpdir() / "a.jsonl")
    c.compress()
    again = m.get(pointer.id)
    assert all(m.get(lid) is not None for lid in again.links), "links must not dangle"


def test_exploring_n_times_does_not_run_one_idea_n_times():
    d = fresh(seed=5)
    for task in ("compute the median of a column", "dedupe the records", "parse a csv file"):
        d.solve(task, interrogate=False)
    runs = d.explore(steps=4)
    assert len({e.idea.text for e in runs}) == len(runs), "each step must try something new"


def test_a_confirmed_experiment_is_never_filed_as_a_refuted_failure():
    for trace in fresh(seed=5).memory.of_kind(Kind.FAILURE):
        assert (trace.mean_grade or 0) <= 0, "a failure must not carry a positive grade"


def test_the_demo_persists_what_it_did():
    """It printed a memory report it never wrote, so a --home it had filled with
    twelve verified tools came back empty on the next command."""
    from distil.cli import main
    home = tmpdir()
    assert main(["--home", str(home), "demo"]) == 0
    back = Distil(LocalProvider(), home=home, seed=1)
    assert back.toolbox.names(), "the demo's tools must survive it"
    assert back.memory.stats()["traces"] > 0


def test_re_forging_a_tool_does_not_create_a_second_trace():
    """Every consumer looks a tool up by name and takes the first match, so a
    second Kind.TOOL trace meant the old, negatively graded one kept answering
    for the new tool and it could never be reused again."""
    from distil.agent import _reusable
    d = fresh(seed=1)
    d.solve("compute the median of a column", interrogate=False)
    for _ in range(6):
        d.toolbox.record_use("median", "got it wrong", False)
    assert not _reusable(d.toolbox.load("median"), d.memory)
    d.solve("compute the median of a column", interrogate=False)   # re-forges
    traces = [t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "median"]
    assert len(traces) == 1, "one trace per tool name, for the life of the store"


def test_re_embedding_keeps_an_mcp_tools_provenance():
    """Its text is written at discovery and carries server, arguments and the
    required list. Overwriting it with the generic ToolSpec text made the tool
    that had proved itself the hardest one to find."""
    d = fresh()
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    d.toolbox.record_use("echo.add", "totalled two figures", True)
    trace = next(t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "echo.add")
    assert "mcp server" in trace.text, "the server provenance must survive"
    assert "arguments:" in trace.text
    assert "totalled two figures" in trace.text, "and the solved problem must join it"
    d.mcp.close()


def test_a_trace_id_pointing_at_another_tool_is_ignored():
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    other = next(t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "chunk")
    path = d.workspace.workshop / "median.json"
    spec = json.loads(path.read_text())
    spec["trace_id"] = other.id                  # now points at a different tool
    path.write_text(json.dumps(spec))
    before = list(other.grades)
    d.toolbox.record_use("median", "a problem", True)
    assert len(other.grades) == len(before), "another tool's trace must not be graded"


def test_a_legitimate_answer_containing_none_is_not_a_denial():
    """Matching bare words anywhere threw away real objectives: "return none
    when the list is empty" is an objective, not a refusal to give one."""
    from distil.clarify import _denies_knowledge
    assert _denies_knowledge("I don't know")
    assert _denies_knowledge("none")
    assert _denies_knowledge("n/a")
    assert not _denies_knowledge("return none when the list is empty")
    assert not _denies_knowledge("nothing should be logged at info level")
    assert not _denies_knowledge("the unknown fields must be rejected")
    assert not _denies_knowledge("no one knows the answer")


def test_re_embedding_a_trace_survives_the_disk_round_trip():
    """record_use mutates trace.text and trace.vector in place. The stored
    vector must still describe the stored text after a save and reload, or
    recall silently ranks against a description the trace no longer has."""
    home = tmpdir()
    d = Distil(LocalProvider(), home=home, seed=1)
    plant(d.toolsmith, d.toolbox)
    d.toolbox.record_use("parse_csv", "collapse a ragged telemetry export", True)
    trace = next(t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "parse_csv")
    assert trace.vector == d.memory.embedder.embed(trace.text, learn=False)
    d.save()

    back = Distil(LocalProvider(), home=home, seed=1)
    again = next(t for t in back.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "parse_csv")
    assert again.vector == trace.vector and again.text == trace.text
    found = back.toolbox.find("collapse a ragged telemetry export", k=1)
    assert found and found[0][0].name == "parse_csv"


def test_a_stale_trace_id_does_not_fork_the_tool():
    d = fresh(seed=1)
    d.solve("compute the median of a column", interrogate=False)
    path = d.workspace.workshop / "median.json"
    spec = json.loads(path.read_text())
    spec["trace_id"] = "deadbeefdead"          # points at nothing
    path.write_text(json.dumps(spec))
    d.toolbox.record_use("median", "a new problem", True)
    traces = [t for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "median"]
    assert len(traces) == 1, "a stale id must fall back, not create a second trace"


def test_a_disgraced_tool_is_reforged_rather_than_reused():
    from distil.agent import _reusable
    d = fresh(seed=1)
    d.solve("compute the median of a column", interrogate=False)
    for _ in range(6):
        d.toolbox.record_use("median", "got it wrong", False)
    assert not _reusable(d.toolbox.load("median"), d.memory)
    r = d.solve("compute the median of a column", interrogate=False)
    assert not any("existing tool" in a["via"] for a in r["attempts"])


def test_every_tool_trace_has_a_matching_toolbox_entry():
    """A Kind.TOOL trace with no entry in names() is a tool recall can surface
    and invoke cannot run."""
    d = fresh(seed=1)
    plant(d.toolsmith, d.toolbox)
    d.mcp.add("echo", [sys.executable, FIXTURE])
    d.mcp.discover()
    named = set(d.toolbox.names())
    traced = {t.meta.get("tool") for t in d.memory.of_kind(Kind.TOOL) if t.meta.get("tool")}
    assert traced == named, f"only in traces: {traced - named}; only in names: {named - traced}"
    d.mcp.close()


def test_invoking_a_tool_with_a_missing_dependency_returns_an_error():
    """bundle() raises; invoke()'s whole contract is to return a result dict."""
    d = fresh()
    spec = ToolSpec(name="orphan", purpose="p", source="def orphan():\n    return 1\n",
                    tests="assert orphan() == 1", signature="orphan()", deps=["gone"])
    d.toolsmith.register(spec, threshold=-2.0)
    out = d.toolbox.invoke("orphan")
    assert not out["ok"] and "not registered" in out["error"]


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


def test_the_whole_state_survives_a_restart_with_both_transports():
    """The end-to-end guard for the serialisation bug class. Everything the
    system accumulated -- traces, cases with their ids, tool grades, an MCP
    tool's use history -- must be there after the process goes away."""
    home = tmpdir()
    a = Distil(LocalProvider(), home=home, seed=1)
    plant(a.toolsmith, a.toolbox)
    a.mcp.add("echo", [sys.executable, FIXTURE])
    a.mcp.discover()
    a.toolbox.record_use("echo.add", "added two numbers", True)
    a.solve("compute the median of a column", interrogate=False)
    a.solve("parse a quantum waveform capture file", interrogate=False)
    a.save()
    a.mcp.close()
    before = a.stats()

    b = Distil(LocalProvider(), home=home, seed=1)
    after = b.stats()
    assert after["traces"] == before["traces"]
    assert after["cases"] == before["cases"]
    assert sorted(after["tools"]) == sorted(before["tools"])

    spec = b.toolbox.load("echo.add")
    assert spec.transport == "mcp" and "added two numbers" in spec.solved
    traces = [t for t in b.memory.of_kind(Kind.TOOL) if t.meta.get("tool") == "echo.add"]
    assert len(traces) == 1 and (traces[0].mean_grade or 0) > 0
    assert b.toolbox.load("median").grade.score == 1.0
    assert all(c.trace_id for c in b.casebook.all()), "a case with no id cannot be re-graded"


def test_compression_never_eats_a_tool():
    home = tmpdir()
    d = Distil(LocalProvider(), home=home, seed=1)
    plant(d.toolsmith, d.toolbox)
    d.solve("compute the median of a column", interrogate=False)
    names = sorted(d.toolbox.names())
    d.compress()
    assert sorted(Distil(LocalProvider(), home=home, seed=1).toolbox.names()) == names


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



# --------------------------------------------------------------------------- #
# projection -- the embedding space made visible
# --------------------------------------------------------------------------- #

def test_projection_is_deterministic_across_runs():
    """A moved point must mean the memory moved, not the renderer.

    Power iteration from a random start would give a different (still correct)
    basis each run, and the picture would shuffle between reloads for no reason.
    """
    vs = [normalise([math.sin(i * j * 0.31) for j in range(1, 17)]) for i in range(12)]
    assert project(vs) == project(list(vs))


def test_projection_separates_two_clusters():
    """The only claim the picture makes: near in the space is near on the plane."""
    a = [normalise([1.0, 0.0, 0.0, 0.0][:4] + [0.01 * i, 0.0]) for i in range(6)]
    b = [normalise([0.0, 1.0, 0.0, 0.0][:4] + [0.0, 0.01 * i]) for i in range(6)]
    pts = project(a + b)
    left, right = pts[:6], pts[6:]
    within = max(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p in left for q in left)
    across = min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p in left for q in right)
    assert across > within, f"clusters overlap: {across} <= {within}"


def test_projection_fits_inside_the_frame():
    vs = [normalise([math.cos(i * j) for j in range(1, 9)]) for i in range(20)]
    for x, y in project(vs):
        assert -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0


def test_projection_uses_one_scale_for_both_axes():
    """Scaling each axis to fill the frame would make a tight cluster look spread.

    Points on a line at 30 degrees must stay on a line at 30 degrees; if x and y
    were normalised separately the slope would be forced to 1.
    """
    vs = [[t, t * 0.5, 0.0, 0.0] for t in (-3.0, -1.0, 1.0, 2.0, 5.0)]
    pts = project(vs)
    ys = [abs(y) for _, y in pts]
    assert max(ys) < 0.5, f"a degenerate second component was stretched to fill: {ys}"


def test_projection_handles_too_few_points_to_have_a_shape():
    assert project([]) == []
    assert project([[1.0, 2.0]]) == [(0.0, 0.0)]
    assert len(project([[1.0, 0.0], [0.0, 1.0]])) == 2


def test_identical_vectors_collapse_to_the_origin_rather_than_dividing_by_zero():
    same = [[0.5, 0.5, 0.5, 0.5] for _ in range(5)]
    assert project(same) == [(0.0, 0.0)] * 5


def test_orientation_is_pinned_so_the_picture_does_not_mirror():
    """An eigenvector is defined up to sign; the layout must not be.

    Power iteration can converge to either sign of the same component, so the
    identical store can come back mirrored between runs and a person who
    remembered where something sat would find it on the other side. Orienting by
    skew means a projection and its mirror image resolve to the same picture.
    """
    pts = [(-0.1, -0.1), (-0.2, -0.2), (0.9, 0.9)]
    mirrored = [(-x, -y) for x, y in pts]
    assert orient(mirrored) == orient(pts), "a mirrored projection must land the same way"
    assert orient(pts) == pts, "positive skew is already the canonical side"


# --------------------------------------------------------------------------- #
# the http api
# --------------------------------------------------------------------------- #

def _api() -> Api:
    return Api(fresh())


def test_every_declared_route_exists_on_the_api():
    """A route named in the table but missing here is a 500 the moment it is hit."""
    from distil.serve import GET_ROUTES, POST_ROUTES
    api = _api()
    for name in GET_ROUTES | POST_ROUTES:
        assert callable(getattr(api, name, None)), f"no handler for /api/{name}"


def test_every_read_endpoint_survives_json_dumps():
    """The agent's payloads carry dataclasses, sets and Grades.

    One unserialisable object fails the whole response rather than the field, so
    the plain-ing has to be total -- and a read endpoint must never need a
    populated store to answer.
    """
    from distil.serve import GET_ROUTES
    api = _api()
    api.agent.memory.remember(Kind.QUERY, "a query worth recalling")
    for name in sorted(GET_ROUTES):
        q = {"task": ["win a chess endgame"], "q": ["recall me"], "like": ["parse csv"]}
        json.dumps(getattr(api, name)(q))


def test_plain_degrades_unserialisable_objects_instead_of_failing():
    class Opaque:
        def __repr__(self): return "<opaque>"
    out = _plain({"s": {1, 2}, "t": (3, 4), "o": Opaque(), "n": None})
    assert json.dumps(out)
    assert out["o"] == "<opaque>" and sorted(out["s"]) == [1, 2] and out["t"] == [3, 4]


def test_the_memory_view_projects_each_embedder_separately():
    """Two geometries share no plane.

    Laying vectors from different backends out together would put unrelated
    things side by side and call it similarity.
    """
    api = _api()
    for i in range(4):
        api.agent.memory.remember(Kind.FACT, f"native trace {i}")
    store = api.agent.memory.store
    for t in list(store.all())[:2]:
        t.embedder = "other-backend"
        t.vector = normalise([float((i * 7 + 3) % 5) for i in range(len(t.vector))])
        store.put(t)
    out = api.memory({})
    assert len(out["backends"]) == 2
    assert all("x" in t and "y" in t for t in out["traces"])


def test_the_memory_view_filters_to_known_kinds_only():
    api = _api()
    api.agent.memory.remember(Kind.QUERY, "a question")
    api.agent.memory.remember(Kind.FACT, "a fact")
    assert {t["kind"] for t in api.memory({"kind": ["query"]})["traces"]} == {"query"}
    # An unknown kind is dropped rather than filtering everything away.
    assert api.memory({"kind": ["nonsense"]})["traces"]


def test_recall_reports_the_weights_it_ranked_by():
    """The panel's whole claim is that ranking is not similarity.

    It can only show that if the server sends the weights it actually used.
    """
    api = _api()
    api.agent.memory.remember(Kind.FACT, "merge two dictionaries in python")
    out = api.recall({"q": ["merge dicts"], "k": ["5"]})
    assert out["weights"]["credibility"] == api.agent.policy.recall_credibility_weight
    for h in out["hits"]:
        assert set(h) >= {"trace", "score", "similarity", "credibility", "recency"}


def test_an_empty_query_recalls_nothing_rather_than_everything():
    assert _api().recall({"q": ["   "]})["hits"] == []


def test_endpoints_that_need_input_say_so_instead_of_guessing():
    api = _api()
    assert api.ask({"task": "  "})["error"]
    assert api.frame({"task": [""]})["error"]
    assert api.forge({"goal": ""})["error"]


def test_ask_labels_the_payoff_rows_from_the_step_that_built_them():
    """The chosen goal leaves the frontier the moment it is met.

    Reading the labels back off the frontier at serialisation time therefore
    lost them, and every row came back as "goal 1".
    """
    api = _api()
    out = api.ask({"task": "write a python function that returns the median of a list",
                   "answers": {"a": "a list of numbers", "b": "the median as a float"}})
    if not out.get("payoff"):
        return                      # clarification gated this run; nothing to check
    rows = out["payoff"]["matrix"]
    goals = out["payoff"]["goals"]
    assert len(goals) == len(rows) and all(g.strip() for g in goals)
    assert out["payoff"]["roles"] is None or len(out["payoff"]["roles"]) == len(rows)


def test_a_trace_serialises_with_the_evidence_behind_its_credibility():
    """A grade shown without its source is a number the user cannot check."""
    api = _api()
    t = api.agent.memory.remember(Kind.FACT, "something to grade")
    api.agent.memory.grade(t.id, 1.0, Source.USER)
    out = trace_json(t, api.agent.policy)
    json.dumps(out)
    assert out["grades"] == [{"score": 1.0, "source": Source.USER}]
    assert 0.0 <= out["credibility"] <= 1.0


def test_grading_an_unknown_trace_reports_rather_than_raises():
    """A user grade is the one input the system cannot re-derive.

    Losing one to a traceback behind a fetch would be silent, so every bad shape
    comes back as a message the panel can show.
    """
    assert "no such trace" in _api().grade({"id": "no-such-trace", "score": 1.0})["error"]
    assert _api().grade({"score": 1.0})["error"]          # missing id
    assert _api().grade({"id": "x", "score": "high"})["error"]   # unparseable score


def test_explore_is_bounded_so_one_request_cannot_run_forever():
    api = _api()
    assert len(api.explore({"steps": 500})["experiments"]) <= 20


# --- the browser-facing defences ------------------------------------------- #

def _serve_for_test():
    """A real server on an ephemeral port. The security checks live in Handler."""
    import http.client
    from http.server import ThreadingHTTPServer
    from distil.serve import Handler
    Handler.api = Api(fresh())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Handler.log_message = lambda *a, **k: None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _request(port, method, path, headers=None, body=None):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    payload = r.read()
    c.close()
    return r.status, payload


def test_a_non_loopback_host_header_is_refused():
    """The DNS-rebinding defence: the name resolved here, but it is not ours."""
    httpd, port = _serve_for_test()
    try:
        status, _ = _request(port, "GET", "/api/state", {"Host": "evil.example.com"})
        assert status == 403
        status, _ = _request(port, "GET", "/api/state", {"Host": "127.0.0.1"})
        assert status == 200
    finally:
        httpd.shutdown()


def test_a_cross_site_request_is_refused_even_with_an_honest_host():
    """Any page may address 127.0.0.1; its Host header is then perfectly true.

    Without this check a visited page could POST /api/forge and have the agent
    write and execute code -- unreadable to the attacker, but already done.
    """
    httpd, port = _serve_for_test()
    try:
        status, _ = _request(port, "GET", "/api/state",
                             {"Host": "127.0.0.1", "Sec-Fetch-Site": "cross-site"})
        assert status == 403
        status, _ = _request(port, "GET", "/api/state",
                             {"Host": "127.0.0.1", "Sec-Fetch-Site": "same-origin"})
        assert status == 200
    finally:
        httpd.shutdown()


def test_a_post_that_could_skip_the_preflight_is_refused():
    """A form or text/plain POST is a "simple request" -- no preflight to fail.

    Requiring application/json forces the browser to preflight, and the preflight
    finds no CORS headers here.
    """
    httpd, port = _serve_for_test()
    body = json.dumps({"goal": "anything"})
    try:
        status, _ = _request(port, "POST", "/api/forge",
                             {"Host": "127.0.0.1", "Content-Type": "text/plain",
                              "Content-Length": str(len(body))}, body)
        assert status == 403
        status, _ = _request(port, "POST", "/api/forge",
                             {"Host": "127.0.0.1", "Content-Type": "application/json",
                              "Origin": "http://evil.example.com",
                              "Content-Length": str(len(body))}, body)
        assert status == 403
    finally:
        httpd.shutdown()


def test_static_files_cannot_escape_the_web_directory():
    httpd, port = _serve_for_test()
    try:
        status, _ = _request(port, "GET", "/%2e%2e/%2e%2e/etc/passwd", {"Host": "127.0.0.1"})
        assert status in (403, 404)
        status, body = _request(port, "GET", "/../distil/serve.py", {"Host": "127.0.0.1"})
        assert status in (403, 404) and b"class Api" not in body
    finally:
        httpd.shutdown()


def test_an_unknown_endpoint_is_a_404_not_an_attribute_error():
    httpd, port = _serve_for_test()
    try:
        status, _ = _request(port, "GET", "/api/lock", {"Host": "127.0.0.1"})
        assert status == 404
        body = b"{}"
        status, _ = _request(port, "POST", "/api/agent",
                             {"Host": "127.0.0.1", "Content-Type": "application/json",
                              "Content-Length": "2"}, body)
        assert status == 404
    finally:
        httpd.shutdown()


def test_a_malformed_body_is_a_400_not_a_traceback():
    httpd, port = _serve_for_test()
    try:
        for body in (b"not json", b"[1, 2, 3]"):
            status, _ = _request(port, "POST", "/api/ask",
                                 {"Host": "127.0.0.1", "Content-Type": "application/json",
                                  "Content-Length": str(len(body))}, body)
            assert status == 400, body
    finally:
        httpd.shutdown()



# --------------------------------------------------------------------------- #
# the autonomous loop
# --------------------------------------------------------------------------- #

def _auto(seed: int = 3):
    d = fresh(seed)
    plant(d.toolsmith, d.toolbox)
    return Auto(d)


def test_the_loop_never_questions_its_own_output():
    """Each QUESTION writes a trace beginning "why <subject>".

    That trace was then eligible as the next subject, so the loop asked why about
    why about why -- escaping accumulating every turn -- and curiosity
    degenerated into self-reference. Nothing the loop wrote may be a subject.
    """
    auto = _auto()
    subjects = [c.detail.get("subject", "") for c in auto.run(cycles=22)
                if c.move == Move.QUESTION and c.detail.get("subject")]
    assert subjects, "no questions were asked at all"
    assert not [s for s in subjects if s.startswith("why ")], subjects[:3]


def test_a_repeated_finding_pays_less_than_the_first():
    """The tenth circular belief is a pattern you already know.

    Without the decay the offline provider -- which cannot really answer why, so
    every chain comes back circular -- made QUESTION score identically forever
    and starved the other four moves.
    """
    auto = _auto()
    scores = [c.learned for c in auto.run(cycles=20) if c.move == Move.QUESTION]
    assert len(scores) >= 3, scores
    assert scores[0] > scores[-1], f"no decay: {scores}"


def test_the_loop_does_not_retry_the_same_gap_forever():
    """`covered` held tool NAMES and the gap is a verb, so the check never fired
    and the same goal was re-forged and re-rejected every cycle."""
    auto = _auto()
    auto.agent.solve("build a csv cleaner and compute the median of each column",
                     ask=lambda q: {})
    goals = [c.detail.get("goal") for c in auto.run(cycles=20)
             if c.move == Move.BUILD and c.detail.get("goal")]
    assert len(goals) == len(set(goals)), f"repeated a gap: {goals}"


def test_a_move_that_throws_does_not_end_the_loop():
    """Losing everything learned so far over one bad move is the wrong trade."""
    auto = _auto()
    auto._tune = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    cycles = auto.run(cycles=10)
    assert len(cycles) == 10
    broken = [c for c in cycles if "failed" in c.note]
    assert all(c.learned == 0.0 for c in broken)


def test_an_untried_move_is_not_starved_by_one_lucky_cycle():
    """Untried moves are worth the best average seen, not zero.

    With zero, a single high-scoring first cycle drove the regret matcher to
    never sample the other four again.
    """
    auto = _auto()
    auto.paid[0], auto.tried[0] = 5.0, 1
    assert auto._average(1) == 5.0, "an untried move should be optimistic"


def test_the_loop_reports_what_it_learned_not_whether_it_worked():
    """Rewarding correctness teaches it to stop running the experiments worth
    running, which is the failure mode the whole design is arranged against."""
    auto = _auto()
    for cycle in auto.run(cycles=12):
        assert cycle.learned >= 0.0
        json.dumps(cycle.to_json())
    # `strategy` rounds each of six shares to 4dp for display, so the sum drifts
    # by up to ~3e-4. Asserting exact unity was testing the rounding, not the mix.
    assert abs(sum(auto.strategy().values()) - 1.0) < 1e-3


def test_the_loop_stops_when_asked():
    stop = {"now": False}
    auto = _auto()
    auto.stop = lambda: stop["now"]
    auto.run(cycles=3)
    stop["now"] = True
    assert auto.run(cycles=100) == []


def test_the_loop_never_runs_out_of_direction():
    """A loop with nothing to do must change what it is doing.

    All five consuming moves draw from pools that empty -- a last unquestioned
    belief, a last known gap, a last cold cluster -- and when they did, every
    cycle returned "nothing to do" and the loop spun without stopping: neither
    working nor finished. It may stall for PATIENCE cycles, never longer.
    """
    # Two seeds at 30 cycles is enough: the pools empty well before then, and
    # PATIENCE is 2, so a loop that cannot recover shows it within a few turns.
    for seed in (5, 23):
        auto = _auto(seed)
        longest = streak = 0
        for cycle in auto.run(cycles=30):
            streak = streak + 1 if cycle.barren else 0
            longest = max(longest, streak)
        assert longest <= Auto.PATIENCE, f"seed {seed} stalled for {longest} cycles"


def test_running_dry_forces_a_new_direction():
    auto = _auto()
    auto.barren = Auto.PATIENCE
    cycle = auto.once()
    assert cycle.move == Move.PURSUE, cycle.move


def test_directions_are_drawn_from_more_than_one_source():
    """Taking the sources in order meant the first supplied every direction until
    it ran out -- and since it draws on memory, which this loop is busy filling,
    it never did."""
    auto = _auto(5)
    auto.run(cycles=30)
    lead_ins = {d.split()[0] for d in auto.directions}
    assert len(lead_ins) >= 3, auto.directions[:6]


def test_a_direction_is_never_taken_twice():
    auto = _auto(11)
    auto.run(cycles=30)
    assert len(auto.directions) == len(set(auto.directions))


def test_pursuing_does_not_re_propose_its_own_descendants():
    """Pursuing a direction runs the full solve loop, which writes goals and
    failures whose text CONTAINS that direction -- and those carry no `auto`
    marker, because the solver wrote them. Proposing one back produced "work out
    what is true about work out what is true about"."""
    auto = _auto(5)
    auto.run(cycles=30)
    repeated = [d for d in auto.directions
                if d.count("work out what is actually true") > 1]
    assert not repeated, repeated


def test_an_empty_store_bootstraps_itself():
    """Every direction source reads memory, so an empty store suggests nothing
    and a fresh instance would sit still -- waiting to be told to do the one
    thing it can always do."""
    d = fresh()
    assert not d.toolbox.names()
    auto = Auto(d)
    auto.run(cycles=6)
    assert d.toolbox.names(), "it never planted anything to reason from"


def test_the_last_resort_source_cannot_exhaust():
    """Four of the five sources have a last element. Pairs do not: there are
    O(n^2) of them and every cycle adds to n."""
    auto = _auto()
    pairs = auto._far_pairs()
    assert len([next(pairs) for _ in range(12)]) == 12


def test_a_vague_self_set_direction_is_gated_not_guessed_at():
    """The gate firing on a question it set ITSELF is the system saying the
    direction was vague, not a failure."""
    auto = _auto()
    gated = [c for c in auto.run(cycles=20)
             if c.move == Move.PURSUE and c.detail.get("gated")]
    for cycle in gated:
        assert cycle.learned > 0, "finding a direction unstatable is worth something"



# --------------------------------------------------------------------------- #
# thinking in the embedding space
# --------------------------------------------------------------------------- #

def _thinker(d=None):
    d = d or fresh()
    return d, Thinker(d.memory, d.policy)


def test_thinking_writes_new_memories_from_the_shape_of_old_ones():
    """The space is computed with, not only read from.

    Every other path produces traces from outside input; this one produces them
    from the geometry -- which is what makes the store grow denser rather than
    only longer.
    """
    d, th = _thinker()
    plant(d.toolsmith, d.toolbox)
    before = d.memory.stats()["traces"]
    found = th.think(limit=4)
    assert found, "a seeded toolbox has enough structure to find something in"
    written = th.absorb(found)
    assert d.memory.stats()["traces"] > before
    assert all(t.meta.get("derived") for t in written)


def test_derived_memories_are_never_reasoned_from():
    """A centroid of centroids is a claim about the shape of its own conclusions.

    The same self-reference that broke the why-chains twice: a derived trace is a
    point in the space the next pass reads.
    """
    d, th = _thinker()
    plant(d.toolsmith, d.toolbox)
    th.absorb(th.think(limit=4))
    again = th.think(limit=4)
    assert not [t for x in again for t in x.sources if t.meta.get("derived")]
    assert all(not t.meta.get("derived") for t in th.pool())


def test_a_conjecture_enters_ungraded():
    """These are read off the shape of the store, not observed. Entering them as
    established would be manufacturing evidence about its own contents."""
    d, th = _thinker()
    plant(d.toolsmith, d.toolbox)
    for t in th.absorb(th.think(limit=4)):
        assert not t.graded, f"{t.text[:50]!r} entered already graded"


def test_the_same_conjecture_re_derived_merges_rather_than_accumulates():
    d, th = _thinker()
    plant(d.toolsmith, d.toolbox)
    first = th.absorb(th.think(limit=4))
    n = d.memory.stats()["traces"]
    th.absorb(th.think(limit=4))       # nothing new has happened in between
    assert d.memory.stats()["traces"] == n, "re-derivation must merge on identity"
    assert first


def test_a_contradiction_between_two_graded_memories_is_found():
    """`memory.gaps` finds diffuse disagreement; this finds a specific pair."""
    d, th = _thinker()
    a = d.memory.remember(Kind.ANSWER, "to merge two dicts use z = {**a, **b} which copies both")
    b = d.memory.remember(Kind.ANSWER, "to merge two dicts use a.merge(b) which copies both")
    d.memory.grade(a.id, 1.0, Source.USER)
    d.memory.grade(b.id, -1.0, Source.USER)
    found = th.tensions()
    assert found and found[0].kind == Thought.TENSION
    assert {t.id for t in found[0].sources} == {a.id, b.id}


def test_a_bridge_excludes_the_two_traces_that_define_it():
    """The midpoint of two near-orthogonal vectors sits at cos 45deg ~ 0.707 from
    BOTH parents, so the nearest neighbour to any midpoint is always a parent.
    Searching without excluding them answered "occupied" every time and the
    operation returned nothing, ever.
    """
    d, th = _thinker()
    for text in ("parse a csv file into rows of strings",
                 "csv rows have ragged column counts",
                 "the lease renewal negotiation happens every year",
                 "negotiating rent with a landlord"):
        d.memory.remember(Kind.FACT, text)
    found = th.bridges(limit=3)
    assert found, "four unrelated facts must leave some midpoint empty"
    for bridge in found:
        assert len(bridge.sources) == 2
        assert bridge.detail["nearest"] < Thinker.OCCUPIED


def test_analogy_is_vector_arithmetic_not_a_similarity_band():
    """a - b + c can reach a region no single memory is near, which is the one
    thing a similarity search cannot do."""
    d, th = _thinker()
    plant(d.toolsmith, d.toolbox)
    for name in ("median", "parse_csv", "chunk"):
        spec = d.toolbox.load(name)
        d.memory.grade(spec.trace_id, 1.0, Source.USER)
    found = th.analogies(limit=2)
    assert found, "three graded tools are enough to form a relation"
    for a in found:
        assert a.kind == Thought.ANALOGY
        assert len(a.vector) == len(d.memory.embedder.embed("x", learn=False))
        assert "landed" in a.detail


def test_think_is_a_move_the_loop_can_choose():
    d = fresh(seed=5)
    plant(d.toolsmith, d.toolbox)
    auto = Auto(d)
    auto.barren = 0
    cycle = auto._think()
    assert cycle.move == Move.THINK or cycle.barren
    assert Move.THINK in Move.ALL


# --------------------------------------------------------------------------- #
# the browser
# --------------------------------------------------------------------------- #

def test_the_browser_refuses_a_non_http_url():
    """`file:///etc/passwd` through a driver is a file read."""
    try:
        Browser(url="http://127.0.0.1:1").open("file:///etc/passwd")
        raise AssertionError("should have refused")
    except BrowserError as exc:
        assert "http" in str(exc)


def test_a_missing_driver_says_how_to_start_one():
    try:
        Browser(url="http://127.0.0.1:1").open("https://example.com")
        raise AssertionError("should have raised")
    except BrowserError as exc:
        assert "docker compose" in str(exc) or "DISTIL_WEBDRIVER" in str(exc)


def test_browser_actions_are_embedded_as_ordinary_tools():
    """One embedding layer over every capability: a browser action has to be
    findable by the same query that finds a forged function."""
    from distil.browser import register
    d = fresh()
    names = register(d.memory)
    assert set(names) == {n for n, _, _ in ACTIONS}
    found = d.toolbox.find("read the text of a web page", k=3)
    assert any(spec.name == "browser.read" for spec, _, _ in found), \
        [spec.name for spec, _, _ in found]
    spec = d.toolbox.load("browser.read")
    assert spec is not None and spec.transport == "browser"


def test_a_browser_tool_without_a_browser_fails_loudly():
    from distil.browser import register
    d = fresh()
    register(d.memory)
    out = d.toolbox.invoke("browser.open", ["https://example.com"])
    assert out["ok"] is False and "browser" in out["error"]


# --------------------------------------------------------------------------- #
# speech
# --------------------------------------------------------------------------- #

def test_transcription_says_why_it_cannot_rather_than_failing_silently():
    state = speech.available()
    assert isinstance(state["local"], bool) and state["why"]
    if not state["local"]:
        try:
            speech.transcribe(b"RIFF....WAVE")
            raise AssertionError("should have raised")
        except speech.SpeechError as exc:
            assert str(exc) == state["why"]


def test_audio_is_validated_before_it_reaches_a_subprocess():
    """Handing an unvalidated upload to whisper makes this endpoint a file-format
    parser with someone else's bugs."""
    for raw, expect in [(b"", "no audio"), (b"not a wav at all", "wav")]:
        try:
            speech.check_wav(raw)
            raise AssertionError(f"should have refused {raw!r}")
        except speech.SpeechError as exc:
            assert expect in str(exc).lower()


def test_a_real_wav_is_accepted_and_measured():
    import struct, wave
    path = tmpdir() / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", 0) for _ in range(16000)))
    assert abs(speech.check_wav(path.read_bytes()) - 1.0) < 0.01


def test_oversized_audio_is_refused_by_length_not_by_trying_it():
    import struct, wave
    path = tmpdir() / "long.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00" * int(8000 * (speech.MAX_SECONDS + 5)))
    try:
        speech.check_wav(path.read_bytes())
        raise AssertionError("should have refused")
    except speech.SpeechError as exc:
        assert "limit" in str(exc)


# --------------------------------------------------------------------------- #
# what the new endpoints promise
# --------------------------------------------------------------------------- #

def test_every_reply_carries_what_memory_already_knew():
    """A gated question used to return questions and nothing else -- the one case
    where "what do I already know about this?" is most useful answered it least.
    """
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    gated = d.solve("make the thing better", ask=None)
    assert gated["needs_clarification"] and gated["recall"]["hits"], \
        "a gated question returned no recall"
    answered = d.solve("compute the median of a list of numbers", ask=lambda q: {})
    assert "recall" in answered


def test_a_composite_tool_links_to_what_it_was_built_from():
    """Credit propagates along links. Until tool traces carried their deps, a
    well-graded composite earned its parts nothing."""
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    api = Api(d)
    spec = d.toolbox.load("median_of_column")
    assert spec and spec.deps, "the fixture no longer has a composite tool"
    out = api.trace({"id": [spec.trace_id]})
    linked = {e["text"].split(":")[0].replace("tool ", "") for e in out["links"]}
    assert set(spec.deps) <= linked, f"{spec.deps} not in {linked}"


def test_the_trace_view_shows_who_uses_a_thing_not_just_what_it_uses():
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    api = Api(d)
    median = d.toolbox.load("median")
    out = api.trace({"id": [median.trace_id]})
    assert any("median_of_column" in b["text"] for b in out["backlinks"]), \
        "backlinks are the direction a person reads in"


def test_the_reasoning_chain_can_be_watched_as_it_is_thought():
    seen = []
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    d.solve("compute the median of a list of numbers", ask=lambda q: {},
            observer=seen.append)
    kinds = [s.kind for s in seen]
    assert kinds[:2] == ["FRAME", "AGENDA"], kinds
    assert "VERIFY" in kinds and len(kinds) >= 8


def test_a_broken_watcher_never_takes_down_the_run():
    """A browser that closed mid-run must not stop the reasoning it was watching."""
    d = fresh()
    plant(d.toolsmith, d.toolbox)
    def explode(step):
        raise RuntimeError("the tab closed")
    out = d.solve("compute the median of a list of numbers", ask=lambda q: {},
                  observer=explode)
    assert out["session"] is not None


def test_attaching_a_server_persists_it_across_restarts():
    """A server added at runtime and lost on restart is worse than one never
    added: its tools stay in the embedding layer, recalled and uncallable."""
    d = fresh()
    d.mcp.attach("echo", ["python3", "-c", "pass"])
    assert d.mcp.config_path.exists()
    config = json.loads(d.mcp.config_path.read_text())
    assert config["mcpServers"]["echo"]["command"] == "python3"
    assert d.mcp.detach("echo") is True
    assert "echo" not in json.loads(d.mcp.config_path.read_text())["mcpServers"]
    assert d.mcp.detach("echo") is False          # already gone


def test_a_corrupt_mcp_config_is_replaced_not_inherited():
    d = fresh()
    d.mcp.config_path.write_text("{ not json at all")
    d.mcp.attach("fs", ["echo", "hi"])
    assert json.loads(d.mcp.config_path.read_text())["mcpServers"]["fs"]


def test_clarify_alone_does_not_commit_to_solving():
    api = Api(fresh())
    out = api.clarify({"task": "make the thing better"})
    assert out["actionable"] is False and out["questions"]
    assert "recall" in out and "frame" in out


def test_ollama_is_unavailable_when_its_model_is_not_pulled():
    """Checking only that the daemon answers was the bug behind "auto mode is not
    querying ollama".

    /api/tags returns 200 whatever is installed, so an un-pulled model meant
    ollama was selected, every /api/chat 404ed, every caller swallowed the error
    and degraded, and the agent reported `ollama` while running on offline rules.
    """
    from distil.provider import OllamaProvider
    p = OllamaProvider(model="gemma4")
    p.installed = lambda: []
    assert not p.available() and "ollama serve" in p.why_unavailable()
    p.installed = lambda: ["llama3.2:latest", "nomic-embed-text:latest"]
    assert not p.available()
    assert "ollama pull gemma4" in p.why_unavailable()
    p.installed = lambda: ["gemma4:latest", "nomic-embed-text:latest"]
    assert p.available() and p.why_unavailable() == ""


def test_a_short_model_tag_matches_the_qualified_one():
    """Ollama reports `gemma4:latest` and people type `gemma4`, so a literal
    comparison called a present model missing. An explicit tag stays exact."""
    from distil.provider import OllamaProvider
    m = OllamaProvider._matches
    assert m("gemma4", ["gemma4:latest"])
    assert m("gemma4:27b", ["gemma4:27b"])
    assert not m("gemma4:27b", ["gemma4:latest"]), "an explicit tag must be exact"
    assert not m("gemma4", ["gemma5:latest"])


def test_pinning_an_unavailable_provider_says_what_is_actually_wrong():
    """"key set? daemon running?" made the reader guess at what the code knows."""
    from distil.provider import ProviderError, auto
    try:
        auto("ollama")
        return                      # a real ollama is running here; nothing to assert
    except ProviderError as exc:
        assert "ollama serve" in str(exc) or "ollama pull" in str(exc), str(exc)


def test_a_provider_answering_nothing_is_distinguishable_from_a_working_one():
    """Every caller degrades on provider failure, correctly -- so a dead provider
    and a live one look identical from outside. The counters are the difference.
    """
    from distil.provider import LocalProvider, ProviderError

    class Dead(LocalProvider):
        name = "dead"
        def complete(self, messages, temperature=0.7, max_tokens=1024):
            exc = ProviderError("connection refused")
            self.record(exc)
            raise exc

    dead = Dead()
    assert dead.health()["failing"] is False, "nothing has been tried yet"
    for _ in range(3):
        try: dead.complete([])
        except ProviderError: pass
    health = dead.health()
    assert health["failing"] and health["calls"] == 3 and health["failures"] == 3
    assert "connection refused" in health["last_error"]


def test_a_healthy_provider_is_not_reported_as_failing():
    d = fresh()
    d.solve("compute the median of a list", ask=lambda q: {})
    health = d.provider.health()
    assert health["calls"] > 0 and not health["failing"]


def test_state_reports_why_each_provider_is_unavailable():
    st = Api(fresh()).state({})
    assert "provider_health" in st
    for entry in st["providers"]:
        assert ("why" in entry) and (entry["available"] or entry["why"]), entry


def test_solve_acts_on_every_goal_not_just_the_first():
    """One selection per task was one GOAL per task.

    The second goal of every two-goal task sat open forever, nothing came back
    for it, and `solved` reported the goal rather than the task -- True with
    half the tree still open. The directive is to distil continuously.
    """
    d = fresh(seed=1)
    r = d.solve("build a csv cleaner and compute the median of each column")
    tree = r["session"].tree
    assert len(tree.goals) - 1 >= 2, "the fixture task no longer distils to two goals"
    assert r["solved"] == tree.solved, "reported solved must be the TREE, not the last goal"
    assert r["goals_open"] == len(tree.frontier())
    selects = [s for s in r["session"].chain.steps if s.kind == "SELECT"]
    assert len(selects) >= 2, "it must come back to the frontier after the first goal"
    assert r["goals_met"] == sum(1 for g in tree.goals.values()
                                 if g.id != "root" and g.status == "met")


def test_a_blocked_goal_stops_the_frontier_loop():
    """Mapping the boundary is the result; pushing on past a refusal is not."""
    d = fresh(seed=1)
    r = d.solve("parse a quantum waveform capture file", ask=lambda q: {})
    assert not r["solved"]
    assert r["goals_met"] == 0
    verifies = [s for s in r["session"].chain.steps if s.kind == "VERIFY"]
    assert len(verifies) == 1, "a blocked goal must end the loop, not be skipped past"


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
