"""FilterBankRadix tests.  Run: python3 -m tests.test_fbradix  (from FilterBankRadix/)

Pure standard library, a few seconds, deterministic.  Every test names the
invariant it protects rather than the function it calls - several of them are
regressions for bugs this architecture actually had, and those say so.
"""

import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fbradix import corpus
from fbradix.activation import DEAD_EPS, is_dead, sine, sine_partials, tanh_partials
from fbradix.bank import (
    ConstantRouter,
    CycleRouter,
    FilteredRadixBank,
    FilterRouter,
    OracleRouter,
    router_from_dict,
)
from fbradix.checkpoint import CheckpointManager
from fbradix.store import read_json, write_json_atomic
from fbradix.check import check_filter, check_pair, check_tree
from fbradix.experiment import contingency, nmi, purity
from fbradix.filter import DIM, NBUCKET, TAU, ActivationFilter, features
from fbradix.tree import RadixTreeNet


# -- layer 1 -------------------------------------------------------------------

def test_features_are_scale_free():
    """A segment's features must not depend on how long it is, or the filter
    learns to route by length - which is a property of the cut, not the text."""
    x = features("abcabcabc")
    y = features("abc" * 40)
    assert max(abs(a - b) for a, b in zip(x, y)) < 1e-12, (x, y)
    assert len(x) == DIM == NBUCKET + 1 and x[-1] == 1.0
    assert abs(sum(x[:NBUCKET]) - NBUCKET) < 1e-9, sum(x[:NBUCKET])
    assert features("") == [0.0] * NBUCKET + [1.0]


def test_address_is_exactly_the_sign_pattern():
    """The address is the only thing layer 1 hands to layer 2; if it stops
    being the sign pattern, nothing else in the design means what it says."""
    f = ActivationFilter(bits=4, seed=3)
    for text in ("def f(x):", "{\"a\": 1}", "ordinary words here", "\t}\n"):
        route, resp = f.address(features(text))
        for j, r in enumerate(resp):
            assert ((route >> j) & 1) == (1 if r > 0 else 0), (route, resp)
        assert 0 <= route < f.n_experts == 16


def test_calibration_splits_every_bit_in_half():
    """Without it a unit can start with every segment on one side, and half of
    the addresses are unreachable before training begins."""
    xs = [features(s["text"]) for s in corpus.synthetic(per_source=20, seed=1)]
    f = ActivationFilter(bits=3, seed=7)
    f.calibrate(xs)
    for j in range(f.bits):
        ones = sum(1 for x in xs if f.responses(x)[j] > 0)
        assert abs(ones / len(xs) - 0.5) <= 0.02, (j, ones, len(xs))


def test_the_hinge_moves_a_bit_across_zero():
    """The whole routing update is this: push a response through zero.  If a
    finite number of steps cannot do it, the filter cannot learn a boundary."""
    f = ActivationFilter(bits=2, seed=0)
    x = features("{\"json\": [1, 2, 3]}")
    before = f.responses(x)[0]
    want = -1.0 if before > 0 else 1.0
    for _ in range(200):
        if not f.push(x, 0, want, 0.2):
            break
    after = f.responses(x)[0]
    assert want * after >= TAU - 1e-9, (before, after)
    assert before * after < 0, (before, after)


def test_filter_gradients_match_finite_differences():
    """Both filter shapes, every parameter of both hinges."""
    for kind in ("sine", "monotone"):
        err = check_filter(kind)
        assert err < 1e-6, (kind, err)
    assert check_pair("sine") < 1e-6
    assert check_pair("monotone") < 1e-6


def test_the_two_address_codes_differ_in_the_right_way():
    """A sign-bit address makes every bit a dichotomy of the data and gives
    `2^m` addresses; an argmax address gives one address per unit and never
    asks the partition to factor.  Which one is in use decides what layer 1 can
    express at all, so it must not be a silent default."""
    sign = ActivationFilter(bits=3, seed=0, code="sign")
    arg = ActivationFilter(bits=3, seed=0, code="argmax")
    assert sign.n_experts == 8 and arg.n_experts == 3
    xs = [features(s["text"]) for s in corpus.synthetic(per_source=6, seed=0)]
    for x in xs:
        route, resp = arg.address(x)
        assert resp[route] == max(resp), (route, resp)
        assert 0 <= route < 3
    try:
        ActivationFilter(bits=2, code="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown code must not be accepted silently")


def test_the_argmax_hinge_makes_one_unit_beat_another():
    """The argmax rule's whole job: raise the address layer 2 wanted above the
    one that won, by a margin."""
    f = ActivationFilter(bits=4, seed=1, code="argmax")
    x = features("func main() { return }")
    start, _resp = f.address(x)
    want = (start + 1) % 4
    for _ in range(400):
        if not f.push_pair(x, want, f.address(x)[0], 0.2):
            break
    assert f.address(x)[0] == want, f.responses(x)


def test_a_sine_unit_can_die_along_a_and_b():
    """`Experiments/ActivationFunctionTest/` finding 4: liveness along x says
    nothing about liveness along the parameters.  Here a dead unit is a bit of
    the address stuck at a constant, so the bank loses half its addresses."""
    assert is_dead(0.0, 1.0) and is_dead(1.0, DEAD_EPS / 2)
    assert not is_dead(-1.0, 1 / 3)
    f = ActivationFilter(bits=2, seed=0)
    assert f.dead_units() == []
    f.a[0] = DEAD_EPS / 100
    assert f.dead_units() == [0]
    # |f'| <= |a*b| everywhere is the reason the test above is the right one
    for x in (-5.0, -1.0, 0.0, 2.5, 9.0):
        assert abs(sine_partials(x, f.a[0], f.b[0], 0.0, 0.0)[1]) <= abs(f.a[0] * f.b[0]) + 1e-12


def test_sine_and_monotone_have_the_same_knobs():
    """The monotone arm is a control, so it has to differ in the shape of one
    function and in nothing else - same four parameters, same call."""
    for x in (-2.0, 0.0, 1.7):
        s = sine_partials(x, -1.0, 1 / 3, 0.2, 0.1)
        t = tanh_partials(x, -1.0, 1 / 3, 0.2, 0.1)
        assert len(s) == len(t) == 6 and s[5] == t[5] == 1.0
    assert abs(sine(0.0, -1.0, 1 / 3, 0.0, 0.0)) < 1e-15


def test_filter_survives_a_round_trip():
    """A filter restored from its dict must address every text identically."""
    f = ActivationFilter(bits=3, seed=5)
    xs = [features(s["text"]) for s in corpus.synthetic(per_source=5, seed=2)]
    f.calibrate(xs)
    g = ActivationFilter.from_dict(f.to_dict())
    assert [f.route(x) for x in xs] == [g.route(x) for x in xs]


# -- layer 2 -------------------------------------------------------------------

def test_every_inserted_window_is_findable():
    """The radix invariant: splitting a run must not lose a string that was
    already in the tree."""
    t = RadixTreeNet(depth=4, alphabet=64, seed=0)
    texts = ["abcdefg", "abcxyz", "abcde", "qqq", "abcdefg"]
    for s in texts:
        t.insert_text(s)
    for s in texts:
        for i in range(len(s)):
            w = s[i : i + t.depth + 1]
            assert t.walk(w) is not None, w
    assert t.walk("zzzz") is None


def test_a_unary_chain_is_stored_once():
    """Path compression is the reason this is a radix tree and not a trie: a
    deterministic continuation costs one node, not one node per character."""
    t = RadixTreeNet(depth=8, alphabet=64, seed=0)
    t.insert_text("abcdefghij" * 3)
    plain = sum(len(s) for s in t.seg)
    assert t.num_nodes < plain, (t.num_nodes, plain)
    assert t.branches < t.num_nodes


def test_the_prediction_is_a_probability_distribution():
    """Three smoothings are composed - the expert, the prior, the floor - and
    the composition is only a model if it still sums to one."""
    data = corpus.synthetic(per_source=6, seed=0)
    alpha = corpus.alphabet(data)
    chars = sorted({c for s in data for c in s["text"]})
    prior = RadixTreeNet(depth=1, alphabet=alpha, seed=0)
    t = RadixTreeNet(depth=3, alphabet=alpha, seed=0)
    for s in data:
        prior.insert_text(s["text"])
        t.insert_text(s["text"])
    t.fallback = prior
    t.train([s["text"] for s in data], epochs=2)
    for ctx in ("", "A", "the", "CGT", "!!"):
        total = sum(t.prob(ctx, c) for c in chars)
        assert abs(total - 1.0) < 1e-6 + t.floor, (ctx, total)
        assert total <= 1.0 + 1e-9, (ctx, total)


def test_an_unseen_context_still_gets_an_answer():
    """Suffix backoff: drop the oldest character until the tree recognises
    something.  A model that returns -inf here cannot be scored at all."""
    t = RadixTreeNet(depth=4, alphabet=64, seed=0)
    t.insert_text("the cat sat on the mat")
    assert t.context("zzz") is not None            # backs off to the root
    assert math.isfinite(t.logprob("zzz", "q"))
    assert math.isfinite(t.logprob("", "t"))


def test_one_hop_gradients_match_finite_differences():
    """The rule that trains every expert, checked against the loss it claims
    to differentiate."""
    err = check_tree()
    assert err < 1e-6, err


def test_training_reduces_the_decision_loss():
    t = RadixTreeNet(depth=4, alphabet=64, seed=2)
    texts = ["the cat sat on the mat", "the dog sat on the log", "the cat ran home"]
    for s in texts:
        t.insert_text(s)
    losses = t.train(texts, epochs=60, lr=0.5)
    assert losses[-1] < losses[0] * 0.8, (losses[0], losses[-1])


def test_the_counts_control_changes_only_the_learning():
    """`counts` is the control that says what the one-hop rule is worth: the
    same tree, the same compression, the same backoff chain and the same
    smoothing, with the softmax replaced by relative traversal counts.  If it
    differed in anything else the comparison would mean nothing."""
    texts = [s["text"] for s in corpus.synthetic(per_source=8, seed=0)]
    a = RadixTreeNet(depth=4, alphabet=64, seed=0)
    b = RadixTreeNet(depth=4, alphabet=64, seed=0, scores="counts")
    for t in texts:
        a.insert_text(t)
        b.insert_text(t)
    assert (a.num_nodes, a.stored_chars, a.branches) == (b.num_nodes, b.stored_chars, b.branches)
    assert b.train(texts, epochs=3) == [0.0, 0.0, 0.0], "counts mode has nothing to train"
    for node in (0, 1):
        assert abs(sum(p for _ch, _c, p in b.child_probs(node)) - 1.0) < 1e-9
    chars = sorted({c for t in texts for c in t})
    for ctx in ("", "A", "the"):
        assert abs(sum(b.prob(ctx, c) for c in chars) - 1.0) < 1e-6 + b.floor
    try:
        RadixTreeNet(depth=2, scores="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown scoring mode must not be accepted silently")


def test_a_single_child_costs_nothing_to_train():
    """radixnet's rule: the softmax over one child is 1, so the loss and every
    gradient are exactly zero - the plan must not contain those positions."""
    t = RadixTreeNet(depth=6, alphabet=64, seed=0)
    t.insert_text("aaaaaaaaaaaa")
    for p, _c in t.plan(["aaaaaaaaaaaa"]):
        assert len(t.kids[p]) > 1, p


# -- the regression that shaped the design -------------------------------------

def test_ignorance_is_never_cheaper_than_knowledge():
    """The bug that made the first version route text to the expert that knew
    least about it.

    Without interpolation an expert that has never seen a context pays the
    floor - about eleven bits a character - while an expert that knows the
    context but is unsure pays four, so on some segments *not knowing* scored
    better and the filter learned to exploit it.  An expert must price its own
    register no worse than a foreign one does.
    """
    data = corpus.synthetic(per_source=25, seed=0)
    alpha = corpus.alphabet(data)
    texts = [s["text"] for s in data]
    prior = RadixTreeNet(depth=1, alphabet=alpha, seed=0)
    for t in texts:
        prior.insert_text(t)
    prior.train(texts, epochs=2)
    experts = {}
    for label in ("digits", "dna", "words", "brackets"):
        own = [s["text"] for s in data if s["source"] == label]
        e = RadixTreeNet(depth=4, alphabet=alpha, seed=1)
        for t in own:
            e.insert_text(t)
        e.fallback = prior
        e.train(own, epochs=3)
        experts[label] = e
    for label, e in experts.items():
        sample = [s["text"] for s in data if s["source"] == label][0]
        mine = e.bits_per_char([sample])[0]
        for other, o in experts.items():
            if other != label:
                assert mine < o.bits_per_char([sample])[0], (label, other, mine)


def test_an_empty_expert_falls_back_to_the_prior():
    """An address with no text must cost about what the prior costs, not the
    floor: the cliff is what drowned the routing signal."""
    data = corpus.synthetic(per_source=10, seed=0)
    alpha = corpus.alphabet(data)
    texts = [s["text"] for s in data]
    prior = RadixTreeNet(depth=1, alphabet=alpha, seed=0)
    for t in texts:
        prior.insert_text(t)
    empty = RadixTreeNet(depth=4, alphabet=alpha, seed=0)
    empty.fallback = prior
    with_prior = empty.bits_per_char(texts)[0]
    alone = RadixTreeNet(depth=4, alphabet=alpha, seed=0).bits_per_char(texts)[0]
    assert with_prior < alone / 2, (with_prior, alone)
    assert abs(with_prior - prior.bits_per_char(texts)[0]) < 0.4, with_prior


# -- the bank ------------------------------------------------------------------

def _fit(router, data, **kw):
    train, test = corpus.split(data, seed=0)
    bank = FilteredRadixBank(router, depth=kw.pop("depth", 4),
                             alphabet=corpus.alphabet(data), seed=0)
    bank.fit(train, rounds=kw.pop("rounds", 2), epochs=kw.pop("epochs", 2), test=test)
    return bank, train, test


def test_the_bank_rebuilds_instead_of_extending():
    """A segment that moves must leave nothing behind, or every expert slowly
    becomes the whole corpus and the partition stops meaning anything."""
    data = corpus.synthetic(per_source=15, seed=0)
    bank, train, _test = _fit(FilterRouter(bits=2, seed=0), data)
    assert sum(e.chars for e in bank.experts) == sum(len(s["text"]) for s in train)


def test_routing_is_deterministic_and_total():
    data = corpus.synthetic(per_source=8, seed=0)
    bank, _train, test = _fit(FilterRouter(bits=3, seed=1), data)
    once = bank.assign(test)
    assert once == bank.assign(test)
    assert all(0 <= r < bank.router.n_experts for r in once)


def test_a_random_split_is_worse_than_one_tree():
    """The control that says routing - not merely splitting - is what would
    have to do the work.  Round robin carries no information about the text."""
    data = corpus.synthetic(per_source=20, seed=0)
    one, _t, test = _fit(ConstantRouter(), data)
    rr, _t2, _test2 = _fit(CycleRouter(4), data)
    assert rr.bits_per_char(test)[0] > one.bits_per_char(test)[0]


def test_the_filter_separates_registers_that_are_separable():
    """Four synthetic registers that share almost no characters: if the filter
    cannot address these apart, it cannot address anything apart."""
    data = corpus.synthetic(per_source=20, seed=0)
    bank, _train, test = _fit(FilterRouter(bits=2, seed=0), data)
    routes = bank.assign(test)
    labels = [s["source"] for s in test]
    assert purity(routes, labels) > 0.9, contingency(routes, labels)
    assert nmi(routes, labels) > 0.8


def test_the_oracle_router_uses_the_labels_and_nothing_else_does():
    """The ceiling arm is allowed to see the answer; every other arm must
    produce the same routing whether or not the labels are there."""
    data = corpus.synthetic(per_source=8, seed=0)
    o = OracleRouter([s["source"] for s in data])
    assert o.n_experts == 4
    assert o.route("anything", "dna") != o.route("anything", "words")
    f = FilterRouter(bits=2, seed=0)
    assert f.route("x = 1", "python") == f.route("x = 1", None)


def test_metrics_agree_at_the_extremes():
    labels = ["a", "a", "b", "b"]
    assert purity([0, 0, 1, 1], labels) == 1.0 and nmi([0, 0, 1, 1], labels) == 1.0
    assert purity([0, 0, 0, 0], labels) == 0.5 and nmi([0, 0, 0, 0], labels) == 0.0
    assert nmi([0, 1, 0, 1], labels) == 0.0


# -- persistence ---------------------------------------------------------------

def test_every_component_round_trips():
    """One contract - `to_dict` / `from_dict` - on the tree, the filter and
    every router, so nothing in the bank is the one piece that cannot be
    written down."""
    texts = [s["text"] for s in corpus.synthetic(per_source=6, seed=0)]
    t = RadixTreeNet(depth=4, alphabet=64, seed=3)
    for x in texts:
        t.insert_text(x)
    t.train(texts, epochs=3)
    u = RadixTreeNet.from_dict(t.to_dict())
    assert (u.num_nodes, u.stored_chars, u.branches, u.chars) == \
           (t.num_nodes, t.stored_chars, t.branches, t.chars)
    for ctx in ("", "A", "the", "CG"):
        for ch in "ACGT the":
            assert abs(t.prob(ctx, ch) - u.prob(ctx, ch)) < 1e-15, (ctx, ch)

    for r in (ConstantRouter(), CycleRouter(4), OracleRouter(["a", "b"]),
              FilterRouter(bits=3, seed=1), FilterRouter(bits=4, seed=2, code="argmax")):
        q = router_from_dict(r.to_dict())
        assert type(q) is type(r) and q.n_experts == r.n_experts
        assert all(q.route(x, "a") == r.route(x, "a") for x in texts)


def test_a_saved_bank_predicts_identically():
    """The point of saving at all: a restored bank is the same model, not a
    similar one.  Both layers travel, and the shared prior is reattached - an
    expert that lost it would price an unknown context at the floor and quietly
    change every number."""
    data = corpus.synthetic(per_source=10, seed=0)
    train, test = corpus.split(data, seed=0)
    bank = FilteredRadixBank(FilterRouter(bits=2, seed=0), depth=4,
                             alphabet=corpus.alphabet(data), seed=0)
    bank.fit(train, rounds=2, epochs=2)
    tmp = tempfile.mkdtemp()
    try:
        for name in ("bank.json", "bank.json.gz"):
            path = bank.save(os.path.join(tmp, name))
            back = FilteredRadixBank.load(path)
            assert back.assign(test) == bank.assign(test), name
            assert abs(back.bits_per_char(test)[0] - bank.bits_per_char(test)[0]) < 1e-15
            assert back.generate("the ", length=40, seed=1) == \
                   bank.generate("the ", length=40, seed=1)
            assert all(e.fallback is back.prior for e in back.experts if e.chars)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_content_decides_not_the_suffix():
    """A plain-JSON file that merely ends in `.gz` still loads - the rule
    `radixnet` settled on, and the one that stops a mislabelled file from
    looking like a corrupt one."""
    tmp = tempfile.mkdtemp()
    try:
        plain = write_json_atomic(os.path.join(tmp, "a.json.gz"), {"x": 1}, use_gzip=False)
        zipped = write_json_atomic(os.path.join(tmp, "b.json"), {"x": 2}, use_gzip=True)
        assert read_json(plain) == {"x": 1} and read_json(zipped) == {"x": 2}
        assert open(zipped, "rb").read(2) == b"\x1f\x8b"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoints_rotate_and_keep_the_latest():
    """`keep` is a promise about disk, and `latest` a promise about which one
    resume picks up.  The oldest go first - but never the one `latest` points
    at, which is the bug this rule exists to avoid."""
    data = corpus.synthetic(per_source=6, seed=0)
    train, _test = corpus.split(data, seed=0)
    bank = FilteredRadixBank(FilterRouter(bits=1, seed=0), depth=3,
                             alphabet=corpus.alphabet(data), seed=0)
    bank.fit(train, rounds=1, epochs=1)
    tmp = tempfile.mkdtemp()
    try:
        cm = CheckpointManager(tmp, keep=2)
        for step in range(4):
            cm.save(bank, step=step, metrics={"test_bits": 1.0 + step})
        names = [r["name"] for r in cm.list()]
        assert len(names) == 2 and names == sorted(names), names
        assert cm.latest()["step"] == 3
        assert cm.latest()["metrics"]["test_bits"] == 4.0
        # every way of naming one resolves to the same file
        newest = cm.latest()["name"]
        assert cm.resolve("latest") == cm.resolve(newest) == cm.resolve("3")
        assert cm.load().assign(train) == bank.assign(train)
        assert cm.delete(newest) and not cm.delete(newest)
        assert cm.latest()["step"] == 2, "latest must fall back when it is deleted"
        assert CheckpointManager(tmp, keep=2).list() == cm.list(), "the index is on disk"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_empty_checkpoint_directory_says_so():
    tmp = tempfile.mkdtemp()
    try:
        cm = CheckpointManager(tmp)
        assert cm.list() == [] and cm.latest() is None and cm.load_latest() is None
        for bad in ("latest", "nope", "7"):
            try:
                cm.resolve(bad)
            except FileNotFoundError:
                continue
            raise AssertionError(f"resolving {bad!r} in an empty directory must fail")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- the corpus ----------------------------------------------------------------

def test_the_split_is_stratified_and_disjoint():
    """An unstratified split can hand one register entirely to the test set,
    and then bits/char is a measurement of that register alone."""
    data = corpus.synthetic(per_source=25, seed=0)
    train, test = corpus.split(data, test_frac=0.2, seed=0)
    assert len(train) + len(test) == len(data)
    assert not ({id(s) for s in train} & {id(s) for s in test})
    for half in (train, test):
        seen = {s["source"] for s in half}
        assert seen == {"words", "digits", "dna", "brackets"}, seen


def test_the_corpus_is_reproducible():
    """Same call, same text, same digest - a result file names the corpus it
    was measured on."""
    a, b = corpus.synthetic(per_source=7, seed=3), corpus.synthetic(per_source=7, seed=3)
    assert corpus.digest(a) == corpus.digest(b)
    assert corpus.digest(a) != corpus.digest(corpus.synthetic(per_source=7, seed=4))
    segs = corpus.segments_of("x" * 700 + "\nshort\n" + "y" * 50)
    assert all(len(s) <= 256 for s in segs) and segs


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for f in fns:
        try:
            f()
            print(f"  ok  {f.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {f.__name__}: {exc}")
    print(f"\n  {len(fns) - failed}/{len(fns)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
