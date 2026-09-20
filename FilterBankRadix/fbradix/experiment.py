"""The arms, the measurements, and the sweeps that answer the design's questions.

Four questions, each with the control that settles it:

| question | arms |
|---|---|
| is a bank of experts better than one tree at all? | ``learned`` vs ``single`` |
| is *learning* the filter worth anything? | ``learned`` vs ``frozen`` |
| does the periodic unit beat a monotone one? | ``learned`` vs ``monotone`` |
| is the routing doing it, or just the split? | ``learned`` vs ``roundrobin`` |
| is the *learning rule* worth anything? | ``single`` vs ``counts`` (the ``baseline`` sweep) |
| does learning the *wave* help, or only the projection? | ``learned`` vs ``fixedwave`` |
| is the sign-bit address the thing holding layer 1 back? | ``learned`` vs ``argmax`` |
| how far short of a perfect router does it fall? | ``learned`` vs ``oracle`` |

and two sweeps, because a single corpus size and a single context depth would
answer none of them properly:

* ``scale`` - the same arms over 100 to 800 segments per source.  Splitting a
  corpus costs every expert data, so whether a bank can win at all is a
  question about how much text there is per expert.
* ``baseline`` - the same trees with the softmax replaced by relative counts,
  so the one-hop rule's contribution is measured rather than assumed.
* ``epochs`` - the one-hop rule against counting as training goes on, which is
  what says whether the rule is under-trained or over-trained.
* ``passes`` - how many hinge passes the refilter step should take, because the
  default was raised from 1 to 4 on an argument and an argument is not a
  measurement.
* ``probe`` - not an arm at all: four supervised measurements that say whether
  the gap between the learned router and the oracle is in the *signal* layer 2
  sends back, in the *address code* layer 1 speaks, or in the rule that fits it.
* ``depth`` - the same arms over context depths 2 to 6.  Routing hands the
  model a fact about the text (which register it is) that the context would
  otherwise have to carry, so the shallower the tree, the more a router should
  be worth.  That is the prediction; the sweep is where it is checked.

Every number reported here is on held-out segments, stratified by source, that
no arm trained on.
"""

from __future__ import annotations

import json
import math
import os
import time
import itertools
from collections import Counter, defaultdict

from . import corpus
from .bank import (  # noqa: F401  (BALANCE is re-exported for the CLI)
    BALANCE,
    ConstantRouter,
    CycleRouter,
    FilteredRadixBank,
    FilterRouter,
    OracleRouter,
)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

ARMS = ("single", "roundrobin", "oracle", "frozen", "learned", "argmax", "frozen-argmax",
        "monotone", "unbalanced", "fixedwave", "deep", "deep-oracle")

COUNT_ARMS = {"counts": "single", "counts-oracle": "oracle"}
"""Arms that keep the whole architecture and delete the learning.

Same trees, same compression, same backoff chain, same prior, same constants -
only the branch probabilities come from relative traversal counts instead of
the softmax of ``w_c * f_p * f_c``.  The difference between ``single`` and
``counts`` is what the one-hop rule is worth, with everything else held fixed;
these arms are run by the ``baseline`` sweep rather than the main one."""

DEEP = {"deep", "deep-oracle"}
"""Arms whose shared prior is as deep as the experts are.

The default prior is one character deep - a unigram floor that stops an empty
address from costing eleven bits.  A *deep* prior is a full tree over all the
training text, which turns every expert from a replacement for the single tree
into a correction to it: the expert answers where it has seen the context often
enough, and the shared model answers everywhere else.  It doubles the stored
structure, and the ``scale`` and ``depth`` sweeps are where that trade is
priced."""


def make_router(arm: str, bits: int, labels, seed: int):
    """One router per arm name."""
    arm = COUNT_ARMS.get(arm, arm)
    if arm == "single":
        return ConstantRouter()
    if arm == "roundrobin":
        return CycleRouter(1 << bits)
    if arm == "oracle":
        return OracleRouter(labels)
    if arm == "frozen":
        return FilterRouter(bits=bits, seed=seed, frozen=True)
    if arm == "argmax":
        # the same number of experts, one unit each instead of log2 of them
        return FilterRouter(bits=1 << bits, seed=seed, code="argmax")
    if arm == "frozen-argmax":
        return FilterRouter(bits=1 << bits, seed=seed, code="argmax", frozen=True)
    if arm == "learned":
        return FilterRouter(bits=bits, seed=seed)
    if arm == "monotone":
        return FilterRouter(bits=bits, seed=seed, kind="monotone")
    if arm == "unbalanced":
        return FilterRouter(bits=bits, seed=seed, balance=0.0)
    if arm == "fixedwave":
        return FilterRouter(bits=bits, seed=seed, act_rate=0.0)
    if arm == "deep":
        return FilterRouter(bits=bits, seed=seed)
    if arm == "deep-oracle":
        return OracleRouter(labels)
    raise ValueError(f"unknown arm {arm!r} (have {ARMS})")


def mean(xs) -> float:
    """Arithmetic mean; ``nan`` for an empty sequence."""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd_of(xs) -> float:
    """Population standard deviation; 0 for fewer than two values."""
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# -- agreement between a routing and the source labels -------------------------

def contingency(routes, labels) -> dict[int, Counter]:
    """``route -> Counter(source)``."""
    out: dict[int, Counter] = defaultdict(Counter)
    for r, l in zip(routes, labels):
        out[r][l] += 1
    return dict(out)


def purity(routes, labels) -> float:
    """Share of segments that sit with the majority source of their address.

    ``1.0`` is a perfect split, ``1/k`` for k equal sources is chance.  It is a
    generous measure - it rewards splitting one source across many addresses -
    so it is reported next to :func:`nmi`, which does not.
    """
    tab = contingency(routes, labels)
    total = sum(sum(c.values()) for c in tab.values())
    if not total:
        return 0.0
    return sum(max(c.values()) for c in tab.values()) / total


def nmi(routes, labels) -> float:
    """Normalised mutual information ``2 I(R;L) / (H(R) + H(L))``, in [0, 1].

    Zero when the addresses say nothing about the source, one when they
    determine it.  Unlike purity it penalises both kinds of disagreement, so a
    router that shatters one source across eight addresses does not score well.
    """
    n = len(routes)
    if not n:
        return 0.0
    cr, cl = Counter(routes), Counter(labels)
    joint = Counter(zip(routes, labels))
    hr = -sum(v / n * math.log(v / n, 2) for v in cr.values())
    hl = -sum(v / n * math.log(v / n, 2) for v in cl.values())
    inf = 0.0
    for (r, l), v in joint.items():
        p = v / n
        inf += p * math.log(p / ((cr[r] / n) * (cl[l] / n)), 2)
    return 0.0 if hr + hl == 0 else 2.0 * inf / (hr + hl)


# -- one run -------------------------------------------------------------------

def run_arm(
    arm: str,
    train: list[dict],
    test: list[dict],
    alphabet: int,
    bits: int = 2,
    depth: int = 5,
    rounds: int = 3,
    epochs: int = 3,
    lr: float = 0.5,
    seed: int = 0,
) -> dict:
    """Fit one arm and measure it on the held-out segments."""
    labels = [s["source"] for s in train]
    router = make_router(arm, bits, labels, seed)
    bank = FilteredRadixBank(router, depth=depth, alphabet=alphabet, seed=seed,
                             prior_depth=depth if arm in DEEP else 1,
                             scores="counts" if arm in COUNT_ARMS else "learned")
    t0 = time.time()
    history = bank.fit(train, rounds=rounds, epochs=epochs, lr=lr, test=test)
    elapsed = time.time() - t0
    test_routes = bank.assign(test)
    test_labels = [s["source"] for s in test]
    bits_test, chars = bank.bits_per_char(test, test_routes)
    rec = {
        "arm": arm,
        "seed": seed,
        "bits": bits,
        "depth": depth,
        "rounds": len(history),
        "test_bits_per_char": bits_test,
        "test_chars": chars,
        "prior_depth": bank.prior_depth,
        "train_bits_per_char": history[-1]["train_bits"],
        "purity": purity(test_routes, test_labels),
        "nmi": nmi(test_routes, test_labels),
        "seconds": elapsed,
        "history": history,
        **bank.size_stats(),
        **bank.load_stats(bank.routes),
        **{k: v for k, v in router.stats().items() if k != "labels"},
        "contingency": {
            str(r): dict(c) for r, c in sorted(contingency(test_routes, test_labels).items())
        },
    }
    return rec


def make_corpus(per_source: int, seed: int = 0):
    """Build (or load the snapshot of) the corpus and split it."""
    data = corpus.take(per_source)
    train, test = corpus.split(data, seed=seed)
    return data, train, test, corpus.alphabet(data)


# -- sweeps --------------------------------------------------------------------

def sweep_main(per_source=200, bits=2, depth=5, seeds=(0, 1, 2), rounds=3, epochs=3, log=print) -> dict:
    """Every arm, every seed, one corpus and one depth: the headline table."""
    data, train, test, alpha = make_corpus(per_source)
    log(f"# main: {len(train)} train / {len(test)} test segments, alphabet {alpha}, "
        f"{sum(len(s['text']) for s in data)} chars, {1 << bits} addresses, depth {depth}")
    runs = []
    for arm in ARMS:
        for seed in seeds:
            r = run_arm(arm, train, test, alpha, bits=bits, depth=depth,
                        rounds=rounds, epochs=epochs, seed=seed)
            runs.append(r)
            log(f"  {arm:11s} seed={seed} bits/char={r['test_bits_per_char']:.4f} "
                f"nodes={r['nodes']:6d} live={r['live_experts']} "
                f"purity={r['purity']:.3f} nmi={r['nmi']:.3f} {r['seconds']:.0f}s")
    return {"sweep": "main", "corpus": corpus.summarise(data), "bits": bits,
            "depth": depth, "runs": runs}


def sweep_depth(per_source=200, bits=2, depths=(2, 3, 4, 5, 6),
                arms=("single", "oracle", "learned", "argmax", "frozen"),
                seed=0, rounds=3, epochs=3, log=print) -> dict:
    """Does routing matter more when the tree has less context of its own?"""
    data, train, test, alpha = make_corpus(per_source)
    log(f"# depth sweep: depths {depths}, arms {arms}")
    runs = []
    for depth in depths:
        for arm in arms:
            r = run_arm(arm, train, test, alpha, bits=bits, depth=depth,
                        rounds=rounds, epochs=epochs, seed=seed)
            runs.append(r)
            log(f"  depth={depth} {arm:9s} bits/char={r['test_bits_per_char']:.4f} "
                f"nodes={r['nodes']:6d} nmi={r['nmi']:.3f} {r['seconds']:.0f}s")
    return {"sweep": "depth", "corpus": corpus.summarise(data), "bits": bits, "runs": runs}


def sweep_scale(sizes=(100, 200, 400, 800), bits=2, depth=5,
                arms=("single", "oracle", "learned", "argmax"), seed=0,
                rounds=3, epochs=3, log=print) -> dict:
    """Splitting costs every expert data - does the cost go away with more text?"""
    log(f"# scale sweep: {sizes} segments per source, arms {arms}")
    runs = []
    for size in sizes:
        data, train, test, alpha = make_corpus(size)
        chars = sum(len(s["text"]) for s in data)
        for arm in arms:
            r = run_arm(arm, train, test, alpha, bits=bits, depth=depth,
                        rounds=rounds, epochs=epochs, seed=seed)
            r["per_source"] = size
            r["corpus_chars"] = chars
            runs.append(r)
            log(f"  n={size:4d} ({chars:6d} chars) {arm:9s} bits/char={r['test_bits_per_char']:.4f} "
                f"nodes={r['nodes']:6d} nmi={r['nmi']:.3f} {r['seconds']:.0f}s")
    return {"sweep": "scale", "bits": bits, "depth": depth, "runs": runs}


def sweep_epochs(per_source=200, bits=2, depth=5, seeds=(0, 1, 2), rounds=1,
                 values=(1, 3, 10, 30), log=print) -> dict:
    """Is the one-hop rule under-trained, or over-trained, against counting?

    The ``baseline`` sweep says a tree whose branches are relative traversal
    counts beats the same tree trained by the one-hop rule.  That has two very
    different explanations - three epochs is not enough to reach what counting
    already knows, or training past it makes the branches sharper than the data
    supports - and they point in opposite directions.  This is the sweep that
    tells them apart: one tree, no routing, the same seeds, more and more
    epochs, against the untrained count line.
    """
    data, train, test, alpha = make_corpus(per_source)
    log(f"# epochs: the one-hop rule against counting, depth {depth}")
    runs = []
    for arm, values_for in (("counts", (0,)), ("single", values)):
        for ep in values_for:
            for seed in seeds:
                r = run_arm(arm, train, test, alpha, bits=bits, depth=depth,
                            rounds=rounds, epochs=max(1, ep), seed=seed)
                r["epochs"] = ep
                r["arm"] = arm
                runs.append(r)
                log(f"  {arm:7s} epochs={ep:2d} seed={seed} "
                    f"bits/char={r['test_bits_per_char']:.4f} "
                    f"train={r['train_bits_per_char']:.4f} {r['seconds']:.0f}s")
    return {"sweep": "epochs", "corpus": corpus.summarise(data), "bits": bits,
            "depth": depth, "runs": runs}


def sweep_passes(per_source=200, bits=2, depth=5, seeds=(0, 1, 2), rounds=3, epochs=3,
                 values=(1, 2, 4, 8), log=print) -> dict:
    """How many hinge passes the refilter step should take.

    The pricing half of :meth:`FilterRouter.learn` is expensive and the hinge
    half is nearly free, so running the hinge more than once over the same
    prices costs almost nothing - which is an argument for more passes, not a
    measurement.  This is the measurement.  The default (``PASSES``) was raised
    from 1 to 4 on the argument; the table says whether the argument was worth
    anything.
    """
    data, train, test, alpha = make_corpus(per_source)
    log(f"# passes: hinge passes per refilter step, depth {depth}")
    runs = []
    for n in values:
        for seed in seeds:
            router = FilterRouter(bits=bits, seed=seed, passes=n)
            bank = FilteredRadixBank(router, depth=depth, alphabet=alpha, seed=seed)
            t0 = time.time()
            history = bank.fit(train, rounds=rounds, epochs=epochs, test=test)
            routes = bank.assign(test)
            labels = [s["source"] for s in test]
            r = {
                "arm": f"passes={n}", "passes": n, "seed": seed, "depth": depth,
                "test_bits_per_char": bank.bits_per_char(test, routes)[0],
                "purity": purity(routes, labels), "nmi": nmi(routes, labels),
                "seconds": time.time() - t0, "history": history,
                **bank.size_stats(), **bank.load_stats(bank.routes),
            }
            runs.append(r)
            log(f"  passes={n:2d} seed={seed} bits/char={r['test_bits_per_char']:.4f} "
                f"live={r['live_experts']} nmi={r['nmi']:.3f} {r['seconds']:.0f}s")
    return {"sweep": "passes", "corpus": corpus.summarise(data), "bits": bits,
            "depth": depth, "runs": runs}


def sweep_baseline(per_source=200, bits=2, depth=5, seeds=(0, 1, 2), rounds=3, epochs=3,
                   log=print) -> dict:
    """What is the one-hop rule worth?  The same trees, with nothing learned.

    ``counts`` and ``counts-oracle`` keep every other part of the architecture
    and replace the softmax of ``w_c * f_p * f_c`` with the relative traversal
    counts.  Whatever is left between them and ``single`` / ``oracle`` is the
    learning rule's contribution, measured rather than assumed.
    """
    data, train, test, alpha = make_corpus(per_source)
    log(f"# baseline: what the one-hop rule is worth, depth {depth}")
    runs = []
    for arm in ("single", "counts", "oracle", "counts-oracle"):
        for seed in seeds:
            r = run_arm(arm, train, test, alpha, bits=bits, depth=depth,
                        rounds=rounds, epochs=epochs, seed=seed)
            runs.append(r)
            log(f"  {arm:14s} seed={seed} bits/char={r['test_bits_per_char']:.4f} "
                f"nodes={r['nodes']:6d} {r['seconds']:.0f}s")
    return {"sweep": "baseline", "corpus": corpus.summarise(data), "bits": bits,
            "depth": depth, "runs": runs}


# -- the probe: is the signal the problem, or the filter? ----------------------

SUP_EPOCHS = 100
"""Epochs for every supervised fit in :func:`probe`.

Not a free parameter to tune - a floor.  At 25 the numbers are not a ceiling
but a snapshot of an unfinished fit, and they are wrong in a way that looks
like a result: one dichotomy read 0.500, exactly chance, and the reason was
that its unit's ``|a*b|`` had collapsed to 0.0100 - the unit had *died* and was
answering a constant.  Four times the epochs and the same split reads 0.777.
A measurement of what a filter can represent has to be taken after the fit has
actually converged, and unit death has to be reported next to it rather than
quietly lowering it.
"""


def wave_reach(filt, xs) -> dict:
    """How far into its own wave each unit has been pushed.

    ``u = b(z - h)`` is the wave's argument, so ``|u| > pi/2`` means the unit
    is past its first peak - the place where a *periodic* response starts
    coming back down and a monotone one merely saturates.  That difference is
    the whole reason the two kinds of unit behave differently as gates, so it
    is measured rather than reasoned about.
    """
    us = []
    for x in xs:
        for j in range(filt.bits):
            us.append(abs(filt.b[j] * (filt.project(x, j) - filt.h[j])))
    us.sort()
    return {
        "median": us[len(us) // 2],
        "max": us[-1],
        "past_first_peak": sum(1 for u in us if u > math.pi / 2) / len(us),
    }


def _fit_supervised(filt, xs, targets, epochs: int = SUP_EPOCHS, lr: float = 0.3):
    """Train a filter *directly on the answers* - the ceiling, not an arm.

    Nothing in the architecture is allowed to do this.  It exists to separate
    two very different failures: a filter that cannot represent the partition,
    and a filter that can but is not being taught it.
    """
    for _ in range(epochs):
        for x, want in zip(xs, targets):
            route, _resp = filt.address(x)
            if route == want:
                continue
            if filt.code == "argmax":
                filt.push_pair(x, want, route, lr)
            else:
                for j in range(filt.bits):
                    w = (want >> j) & 1
                    if ((route >> j) & 1) != w:
                        filt.push(x, j, 1.0 if w else -1.0, lr)
    return [filt.route(x) for x in xs]


def probe(per_source=200, bits=2, depth=5, epochs=3, seeds=(0, 1, 2), log=print) -> dict:
    """Where the routing gap actually is, in four measurements.

    1. **Is the signal right?**  Give the bank perfect experts and ask which
       one prices each segment cheapest.  That is exactly the target the
       refilter step chases, so if it does not agree with the sources, nothing
       downstream can.
    2. **Can the address code express the answer?**  Fit a filter *supervised*
       on the labels, under both codes - and for the sign code, under every one
       of the 24 ways to assign four sources to four addresses, since the
       unsupervised filter is free to pick any of them.
    3. **Which questions are answerable at all?**  The three ways to split the
       four registers two-against-two, one unit each.  A sign-bit address needs
       two of the three, and their product bounds what it can reach.  Read them
       next to ``amp_freq``: a unit that died mid-fit reads as an unanswerable
       question, and it is not one.
    4. **And one source against the rest?**  What an argmax unit is asked for.

    Every supervised fit also reports :func:`wave_reach` - how far into its own
    wave the hinge pushed each unit.  That is what separates the two shapes: a
    hinge can only raise a response by moving the projection, and past the
    first peak a *periodic* unit starts coming back down while a monotone one
    only flattens.

    Everything but (1) is averaged over ``seeds``, because a single supervised
    fit of this filter is not a stable number - which is itself one of the
    findings.
    """
    from .filter import ActivationFilter, features

    data, train, test, alpha = make_corpus(per_source)
    labels = [s["source"] for s in train]
    names = sorted(set(labels))
    xs = [features(s["text"]) for s in train]
    out: dict = {"corpus": corpus.summarise(data), "bits": bits, "depth": depth,
                 "seeds": list(seeds), "sup_epochs": SUP_EPOCHS}

    # 1 -- the target layer 2 hands back
    bank = FilteredRadixBank(OracleRouter(labels), depth=depth, alphabet=alpha, seed=0)
    bank.fit(train, rounds=1, epochs=epochs)
    cheapest = []
    for s in train:
        costs = [e.bits_per_char([s["text"]])[0] if e.chars else float("inf") for e in bank.experts]
        cheapest.append(min(range(len(costs)), key=lambda j: costs[j]))
    out["signal"] = {
        "nmi": nmi(cheapest, labels),
        "purity": purity(cheapest, labels),
        "contingency": {str(r): dict(c) for r, c in sorted(contingency(cheapest, labels).items())},
    }
    log(f"  1. the target layer 2 hands back: nmi={out['signal']['nmi']:.3f} "
        f"purity={out['signal']['purity']:.3f}")

    def fit(make, targets):
        """Supervised fit over every seed; returns means and the spread."""
        acc, pur, nm, dead = [], [], [], 0
        reach0, reach1, peak1, freq = [], [], [], []
        for sd in seeds:
            f = make(sd)
            f.calibrate(xs)
            reach0.append(wave_reach(f, xs)["median"])
            got = _fit_supervised(f, xs, targets)
            acc.append(sum(1 for g, t in zip(got, targets) if g == t) / len(targets))
            pur.append(purity(got, labels))
            nm.append(nmi(got, labels))
            dead += len(f.dead_units())
            after = wave_reach(f, xs)
            reach1.append(after["median"])
            peak1.append(after["past_first_peak"])
            freq.append(mean(f.b))
        return {"accuracy": mean(acc), "accuracy_sd": sd_of(acc), "purity": mean(pur),
                "nmi": mean(nm), "dead_units": dead, "per_seed": acc,
                "wave_u_before": mean(reach0), "wave_u_after": mean(reach1),
                "past_first_peak": mean(peak1), "mean_b": mean(freq)}

    # 2 -- what each address code can express, when handed the answer
    out["ceiling"] = {}
    best = None
    for perm in itertools.permutations(range(1 << bits)):
        if len(names) > len(perm):
            break
        lab = dict(zip(names, perm))
        r = fit(lambda sd: ActivationFilter(bits=bits, seed=sd), [lab[l] for l in labels])
        if best is None or (r["purity"], r["nmi"]) > (best["purity"], best["nmi"]):
            best = dict(r, assignment=list(perm))
    out["ceiling"]["sign"] = best
    log(f"  2. sign code, supervised, best of {math.factorial(1 << bits)} assignments: "
        f"purity={best['purity']:.3f} nmi={best['nmi']:.3f} dead={best['dead_units']}")

    lab = {l: i for i, l in enumerate(names)}
    for kind in ("sine", "monotone"):
        for rate, tag in ((None, ""), (0.0, "-fixedwave")):
            def make(sd, kind=kind, rate=rate):
                kw = {"bits": 1 << bits, "seed": sd, "kind": kind, "code": "argmax"}
                if rate is not None:
                    kw["act_rate"] = rate
                return ActivationFilter(**kw)
            r = fit(make, [lab[l] for l in labels])
            out["ceiling"][f"argmax_{kind}{tag}"] = r
            log(f"     argmax code, supervised, {kind + tag:19s}: "
                f"accuracy={r['accuracy']:.3f} +- {r['accuracy_sd']:.3f} "
                f"nmi={r['nmi']:.3f} | median |u| {r['wave_u_before']:.2f} -> "
                f"{r['wave_u_after']:.2f}, {r['past_first_peak']:.1%} past the first "
                f"peak, mean b {r['mean_b']:.3f}")

    # 3 -- the two-against-two questions a sign-bit address is made of
    out["dichotomies"] = {}
    for i in range(1, len(names)):
        group = {names[0], names[i]}
        tg = [1 if l in group else 0 for l in labels]
        row = {}
        for kind in ("sine", "monotone"):
            row[kind] = fit(lambda sd, kind=kind: ActivationFilter(bits=1, seed=sd, kind=kind),
                            tg)["accuracy"]
        key = " + ".join(sorted(group))
        out["dichotomies"][key] = row
        log(f"  3. {key:18s} vs the rest: sine={row['sine']:.3f} monotone={row['monotone']:.3f}")

    # 4 -- the one-against-the-rest question an argmax unit is made of
    out["one_vs_rest"] = {}
    for name in names:
        tg = [1 if l == name else 0 for l in labels]
        r = fit(lambda sd: ActivationFilter(bits=1, seed=sd), tg)
        out["one_vs_rest"][name] = r["accuracy"]
        log(f"  4. {name:18s} vs the rest: {r['accuracy']:.3f}")
    return out


# -- driver --------------------------------------------------------------------

def summarise_main(result: dict) -> list[dict]:
    """Mean and spread per arm across seeds, for the README table."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in result["runs"]:
        by_arm[r["arm"]].append(r)
    out = []
    for arm in ARMS:
        rs = by_arm.get(arm)
        if not rs:
            continue
        vals = [r["test_bits_per_char"] for r in rs]
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        out.append({
            "arm": arm,
            "bits_per_char": mean,
            "sd": sd,
            "nodes": sum(r["nodes"] for r in rs) / len(rs),
            "live": sum(r["live_experts"] for r in rs) / len(rs),
            "purity": sum(r["purity"] for r in rs) / len(rs),
            "nmi": sum(r["nmi"] for r in rs) / len(rs),
            "seconds": sum(r["seconds"] for r in rs) / len(rs),
        })
    return out


def main(argv=None) -> int:
    """``python3 -m fbradix.cli experiment`` lands here."""
    import argparse

    ap = argparse.ArgumentParser(prog="fbradix experiment", description=__doc__)
    ap.add_argument("--sweep", default="all",
                    choices=("all", "main", "depth", "scale", "probe", "baseline",
                             "passes", "epochs"))
    ap.add_argument("--quick", action="store_true", help="one seed, small corpus, shallow")
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args(argv)

    seeds = tuple(range(args.seeds))
    depths = (2, 3, 4, 5, 6)
    sizes = (100, 200, 400, 800)
    epoch_values = (1, 3, 10, 30)
    pass_values = (1, 2, 4, 8)
    if args.quick:
        seeds = (0,)
        args.per_source = 60
        args.rounds = 2
        depths = (2, 4)
        sizes = (60, 120)
        epoch_values = (1, 3)
        pass_values = (1, 4)

    os.makedirs(args.out, exist_ok=True)
    lines: list[str] = []

    def log(msg):
        text = msg if isinstance(msg, str) else json.dumps(msg)
        print(text, flush=True)
        lines.append(text)

    started = time.time()
    out: dict[str, dict] = {}
    if args.sweep in ("all", "main"):
        out["main"] = sweep_main(args.per_source, args.bits, args.depth, seeds,
                                 args.rounds, args.epochs, log)
        log("")
        log("  arm          bits/char      nodes  live  purity    nmi")
        for row in summarise_main(out["main"]):
            log(f"  {row['arm']:11s} {row['bits_per_char']:.4f} +- {row['sd']:.4f} "
                f"{row['nodes']:8.0f} {row['live']:5.1f} {row['purity']:6.3f} {row['nmi']:6.3f}")
        log("")
    if args.sweep in ("all", "depth"):
        out["depth"] = sweep_depth(args.per_source, args.bits, depths, seed=0,
                                   rounds=args.rounds, epochs=args.epochs, log=log)
        log("")
    if args.sweep in ("all", "epochs"):
        out["epochs"] = sweep_epochs(args.per_source, args.bits, args.depth, seeds,
                                     values=epoch_values, log=log)
        log("")
    if args.sweep in ("all", "passes"):
        out["passes"] = sweep_passes(args.per_source, args.bits, args.depth, seeds,
                                     args.rounds, args.epochs, values=pass_values, log=log)
        log("")
    if args.sweep in ("all", "baseline"):
        out["baseline"] = sweep_baseline(args.per_source, args.bits, args.depth, seeds,
                                         args.rounds, args.epochs, log)
        log("")
    if args.sweep in ("all", "probe"):
        log("# probe: where the routing gap is")
        out["probe"] = probe(args.per_source, args.bits, args.depth, args.epochs, log=log)
        log("")
    if args.sweep in ("all", "scale"):
        out["scale"] = sweep_scale(sizes, args.bits, args.depth, seed=0,
                                   rounds=args.rounds, epochs=args.epochs, log=log)
        log("")
    out["meta"] = {
        "seconds": time.time() - started,
        "quick": args.quick,
        "per_source": args.per_source,
        "bits": args.bits,
        "depth": args.depth,
        "rounds": args.rounds,
        "epochs": args.epochs,
        "seeds": list(seeds),
    }
    # One file per sweep, whether it was run alone or as part of --sweep all,
    # so `summarize.py` finds the same names either way.
    tag = "quick" if args.quick else args.sweep
    written = []
    if args.quick:
        written.append(_write(args.out, "quick", out))
    else:
        for name in ("main", "baseline", "epochs", "passes", "depth", "scale", "probe"):
            if name in out:
                written.append(_write(args.out, name, {name: out[name], "meta": out["meta"]}))
    lpath = os.path.join(args.out, f"{tag}_run.log")
    with open(lpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {', '.join(written)} and {lpath} in {out['meta']['seconds']:.0f}s")
    return 0


def _write(out_dir: str, name: str, payload: dict) -> str:
    """Write one sweep's JSON and return its path."""
    path = os.path.join(out_dir, f"{name}_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    return path
