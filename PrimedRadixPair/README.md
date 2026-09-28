# PrimedRadixPair

**Two radix trees, primed with every option, connected at every equal node —
one written by reading, the other by what worked.**

A radix tree that is **primed** — every sequence of `1..L` units over a closed
vocabulary is inserted before the tree sees a single text — never changes its
structure again: training is a counter going up at an address that is
*computed*, not found. A second primed tree over the same sequences is the
**reward tree**: it is written only by the outcomes of what the model produced
— a step rewarded when the output was judged correct, punished when it was
not — and never by a text merely read. The two are connected wherever they hold
an equal sequence, at every level, and the connection is where a step's two
numbers — how often it was read, what it earned — are read together into one
answer. Text goes in and comes out through an **encoder/decoder** behind a
tokenizer of your choosing: characters, bytes, the byte-pair encoding LLMs
use, or the repository's phonetic tokenizer.

## Contents

| file | what it is |
|---|---|
| `PRD.md` | the requirement in the author's own words, the hypothesis, goals and non-goals, the requirements, the success criteria, the phases, and the decisions still to be made |
| `DESIGN.md` | the specification — the contract every module is implemented against, in 19 sections |

**There is no code in this directory yet.** The two documents are the whole of
it. Read `PRD.md` first — it is short and it names the decisions the design
rests on — then `DESIGN.md` before writing anything here.

## The three things worth knowing

**Priming is arithmetic.** A complete `R`-ary tree needs no pointers: the path
to a node *is* its address, read as a number in base `R` (the radix). Inserting
every option by brute force is therefore free in structure and costs only the
numbers written on the nodes, and the literal brute-force insertion is kept as
the oracle the arithmetic is tested against (`DESIGN.md` §6, §15).

**Two trees, two signals, met at every level.** The count tree holds what the
corpus did; the reward tree holds what the judges said. Kept apart, each can be
walked alone — the punishment traversal finds the way the least has gone wrong
on, the family's answer to a step that is popular and known to be a mistake —
and read together at the rung they give the family's dual function: the share
of what was read, times `e` to the reward (`DESIGN.md` §9.2). A reward is
written at every context length of the step it judges, so it generalises at the
shallow levels and specialises at the deep; connected only at the final nodes
it would do neither, and that difference is measured, not assumed (§9.6).

**The tokenizer decides the depth.** A primed tree is `R^L` big, and the codec
sets `R`: letters and sounds prime to three or four units of context, a
byte-pair vocabulary of about a thousand tokens to two, an LLM's own tokenizer
to one. The budget is a table, not a surprise (`DESIGN.md` §16).

## Planned shape

Python package `radixpair`, Python 3.11+, **standard library only**; `tiktoken`
and `tokenizers` are optional extras for reading with a published LLM tokenizer,
imported lazily and never required. The familiar four verbs — `train / predict
/ generate / score` — and the family's feedback primitives — `reward / punish /
two_nrl / feedback` — on the same CLI shape as `RadixCyclicNN/` and `GTMNN/`.
The `count` kind first (deterministic, no learning rate, the measurements are
made on it), the `sine` kind second.

## Related

* `RadixCyclicNN/radixnet/countnet.py` and `DECISIONS.md` D-022, D-026, D-050 — the count / reward model whose two halves this pair pulls apart into two trees, and the feedback primitives it keeps.
* `RadixCyclicNN/SPEC-LeastPunished.md` and `RadixCyclicNN/DESIGN.md` §31 — why rewards and punishments are kept in separate ledgers, and the two traversals that read them.
* `FilterBankRadix/DESIGN.md` §5 — the context tree whose every-context-length prediction the fold reproduces, and the smoothing constants shared so bits per unit are comparable.
* `RadixCyclicNN/DESIGN.md` §4, §7 — the encoder / decoder halves this codec is modelled on, and the shortest-path search reused here.
* `RadixTrieLLM_RNN/main.py` — the smallest radix insertion in the repository; its splitting logic is what a primed tree never needs.
* `PhoneticTokenizer/` — the tokenizer of sounds this reads with, at the phoneme level.
