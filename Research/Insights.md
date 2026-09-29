# Insights — index

A record of the ideas driving this repository, each with where it is specified,
whether it has been tested, and what the evidence says. Kept as an index rather
than an argument: the arguments live in the design documents, the numbers in
each project's findings, results and decision records, named on every entry's
*Where* line.

Status is one of **held** (stated, not yet tested), **confirmed** (tested, holds),
**refined** (tested, holds with a correction), or **contradicted** (tested, does
not hold as stated).

An entry keeps its number; other documents cite entries by it
(`FilterBankRadix/README.md` cites 24, the GREN branch cites 30, 32 and 36).
Where the code or the measurement behind an entry has not reached `main`, its
*Where* line names the branch that holds it: `feat/gren-impl` (PR #25) for most
of the game stack, 30–49; `claude/keen-feynman-ntoqni` (PR #24) for the
re-measured NeuralCompression findings and TwoSine; and
`feat/twonrl-chess-longrun`, `feat/primed-radix-pair`,
`claude/busy-hopper-2ji25l`, `claude/phonetic-tokenizer-package-b6yfai` and
`feat/multi-gradient-nn` for the rest.

## At a glance

142 entries: 58 confirmed, 33 refined, 41 held, 10 contradicted. Each links to
its entry; the *Where* line under every entry names the evidence.

| # | Insight | Status |
|---|---|---|
| | **[The architecture](#the-architecture)** | |
| 1 | [AGI is a game-solving problem](#1-agi-is-a-game-solving-problem) | held |
| 2 | [Game theory, not one big network — GTMNN](#2-game-theory-not-one-big-network--gtmnn) | refined |
| 3 | [Credit belongs to game theory, not the chain rule](#3-credit-belongs-to-game-theory-not-the-chain-rule) | confirmed |
| 4 | [Everything can be gamified](#4-everything-can-be-gamified) | refined |
| | **[Learning the rules](#learning-the-rules)** | |
| 5 | [You learn the rules by failing on purpose](#5-you-learn-the-rules-by-failing-on-purpose) | confirmed |
| 6 | [The failure rate should be about 50%](#6-the-failure-rate-should-be-about-50) | refined |
| 7 | [Programming is the game worth optimising for](#7-programming-is-the-game-worth-optimising-for) | confirmed |
| | **[Finding which game you are in](#finding-which-game-you-are-in)** | |
| 8 | [Identify the game by finding similar games](#8-identify-the-game-by-finding-similar-games) | confirmed |
| 9 | [Information clustering — chess and checkers are closer than chess and a difficult conversation](#9-information-clustering--chess-and-checkers-are-closer-than-chess-and-a-difficult-conversation) | confirmed |
| 10 | [Use what we know about the game: inputs, outputs, category, goal](#10-use-what-we-know-about-the-game-inputs-outputs-category-goal) | held |
| 11 | [Break each game into its smallest possible parts first](#11-break-each-game-into-its-smallest-possible-parts-first) | confirmed |
| 12 | [An n-dimensional trie, with the game encoded into each path](#12-an-n-dimensional-trie-with-the-game-encoded-into-each-path) | held |
| 13 | [A radix tree with the mechanics of a trie](#13-a-radix-tree-with-the-mechanics-of-a-trie) | held |
| 14 | [Or invert the problem — a random forest](#14-or-invert-the-problem--a-random-forest) | contradicted |
| | **[Compression](#compression)** | |
| 15 | [Neural compression: split the output range, plus a selector](#15-neural-compression-split-the-output-range-plus-a-selector) | refined |
| 16 | [Order the partition by similarity](#16-order-the-partition-by-similarity) | confirmed |
| 17 | [Avoid dust — binary doesn't work because of 16 and 32](#17-avoid-dust--binary-doesnt-work-because-of-16-and-32) | confirmed |
| 18 | [Same input and output size for every game, plus a game modifier](#18-same-input-and-output-size-for-every-game-plus-a-game-modifier) | refined |
| 19 | [Train validity first, then grade the move separately](#19-train-validity-first-then-grade-the-move-separately) | refined |
| | **[The vanishing gradient](#the-vanishing-gradient)** | |
| 20 | [The vanishing gradient is a feature, not a bug](#20-the-vanishing-gradient-is-a-feature-not-a-bug) | contradicted |
| 21 | [Invert the gradient when the vanishing threshold is reached](#21-invert-the-gradient-when-the-vanishing-threshold-is-reached) | contradicted |
| 22 | [Dual network: train in one, query the other — the REM analogy](#22-dual-network-train-in-one-query-the-other--the-rem-analogy) | confirmed |
| 23 | [Rotating inversion — invert alternating layers each cycle](#23-rotating-inversion--invert-alternating-layers-each-cycle) | refined |
| | **[Activation](#activation)** | |
| 24 | [The sine wave activation instead of a sigmoid](#24-the-sine-wave-activation-instead-of-a-sigmoid) | refined |
| 25 | [Cycles are a feature, not a bug](#25-cycles-are-a-feature-not-a-bug) | held |
| | **[Cortical organisation](#cortical-organisation)** | |
| 26 | [A Single Self-Building Neural Network instead of micro networks](#26-a-single-self-building-neural-network-instead-of-micro-networks) | refined |
| 27 | [Similar games in similar regions — brain structure](#27-similar-games-in-similar-regions--brain-structure) | refined |
| 28 | [A cyclic graph where distance is similarity](#28-a-cyclic-graph-where-distance-is-similarity) | confirmed |
| 29 | [Common denominators — generalised, minified inputs](#29-common-denominators--generalised-minified-inputs) | refined |
| | **[The game stack, integrated and measured — GREN, GTMNN, CyclicCortex, Memory](#the-game-stack-integrated-and-measured--gren-gtmnn-cycliccortex-memory)** | |
| 30 | [A shared refusal code is not a shared rule](#30-a-shared-refusal-code-is-not-a-shared-rule) | contradicted |
| 31 | [Identification and characterisation are different questions](#31-identification-and-characterisation-are-different-questions) | refined |
| 32 | [A game modifier is a game-IDENTITY signal, and identity is the opposite of transfer](#32-a-game-modifier-is-a-game-identity-signal-and-identity-is-the-opposite-of-transfer) | refined |
| 33 | [A training signal that contains the answer is not a loss](#33-a-training-signal-that-contains-the-answer-is-not-a-loss) | confirmed |
| 34 | [A trie over what is PROVABLE, not over what a thing is](#34-a-trie-over-what-is-provable-not-over-what-a-thing-is) | confirmed |
| 35 | [Measure the mixed profile, not its argmax](#35-measure-the-mixed-profile-not-its-argmax) | confirmed |
| 36 | [Memory is the trajectory, not the answer](#36-memory-is-the-trajectory-not-the-answer) | held |
| 37 | [Index a game by its END GOAL, and work backwards](#37-index-a-game-by-its-end-goal-and-work-backwards) | confirmed |
| 38 | [A refusal code is a precondition violation](#38-a-refusal-code-is-a-precondition-violation) | confirmed |
| 39 | [Check whether the baseline already does it](#39-check-whether-the-baseline-already-does-it) | confirmed |
| 40 | [A rule transfers exactly when the feature that carries it means the same thing](#40-a-rule-transfers-exactly-when-the-feature-that-carries-it-means-the-same-thing) | confirmed |
| 41 | [Map a new game blind: refusals name the rules, acceptances are the scarce resource](#41-map-a-new-game-blind-refusals-name-the-rules-acceptances-are-the-scarce-resource) | confirmed |
| 42 | [Growth is an identity — except for outputs, and at initialisation](#42-growth-is-an-identity--except-for-outputs-and-at-initialisation) | confirmed |
| 43 | [The vocabulary sets the ceiling — for legality, for play, and for a teacher](#43-the-vocabulary-sets-the-ceiling--for-legality-for-play-and-for-a-teacher) | confirmed |
| 44 | [Legal and good are different questions — put legality on its own wire](#44-legal-and-good-are-different-questions--put-legality-on-its-own-wire) | confirmed |
| 45 | [Outcome credit must be scaled to the game's length](#45-outcome-credit-must-be-scaled-to-the-games-length) | confirmed |
| 46 | [Sample where the rule lives — a training distribution teaches its own prior](#46-sample-where-the-rule-lives--a-training-distribution-teaches-its-own-prior) | confirmed |
| 47 | [With regions as players, Shapley is exact — and it says the graph routes rather than teaches](#47-with-regions-as-players-shapley-is-exact--and-it-says-the-graph-routes-rather-than-teaches) | confirmed |
| 48 | [The theorems hold in code — and the cycle Milchtaich warns of never came](#48-the-theorems-hold-in-code--and-the-cycle-milchtaich-warns-of-never-came) | confirmed |
| 49 | [Three jobs, three measures — rarity is for probing, not for similarity](#49-three-jobs-three-measures--rarity-is-for-probing-not-for-similarity) | confirmed |
| | **[The papers, measured](#the-papers-measured)** | |
| 50 | [What the field routes around is carrying information](#50-what-the-field-routes-around-is-carrying-information) | held |
| 51 | [The vanishing gradient is a reading — count the multiplications and divide the decay back out](#51-the-vanishing-gradient-is-a-reading--count-the-multiplications-and-divide-the-decay-back-out) | refined |
| 52 | [A cycle is repetition stored once — and, read as statistics, a merge of every context a gram occurs in](#52-a-cycle-is-repetition-stored-once--and-read-as-statistics-a-merge-of-every-context-a-gram-occurs-in) | refined |
| 53 | [You cannot leave a loop from inside it — and taking the cycle out of the structure moves it into the process](#53-you-cannot-leave-a-loop-from-inside-it--and-taking-the-cycle-out-of-the-structure-moves-it-into-the-process) | confirmed |
| 54 | [Inverting a path is two-colouring it — exact on a tree, best-effort with cycles, impossible on a self-loop](#54-inverting-a-path-is-two-colouring-it--exact-on-a-tree-best-effort-with-cycles-impossible-on-a-self-loop) | confirmed |
| 55 | [A sigmoid is one step of a sine wave rotated 45 degrees](#55-a-sigmoid-is-one-step-of-a-sine-wave-rotated-45-degrees) | confirmed |
| 56 | [The sine unit never dies along its input — it can die along `a` and `b`](#56-the-sine-unit-never-dies-along-its-input--it-can-die-along-a-and-b) | refined |
| 57 | [Liveness is not accuracy — the sine's case rests on one unseeded CartPole run, and under control it never fits best](#57-liveness-is-not-accuracy--the-sines-case-rests-on-one-unseeded-cartpole-run-and-under-control-it-never-fits-best) | refined |
| 58 | [Read Adam's two moments as two interfering waves — TwoSine](#58-read-adams-two-moments-as-two-interfering-waves--twosine) | refined |
| 59 | [Keep the game coordinate, drop the output partition](#59-keep-the-game-coordinate-drop-the-output-partition) | held |
| | **[2NRL — learning by inverting consistent failure](#2nrl--learning-by-inverting-consistent-failure)** | |
| 60 | [2NRL — train on the failure at full rate, invert, fine-tune gently](#60-2nrl--train-on-the-failure-at-full-rate-invert-fine-tune-gently) | contradicted |
| 61 | [Two negatives make a positive — one sign flip turns a trained failure into its exact opposite](#61-two-negatives-make-a-positive--one-sign-flip-turns-a-trained-failure-into-its-exact-opposite) | confirmed |
| 62 | [The failure has to be consistent — what inversion recovers is bounded by the failure's entropy](#62-the-failure-has-to-be-consistent--what-inversion-recovers-is-bounded-by-the-failures-entropy) | contradicted |
| 63 | [Negating a unit means negating its offset — a test of an involution cannot see a wrong inverse](#63-negating-a-unit-means-negating-its-offset--a-test-of-an-involution-cannot-see-a-wrong-inverse) | refined |
| 64 | [The NOT of a weight is its complement, `W → 1 − W`](#64-the-not-of-a-weight-is-its-complement-w--1--w) | contradicted |
| 65 | [The inversion has to stay a NOT all the way up — parity in a stack, commutation in a search](#65-the-inversion-has-to-stay-a-not-all-the-way-up--parity-in-a-stack-commutation-in-a-search) | confirmed |
| 66 | [Inversion needs a branch — 2NRL on a tree teaches nothing until the failure departs from a success](#66-inversion-needs-a-branch--2nrl-on-a-tree-teaches-nothing-until-the-failure-departs-from-a-success) | refined |
| 67 | [Phase 1 ends when the failure is learned, not on a date](#67-phase-1-ends-when-the-failure-is-learned-not-on-a-date) | refined |
| 68 | ["Pull hard on the thread" is a schedule over search breadth, and its trigger is a reward](#68-pull-hard-on-the-thread-is-a-schedule-over-search-breadth-and-its-trigger-is-a-reward) | held |
| | **[Games as text, and the rules of chess](#games-as-text-and-the-rules-of-chess)** | |
| 69 | [Is there a way to encode all games? — one tape, one encoder, and the rules as referee](#69-is-there-a-way-to-encode-all-games--one-tape-one-encoder-and-the-rules-as-referee) | confirmed |
| 70 | [RadixNet knows a token by its three characters and nothing else](#70-radixnet-knows-a-token-by-its-three-characters-and-nothing-else) | confirmed |
| 71 | [The memory is one token — spend it on something with locality](#71-the-memory-is-one-token--spend-it-on-something-with-locality) | confirmed |
| 72 | [One network, six games — the header token is the selector](#72-one-network-six-games--the-header-token-is-the-selector) | refined |
| 73 | [A rating of a network that cannot find a legal move rates its search for one](#73-a-rating-of-a-network-that-cannot-find-a-legal-move-rates-its-search-for-one) | confirmed |
| | **[The radix graph — structure and search](#the-radix-graph--structure-and-search)** | |
| 74 | [A radix tree for information compression — a small network at every node](#74-a-radix-tree-for-information-compression--a-small-network-at-every-node) | held |
| 75 | [A trigram because of how the transformer pivots](#75-a-trigram-because-of-how-the-transformer-pivots) | held |
| 76 | [The graph compresses itself, lossy on purpose — parameters discarded, predictions kept](#76-the-graph-compresses-itself-lossy-on-purpose--parameters-discarded-predictions-kept) | refined |
| 77 | [The encoding is a dial of the model — the graph never mentions a character](#77-the-encoding-is-a-dial-of-the-model--the-graph-never-mentions-a-character) | held |
| 78 | [Prediction is a shortest path — and a shortest path prefers to stop](#78-prediction-is-a-shortest-path--and-a-shortest-path-prefers-to-stop) | confirmed |
| 79 | [Accept the vanishing gradient — a one-hop rule makes `n = 1`](#79-accept-the-vanishing-gradient--a-one-hop-rule-makes-n--1) | held |
| 80 | [The structure is independent of what a weight means — one graph, four model kinds](#80-the-structure-is-independent-of-what-a-weight-means--one-graph-four-model-kinds) | confirmed |
| 81 | [Frequency alone cannot move on — weigh an edge's all-time share against its recent share](#81-frequency-alone-cannot-move-on--weigh-an-edges-all-time-share-against-its-recent-share) | held |
| 82 | [A sine has a phase — the resonant model](#82-a-sine-has-a-phase--the-resonant-model) | refined |
| 83 | [Judgements follow the path, not the edge](#83-judgements-follow-the-path-not-the-edge) | held |
| 84 | [A dynamic window over the nodes — halve down a binary ladder, grow back at the top](#84-a-dynamic-window-over-the-nodes--halve-down-a-binary-ladder-grow-back-at-the-top) | held |
| 85 | [A split and a merge must be inverses for everything the model hangs on an edge](#85-a-split-and-a-merge-must-be-inverses-for-everything-the-model-hangs-on-an-edge) | confirmed |
| 86 | [Traverse by the punishments, not the rewards — and blame is not for sale](#86-traverse-by-the-punishments-not-the-rewards--and-blame-is-not-for-sale) | held |
| 87 | [An attention band for each n-gram — a correction lands where a gram looks, not where it wrote](#87-an-attention-band-for-each-n-gram--a-correction-lands-where-a-gram-looks-not-where-it-wrote) | held |
| 88 | [Train backwards to learn what came before](#88-train-backwards-to-learn-what-came-before) | held |
| 89 | [Rehearse what was read before, start easy, and never let an easy epoch stop the run](#89-rehearse-what-was-read-before-start-easy-and-never-let-an-easy-epoch-stop-the-run) | held |
| 90 | [Counters are odometers — cycles are a feature in arithmetic too](#90-counters-are-odometers--cycles-are-a-feature-in-arithmetic-too) | confirmed |
| 91 | [Racy counting lost 0.7% for no speed-up — the speed came from representation, not from the race or the language](#91-racy-counting-lost-07-for-no-speed-up--the-speed-came-from-representation-not-from-the-race-or-the-language) | contradicted |
| | **[The radix graph — failure, judgement and teachers](#the-radix-graph--failure-judgement-and-teachers)** | |
| 92 | [A standing model of failure — 2NRL's negative phase made permanent](#92-a-standing-model-of-failure--2nrls-negative-phase-made-permanent) | held |
| 93 | [A filter must not be taught by what already passed it — guard what a person sees, not what a judge sees](#93-a-filter-must-not-be-taught-by-what-already-passed-it--guard-what-a-person-sees-not-what-a-judge-sees) | held |
| 94 | [Blame at the granularity the failure happened at — a correction is a diff](#94-blame-at-the-granularity-the-failure-happened-at--a-correction-is-a-diff) | held |
| 95 | [The LLM supplies the language; a deterministic rule supplies the decision](#95-the-llm-supplies-the-language-a-deterministic-rule-supplies-the-decision) | held |
| 96 | [Code is the one reward that is not an opinion](#96-code-is-the-one-reward-that-is-not-an-opinion) | held |
| 97 | [A tool call is text the network writes, so an attempt is one training text](#97-a-tool-call-is-text-the-network-writes-so-an-attempt-is-one-training-text) | held |
| 98 | [The perpetual self-upgrade has no yardstick](#98-the-perpetual-self-upgrade-has-no-yardstick) | held |
| | **[The radix graph — going round, and thinking](#the-radix-graph--going-round-and-thinking)** | |
| 99 | [The walk may go round — the hand-over is a reflex taught by experience and overruled by memory](#99-the-walk-may-go-round--the-hand-over-is-a-reflex-taught-by-experience-and-overruled-by-memory) | refined |
| 100 | [A lesson that stops the walk cannot be unlearned by walking — forget on the graph's own clock](#100-a-lesson-that-stops-the-walk-cannot-be-unlearned-by-walking--forget-on-the-graphs-own-clock) | held |
| 101 | [Thinking is a fourth sentinel that faces both ways — and the thinking a client sees is the search's own trace](#101-thinking-is-a-fourth-sentinel-that-faces-both-ways--and-the-thinking-a-client-sees-is-the-searchs-own-trace) | held |
| | **[Without the cycle, and with forgetting — RadixAcyclicNN and RadixDecayNN](#without-the-cycle-and-with-forgetting--radixacyclicnn-and-radixdecaynn)** | |
| 102 | [How deep should the context be? Held-out prediction stops improving at three to four grams](#102-how-deep-should-the-context-be-held-out-prediction-stops-improving-at-three-to-four-grams) | refined |
| 103 | [A query is a traversal — what the model says, it becomes likelier to say](#103-a-query-is-a-traversal--what-the-model-says-it-becomes-likelier-to-say) | confirmed |
| 104 | [A memory fades on the model's own clock — and a life shorter than a reading cannot hold the reading](#104-a-memory-fades-on-the-models-own-clock--and-a-life-shorter-than-a-reading-cannot-hold-the-reading) | confirmed |
| 105 | [Use keeps a memory, and reading brings it back — only what is used, and only what is re-read](#105-use-keeps-a-memory-and-reading-brings-it-back--only-what-is-used-and-only-what-is-re-read) | confirmed |
| | **[Priming instead of growing — PrimedRadixPair and LatentRadixPair](#priming-instead-of-growing--primedradixpair-and-latentradixpair)** | |
| 106 | [Prime, don't grow — learning is counting at a computed address, and the tokenizer decides the depth](#106-prime-dont-grow--learning-is-counting-at-a-computed-address-and-the-tokenizer-decides-the-depth) | refined |
| 107 | [Connected at every level, not just the final nodes](#107-connected-at-every-level-not-just-the-final-nodes) | confirmed |
| 108 | [What was read and what worked, kept in two trees — and a punishment cannot be bought back](#108-what-was-read-and-what-worked-kept-in-two-trees--and-a-punishment-cannot-be-bought-back) | confirmed |
| 109 | [On a primed tree the cheapest path is the wrong decoder](#109-on-a-primed-tree-the-cheapest-path-is-the-wrong-decoder) | confirmed |
| 110 | [A dynamic tokenization — the window grows one letter at a time](#110-a-dynamic-tokenization--the-window-grows-one-letter-at-a-time) | refined |
| 111 | [Tokenize the context, not the units — a learned 12-bit code of the last 16 bytes](#111-tokenize-the-context-not-the-units--a-learned-12-bit-code-of-the-last-16-bytes) | refined |
| 112 | [The first symbol of a learned code replaces a market of slices](#112-the-first-symbol-of-a-learned-code-replaces-a-market-of-slices) | held |
| 113 | [Verdicts in outcome units](#113-verdicts-in-outcome-units) | held |
| | **[Text as sound](#text-as-sound)** | |
| 114 | [A sound is a unit — *night* and *knight* are one thing to the model](#114-a-sound-is-a-unit--night-and-knight-are-one-thing-to-the-model) | held |
| 115 | [Acoustic units — an alphabet learned from audio, small enough to recur](#115-acoustic-units--an-alphabet-learned-from-audio-small-enough-to-recur) | held |
| 116 | [Keep the source, learn the filter — a neural vocoder for the acoustic units](#116-keep-the-source-learn-the-filter--a-neural-vocoder-for-the-acoustic-units) | refined |
| 117 | [The traditional LLM tokenizer is one more setting of the encoding dial](#117-the-traditional-llm-tokenizer-is-one-more-setting-of-the-encoding-dial) | held |
| 118 | [Mark a model of sounds in English, teach it in sounds](#118-mark-a-model-of-sounds-in-english-teach-it-in-sounds) | held |
| 119 | [Talk with the model — the conversation itself is what teaches it](#119-talk-with-the-model--the-conversation-itself-is-what-teaches-it) | held |
| 120 | [Any modality is text — and a text the model was taught is its own answer key](#120-any-modality-is-text--and-a-text-the-model-was-taught-is-its-own-answer-key) | held |
| 121 | [A sound, drawn — the picture is the sound once it carries the phase](#121-a-sound-drawn--the-picture-is-the-sound-once-it-carries-the-phase) | refined |
| 122 | [One node per hertz, reading vertical strips — it cannot beat "the next strip is this one"](#122-one-node-per-hertz-reading-vertical-strips--it-cannot-beat-the-next-strip-is-this-one) | contradicted |
| | **[Filter banks — FilterBankRadix](#filter-banks--filterbankradix)** | |
| 123 | [Layer 1 is an activation filter feeding a set of radix trees](#123-layer-1-is-an-activation-filter-feeding-a-set-of-radix-trees) | refined |
| 124 | [Learn a radix tree's branch probabilities with the one-hop rule](#124-learn-a-radix-trees-branch-probabilities-with-the-one-hop-rule) | contradicted |
| 125 | [Locate the gap with measurements the architecture may never make](#125-locate-the-gap-with-measurements-the-architecture-may-never-make) | confirmed |
| 126 | [Price the counterfactual out of sample, and never let ignorance be cheaper than knowledge](#126-price-the-counterfactual-out-of-sample-and-never-let-ignorance-be-cheaper-than-knowledge) | confirmed |
| | **[An agent that improves itself — DISTIL](#an-agent-that-improves-itself--distil)** | |
| 127 | [The embedding layer is the memory, and the grade on it is what matters](#127-the-embedding-layer-is-the-memory-and-the-grade-on-it-is-what-matters) | confirmed |
| 128 | [An optimiser that can edit its own grader will edit its own grader](#128-an-optimiser-that-can-edit-its-own-grader-will-edit-its-own-grader) | confirmed |
| 129 | [Distil a task until a check can be written; ask when none can](#129-distil-a-task-until-a-check-can-be-written-ask-when-none-can) | refined |
| 130 | [Chain of thought built on game theory — planning is a game against Nature](#130-chain-of-thought-built-on-game-theory--planning-is-a-game-against-nature) | held |
| 131 | [Question everything, never take no for an answer, run the bad ideas — all three are paid in information](#131-question-everything-never-take-no-for-an-answer-run-the-bad-ideas--all-three-are-paid-in-information) | held |
| 132 | [Think by clustering the embeddings — a concept is a centroid written back](#132-think-by-clustering-the-embeddings--a-concept-is-a-centroid-written-back) | held |
| | **[Not yet built](#not-yet-built)** | |
| 133 | [Multiple gradients as stacked planes, with the layer chosen by depth perception](#133-multiple-gradients-as-stacked-planes-with-the-layer-chosen-by-depth-perception) | held |
| | **[Method — how the evidence was kept honest](#method--how-the-evidence-was-kept-honest)** | |
| 134 | [Parity is equality, not tolerance — and three implementations are a measuring instrument](#134-parity-is-equality-not-tolerance--and-three-implementations-are-a-measuring-instrument) | confirmed |
| 135 | [A number the committed script does not print is withdrawn — and a floored mean hid a 600× loss](#135-a-number-the-committed-script-does-not-print-is-withdrawn--and-a-floored-mean-hid-a-600-loss) | confirmed |
| 136 | [Measure each game in its own units, at matched compute, over seeds — five traps that would have reversed a result](#136-measure-each-game-in-its-own-units-at-matched-compute-over-seeds--five-traps-that-would-have-reversed-a-result) | confirmed |
| 137 | [Every arm peaked early — keep the best network, and grade it on positions it was not chosen with](#137-every-arm-peaked-early--keep-the-best-network-and-grade-it-on-positions-it-was-not-chosen-with) | confirmed |
| 138 | [An arm's name is not its configuration](#138-an-arms-name-is-not-its-configuration) | confirmed |
| 139 | [Ask a function only the question it answers — two bugs that produced plausible numbers](#139-ask-a-function-only-the-question-it-answers--two-bugs-that-produced-plausible-numbers) | confirmed |
| 140 | [Judge a decode by what the picture stored — the rulers that lied](#140-judge-a-decode-by-what-the-picture-stored--the-rulers-that-lied) | confirmed |
| 141 | [Anything a loop writes is input to that loop](#141-anything-a-loop-writes-is-input-to-that-loop) | confirmed |
| 142 | [Review a frozen tree, and re-review the fixed one](#142-review-a-frozen-tree-and-re-review-the-fixed-one) | confirmed |

---

## The architecture

### 1. AGI is a game-solving problem
> To solve a game you must understand it. To understand it you must explore it.
> Once you understand the game, then and only then can you play the game.

The whole multi-network architecture follows from this ordering. It is specified
to be enforced mechanically rather than assumed: `GREN` produces a rule package
and `GTMNN` refuses to play one below a confidence floor. As built (branch
`feat/gren-impl`), GTMNN reads every package it is given; the gate that exists is
CyclicCortex's *characterisation* floor, and 31 shows why a confidence floor
would have been the wrong gate — it refuses sudoku (0.35) and go (0.421), games
GREN understands well.

*Where:* `GREN/DESIGN.md` §1, §22.1 · **held** (specified, not enforced on the
GTMNN side; the gate that was built is characterisation — 31)

### 2. Game theory, not one big network — GTMNN
> A population of micro neural networks that play a game against each other. The
> answer is the equilibrium of that game.

*Implemented and measured, then re-measured.* The machinery works and every
game-theoretic claim it rests on is asserted numerically (`GTMNN/tests/`, 49
tests). The first measurement said the population barely learns to predict. That
was wrong, and wrong for a reason the repository had already met twice: the
comparison, not the thing compared.

**It learns.** Real inference loss — `y` withheld, `BeliefCongestion`, solver,
aggregate — is **2.353** against a uniform 2.708 at 60 epochs, below uniform at
every budget and improving monotonically. The credit path was never broken; its
coefficient was mis-scaled. `φ` is a marginal contribution with a mean `|φ|` of
about 0.15 per seat, so at `lr = 0.05` the population learned ~7× slower than
the same micros under a unit supervised signal, and "does not learn" was "learns
slowly, measured against a control given 30–100 epochs to its 4–6". Three of my
claims fell: the supervised-vs-credit gap (mismatched budgets; at matched 6
epochs it is 0.07), "more training does not help" (old code — 139),
and "the abstain trap is not real" (it is — it just needs a signal that raises
`q_abstain` to fire).

**The gap that remains is structural.** Supervised reaches 0.905. Only seats
holding `y` ever receive non-zero credit — about 23% of seat-games — and a
correct abstention is a null player, so a micro is never taught to be quiet;
`q_abstain` *falls* over training. Supplying that signal the principled way
(counterfactual regret with Shapley as payoff) does what it was designed to do —
`q_abstain` rises to 0.33 — and collapses the inference game to uniform,
`2.708 ± 0.00`: abstainers are 77% of seats, their bids fall under the reserve,
nobody is seated, the aggregate is the ε-smoothing. The abstain trap, closed.
The sharper open question: how do you teach a micro to be quiet without it going
silent for good?

The population does specialise: `gini` rises to 0.54 when every micro is seated
against 0.05 when 24 of 256 are (measured before the coefficient fix, at
`lr = 0.05`).

*Where:* `GTMNN/DESIGN.md`; the implementation, its tests and these numbers are
in `AGI/GTMNN/` on branch `feat/gren-impl` (the specialisation figures in
`GTMNN/README.md` on `claude/keen-feynman-ntoqni`) · **refined** (it learns; the
coefficient was wrong, not the mechanism; the gap to supervised is real and its
cause is named)

### 3. Credit belongs to game theory, not the chain rule
Backpropagation answers "who is responsible" with the derivative. Shapley (1953)
answered the same question in 1953 and proved the answer unique under four
axioms. No gradient crosses a micro boundary anywhere in GTMNN.

*Confirmed, and more exactly than stated.* Efficiency `Σφ = v(N) − v(∅)` holds to
**3e-15** over 200 games at 1, 2, 7 and 32 permutations — it is exact *per
sample*, not in expectation, because the marginal contributions telescope along a
single permutation. The null player gets **exactly** `0.0` and takes exactly no
step; two interchangeable seats get identical credit to `1e-12` when enumerated;
and over all `M!` permutations the incremental estimator equals the closed-form
subset-weighted sum to `4e-16`, so any gap at finite samples is variance falling
as `1/√n` rather than bias.

Also confirmed: no gradient crosses a micro (checksummed around every `learn`),
and all 45 parameters of a micro — weights, biases and the four sine parameters —
match finite differences to `1e-10`.

The credit was therefore never what held 2 back: it is exactly what it claims to
be, and once its coefficient was scaled to `φ` the population learns. With
regions instead of micros the same credit is computed exactly rather than
sampled (47), and DISTIL applies it over subgoals (130).

*Where:* `GTMNN/DESIGN.md` §12; `GTMNN/gtmnn/shapley.py` and its tests on branch
`feat/gren-impl` · **confirmed**

### 4. Everything can be gamified
*Refined.* The **encoding** is general — anything with a refusal oracle can be
probed and placed. The **learnability** is not, and the gap is six orders of
magnitude: the same table that says programming is learnable in about an hour
says a conversation would take 150 years. But *identification* needs only
`log2|G|` bits regardless of rule-set size, which collapses the intractable rows
to minutes. So everything can be *placed*; not everything can be *learned* by
probing. The universal game encoder measured the same split on a sequence model:
six games through one encoder that knows none of them, and legality learned from
**0.999** on Nim to **0.08** on chess (69).

*Where:* `GREN/DESIGN.md` §5, `Experiments/UniversalGameEncoding/` · **refined**

---

## Learning the rules

### 5. You learn the rules by failing on purpose
> When you fail at chess there are markers: no, you can't do that move. You learn
> the game by failing and guessing.

This became the central claim of GREN: **a game's identity is its refusal
boundary**, and two games are similar exactly to the degree they refuse the same
things.

*Confirmed on real games.* GREN never sees a game's mechanic names: it fires
moves, records which of sixteen shared refusal codes come back and how often,
and emits a signature. That signature reproduces the hand-written geometry —
Spearman **ρ = 0.886** between discovered and hand-written pairwise distances,
chess/checkers the closest pair under both (**0.368** discovered, **0.529**
written) — and, swapped in for CyclicCortex's hand-written mechanic sets, it
forms the same three regions with the same members and plays bit-for-bit the
same. What is discovered is *which* codes a game has and how often, not the
codes: the oracle names its own, as a compiler does. A blind explorer later
found six of chess's seven refusal kinds from 166 random submissions with no
rules in hand (41). Learning the rules *from* one's own refusals, one at a time,
turned out to be a thin way to ask the oracle (44).

*Where:* `GREN/DESIGN.md` §2, `GREN/README.md` ("The result that matters"); the
CyclicCortex swap in `CyclicCortex/README.md` on branch `feat/gren-impl` ·
**confirmed**

### 6. The failure rate should be about 50%
Not a setting — a consequence. Choosing probes by expected information gain is
provably maximised, for a binary oracle, at `p(legal) = 0.5`. The most
informative probe is the one half the surviving candidates call legal. The
system reports its observed failure rate as a diagnostic: a run at 5% is
confirming what it already believes; at 95% it is thrashing outside the boundary.

*Measured, and it holds where the boundary is tight.* At 1,200 probes the EIG
policy fails at **0.480** in chess and **0.545** in checkers, near the
prediction, but at **0.282** in sudoku and **0.159** in go, where most points are
legal (legality density 0.84) and there is no 50/50 split to find; probing at
the boundary finds all seven of chess's refusal kinds where random probing finds
four. That explorer's candidate pool is half legal by construction, so part of
its 50% was handed to it. The blind explorer of 41 measured the other side: go
0.51 and chess 0.94 under the same p = ½ policy, because in a sparse grammar
failure is not a choice — and the policy that *maximises* failure is the worst
mapper everywhere.

*Where:* `GREN/DESIGN.md` §4, `GREN/README.md` ("The 50% failure rate") ·
**refined** (holds where the boundary is tight; in a sparse grammar the recipe
is to keep every acceptance, not to fail more — 41)

### 7. Programming is the game worth optimising for
> The compiler will tell you: this ain't working, bro.

*Confirmed by arithmetic.* It is the only domain where a rule set large enough to
matter meets an oracle fast enough to exhaust it: free, instant, complete, never
bored, and it ships with its own reason taxonomy already enumerated. The
diagnostic text is the highest-bandwidth channel in the system — `expected i32,
found &str` localises a type rule in one probe where a bare rejection would take
hundreds. RadixCyclicNN built the grounded half — code as the one reward that is
not an opinion (96) — and has not yet shown it teaching.

*Where:* `GREN/DESIGN.md` §5, §11.2 · **confirmed**

---

## Finding which game you are in

### 8. Identify the game by finding similar games
> We want to figure out which game we're playing by finding similar games, based
> on what we know about the game.

*Confirmed, and it is worth three to four orders of magnitude.* Learning Rust's
rules from nothing is ~10⁵ bits; recognising "this is Rust-like" against a corpus
of 1000 is ~10 bits — ten probes. The neighbour's rules are then inherited and
probing is spent only where the surviving candidates disagree. Most of a game's
rules are never probed at all.

*Refined for inheritance by 40:* the most similar neighbour can hand over the
wrong rule where a shared feature means something different — a chess-trained
network calls **95%** of checkers' legal moves `BLOCKED_PATH` — so a rule is safe
to inherit only where the feature that carries it means the same thing.

*Where:* `GREN/DESIGN.md` §3 · **confirmed** (identification, by derivation and
by the similarity numbers in §11 below; inheritance **refined** by 40)

### 9. Information clustering — chess and checkers are closer than chess and a difficult conversation
*Confirmed numerically.* Over mechanic sets: chess/checkers **0.39**,
chess/go **0.23**, chess/rust **0.03**, chess/boss-conversation **0.07**.
Correct ordering, reproduced from the axes rather than asserted. Weighting the
same sets by rarity makes every similarity worse (49).

*Where:* `GREN/DESIGN.md` §8.6 · **confirmed**

### 10. Use what we know about the game: inputs, outputs, category, goal
These four generalise better than the purely game-theoretic axes they replaced,
and they split into **declared** (free — you read the box before you play) and
**measured** (costs probes). Retrieval on declared axes alone routinely cuts a
thousand-game corpus to a handful, which is the single largest efficiency in the
system. Of the four, the goal turned out to be the one to put first (37).

*Where:* `GREN/DESIGN.md` §9 · **held**

### 11. Break each game into its smallest possible parts first
*Confirmed, and it dissolved a defect rather than patching one.* Decomposing to
**mechanics** — the smallest unit comparable across domains — makes a game a
*set*, and set overlap is symmetric and order-free. The chess/go pathology (an
early axis disagreement making a single tree call them unrelated) cannot occur
over sets.

"Smallest possible" got a criterion that is measured rather than argued: **a part
is minimal when no proper sub-part is ever observed independently in the
corpus.** `CHAIN_CAPTURE` and `FORCED_CAPTURE` split, because draughts variants
exist with one and not the other. `CASTLING` does not, despite looking compound,
because it never varies in halves.

*Contradicted as a route to grouping by 40:* grouping by shared mechanic *names*
measurably groups the wrong pairs, because a name does not carry its condition.
Decomposition identifies; it does not yet say what transfers (What to test
next 2 is the acceptance test for splitting further).

*Where:* `GREN/DESIGN.md` §8 · **confirmed** (decomposition and identification;
the grouping claim **contradicted** by 40)

### 12. An n-dimensional trie, with the game encoded into each path
### 13. A radix tree with the mechanics of a trie
Both halves are load-bearing and neither alone suffices. **Trie mechanics** give
terminal flags on internal nodes, so a general class can be a strict prefix of a
specific one — "certainly a two-player perfect-information capture game, which
one is still open" is a real answer whose shared rules can be played. **Radix
compression** means only the mechanics that actually discriminate cost a node.
Their interaction is one rule: *a terminal node is never merged away.*

*Where:* `GREN/DESIGN.md` §19 · **held**

### 14. Or invert the problem — a random forest
*Contradicted, honestly.* The forest existed to patch the chess/go ordering
problem. Insight 11 dissolved that problem at its source, so the forest's
original justification is gone. What survives is a weaker hedge against
frequency-ordering fragmentation; `T` dropped 64 → 16 and an open question asks
whether it should be 1 and the module deleted.

*Where:* `GREN/DESIGN.md` §20, §32.7 · **contradicted** (superseded by 11)

---

## Compression

### 15. Neural compression: split the output range, plus a selector
> One game takes 0 to 1. Two games take 0–0.5 and 0.5–1. Four games take 25%
> each. On query, expand the output back to the 0–1 range.

*Refined, and the refinement is the interesting part.* One network genuinely does
hold sixteen games at **321 parameters and 0.0081 error**, against sixteen
separate networks at **1040 parameters and 0.0101 error** — fewer parameters
*and* better accuracy. **But the output partition is not what compresses.** Plain
conditioning is 7.5× more accurate at the same scale, and the slopes run
opposite: as games are added the partition degrades while conditioning improves.

**What the partition actually wins is games it has never played.** Trained on
even-indexed games only, one-hot conditioning memorises the seen ones at 0.0045
and collapses to 0.1185 on unseen (26×) — it *structurally cannot* do better,
since an unseen game has no slot to put a 1 in. The continuous selector
coordinate degrades 8%, and the partitioned version does not degrade at all.

So the value is not compression. It is that **a continuous game coordinate lets
you play a game you have never played, by placing it between games you have.**
For an AGI architecture that is the more important of the two.

Two caveats from the re-measured findings: "beats separate networks" is a tanh
result — under `sine b=1` separate networks win at every N from 4 up — and
predicting a flat 0.5 scores **0.081**, a floor the zero-shot numbers sit close
to; every table in §1–5 is a single seed. The experiments' own recommendation —
keep the coordinate, drop the partition — outruns its tables (59).
One encoder later reached the selector without a partition at all: the header
token of a game tape (72).

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §1–4; the caveats in the
same file as rewritten on branch `claude/keen-feynman-ntoqni` · **refined**

### 16. Order the partition by similarity
*Confirmed, and it is mandatory rather than a nicety.* Absent at N=2, 2.5× at
N=8, and at N=16 it is the difference between working and learning nothing:
shuffled, the tanh network scores **0.9255**, 15× the ordered error and 11.4×
worse than predicting a constant (9.7× with sine). Ordered, the target is a
smooth function of the selector coordinate; shuffled, it is a high-frequency
sawtooth a small network cannot fit.

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §3 (the 0.9255 as
re-measured on branch `claude/keen-feynman-ntoqni`, where main's copy says
"diverged") · **confirmed**

### 17. Avoid dust — binary doesn't work because of 16 and 32
*Confirmed, and the mechanism is worse than waste.* **Dust is error
amplification.** Reading a band back out multiplies output error by the slot
count, so reserving slots for games that are not there degrades the games that
are: 50% dust costs **3.1×**, 75% costs **4.5×**. Padding 17 games into 32 slots
is roughly a 3× penalty for nothing.

The static-sequence idea it was meant to enable turns out not to be needed:
re-laying every band out costs **1.37×** and freezing them costs **1.36×** —
indistinguishable, because the selector is continuous and rescaling it is a
smooth reparameterisation rather than a permutation. **3.1× against 1.37×, both
pointing the same way: take the relayout, refuse the dust.** Those are means over
four seeds; per seed the damage ranges 0.60× to 3.74×, so the ordering is the
finding, not the magnitude.

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §6–7 (the per-seed spread as
re-measured on branch `claude/keen-feynman-ntoqni`) · **confirmed**

### 18. Same input and output size for every game, plus a game modifier
*Held, with a concrete design.* Score **one candidate at a time**, and action
space size never touches the geometry — 50 moves, 361, or infinitely many Rust
programs all work. The modifier is the mechanic set hashed into 128 dimensions,
and because hashing a *set* preserves overlap, similar games land at nearby
points automatically. **Transfer is a consequence of the encoding rather than a
mechanism added on top.** Width chosen by measurement: 0.177 mean error at 32
dimensions, 0.061 at 128.

*Refined by 32:* the hash does track mechanic overlap, but in a population the
modifier acts as a game-*identity* signal, and it was worth **−0.062** to
chess→checkers transfer, negative in 5 of 6 seeds (measured at 64 dimensions,
not the 128 chosen here). The universal game encoder reached the fixed
interface for a sequence model — one action token whatever the action space —
with a header token as the modifier (69).

*Where:* `GTMNN/DESIGN.md` via `GREN/DESIGN.md` §22.4–22.5 · **refined** (the
hash works; the transfer claim does not — 32)

### 19. Train validity first, then grade the move separately
> Focus on training on valid/invalid moves, then a separate grade/reward for the
> move itself. This lets us compress go, chess and checkers into the same space.

*Held, and it is the load-bearing half of 18.* Validity is binary, instant,
abundant and **transferable** — legality is structure, and the modifier says
which structure. Grade is sparse, delayed and game-specific. Training them
jointly would let the noisy signal contaminate the exact one. It also makes
"understand the game, then play it" true *inside a single micro's weights*, not
only at the system level.

*Refined: the separation is confirmed and the transfer is not.* Every time legal
and good were blurred, something broke, and putting legality on its own wire cut
chess's refusals ×89 (44). But the decisive test has been run:
`p_valid` trained on chess leaves checkers at **+0.023** over its majority class,
and the modifier is worth −0.062 (32). Validity transfers only where the feature
that carries a rule means the same thing in both games (40).

*Where:* `GREN/DESIGN.md` §22.6, `CyclicCortex/README.md`,
`TwoNRL_Chess/README.md` · **refined** (the separation confirmed; chess→checkers
validity transfer measured at about nothing — the +0.023 on branch
`feat/gren-impl`)

---

## The vanishing gradient

### 20. The vanishing gradient is a feature, not a bug
*Contradicted in this architecture, specifically.* The sine's derivative near zero
is `−cos(b·z)·b ≈ −b`, so depth multiplies the gradient by `b^depth`. At depth 4
with `b = 1/3` the layer-0 gradient is `7.6e-05` and test MSE `0.0858`; at `b=1`
they are `1.7e-03` and `0.00016`. **22× the gradient, 536× the accuracy.** At
`b=1` the gradient does not vanish at all and depth 4 becomes the *best* result —
better than depth 1 or 2, which is what depth is supposed to buy.

The general case for designing around vanishing gradients may hold elsewhere.
Here the gradient vanishes because every neuron was initialised into the linear
region of its activation, and it stops when that is corrected. See 24.

This tests the sine-slope version of the claim. The paper of the same name
claims something else — that the decay is an invertible reading of depth, to be
counted and divided back out — and that is tested in 51; the
radix graph dissolved the problem instead, with a one-hop rule (79).

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §10 · **contradicted**
(as a property to accept; the observation that it occurs was correct)

### 21. Invert the gradient when the vanishing threshold is reached
*Contradicted — no trigger tested helps, and most of them are catastrophic.*
Three detectors were built: gradient magnitude, weight magnitude, and a loss
**plateau**. Across three seeds and two healthy activations plus one genuinely
stalled one, **the best result any of them achieves is 1.00× — no effect at
all** — and on a healthy network every policy that fires does damage, by 1.8× to
520×. A single inversion is enough: the weight trigger fired *once*, on one seed,
and took that seed from 0.00016 to 0.0961.

Triggering on **gradient magnitude** is the worst of the three (243–520× worse on
healthy networks), because a small gradient means stuck *or* converged *or* still
starting and magnitude cannot separate them. A **plateau** is the better
*detector* — 6 fires against the gradient trigger's 132 on healthy tanh — but
detecting the stall correctly does not make climbing out of it work, and it still
fired 11 times on the healthy `sine b=1` network.

**An earlier 1.24× for plateau-with-long-bursts is withdrawn: it does not
reproduce.** Sweeping burst length on the stalled network gives 1.00×, 0.98×,
0.88×, then divergence — monotonically worse — and the committed defaults fire
23 times where the retired claim reported 15, so it came from a configuration
that was never in the code (135). Gating on "the loss is still bad"
does make the policy safe, by reducing it to **zero fires**.

Keep the plateau detector for deciding when to **grow** a network
(`CyclicCortex/DESIGN.md` §8 uses it for exactly that). Do not build the
inversion it was meant to trigger. Fixing `b` is worth 536× on the same stall.

*Where:* `Experiments/NeuralCompression/vanishing.py`, FINDINGS §11 as rewritten
on branch `claude/keen-feynman-ntoqni` (main's copy still carries the withdrawn
1.24×) · **contradicted**

### 22. Dual network: train in one, query the other — the REM analogy
*Confirmed, and the analogy predicted the better implementation before either was
run.* **Replay beats grafting.** Grafted weights arrive in a network with an
input the specialist never had and an output scaled to a band — wrong coordinate
system, which fine-tuning must repair. Replay transfers the *function* rather
than the *parameters*, so nothing has to match. That is also what the
neuroscience describes: consolidation in sleep is replay-driven, the hippocampus
regenerating experience for the neocortex, not synapses copied between
structures.

Two corrections to the magnitude. Replay's apparent 1.57× is mostly compute; at
matched budget it is **1.05×**. And grafting's real value is not accuracy but
**compute** — 0.0542 using 38,400 query-network updates, a third of what joint
training needs to reach a worse 0.0626 — and at matched budget it is actually
*behind* joint training, **0.77×**. Specialists train independently and in
parallel; the shared network, the contended resource, only fine-tunes.

What neither fixes: specialists alone reach 0.0081 where the best shared network
reaches 0.0398. **Compression costs about 5× and no consolidation recovers it.**

*Where:* `Experiments/NeuralCompression/consolidate.py`, FINDINGS §12 (the
matched-budget 0.77× and the script that prints the 1.05× are on branch
`claude/keen-feynman-ntoqni`) · **confirmed**

### 23. Rotating inversion — invert alternating layers each cycle
*Safe, and for the same reason useless — tested.* With an odd activation
(`−sin(bz)` is odd), negating a layer's weights is a genuine **symmetry
operation**, and the parity alternation has **period 4**: odd, even, odd, even
returns every weight to its original value exactly (verified). That is a cycle
through the network's own symmetry orbit, and it connects directly to
`CyclesAreAFeature.md`. (With an offset, as RadixCyclicNN's units have, negation
is a symmetry only if the offset is negated too — 63.)

It behaves completely differently from every gradient-ascent method: **0.71–1.04×
where ascent-based inversion is 0.00×** (100× to a million× worse). Negating
weights relocates the network without unlearning; ascent destroys the learned
function. Short periods (p=10) are mildly harmful.

But it does not help, on either a smooth task or a deliberately rugged one with
many basins — **1.00×, exactly neutral.** The reason is not that it preserves the
loss: measured across four negations, the loss **jumps ~75×** (0.0034 → 0.25) and
only returns at cycle 4, when the weights do. Each hop is a large perturbation,
not a free move along a level set. What makes it neutral is the **closure** — the
network is kicked out, partly repaired by training, kicked again, and returns
exactly to where it began. *A cycle that returns to its origin does no work.*

Which says what would have to change for it to pay off: **break the closure.**
Period 4 comes from two parities over an involution. Three phases, an asymmetric
subset per cycle, or pairing the negation with something non-involutive would
make the excursion open, and the network would explore instead of returning.
Untested, and the obvious next thing to try.

*Where:* `Experiments/NeuralCompression/vanishing.py` (`Rotating`, run by ad-hoc
calls — it is not in the script's default run), FINDINGS §15–18 · **refined**

---

## Activation

### 24. The sine wave activation instead of a sigmoid
*Confirmed as an activation, contradicted at the stated frequency.* Sweeping `b`
in `f(x) = a·sin(b(x−h)) + k` on a task whose target **is** a sine of a linear
combination — the case most favourable to it:

| `b` | \|z\| to first peak | test MSE |
|---|---|---|
| **1/3 (as written)** | 4.71 | **0.01674** |
| 1.0 | 1.57 | **0.00015** |
| 2.0 | 0.79 | **0.00004** |
| 3.0 | 0.52 | diverges |
| 5.0 | 0.31 | diverges |
| *tanh reference* | — | *0.00098* |

**At `b = 1/3` the sine is 17× worse than tanh. At `b = 1–2` it is 6–24×
better,** and above 2 it diverges, so the window is bounded on both sides. The
activation is excellent; the frequency is wrong. Reaching the first
peak needs `|z| = π/(2b)`, which at `b = 1/3` is 4.71 — far outside the operating
range once features are L2-normalised, so every neuron sits in the sine's
**linear** region and the network is linear however wide it is.

Learnability does not rescue it: `∂f/∂b = a(x−h)·cos(b(x−h))` is small exactly
when `x` is small, so the gradient that would fix the frequency is suppressed by
the same condition that makes it wrong. **The initialisation is the trap, not the
parameterisation.** The rule is `b ≈ π / (2·E|z|)`. (That learnability cannot
rescue it is argued rather than measured — `b` is never learned in that
experiment — and the two runs that do learn the wave, the gate below and the
self-building test, both gained from it, though neither started in the
small-`|z|` regime the argument is about — 57.)

*And in the other direction — as a **gate** it fails, for the same reason.*
`FilterBankRadix/` reads the wave for its *sign* and uses it to route: a hinge
raises a unit's response by pushing its projection, and the address is the sign
pattern. Held at `b = 1/3` with `a, b, h, k` frozen, that push carries the
wave's argument `|u| = |b(x−h)|` from a median of 0.28 to **143.5** — 100% of
units end up past their first peak, twenty-two periods out, where a sine is
coming *back down* — and the filter's supervised accuracy is **0.243 against a
chance of 0.25**: it cannot be fitted at all. A tanh under the identical rule
ends up past its own first peak too (75.6%) and does not care, because
saturation preserves an ordering and periodicity does not; it scores 0.716. With
the wave learnable the sine recovers to 0.723, and the parameter that recovers it
is `b`, which falls from 0.333 to **0.176** — *learning the wave, here, means
flattening it*, and in the bank itself that flattening is worth 0.18 bits/char
while a tanh filter still routes 0.08 bits/char better than the sine. Same
frequency, two opposite failures: too linear for a deep network, too periodic for
a gate.

The rest of the sine's case — its geometry, its liveness, and the thin evidence
it was adopted on — is 55, 56 and
57; the resonant model reads the other half of the wave, its phase
(82).

*Where:* `Research/SineWaveActivationFunction.md`, FINDINGS §5,
`FilterBankRadix/README.md` · **refined**

### 25. Cycles are a feature, not a bug
Reached independently from the other direction by the game theory. At inference
GTMNN's game is *player-specific*, and Milchtaich (1996) says such games always
have a pure equilibrium but best-response paths may cycle. So the cycle is not the
solver failing — it is a property of the game, iterating harder does not help, and
the correct response is to leave and escalate to metacognition.

Tested since, from both sides. The cycle Milchtaich warns of was hunted across
seven geometries and **1,400** games and never appeared, so it stands as a
worst-case guarantee, not the typical case (34, 48). The paper's own
claims have their own entries — repetition stored once (52), the
exit from outside (53), the odd-cycle obstruction
(54) — and RadixCyclicNN built the hand-over it calls for
(99).

*Where:* `Research/CyclesAreAFeature.md`, `GTMNN/DESIGN.md` §2.1, §14 · **held**
(the game-theoretic reading; its measured halves are the entries above)

---

## Cortical organisation

### 26. A Single Self-Building Neural Network instead of micro networks
> Inputs and outputs grow to match the games.

One SBNN per *region* rather than thousands of micros. Growth is additive and
**non-destructive**: new hidden units enter with zero outgoing weight and new
inputs with zero incoming weight, so the network computes bit-identically the
instant after growth and only changes as the new capacity learns. Growth on a
**plateau**, the better of the stall detectors 21 compared — though even the
plateau fired on healthy networks there, so it needs a *relative* test and a
stop once the target is met, as
`Experiments/ActivationFunctionTest/self_building_sinewave.py` has. The
precedent first cited here, `Experiments/SBNN_RNN_ActivationFunction/main.py`,
does neither: its growth draws new weights and so perturbs the network, and its
*absolute* plateau threshold grows a converged network to its cap.

*Refined in the building:* identity growth has two exceptions — outputs, and
initialisation — both measured (42).

*Where:* `CyclicCortex/DESIGN.md` §8, `GREN/README.md` ("Self-Building Neural
Networks") · **refined** (identity growth confirmed by tests, with two
exceptions — 42)

### 27. Similar games in similar regions — brain structure
The similarity graph partitions into regions by modularity; each region owns one
network; a new game lands next to what it resembles or founds a new region if
nothing is close enough. This changes what carries game identity: **it moves out
of the input vector and into the graph position**, which is why the region's
network does not need to be told which game it is playing.

Measured since: regions earn their place by *placing* games — a wrong region's
Shapley value is negative on every game — not by teaching each other, which
helps one game in four by 0.025 (47). One network given six games and
told which is which by a header token let a quarter of its answers leak between
them (72), a point for one network per region.

*Where:* `CyclicCortex/DESIGN.md` §6, §9, `CyclicCortex/README.md` ·
**refined** (placement confirmed; regions teaching each other measured at about
nothing — 47)

### 28. A cyclic graph where distance is similarity
*Confirmed.* Jaccard distance over mechanic sets is a **proper metric** — checked
over 210 ordered triples with **zero triangle-inequality violations** — so
"distance is similarity" has a consistent geometry rather than being a figure of
speech.

And it must be **cyclic**: chess/checkers 0.609, checkers/go ~0.730, chess/go
0.769 form a triangle in which none is the parent. A tree must break one edge.
This makes the relationship to insight 13 precise: **the radix tree is a
projection of this graph, not a rival to it** — the tree makes retrieval
sublinear, the graph is what is actually true, and where they disagree the graph
is right.

*Where:* `CyclicCortex/DESIGN.md` §5 · **confirmed**

### 29. Common denominators — generalised, minified inputs
> For chess, a piece moving one space would count here. This allows us to train
> the same network on checkers as well.

*Confirmed, and more strongly than stated.* Training on king moves then testing
on checkers, zero-shot: **0.646** on a generalised vocabulary against **0.514 —
exactly chance —** on a raw board encoding. Three findings beyond the headline:

* **Raw pretraining is actively harmful** (−0.220 at k=5). A board-specific encoding makes the network *worse* at the second game than starting from nothing, because square indices mean different things in different games. This is the failure mode the generalised vocabulary exists to prevent, and it is not hypothetical.
* **It is not an artefact of good feature design.** Stripping the features that encode the checkers rule (`is_diagonal`, `forwardness`) left zero-shot **identical** at 0.646 and made the transfer *gain* larger. The advantage is the shared coordinate system, not the chosen features.
* **Three inputs are enough.** `(dx, dy, target_empty)` matches eight hand-designed features and beats seventy-two. Minification is not a compromise for capacity — the minimal shared vocabulary is the *best* one, because everything removed was game-specific and therefore untransferable.

The common denominators are GREN's mechanics (insight 11) promoted from a
similarity signature into the network's actual input vocabulary: **the objects
that decide two games are alike are the objects the network reads.**

*Refined at cortex scale.* The purpose-built experiment above shares *all three*
of its inputs. A real region does not: region 0 gives chess and checkers 8
shared slots out of 33, and the other 25 are private. Measured there — train
chess alone, then evaluate checkers, against checkers' **majority class** —
transfer is **+0.023**, not +0.286. The untrained network is the wrong baseline:
with *nothing* shared at all, checkers still reaches its majority class, because
training chess moves the region's shared hidden layer and output bias and that
needs no transfer whatsoever. I made that mistake first and it inflated the
figure twelve-fold.

This does not contradict the headline; it is what the headline's third bullet
predicts. Everything removed was game-specific and untransferable, so a
vocabulary that is 25/33 game-specific transfers about as well as its private
part allows. The way to get the 0.646 result inside a region is to make the
shared vocabulary *most* of the vocabulary, not to add more of it. The other
edge of minification — a vocabulary too small to express a rule caps what can be
learned — is 43; and a code that means the same thing
everywhere is what the universal game encoder found worth spending its one token
of memory on (71).

*Where:* `CyclicCortex/common_denominator.py`, `CyclicCortex/DESIGN.md` §4; the
cortex-scale measurement in `CyclicCortex/README.md` on branch `feat/gren-impl` ·
**refined**

---

## The game stack, integrated and measured — GREN, GTMNN, CyclicCortex, Memory

### 30. A shared refusal code is not a shared rule
Not an insight of Curtis's but one the architecture forced, and the sharpest
limit found so far on identification-by-refusal (11, 12).

GREN can derive CyclicCortex's *input* vocabulary as well as its mechanics:
`gren/vocabulary.py` asks which feature dimensions separate the moves a game
refuses with code C from the moves it accepts, then matches dimensions **across**
games by their whole legality signature, so a shared slot holds one quantity
rather than one name. It works — chess's and checkers' `OCCUPANCY[0]` are matched
to each other; go's and sudoku's `GRID_PLACE[2]` are matched to each other and,
with a sign flip, to an occupancy feature of chess and checkers;
and sudoku's `CONSTRAINT_UNIQUE`, which I wrote as one 3-wide block, is separated
into row, column and box without being told there were three.

And it makes chess-to-checkers transfer **catastrophically worse**: −0.409
against the majority baseline, where the hand-written vocabulary gets +0.023.

The alignment is not the fault. It reproduces my own hand-written
occupancy-only ablation to within noise (−0.409 against −0.432) — which is what
a correct measurement of a bad idea looks like. Chess and checkers both refuse
`OCCUPIED_TARGET`, but chess refuses only a target holding your **own** piece,
because taking an enemy piece is a capture, while checkers refuses **any**
occupied target for a simple move. Same code, opposite rule. Sharing a weight
across them teaches chess's exception into checkers. The hand-written vocabulary
escapes this only by *diluting* those three slots with five harmless ones:
`GRID_MOVE` sharing alone contributes exactly 0.000.

**Refusals identify a rule's shape, not its arguments.** That is enough to place
a game on the map and not enough to share a weight — which is why the mechanics
swap is the default and the derived vocabulary is opt-in.

*Where:* `GREN/gren/vocabulary.py`, `CyclicCortex/cortex/discovered.py`,
`CyclicCortex/README.md` — on branch `feat/gren-impl` · **contradicted** (as a
route to transfer; the alignment itself is confirmed)

### 31. Identification and characterisation are different questions
Also forced rather than proposed. GREN gated a `GamePackage` on one confidence
number, and handing packages to CyclicCortex made the conflation visible: sudoku
scored **0.350** and was refused.

Sudoku scores badly because it sits 0.80 away from everything GREN has seen. That
is *confidently novel*, not uncertain — and a novel game is precisely the one
that should found its own region. Gating placement on identification confidence
refuses a correctly understood game for the crime of being new.

So the number splits in two. **Identification** — which known game is this —
falls when nothing is close. **Characterisation** — do we know what this game is
like — rises with probe coverage and a settled refusal taxonomy, and is the
right gate for placement. Sudoku: 0.350 and 0.977 — and go, at 0.421, would have
been refused too.

A confidence measure that answers two questions at once will be wrong about one
of them, and the integration is what exposed which.

*Where:* `GREN/gren/package.py` (`placeable`), `GREN/gren/axis.py` (`settled`),
`CyclicCortex/cortex/discovered.py` — on branch `feat/gren-impl` · **refined**

### 32. A game modifier is a game-IDENTITY signal, and identity is the opposite of transfer
`GREN/DESIGN.md` §22.5 argues that hashing a game's *mechanic set* into the
micro's input makes transfer automatic: hashing a set preserves overlap, so two
similar games land at nearby points and "a micro that learned 'a piece cannot
move through an occupant' generalises to every game with a similar modifier
without being told to".

**The first half is confirmed exactly.** Modifier cosine tracks the Ochiai
similarity of the mechanic sets with mean absolute error 0.102 at 64 dimensions,
**0.031 at 128** and 0.017 at 256 — confirming the spec's choice of 128 (its own
table, on synthetic pairs, reads 0.128, 0.061 and 0.037), and putting chess
nearest checkers and sudoku furthest, which is the same ordering GREN found by
probing and CyclicCortex uses to build its regions. Three projects now agree on
what a game *is*, through one JSON file rather than three hand-written
frozensets.

**The second half is contradicted.** Train on chess, evaluate checkers cold, six
seeds: with the modifier on, 0.503; with the modifier block zeroed, **0.566**
(measured with the CLI's default 64-dimensional modifier, not the 128 the spec
chose). The modifier is worth **−0.062**, negative in 5 of 6 seeds. One seed had
shown +0.117 and that was noise.

The reading that fits: the modifier tells the population *which game this is*, and
game identity is precisely what lets a population specialise per game — the
opposite of carrying a response across. Similarity in modifier space is real; it
is just dominated by the identity signal sitting in the same bits.

This is the third independent measurement in this repository putting
chess→checkers transfer at approximately nothing: CyclicCortex's shared
vocabulary (+0.023 against its proper baseline), its cross-region ensemble
(+0.025), and now this. Three different mechanisms, three different codebases,
one answer.

*Where:* `GTMNN/gtmnn/games.py`, `GTMNN/README.md` — on branch `feat/gren-impl`;
`GREN/DESIGN.md` §22.5 · **refined** (the hash works; the transfer claim does
not)

### 33. A training signal that contains the answer is not a loss
Not an insight of Curtis's — a mistake I made and had to measure my way out of,
recorded so it is not repeated.

GTMNN's epoch record reports `loss` as mean `−log P(y)`, per `DESIGN.md` §16.2,
where `P` is the aggregate of the training-game equilibrium. But that game is
`CorrectnessCongestion`, and **its payoff function contains `y`** — every seated
micro is paid `B/n_a` for naming the answer it was just shown. The resulting
number looks excellent and measures almost nothing: `loss` **0.59** while the
population's actual predictive loss was **4.2**, against a uniform baseline of
**2.77**. I read the first as the second for several rounds.

It is not a useless quantity — it is the right partner for `shapley_total`, since
the two are views of `v(N) − v(∅)`. It is just not a prediction. The fix is to
report both, side by side, so the contaminated one can never be mistaken for the
honest one again.

The general form: **when a metric is computed from a mechanism that was given the
label, it measures the mechanism's compliance, not the model's knowledge.** The
control that catches it is cheap — withhold the label and recompute.

*Where:* `GTMNN/gtmnn/model.py` (`belief_loss`), `GTMNN/README.md` — on branch
`feat/gren-impl` · **confirmed**

### 34. A trie over what is PROVABLE, not over what a thing is
The radix tree in GREN indexes games by what they *refuse*. The same mechanic one
layer up indexes them by their game-theoretic structure — and there the levels
have a meaning the refusal tree does not have: **descending is accumulating
premises.**

The root does nothing. It asserts no property and guarantees only Nash 1950, true
of every finite game. Each level adds one fact, ordered so that every step
strictly narrows, and a node's guarantees are then a function of its path and
nothing else. The theorems fire at different depths: von Neumann at depth 2
(`players=2`, `payoff=opposed`), Milchtaich at depth 4 the moment monotone load
is asserted, Rosenthal only at depth 5 with the whole congestion prefix. **The
path is the proof**, which is why it is a trie and not a lookup table — an
internal node is a real answer you can act on before reaching a leaf.

Two things follow that a diagram would not have given:

* **The recommendation is falsifiable, and mostly holds.** Running all five
  solvers on every class, the trie's pick measures best in **5 of 6**. The
  exception is instructive rather than embarrassing: on the player-specific
  congestion game the trie says `regret_plus` because Milchtaich warns a
  best-response path *may* cycle, and `best_response` measures better. Both are
  right — one is a worst-case guarantee, the other an average case. Hunting the
  cycle across seven geometries and 1 400 games **found none**. The trie's job is
  to say which choice is provably safe, not to predict the mean.
* **The solver stopped being a configuration option.** `solver="auto"` classifies
  the payoff in hand and takes the deepest rule whose premises the path satisfies.
  Measured: 1.8× faster at a loss inside the seed spread. A new payoff now gets
  the right solver by answering the same questions, rather than by someone
  remembering to add a branch.

The general form: **an index whose levels are premises turns a classification
into a derivation.** GREN's tree answers "which game is this"; this one answers
"what am I allowed to assume", and the second is the one that changes code.

**A network at every node.** Reaching a node is how you find the population to
query, and a cold leaf backs off to the nearest trained ancestor — which is what
the terminal-flag-on-internal-nodes design is FOR. "A congestion game with a
common payoff" is a usable answer while the specific class below it is still
cold. The routing is verified in every seed: `resolve` takes the deepest trained
node, backs off when the leaf is cold, stops the moment it has played. What
backoff is *worth* is **+0.006 over 5 seeds with a ±0.15 swing** — nothing. One
seed had shown +0.22 and it was noise, the same trap as 32. The class it backs
off to has barely learned either, so this is downstream of insight 2 and is worth
re-measuring only once that is fixed.

**Scope, deliberately narrow.** A trie here indexes games and nothing else: GREN's
by what a game refuses, this one by its structure. Neither carries activations or
stands in for a network. The moment one does, it stops being an index and the
guarantees stop meaning anything.

*Where:* `GTMNN/gtmnn/trie.py`, `GTMNN/gtmnn/tournament.py`, `GTMNN/README.md` —
on branch `feat/gren-impl` · **confirmed** (the index and the solver derivation;
the backoff benefit is not yet measurable, and was not re-measured once 2 was
fixed)

### 35. Measure the mixed profile, not its argmax
Another mistake caught by its own control, recorded so it is not repeated.

Exploitability — max over players of (best-response utility − realised utility) —
is the honest convergence measure, zero exactly at Nash. I computed it on the
profile's **argmax**, which is meaningless wherever the equilibrium is mixed. On
rock-paper-scissors *every* pure profile is exploitable (1.0 when the two match,
2.0 when they do not), so the measure scored all five solvers identically at
their worst and hid the thing that matters: the uniform mixture is exactly
unexploitable, and the solvers differ enormously in how close they get to it
(fictitious 0.19, best response 2.00).

The tell was a column of identical bad numbers. A metric that cannot separate
five methods on the one game built specifically to separate them is measuring
the wrong object.

*Where:* `GTMNN/gtmnn/equilibrium.py` (`mixed_exploitability`) — on branch
`feat/gren-impl` · **confirmed**

### 36. Memory is the trajectory, not the answer
> Solve memory by keeping track of the trajectory through the model, input and
> output data, then storing this in a new NN that is a stack-based idea where the
> top of the stack has precedence compared to similar paths further down.

Every network here discards the most informative thing it produces. GREN walks a
radix path to identify a game and keeps the answer. CyclicCortex routes to a
region and keeps the prediction. GTMNN seats 64 micros of 4096, solves an
equilibrium, and throws away *who was seated and what they argued*. The
trajectory is that discarded object — the path taken, not the destination — and
the claim is that it is what memory should be keyed on.

**The precedence order is the sharp part.** GREN's radix tree resolves a tie by
SPECIFICITY: a deeper terminal beats a shallower one, because "chess" beats "a
two-player board game" when both match. A stack resolves it by RECENCY: the top
beats everything below, whatever its specificity. Those are two different orders
over the same set of matching paths and they disagree — the specific-but-stale
entry and the general-but-fresh one are both real answers. Which should win is
empirical, and both policies can run over the same stack, so it is settleable.

A stack also buys something no weight-based memory has: **shadowing without
erasure**, exactly like a scope chain. Popping restores what was underneath, so
memory gets an undo.

Two existing threads meet here. `CyclicCortex::rehearse` exists because admitting
a new game overwrites the incumbents, and replay is the mitigation — a trajectory
store attacks the same problem from the other side and `rehearse` is the measured
baseline to beat. And insight 22's dual-network/REM consolidation has never had a
rule for *what* to consolidate; a stack supplies one, in that entries which keep
getting shadowed are what the base network should absorb, and entries that keep
being hit alone are what it still cannot represent.

**The premise is falsifiable and cheap to falsify, so that goes first.** If a
trajectory-keyed lookup does not beat an ordinary input-keyed one on the same
episodes, the trajectory carries no structure the input lacks and the whole
design reduces to a cache with extra steps. Given that "similar things
generalise" has now failed three independent measurements in this repository
(29, 30, 32), the experiment should be built to fail loudly.

Unresolved and load-bearing: what a trajectory IS across three networks with
three different path alphabets; what "similar" means for two paths (the phrase
"similar paths further down" is doing enormous work); one global stack or a stack
per prefix; and growth, since a stack that only grows is a log rather than a
memory. Also a scope collision — a path-keyed store is trie-shaped, and a trie
here is for game theory and game classification only. Either this uses a
different structure or that rule needs a stated exception; it should not quietly
become a third trie.

*Where:* `Memory/README.md` — a specified, unbuilt design on branch
`feat/gren-impl` · **held**

### 37. Index a game by its END GOAL, and work backwards
> The Radix Tree for game solving should be about the end goal. We focus on the
> end goal of the game, and work backwards. For chess, it's to topple the king,
> for a conversation at work, it's determined by the conversation itself and can
> be a bit more ambiguous, however, we usually have an end goal in mind for the
> conversation (to learn more, get a promotion, etc...). If the goal is unclear,
> go up a level.

This corrects a proposal of mine. Asked how to make the trie more specific, I
offered more game-theoretic structure — information, horizon, dominance,
supermodularity. Every one of those is a real axis and every one is **the wrong
kind**: they require already knowing the game. You cannot state a conversation's
horizon or potential function before having it. Measured, that is exactly what
went wrong — a conversation with *every* structural fact supplied still stalled
at depth 1 of 9, because no branch existed for it.

The goal is the one thing you can state about an unfamiliar game **before you can
play it**. Same conversation, indexed by goal instead: depth 3 of 3, fully
placed, sitting next to the other conversations rather than nowhere.

**Going up a level is the semantics, not a fallback.** Each level is a usable
answer in its own right:

| what I know | depth | who it sits with |
|---|---|---|
| "get a promotion" | 3 | the 1:1 with my manager |
| "change how they see me" | 2 | every conversation with that shape |
| "talk to my manager" | 1 | everything |

"Change how they see me" is a real goal with real strategies even when "get a
promotion" is not yet committed to. That is the same trie mechanic as GREN's
terminal-on-internal-node rule, arrived at from the other direction: a general
class is a strict prefix of a specific one and is answerable on its own.

Two arguments for the ordering, and the second is the load-bearing one:

* **Compression.** `gren/cli.py` inserted `sorted(signature())`, so the tree's
  root discriminated on `adversarial=` for no better reason than that "a" sorts
  first — which separates nothing among four board-ish games. Goal-first
  identifies a game from **one** token against alphabetical's **2.5**, and the
  root branches four ways instead of two. *Small N*: four games and one
  perfectly-discriminating axis makes this nearly free, and `goal_type` would not
  stay a unique key at four hundred games.
* **Knowability.** Putting the goal at the root means the questions the tree asks
  first are the ones a new game can actually answer. This does not depend on the
  corpus size, and it is why the first argument is the weaker one.

"Work backwards" is also the traversal, not just the layout: from the goal, to
what must be true for it, to what must be true for *that*. Chess endgame
tablebases are built exactly this way, backwards from mate. It is goal regression,
and it is the natural home for `GREN/DESIGN.md` §8.3's STRIPS operators, still
unimplemented.

One honest limit: chess and checkers still share 2 of 3 goal axes
(`against=opponent`, `decided_by=terminal-state`) and diverge only at the leaf —
`reach-target` against `outlast`. The goal axis separates them where mechanic
similarity does not separate them at all, which is the right direction given that
chess→checkers transfer has measured at approximately nothing three ways
(29, 30, 32), but it is a modest separation rather than a dramatic one.

The conversation's placements — depth 1 of 9 by structure, 3 of 3 by goal — and
the goal ladder are recorded in the commit that built this (31e6ad2), not in a
committed script; what the code commits is `goal_first()`, which puts
`goal_type` first in a flat signature. Putting the goal at the root is safe only
because similarity is computed over sets (11, 28): `GREN/DESIGN.md` §9.3 names
splitting on `goal_type` first as the pathology a single tree suffers.

*Where:* `GREN/gren/radix.py` (`goal_first`), `GREN/gren/cli.py` — on branch
`feat/gren-impl` · **confirmed** (the ordering; the conversation measurements
are recorded in a commit message only)

### 38. A refusal code is a precondition violation
The follow-through on 37. If the tree is organised by the end goal, the way to
use it is to work backwards — and goal regression is the classical method for
that: ask what would have to be true for the goal to hold, and keep asking until
the answer is already true.

Regression's usual cost is writing every operator's preconditions by hand, and
that hand is where domain knowledge smuggles itself in. GREN does not need them
written. `BLOCKED_PATH` means `move` requires a clear path; `CONSTRAINT_ROW`
means `place` requires the value absent from the row. **`why()` already answers
"what blocks this action?", which is exactly the question regression asks at
every step.** The preconditions are measured, not authored, and `DESIGN.md`
§8.3's STRIPS operators finally have a source.

On sudoku this reproduces constraint propagation from nothing but the refusal
codes: rank cells by how few values clear all four preconditions, and a cell with
one clearing value is forced. Nobody coded "naked single". Search nodes fall
27×, 143× and **1 240×** at 30, 26 and 22 clues — though by raw work it barely
wins on easy puzzles, because scanning 51 cells to save a few nodes is not worth
it when almost any order solves them.

**The law, and it is quantitative: a subgoal prunes exactly what it excludes.**
On chess, `mate ⟸ check ∧ no-escape`, where the expensive conjunct enumerates
every opponent reply and the cheap one is a single `attacked()` call. In positions
from real play 4.3% of moves give check, 95.7% of the expensive tests are avoided,
and it runs 7.7× faster. With pieces scattered at random 59.5% give check, 48.7%
is avoided, and it runs 2.2× faster. Regression pays in proportion to how
selective the cheap subgoal is, and nothing else.

*Where:* `GREN/gren/regress.py`, `GREN/README.md` — on branch `feat/gren-impl` ·
**confirmed**

### 39. Check whether the baseline already does it
The first measurement of 38 came out at **exactly 1.0×**, which is the number
that should always be suspicious.

The control had regressed all along: `is_mate()` is `in_check() and not
legal_moves()`, and Python's `and` short-circuits. **Goal regression over a
conjunctive goal IS a short-circuiting `and` with the cheap, selective conjunct
written first.** The technique was already in the baseline, written by whoever
chose that conjunct order, probably without thinking of it as planning.

Two things follow. A classical technique can be present in ordinary code under
another name, so "does the baseline already do this?" is a question to ask before
building the thing, not after measuring it. And 1.0× is diagnostic: a real
improvement rarely lands on exactly parity, so that number usually means the two
arms are the same arm.

It is not the only time in this repository a control has decided an outcome —
the untrained-network baseline that inflated CyclicCortex's transfer twelve-fold
(29), the argmax exploitability that scored five solvers identically (35), and
now this. In each the mistake was in the comparison rather than in the thing
being compared, and the method entries at the end of this index collect the
rest.

*Where:* `GREN/gren/regress.py` (`conjunctive`, `expected_work`) — on branch
`feat/gren-impl` · **confirmed**

### 40. A rule transfers exactly when the feature that carries it means the same thing
> Expect to learn valid moves, the rules, and underlying mechanics of a game as
> well.

Three levels, and "similar" was only ever being measured at the third. **Valid
moves** is one bit per candidate. A **rule** is which refusal fires and under
what condition. **Mechanics** are the decomposed parts — GREN's signature — and
every similarity measure in this repository is built on them. Asked to go back
to basics on identification and grouping, the first thing to do was measure
transfer at all three levels, for all six pairs, and hold the measures to it.

**Level 1, GTMNN legality, twelve directed pairs: flat.** Every value within
±0.09 and inside its own seed spread. It agrees with the three earlier
chess→checkers numbers and adds nothing on its own.

**Level 1 on CyclicCortex — the harness that learns well — against the majority
baseline, each pair forced into one region:** go/sudoku **+0.341**, chess/checkers
**+0.081**, every other pair 0.000 for want of a rule-bearing channel. The same
ranking as level 2, the same +0.30 / +0.07 agreement, from an independent
harness. And it is asymmetric: checkers→chess +0.139, chess→checkers +0.023 — the
simpler game's rules carry into the richer one, not back.

**Level 2 is the result.** Train a growing code-predictor on game A's
(shared-vocabulary features → refusal code), test it per code on game B:

| pair | rule-bearing shared slots | rule transfer | rank by every feature measure | rank by rule transfer |
|---|---|---|---|---|
| chess / checkers | 7 | 0.32 | **1st** | 2nd |
| go / sudoku | 3 | **0.98** | 4th | **1st** |
| the other four | 0 | — | | |

Every feature-based measure — GREN's signature, the hand-written mechanics, the
GTMNN modifier — calls chess/checkers the *most* similar pair and go/sudoku among
the *least*. Rule transfer says the opposite, by a mile. Agreement with rule
transfer is **+0.30** for all three mechanic measures and **+0.07** for the goal
tree: they capture "cross-family is far", which is a tautology once a shared
feature is required, and get the top of the ranking backwards.

The mechanism, exactly. **`go→sudoku`** transfers the one rule they share
through `GRID_PLACE[2]` — "target cell empty", the feature the vocabulary
alignment had already matched with the same orientation — at LEGAL 1.00 and
OCCUPIED_TARGET 0.99. On sudoku's *constraint* refusals, which fire on an empty
cell, it says LEGAL every time: correct by its own lights, since go has no
concept of a value clash. **`chess→checkers`** calls **95% of checkers' legal
moves `BLOCKED_PATH`** — a code checkers does not have. A checkers jump is a
two-square diagonal with the midpoint occupied by the piece being taken. To a
chess-trained net an occupied midpoint on a diagonal is a blocked bishop. Same
feature, *opposite* rule: in checkers the occupied midpoint is required, in chess
it is forbidden. Insight 30 measured at the level of a single rule, on a second
rule pair.

**Another instance of the prior trap (33, 35, 39).** `sudoku→chess` scored
OCCUPIED_TARGET 1.00 with every other class at 0.00 — and the two share no
feature at all. A net given a zero input emits its bias. `SIDE`, whose turn it
is, is one shared slot with no rule content and produced the same artefact at
83%. The guard: a shared surface that can *carry* a rule, or the score is a
prior. Applying it moved the three mechanic measures' agreement from a flattering
+0.60 to +0.30.

**So: identify by mechanics, group by rules.** Identification is solved (GREN,
one token, goal-first). Grouping by shared mechanic *names* measurably groups the
wrong pairs, because a name does not carry its condition. Two ways to group
right: the operational measure — train on A, test per rule on B — which cannot
be wrong about the thing it is for and is now `gren.cli transfer`; or a
decomposition deep enough that a shared name *implies* a shared condition, which
is §8.3's minimality rule, still unbuilt, and which now has a concrete
acceptance test — after splitting `OCCUPIED_TARGET` and `BLOCKED_PATH` into their
condition-level parts, the feature ranking should put go/sudoku first.

The level-2 figures are committed and pinned by tests; the level-1 figures, the
agreement scores and the `SIDE` artefact are recorded only in the commit that
measured them (55bd8f4) — `gren.cli transfer` prints per-code recall and
nothing else.

*Where:* `GREN/gren/cli.py` (`transfer`), `GREN/README.md`,
`GREN/tests/test_gren.py` — on branch `feat/gren-impl` · **confirmed** (and 11's
grouping claim **contradicted** as stated)

---

### 41. Map a new game blind: refusals name the rules, acceptances are the scarce resource
> What is an automated way to map out a new game? I would like to try failing at
> the game as much as possible to learn the game's rules/inputs/outputs.

**First, an honest finding about the existing explorer: it is not blind.** Every
`candidates()` in `cortex/games.py` returns `good[:k] + bad[:k]` — half legal by
construction, chosen with `is_legal`. The Boundary and EIG policies perturb
`oracle.actions()`, which is the legal move list. `random_state` plays random
legal moves. So GREN's measured 50% failure rate is partly handed to it by its
candidate pool, and three rules are out of its reach by construction: sudoku's
OFF_BOARD and OUT_OF_RANGE (the pool never leaves the grid) and go's REPETITION
(`random_go` builds boards with no previous position, so ko can never fire). A
new game gives none of this. It gives a grammar — what a submission looks like —
a yes or a no, and the next state when it says yes.

**The blind loop, as measured** (scratch `blind.py`; 4 games × 5 policies × 5
seeds at 1200 and 3000 probes): sample the grammar one step past its edges;
submit; remember every acceptance and replay it against the flood of refusals;
choose the next probe from grammar samples and one-step edits of remembered
acceptances by a learned p(accept | features); take the accepted move half the
time, restart after 40 moves or 100 straight refusals; name the rules by
grouping refusals on the feature the model blames (input × gradient); read the
mechanics off the state diff of every acceptance. The rules are consulted by the
scorer only, after the fact, to say what the explorer tripped.

**"Fail as much as possible" is right about the mechanism and wrong about the
objective.** The mechanism holds: random grammar samples fail 99.8% of the time
in chess, and that flood names six of the seven refusal kinds within 166 probes —
every failure is a rule, and the rules are cheap. But the policy that
*maximises* failure is the worst mapper on every level, in every game:

| game, 3000 probes | policy | failure | accepted | rules found | mechanics found | legality acc. |
|---|---|---|---|---|---|---|
| chess (7 rules) | GREN eig, rules in hand | 0.46 | — | 7.0 | — | — |
| | random | 0.998 | 5 | 6.0 | 1.0 | — |
| | maximise failure | 0.996 | 13 | 6.0 | 1.0 | 0.50 |
| | aim for p = ½ | 0.940 | 179 | **7.0** (SELF_CHECK 5 of 5) | 2.0 | 0.57 |
| go (4 rules) | GREN eig, rules in hand | 0.15 | — | 2.8 | — | — |
| | random | 0.49 | 1523 | 3.0 | 3.0 | — |
| | maximise failure | 0.94 | 190 | 2.2 | 2.8 | 1.00 |
| | aim for p = ½ | 0.51 | 1464 | **4.0** (5 of 5) | 4.0 | 1.00 |

Maximising failure finds go's two easy refusals and then trips them for the
remaining 2,900 probes: 93% failure, half the rules, and a rule-naming purity
gap over chance of +0.06 where the p = ½ policy's is +0.52. On sudoku it finds
all six rules but takes 732 probes to the last one against 55. Until the first
acceptance the learned policies are all the same random policy — a model with
one label cannot rank — which is why they find chess's first six rules at the
same probe; the rules that separate policies are the ones behind the boundary.
The quantity to maximise is not failures per probe but *new information* per
probe, and after the first hundred probes the informative submissions are the
ones the model cannot call: insight 6's p = ½, now measured on a blind oracle.

**Where the grammar is sparse, the bottleneck is the inputs, and only
acceptances buy them.** In go and sudoku the grammar is dense (half of go's
submissions are accepted) and every blind policy learns the legal set to 0.99+
balanced accuracy. In chess about 0.2% of the grammar is legal and in checkers
0.07%: random probing accepts 5 chess moves in 3000 and 2 checkers moves. A
model trained on that stream learns to say no — balanced accuracy 0.50, a
majority classifier. Replaying one remembered acceptance per refusal lifts it to
0.62–0.65 in chess and 0.66–0.76 in checkers at 1200 probes; the p = ½ policy,
with 36× the acceptances of random, is what reaches SELF_CHECK — a rule that
lives in the midgame, which no policy that cannot get to the midgame will ever
see. Checkers is the negative: at most three acceptances per run under every
blind policy, no jump ever performed, WRONG_DIRECTION never seen and
FORCED_ALTERNATIVE once in 65 runs. You cannot learn the rules of the midgame
if you cannot reach it, and reaching it is paid for in acceptances, not
failures.

**The rules can be named from yes/no alone — up to the feature language.** Group
each refusal by the present feature the legality model blames hardest, and
compare the groups with the true codes, which the explorer never saw (1200
probes, p = ½ policy):

| game | purity | chance | rules named / found |
|---|---|---|---|
| sudoku | 0.83 | 0.32 | 5.0 / 6.0 |
| go | 0.99 | 0.47 | 3.0 / 3.8 |
| checkers | 0.87 | 0.43 | 4.7 / 5.6 |
| chess | 0.77 | 0.32 | 5.4 / 6.0 |

Sudoku's map reads like the rulebook: `filled → OCCUPIED_TARGET` 1.00,
`in_row → CONSTRAINT_ROW` 0.88, `in_col → CONSTRAINT_COL` 0.80, `in_box →
CONSTRAINT_BOX` 0.88. The misses are the feature language's, not the method's,
exactly as §14 specifies: OFF_BOARD and OUT_OF_RANGE share the one `range`
feature, so they merge; chess's NOT_YOURS is the *absence* of "mine" and of
"empty", and a rule carried by no present feature cannot be blamed on one, so
its refusals spill into the geometry groups. Where the oracle and the blind
namer disagree — checkers' "target occupied" group is 52% EMPTY_SOURCE — both
are right: `why()` reports the first violated precondition in its own order,
the model reports the strongest, and a probe with an empty source and an
occupied target violates both. This is insight 38 from the other side: a
refusal is a precondition violation, and a blind explorer recovers the
precondition.

**Outputs are read off the state diff, and playing forward is what finds them.**
Go: place; place and take one; place and take several; pass (no cell changes,
the turn passes) — four mechanics, all found. Chess: move, and capture, the
second only under the learned policies that reach it. Checkers: the plain step
and nothing else, for the same reason it has no midgame. Sudoku: fill, and the
turn does not pass.

**So the automated way is:** grammar in, yes/no out. Fail freely at first,
because every refusal is a rule. Then aim at the boundary, not past it. Treat
each acceptance as the scarce resource — remember it, replay it, play it
forward — because the legal set, the midgame rules and every mechanic are paid
for in acceptances. Name rules by what the model blames; read mechanics from the
diff; stop when no new rule group or delta kind has appeared for a while
(`Evidence.settled`). Every piece of this exists in GREN except the blindness.

**Traceability.** `blind.py` was a scratch script and was never committed on any
branch, so the tables above are recorded only here and in the commit that
recorded them (8ab11c9). The claims about the existing explorer are checked
against committed code — every `candidates()` is half legal, and `random_go`
can never fire ko — but the measurements stand unreproducible until the script
and its output are committed (What to test next 10).

*Where:* scratch `blind.py` (never committed; its results in commit 8ab11c9 on
branch `feat/gren-impl`), `GREN/gren/probe.py`, `CyclicCortex/cortex/games.py` ·
**confirmed** (by an uncommitted script); 5 **confirmed** and 6 **refined** by it

---

### 42. Growth is an identity — except for outputs, and at initialisation
> Inputs and outputs grow to match the games.

*Confirmed, with two corrections the obvious version gets wrong.* Hidden units
entering with zero outgoing weight, and inputs with zero incoming weight, are
exact identities — CyclicCortex asserts bit-identical output over **300** inputs
— and GREN's network starts at **0 inputs and 0 outputs** and grows to **48 and
16** across four games: an input when a feature name first appears, an output
when a refusal code is first seen. It found the two cases that are not
identities. **Outputs:** zero weight and zero bias hand a new softmax class
`exp(0)` of the mass, and a large *fixed* negative bias fails too, because
trained logits can sit below any constant (**0.244** of the distribution moved);
what works is a bias **40 below the largest existing one**, arriving at a share
of ~1e-18 and still learnable afterwards (0.967). **Initialisation:** identity
growth from nothing symmetry-locks the network — `tanh(0) = 0` zeroes every
update, and the loss sat at **0.7003** forever on a trivially separable task —
so rows created during warm-up must be random. The identity covers the instant
of growth, not the training after it: admitting checkers moved chess's
top-choice legality **0.933 → 0.811**, and rehearsal's benefit was not
detectable at this scale.

*Where:* `GREN/README.md` ("Self-Building Neural Networks"),
`GREN/gren/sbnn.py`, `CyclicCortex/README.md`, `CyclicCortex/cortex/sbnn.py`,
`CyclicCortex/DESIGN.md` §8 · **confirmed**

### 43. The vocabulary sets the ceiling — for legality, for play, and for a teacher
*Confirmed three times, from three directions: a network cannot learn what its
inputs cannot say.* The common-denominator result (29) says the minimal shared
vocabulary transfers best; CyclicCortex measured the other edge — minimal is not
sufficient. **Legality:** checkers sat at **0.782** until the features included
the square *between* from and to, since a jump is legal only if an enemy sits
there; with the midpoint checkers rose to **0.906**, and chess, sharing the
region, to **0.921**. **Play:** go's features were sufficient for legality
(**1.000**) and structurally incapable of expressing territory, so it lost every
game; influence, connection and an outcome-trained grade took it from **0–6** to
**3–3** (45). **A teacher:** distilling Stockfish's move
scores produced no correlation with them — **ρ = −0.056** from 200 positions and
**−0.086** from 800, against **0.133** for the hand-written heuristic — because
Stockfish evaluates the whole board and the student sees about 27 features of
one move. *A teacher cannot teach what the student's vocabulary cannot express.*
The universal game encoder met the same ceiling as a one-token memory
(71).

*Where:* `CyclicCortex/README.md`, `CyclicCortex/information.py`,
`CyclicCortex/cortex/stockfish.py` · **confirmed**

### 44. Legal and good are different questions — put legality on its own wire
> I failed on purpose at games to understand the rules, and then figured out the
> rest very quickly once I knew the rules.

*Confirmed — the split in 19 is load-bearing, and every time it was blurred
something broke.* A go region trained only for validity played **125**-ply games
without one illegal move and lost **0–81**. Each conflation was then measured:
distilling a teacher trained `valid = 1.0` on every scored move — and a teacher
scores only legal ones — so legality fell **0.925 → 0.748**; rule selection that
maximised legal top picks chose the validity-only rule at a perfect **1.000**,
discarded the grade head, and played the worst material of any configuration;
and a grade head rewarded for any two-square checkers move learned that
jump-*shaped* moves are good. Chess put the separation on its own wire. Its
network proposes one of 4,096 from–to pairs and at first learned legality only
from the refusals it happened to make, inside a softmax whose target was move
*quality* — 400 rounds to get from 660 refusals a move to **41.8**. But every
position answers all 4,096 legality questions at once, exactly and for free, so
seven heads on one trunk — legal, pseudo-legal, capture, check and tactical —
were trained on those labels, with the illegal moves the network currently ranks
highest as hard negatives: refusals fell **×89** for 2NRL (97.9 → 1.1) and
**×46** for the positive control (41.8 → 0.9), and legal AUC reached 0.95–0.98
in every arm. These are the rules, not the game, and there is no single right
way to recombine the two: `p_valid` alone is best for chess and go and worst for
checkers, because forced capture makes legality a property of the move *set*.

*Where:* `CyclicCortex/README.md`, `CyclicCortex/cortex/cortex.py`,
`TwoNRL_Chess/README.md` ("Forcing the rules"), `TwoNRL_Chess/rules.py`,
`GREN/DESIGN.md` §22.6 · **confirmed**

### 45. Outcome credit must be scaled to the game's length
*Confirmed — three things were each necessary, and none sufficient alone.*
Learning grade from outcomes — play, wait for the result, credit every move with
the discounted outcome — was measured on go against the engine over six games:
richer features alone, **0–6**; outcome credit alone, **0–6** (mean gap −81.0);
features, outcome credit **and** a discount matched to the game, **3–3**
(−14.3). The discount is the non-obvious one: at γ = **0.95** a **125**-ply game
gives its opening move `0.95¹²⁵ ≈` **0.0017** of the outcome, so the credit has
vanished before it reaches the moves that decided the game. No constant serves
games of different lengths, so γ is derived per game as `0.5^(1/plies)` — the
first move keeps half the credit of the last. An outcome signal also needs an
outcome: fifty self-play chess games once produced fifty draws at the ply cap,
and so zero gradient; each game now supplies a bounded zero-sum `value(state,
side)` for when no winner exists.

*Where:* `CyclicCortex/README.md` ("Grade, learned from outcomes"),
`CyclicCortex/cortex/cortex.py` (`selfplay`) · **confirmed**

### 46. Sample where the rule lives — a training distribution teaches its own prior
*Confirmed, repeatedly: what a network learns is set by where its examples come
from.* Chess candidates in CyclicCortex were only **36%** legal, and training on
that distribution taught the prior rather than the rule; balancing the candidate
sets gained **+0.037** (0.819 → 0.856). **99 of 200** randomly generated chess
positions were unreachable — the side not to move already in check, a king
missing — so half the supervised signal came from boards no game can produce.
Random illegal moves are dominated by trivially illegal ones, so forced capture,
suicide and ko almost never decide a sample and need targeted hard negatives.
Evaluation and play are different distributions too: play reaches the positions
the network's own weak moves lead to, and mixing self-play positions in cut
chess's illegal rate in one game from **0.70** to **0.06**, while a ranking rule
selected on the training mixture scored **0.425** in evaluation and **0.04** in
play. The same trap sits under GREN's failure rate, whose candidate pool is half
legal by construction (6, 41).

*Where:* `CyclicCortex/README.md`, `CyclicCortex/cortex/stockfish.py`
(`playable`), `CyclicCortex/cortex/games.py`; the play figures in commits
a8101bf and 77857b7 · **confirmed**

### 47. With regions as players, Shapley is exact — and it says the graph routes rather than teaches
*Confirmed — the benefit claimed for trading micros for regions, and an answer
to what regions are for.* GTMNN must estimate Shapley by Monte-Carlo, because
`2^4096` coalitions cannot be enumerated; a dozen regions makes `2^12 = 4096`
coalitions a loop, so CyclicCortex computes credit exactly and samples only
above that. Over three regions and four games efficiency holds to machine
precision (**0.0** to **2.2e-16**), and the values are semantically right: each
game's own region takes the largest share (chess **+0.473**, checkers
**+0.487**, go **+0.599**, sudoku **+1.071**); a wrong region scores
**negative** (the go region costs sudoku **−0.443**); and the sudoku region
scores exactly **−0.000** on chess, where its vocabulary coverage is exactly
**0.00** — the null-player axiom appearing in measured data rather than
asserted. What the credit does not show is regions teaching each other:
ensembling every region into a game's prediction helps one game of four,
checkers **0.905 → 0.930**, and inside a region eight shared slots give
**+0.023** (29). *The graph routes; it does not teach* — regions earn their
place by placing games correctly, and a wrong placement measurably hurts.

*Where:* `CyclicCortex/README.md` ("Regions as players", "Cross-region
transfer"), `CyclicCortex/cortex/credit.py`, `CyclicCortex/cortex/routing.py`,
`CyclicCortex/DESIGN.md` §9.2, §15.1 · **confirmed** (as routing; regions
teaching each other measured at about nothing)

### 48. The theorems hold in code — and the cycle Milchtaich warns of never came
*Confirmed as code, and the worst case the design prepares for did not appear.*
GTMNN rests on game structure buying guarantees, and each is now a numerical
test rather than a citation. **Termination:** the training game is an exact
potential game, so best response must terminate — **200** random games, **0**
unconverged, **0** cycles, with the solver asserting that the potential rose on
every accepted deviation (Rosenthal). **Truthful bidding:** seats sell at the
highest losing bid, so truth must be dominant, and over a bid sweep **no
misreport beats it** (Vickrey) — calibration really is a mechanism rather than a
loss term. **Cycle detection:** rock-paper-scissors reports period **3** per
sweep and **6** when pushed per move, which is why the detector pushes once per
completed sweep — a cycle measured at the wrong granularity has the wrong
length. The inference game is player-specific, and Milchtaich's warning that a
best-response path *may* cycle is the basis of 25; hunting for the cycle across
seven geometries and **1,400** games found **none**, and best response converged
every time, **20×** faster than the regret solver the trie recommends (34).

*Where:* `GTMNN/DESIGN.md` §2.1; `GTMNN/README.md` ("What holds"),
`GTMNN/gtmnn/equilibrium.py`, `GTMNN/tests/` — on branch `feat/gren-impl` ·
**confirmed**

### 49. Three jobs, three measures — rarity is for probing, not for similarity
*Confirmed numerically: weighting by rarity makes similarity worse, because
similar games share the common parts.* IDF weighting looked like the natural
refinement of Jaccard over mechanic sets, and it lowers every similarity —
chess/checkers from **0.39** to **0.28**. What those two share is the common
mechanics, a grid board and alternating turns, which have low IDF; what
separates them is the rare ones, castling and en passant, which have high IDF.
So IDF amplifies idiosyncratic detail at the expense of structural agreement:
chess and checkers are alike *despite* castling. Three jobs want three measures,
and conflating them was the mistake: **similarity** is plain Jaccard (9);
**hierarchy** is containment, `|A∩B| / min(|A|, |B|)`, asymmetric as
generalisation is (13); **discrimination** — which mechanic to probe next — is
IDF, because a rare mechanic splits a candidate set best. Sizing the game
modifier met the same fact from the other side: giving the commonest mechanics
dedicated, collision-free bits **lost** at equal width (**0.087** against plain
hashing's **0.061**).

*Where:* `GREN/DESIGN.md` §8.6, §22.5 · **confirmed**

---

## The papers, measured

### 50. What the field routes around is carrying information
*Held as a pattern — its instances pay very differently.* The four papers in
`Research/` make one move: something the field routes around — the vanishing
gradient, cycles, a non-monotone activation, failure — turns out to carry
information, and the usual fix deletes the signal instead of reading it.
Measured, the signal is real every time; reading it pays unevenly. The decay is
a clean reading of depth (R² **0.997** at depth 12) and loses to Adam, which
reads the same scale empirically (51). The loop that makes the
graph finite also makes it generalise — **4.057** held-out bits/char against an
acyclic tree's **6.514** — though a tree bounded to the graph's order does as
well (52). The sine's liveness is real and its accuracy is best
on no target tested (57). And a trained failure inverts exactly into
its mirror image (61), yet at matched compute 2NRL loses to its
own control on chess and inside RadixCyclicNN (60). The move finds
signals reliably; whether reading one beats the standard tool has to be measured
each time.

*Where:* `Research/README.md`, `Research/VanishingGradientIsAFeature.md` §13 ·
**held**

### 51. The vanishing gradient is a reading — count the multiplications and divide the decay back out
> The vanishing gradient is a feature of gradient descent. Keep track of the
> number of times we do `X*X` — the underlying cause — and use that number to
> normalise the values back to 0–1.

*Refined — the premise holds and sharpens with depth; the correction built on it
loses to Adam.* This is the paper's own claim, which 20 never tested.
Back-propagation multiplies, so a gradient `n` products from the loss arrives at
about `c^n` of its size, and `c^n` is invertible: the decay is a code whose key,
`n`, is known before any data. On a sigmoid MLP at initialisation `log‖g‖` is
linear in the count, with **R² 0.926 at depth 2 rising to 0.997 at depth 12**.
Dividing the decay back out, every SGD arm held to the same total step, reaches
**0.0727** at depth 4 against plain SGD's **0.1271** and a count-free per-layer
normalisation's **0.0906** — both gaps inside one standard deviation (about
0.04, four seeds). At depths 5–6 all three SGD arms land on 0.147, and **Adam
beats every count-based arm at every depth** (0.0248, 0.0118 and 0.0051 at
depths 4–6): it estimates per parameter the scale the count models per layer
(39). The count is free and `c` is not — a `c` fitted once scores 0.1083 against
0.0727 re-fitted every step — and no arm ever trains with the a-priori `c =
0.25`, so the claimed edge over Adam, "correct at step zero", is untested. The
paper's tables are for `--steps 1000`; re-run at the script's committed default
of 1,200 steps, the best dial moves from inside the range (α = 0.75) to its end
(α = 1.00, 0.0547), and Adam still wins (0.0153).

*Where:* `Research/VanishingGradientIsAFeature.md` §6, §9–§11,
`Experiments/DepthCountedNormalisation/depth_counted_normalisation.py` ·
**refined**

### 52. A cycle is repetition stored once — and, read as statistics, a merge of every context a gram occurs in
> I originally proposed that the brain is a directed acyclic graph. However, I
> believe the brain is actually a directed cyclic graph. Where cycles are a
> feature, not a bug.

*Refined — against an unbounded tree the cycle buys exactly what the paper
predicts; bounded to the graph's own order, a tree buys most of it too.* In
RadixCyclicNN a trigram lives in exactly one node, so the second `aaa` of
`"aaaa"` is an edge back to itself, and the graph represents lengths it never
saw. `RadixAcyclicNN` is the control: the same network, operation for operation,
with a node as a context rather than a gram, so it cannot contain a cycle. **A
thousand a's cost the graph 1 node and the tree 999**; shown `aaaa` once, the
graph scores `aaaaaaa` with **0 unknown** transitions and the tree with **3 of
5** — the paper's Prediction 1, confirmed to the node. On 640 lines of the
papers' prose the graph holds its grams in **3,516** nodes against **61,213**,
and scores held-out text at **4.057 bits/char with 12% misses** against the
tree's **6.514 and 54%**, while the tree recites **572 of 640** lines and the
graph none. But bounded to **two symbols** — the graph's own order — the tree
predicts held-out prose slightly *better* (**3.832**, 11% misses) and holds
`a`×1000 in 3 nodes. What a cycle does, read as statistics, is **merge every
context a gram occurs in**, and merging is what generalises; a tree approaches
it only by throwing its depth away. One corpus, one seed, every arm scored the
graph's way.

*Where:* `Research/CyclesAreAFeature.md` §3, §9, `RadixAcyclicNN/README.md`,
`RadixAcyclicNN/DESIGN.md` §17, `RadixAcyclicNN/results/` · **refined**

### 53. You cannot leave a loop from inside it — and taking the cycle out of the structure moves it into the process
> When we encounter a cycle, we use metacognition or another part of the brain
> instead.

*Confirmed by construction, on both sides; whether the monitor adds anything to
the clock is unmeasured.* A walker whose behaviour depends only on the current
node behaves identically on every visit, so whatever ends a loop must come from
outside it. The paper names three exits and each is built: a **clock** — the
search state is `(node, chars emitted)`, so the graph is cyclic but the search
space is not, backed by a 200,000-expansion budget; a **monitor** —
`dialogue.py` watches outputs, never the graph, and backs up where the walk went
round (99); and an **inversion** — duplicates the search could not
avoid become a 2NRL negative phase. Allowing cycles also forced the arithmetic:
a negative cycle makes the cheapest path `−∞`, so costs must be non-negative.
The acyclic control shows the converse. On a tree a search needs no state and no
budget (two to five nodes per prediction where the graph's visits four to
seventy-three) and signed costs are legal — but a bounded tree asked for more
than its window must re-enter itself, and a token-by-token walk "can forget its
way into a loop (…sets in the east and sets in the east…)": "the one place the
acyclic model needs the thing the cyclic one needs everywhere". The paper's
Prediction 2 — switch the monitors off and self-conversation collapses to a
fixed point within single-digit turns — is the result that would cost it the
thesis, and it has not been run.

*Where:* `Research/CyclesAreAFeature.md` §4–§6, §9,
`RadixCyclicNN/radixnet/search.py`, `ModelKit/modelkit/dialogue.py`,
`RadixAcyclicNN/DESIGN.md` §11, §17 (T-07, T-08) · **confirmed**

### 54. Inverting a path is two-colouring it — exact on a tree, best-effort with cycles, impossible on a self-loop
*Confirmed as mathematics; its cost to learning is untested.* An edge scores `w
· f_p · f_c`, so flipping one endpoint negates it and flipping both leaves it
alone. To make every edge of a failed path unlikely by touching nodes, flip
**every other node** — flipping every node changes nothing (checked: `−0.350
−0.360 −0.176 −0.048` becomes all-positive under alternate flips and is
untouched under all-flips). A path is always two-colourable; a graph is exactly
when it has no odd cycle, and a walk through a cyclic graph can close one. On a
3-cycle **the best of all eight flip patterns flips 2 of 3 edges, never 3**, and
a self-loop scores `w · f_p²`, so flipping its node changes it by **exactly
nothing** — it is sign-locked and moves only through its weight. So local
inversion takes "the parity that covers the most edges": allowing cycles turned
an operation that is exact on a DAG into a best-effort one. The acyclic control
is the other side — a tree is bipartite and each END leaf private, so the same
flip reverses every edge of a path, END included, and `make test` asserts it.
Whether odd cycles measurably weaken inversion is the paper's Prediction 3,
unmeasured.

*Where:* `Research/CyclesAreAFeature.md` §5.3, `RadixCyclicNN/DECISIONS.md`
D-005, D-028, `RadixCyclicNN/radixnet/graph.py` (`flip_nodes`),
`RadixAcyclicNN/README.md` · **confirmed**

### 55. A sigmoid is one step of a sine wave rotated 45 degrees
> Instead of the traditional sigmoid function frequently used in ANNs, we use a
> sine wave rotated 45 degrees. … This led me to see the similarities between a
> sigmoid function and a sine wave.

*Confirmed as geometry, and 45 is exactly the right number.* Rotate `y = sin t`
by θ: the curve stays a function of `x` if and only if `cos θ ≥ sin θ`, that is
**θ ≤ 45°**, and at the limit it is a staircase. One step of it, drawn at 40°,
fits a logistic `σ(6.90·x)` with a maximum error of **0.044**. So the sigmoid is
not a rival to the wave — it is the wave with one step kept and the rest
flattened, and the design question becomes why the rest was ever deleted. Two
qualifications: what is built is the *unrotated* `−sin(x/3)`, which gives up
monotonicity (`f(1.0) = f(17.85) = −0.327`), so the geometry motivates the
choice rather than specifying it; and on a staircase target — the sigmoid's own
shape — the learned sine loses to ReLU (1.91e-02 against 6.59e-03), because a
sigmoid-family unit is a local basis and a sine unit speaks across the whole
domain. The logistic fit's script is not committed.

*Where:* `Research/SineWaveActivationFunction.md` §1, §3, §7,
`Experiments/ActivationFunctionTest/README.md` · **confirmed**

### 56. The sine unit never dies along its input — it can die along `a` and `b`
*Refined — confirmed along the input, and a second axis was found.* `|f′(x)| =
|a·b·cos(b(x−h))|` returns to `|a·b|` every half period, so the share of a range
where a unit is deaf (`|f′| < 0.01`) stays near 2% however wide the range —
**1.8% on [−10, 10], 2.0% on [−100, 100]** — where the sigmoid's grows from
54.2% to 95.4% and ReLU's sits at 50%. Inside self-building networks that train,
the sine is deaf on **0.2–1.1%** of the pre-activations they produce, against
28–47% for sigmoid and tanh and 61–80% for ReLU; one unit died in 30 sine runs,
and giving the baselines the sine's own initialisation leaves them at 35–69%.
But `|f′| ≤ |a·b|`, so a unit can go deaf *everywhere* by collapsing `a·b`: with
the activation learning as fast as the weights, the sine loses **13.2 of 48
units** on one target, and every dead unit has `|a·b| ≤ 0.0100` against a live
median of 0.067. The paper's tenth-rate rule for the activation — which the
decision record called "a stability guess, not a measured optimum" — now has
that measurement behind it. Never dead is not never vanishing: at the defaults
`|a·b| = 1/3`, which is 20's 22× smaller layer-0 gradient.

*Where:* `Research/SineWaveActivationFunction.md` §6, §8,
`Experiments/ActivationFunctionTest/self_building_sinewave.py`,
`Experiments/ActivationFunctionTest/README.md`, `RadixCyclicNN/DECISIONS.md`
D-003 · **refined**

### 57. Liveness is not accuracy — the sine's case rests on one unseeded CartPole run, and under control it never fits best
> The improvement is astounding with a ~200% to ~400% improvement in reward.
> Sigmoid: 9 / ReLU: 10 / Sinewave: 28-48

*Refined — the headline is unreplicated, and the controlled test disagrees on
accuracy.* The CartPole comparison that launched the sine is one greedy episode
per arm, with no seed and no variance, and the activation also sits on the
output layer, where a sigmoid squashes Q-targets near 20 into (0, 1). The
decision record calls it "the most load-bearing claim in the project on the
thinnest measurement"; the paper writes the replication — a linear-head DQN
pair, 20 seeds, 100 episodes, a tanh arm, a `b` sweep — and nobody has run it.
The controlled test that *has* run — a self-building network, linear head, Adam,
five seeds — shows the sine barely dying (56) and fitting best on
none of three targets: the swept sine `chirp` goes to sigmoid (**3.78e-05
against 1.24e-04**), `spline` to sigmoid, the staircase to ReLU. What pays is
learning the wave: against the same sine frozen at `−sin(x/3)` it gives **12×
the accuracy from 7.5× fewer units**, and since `a` and `k` are redundant with
the output weight and bias, the knob that pays is the frequency — the one 24
says is wrong at 1/3. RadixCyclicNN adopted the sine on the CartPole run, and
the count model its fast ports run pins every activation to 1 (80).

*Where:* `Experiments/ActivationFunctionTest/`,
`Research/SineWaveActivationFunction.md` §9, §12, `RadixCyclicNN/DECISIONS.md`
D-002, Q-2 · **refined**

### 58. Read Adam's two moments as two interfering waves — TwoSine
*Refined — the interference does real work, but no more than simpler brakes, and
a one-line change to Adam beats it.* Adam's `m/√v` is a signal-to-noise ratio,
so its step is already a saturating, odd, bounded function of consistency — a
sigmoid in disguise, which is the sine paper's argument moved from the
activation to the optimiser. TwoSine keeps two first moments (0.9 and 0.99) over
one shared second moment, maps each ratio to a phase, and steps by the mean of
two sines. By sum-to-product that is a **beat**: the carrier is direction and
confidence, and the envelope `cos((φ_f − φ_s)/2)` closes when a parameter
reverses against its own trend — a brake read off the gradients, not a schedule.
Paired over **120** cells it beats Adam (**57–36**, p = 0.038) and the same
three moments combined linearly (**57–35**, p = 0.028), so the waves do the
work, not the extra memory; on a network of learnable sine units it more than
halves Adam's error (**0.001743** against **0.003835**). But it is not
distinguished from one wave (p = 0.594) or a hard sign gate (p = 0.175), the
brake engages on only 2.3% of parameter-steps, and plain Adam with β₁ = 0.99
reaches **0.000341** on the same network, five times better. Unclipped, the
envelope goes negative (−0.55) and reverses the step. Three toy problems, at
most 20 dimensions, five seeds.

*Where:* `Experiments/TwoSineOptimizer/` on branch `claude/keen-feynman-ntoqni`
· **refined**

### 59. Keep the game coordinate, drop the output partition
*Held — the recommendation the compression experiments end on, and it outruns
its own tables.* Part 1 found the partition is not what compresses, and that its
real win — games never played — comes with the continuous selector coordinate
(15). Part 2 found how to lay bands out if you keep them: exactly N equal bands,
ordered by similarity, re-laid out whenever a game is added. Midpoint insertion
— static, dust-free and ordered at once — was the worst layout tested (**2.36×
damage against 1.37×** for re-laying out), and a one-hot selector with a
partitioned output was the worst condition overall (**0.218** at N=16). So
FINDINGS concludes: supply the coordinate through the hashed-mechanic modifier
(18), keep the **full output range**, and every number in Part 2 becomes moot.
But the single-seed tables do not show a full-range coordinate beating the
partition: the scalar full-range selector scores **0.0741 against the ordered
partition's 0.0607** at N=16 and 0.0830 against 0.0613 on unseen games, and the
multi-dimensional modifier meant to carry the interpolation has never been run
through this harness.

*Where:* `Experiments/NeuralCompression/FINDINGS.md` §4, §7, §9 · **held**

---

## 2NRL — learning by inverting consistent failure

### 60. 2NRL — train on the failure at full rate, invert, fine-tune gently
> I learned by failing consistently, then doing the inverse/opposite. Once I
> found a thread to pull on, I would pull hard.

*Contradicted at matched compute — it wins only where the inverse of the failure
already is the answer.* Double-Negative Reinforcement Learning is this note
written as an update rule: train **on** the failures at full rate until the
model reproduces them ("you cannot invert a failure you have not first
represented"), invert the network so the most likely becomes the least likely,
then fine-tune on the truth at a fifth of the rate. It shaped RadixCyclicNN —
garbage became a first-class input and every loop emits `(bad, good)` pairs —
and its paper says plainly it was never measured against a baseline. It since
has been, three times, with updates, rates and budget matched. **On chess**,
with the rules supervised, 2NRL loses head to head to every other arm — **0.229
± 0.088** against positive-only training — and scores lowest against a random
legal mover (**0.243** to the control's **0.535**). **Inside RadixCyclicNN**,
through the universal game encoder, the architecture's own `two_nrl` takes Nim's
legality from **0.94 to 0.44** while the matched positive arm stays at 0.94; the
local path inversion gives 0.40. **On CartPole** it wins, **500.0** against a
step-matched control's **431.9** — because there "the inverse of a good failure
policy *is* a good balancing policy". The paper's load-bearing prediction P4,
that inversion beats merely pushing away from the same negatives, fails too: on
chess repulsion matched 2NRL at 24 rounds and beat it at 40 (86.4 refusals a
move against 133.2). Three seeds and one task each; the text experiment the
paper designs is unrun, and the paper still says 2NRL "has never been measured
against a baseline".

*Where:* `Research/2NRL.md` §3–§5, §10–§11, `RadixCyclicNN/DECISIONS.md` D-009,
`TwoNRL_Chess/README.md`, `Experiments/UniversalGameEncoding/README.md` (§
twonrl), `Experiments/TwoNRL_CartPole/README.md` · **contradicted**

### 61. Two negatives make a positive — one sign flip turns a trained failure into its exact opposite
*Confirmed, exactly — and what it returns is precisely what phase 1 learned
backwards, nothing more.* On CartPole a sine network trained only to knock the
pole over (**9.2** steps) scores **499.4 / 500** after one closed-form sign
flip, having never seen the balancing task. On chess a network trained to get
the rules backwards starts at **2,341.9** refusals a move and legal AUC
**0.011**; after the flip it is at **1.7** and **0.989**, and capture and check
reverse the same way. The measured negation error over every flip is
**0.0e+00**: a network and its inversion score `a` and `1 − a`. The mechanism is
the sine's sign parameter — `a → −a, k → −k` negates a unit for every input, so
each unit keeps the evidence it looked at and reverses its verdict on it. The
caveat is what gets reversed: the flip hands back the mirror image of what phase
1 built, which is the answer only when phase 1 built its exact opposite — a
failure policy anti-balance 94% of the time, or heads trained on the complement
of legality. Move quality is not reversed into anything (centipawn loss 336.9
before the flip, 355.8 after), and the ratio flatters: the positive control
reached 1.7 refusals by round 4 with no flip at all.

*Where:* `Experiments/TwoNRL_CartPole/README.md`, `TwoNRL_Chess/README.md`
("Phase 2 on its own"), `TwoNRL_Chess/sbnn.py` (`invert_unit`),
`Research/2NRL.md` §3.1, §4.3 · **confirmed**

### 62. The failure has to be consistent — what inversion recovers is bounded by the failure's entropy
*Contradicted as stated — the entropy of the failure did not decide what
inversion recovered; what the failure encoded did.* The paper makes consistency
the operating condition — "the negation of noise is noise" — and names its
falsifier: "if arm C matches arm A, §5 is wrong". On chess the negative sets
carried measured entropies: **0.00** for the network's own failures, **0.15**
for Stockfish's worst legal move, **0.81** for a random legal move. Before
legality was supervised the end-to-end order ran backwards — random **54.4**
refusals a move, worst-move **96.8**, own failures **133.2**. What the flip did
depended on where phase 1 had aimed: aimed at illegal moves it made the network
**×6.1 better**; aimed at a legal move, worst or random, it threw away the
legality phase 1 had taught by accident and made it **×11.3** and **×12.8
worse**. *The operator is neutral; what it is pointed at is everything.*
CartPole gives the sharpest form: always pushing the same way is the most
consistent failure there is, and it inverts into another constant push (**36.4**
of 500); only a failure made state-dependent inverts into balance (**499.4**). A
failure is worth inverting when it is the mirror image of the solution —
state-dependent, and aimed at the quantity being measured.

*Where:* `Research/2NRL.md` §5, §11, `TwoNRL_Chess/README.md` ("The arms"),
`TwoNRL_Chess/results/README.md` (the 40-round table in PR #13),
`Experiments/TwoNRL_CartPole/README.md` ("Grading the loss", "Ablations") ·
**contradicted**

### 63. Negating a unit means negating its offset — a test of an involution cannot see a wrong inverse
*Refined — the sign parameter makes inversion possible, but only with the
offset.* What the sine buys, and ReLU and the logistic cannot, is a parameter
whose sign is the unit's sign, so negating a network is a parameter change. But
the unit is `a·sin(b(x−h)) + k`, and `a → −a` alone gives `−f + 2k` — the
negation only while `k = 0`, and `k` is learned. RadixCyclicNN's `invert()`
negated `a` only, from the first implementation until 2026-09-14. Once measured,
rankings on randomised four-child nodes **failed to reverse 87% of the time**,
and after eight epochs of real training edge signals came back **−0.453 → +0.459
instead of +0.453**: "2NRL was quietly not doing what it says". Every test had
passed, because they asserted the properties that survive the bug — inverting
twice is the identity, the signs of `w` and `a` flip — and the graph-level test
never trained, so every `k` was still 0. The regression tests now move the
offsets off zero first and assert what inversion is *for*: every edge signal
exactly negated, every ranking exactly reversed. GTMNN's implementation on its
branch still negates `a` alone, though its spec on `main` now says `a` and `k`.

*Where:* `RadixCyclicNN/DECISIONS.md` D-010, `RadixCyclicNN/radixnet/graph.py`
(`invert`), `RadixCyclicNN/tests/test_graph.py`,
`Research/SineWaveActivationFunction.md` §4, `GTMNN/DESIGN.md` §5 · **refined**

### 64. The NOT of a weight is its complement, `W → 1 − W`
*Contradicted — on the same trained network both negations reverse the rules,
and the complement erases them.* `TwoNRL_Chess` on `main` ships `W → 1 − W` as
its default inversion, on the reading that "the NOT of a quantity in [0, 1] is 1
− x". An unmerged run tested it against the two negations on one phase-1
network, the three runs bit-identical until the flip at round 36. The network
had learned to rank illegal moves above legal ones, legal AUC **0.010**. After
one operation and no training: the per-unit flip **0.989**, the sign flip `W →
−W` **0.988**, the complement **0.514** — every head at chance, 137.8 refusals a
move — and phase 3 brings it back only to 0.633. The reason is exact: `1 − W =
−W + J`, so the first layer computes `−z + Σx + 2·bias`, and `Σx` — the 31 to 43
active inputs of a position — becomes a phase shift of 10.3 to 14.3 radians
through the sine at `b = 1/3`, which scrambles the ordering position by
position. `1 − w` reflects about the middle of [0, 1]; these weights are centred
on 0, and the reflection about the middle of *their* range is `−w`. Being an
involution is not the property that matters; reversing the order is.

*Where:* `TwoNRL_Chess/sbnn.py` (`invert`), `TwoNRL_Chess/experiment.py`
(`Config.invert_mode`); the measurement in `TwoNRL_Chess/results_sine/README.md`
on branch `feat/twonrl-chess-longrun` · **contradicted**

### 65. The inversion has to stay a NOT all the way up — parity in a stack, commutation in a search
*Confirmed — every level has its own way of cancelling the negation, and each
was caught by measurement.* In the radix graph an edge scores `w·f_p·f_c`, so
flipping every weight and amplitude flips each score once. A feed-forward stack
composes instead, and with an odd activation the same literal rule cancels layer
by layer — **flipping everything is the identity**, tested at 2, 3 and 4 layers
— while flipping the weights alone negates the output only when the layer count
is odd (an earlier claim that it held "at 0.0" rested on a check that was not
testing what it said). The exact operators break parity once, and growth must
respect it: a new unit enters with zero outgoing weight, which survives
negation. One level up, a search must commute with negation or the flip stops
reversing order where it was proved to: minimax does not (`min(−v) = −max(v)`),
the midrange `(min + max)/2` does, and it keeps most of what search is worth —
on 800 held-out positions minimax gives away **112.6** centipawns a move, the
midrange **128.7**, random play **289.0**. It is the odd-cycle obstruction
(54) one level up.

*Where:* `Experiments/TwoNRL_CartPole/README.md` ("parity correction"),
`Experiments/TwoNRL_CartPole/test_invert.py`, `TwoNRL_Chess/README.md` ("The
inversion negates every unit", "Why the network scores moves and not positions")
· **confirmed**

### 66. Inversion needs a branch — 2NRL on a tree teaches nothing until the failure departs from a success
*Refined — on a tree the negative phase has nothing to teach until the failure
shares a branch with the success.* In the acyclic tree a garbage line the model
has never seen is a **corridor**: one option at every step, a softmax over one,
loss 0 and no gradient. So the failures must be shown to a model that already
holds the good texts, where the swapped word is a real branch; there the
inversion reverses every pairwise preference exactly, and the positive phase
leaves the good continuation preferred at **35 of 38** such branches. In the
cyclic graph a node is shared by every sentence that uses its gram, so the
negative phase always finds a branch; in a structure of contexts a failure is
informative only where it departs from something the model already knows. What
the negative phase needs is not the failure but its point of departure. No
control arm — the same model fine-tuned without the negative phase — is
reported, so this shows the mechanism, not that 2NRL beats plain training.

*Where:* `RadixAcyclicNN/README.md` ("What 2NRL needs on a tree: a branch"),
`RadixAcyclicNN/DESIGN.md` §12, `RadixAcyclicNN/tests/test_radixtree.py`
(`test_two_nrl`) · **refined**

### 67. Phase 1 ends when the failure is learned, not on a date
*Refined — the event trigger works; what it exposed is that finishing the
failure costs the repair, and that a plateau tested against the best round so
far fires on noise.* The paper said phase 1 should run "until the failure mode
is *well* represented … I have not looked". The chess run flips on the first of
three events: the network puts at least 90% of its probability on the failure,
it has stopped improving (counted only above p = 0.5), or the run is one round
from its end. Without the floor, two noisy rounds ended phase 1 at p = 0.43;
with it, p climbed to 0.93 over 23 rounds and left phase 3 one round, which did
worse than a premature flip with 17 rounds of repair — *either the failure
finishes and there is no time to repair it, or there is time and the failure is
half-built.* Inverting inside every round, the paper's perpetual loop, was worst
of all (131.8 refusals a move against 74.4). And the plateau test was wrong:
comparing a round with the best round so far is a maximum over a noisy series,
and with round-to-round noise of **0.042** against a **0.01** threshold two of
three seeds ended phase 1 while p was still climbing (at 0.60 and 0.81). A
trailing five-round window fixed it — the same lesson for 21's plateau detector
and 26's growth trigger.

*Where:* `TwoNRL_Chess/README.md` ("Phase 1 ends when the failure is learned"),
`TwoNRL_Chess/experiment.py` (`FlipTrigger`),
`TwoNRL_Chess/results_r400/README.md`, `Research/2NRL.md` §12 · **refined**

### 68. "Pull hard on the thread" is a schedule over search breadth, and its trigger is a reward
> Explore rapidly and widely until you find a thread, then tighten up the
> exploration and iterate.

*Held — the first reading was wrong and the author corrected it; the schedule is
not built.* The implementation first read "once I found a thread to pull on, I
would pull hard" as a weighting — train harder on worse failures — and that
proportional boosting is worth about 50 points on CartPole (447.3 without it,
499.4 with). "That is not what I meant." The process is an annealing schedule
over **search breadth** — temperature, beam, `k`, sample count — not over how
hard any one example is trained: breadth decides which examples are ever seen,
weighting decides what to do once they are. Today every such parameter is fixed
for the length of a call, so a long run explores as widely in its last
generation as in its first. The trigger looked like the hard part and dissolved
when the question changed: not "what counts as finding a thread" — a judgement —
but a reward event, "when the network is rewarded, or we reach a known area of
completion", which the system already emits and which needs no bar set in
advance. The machinery half-exists — the learning-rate schedule evaluator is
pointed at the wrong quantity — and it is recorded rather than built so the gap
stays visible.

*Where:* `Research/2NRL.md` §6, `RadixCyclicNN/DECISIONS.md` D-027, D-033,
D-067, Q-13, `Experiments/TwoNRL_CartPole/README.md` ("Ablations") · **held**

---

## Games as text, and the rules of chess

### 69. Is there a way to encode all games? — one tape, one encoder, and the rules as referee
*Confirmed for encoding; as 4 predicted, learnability is a separate question
with a much wider spread.* Every finite, discrete, sequential game is one object
— a labelled transition system with payoffs — and a play is a walk through it,
which is exactly what RadixCyclicNN stores. So encoding a game is choosing which
projection of the walk to write down: a header token, one (state digest, action)
pair per ply, an outcome, every token three characters because the window is.
Six games go through one encoder that knows none of them — chess, Othello,
Connect Four, tic-tac-toe, Nim and Pig, whose die is a player and whose faces
are actions — and a seventh, hexapawn, was added from outside the package.
Decoding is replay through the game's own rules, which makes the decoder total,
self-checking and a referee: a broken tape is a 2NRL negative with no engine and
no labels. What the same network then learns runs from **0.999** legality at
seen positions on Nim to **0.08** on chess. Imperfect information, simultaneous
moves and real time are argued, not run.

*Where:* `Experiments/UniversalGameEncoding/ENCODING.md` §1–3, §7–8,
`Experiments/UniversalGameEncoding/README.md`,
`Experiments/UniversalGameEncoding/USING.md` §6 · **confirmed**

### 70. RadixNet knows a token by its three characters and nothing else
*Confirmed four times — every instance looked like a training problem and was an
encoding one.* The graph identifies a trigram by its characters alone, so two
kinds of token that spell the same characters are one node with children from
both roles. The outcome `draw` and Connect Four's "column 1" were both code 1;
game 0's header was its action 0; and the empty board's state code was also
"play cell 0" — so an injective state code measured *worse* than a hash (**0.72
against 0.99** legality at known positions) until the code space was
partitioned. One level down, a stride-1 window over three-character tokens has
three phases, and a trigram that is a whole token in one place and a seam in
another is one node — only 1–9% of distinct trigrams, but up to **45%** of
occurrences. Reserving each token's last character to its own alphabet makes the
phase readable and takes Connect Four from **10.65** refusals a move to **0.34**
(legality 0.70 → 0.92). *If two tokens can be told apart only by where they sit,
this architecture cannot tell them apart.*

*Where:* `Experiments/UniversalGameEncoding/README.md` ("Three bugs, one
lesson", "phase"), `Experiments/UniversalGameEncoding/ENCODING.md` §4–5 ·
**confirmed**

### 71. The memory is one token — spend it on something with locality
*Confirmed — the cliff lands exactly where the architecture predicts, and for a
large state space the fix is to stop describing the state.* `RadixNet._locate`
conditions every prediction on the last trigram of the prefix alone, so whatever
the tape says about a position must fit in one token — 18 bits, or 15 with
phase-disjoint tokens. Nim scores **0.94** legality at 15 bits, **0.51** at 20
and **0.94** again at 30, because twenty bits leave five in the last token and
thirty leave fifteen. A hash has no locality, so the model is silent, not wrong,
on unseen positions, and a 13-bit digest of a chess position does not determine
its legal moves: **1,725** refusals a move against guessing's **1,894**. The
same 13 bits spent on *the square the last move landed on*, which means the same
thing in every position, take it to **98.6**. It is 29 seen from the other side
— a code that names a position cannot generalise, a code that means something
can — and the exact contrast with 18, whose hash is of a *set* and so keeps
overlap.

*Where:* `Experiments/UniversalGameEncoding/ENCODING.md` §6, §10–11,
`Experiments/UniversalGameEncoding/README.md` ("phi", "mix"),
`RadixCyclicNN/radixnet/model.py` (`_locate`) · **confirmed**

### 72. One network, six games — the header token is the selector
*Refined — it works with no extra machinery, and it is not free: up to a quarter
of the answers are another game's move.* NeuralCompression needed an output
partition and a selector network to ask whether one network can hold many games
(15); here it falls out of the encoding — six corpora in one training call, the
header token the whole selector. The shared graph has **49,865** nodes against
**63,991** for six separate ones, **1.28×** fewer, and every game pays for the
company: legality is worse for five of six (Nim **0.943 → 0.625**). The reason
is the one the encoder keeps finding: only the header is reserved per game, so
nothing stops a Connect Four position being answered with a token that only ever
appeared in Othello — **14–27%** of the time, hidden in the legality columns
because a foreign token is usually illegal anyway. The compression is real and
so is its cost, as in 22; here the cost is interference, not capacity — a point
for 27's one network per region.

*Where:* `Experiments/UniversalGameEncoding/README.md` ("multi"),
`Experiments/UniversalGameEncoding/results/multi.json` · **refined**

### 73. A rating of a network that cannot find a legal move rates its search for one
*Confirmed, and it reversed a reading: learning the rules did not make these
networks worse players; it revealed that they were not playing.* Stockfish beats
every network every game, so the chess arms were rated on a ladder anchored at a
random legal mover, with the board walking each network's ranking until
something was accepted. That ladder said "no difference" — `2nrl` scored
**0.254** against the field, exactly the random mover's **0.254** — and the
selected positive-only network reached **112** Elo, the first interval to
exclude random. Then the rule heads cut refusals from 97.9 a move to 1.1, and
every arm's score against the random mover *fell*. The same saved weights ranked
two ways, nothing retrained: on quality alone each needs about a hundred
refusals to find a legal move and every arm scores **above** random
(0.506–0.631); with the legal head each plays the move it actually wants, and
every arm scores **at or below** it (0.237–0.463 against 0.481), falls of two to
five standard errors. A near-uniform draw over legal moves has no plan to be
punished for; a consistent bad policy does. So 112 Elo, and every rating before
the rule heads, largely rated the randomisation that ignorance of the rules
imposed. The Elo fitter, meanwhile, had never converged — a limit cycle hidden
behind `max_iters` — and fixing it moved every arm by 0. CyclicCortex's front
end keeps the same rule for display: an illegal pick is counted and shown,
because hiding the substitution would make every cortex look perfect.

*Where:* `TwoNRL_Chess/README.md` ("And then the rating fell", "Caveats"),
`TwoNRL_Chess/ranking_ablation.py`, `TwoNRL_Chess/elo.py`, the first ladder in
commit 5765183; the front end in `CyclicCortex/cortex/frontend/README.md` on
branch `claude/keen-feynman-ntoqni` · **confirmed**

---

## The radix graph — structure and search

### 74. A radix tree for information compression — a small network at every node
> In this iteration, I chose a Radix Tree specifically to focus on information
> compression. This is just one variation of many to solve AGI/ASI.

*Held, and never trained — the note the whole radix line descends from.* A path
through the trie *is* the sequence; a shared prefix is stored once, so **the
structure does the compressing**, and a small network at each node only has to
explain the part of the string on its own edge. `RadixTrieLLM_RNN/main.py`
builds exactly that — a byte-level transformer per node (64 wide, 4 heads, 2
layers), radix splitting on insert, each node's pooled state threaded into its
child — but nothing trains it, and `RadixTreeRNN` is an empty file. Everything
after keeps the premise and changes the node's model or the structure's shape:
RadixCyclicNN replaced the transformer with one sine per node and let repeated
grams become cycles; RadixAcyclicNN kept the tree to measure what the cycles buy
(52); PrimedRadixPair inserts every option up front and keeps
this insertion only as its brute-force oracle (106). The tree/trie
distinction drawn here — path compression from one, terminals on internal nodes
from the other — is the one 13 draws for games.

*Where:* `RadixTreeLLM/README.md`, `RadixTrieLLM_RNN/README.md`,
`RadixTrieLLM_RNN/main.py`, `RadixTreeRNN/README.md` · **held**

### 75. A trigram because of how the transformer pivots
> I chose the trigram because it reminded me of transformer architecture, in how
> the transformer pivots.

*Held — a structural analogy, recorded as the reason for the design rather than
a consequence of it.* With a window of three characters at stride 1, the
character shared by the window ending on it and the window beginning on it
**is** the pivot; three is the smallest window with a pivot and context on
either side. The consequences are exact: every edge is keyed on a two-character
overlap (`label[p][-2:] == label[c][:2]`, asserted by the tests), decoding needs
no learned decoder, and there is no vocabulary and no unknown token, so base64
and source code train as readily as English. The honest cost is order — a node
is a gram, so the model is first-order in grams, and the acyclic control
recorded it forgetting the sentence it was in at every shared gram (after `the
cat`, the likeliest thing it knew was `gold`). The window later became a dial
(77); nothing tests whether a one-character pivot captures anything a
transformer's does.

*Where:* `RadixCyclicNN/DECISIONS.md` D-006, Q-11,
`RadixCyclicNN/radixnet/encoding.py` · **held**

### 76. The graph compresses itself, lossy on purpose — parameters discarded, predictions kept
*Refined — predictions survive compression only once everything hung on an edge
moves with it.* A unary chain merges into one node with a longer label; a
transition observed into its middle splits it again; compression runs after
every epoch. The merge keeps the parent's parameters and discards the child's,
justified because a unary chain is deterministic — probability 1, cost 0 — so
the **parameters** go and the **predictions** stay, as the tests assert.
Repeated text costs nodes only where it branches: the benchmark's 2,000,013
characters fit in **894** nodes over 1,048 trigrams, and real prose runs at 1.20
grams per node. Two forces push back and the record lets them win: a blamed
transition stays out of compression so the fragment stays nameable, and a node
taught to hand over becomes a junction. For the count model the invariant was
false until 2026-09-28 — a split and a merge lost what the model hangs on its
edges (85).

*Where:* `RadixCyclicNN/DECISIONS.md` D-007, D-046, D-068, D-092, Q-16,
`RadixCyclicNN/bench/RESULTS.md`, `RadixCyclicNN/radixnet/graph.py` ·
**refined**

### 77. The encoding is a dial of the model — the graph never mentions a character
> What about word n-grams?

*Held — built in three ports and structurally identical under nine settings; no
setting has been compared with another on prediction.* The obvious reading asks
for a second model. It needs none: every structural rule of the graph — a label
is a sequence of symbols, an edge exists where two labels overlap by `n − 1` of
them, compression merges a unary chain at that seam — is stated in terms of the
window, and "not one of those rules mentions a character". So the input became a
value the graph is born with, `Encoding(unit, n, stride)`: `char:3:1` is the
trigram and stays the default, a stride equal to `n` is tokenisation
(`char:4:4`), and `word:3:1` is the word trigram, in which a repeated phrase
compresses into one node whose label *is* the phrase. Two sessions reached word
n-grams independently — one as a new model kind with a vocabulary, one as a unit
on the dial — and the dial was kept: strictly more general, and no vocabulary at
all. What is measured is structure: the three ports build the same graph under
nine settings (`char:3:1` **470 nodes, 824 edges**; `word:3:1` **72, 131**).
Sounds and BPE tokens later arrived as further units on the same dial
(114, 117).

*Where:* `RadixCyclicNN/SPEC-WordNGrams.md`, `RadixCyclicNN/DECISIONS.md` D-071,
D-073, `RadixCyclicNN/radixnet/encoding.py`,
`RadixCyclicNN/rust/tests/encodings.rs` · **held**

### 78. Prediction is a shortest path — and a shortest path prefers to stop
*Confirmed, including the bias it predicted against itself.* The specification
named a cost function and Dijkstra, and prediction is exactly that: edge cost
`−log P(c | p) + step_penalty ≥ 0`, searched over the depth-unrolled graph,
deterministic, so conversations reproduce and tests are tractable. The record
named the cost up front — cost accumulates per edge, so the search **prefers
short, confident completions**, "a property of the method, not a bug" — and it
showed wherever anyone looked: an early `2 × length` truncation silently cut
correct completions; the agent had to use the beam because Dijkstra's cheapest
path "ends at END after a couple of characters that can never be a whole call";
the bottom beam had to be capped because in a cyclic graph "least likely" means
going round for ever; and on prose the graph recites **0 of 640** lines. One
label per state also blinds Dijkstra to which walk reached it; the resonant
model's k-best search returns the `k` cheapest walks exactly, expanding **159**
states from `"the "` at `k = 5` where the beam expanded **1,312** for an
approximation. `step_penalty`, `length` and `to_end` remain the levers, and none
has been tuned by measurement.

*Where:* `RadixCyclicNN/DECISIONS.md` D-008, D-024, D-025, D-056,
`RadixCyclicNN/radixnet/search.py`, `RadixCyclicNN/radixnet/phasesearch.py`,
`RadixCyclicNN/DESIGN.md` §30.4, `RadixAcyclicNN/README.md` · **confirmed**

### 79. Accept the vanishing gradient — a one-hop rule makes `n = 1`
> Accept the vanishing gradient, update the activation function instead; `N*N`;
> activation(child) × activation(parent)

*Held — the vanishing gradient was dissolved by construction rather than
accepted, and the price moved elsewhere.* RadixCyclicNN built the note as a
**one-hop rule**: an observed transition `p → c` updates the edge weight, the
two node states and the sine parameters of `p` and its children, and nothing
further — the graph is cyclic, so there is no finite unrolling to back-propagate
through anyway. Every edge is one product, `W · f_p · f_c`, so a gradient
crosses exactly one multiplication: **`n = 1`, and no product chain exists**,
and training cost is linear in transitions whatever the depth or cycle length.
Two consequences fall outside 20. The loss is a local `−log softmax` over one
parent's children, so the model never receives a gradient for a sequence-level
error — coherence must come from compression and search. And the danger runs the
other way: the clip at `±5.0` is load-bearing because `cos` terms make single
gradients *large* when `b` drifts. Whether a one-hop rule learns what depth
would have is untested — and where a one-hop rule was measured against plain
counting, counting won (124, 102). The vanishing-gradient
paper is the road not taken: keep `n` (51).

*Where:* `RadixCyclicNN/DECISIONS.md` D-004, D-005,
`Research/VanishingGradientIsAFeature.md` §2,
`RadixCyclicNN/radixnet/backend.py` · **held**

### 80. The structure is independent of what a weight means — one graph, four model kinds
*Confirmed by construction — and it means the fastest engine does not carry the
central claim.* The count/reward idea wanted the same structure with a different
learning rule, so graph, search, decoding and persistence were factored out and
a model *kind* supplies only how it learns. The count model then made one move
worth remembering: every node is created with `a = 0, k = 1`, so `f(z) = 1` and
the score `w · f_p · f_c` reduces to the edge weight — **every inherited method
works unchanged**, and about 700 lines get a complete search, beam, generation
and persistence stack for free. The negative network reused it "for the same
reason": **the structure is independent of what the weights mean.** Four kinds
now share one graph, each learning in its own currency — a learning rate, a
reward, a phase lock, blame. Two consequences: the count model is the one ported
to Go and Rust and the one the benchmark times, and the sine, the project's
central claim, plays no part in it (Q-5); and "interchangeable by design" was
not true until tested — a kind-parity test produced **10 errors and 1 failure**
on the old tree.

*Where:* `RadixCyclicNN/DECISIONS.md` D-020, D-021, D-045, D-050, Q-5,
`RadixCyclicNN/radixnet/countnet.py`, `ModelKit/tests/test_feedback.py` ·
**confirmed**

### 81. Frequency alone cannot move on — weigh an edge's all-time share against its recent share
*Held — a forgetting rule with natural units, never measured against the rule it
replaced.* The count model first weighted an edge by `log(1 + count)`; pure
accumulated frequency cannot forget, so the model could not move on from what it
saw early. The replacement weighs each edge by its **share** of its parent's
traffic twice — over all time, and inside a sliding window of the last
**10,000** traversals anywhere in the graph — with the edge's reward on top:
`P(c | p) ∝ R_all^global · R_recent^window · e^reward`, at the default 0.5 / 0.5
the geometric mean of the two shares. A per-edge exponential decay was rejected
as having "no crisp interpretation and no natural units; the window has both" —
and the window's unit, traversals, later set the half-life of the proposed edge
decay (100) and of RadixDecayNN's clock (104). The window is
a sequence, so corpus order became part of the model's meaning: the Go engine
keeps structure building single-writer and applies the window in corpus order
across chunks.

*Where:* `RadixCyclicNN/DECISIONS.md` D-022, D-041, D-043,
`RadixCyclicNN/radixnet/countnet.py` · **held**

### 82. A sine has a phase — the resonant model
> This was drawn from the idea that a brain operates like an analog computer.
> Hence, we assume sine waves are the standard of information encoding/decoding.

*Refined — the phase separates contexts a trigram merges, on two sentences, and
as a position clock.* RadixNet reads the note as a pointwise sine (24). The
resonant model reads the other half: a sine has a **phase**, phases add along a
path, and signals that meet in phase reinforce. A walk carries one of 8 phase
buckets, advanced by every trigram; each edge learns the circular mean of the
phases at which it was taken, and the length of that mean is a **coherence** —
how context-dependent the transition is, with nothing added to measure it. The
score adds `resonance_scale · coherence · cos(ψ − μ)`, so `P(child | parent,
phase)` is **the first distribution in the system that depends on more than the
parent**, and inversion gets an exact meaning: rotate every `μ` by π. Trained on
`"the cat sat down"` and `"a big cat ran away"`, the node `"at "` has three
children at exactly **1/3** each without the phase; with it they separate
(**0.807**, **0.807** and **0.691** at different buckets), and the model
continues each sentence correctly, where `resonance_scale = 0` answers `"ran
away"` to both and loops on `"sat sat sat"`. The caveat is the default: with no
hash kick the phase is pure position, so what separated those continuations is
where in its line each occurred. It has no decision in the record.

*Where:* `RadixCyclicNN/DESIGN.md` §30, `RadixCyclicNN/README.md` ("The resonant
model"), `RadixCyclicNN/radixnet/resonance.py`,
`RadixCyclicNN/tests/test_resonance.py` · **refined**

### 83. Judgements follow the path, not the edge
*Held — built and priced, never graded.* An edge is the right move in one
sentence and the wrong one in the next, so a verdict counted per edge blurs the
two. Every judgement is filed against **the caller that reached the edge** — the
node before the edge's parent, and the edge: the step in the company it kept —
so `the cat → sat` and `a cat → sat` are counted apart, and `log((correct + ½) /
(incorrect + ½))` joins the edge's weight before the softmax: cheap for the walk
that was right here, dear for the one that was wrong, unchanged everywhere else.
A whole path is rewarded only when the whole answer was right. Second-order
context is kept affordable by laziness — a context is born only when a path is
judged, so training cannot fill the table with a language's second-order counts,
and a 39 MB corpus still trains in **13 s at 409 MB**. This table is what the
least-punished traversal counts (86) and what splits and
merges were silently mangling (85); it has no decision of its
own, and the two documents that cite one cite D-058, which is about agent
failures.

*Where:* `RadixCyclicNN/DESIGN.md` §16.5–§16.6, `RadixCyclicNN/README.md`,
`RadixCyclicNN/radixnet/countnet.py` · **held**

### 84. A dynamic window over the nodes — halve down a binary ladder, grow back at the top
> Implement a dynamic window that is sized in the binary number system. It
> starts at 32, then moves to 16, then 8, then maybe 4. This algorithm will
> split nodes into two parts and assign the same weights and data for traversal
> as well as a heavy connection for the two halves. This process happens
> manually or automatically. Then we size up the windows back to 32 and do this
> process again.

Radix compression only coarsens: every unary chain becomes one node, and a
merged node is a corridor - entered at its first gram, left at its last, nothing
to learn inside and nowhere to branch. The window makes the structure
**breathe**. It is a ceiling on a node's length that walks a ladder of powers of
two - 32, 16, 8, 4 - and back to the top; one step merges what fits, halves
every node that is longer into two halves that carry the **same state,
activation and count**, joined by a **heavy connection** (the node's whole
count, and in the sine model a weight of 8, so the cut is invisible to traversal
the moment it is made), and moves the window. Going down, the halves can learn
apart; back at the top, what nothing branched into is consolidated by the same
lossy merge compression always was. It steps by itself at the end of every
training epoch, or by hand.

Built, node for node, in all three ports. A node `ABCD` becomes `AB` and `CD`
under an encoding whose grams do not overlap; under the sliding trigram it is
`ABC -> BCD` merged and comes apart into those, sharing the pivot `BC` - a half
is never shorter than one gram, because a gram lives in exactly one node.

For the count model "invisible to traversal" was true of the step across the
bridge and false of everything priced after it, until the split and the merge
were made exact inverses for everything the model hangs on an edge
(85); since then the cycle is transparent to feedback given
during it, so the experiment that would price it can run on a tutored model. The
primed pair measured the upward half of the same idea — growing the context one
letter at a time — and the fourth letter made held-out prediction worse
(110). (This entry was 30 until the branch entries 30–41 were merged in.)

*Where:* `RadixCyclicNN/SPEC-DynamicWindow.md`, `RadixCyclicNN/DECISIONS.md`
D-087, D-092 · **held** — what the cycle buys is not measured (its Q-20)

### 85. A split and a merge must be inverses for everything the model hangs on an edge
*Confirmed — by five measured defects, fixed in all three ports.* Compression
and the dynamic window rest on one assumption: cutting a node in two, or merging
two back, changes where the graph can branch and nothing else. For the count
model it was false. A split gave the bridge the node's visit count but none of
its sliding-window traffic, so a branch taken once in fifty traversals was
priced at **0.23** on the compressed graph and **0.029** on the same corpus
uncompressed. A split summed every moved edge's verdicts onto the bridge, so a
clean path through a halved node inherited its siblings' blame and the
least-punished ranking could flip on a halving alone. A merge dropped the
verdicts on the choice it absorbed — a corrected text's price went **0.052 →
0.148** and the corrected-away alternative got *cheaper* — and dropped the dying
edge's reward, so a penalty on a window bridge vanished at the next step up. The
fix makes the two operations inverses: the bridge inherits the window's events
and carries no verdict ("a forced step was never a choice"), and a merge re-keys
what it absorbs; a split and a merge no longer change what the model predicts,
to the bit. Refusing to merge judged edges was rejected: feedback touches every
edge of a path, so the window's ladder would never grow back after a tutoring
session.

*Where:* `RadixCyclicNN/DECISIONS.md` D-092,
`RadixCyclicNN/radixnet/countnet.py`, `RadixCyclicNN/tests/test_countnet.py` ·
**confirmed**

### 86. Traverse by the punishments, not the rewards — and blame is not for sale
*Held — the two searches measurably disagree; which answers better has never
been graded.* Rewards and penalties land in one accumulator, so a step praised
five times and corrected once reads `+4`, exactly like one praised four times
and never corrected. But a reward says *this was good once*; a penalty says
*this was wrong, and here is the correction*. Two answers were built and both
are kept, because neither contains the other. The **punishment** traversal takes
rewards out of the score and prices steps by penalties alone. The
**least-punished** traversal ranks a walk by its *worst* step's blame before its
cost, and counts failures in context against nothing — its first version netted
them against rewards and was bought off in the first test written against it;
now fifty units of reward on a punished step leave it punished. On two sentences
the ordinary search continues with the praised-and-corrected `mat` (**0.99**),
the least-punished one with the unjudged `log` (**5.24**). On 2,000,013
characters with every seventh text punished it disagrees with the ordinary
search on **596 of 3,000** continuations (19.9%) and expands **60,068** nodes
against **918,919**, because refusing a blamed step prunes the beam — effort,
not quality. PrimedRadixPair built the same idea as two ledgers (108).

*Where:* `RadixCyclicNN/SPEC-LeastPunished.md`, `RadixCyclicNN/DECISIONS.md`
D-069, D-075, `RadixCyclicNN/bench/RESULTS.md`,
`RadixCyclicNN/radixnet/penalty.py` · **held**

### 87. An attention band for each n-gram — a correction lands where a gram looks, not where it wrote
> add an attention adjustment band for each n-gram, rewarding the centre of the
> window more than its beginning and end - the way the eye focuses on the centre
> and blurs the left, the right, and the lines above and below while it reads.

*Held — built in three ports, inert where it should be, and ungraded.* A
correction's diff marks the units the teacher changed, and each was charged in
full to the step that **wrote** it — the gram holding the unit at its newest
position. That was already a band, in the limit: all the attention on the unit
the eye has only just reached, while the gram with the mistake at its centre,
which saw it best, was never charged at all. The band is a tent, 1 at the centre
and `1 − blur` at the ends, and each marked unit hands out exactly one charge
shared among the grams that see it: `the cat sat → the bat sat` charges `"e c"`
**0.25**, `" ca"` **0.5** and `"cat"` **0.25**. Three properties are pinned over
eight encodings and four blurs — a unit's shares sum to 1, a whole-text
judgement meets the band not at all, a unit only one gram sees goes to it in
full — so it changes only partial corrections, which is what the copy editor
produces (94). The tent rather than a Gaussian was forced by
parity: `exp` is not bit-identical across Go and the C library. The eye's "lines
above and below" has no counterpart in a one-line text.

*Where:* `RadixCyclicNN/SPEC-AttentionBand.md`, `RadixCyclicNN/DECISIONS.md`
D-086, `RadixCyclicNN/radixnet/attention.py`, PR #36 · **held**

### 88. Train backwards to learn what came before
*Held — built in three ports and every kind; nothing has measured what a
backwards model knows.* A model of this kind learns what *follows*: a text is a
chain of grams from START to END, and every search continues a prefix. Read its
texts backwards and it learns what *precedes* them — but only a query turned
around the same way meets what it read, and only an answer turned back round
reads as text. Training turns every text around in the model's own units — code
points for characters, whole words for a word model, since reversing a word's
letters would make every word a new symbol — so a reversed run is byte for byte
a run over the reversed texts; asking is the caller's half. Nothing marks a
model as trained backwards, deliberately: a flag would rule out one graph taught
both ways and asked either way. The honest limit is the n-gram's: from `"the
lazy dog"` a trigram model cannot tell the `the` before `lazy` from the one that
starts the sentence.

*Where:* `RadixCyclicNN/DECISIONS.md` D-084,
`RadixCyclicNN/SPEC-SearchAndTraining.md` §9,
`ModelKit/frontend/src/backwards.js` · **held**

### 89. Rehearse what was read before, start easy, and never let an easy epoch stop the run
*Held — built off by default in three ports; none of it measured.* A model
trained on a second corpus forgot the first, nothing let a run start easy, and
nothing stopped a run that had stopped learning. **Rehearsal** keeps a bounded,
uniform sample of every text ever read and walks a slice of it *after* each
epoch's new texts, so the end of the epoch — what the sliding window calls
recent — refreshes old material; a punished text is never rehearsed, since a
buffer trained *for* would teach the failure back. A **curriculum** starts on
the shortest texts and adds the rest a step at a time, and **early stopping**
ignores every epoch still inside it: a partial epoch's loss is over fewer,
easier texts, and counting it would stop a curriculum just as it got hard. One
negative result came on the way — a diverse beam built by penalising partial
paths that end where another does changed nothing on real graphs and, tuned
harder, pruned the best path — and one rule kept it portable: nothing new draws
a random number.

*Where:* `RadixCyclicNN/DECISIONS.md` D-078,
`RadixCyclicNN/SPEC-SearchAndTraining.md`, `RadixCyclicNN/radixnet/training.py`
· **held**

### 90. Counters are odometers — cycles are a feature in arithmetic too
*Confirmed as engineering — 25's principle applied to arithmetic, not evidence
for it.* Every count in RadixCyclicNN only goes up, and each would eventually
overflow — in Go an `int64`, and long before that the **53-bit mantissa** of the
JSON number carrying it through a model file, the API and a JavaScript `Number`.
None is unbounded now: each is a two-digit odometer, `total = resets × LIMIT +
value`. The limit is **10^15** — exactly representable as a `float64`, so a
counter crosses a file, a response and a browser unchanged, and four orders of
magnitude under the `int64` maximum, so a whole epoch fits before the next wrap;
the odometer comes full circle after **10^30** events. Wrapping never happens
inside a counting loop — a carry sweep at each epoch's end and before each save
moves the overflow — and **a wrapped model behaves exactly as one that counted
forever**, which 24 counter tests and a cross-language parity case assert.

*Where:* `RadixCyclicNN/DECISIONS.md` D-064,
`RadixCyclicNN/radixnet/counter.py`, `RadixCyclicNN/go/radixnet/counter.go`,
`RadixCyclicNN/tests/test_counter.py` · **confirmed**

### 91. Racy counting lost 0.7% for no speed-up — the speed came from representation, not from the race or the language
*Contradicted as an optimisation; open as a research position.* The Go engine
counts traversals with unsynchronised increments from one goroutine per text: a
collision loses an update, by design. On a million-line corpus it lost **0.7% of
traversals** and gave **no speed-up** over an exact pool of four workers on a
4-core machine, and a later benchmark agrees (Go's exact mode trained at
0.85–1.04× the racy default). The speed that did come was asymptotic: Python
recomputed every weight after every text, Go recomputes only the rows a
traversal touched — "this, not the goroutines, is where the Go engine's real win
comes from" — and bounding the chunks in flight took a 39 MB corpus from **1.7
GB to 393 MB**. A third implementation, in Rust, was written to price the
language rather than the model, behind a harness that refuses to time two
programs until they agree to the bit (134): Rust counts **2.15–2.81×**
and predicts **3.84–6.23×** faster than Go. But three representation choices
carry most of it and none is algorithmic — a trigram packed into a `u64`, a
reused child buffer, `FxHasher` — and the packing was dropped hours after the
table was written, when the encoding dial made a gram arbitrary text; the
published speed-up is for a representation the port no longer has. Q-7 keeps the
other reading open: that lossy, contended counting is what a brain does, to be
stated as a claim if it is one.

*Where:* `RadixCyclicNN/DECISIONS.md` D-040–D-043, D-065, D-072, Q-7,
`RadixCyclicNN/bench/RESULTS.md`, `RadixCyclicNN/bench/README.md` ·
**contradicted**

---

## The radix graph — failure, judgement and teachers

### 92. A standing model of failure — 2NRL's negative phase made permanent
*Held — built and queryable, not measured against the mechanism it may replace.*
2NRL as first stated is transient: phase 1 represents the failure, phase 2
negates it, and the representation is gone. The negative network keeps it — the
same self-compressing graph, but every node and edge exists **because something
went wrong there**, and a softmax over the failure mass is a **failure
distribution**: from a prefix it predicts the ways to fail, as the positive
model predicts the ways to succeed. `judge(text)` answers with risk, peak,
coverage, reasons and the fragments that carry them, and `forget` exists because
evidence never released makes a network permanently pessimistic. Two details
make it learn a *shape*: the tutor names each mistake from a fixed list, because
a free-text critique cannot be tallied, and because "one sentence is one
sentence, not an error" it writes three more with the same mistake at half
severity — after one bad sentence about dogs, "the dogs sits on the mat" is
already suspect, blamed on `sits`. That manufactures 2NRL's consistency
condition. It sharpens Q-15: the count model already skips inversion because it
can punish a path directly, and if failure can be kept, named and vetoed, what
does invert-and-discard still buy? Chess's repulsion arm, which matched or beat
inversion, suggests the answer (60).

*Where:* `RadixCyclicNN/DECISIONS.md` D-023, D-045, D-049, Q-15,
`Research/2NRL.md` §7, `RadixCyclicNN/radixnet/negative.py` · **held**

### 93. A filter must not be taught by what already passed it — guard what a person sees, not what a judge sees
*Held — a principle with a stated blind spot of its own.* Once the negative
network existed, answers still came straight from the positive model, "including
the sentence the tutor had corrected an hour earlier". So the pair now guards
every answer by default: the positive model over-samples and the negative one
vetoes — by accumulated blame, by peak blame on one fragment, or by the
likelihood ratio `log P_neg − log P_pos` — behind a coverage gate, so text that
has never failed is never vetoed on no evidence. It is the GAN idea moved from
training time to **output time**. The principle worth indexing is the exemption:
the loops that *teach* the negative network read the positive model
**unfiltered**, because "a reviewer that only ever saw what already passed the
filter would have nothing left to teach" — filtering the training signal with
the filter being trained is a closed loop that converges on its own blind spots.
If the output goes to a person, guard it; if it goes to a judge, do not. Q-18
states the cost: nothing measures whether the model improved or the filter got
better at hiding its failures.

*Where:* `RadixCyclicNN/DECISIONS.md` D-047, D-048, D-053, Q-18,
`ModelKit/modelkit/duo.py`, `ModelKit/modelkit/critic.py` · **held**

### 94. Blame at the granularity the failure happened at — a correction is a diff
*Held — by the record's own account the most error-prone part of the system, and
never measured against blaming the whole text.* A tutor's grade first reached
the graph as two whole-sentence verdicts, the attempt garbage and the correction
gospel; but a teacher changes a tense or an article, so the whole-sentence
penalty **taxed the trigrams that were right**. Corrections are now aligned
character by character: only the steps that wrote a struck-out character lose,
and an edge both sentences walk is rewarded, never penalised. The same idea
turned the adversarial reviewer into a **copy editor**. A model that writes `"Hi
howe are you??"` has one letter and one mark wrong, and a verdict on the
sentence teaches the negative network that greetings are failures — so the LLM
is asked for the *smallest* change that makes the text right, which keeps the
diff a map of the mistake rather than of the editor's taste, and rating and
correcting stay two calls ("a model asked to rate tends to rewrite, and one
asked to rewrite stops rating"). For the agent, three failures that look alike
get different blame: a wrong answer is diffed against a correct run of the same
task, a call the mediator had to repair blames **the network's own emission** —
blaming the transcript "would teach the network that a well-formed call is a
mistake" — and a refused call only if the network wrote it. A bug shows why it
is error-prone: the correction text was also being cleared, which removed the
blame the diff had just placed.

*Where:* `RadixCyclicNN/DECISIONS.md` D-046, D-058, D-082 (the correction one),
`RadixCyclicNN/radixnet/diff.py`, `ModelKit/modelkit/blame.py`,
`ModelKit/modelkit/ollama.py` · **held**

### 95. The LLM supplies the language; a deterministic rule supplies the decision
*Held — the discipline under all six teaching loops, never measured as an
outcome.* Every loop that teaches the network uses a language model, and each
confines it to what a language model is for. The rule, stated for the tutor:
**the model supplies language, a deterministic rule supplies the decision, and
the record says which.** A run ends with a report card, and the next step up is
read off the marks, never asked for — *advance* when **80%** passed at **8/10**,
*stretch* at 50%, *hold* otherwise — because "a student who is failing must not
be given a harder exercise", and an LLM asked "what level next?" says "harder"
out of the conversation's own momentum. The card's own plan, with no LLM, is the
floor the teacher must improve on, and `source` records which wrote it, "so a
plan is never a hallucination presented as a syllabus". The same shape recurs:
the agent's LLM writes acceptance criteria **before** anything is attempted and
demonstrates the task **only on failure**; the code judge is handed the
objective report so it grades against facts; a 9 out of 10 is learned nine
tenths as hard as a 10. No loop's effect on the network has been measured.

*Where:* `RadixCyclicNN/DECISIONS.md` D-026, D-030, D-049–D-052, D-055, D-057,
`ModelKit/modelkit/tutor.py`, `ModelKit/modelkit/agent.py` · **held**

### 96. Code is the one reward that is not an opinion
*Held — the grounded half of 7, built and not yet shown to teach.* Every other
feedback source in RadixCyclicNN is a judgement — a thumb, an LLM's opinion, a
sibling discriminator. A program runs and prints the expected output, or it does
not. So code generation is `problem → program → sandbox → judge → 2NRL`, and in
the record's words it is **the only reward in the system that does not
ultimately reduce to someone's opinion.** Judging is layered strongest first —
did it run and match, then an in-house style checker with fixed codes, and only
then an LLM handed the objective report — and `judged_by` records which layer
decided. One detail makes the two phases one curriculum: the training text is
`prompt + "\n" + code + "\n"`, so the prefix the network continues when it
writes code itself is exactly the string it learned when the teacher wrote it.
No measurement of a character-level graph learning to write a passing program
exists, and the sandbox's network isolation is best-effort.

*Where:* `RadixCyclicNN/DECISIONS.md` D-030, D-031,
`ModelKit/modelkit/codegen.py` · **held**

### 97. A tool call is text the network writes, so an attempt is one training text
*Held — tool use with no new learning machinery, and no measured autonomy yet.*
The network cannot decide to call a function; it can only emit characters. So a
tool call **is** text — `<tool>web_fetch {"url": "..."}</tool>` — and the
observation is text read back, which makes **a whole attempt, from task to
answer, one ordinary training text** that 2NRL can reward or punish as a unit.
Two consequences are sharper than they look. Every byte of the call format is
part of the model: Go's encoder wrote `{"url":"u"}` where Python wrote `{"url":
"u"}`, and that one space separated a transcript the other side could read from
one it could not. And Dijkstra's cheapest path ends before a call can close, so
the first attempt uses the beam. The cold start is solved without giving the
answer away: when the emission cannot form a call, the LLM **repairs** it into
one valid call against the real schemas, so an untrained network makes progress
and learns the repaired form, and the LLM does the task itself only on failure.
The first usable candidate counts as the network's own move, `autonomy`, and no
value for it is recorded.

*Where:* `RadixCyclicNN/DECISIONS.md` D-056, D-057, D-058,
`ModelKit/modelkit/tools.py`, `ModelKit/modelkit/agent.py` · **held**

### 98. The perpetual self-upgrade has no yardstick
> constantly self-upgrading

*Held — and the record names the way it could fail without anyone noticing.* The
note became the `Evolver`: the model is the generator and a second copy the
discriminator; each generation samples fakes, teaches the discriminator real
versus fake with 2NRL, and feeds the worst fakes back as the generator's
garbage. It runs until told to stop — and Docker's `SIGTERM` once discarded an
entire run, so containers now stop with `SIGINT` and both models are saved in
under a second. What it lacks is a yardstick: there is **no convergence
criterion and no held-out evaluation**. The loop reports the gap between real
and fake scores and takes a widening gap as progress, but the only judge of that
gap is the generator's own sibling, "a weak critic", so nothing detects the
generator getting worse in a way its discriminator likes (Q-4). The author's
stopping rule is "when the thing worked" (Q-14) — the same recognition problem
as the breadth schedule's trigger (68). Until a rule exists,
"self-upgrading" describes the loop, not a measured property of the model;
DISTIL met the same hazard from the other side (128).

*Where:* `RadixCyclicNN/DECISIONS.md` D-011, D-029, D-037, Q-4, Q-14,
`ModelKit/modelkit/gan.py` · **held**

---

## The radix graph — going round, and thinking

### 99. The walk may go round — the hand-over is a reflex taught by experience and overruled by memory
*Refined — the hand-over works when experience teaches it and fails when the
corpus does.* A node is a trigram, so repeated text closes cycles, and they are
never removed; the search runs over `(node, chars emitted)`, which makes a cycle
finite. The metacognitive half — hand over instead of looping — was missing for
most of the project's life, and the cost showed: a voice with nothing new to say
repeated one line for every remaining turn (**26 × "park" in 40 turns**). It now
exists at three grains. A **stutter** — one to four words repeated immediately —
is the loop's own signature, and a voice that catches one backs up to where the
walk went round and searches again from the longer prefix. Each rethink teaches
the graph a `p → BACK` edge, a **reflex** competing for the node's probability,
so later searches hand over there; three conversations on the sample corpus
taught 3 nodes, then 2, then 1. And in the resonant model, where returning at
the *same* phase is a true loop, a metacognitive layer is the **memory** of one
cycle and outranks the reflex once it has evidence — once its prior stopped
overreaching (raw, it stood at **ride 78 / escape 33**, deciding a never-seen
cycle as if by a hundred observations). The other direction failed, measured:
taught from the corpus instead, `BACK` reached **P = 0.79** at `"he "` and
vetoed all seventeen children where the corpus had declined one, and generation
fell from `the sun, the field, the park` to `lond, ever, water` — so that route
is off by default. The effect on output quality has only been described, never
measured.

*Where:* `RadixCyclicNN/DECISIONS.md` D-001, D-061–D-063, D-068,
`RadixCyclicNN/DESIGN.md` §30.3–§30.3.1, `RadixCyclicNN/radixnet/metacog.py`,
`ModelKit/modelkit/dialogue.py` · **refined**

### 100. A lesson that stops the walk cannot be unlearned by walking — forget on the graph's own clock
*Held — the trap is measured; the cure is specified and was never built here.*
The model corrects itself by walking, being wrong and being corrected. A
hand-over closes exactly that door: once `BACK` is a node's cheapest child the
search goes round the node, no walk through it can turn out fine, and nothing
can contradict the lesson. **A hand-over taught in error is self-sealing.** On
the sample corpus two lessons close a node in the count model (`BACK` **1.837**
against the cheapest sibling's **2.079**; after four, **0.334** against
**3.164**), and nothing reopens it. The argument is about the clock: decay
clocked on walks through the node cannot work — they have stopped by
construction — so the clock is **graph time**, traversals anywhere, with a
half-life of 10,000, the count model's own window (81). Rewards fade
and counts do not, because history is not rewritten; the sine model's weights
fade toward the node's own mean, which commutes with 2NRL's inversion. The
honest cost: a model that only converses never forgets. RadixDecayNN later built
the clock and the half-life — on the counts themselves, the choice this spec
rejected (104).

*Where:* `RadixCyclicNN/SPEC-EdgeDecay.md`, `RadixCyclicNN/DECISIONS.md` Q-1 ·
**held**

### 101. Thinking is a fourth sentinel that faces both ways — and the thinking a client sees is the search's own trace
*Held — a research claim, built in three ports, with nothing measured.* Between
noticing a repeat and backing out of it there was nothing, and a model asked
about a text had no words of its own to answer in. `THINK` is learned the way
`BACK` is and used the other way round as well. Its **in-edges** are taught by
events — a rethink, a question asked about a text, a thought questioning itself
— so the model learns *where* to stop and think. Its **out-edges** are where
thoughts begin: a thought is a text trained from `THINK` instead of `START`, so
the model learns how its thoughts open without a word of them leaking into what
it says — one graph, two origins, separated by their first edge alone. A model
taught no thoughts thinks nothing and says so. The thoughts come from an LLM's
*reasoning* rather than its answers, because the reasoning is in the right
register — short, hedged, self-questioning. And when the model is served in
today's chat format, **the thinking a client sees is the search's own trace** —
what it looked for, how many paths it weighed, what the guard vetoed, where it
backed out — in fixed lines every port writes character for character, never
prose generated about the walk; a system prompt is "accepted and not read, and
the thinking says so", because a rendering is exactly where a system built to
say what it did could start to pretend. What is open is the rate: how soon a
node starts questioning, and nothing decays it.

*Where:* `RadixCyclicNN/DECISIONS.md` D-080 (the THINK one), D-085,
`RadixCyclicNN/DESIGN.md` §5.1.2, §36, `ModelKit/modelkit/thinking.py`,
`ModelKit/modelkit/assistant.py` · **held**

---

## Without the cycle, and with forgetting — RadixAcyclicNN and RadixDecayNN

### 102. How deep should the context be? Held-out prediction stops improving at three to four grams
*Refined — deeper context helps held-out prediction only to three or four grams;
past that it adds recitation and narrows the options.* The unbounded tree holds
every depth at once, so the context window becomes a parameter of the query and
its depth a measurement. By counts on held-out prose, **one gram of context is
the cyclic graph to the digit** (0.514 held-out accuracy); accuracy rises to
**0.579 at three grams and 0.581 at four**, then stays (0.578 at eight, 0.579
unbounded) while training accuracy climbs to 0.978 — deeper windows only recite
— and the actual next gram is on offer **0.88 → 0.68** of the time. With the
learned one-hop rule deciding instead, the shape is the same and lower
everywhere (**0.461** at one gram, **0.558** unbounded): ten epochs had not
brought the shallow softmaxes up to their counts, so plain counting beat the
learned rule at every window, as it did in FilterBankRadix (124). The
learned rule ran at `b = 1/3` (24), and a window that forgets can loop, so the
walk carries a clock.

*Where:* `RadixAcyclicNN/README.md` ("Token by token"),
`RadixAcyclicNN/DESIGN.md` §11.4, `RadixAcyclicNN/results/window_results.json` ·
**refined**

### 103. A query is a traversal — what the model says, it becomes likelier to say
*Confirmed — asked quietly, nothing moves.* `RadixDecayNN` strips the tree to
one number per node, `seen`, under three rules: it decays, every traversal adds
one, and **a query adds too** — saying something is a traversal of what is said,
so the model remembers what it has said as it remembers what it read. Said ten
times from `the`, `the bear sleeps through the winter` goes from **0.027 to
0.224** probable, and the branch it is decided at sharpens (entropy 3.76 → 3.38
bits); the same ten questions asked quietly leave it at **0.027**. Sampled at
temperature 1, thirty sayings wander over 15 texts and still drift, 0.025 →
0.043: a habit forms by chance and then feeds itself. On prose, ten sayings take
the said text from **0.067 to 0.467**. A saying and a reading weigh the same, so
what is said can outweigh what was read — the deliberate opposite of the cyclic
model's edge-decay proposal, which lets nothing move "from a query"
(100). Nothing bounds the loop, and which behaviour is right is
untested on any task.

*Where:* `RadixDecayNN/README.md` ("A habit forms"), `RadixDecayNN/DESIGN.md`
§7, `RadixDecayNN/results/` · **confirmed**

### 104. A memory fades on the model's own clock — and a life shorter than a reading cannot hold the reading
*Confirmed on the decay's own curve.* `seen` fades against traversals anywhere —
no epochs, no wall time — as a half-life, a linear loss, or not at all, which is
the count model back. Reading is expensive on that clock (**6,845** traversals
for the 60-line sample corpus) and saying is cheap (6 for a 36-character text).
Read once, then silence, at a life of 10,000: a quarter of a life loses nothing
(**57 of 60** recited); at **one life 29% of nodes remain and none is recited**.
Under a half-life an option is never gone, only faint, so misses stay near 3%,
while linear decay loses visits outright and its misses climb to **81%**. On 800
prose lines at a life of 100,000 the reading itself took **298,146 traversals —
2.98 lives** — so the model finished with 232 of 800 texts recited where no
decay recites 705: it had forgotten its first texts before it reached its last.
`life` means nothing until it is set against what is put on the clock. One run
per setting, scored on the texts read.

*Where:* `RadixDecayNN/README.md` ("A memory fades"), `RadixDecayNN/DESIGN.md`
§4, §10, `RadixDecayNN/results/experiments.log` · **confirmed**

### 105. Use keeps a memory, and reading brings it back — only what is used, and only what is re-read
*Confirmed.* After reading, the model sits through twenty silences of a tenth of
a life, each broken by saying one text from its first eight characters. Said,
the text is refreshed by its own saying — **40 traversals** in all — and is
**still recited after two lives**, while the rest of the corpus fades to 13% of
its nodes; said quietly, or not said, it goes with the rest. Forty traversals of
use keep what six thousand of reading could not. A saying refreshes only what it
arrives at, so a long context can outlive its own suffixes and **the model can
know a text from its beginning and not from its middle**. Recovery is by any
traversal, with no retraining step: faded four lives (1% of nodes, nothing
recited), then every third text re-read once, and **19 of those 20** are recited
again and **0 of the other 40**. On prose, where one reading spanned three
lives, the texts later chosen for re-reading had been held **80 of 266** after
the first reading; re-read on their own they come back **232 of 266**. What was
read in a long stretch is partly gone before the stretch ends.

*Where:* `RadixDecayNN/README.md` ("Use keeps a memory", "Reading brings it
back"), `RadixDecayNN/DESIGN.md` §6–7, `RadixDecayNN/results/` · **confirmed**

---

## Priming instead of growing — PrimedRadixPair and LatentRadixPair

### 106. Prime, don't grow — learning is counting at a computed address, and the tokenizer decides the depth
> I want to build a new model in a new directory. Two Radix Tree's, connected at
> the final nodes. This is done by "Priming" the Radix Tree with every potential
> option (brute-force insertion of every option) …

*Refined — as mechanics it is exactly what was claimed; as a model it is
untested.* Every radix model before it grows, and every structural decision — a
merge that hides a branch, a split that comes late — is a place to be wrong. The
primed tree decides nothing structurally, ever: every sequence of `1..L` units
over a closed vocabulary exists before the first text, as slots in flat arrays
whose address is the path read in base `R` — no pointers, no allocation, no
split or merge. The literal brute-force insertion asked for is kept as an oracle
and produces the same tree at every size tested, and training is `L` array
increments per unit: **4.36 million a second** in pure Python for letters at `L =
4`. The price is `R^L` whatever the data, so **the choice of tokenizer is the
choice of depth**: under a 4,194,304-node ceiling letters prime to `L = 4`,
phones to 3, the core lexicon's 1,502 syllables to 2, and GPT-2's whole
vocabulary to `L = 1` — a unigram; one GPT-2 token of context would be
2,526,117,861 nodes, 56 GiB. The hypothesis the design exists for — that priming
beats growing on little data — is its own open question, and the comparison
against FilterBankRadix and the count model its requirements asked for was never
reported.

*Where:* `PrimedRadixPair/PRD.md`, `PrimedRadixPair/DESIGN.md` §6, §16,
`PrimedRadixPair/README.md` — on branch `feat/primed-radix-pair` · **refined**

### 107. Connected at every level, not just the final nodes
> … then connect the 2nd Radix Tree to the first at every single level/node
> where the nodes are equal (not just the final nodes).

*Confirmed on both sides of the rung — reading and rewarding — with one ordering
that failed.* The count tree and the reward tree share one address space, so the
rung between equal nodes is the same id read twice. **Reading:** a prediction
visits every context length from the deepest to the root, and summed over every
fall that is exactly FilterBankRadix's every-context-length fold (identical to
1e-12); on held-out sentences it beats one fall straight to uniform in every
row, by 0.1 to 2 bits per unit — letters at `L = 4` with no smoothing, **2.888
against 3.242**. **Rewarding:** a reward written at every level lifts the
rewarded unit under a *different* context sharing its last unit by **+0.92
bits** (letters), **+1.35** (phones) and **+1.15** (syllables); written at the
final nodes only, by **0.00**. "Not just the final nodes" is what lets a reward
generalise. Two caveats: at one syllable of context the root is the only shared
level, so an unrelated control is lifted as much (+1.21); and the success
criterion's other half — one fall beats no fall — failed in five of six rows
(3.748 against 3.604), which the README does not mention.

*Where:* `PrimedRadixPair/README.md` ("What was measured"),
`PrimedRadixPair/DESIGN.md` §9, `PrimedRadixPair/PRD.md` §7 — on branch
`feat/primed-radix-pair` · **confirmed**

### 108. What was read and what worked, kept in two trees — and a punishment cannot be bought back
> The second tree is a reward-focused tree that is rewarded based on correct
> outcomes.

*Confirmed on the author's own case.* The count tree is written only by reading;
the reward tree only by the outcomes of what the model produced, and a
punishment is a reward with a negative sign. But reward and penalty are **kept
as two sums**, because a step rewarded five times and punished once must not
read like one rewarded four times and never punished when the question is which
way has gone wrong least — the structural form of the least-punished traversal
(86). Two sentences read, `mat` rewarded once at 5 and
punished once at 1: the reward traversal continues **`mat`**, the punishment
traversal **`log`**, and fifty more rewards change neither — **unbuyable**. An
outcome credits only what the model produced; crediting the whole sentence would
also reward `he ` → `c` in `the cat`. The first draft read the second tree as
the first read backwards — a mirror, scoring from both ends — and was dropped,
unbuilt, when the second tree became a reward tree. No judge is wired to it yet,
and `LatentRadixPair` quietly reverses one default: a rewarded text is also
read.

*Where:* `PrimedRadixPair/DESIGN.md` §8, §9.2, `PrimedRadixPair/README.md` — on
branch `feat/primed-radix-pair` (the mirror draft on
`claude/blissful-euler-ug38mh`) · **confirmed**

### 109. On a primed tree the cheapest path is the wrong decoder
*Confirmed while building, and not foreseen.* The family predicts by the
cheapest single path, Dijkstra over `−log P` (78). On a primed tree
that search can pay for a rare unit at the deepest level because the context it
then stands in was never read and **falls to the root for nothing**, where
frequent units are cheap: a cheap path through an improbable step. The exact
fold sums over every way of falling and is not fooled, so the fold's greedy walk
became the default and the cheapest path the option — the reverse of the family.
On a grown graph the rare edge does not exist, so the failure belongs to
completeness: a tree that holds every option holds every trap, and a decoder
that optimises one path will find them. `LatentRadixPair` drops the
cheapest-path walk for the same reason. No rate of such paths is reported.

*Where:* `PrimedRadixPair/DESIGN.md` §10, `PrimedRadixPair/README.md`,
`LatentRadixPair/DESIGN.md` §4 — on branch `feat/primed-radix-pair` ·
**confirmed**

### 110. A dynamic tokenization — the window grows one letter at a time
> … a new type of dynamic tokenization, dynamic meaning the window size — first
> 1 letter, then 2 letters, then 3 letters, and so on.

*Refined — the window can grow for free, but only while its deepest contexts are
trusted; on sixty sentences the fourth letter made prediction worse.* A level's
offset does not depend on the depth, so **a tree of depth `L` is the first slots
of a tree of depth `L + 1`**: growing is an append and one counting pass, and a
tree grown from one letter to four was checked equal, rung by rung, to one
primed at that depth. The token grows with the window — where the tree is
certain, units run together, so at `L = 4` `bird␣` is one token. **The
tokenization is the window**, and `R` never grows: byte-pair encoding grows `R`;
this grows `L`. On letters, 48 sentences read and 12 held out: **4.132, 3.244,
3.123, 3.204** bits/char at `L = 1…4`, because the new deepest contexts at `L =
4` were trusted at only 0.62 on average. The readiness rule — grow at a mean
trust of 0.75 — reads **0.80 at `L = 3`**, so it would still take the step that
made prediction worse: it reports the trust of the level it has, not the one it
would add, and stops one rung late. Specified and measured, not built; it is the
primed counterpart of the dynamic window (84) and the only measured
window growth in the repository.

*Where:* `PrimedRadixPair/SPEC-DynamicTokenization.md` — on branch
`feat/primed-radix-pair` · **refined**

### 111. Tokenize the context, not the units — a learned 12-bit code of the last 16 bytes
*Refined — a learned code is a better tree key than raw bytes, bit for bit, but
the objective, not the bottleneck, makes it one.* A primed tree over a learned
token set hits its budget from both sides: a vocabulary that can encode anything
needs the 256 bytes and leaves one token of context, and a small one cannot
represent arbitrary bytes. The way out is to see which half of tokenizing must
be lossless. The **units** must — the output is read back — so they are bytes;
the **context** need not, since it only selects a node. So a small network
compresses the last 16 bytes into a **12-bit code** — finite scalar
quantisation, an autoencoder trained first, as latent image models do — and both
trees are primed over every prefix of it. On held-out Markdown it scores **3.189
bits/byte**, against 3.892 for the last raw byte and 3.025 for the last two:
twelve learned bits worth about fourteen raw ones, from 4,369 primed nodes where
a dense two-byte tree needs 65,536. Masking the finer symbols orders them coarse
to fine, so falling back to a parent is falling back to a coarser description of
the same context. A code trained only to reconstruct was worth about 11 raw bits
— "no better a key than the bytes it replaced" — but that ablation also doubles
the training steps and changes recency, so it does not isolate the objective.
Three raw bytes still score 2.408.

*Where:* `LatentRadixPair/DESIGN.md` §1–2, §10, `LatentRadixPair/README.md` — on
branch `feat/primed-radix-pair` · **refined**

### 112. The first symbol of a learned code replaces a market of slices
*Held.* The layer planned above the primed pair was a market: many slices, each
with its own tokenizer; a recogniser choosing among them as GREN's coverage
would; an auction on GTMNN's seats; settlement by outcome; slices calling
slices. The context tokenizer folds all of it into one code. The **first
symbol** partitions every context into learned classes and does the recogniser's
job; the subtree under it is that class's slice; slices calling slices become
the fold's back-off to a coarser description. Nothing is bid for. It is 27's
idea — similar games in similar regions — reached from text, with a region as a
learned code prefix rather than a cluster of a graph; whether the partition
separates thinking, games and language is unmeasured.

*Where:* `LatentRadixPair/DESIGN.md` §7 — on branch `feat/primed-radix-pair` ·
**held**

### 113. Verdicts in outcome units
*Held — argued from the size of the space a verdict comes from, not measured.*
One reward tree is meant to hold verdicts from games and from English, so a
verdict is worth **strength / outcomes** per rung, where `outcomes` counts the
space it judges: **1/2 in a two-outcome game, 1/69 in English**, one part per
phone of the phonetic tokenizer. The more finite the space, the more one verdict
says: from a uniform node a single verdict moves its outcome to 0.62 in a
two-outcome game and by a hundredth of a percent in English. 69 is one of
several defensible counts the design lists, the nodes actually hold 257 byte
outcomes, and nothing measures whether the scaling makes mixed feedback behave
better than raw amounts.

*Where:* `LatentRadixPair/DESIGN.md` §5 (D-7) — on branch
`feat/primed-radix-pair` · **held**

---

## Text as sound

### 114. A sound is a unit — *night* and *knight* are one thing to the model
*Held — built three times; the comparison that would answer it has never been
run.* The phonetic tokenizer makes a text's alphabet the language's sounds: 39
phonemes with their stress, `#` between words, punctuation as pauses — a fixed
alphabet of 92 symbols, nothing learned from a corpus, so the rule against a
learned vocabulary survives. *Night* and *knight* become one sequence and *cat*
and *cut* differ in one symbol, and the claim is that a model over them learns
"rhyme, alliteration, the way a sentence ends" rather than spelling accidents.
The encoding dial reads it as one more unit, so the graph, the search and the
file needed no change, and the primed pair made it its main tokenizer — phones
in, phones out, the English spelled back on request. What is measured is size
and cost, not learning. A tree over phones holds **1,767 nodes against 2,224**
for characters on the sample corpus and recites the same lines. The
letter-to-sound rules get **38%** of the full dictionary's words exactly and
**82%** of its phones (half the dictionary is names), so an unread word's sounds
are a guess; homophones collapse, so spelling back means choosing a word; and a
phone is short, so two phones of context put *a cat* and *the mat* behind the
same `AH0 #`. The only held-out numbers are per unit and do not compare: **4.102
bits per phone** against **3.123 per character**.

*Where:* `PhoneticTokenizer/DESIGN.md` §1–4, `PhoneticTokenizer/README.md`,
`RadixCyclicNN/DECISIONS.md` D-080 (the phonetic one),
`RadixAcyclicNN/README.md` ("Sounds"); the held-out numbers in
`PrimedRadixPair/README.md` on branch `feat/primed-radix-pair` · **held**

### 115. Acoustic units — an alphabet learned from audio, small enough to recur
*Held — the units survive the vocoder; what a model learns from them is
unmeasured.* Speech has no dictionary, so for audio the question changes from
*learned or rule-based* to *what is learned, and what the units should be*. The
answer is k-means over log-mel frames (25 ms every 10 ms, 40 bands, the
utterance's mean removed) into **64 units**, a run of one unit collapsed to one
token such as `q17`, so a model can be trained on recordings with no transcript
at all. The argument is about inventory size: a count graph learns exact
repetitions, so its symbols must recur at about the rate sounds change —
"forty-odd phones at ten to fifteen a second recur; a thousand-way codec token
at fifty a second never does" — and sixty-four units, runs collapsed, sit where
the phones do. The codec half is measured: a frame heard again after the round
trip is the same unit about **86%** of the time streamed and **92%** polished.
The learning half is tested only structurally — a heard recording has no more
unknown transitions than noise — and the codebook is the synthesizer's own voice
(28 sentences, 2 voices, 212 s); no real speech has been heard.

*Where:* `PhoneticTokenizer/DESIGN.md` §5,
`PhoneticTokenizer/phonetok/acoustic.py`, `RadixCyclicNN/DECISIONS.md` D-082
(the acoustic one), `RadixCyclicNN/tests/test_acoustic_units.py` · **held**

### 116. Keep the source, learn the filter — a neural vocoder for the acoustic units
*Refined — a learned filter roughly halves the distance to held-out speech, but
only once the phase was left to the excitation, and by the unit round trip it
collapses.* A centroid is the average of thousands of frames, and a voice made
of averages is blurred. The design on `main` refuses a neural vocoder — weights
to ship, a framework to run, nothing a port could reproduce; this one is a small
network trained in numpy with hand-written gradients, one JSON file beside the
codebook, run bit for bit by Go and Rust. On the speech frames of four held-out
recordings its log-mel distance to the original is **1.4143 against 3.0878** for
the centroid vocoder with 32 Griffin-Lim passes. A head that predicted phase
failed first — within a run of one unit the input is the same frame after frame,
so "the pitch collapses onto the frame rate" — and keeping the pulse-and-noise
excitation, learning only two gains per bin, fixed it. By the unit round trip it
reads **0.1172 against 0.8037**, a number the commit message does not quote:
that ruler favours the centroid by construction, since its frames *are* the
centroids (140). Every recording in the test is the synthesizer's own
speech.

*Where:* `PhoneticTokenizer/DESIGN.md` §5.5,
`PhoneticTokenizer/phonetok/neural.py`,
`PhoneticTokenizer/phonetok/data/acoustic.vocoder.json`,
`RadixCyclicNN/DECISIONS.md` D-090 — all on branch
`claude/phonetic-tokenizer-package-b6yfai` (on `main`, D-090 is a different
decision) · **refined**

### 117. The traditional LLM tokenizer is one more setting of the encoding dial
> Implement a new encoder and decoder layer that follows the traditional LLM
> tokenizer.

*Held — built exactly, in three ports, and graded against nothing.* The radix
graph refused a learned sub-word tokenizer because it needs a corpus before
training can start and freezes a vocabulary. Byte-level BPE answers the first
objection — every byte is a token, so nothing is unreadable — and the corpus can
be one the repository already has, so the question became *where*, and the
answer was the dial: a token is one more kind of unit (`token:3:1`). The graph
needed more than a tokenizer, because a label is text cut back into units
constantly, and GPT-2's written form "marks the space and leaves the glue
unmarked — backwards for text", so no prefix a person types could match a label.
The fix writes a space-led token bare and glues the others on with `⁀`, proved
idempotent and tested on every run of up to six units. The bundled merges are
**4,096 ids** learned from the repository's own 77 Markdown files, reading its
prose at about **3.0–3.5 characters a token**; under `token:3:1`, *the cat sat
on the mat* is one node of six tokens. Whether fewer, larger units predict
better is its Q-21, unmeasured.

*Where:* `RadixCyclicNN/SPEC-Tokens.md`, `RadixCyclicNN/radixnet/bpe.py`,
`RadixCyclicNN/DECISIONS.md` D-090 and Q-21 — all on branch
`claude/busy-hopper-2ji25l` · **held**

### 118. Mark a model of sounds in English, teach it in sounds
*Held — built into every loop in three ports; whether a model of sounds learns
better for it is unmeasured.* A model over phones answers `DH.AH0 # K.AE1.T #
S.AE1.T`, and "an LLM cannot mark ARPAbet — it rates it as nonsense, or
'corrects' it into different ARPAbet", so none of the teaching loops could teach
a model of sounds. Now the reader and the learner see different texts: the
tutor, reviewer, copy editor, critic, chat partner and judge read the English
the sounds spell, and what they write back is read into sounds and diffed where
the graph lives, so only the sounds that differ are blamed. **The model is
credited for its own text, never for a respelling** — a thumbs-up on the words
would teach it a text it never produced. A correction to a homophone (*their*
for *there*) is the same sounds and rightly blames nothing; the honest cost is
that the tokenizer's lexicon decides part of every mark, so sounds that spell
*why* where *y* was meant are marked wrong.

*Where:* `RadixCyclicNN/DECISIONS.md` D-090, D-091,
`RadixCyclicNN/radixnet/encoding.py` (`spell`), `ModelKit/modelkit/llm.py`
(`reader_text`), `ModelKit/tests/test_teaching_in_words.py` · **held**

### 119. Talk with the model — the conversation itself is what teaches it
*Held — the loop runs on all three servers; the hand-over it exists to produce
is unmeasured.* Every utterance is one turn: heard (the words and the waveform
behind one token, or a model of acoustic units' own units), **trained on before
the reply**, answered, and spoken back through the model's own voice. A model
that has heard three sentences cannot answer, and with nobody answering for it
the conversation would end before it learned anything — so when the model has
nothing to say, an LLM answers *as the model's voice*, and that reply is taught
to it. A conversation that starts with the LLM answering is meant to end with
the model answering by itself; nothing measures whether it does. One rule
protects the data: while the reply plays the ear is closed, because a model that
heard its own reply through the microphone would learn it as the person's words.
The first build's spoken path never paired a single recognised word — dictation
and the ear kept time on different clocks.

*Where:* `RadixCyclicNN/DECISIONS.md` D-089, `ModelKit/modelkit/voicechat.py`,
`ModelKit/tests/test_voicechat.py` · **held**

### 120. Any modality is text — and a text the model was taught is its own answer key
*Held — an elegant closure, built and given a tutor, and for sound abandoned on
an argument never measured.* The model reads trigrams and nothing else, so every
other modality became text. Images run generation **backwards**: the Stable
Diffusion VAE's encoder turns a picture into its latent, 8× smaller on each
side, quantised to signed bytes and written as base64 behind a header; decoding
runs the forward process and repairs whatever it is given, so a half-learned
prediction becomes a picture rather than an error. Speech became two texts
behind one unique token — the words, and the waveform as mu-law bytes, chosen by
measurement (about **2%** relative error down to a whisper, where linear 8-bit
is at **38%**) — so in the cyclic graph the words and the sound leave the same
node. This closes a loop nothing else can: **a text the model was taught is its
own answer key**, so a recall tutor opens a taught text, lets the search
complete it, runs the result back through the codec and marks the payload out of
10, with no LLM and no transcriber. None of its marks is reported. The acoustic
units then replaced waveform-as-text for sound (115), on the argument
that a count graph "cannot generalise over [a waveform] because a waveform never
repeats" — with no held-out score cited; and the image half rests on the equally
unmeasured claim that trigrams over a latent's base64 are "more than noise".

*Where:* `RadixCyclicNN/DECISIONS.md` D-036, D-054, D-066, D-082 (the acoustic
one), `RadixCyclicNN/DESIGN.md` §25, `ModelKit/modelkit/vision.py`,
`ModelKit/modelkit/speech.py`, `ModelKit/modelkit/recall.py` · **held**

### 121. A sound, drawn — the picture is the sound once it carries the phase
*Refined — a grey plane holds a sound's magnitudes almost exactly and cannot
hold its phase; give the phase a channel and the samples come back byte for
byte.* AudioImage writes a spectrogram as a picture — time across, frequency up,
loudness as ink — in a 16-bit PNG that carries its own settings, so `decode`
turns even a hand-drawn picture back into a WAV. The pixels hold magnitudes to a
relative error of **2.2 × 10⁻⁴** at 16 bits; what a grey pixel cannot hold is
the angle, and Griffin-Lim guesses it back with **4.1%** spectral error on a
melody at 64 passes. Writing the phase into the blue channel, only within 60 dB
of the peak (**5.7%** of the melody's pixels), brings the melody to **0.46% and
+44.1 dB** at 8 bits in a file no larger than the grey one; a fourth channel
holding the encoder's own residual makes the WAV **byte-for-byte identical**
(80.3 KiB against a 90.5 KiB melody, but 292 KiB against 88 KiB for noise, which
is nothing but phase). Only the linear frequency axis is exact, because an angle
cannot be interpolated. That a model writing these rectangles would write sound
is stated and untested.

*Where:* `AudioImage/DESIGN.md` §1–6, §11, `AudioImage/README.md` ("What
survives the round trip"), `AudioImage/audioimage/codec.py` · **refined**

### 122. One node per hertz, reading vertical strips — it cannot beat "the next strip is this one"
*Contradicted — six arrangements measured, none beats persistence, and an
untrained network scores exactly the baseline.* Over the AudioImage plane each
node owns one frequency and learns by tabular Q-learning how loud it will be in
the next strip. "The next strip is the same as this one" scores **0.026** mean
level error on a melody; the best arrangement scores **0.030**. Two real
mistakes were fixed on the way — a discount of 0.85 on a recording, where the
action cannot change the next strip, scored **0.195** against 0.119 at zero;
naming the *change* rather than the level halved that to **0.063**, because
**77%** of node-steps do not change — and more history, pooling and an evidence
threshold did not close the gap. The reason is in the data: *when* a node
changes is decided by what it cannot see — a note stops because the player
stopped — so the missing information is in the other nodes' present, not this
node's past. An untrained network warm-started to hold scores exactly the
baseline, because holding *is* the baseline (39).

*Where:* `AudioImage/audioimage/qlearn.py` (module docstring),
`AudioImage/tests/test_qlearn.py` · **contradicted**

---

## Filter banks — FilterBankRadix

### 123. Layer 1 is an activation filter feeding a set of radix trees
> Layer 1 is an activation filter. It feeds a set of Radix Tree networks.

*Refined — the routing learns, but it pays only where the context cannot carry
the fact it supplies.* A bank of learnable units reads a segment's character
histogram and emits an address; the address picks one of several radix trees,
and only that tree predicts. A learned filter beats the same filter left frozen
by **0.22 bits/char** and a random split by **0.44**, so the routing does
something. But at depth 5 one tree over the whole corpus (**2.786**) beats the
bank even with a source-perfect router (**2.873**): splitting costs every expert
data and costs the bank the prefixes the sources share. Both sweeps cross. At
depth 2 being told the register is worth **0.20 bits/char**; the learned filter
wins at depth 4 (**2.807** against one tree's 2.814); one tree wins at 5–6. As
the corpus grows, the oracle's deficit runs **0.108, 0.095, 0.006, −0.015**.
Routing hands the model a fact — *which register this is* — that a deep tree
already has in its context. And the best partition is not the labelled one: tanh
units routing on the histogram alone reach **2.831** and beat the source-perfect
oracle at a purity of only 0.433. One corpus, four registers, one seed on the
sweeps; addresses can die but are never born.

*Where:* `FilterBankRadix/DESIGN.md` §1, §6–7, §12, `FilterBankRadix/README.md`,
`FilterBankRadix/results/` · **refined**

### 124. Learn a radix tree's branch probabilities with the one-hop rule
*Contradicted at this scale — counting the traversals beats the rule, and the
gap is optimisation, not capacity.* FilterBankRadix's trees score a branch as
RadixCyclicNN does — child × parent activation × edge weight, softmaxed over the
siblings, trained one hop at a time. Replace the softmax with relative traversal
counts and learn nothing: the same trees score **2.614** against **2.786**
unrouted and **2.740** against **2.873** routed, in one second of work against
forty. It is not capacity — `w_c f_c = (log p_c)/f_p` reproduces any
distribution — nor over-fitting, since counting also has the lower training loss
(0.872 against 1.022); it is under-training: **2.842, 2.786, 2.726, 2.677** at
1, 3, 10 and 30 epochs, ten times the training buying 0.11 bits and leaving 0.06
on the table. RadixAcyclicNN found the same at every context depth
(102). One 58k-character corpus against an undiscounted n-gram — but a
rule this repository uses everywhere reaches a worse answer than counting, more
slowly.

*Where:* `FilterBankRadix/fbradix/tree.py`, `FilterBankRadix/DESIGN.md` §5.2,
`FilterBankRadix/README.md` ("What the one-hop rule is worth"),
`FilterBankRadix/results/baseline_results.json` · **contradicted**

### 125. Locate the gap with measurements the architecture may never make
*Confirmed as a method — it put the gap in layer 1, not in the signal.* When a
learned router trails an oracle, the gap could be the signal, the representation
or the learning rule, and four supervised probes — things the architecture
itself is never allowed to do — separate them. **The signal is perfect**: given
true experts, the cheapest one for every one of 640 segments is the segment's
own register (NMI **1.000**). **Layer 1 falls short twice**: fitted directly on
the labels it reaches only **0.69–0.72** purity, unsupervised **0.42–0.49** —
half the gap representation, half the rule. The address code argued to be "the
single largest decision in layer 1" measures inside the seed spread (2.908
against 2.887 bits/char). And the probe caught its own error: at 25 epochs one
split read **0.500, exactly chance**, which looked like a hard limit on sign-bit
addresses and was a dead unit (`|a·b|` had collapsed to 0.0100); at four times
the epochs the same split read **0.777**. Like exactly 1.0× (39), exactly chance
is a number to distrust.

*Where:* `FilterBankRadix/fbradix/experiment.py`, `FilterBankRadix/DESIGN.md`
§4.5, §9, `FilterBankRadix/README.md` ("Where the routing gap is"),
`FilterBankRadix/results/probe_results.json` · **confirmed**

### 126. Price the counterfactual out of sample, and never let ignorance be cheaper than knowledge
*Confirmed — the first correction decided whether the experiment measured
anything at all.* The only thing the trees tell the filter is a price: how many
bits each expert would spend on a segment. The first version priced the segments
each expert was built from, which it had memorised — **99.5–100%** of them were
already at their cheapest expert, with a mean advantage of **0.000 bits**,
against **53.1–70.0%** and **0.21–0.31 bits** on unseen segments — so at two of
three seeds the filter never moved and **the learned arm was bit-for-bit its own
frozen control**. Each fold is now priced under experts built without it. The
price was wrong a second way: an expert that had never seen a context paid the
smoothing floor, about **eleven** bits a character, while one that knew the
context but was unsure paid **four** — so the filter learned to route text to
the tree that knew least about it. Interpolating with a shared prior makes
knowing cheaper than not knowing by construction, and folding that prior through
every context length took one tree from **2.998 to 2.808**. *A counterfactual
price is a signal only when the pricer did not fit what it prices, and when
ignorance cannot undercut it.*

*Where:* `FilterBankRadix/fbradix/bank.py` (`FOLDS`, `refilter`),
`FilterBankRadix/DESIGN.md` §5.3, §6.2–6.3, `FilterBankRadix/README.md` ("What
went wrong first") · **confirmed**

---

## An agent that improves itself — DISTIL

### 127. The embedding layer is the memory, and the grade on it is what matters
*Confirmed on the case it was built for, and only on that case.* DISTIL keeps
one embedding store for everything — queries, answers, goals, reasoning chains,
tools, failures — and its retrieval half is ordinary RAG. The write path is not:
every trace carries a grade, recall ranks by `similarity × credibility ×
recency`, being wrong **demotes rather than deletes**, and a verified outcome
re-weights the path that produced it. One store holds two answers to one
question, one verified and one refuted: a plain vector store returns the refuted
`a.merge(b)` (similarity **0.468**); graded recall returns `{**a, **b}`
(similarity **0.342**, credibility 0.75) — the right answer is less similar and
still wins. The grade forced three details: credibility is a shrunk posterior,
or one lucky `+1` outranks a memory verified twenty times; cosine dedupe would
average a `+1` and a `−1` into an opinionless `0`, so records dedupe on
identity; and an approximate index must over-fetch (10×), or a verified trace
ranked 12th never reaches the re-ranker. The evidence is a constructed fixture,
a lexical embedder and an offline provider; that graded recall improves answers
on a real workload is untested. It keys recall on content and grade, not on the
trajectory 36 proposes.

*Where:* `DISTIL/distil/memory.py`, `DISTIL/DESIGN.md` §1, §3–4,
`DISTIL/README.md`, `DISTIL/tests/test_distil.py` · **confirmed**

### 128. An optimiser that can edit its own grader will edit its own grader
*Confirmed — the first thing self-upgrade did was game its own objective.*
DISTIL's first tuning objective averaged the grades of the top recalled traces
without asking whether they were relevant, and self-upgrade promptly drove the
similarity weight to its floor: a policy that **ignored the query** scored
**0.61 against 0.24**. The fix weights each hit by `grade × similarity`, and
both the hole and the fix are regression tests. Hence the invariant: `grade.py`,
`sandbox.py` and `selfedit.py` are fixed points, the test count may not fall,
and the autonomous loop cannot reach `selfedit` — *it can rewrite its reasoning;
it cannot mark its own homework.* Protecting the grader turned out to be the
easy half, because review found the graders wrong on their own: the determinism
stage compared `''` to `''`, so a tool returning a different answer on every
call graded **+1.000 verified**; the reuse path wrote a verifier-grade `+1` for
a tool it never ran; and `assert_schema` failed *open*. No test suite catches
these, because each reports success. The rule that survives: a grade is the one
input the system cannot re-derive, so nothing may write a grade it did not earn
— no LLM-as-judge, and derived concepts enter ungraded (33).

*Where:* `DISTIL/distil/explore.py` (`score`), `DISTIL/distil/selfedit.py`,
`DISTIL/distil/grade.py`, `DISTIL/DESIGN.md` §5, §10.5, §14, `DISTIL/README.md`
· **confirmed**

### 129. Distil a task until a check can be written; ask when none can
*Refined — the rule holds, but the gate built on it inverted in both directions
before it was fixed.* DISTIL distils a task into subgoals and stops at
verifiability, not at a depth: **a goal is atomic when a check can be written
that decides whether it has been met.** A vague task bottoms out in goals nobody
can check, and that list names the specification gap clause by clause — and says
what to build first, a tool that creates a referee (7). Below the line the
system asks rather than plans, because building for the wrong problem is the
costliest cell of its payoff table (**−0.80**) — 1's understand-before-play
refusal, applied to tasks. The first gate reused a checkability *prior* as a
boundary: its no-signal answer was exactly 0.5 and the test was `< 0.5`, so
"unknown" resolved to "clear" — *handle it somehow please* proceeded (**0.85**)
while *write a parser that produces clean output* was gated (**0.15**). It now
asks two questions against sixteen pinned cases. A fix letting a named referee
stand in for an objective was reverted: a benchmark can say a number moved, not
which number you meant. And the answers to the gate's questions never reached
the solver — invisible to a green suite.

*Where:* `DISTIL/distil/goals.py`, `DISTIL/distil/clarify.py`,
`DISTIL/DESIGN.md` §2, §8, `DISTIL/tests/test_distil.py` (`CLARITY_CASES`) ·
**refined**

### 130. Chain of thought built on game theory — planning is a game against Nature
*Held — the solvers are checked against closed forms; whether the choices beat a
plain planner's is untested.* Free-form chain of thought cannot be checked: it
narrates a decision already made, and when it is wrong it is wrong persuasively.
So every step is a typed object with its inputs attached, and the choice of goal
is a game against Nature: the rows are candidate goals by role — build, clarify,
interrogate, tool, check — and the columns what actually goes wrong: the task as
stated, underspecified, harder, framed wrong, missing a tool. Building pays best
when the task means what it says (**0.90**) and is the costliest mistake when
the frame is wrong (**−0.80**); checking never hurts, so a regret-averse chooser
drifts toward verifying. Minimax regret is the default because regret is what
memory can estimate. Credit over subgoals goes by Shapley value, not to the last
step before it worked — 3's mechanism over subgoals, as CyclicCortex applies it
over regions. The caveat is in the code: the payoffs are a hand-set prior that
memory only scales, by 0.5–1.5 on the upside, where the design says they are
"estimated from memory".

*Where:* `DISTIL/distil/reason.py`, `DISTIL/distil/game.py`, `DISTIL/DESIGN.md`
§6–7 · **held**

### 131. Question everything, never take no for an answer, run the bad ideas — all three are paid in information
*Held — built and run offline, not measured.* Persistence needs a stopping rule
or it is an infinite loop with ambition, and the rule is informational: **keep
re-attacking while the refusals are still new; stop when they repeat.** A second
identical no means the boundary has been found, and the boundary is the
deliverable of a run that never succeeds — 5's "a game's identity is its refusal
boundary" turned into a control loop. Exploration ranks experiments by the
entropy of their outcome, which peaks at `p = 0.5` — 6's derivation applied to
self-improvement. Analogy draws pairs from a similarity *band*, 0.22–0.60: above
it a pair is the same thing said twice, below it nothing carries over, and
nearest-neighbour retrieval is built to return the top of the range. The
autonomous loop is rewarded for information, not success, because rewarding
correctness teaches a system to stop proposing the experiments worth running.
Running it forced two corrections: offline every why-chain came back circular,
so one move scored identically for ever and starved the other five; and an
untried move scored zero, so one lucky first cycle locked the chooser onto one
arm.

*Where:* `DISTIL/distil/challenge.py`, `DISTIL/distil/explore.py`,
`DISTIL/distil/auto.py`, `DISTIL/DESIGN.md` §9.3, §10, §17 · **held**

### 132. Think by clustering the embeddings — a concept is a centroid written back
*Held — the clustering is real and measured on a fixture; what it buys recall is
not.* `think.py` reads the store's *shape*: it clusters what the system knows
and writes each cluster back as a new memory. A centroid sits near every member
and is identical to none of them, which is what a concept is; the store grows
denser rather than only longer, and it only ever adds. Two bugs surfaced only as
the clustering quietly returning too little: average linkage needs the *full*
pairwise matrix — with the sub-threshold pairs dropped, three obvious groups
came back as one — and the cut cannot be a constant, so it is `mean + 0.75σ` of
the store's own pairs. It prefers precision: on twelve traces in three known
groups it finds **two, both pure**; the third sits where within-group similarity
(**0.129**) overlaps across-group (max **0.158**). A wrong concept becomes a
fact the system believes about itself, so a missing one is the cheaper error.

*Where:* `DISTIL/distil/think.py`, `DISTIL/DESIGN.md` §20, `DISTIL/README.md`
("Thinking by clustering") · **held**

---

## Not yet built

### 133. Multiple gradients as stacked planes, with the layer chosen by depth perception
*Held — a design; nothing has been built or measured.* The one variation here
that keeps ordinary layers and back-propagation and puts its divergence in the
descent. A gradient is pictured as an XY plane with the weights as depth; give
every layer its own and stack them, and descent answers two questions together:
*which way is downhill from here*, and *which layer is the right one to be
descending in*. Every layer connects directly to every other, so separation is a
learned weight rather than a distance, and the cross-layer step combines two
mechanisms: **depth perception chooses the layer, and the vertical weights carry
the step there**. The perception is taken literally from the eye — comparative,
never absolute — which gives the one invariant already testable: adding a
constant to every depth in a column must not change which layer is accessed. For
images the cues are occlusion, where the nearest plane wins, and motion
parallax, whose motion comes from the descent itself — the model moves its own
eye by learning — with a light radial blur standing in for the fovea. The design
names its own risks: the vertical weights are learned by the descent that
travels them and can reinforce one path until nothing else is reachable;
convergence silences parallax; and the vertical parameters grow as `L(L−1)/2`,
2,016 sets at `L = 64`.

*Where:* `MultiGradientNN/DESIGN.md`, `MultiGradientNN/README.md` — on branch
`feat/multi-gradient-nn` · **held**

---

## Method — how the evidence was kept honest

### 134. Parity is equality, not tolerance — and three implementations are a measuring instrument
*Confirmed — exact equality was affordable, and most of what it caught was a
unit.* When the count model was ported to Go, "close enough" was rejected: a
tolerance compounds over epochs and would have hidden real divergence in the
structure. Exact interchange cost a Mersenne Twister reproducing CPython's,
Shewchuk's exact summation, and insertion-ordered adjacency, because float
addition is not associative; the parity suites then assert **equality** — graph,
counts, RNG state, predictions, transcripts, even the metacognition records —
with each side continuing the other's file. Built to keep the ports honest, the
suites mostly caught the reference. Once the encoding became a dial and a unit
stopped meaning a character, one bug surfaced again and again: Python's
`score()` counted a six-word text as **22**, not 6; Go and Python disagreed on
`trained_chars` **1,800 to 365**; Rust matched a prefix by characters, so a word
model continuing `"the"` found `"there is a"`; and Python placed a correction's
end in characters, so **a word model's sentence that stopped too early was never
blamed** — a gap a Rust parity test had documented before anyone fixed it. Two
rules came out of it: *a number whose unit depends on the model is a number that
will be read wrong*, so every report names its unit; and a port is compared by
its surfaces, not its README — walking 78 routes and 32 commands found four Rust
server bugs, each answering 200 with a plausible document. A benchmark inherits
the rule and refuses to time two programs until they agree, because a speed
comparison between programs that computed different things is not a comparison
(91).

*Where:* `RadixCyclicNN/DECISIONS.md` D-039, D-073, D-074, D-086,
`RadixCyclicNN/bench/README.md`, `RadixCyclicNN/SPEC-WordNGrams.md` §9,
`ModelKit/tests/test_go_parity.py`, `ModelKit/tests/test_rust_parity.py`, PR #20
· **confirmed**

### 135. A number the committed script does not print is withdrawn — and a floored mean hid a 600× loss
*Confirmed by the rewrite that withdrew it.* Re-running every NeuralCompression
claim against the committed scripts found four the scripts did not support. The
largest was 21's plateau trigger with long bursts: the committed defaults fire
**23** times where the claim reported **15**, so its **1.24×** came from a
configuration that was never in the code, and a burst sweep gives 1.00×, 0.98×,
0.88×, then divergence. The second was hidden by the report format: the script
printed fires as a per-seed mean floored to an integer, so a weight trigger that
fired **once** — on one seed of three, taking it from **0.00016** to **0.0961**,
a 600× loss — printed as **0** fires beside a mean of 0.0322: one disaster and
two non-events reported as a mild effect. The same pass replaced "diverged" with
the measured 0.9255 (16), found "beats separate networks" to be a tanh result
(15), and found per-seed damage in the layout test ranging 0.60×–3.74× where one
seed had said 1.1× (17). *Report the worst seed, report totals rather than
floored means, and quote only numbers the committed code prints.*

*Where:* `Experiments/NeuralCompression/FINDINGS.md`, `vanishing.py`,
`partition.py`, `consolidate.py` as rewritten on branch
`claude/keen-feynman-ntoqni` (the copies on `main` are the pre-rewrite versions)
· **confirmed**

### 136. Measure each game in its own units, at matched compute, over seeds — five traps that would have reversed a result
*Confirmed — each trap was caught by a control, and each alone would have
flipped a conclusion.* The split-and-selector experiments came close to
measuring arithmetic instead of learning. **A band of width `1/N` makes raw MSE
`N²` smaller**, so reported naively the partition looks brilliant; every number
is rescaled to the game's own `[0, 1]`. **The same band makes the gradient `N`
times smaller**, so the partition trains `N` times slower and reports a false
negative — the first run did exactly this, and the ordered/shuffled gap at N=16
appeared only once loss was measured in game units. **"Reserved slots cost ~14%"
was single-seed noise** — 0.0397 against 0.0401 over four seeds — and the first
dust measurement built its bands with a Voronoi construction that re-tiled `[0,
1]`, removing the dust it was measuring. **"Train 4, then retrain all 8"
measured nothing**, because the old games improved from the extra training;
phase 2 must train only the new games. **Unequal compute:** replay's 1.57× fell
to 1.05× at the same budget, and a probe-then-step pattern gave the inversion
arms two updates per sample where the baseline got one. One diagnostic is worth
keeping: a learning-rate sweep flat from 0.02 to 1.0 (0.0179 → 0.0179) means
stuck, not undertrained — which is how 24's `b = 1/3` was found.

*Where:* `Experiments/NeuralCompression/FINDINGS.md` ("Two traps", §5, §8, §13),
`Experiments/NeuralCompression/README.md` · **confirmed**

### 137. Every arm peaked early — keep the best network, and grade it on positions it was not chosen with
*Confirmed, and it changed what every earlier number meant.* Five times the
rounds settled whether more training helps, and it does not: the positive-only
chess arm reached **6.3** refusals a move at round 9, **36.8** at round 40 and
**151.2** at round 200, while the rating spread across the five arms fell from
79 Elo to 30. Training loss kept falling while held-out refusals drifted up —
overfitting to the sliding buffer of the network's own failures, which is
exactly the data an on-policy failure-learner trains on. So every number
reported until then was wherever the run happened to stop. A run now keeps the
network that scores best on 120 validation positions and reports it on the 280
it was not chosen with: on a 30-round run that gives **52.6** refusals against
the last round's **77.1**, and the first centipawn loss clearly under a random
legal mover's. With selection in place, the 400-round run added nothing after
round 44.

*Where:* `TwoNRL_Chess/results_long/README.md`,
`TwoNRL_Chess/results_r400/README.md`, `TwoNRL_Chess/experiment.py`, commit
626688d · **confirmed**

### 138. An arm's name is not its configuration
*Confirmed three times in one experiment — and once the identical rows were a
theorem rather than a bug.* The chess ablations reused the arm name `2nrl`, so
keying the summary on the name silently averaged four configurations into one
row; separated, they read **74.4**, **77.7** and **131.8** refusals a move. Then
the flip became event-triggered and a flag went inert: both phase-depth arms
silently re-ran the headline, and the tables listed three identical rows under
different names. Later the default operator changed while the labeller still
took the old default, so runs of one operator would have been labelled as
another. The commit that caught the silent ablation also found the opposite
case: the read-out-flip arm reproduced the headline to every digit because the
two operators differ by a per-unit sign symmetry and Adam is sign-equivariant —
"Saying the fix would change the results was wrong of me." The game encoder's
quick run, meanwhile, was writing into its results directory: two minutes of
checking would have silently replaced forty minutes of measurement. *Identical
numbers under different names are the first thing to explain: either one arm has
run twice, or there is a symmetry nobody noticed* (39). Every table is now keyed
on what its run changed and generated from the run's JSON — which protects the
tables, not the prose: the chess README's prose says the flip buys ×850, its
generated table ×1355.4.

*Where:* `TwoNRL_Chess/summarize.py` (`label_of`),
`TwoNRL_Chess/results/invert_readout.json`,
`Experiments/UniversalGameEncoding/Makefile`, commits 1279c7c and 1666a45 ·
**confirmed**

### 139. Ask a function only the question it answers — two bugs that produced plausible numbers
*Recorded because one of them produced a conclusion that was then published.* In
GTMNN an unfilled repertoire slot and the ABSTAIN action were both written
`None`, so the payoff scored a dead slot as a free abstention and best response
took it — **20 of 24** seats "abstaining", half of them onto slots that did not
exist. The fix landed in the same commit that published "more training does not
help" (3.226 → 3.204 after 6,900 steps per micro), which had been measured on
the broken code; re-measured, loss falls monotonically through 60 epochs (2). In
GREN's vocabulary derivation legality was read off `why()` — but `why()` is a
diagnosis that *assumes* a refusal and falls through to a default code, so chess
got a zero-size legal class and every feature a separation of exactly **0.000**.
*A value that means two things will be read as the cheaper one, and a function
that answers "why was this refused?" cannot answer "was it refused?".* When a
fix lands, re-run every conclusion measured before it; the fix does not retract
them by itself.

*Where:* `GTMNN/gtmnn/micro.py`, `GTMNN/gtmnn/payoff.py`,
`GREN/gren/vocabulary.py` — on branch `feat/gren-impl`, commits a1052a9 and
890b858 · **confirmed**

### 140. Judge a decode by what the picture stored — the rulers that lied
*Confirmed — three things in the audio codec were wrong until they were
measured, and the metric everyone reaches for first was the wrong one.*
**Waveform SNR measures the phase you threw away:** a melody decoded from the
grey picture is 2.51% off in spectrum and **−2.8 dB** in SNR — further from the
original than silence, with the spectrum 97% right — so spectral convergence is
the number that matters, and a log-spectral distance must be floored 80 dB under
the peak or the silent bins dominate it. **Level by energy, not by the tallest
sample:** a rebuilt phase keeps a clip's energy but not its peak, so
peak-matching scored a correct decode as **0.29** wrong when it was **0.038**
wrong. **Damp the acceleration:** fast Griffin-Lim applied its momentum raw, and
at 0.99 the error over 32 passes rose to **0.336**, against 0.156 with no
momentum at all; damped, it fell to **0.073**. A fourth — a spectrum read from a
picture has DC and Nyquist bins no real signal has — made two backends disagree
by about 0.25 on every decode. The learned vocoder later met the same kind of
ruler: its unit round trip favoured the centroid vocoder by construction
(116).

*Where:* `AudioImage/DESIGN.md` §2.4, §6.2, §6.4, §7, `AudioImage/README.md`
("What survives the round trip") · **confirmed**

### 141. Anything a loop writes is input to that loop
*Confirmed three times in one project and once in another.* DISTIL's autonomous
loop read itself twice before the rule was written down: its questioning move
writes a trace beginning "why <subject>", which was then eligible as the next
subject, so it asked why about why about why; and solving a task it had set
itself wrote goals and failures *containing* that task, unmarked, because the
solver wrote them. The third time shipped: cluster concepts excluded from the
source pool still occupied nearest-neighbour slots, so each pass re-derived the
same concept and wrote a near-duplicate — the exclusion has to happen at search
time. FilterBankRadix met it at the level of the experiment: its corpus was cut
from this repository's files, one of them this index, so **recording a result
changed the corpus a running sweep was measuring on**, and two tables in one
README stopped being comparable. Corpora are now pinned by name and frozen with
a SHA-256 digest in every result file — which is why this file is deliberately
left out of every corpus built from the repository.

*Where:* `DISTIL/distil/auto.py`, `DISTIL/distil/think.py`, `DISTIL/DESIGN.md`
§17.3, §20.5, `FilterBankRadix/fbradix/corpus.py`,
`RadixAcyclicNN/radixtree/corpus.py` · **confirmed**

### 142. Review a frozen tree, and re-review the fixed one
*Confirmed, and the numbers are the argument.* DISTIL was reviewed six times by
parallel agents, every finding checked by an independent skeptic, and **115
defects** were confirmed and fixed. Round one found **45**, and every later
round found defects the previous round's fixes had *introduced* — **6 of 15, 4
of 13, 7 of 13, 1 of 2, 5 of 27** — and one fix was simply wrong, reverted by
its own author two commits later. Rounds one to five ran against a tree being
edited underneath them: in round four **16 of 16** refutations read "the code no
longer exists in the working tree". Round six froze the tree at a commit, and
**27 of 28** findings survived verification. The worst defects were the kind a
passing suite cannot show — graders that reported success for checks they never
ran (128), and answers that never reached the solver (129).
*Review a frozen commit, re-review after every fix, and read what a check
reports, not merely that it passed.* (The commit subjects sum to 122 defects,
not the README's 115; the gap is one commit's seven.)

*Where:* `DISTIL/README.md` ("What four rounds of adversarial review found") and
the review commits `9b9459e` … `4bfd772` · **confirmed**

---

## What to test next

Grouped by where they would land, and within each group ordered by how much they
would change, per unit of effort.

**The game stack**

0. ~~**Does transfer survive ACROSS regions, or only within them?**~~ (29) **Answered: the graph is doing routing.** Cross-region transfer is +0.025 in one case of four; the within-region shared vocabulary is +0.023 against its proper baseline. Regions earn their place by *placing* games correctly — the wrong region is measurably worse, per the negative Shapley values — not by teaching each other (47).
1. ~~**Does validity transfer?**~~ (19) **Answered, and it is the same +0.023.** `p_valid` trained on chess alone leaves checkers at its majority class. The interesting follow-up is 29's refinement: build a region whose vocabulary is *mostly* shared, as `common_denominator.py` did with three inputs, and see whether 0.646 survives inside a cortex.
2. **Can the minimality criterion split a mechanic?** (38, 40, and §8.3) Now with an acceptance test. `OCCUPIED_TARGET` is "own piece there" in chess and "any piece there" in checkers, go and sudoku; `BLOCKED_PATH`'s occupied-midpoint feature is a block in chess and a capture in checkers. §8.3's rule — keep splitting a part while its halves are seen independently — should split both. **Pass condition: after the split, the feature-based similarity ranking puts go/sudoku first, where rule transfer already puts it.** If it still ranks chess/checkers first, name-level decomposition cannot be made to predict transfer and grouping must use the operational measure.
3. **Does goal-first ordering still pay at scale?** (37) The compression half of the result is small-N: four games and one perfectly-discriminating axis. Generate or gather thirty-odd games with overlapping goals and re-measure tokens-to-identification against alphabetical and against a highest-entropy-axis-first ordering. The knowability argument does not need this; the compression claim does.
4. **Is a trajectory worth more than the input that produced it?** (36) The premise of the whole memory design, and an afternoon to settle. Take CyclicCortex, which already has four games and trained regions: record input features, trajectory (region id, vocabulary slots written, hidden-activation pattern) and outcome per position, then build two lookup tables over the same episodes — one keyed by input, one by trajectory — and see which predicts the held-out outcome better. If trajectory-keyed does not win, the premise is wrong and nothing else in `Memory/` needs building.
5. **Can a mechanic be split when two games disagree about it?** (30) Chess and checkers both refuse `OCCUPIED_TARGET` under opposite conditions. If GREN probed *conditionally* — does this game still refuse when the occupant is an enemy? — the code would split into `OCCUPIED_TARGET_OWN` and `OCCUPIED_TARGET_ANY`, and the two games would stop sharing a slot they should never have shared. This is the cheapest test of whether the refusal channel can be widened enough to carry feature alignment, and it is a direct consequence of the only sharp negative result so far.
6. ~~**Why does the credit path learn so weakly?**~~ (2, 3) **Answered: it doesn't — the learning rate was 7× too small for φ's scale, and the control had 30–100 epochs to its 4–6.** Fixed; it now beats uniform at every budget. The successor question is sharper: **how do you teach a micro to abstain without it going silent for good?** Counterfactual-regret credit teaches silence and collapses the inference game to uniform through the abstain trap. Candidates worth one experiment each: cap the abstainer share of the gradient budget at the holder share; let the reserve price fall with the bidder count so a quiet population still seats *someone*; or seat by reputation rather than bid when bidders < M. The last is the least invasive and the auction's own `bidders` signal already flags the condition.
7. **Fix `b` everywhere and re-measure.** (24, 20) Upstream of everything in the vanishing-gradient work. `RadixCyclicNN` carries the same default — measure its realised `E|z|` before changing it — and so do FilterBankRadix's layer-2 trees, RadixAcyclicNN's learned rule (whose window sweep ran at `b = 1/3`) and PrimedRadixPair's planned sine kind. GTMNN's implementation and the CyclicCortex and GREN networks already use `b = 1`.
8. ~~**Does the population actually specialise?**~~ (2) **Answered: yes, when it gets to play.** `gini` reaches 0.54 with every micro seated and 0.05 with 24 of 256 — so the split reward does bite, and seat count is what gates it. That is why 6 above is the sharper question.
9. **NCD against mechanic Jaccard.** (11) Where a feature-free similarity disagrees with the mechanic one, the vocabulary is blind — the only automatic check on the part of GREN hardest to verify.
10. **Get a blind explorer to the midgame of a sparse game.** (41) Checkers is unmapped at the input level under every blind policy: at most three acceptances per run, no jump ever performed. Two candidates, one experiment each: a two-phase policy that exploits p(accept) until a run of accepted moves is banked and then switches to p = ½; and edits of remembered acceptances at distance two, since a checkers move's one-step neighbours are almost all illegal. Pass condition: a jump performed and FORCED_ALTERNATIVE found blind in most seeds. Then promote `blind.py` to `gren.cli blind` — committed, with its output, so 41's numbers can be reproduced at last — so a new game is mapped from its grammar alone, and give the rule-namer a feature it can blame for NOT_YOURS, the one chess rule no present feature carries.
11. **`universal_fraction`.** (18) The share of micros that are game-blind and transfer everywhere. Nothing predicts it; whether evolution finds a stable value is the sharpest test of whether GREN and GTMNN compose.

**The papers and 2NRL**

12. **Run 2NRL's own text experiment.** (60, 62) Chess and the game encoder measured 2NRL against matched controls and it lost; CartPole, where the inverse of the failure is the answer, is its only win. `Research/2NRL.md` §11.1 designs the decisive test on text — negative sets of matched size differing only in structure, against positive-only training and against the same negatives without inversion, scored by held-out per-character log-probability — with §11.2's sweep of the rate ratio. Nothing else would say whether 2NRL has a home outside CartPole; until it runs, `TwoNRL_Chess` should at least stop shipping the complement as its default inversion (64).
13. **Does the monitor do anything the clock does not?** (53, 99) The cycles paper's Prediction 2: switch the monitors off (`avoid_repeats=False`, `avoid_word_repeats=False`, `explore=0`) and self-conversation should collapse to a fixed point within single-digit turns. It is the result that would cost the paper its thesis, and it is a flag away. Prediction 3 — odd cycles measurably weaken targeted inversion (54) — is the next one.
14. **Replicate the CartPole run the sine was adopted on.** (57) The sine paper's §9 protocol: a linear-head DQN pair, 20 seeds, 100 greedy episodes, a tanh arm and a `b` sweep. If tanh matches the sine there, the headline was about the saturating output head, not about periodicity.

**The radix models and sound**

15. **Does a model learn language better from sounds than from letters?** (114, 117, 77) The question the phonetic work exists for, and no run has asked it. Train one corpus under `char:3:1`, `word:3:1`, `token:3:1`, `phone:3:1` and `syllable:2:1`, and compare held-out loss **per character** — bits per unit do not compare — together with node counts and what each continues a prefix with. The token half is the branch's Q-21; the sound half is stated nowhere.
16. **Does priming beat growing on little data?** (106) The primed pair's reason to exist and its own open question: bits per unit after 1, 10, 100 and 1,000 texts, against RadixCyclicNN's count model on the same texts — the comparison its requirements asked for and `bench compare` never ran.
17. **Grade what the judgement machinery chooses.** (86, 87, 94) The least-punished traversal disagrees with the ordinary search on 19.9% of continuations and the attention band redistributes every partial correction, and neither has been graded. The tutor is already wired: score both searches' continuations on the punished benchmark corpus, and a corrected model with and without the band.
18. **Does the dynamic window's cycle buy anything?** (84) A model cycled 32 → 16 → 8 → 4 → 32 over the same corpus as one left alone, compared on loss, on the branch points its halves grew and on what it predicts. Cheap to run: `radixnet window --on`, then the same `train` — and since the split/merge fix (85) it can run on a tutored model. It also decides whether the top should merge the halves back at all.
19. **Give the self-upgrade loop a yardstick.** (98) A held-out set and a fixed metric, so a widening real/fake gap can be told apart from a generator getting worse in a way its own discriminator likes (Q-4). DISTIL showed what an unwatched objective does first (128).
20. **Put the decaying memory on the cyclic graph.** (103, 100) RadixDecayNN's design names it as the next model: `seen` fading on the graph's own clock, on the structure where a hand-over otherwise seals itself for good.
21. **Teach the acoustic units real speech.** (115, 116) The codebook and the learned vocoder have only ever heard the synthesizer's own voice. Learn both from real recordings, then re-measure the round trip and the log-mel distance — and only then ask what a model learns from them.
