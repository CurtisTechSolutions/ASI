# RadixCyclicNN — run commands.  `make help` lists every target.
#
# Local targets call the zero-dependency Python package directly (python3 only).
# Docker targets drive the Compose stack in docker-compose.yml.
# Override any variable on the command line, e.g. `make train EPOCHS=20 LR=0.8`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY          ?= python3
RADIXNET    ?= $(PY) -m radixnet
MODEL       ?= model.json
DATA        ?= data/sample_corpus.txt
GARBAGE     ?= data/sample_garbage.txt
CKPT_DIR    ?= checkpoints
UPLOAD_DIR  ?= uploads
BACKEND     ?= auto
DEVICE      ?=
SEED        ?= 0
EPOCHS      ?= 10
LR          ?= 0.5
BATCH       ?= 4
NEG_LR      ?= $(LR)
POS_LR      ?= 0.1
PREFIX      ?= the quick brown
LENGTH      ?= 20
MODE        ?= dijkstra
TRAVERSAL   ?= reward
COUNT       ?= 5
MAX_LENGTH  ?= 60
TEXT        ?= the cat sat on the mat
SAY         ?= tell me about the cat
FORMAT      ?= openai
GENERATIONS ?= 3
SAMPLES     ?= 8
NEGATIVE    ?= $(basename $(MODEL)).negative$(suffix $(MODEL))
REASON      ?= gibberish
SEVERITY    ?= 1
DISC        ?= $(dir $(MODEL))discriminator.json
NAME        ?= latest
HOST        ?= 127.0.0.1
PORT        ?= 8000
CHARS       ?= 50000
OLLAMA_URL  ?=
OLLAMA_MODEL ?=
PROMPT      ?= short true sentences about animals and the weather
LINES       ?= 20
STYLE       ?= good
THRESHOLD   ?= 6
VARIANTS    ?= 3
TOPIC       ?= everyday life
FOCUS       ?= subject-verb agreement
ROUNDS      ?= 3
EXERCISES   ?= 5
DRILLS      ?= 0
PLAN        ?= 3
BATCHES     ?= 3
BRIEF       ?=
PROBLEMS    ?= data/sample_problems.jsonl
AUDIO       ?= utterance.wav
TRANSCRIPT  ?=
SECONDS     ?= 5
SPEECH_RATE ?= 8000
PHASE       ?= both
# Who teaches: the English lessons and the code-generation tutor both follow it
TUTOR       ?= ollama
TUTOR_MODEL ?=
TUTOR_URL   ?= $(if $(filter chatgpt,$(TUTOR)),$(CHATGPT_URL),$(OLLAMA_URL))
CHATGPT_MODEL ?=
CHATGPT_URL ?=
# blank for the chatgpt tutor: it then uses CHATGPT_MODEL, $$RADIXNET_OPENAI_MODEL or gpt-4o-mini
CODEGEN_MODEL ?= $(if $(filter chatgpt,$(TUTOR)),$(CHATGPT_MODEL),gemma4)
CODEGEN_URL ?= $(if $(filter chatgpt,$(TUTOR)),$(CHATGPT_URL),$(OLLAMA_URL))
CODEGEN_MODEL ?= gemma4
TASKS       ?= data/sample_tasks.txt
BLAME       ?=
BROWSER     ?=
STEPS       ?= 10
AGENT_MODEL ?=
MAX_STEPS   ?= 6
COMPOSE     ?= docker compose
GPU_COMPOSE ?= $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml
ALL_PROFILES := --profile auto --profile dev --profile evolve --profile tools

# Compose reads these from the environment (see .env.example for the full list).
RADIXNET_PORT    ?= $(PORT)
RADIXNET_BACKEND ?= $(BACKEND)
RADIXNET_CORPUS  ?= $(notdir $(DATA))
WITH_TORCH       ?= 0
RADIXNET_RESUME  ?= 0
export EPOCHS LR BATCH RADIXNET_PORT RADIXNET_BACKEND RADIXNET_CORPUS WITH_TORCH RADIXNET_RESUME

DEVICE_FLAG := $(if $(DEVICE),--device $(DEVICE),)
GLOBAL      := --model $(MODEL) --backend $(BACKEND) $(DEVICE_FLAG) --seed $(SEED)

.PHONY: help install install-gpu check test go-build go-test go-parity go-serve go-negative \
        phone-train phone-predict phone-generate phone-words phone-test phone-demo speak speak-play acoustic-train acoustic-speak phone-go phone-rust phone-parity \
        rust-build rust-test rust-parity rust-talk go-talk bench-compare train resume predict generate score talk talk-repl 2nrl invert compress \
        negative-blame negative-clear negative-why negative-filter negative-auto negative-reasons negative-forget tutor-blame \
        evolve evolve-forever evolve-blame info checkpoints restore bench demo serve frontend-install frontend-build \
        frontend-dev frontend-test ollama-models ollama-corpus ollama-garbage ollama-review ollama-blame ollama-correct ollama-2nrl \
        chatgpt-models chatgpt-ask codegen codegen-teacher codegen-model \
        speech-info speech-teach speech-listen speech-tutor speech-decode \
        tools agent explore explore-forever mcp \
        docker-build up up-auto up-dev up-gpu up-ollama down logs shell docker-train docker-evolve \
        docker-test docker-bench docker-reload docker-export docker-clean clean clean-all

help: ## Show this help
	@echo "RadixCyclicNN make targets  (override variables like: make train EPOCHS=20 LR=0.8)"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: MODEL=$(MODEL) DATA=$(DATA) GARBAGE=$(GARBAGE) CKPT_DIR=$(CKPT_DIR) BACKEND=$(BACKEND)"
	@echo "          EPOCHS=$(EPOCHS) LR=$(LR) BATCH=$(BATCH) PORT=$(PORT)"

##@ Setup
install: ## Install the package in editable mode (zero runtime dependencies)
	$(PY) -m pip install -e .

install-gpu: ## Install with the optional torch backend (GPU / vectorised CPU)
	$(PY) -m pip install -e ".[gpu]"

check: ## Byte-compile the package and show which backends are available
	$(PY) -m compileall -q radixnet
	$(PY) -c "from radixnet import describe_backends; print(describe_backends())"

test: ## Run the unit test suite
	$(PY) -m unittest discover -s tests -v

GO ?= go
go-build: ## Build the Go count / reward model CLI into go/bin/radixnet-count
	cd go && $(GO) build -o bin/radixnet-count ./cmd/radixnet-count

go-test: ## Run the Go tests with the race detector
	cd go && $(GO) vet ./... && $(GO) test -race ./...

go-talk: go-build ## Talk to the Go model in today's format: SAY is the message (FORMAT=openai|anthropic)
	go/bin/radixnet-count --model $(MODEL_COUNT) talk --message "$(SAY)" --format $(FORMAT)

go-negative: go-build ## The Go negative network: blame GARBAGE, then judge TEXT through it
	go/bin/radixnet-count --model $(MODEL) --negative $(NEGATIVE) negative blame --data $(GARBAGE) --reason $(REASON) --severity $(SEVERITY) --source make
	go/bin/radixnet-count --model $(MODEL) --negative $(NEGATIVE) negative why --text "$(TEXT)"

go-parity: ## Cross-language parity of the Go port against the Python model
	$(PY) -m unittest tests.test_go_parity -v

MODEL_COUNT ?= model.count.json
go-serve: go-build ## Serve the frontend from the Go count / reward model on http://HOST:PORT (uploads / checkpoints as for `serve`)
	go/bin/radixnet-count --model $(MODEL_COUNT) --seed $(SEED) serve --host $(HOST) --port $(PORT) --frontend-dir frontend/dist --upload-dir $(UPLOAD_DIR) --checkpoint-dir $(CKPT_DIR)

go-mcp: go-build ## The Go MCP server on stdin / stdout: the tools and the network, for any MCP client
	go/bin/radixnet-count --model $(MODEL_COUNT) --seed $(SEED) mcp --upload-dir $(UPLOAD_DIR)

##@ Rust port and the cross-language benchmark
CARGO ?= cargo
LIMIT         ?= 10
TRAVERSAL     ?= reward
MERIT_SCALE   ?= 1
PENALTY_SCALE ?= 1
BENCH_CHARS ?= 2000000
BENCH_PREDICTIONS ?= 3000
BENCH_EPOCHS ?= 3
BENCH_REPEAT ?= 3
PUNISH_EVERY ?= 7

RUST        ?= rust/target/release/radixnet
MODEL_RUST  ?= $(MODEL_COUNT)
RUST_GLOBAL := --model $(MODEL_RUST) --seed $(SEED)
RUST_ENC    ?= char:3:1

rust-build: ## Build the Rust port's CLI and benchmark into rust/target/release (needs Rust 1.82+)
	cd rust && $(CARGO) build --release

rust-test: ## Run the Rust tests, clippy and the formatter check
	cd rust && $(CARGO) test && $(CARGO) clippy --all-targets -- -D warnings && $(CARGO) fmt --check

rust-parity: ## Cross-language parity of the Rust port against the Python model (every tests/test_rust_parity*.py)
	$(PY) -m unittest discover -s tests -p 'test_rust_parity*.py' -t . -v

rust-train: rust-build ## Train the Rust model on DATA (RUST_ENC=char:3:1|word:3:1|..., MODEL_RUST is its file)
	$(RUST) $(RUST_GLOBAL) --encoding $(RUST_ENC) train --data $(DATA) --epochs $(EPOCHS)

rust-predict: rust-build ## Continue PREFIX: the K likeliest and K least likely (TRAVERSAL picks what it follows)
	$(RUST) $(RUST_GLOBAL) --encoding $(RUST_ENC) predict --prefix "$(PREFIX)" --length $(LENGTH) --k $(COUNT) \
	    --traversal $(TRAVERSAL) --merit-scale $(MERIT_SCALE) --penalty-scale $(PENALTY_SCALE)

rust-generate: rust-build ## Generate COUNT whole texts of at most MAX_LENGTH symbols each
	$(RUST) $(RUST_GLOBAL) --encoding $(RUST_ENC) generate --count $(COUNT) --max-length $(MAX_LENGTH) --mode beam \
	    --traversal $(TRAVERSAL)

rust-talk: rust-build ## Talk to the Rust model in today's format: SAY is the message (FORMAT=openai|anthropic)
	rust/target/release/radixnet --model $(MODEL_RUST) talk --message "$(SAY)" --format $(FORMAT)

rust-score: rust-build ## Log-probability of TEXT under the Rust model
	$(RUST) $(RUST_GLOBAL) --encoding $(RUST_ENC) score --text "$(TEXT)"

rust-feedback: rust-build ## Thumbs up DATA and thumbs down GARBAGE (2NRL when both are given)
	$(RUST) $(RUST_GLOBAL) --encoding $(RUST_ENC) feedback --good $(DATA) --bad $(GARBAGE) --strength $(SEVERITY)

rust-paths: rust-build ## What the judged walks did, step by step
	$(RUST) $(RUST_GLOBAL) paths --limit $(LIMIT)

rust-nodes: rust-build ## Each node against the nodes around it
	$(RUST) $(RUST_GLOBAL) nodes --limit $(LIMIT)

rust-words: rust-build ## A word encoding's alphabet, most read first (needs RUST_ENC=word:3:1)
	$(RUST) $(RUST_GLOBAL) words --limit $(WORD_LIMIT)

rust-weights: rust-build ## Show the Rust model's weight function
	$(RUST) $(RUST_GLOBAL) weights

rust-info: rust-build ## The Rust model's statistics and history tail
	$(RUST) $(RUST_GLOBAL) info

rust-check: rust-build ## The structural invariants of the Rust model, over DATA
	$(RUST) $(RUST_GLOBAL) check --data $(DATA)

rust-compress: rust-build ## Merge every unary chain by hand, then save
	$(RUST) $(RUST_GLOBAL) compress

rust-invert: rust-build ## Flip the sign of every reward
	$(RUST) $(RUST_GLOBAL) invert

rust-serve: rust-build ## Serve the API + the prebuilt frontend from the Rust port on http://HOST:PORT
	$(RUST) $(RUST_GLOBAL) --encoding $(RUST_ENC) serve --host $(HOST) --port $(PORT) \
	    --frontend-dir frontend/dist --upload-dir $(UPLOAD_DIR) --checkpoint-dir $(CKPT_DIR)

rust-bench: rust-build ## The Rust port's own benchmark (BENCH_CHARS characters, BENCH_EPOCHS epochs)
	rust/target/release/radixnet-bench --chars $(BENCH_CHARS) --epochs $(BENCH_EPOCHS) --traversal $(TRAVERSAL)

rust-demo: ## Train on the sample corpus with the Rust port, then predict, generate and read it back
	$(MAKE) rust-train
	$(MAKE) rust-predict
	$(MAKE) rust-generate
	$(MAKE) rust-info

bench-compare: ## Both ports over one corpus, checked against each other -> bench/RESULTS.md
	$(PY) bench/compare.py --chars $(BENCH_CHARS) --predictions $(BENCH_PREDICTIONS) \
	    --epochs $(BENCH_EPOCHS) --repeat $(BENCH_REPEAT) --punish-every $(PUNISH_EVERY) --seed $(SEED)

##@ Train / predict (local)
train: ## Train on DATA (incremental: MODEL is loaded first if it exists), checkpointing into CKPT_DIR
	$(RADIXNET) $(GLOBAL) train --data $(DATA) --epochs $(EPOCHS) --lr $(LR) --batch-size $(BATCH) --checkpoint-dir $(CKPT_DIR)

resume: ## Continue training from the latest checkpoint in CKPT_DIR
	$(RADIXNET) $(GLOBAL) train --data $(DATA) --epochs $(EPOCHS) --lr $(LR) --batch-size $(BATCH) --checkpoint-dir $(CKPT_DIR) --resume

predict: ## Continue PREFIX by shortest path (MODE=dijkstra|sample, TRAVERSAL=reward|punishment) for LENGTH characters
	$(RADIXNET) $(GLOBAL) predict --prefix "$(PREFIX)" --length $(LENGTH) --mode $(MODE) --traversal $(TRAVERSAL)

generate: ## Generate COUNT texts from scratch (up to MAX_LENGTH characters each; TRAVERSAL=reward|punishment)
	$(RADIXNET) $(GLOBAL) generate --count $(COUNT) --max-length $(MAX_LENGTH) --traversal $(TRAVERSAL)

score: ## Log-probability of TEXT under the model
	$(RADIXNET) $(GLOBAL) score --text "$(TEXT)"

talk: ## Talk to the model in today's format: SAY is the message, the thinking comes first, then the reply (FORMAT=openai|anthropic)
	$(RADIXNET) $(GLOBAL) talk --message "$(SAY)" --format $(FORMAT)

talk-repl: ## Talk to the model line by line at a prompt (Ctrl-D ends it); --save writes what it learned back
	$(RADIXNET) $(GLOBAL) talk --format $(FORMAT)

2nrl: ## 2NRL: train on GARBAGE, invert the network, fine-tune on DATA
	$(RADIXNET) $(GLOBAL) 2nrl --bad $(GARBAGE) --good $(DATA) --neg-lr $(NEG_LR) --pos-lr $(POS_LR) --batch-size $(BATCH)

invert: ## Invert the network (flip every edge weight and activation amplitude)
	$(RADIXNET) $(GLOBAL) invert

compress: ## Merge unary chains (radix-tree path compression) and save
	$(RADIXNET) $(GLOBAL) compress

WINDOW_TOP   ?= 32
WINDOW_FLOOR ?= 4
STEPS        ?= 1

window: ## Show the dynamic window: the ladder of node sizes (32 -> 16 -> 8 -> 4 -> 32) and where it stands
	$(RADIXNET) $(GLOBAL) window

window-on: ## Switch the dynamic window on (WINDOW_TOP=32 WINDOW_FLOOR=4; it steps at the end of every training epoch) and save
	$(RADIXNET) $(GLOBAL) window --on --top $(WINDOW_TOP) --floor $(WINDOW_FLOOR)

window-step: ## Step the dynamic window by hand STEPS times: merge what fits, halve what is longer, move the window; then save
	$(RADIXNET) $(GLOBAL) window --step $(STEPS)

window-off: ## Switch the dynamic window off (compression is unbounded again) and save
	$(RADIXNET) $(GLOBAL) window --off

##@ Word n-grams (the encoding dial's word unit: the same model over words)
MODEL_WORD  ?= model.word.json
WORD_ENC    ?= word:3:1
WORD        := --model $(MODEL_WORD) --kind count --encoding $(WORD_ENC) --backend $(BACKEND) $(DEVICE_FLAG) --seed $(SEED)
WORD_PREFIX ?= the cat sat on
WORD_LENGTH ?= 6
WORD_TEXT   ?= $(TEXT)
WORD_LIMIT  ?= 20

word-train: ## Train the word model on DATA (MODEL_WORD is its file; lengths and counts are per word)
	$(RADIXNET) $(WORD) train --data $(DATA) --epochs $(EPOCHS) --checkpoint-dir $(CKPT_DIR)

word-predict: ## Continue WORD_PREFIX with WORD_LENGTH words (top-K and bottom-K)
	$(RADIXNET) $(WORD) predict --prefix "$(WORD_PREFIX)" --length $(WORD_LENGTH) --mode beam

word-generate: ## Generate COUNT texts of at most MAX_LENGTH words each
	$(RADIXNET) $(WORD) generate --count $(COUNT) --max-length $(MAX_LENGTH) --mode beam

word-score: ## Log-probability of WORD_TEXT, per word (an unread word is <unk>)
	$(RADIXNET) $(WORD) score --text "$(WORD_TEXT)"

word-vocab: ## The words the model has read, most read first (WORD_LIMIT rows, 0 = all)
	$(RADIXNET) $(WORD) words --limit $(WORD_LIMIT)

word-info: ## What the word model holds: its statistics and the training history tail
	$(RADIXNET) $(WORD) info

word-serve: ## Serve the frontend with the word model selected (the Words tab shows its alphabet)
	$(RADIXNET) $(WORD) serve --host $(HOST) --port $(PORT) --checkpoint-dir $(CKPT_DIR) --upload-dir $(UPLOAD_DIR)

word-test: ## The encoding dial's unit tests (Python)
	$(PY) -m unittest tests.test_encoding tests.test_encodings_end_to_end -v

word-demo: ## Train a word model on the sample corpus, then predict, generate and list its alphabet
	$(MAKE) word-train
	$(MAKE) word-predict
	$(MAKE) word-generate
	$(MAKE) word-vocab

word-go: ## The Go port over words: train on DATA and continue WORD_PREFIX (needs `make go-build`)
	go/bin/radixnet-count --encoding $(WORD_ENC) --model $(MODEL_WORD) --seed $(SEED) train --data $(DATA) --epochs $(EPOCHS)
	go/bin/radixnet-count --model $(MODEL_WORD) predict --prefix "$(WORD_PREFIX)" --length $(WORD_LENGTH)

word-rust: ## The Rust port over words: train on DATA and continue WORD_PREFIX (needs `make rust-build`)
	rust/target/release/radixnet --encoding $(WORD_ENC) --model $(MODEL_WORD) --seed $(SEED) train --data $(DATA) --epochs $(EPOCHS)
	rust/target/release/radixnet --model $(MODEL_WORD) predict --prefix "$(WORD_PREFIX)" --length $(WORD_LENGTH)

word-parity: ## Cross-language parity of the word model (Go and Rust against Python)
	$(PY) -m unittest tests.test_go_parity.TestGoWordParity tests.test_rust_parity.TestRustWordParity -v

# ---- the phonetic units: the text read as sounds (../PhoneticTokenizer) ------------
PHONE_ENC   ?= phone:3:1
MODEL_PHONE ?= model.phone.json
PHONE       := --model $(MODEL_PHONE) --kind count --encoding $(PHONE_ENC) --backend $(BACKEND) $(DEVICE_FLAG) --seed $(SEED)
PHONE_PREFIX ?= the cat sat
PHONE_LENGTH ?= 8
SPEECH      ?= speech.wav
export PYTHONPATH := $(PYTHONPATH):../PhoneticTokenizer

phone-train: ## Train a model over sounds on DATA (PHONE_ENC = phone:3:1 | syllable:2:1; MODEL_PHONE is its file)
	$(RADIXNET) $(PHONE) train --data $(DATA) --epochs $(EPOCHS) --checkpoint-dir $(CKPT_DIR)

phone-predict: ## Continue PHONE_PREFIX with PHONE_LENGTH sounds, and say what they spell
	$(RADIXNET) $(PHONE) predict --prefix "$(PHONE_PREFIX)" --length $(PHONE_LENGTH) --mode beam

phone-generate: ## Generate COUNT texts of at most MAX_LENGTH sounds each, spelled back into words
	$(RADIXNET) $(PHONE) generate --count $(COUNT) --max-length $(MAX_LENGTH) --mode beam

phone-words: ## The sounds the model has read, most read first
	$(RADIXNET) $(PHONE) words --limit $(WORD_LIMIT)

phone-test: ## The phonetic units' tests (Python), and the encoding suites they ride in
	$(PY) -m unittest tests.test_phonetic tests.test_voice tests.test_encoding tests.test_encodings_end_to_end -v

phone-demo: ## Train a model over sounds on the sample corpus, then predict, generate, list its sounds and speak
	$(MAKE) phone-train
	$(MAKE) phone-predict
	$(MAKE) phone-generate
	$(MAKE) phone-words
	$(MAKE) speak

speak: ## Hear the model walk: COUNT utterances from PHONE_PREFIX, synthesized as it traverses, into SPEECH
	$(RADIXNET) --model $(MODEL_PHONE) --seed $(SEED) speak --prefix "$(PHONE_PREFIX)" --count $(COUNT) --max-length $(MAX_LENGTH) --out $(SPEECH)

speak-play: ## The same, streamed to a player (aplay, paplay, ffplay, play or afplay) as the walk goes
	$(RADIXNET) --model $(MODEL_PHONE) --seed $(SEED) speak --prefix "$(PHONE_PREFIX)" --count $(COUNT) --max-length $(MAX_LENGTH) --play

WAVS ?= recordings/*.wav
MODEL_EARS ?= model.acoustic.json
acoustic-train: ## Train a model on recordings alone: WAVS heard as acoustic units (--encoding acoustic:3:1) into MODEL_EARS
	$(RADIXNET) --model $(MODEL_EARS) --encoding acoustic:3:1 --seed $(SEED) train --data $(WAVS) --epochs $(EPOCHS)

acoustic-speak: ## Hear the acoustic model walk: COUNT utterances through the vocoder, into SPEECH
	$(RADIXNET) --model $(MODEL_EARS) --seed $(SEED) speak --count $(COUNT) --max-length $(MAX_LENGTH) --out $(SPEECH)

phone-go: ## The Go port over sounds: train on DATA and continue PHONE_PREFIX (needs `make go-build`)
	go/bin/radixnet-count --encoding $(PHONE_ENC) --model $(MODEL_PHONE) --seed $(SEED) train --data $(DATA) --epochs $(EPOCHS)
	go/bin/radixnet-count --model $(MODEL_PHONE) predict --prefix "$(PHONE_PREFIX)" --length $(PHONE_LENGTH)

phone-rust: ## The Rust port over sounds: train on DATA and continue PHONE_PREFIX (needs `make rust-build`)
	rust/target/release/radixnet --encoding $(PHONE_ENC) --model $(MODEL_PHONE) --seed $(SEED) train --data $(DATA) --epochs $(EPOCHS)
	rust/target/release/radixnet --model $(MODEL_PHONE) predict --prefix "$(PHONE_PREFIX)" --length $(PHONE_LENGTH)

phone-parity: ## Cross-language parity of the models over sounds (Go and Rust against Python)
	$(PY) -m unittest tests.test_go_parity.TestGoEncodingParity tests.test_rust_parity.TestRustWordParity.test_the_other_dials_agree_too -v

##@ The negative network (failures only; it filters the positive model)
negative-blame: ## Blame every line of GARBAGE for REASON with SEVERITY (the negative model is NEGATIVE)
	$(RADIXNET) $(GLOBAL) negative --negative $(NEGATIVE) blame --data $(GARBAGE) --reason $(REASON) --severity $(SEVERITY) --source make

negative-clear: ## The tutor passed DATA: take blame off the fragments it shares with known failures
	$(RADIXNET) $(GLOBAL) negative --negative $(NEGATIVE) clear --data $(DATA)

negative-why: ## Why TEXT looks like a failure: the reasons and the fragments carrying them
	$(RADIXNET) $(GLOBAL) negative --negative $(NEGATIVE) why --text "$(TEXT)"

negative-filter: ## Generate COUNT texts through the pair: the positive model writes, the negative one vetoes
	$(RADIXNET) $(GLOBAL) negative --negative $(NEGATIVE) filter --count $(COUNT) --max-length $(MAX_LENGTH)

REVIEWER_FLAGS := $(if $(OLLAMA_URL),--url $(OLLAMA_URL),) $(if $(OLLAMA_MODEL),--reviewer-model $(OLLAMA_MODEL),)
negative-auto: ## Teach it automatically for ROUNDS rounds: the model writes, Ollama reviews, the failures are blamed
	$(RADIXNET) $(GLOBAL) negative --negative $(NEGATIVE) auto --rounds $(ROUNDS) --count $(COUNT) --threshold $(THRESHOLD) $(REVIEWER_FLAGS)

negative-reasons: ## What the tutor has blamed, and the journal of what it said
	$(RADIXNET) $(GLOBAL) negative --negative $(NEGATIVE) reasons

negative-forget: ## Forget the blame behind REASON (the tutor can be wrong too)
	$(RADIXNET) $(GLOBAL) negative --negative $(NEGATIVE) forget --reason $(REASON)

##@ Evolve (GAN-style self-upgrading)
evolve: ## GAN-style self-upgrade for GENERATIONS generations (checkpoints in CKPT_DIR, discriminator in DISC)
	$(RADIXNET) $(GLOBAL) evolve --data $(DATA) --generations $(GENERATIONS) --samples $(SAMPLES) --batch-size $(BATCH) --checkpoint-dir $(CKPT_DIR) --discriminator $(DISC)

evolve-forever: ## Self-upgrade until Ctrl-C (the model is saved on stop)
	$(MAKE) evolve GENERATIONS=0

evolve-blame: ## Self-upgrade and let the discriminator teach the negative network as it goes
	$(RADIXNET) $(GLOBAL) evolve --data $(DATA) --generations $(GENERATIONS) --samples $(SAMPLES) --batch-size $(BATCH) --checkpoint-dir $(CKPT_DIR) --discriminator $(DISC) --blame --negative $(NEGATIVE)

info: ## Model statistics and training history
	$(RADIXNET) $(GLOBAL) info

checkpoints: ## List the checkpoints in CKPT_DIR
	$(RADIXNET) $(GLOBAL) checkpoints --dir $(CKPT_DIR)

restore: ## Restore checkpoint NAME (default: latest) from CKPT_DIR into MODEL
	$(RADIXNET) $(GLOBAL) checkpoints --dir $(CKPT_DIR) --restore $(NAME)

bench: ## Throughput benchmark with BACKEND on CHARS synthetic characters
	$(RADIXNET) --backend $(BACKEND) $(DEVICE_FLAG) bench --chars $(CHARS)

demo: ## Train on the sample corpus, then predict, generate and score
	$(MAKE) train
	$(MAKE) predict
	$(MAKE) generate
	$(MAKE) score

##@ Ollama (local LLM: corpus from a prompt, adversarial review)
OLLAMA_FLAGS := $(if $(OLLAMA_URL),--url $(OLLAMA_URL),) $(if $(OLLAMA_MODEL),--ollama-model $(OLLAMA_MODEL),)

ollama-models: ## List the models installed in Ollama (OLLAMA_URL / OLLAMA_MODEL override $$OLLAMA_HOST / $$RADIXNET_OLLAMA_MODEL)
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) models

ollama-corpus: ## Ask Ollama for LINES lines about PROMPT (STYLE=good|garbage), write them to ollama-$(STYLE).txt and train on them
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) corpus --prompt "$(PROMPT)" --lines $(LINES) --style $(STYLE) --out ollama-$(STYLE).txt --train --epochs $(EPOCHS) --lr $(LR) --batch-size $(BATCH)

ollama-garbage: ## Ask Ollama for LINES deliberately wrong lines about PROMPT and write them to ollama-garbage.txt (no training)
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) corpus --prompt "$(PROMPT)" --lines $(LINES) --style garbage --out ollama-garbage.txt

ollama-review: ## Let Ollama adversarially rate COUNT samples from the model (pass at THRESHOLD)
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) review --count $(COUNT) --max-length $(MAX_LENGTH) --threshold $(THRESHOLD)

ollama-blame: ## Review COUNT samples and teach the negative network why the failures failed (critique -> reason)
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) review --count $(COUNT) --max-length $(MAX_LENGTH) --threshold $(THRESHOLD) --blame --negative $(NEGATIVE)

ollama-correct: ## Copy-edit COUNT samples letter by letter and blame only the characters Ollama changed (SEVERITY per corrected text)
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) correct --count $(COUNT) --max-length $(MAX_LENGTH) --blame --severity $(SEVERITY) --negative $(NEGATIVE)

ollama-2nrl: ## Review COUNT samples, then 2NRL: failed samples as garbage, passed samples + DATA as correct data
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) review --count $(COUNT) --max-length $(MAX_LENGTH) --threshold $(THRESHOLD) --2nrl --good $(DATA) --neg-lr $(NEG_LR) --pos-lr $(POS_LR) --batch-size $(BATCH)

##@ Tutor (the teacher sets the exercise, the model answers, the teacher marks it)
TUTOR_FLAGS := --tutor-provider $(TUTOR) $(if $(TUTOR_URL),--url $(TUTOR_URL),) \
               $(if $(TUTOR_MODEL),--tutor-model $(TUTOR_MODEL),)

tutor: ## English lessons about TOPIC (TUTOR=ollama|chatgpt teaches): ROUNDS rounds of EXERCISES openings, marked out of 10, learned by 2NRL
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --rounds $(ROUNDS) --exercises $(EXERCISES) --drills $(DRILLS) --threshold $(THRESHOLD) --report tutor-report.json

tutor-blame: ## English lessons that also teach the negative network why each sentence failed, and VARIANTS more sentences with the same mistake
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --rounds $(ROUNDS) --exercises $(EXERCISES) --blame --negative $(NEGATIVE) --variants $(VARIANTS)

tutor-dry: ## One round of lessons about TOPIC, marked and reported but not trained on
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --rounds 1 --exercises $(EXERCISES) --threshold $(THRESHOLD) --dry-run

tutor-focus: ## Lessons drilling one point of grammar: FOCUS="past tense"
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --focus "$(FOCUS)" --rounds $(ROUNDS) --exercises $(EXERCISES) --threshold $(THRESHOLD)

tutor-plan: ## Lessons about TOPIC that end with the next PLAN lessons and the brief that teaches them; BRIEF="..." runs a batch to one
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --rounds $(ROUNDS) --exercises $(EXERCISES) --threshold $(THRESHOLD) $(if $(BRIEF),--brief "$(BRIEF)",) --plan $(PLAN) --report tutor-report.json

tutor-auto: ## Auto run: BATCHES batches of ROUNDS rounds about TOPIC, each planned from the report card of the one before (BATCHES=0 runs until Ctrl-C)
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --rounds $(ROUNDS) --exercises $(EXERCISES) --threshold $(THRESHOLD) --batches $(BATCHES) --report tutor-report.json

##@ ChatGPT (OpenAI as the code-generation tutor; needs $$OPENAI_API_KEY)
CHATGPT_FLAGS := $(if $(CHATGPT_URL),--url $(CHATGPT_URL),) $(if $(CHATGPT_MODEL),--chatgpt-model $(CHATGPT_MODEL),)

chatgpt-models: ## List the models $$OPENAI_API_KEY may use (the quickest check that ChatGPT can tutor)
	$(RADIXNET) $(GLOBAL) chatgpt $(CHATGPT_FLAGS) models

chatgpt-ask: ## Send PROMPT to ChatGPT and print the answer
	$(RADIXNET) $(GLOBAL) chatgpt $(CHATGPT_FLAGS) ask --prompt "$(PROMPT)"

##@ Code generation (sandbox + LLM tutor / judge + 2NRL rewards)
CODEGEN_FLAGS := --teacher-provider $(TUTOR) $(if $(CODEGEN_MODEL),--teacher-model $(CODEGEN_MODEL),) \
                 $(if $(CODEGEN_URL),--url $(CODEGEN_URL),)

codegen: ## Solve PROBLEMS with the tutor (TUTOR=ollama|chatgpt, CODEGEN_MODEL) and then the model (PHASE=both|teacher|model), 2NRL after every problem
	$(RADIXNET) $(GLOBAL) codegen --problems $(PROBLEMS) --phase $(PHASE) $(CODEGEN_FLAGS) --checkpoint-dir $(CKPT_DIR) --report codegen-report.json

codegen-teacher: ## Phase 1 only: the tutor writes and fixes the solutions, the network learns them (wrong attempts as garbage)
	$(MAKE) codegen PHASE=teacher

codegen-model: ## Phase 2 only: the network writes the solutions, punished / rewarded by the sandbox and the judge
	$(MAKE) codegen PHASE=model


##@ Speech (teach the model by talking to it)
SPEECH_TEXT := $(if $(TRANSCRIPT),--text "$(TRANSCRIPT)",)

speech-info: ## Which transcription backends, microphone recorders and codecs are available
	$(RADIXNET) $(GLOBAL) speech info

speech-teach: ## Teach AUDIO: its transcript and its waveform, behind one unique token (TRANSCRIPT=... skips the ASR)
	$(RADIXNET) $(GLOBAL) speech teach $(AUDIO) $(SPEECH_TEXT) --rate $(SPEECH_RATE) --train --epochs $(EPOCHS) --lr $(LR) --batch-size $(BATCH)

speech-listen: ## Record SECONDS from the microphone and teach the model what was said and how it sounded
	$(RADIXNET) $(GLOBAL) speech listen --seconds $(SECONDS) $(SPEECH_TEXT) --rate $(SPEECH_RATE) --train --epochs $(EPOCHS) --lr $(LR) --batch-size $(BATCH)

speech-tutor: ## Ask the model to say AUDIO back, mark what comes back and blame what it misremembered
	$(RADIXNET) $(GLOBAL) speech tutor $(AUDIO) $(SPEECH_TEXT) --rate $(SPEECH_RATE) --train --length 400 --blame

speech-decode: ## Turn an encoded or predicted waveform text (TEXT=...) back into AUDIO
	$(RADIXNET) $(GLOBAL) speech decode --text "$(TEXT)" --out $(AUDIO)
OLLAMA_FLAGS_URL := $(if $(OLLAMA_URL),--url $(OLLAMA_URL),)
AGENT_FLAGS := $(OLLAMA_FLAGS_URL) $(if $(AGENT_MODEL),--agent-model $(AGENT_MODEL),) --max-steps $(MAX_STEPS) \
               $(if $(BLAME),--blame,) $(if $(BROWSER),--browser,)

##@ Tool use (browsing + Ollama criteria, judging and teaching)
tools: ## List the external tools the network can call
	$(RADIXNET) $(GLOBAL) tools list

agent: ## Solve TASKS with tools: Ollama writes the acceptance criteria, judges and teaches; 2NRL trains on the failures
	$(RADIXNET) $(GLOBAL) agent --tasks $(TASKS) $(AGENT_FLAGS) --checkpoint-dir $(CKPT_DIR) --report agent-report.json

explore: ## Let the network choose its own tasks and browse for STEPS steps
	$(RADIXNET) $(GLOBAL) explore --steps $(STEPS) $(AGENT_FLAGS) --checkpoint-dir $(CKPT_DIR)

explore-forever: ## Explore until Ctrl-C (the model is saved on stop)
	$(MAKE) explore STEPS=0

mcp: ## Serve the tools and the network over the Model Context Protocol (stdio)
	$(RADIXNET) $(GLOBAL) mcp

##@ Serve (local)
serve: ## Start the HTTP API + prebuilt frontend on http://HOST:PORT (uploads go to UPLOAD_DIR)
	$(RADIXNET) $(GLOBAL) serve --host $(HOST) --port $(PORT) --checkpoint-dir $(CKPT_DIR) --upload-dir $(UPLOAD_DIR)

frontend-install: ## Install the frontend dependencies (npm)
	cd frontend && npm install --no-audit --no-fund

frontend-build: ## Rebuild frontend/dist (what `make serve` serves)
	cd frontend && npm run build

frontend-dev: ## Vite dev server with hot reload on http://localhost:5173 (expects `make serve` running)
	cd frontend && npm run dev

frontend-test: ## Unit tests of the frontend's settings store and site-wide settings (node --test, no npm install needed)
	cd frontend && npm test

##@ Docker (Compose stack)
docker-build: ## Build the image (WITH_TORCH=1 adds torch)
	$(COMPOSE) build

up: ## Start the API + frontend in the background on http://localhost:PORT
	$(COMPOSE) up -d api

up-auto: ## Start the API and immediately kick off the self-upgrade loop inside it
	$(COMPOSE) --profile auto up -d

up-dev: ## API + Vite dev server with hot reload on http://localhost:5173 (foreground)
	$(COMPOSE) --profile dev up

up-gpu: ## Build with torch and start the API with the host's NVIDIA GPUs
	$(GPU_COMPOSE) up -d --build api

up-ollama: ## Start the API plus an Ollama container (pull a model once: docker compose exec ollama ollama pull llama3.2)
	OLLAMA_HOST=http://ollama:11434 $(COMPOSE) --profile ollama up -d

down: ## Stop the stack (all profiles); the data volume is kept
	$(COMPOSE) $(ALL_PROFILES) down

logs: ## Follow the API logs
	$(COMPOSE) logs -f api

shell: ## Shell inside the running API container
	$(COMPOSE) exec api /bin/bash

docker-train: ## One-shot training into the data volume (EPOCHS / LR / BATCH apply)
	$(COMPOSE) run --rm train

docker-evolve: ## Headless self-upgrade loop beside the API (writes /data/model.json when stopped)
	$(COMPOSE) --profile evolve up -d evolve

docker-test: ## Run the unit tests inside the image
	$(COMPOSE) run --rm test

docker-bench: ## Run the benchmark inside the image
	$(COMPOSE) run --rm bench

docker-reload: ## Make the running API load /data/model.json (after docker-train / docker-evolve)
	curl -sS -X POST -H 'Content-Type: application/json' -d '{"path": "/data/model.json"}' http://localhost:$(PORT)/api/load; echo

docker-export: ## Copy the model out of the data volume to ./model-from-docker.json
	$(COMPOSE) cp api:/data/model.json ./model-from-docker.json

docker-clean: ## Remove the containers AND the data volume
	$(COMPOSE) $(ALL_PROFILES) down -v --remove-orphans

##@ Housekeeping
clean: ## Remove Python caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

clean-all: clean ## Also remove local models, checkpoints and frontend node_modules
	rm -rf $(CKPT_DIR) model.json model.json.gz discriminator.json model-from-docker.json frontend/node_modules
