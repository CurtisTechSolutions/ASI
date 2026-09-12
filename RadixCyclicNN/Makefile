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
COUNT       ?= 5
MAX_LENGTH  ?= 60
TEXT        ?= the cat sat on the mat
GENERATIONS ?= 3
SAMPLES     ?= 8
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
TOPIC       ?= everyday life
FOCUS       ?= subject-verb agreement
ROUNDS      ?= 3
EXERCISES   ?= 5
DRILLS      ?= 0
PROBLEMS    ?= data/sample_problems.jsonl
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

.PHONY: help install install-gpu check test go-build go-test go-parity go-serve train resume predict generate score 2nrl invert compress \
        evolve evolve-forever info checkpoints restore bench demo serve frontend-install frontend-build \
        frontend-dev ollama-models ollama-corpus ollama-garbage ollama-review ollama-2nrl \
        chatgpt-models chatgpt-ask codegen codegen-teacher codegen-model \
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

go-parity: ## Cross-language parity of the Go port against the Python model
	$(PY) -m unittest tests.test_go_parity -v

MODEL_COUNT ?= model.count.json
go-serve: go-build ## Serve the frontend from the Go count / reward model on http://HOST:PORT (uploads / checkpoints as for `serve`)
	go/bin/radixnet-count --model $(MODEL_COUNT) --seed $(SEED) serve --host $(HOST) --port $(PORT) --frontend-dir frontend/dist --upload-dir $(UPLOAD_DIR) --checkpoint-dir $(CKPT_DIR)

##@ Train / predict (local)
train: ## Train on DATA (incremental: MODEL is loaded first if it exists), checkpointing into CKPT_DIR
	$(RADIXNET) $(GLOBAL) train --data $(DATA) --epochs $(EPOCHS) --lr $(LR) --batch-size $(BATCH) --checkpoint-dir $(CKPT_DIR)

resume: ## Continue training from the latest checkpoint in CKPT_DIR
	$(RADIXNET) $(GLOBAL) train --data $(DATA) --epochs $(EPOCHS) --lr $(LR) --batch-size $(BATCH) --checkpoint-dir $(CKPT_DIR) --resume

predict: ## Continue PREFIX by shortest path (MODE=dijkstra|sample) for LENGTH characters
	$(RADIXNET) $(GLOBAL) predict --prefix "$(PREFIX)" --length $(LENGTH) --mode $(MODE)

generate: ## Generate COUNT texts from scratch (up to MAX_LENGTH characters each)
	$(RADIXNET) $(GLOBAL) generate --count $(COUNT) --max-length $(MAX_LENGTH)

score: ## Log-probability of TEXT under the model
	$(RADIXNET) $(GLOBAL) score --text "$(TEXT)"

2nrl: ## 2NRL: train on GARBAGE, invert the network, fine-tune on DATA
	$(RADIXNET) $(GLOBAL) 2nrl --bad $(GARBAGE) --good $(DATA) --neg-lr $(NEG_LR) --pos-lr $(POS_LR) --batch-size $(BATCH)

invert: ## Invert the network (flip every edge weight and activation amplitude)
	$(RADIXNET) $(GLOBAL) invert

compress: ## Merge unary chains (radix-tree path compression) and save
	$(RADIXNET) $(GLOBAL) compress

evolve: ## GAN-style self-upgrade for GENERATIONS generations (checkpoints in CKPT_DIR, discriminator in DISC)
	$(RADIXNET) $(GLOBAL) evolve --data $(DATA) --generations $(GENERATIONS) --samples $(SAMPLES) --batch-size $(BATCH) --checkpoint-dir $(CKPT_DIR) --discriminator $(DISC)

evolve-forever: ## Self-upgrade until Ctrl-C (the model is saved on stop)
	$(MAKE) evolve GENERATIONS=0

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

ollama-2nrl: ## Review COUNT samples, then 2NRL: failed samples as garbage, passed samples + DATA as correct data
	$(RADIXNET) $(GLOBAL) ollama $(OLLAMA_FLAGS) review --count $(COUNT) --max-length $(MAX_LENGTH) --threshold $(THRESHOLD) --2nrl --good $(DATA) --neg-lr $(NEG_LR) --pos-lr $(POS_LR) --batch-size $(BATCH)

##@ Tutor (the teacher sets the exercise, the model answers, the teacher marks it)
TUTOR_FLAGS := --tutor-provider $(TUTOR) $(if $(TUTOR_URL),--url $(TUTOR_URL),) \
               $(if $(TUTOR_MODEL),--tutor-model $(TUTOR_MODEL),)

tutor: ## English lessons about TOPIC (TUTOR=ollama|chatgpt teaches): ROUNDS rounds of EXERCISES openings, marked out of 10, learned by 2NRL
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --rounds $(ROUNDS) --exercises $(EXERCISES) --drills $(DRILLS) --threshold $(THRESHOLD) --report tutor-report.json

tutor-dry: ## One round of lessons about TOPIC, marked and reported but not trained on
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --rounds 1 --exercises $(EXERCISES) --threshold $(THRESHOLD) --dry-run

tutor-focus: ## Lessons drilling one point of grammar: FOCUS="past tense"
	$(RADIXNET) $(GLOBAL) tutor $(TUTOR_FLAGS) --topic "$(TOPIC)" --focus "$(FOCUS)" --rounds $(ROUNDS) --exercises $(EXERCISES) --threshold $(THRESHOLD)

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

##@ Serve (local)
serve: ## Start the HTTP API + prebuilt frontend on http://HOST:PORT (uploads go to UPLOAD_DIR)
	$(RADIXNET) $(GLOBAL) serve --host $(HOST) --port $(PORT) --checkpoint-dir $(CKPT_DIR) --upload-dir $(UPLOAD_DIR)

frontend-install: ## Install the frontend dependencies (npm)
	cd frontend && npm install --no-audit --no-fund

frontend-build: ## Rebuild frontend/dist (what `make serve` serves)
	cd frontend && npm run build

frontend-dev: ## Vite dev server with hot reload on http://localhost:5173 (expects `make serve` running)
	cd frontend && npm run dev

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
