# DISTIL — run commands.  `make help` lists every target.
#
# Everything here drives the zero-dependency Python package through its command
# line (python3 only, no pip install).  `make ui` additionally serves the React
# frontend, whose build output is committed — so the UI runs without node, and
# only changing `ui/` needs `make ui-build`.
#
# Every target works offline: LocalProvider is a deterministic decomposer, not a
# mock, and it is what the suite runs against.  Attach a real model by overriding
# PROVIDER on the command line, e.g. `make ask PROVIDER=ollama TASK="..."`.
#
# HOME_DIR is the state directory.  It defaults to ./state rather than ~/.distil
# so that experimenting here cannot quietly rewrite a real memory — set
# `HOME_DIR=~/.distil` when you want the one the bare CLI uses.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY        ?= python3
HOME_DIR  ?= state
PROVIDER  ?= local
SEED      ?= 0
DISTIL    ?= $(PY) -m distil.cli --home $(HOME_DIR) --provider $(PROVIDER) --seed $(SEED)

TASK      ?= build a csv cleaner and compute the median of each column
CLAIM     ?= we must rewrite the parser in Rust because Python is too slow
QUERY     ?= merge two dicts
GOAL      ?= count how often each word appears in a text
LIKE      ?= the csv has ragged rows
IDEA      ?=
MODEL     ?=
MODULE    ?= reason.py
CHANGE    ?= cache the payoff matrix between calls
TRACE     ?=
SCORE     ?= 1.0
COMMENT   ?=
DEPTH     ?= 3
STEPS     ?= 3
TRIALS    ?= 6
ROUNDS    ?= 2
K         ?= 8
GENERATION ?=

HOST      ?= 127.0.0.1
PORT      ?= 8765
NPM       ?= npm

# MCP servers are added as: make mcp-add NAME=fs CMD="npx -y @modelcontextprotocol/server-filesystem /tmp"
# CMD is the one variable deliberately left unquoted at the point of use: it is
# a command and its arguments, and --add needs them as separate argv words.
NAME      ?=
CMD       ?=

.PHONY: help check test lint demo home \
        up up-ollama pull down logs docker-test \
        seed tools forge \
        ask clarify frame distill why cases recall grade \
        explore upgrade compress compress-now restore auto \
        selfedit lineage rollback \
        mcp mcp-add providers stats stores \
        ui ui-install ui-build ui-dev ui-clean \
        clean clean-state clean-all

help: ## Show this help
	@echo "DISTIL make targets  (override variables like: make ask TASK=\"parse this log\")"
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Defaults: HOME_DIR=$(HOME_DIR) PROVIDER=$(PROVIDER) SEED=$(SEED) PORT=$(PORT)"
	@echo "          State lives in $(HOME_DIR)/, not ~/.distil — this Makefile will not"
	@echo "          write into a real memory unless you ask it to."

##@ Setup
check: ## Byte-compile the package and report which providers are reachable
	$(PY) -m compileall -q distil tests
	$(DISTIL) providers

test: ## Run the suite (335 tests, ~60s, no network, no key)
	$(PY) -m tests.test_distil

lint: ## Byte-compile everything (a syntax check without extra tools)
	$(PY) -m compileall -q distil tests && echo "ok"

demo: ## The whole system end to end, offline: frame, distil, forge, explore, compress
	@# No --home and no --provider on purpose: demo runs LocalProvider in a
	@# throwaway directory, so it cannot touch HOME_DIR or cost anything.
	$(PY) -m distil.cli demo

home: ## Create HOME_DIR, the state directory every other target writes into
	@mkdir -p $(HOME_DIR)

##@ Tools — capability it has to prove before it keeps
seed: | home ## Plant the starter toolkit (12 tools, each through the grader)
	$(DISTIL) seed

tools: | home ## List every registered tool, local and MCP alike
	$(DISTIL) tools

forge: | home ## Write, verify and register a tool for GOAL (non-zero exit: not registered)
	@# A tool that is not strictly better than one already registered is
	@# rejected, and the CLI exits non-zero to say so. That is the contract,
	@# not a build failure -- `make forge GOAL="the middle value of a list"`
	@# after `make seed` will hit it, because median is already in there.
	$(DISTIL) forge "$(GOAL)"

##@ Reasoning — the game, then the goals, then the act
ask: | home ## Reason about TASK and act on it (the full twelve-step chain)
	$(DISTIL) ask "$(TASK)"

clarify: | home ## Ask about TASK until the first step is actionable, instead of guessing
	$(DISTIL) clarify --rounds $(ROUNDS) "$(TASK)"

frame: | home ## Understand TASK as a game: players, payoff, horizon, referee
	$(DISTIL) frame "$(TASK)"

distill: | home ## TASK into goals, stopping where each one becomes checkable
	$(DISTIL) distill "$(TASK)"

why: | home ## Interrogate CLAIM and attack its premises, DEPTH levels down
	$(DISTIL) why --depth $(DEPTH) "$(CLAIM)"

##@ Memory — the embedding layer that is the memory
recall: | home ## Rank K memories for QUERY by similarity x credibility x recency
	$(DISTIL) recall -k $(K) "$(QUERY)"

cases: | home ## Problems and what actually solved them; LIKE="..." finds precedent
	$(DISTIL) cases $(if $(LIKE),--like "$(LIKE)",)

grade: | home ## Grade TRACE=<id> with SCORE (in -1..1); a user grade counts double
	@test -n "$(TRACE)" || { echo "set TRACE=<trace id> (get one from: make recall)"; exit 1; }
	$(DISTIL) grade $(TRACE) $(SCORE) $(if $(COMMENT),--comment "$(COMMENT)",)

compress: | home ## Preview what cold memory would consolidate into digests
	$(DISTIL) compress --dry-run

compress-now: | home ## Consolidate for real (reversible: originals are archived first)
	$(DISTIL) compress

restore: | home ## Bring archived traces back out of the archive
	$(DISTIL) restore

stats: | home ## Memory and tool counts
	$(DISTIL) stats

##@ Memory backend
stores: ## Which memory backend is configured, and what each one needs
	$(DISTIL) stores

##@ Running by itself
CYCLES ?= 0

auto: | home ## Run the autonomous loop for CYCLES cycles (0 = until Ctrl-C)
	@# Six moves, regret-matched and rewarded for information rather than success.
	@# It cannot reach selfedit: an unattended optimiser with write access to its
	@# own grader will edit its own grader.
	$(DISTIL) auto --cycles $(CYCLES)

##@ Exploration — bad ideas are ranked, not avoided
explore: | home ## Brainstorm and run STEPS experiments (IDEA="..." seeds the brainstorm)
	$(DISTIL) explore --steps $(STEPS) $(if $(IDEA),--seed-idea "$(IDEA)",)

upgrade: | home ## Tune the policy against measured outcomes over TRIALS trials
	$(DISTIL) upgrade --trials $(TRIALS)

##@ Self-editing — guarded by the invariants and this suite
selfedit: | home ## Propose CHANGE to MODULE, sandbox it, promote it if it holds (needs a real PROVIDER)
	@# Offline this is expected to be REJECTED and to exit non-zero: LocalProvider
	@# cannot write source, so the edit does not parse and the invariant catches
	@# it. That is the guard working. Set PROVIDER=ollama|openai|anthropic to see
	@# it promote anything. grade.py, sandbox.py and selfedit.py are never
	@# editable: an optimiser that can edit its own grader will edit its own grader.
	$(DISTIL) selfedit $(MODULE) "$(CHANGE)"

lineage: | home ## The history of self-edits, and what each one was allowed to touch
	$(DISTIL) lineage

rollback: | home ## Restore the package to before GENERATION (default: the last one)
	$(DISTIL) rollback $(GENERATION)

##@ MCP — remote tools in the same embedding layer
mcp: | home ## List the configured MCP servers and their tools
	$(DISTIL) mcp

mcp-add: | home ## Add a server: make mcp-add NAME=fs CMD="npx -y @modelcontextprotocol/server-filesystem /tmp"
	@test -n "$(NAME)" -a -n "$(CMD)" || { echo 'usage: make mcp-add NAME=fs CMD="npx -y ..."'; exit 1; }
	$(DISTIL) mcp --add $(NAME) $(CMD)

providers: ## Which providers are reachable, and which of them can embed
	$(DISTIL) providers

##@ Frontend — the claims, made checkable against a running instance
ui: | home ## Serve the frontend on HOST:PORT (loopback only; it executes generated code)
	$(DISTIL) ui --bind $(HOST) --port $(PORT)

ui-install: ## Install the frontend build dependencies (only needed to change ui/)
	cd ui && $(NPM) install --no-audit --no-fund

ui-build: ## Rebuild web/ from ui/ — the committed output `make ui` serves
	cd ui && $(NPM) run build

ui-dev: ## Vite dev server with hot reload, proxying /api to a running `make ui`
	cd ui && $(NPM) run dev

##@ Containers — the agent, plus a browser it is allowed to drive
COMPOSE ?= docker compose

up: ## Start the agent on 127.0.0.1:$(PORT) with a Chrome container beside it
	$(COMPOSE) up -d

up-ollama: ## Also start a local model (then: make pull MODEL=gemma4)
	$(COMPOSE) --profile ollama up -d

pull: ## Pull MODEL into the ollama container
	@test -n "$(MODEL)" || { echo 'usage: make pull MODEL=gemma4'; exit 1; }
	$(COMPOSE) exec ollama ollama pull $(MODEL)

down: ## Stop the stack; the state volume is kept
	$(COMPOSE) --profile ollama down

logs: ## Follow the agent's logs
	$(COMPOSE) logs -f agent

docker-test: ## Run the suite inside the image
	$(COMPOSE) --profile test run --rm test

##@ Housekeeping
ui-clean: ## Remove the frontend's node_modules (web/ is committed and kept)
	rm -rf ui/node_modules ui/.vite

clean: ## Remove Python caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-state: ## Delete HOME_DIR — every memory, tool and case this Makefile made
	rm -rf $(HOME_DIR)

clean-all: clean clean-state ui-clean ## All of the above
