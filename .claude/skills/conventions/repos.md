# Repository quick reference

Facts the assistant established in prior sessions. Verify commands against the repo's Makefile before relying on them.

## CurtisTechSolutions/ICPSignal (130 sessions)
Lead-finding SaaS: scrapes Reddit and other communities, scores leads against an ICP, drafts replies, syncs to CRMs (HubSpot, Salesforce, Pipedrive), exposes an MCP server. Go backend with River jobs and sqlc, React/TS in `web/`, Postgres migrations. Vocabulary: "ICP" and "Leads", not "profile" or "signals". CI: `go vet`, `go test ./... -race`, `npm audit`, `make generate-check`; `make deploy`; `docker-publish.yml` on release. Prod: Docker Compose behind Caddy on a DigitalOcean droplet (`icpsignal.com`), staging deploys on every push to main, Kubernetes with Helm and Flux via `CurtisTechSolutions/gitops` (`staging.icpsignal.com`). Stripe plans $99/$299/$799 monthly, annual with free months, `icpsignal stripe setup` CLI. Settings in the admin panel override `.env`. Email via Resend; feedback and alerts via Discord webhooks; GA4 + Meta Pixel. Browser worker nodes (Chrome WebDriver / Selenium grid) reached over private IPs.

## CurtisTechSolutions/ASI (64 sessions)
ML research: 2NRL, RadixCyclicNN, LatticeFSM, GTMNN, phonetic tokenizer, sine-wave activation (never modify the formula). Parallel Rust, Go and Python implementations kept in parity; each new model gets its own directory and branch, a Rust backend with `make serve`, and the ModelKit React frontend. Ollama (gemma4) as grader, tutor and critic; Stockfish for chess experiments. Commands: `make serve`, `make test`, `make go-test`, `make rust-test`, `go vet`, `cargo fmt`, `cargo test`. Insights go in the research directory and the insights markdown file. No artificial limits on uploads; uncapped goroutines when training.

## CurtisTechSolutions/crate-arcade and loot-arcade (33 sessions)
Token and crate casino site (creator panel, wallets, loyalty, affiliates, bonus codes). Go backend (chi, GORM, Postgres enums) plus React/TS; PhaserJS crate-opening game. Squash merges; PR template. CI: `tsc --noEmit`, `eslint`, `prettier`, `go vet`, `go test ./...`. Compose stack with external Postgres URL. Settings (caps, rewards, expiries, RTP, commission) are DB-adjustable from the admin panel. Mobile layout must be checked.

## CurtisTechSolutions/games (17 sessions)
Slot machines served to the arcade sites ("CTS Games"). Go backend, wallet and webhook flow (currency comes from the site in the request, then to the DB); CLI mirrors admin-panel commands. Engine tests with idempotency keys. Deploy script for DigitalOcean via Compose; Helm chart combining games and crate-arcade lives in `CurtisTechSolutions/infrastructure`; K3s with external DB.

## CurtisTechSolutions/FinanceAI (12 sessions)
Trading automation (Coinbase integration, strategies, backtests, paper and live modes). `./fai` wrapper over Docker Compose; `make paper`, `make live`, `make lint`, `make ghcr-login push`. Rust `financeai-engine` crate. Deployed to a DigitalOcean droplet. Trading rules: price, volume and volatility only; "less is more".

## CurtisTechSolutions/FinTools and TopstepX (10 sessions)
FinTools: Python (Typer CLI, ruff, mypy) with a web frontend, broker accounts (TopstepX), MCP server; credentials entered through frontend forms. TopstepX: multi-language SDK (`python/`, `go/`, `rust/`) published to PyPI, crates.io and Go modules; `mypy --strict`, `ruff`, `cargo fmt`.

## CurtisTechSolutions/LeadFinder (4 sessions)
Reddit scraping pipeline with Postgres and NATS, Compose stack with external DB and NATS from `.env`, Makefile with `make check`, `make up`, `make help`. Where the global rules on authorship and semver PR names were stated.

## CurtisTechSolutions/CTS-Website, backend-rescue (5 sessions)
Company site (Curtis Tech Solutions LLC): GA4 via the React package (not gtag.js), privacy policy and terms with triple-checked facts, SEO and AI optimization, release-please releases, Fly deploy needs `FLY_API_TOKEN`.

## CurtisTechSolutions/gitops and infrastructure
Flux-managed Kubernetes config (`kubectl`, `kubectl kustomize`, `terraform validate`, `./scripts/setup.sh`) and Helm charts for the arcade and games services.

## CurtisTechSolutions/EchoApp + echobackend, AmyAI-CORE, Finny, GamesForAI
EchoApp: React Native/Expo (iOS via Xcode), `npm ci`, `tsc --noEmit`, `eslint .`, swipe-card UX, violet palette with a dark-mode variant. AmyAI-CORE: Go port of Python functionality, PRD-first. Finny: trading toolkit PRD (backtesting, forward testing, MCP server, Ollama/ChatGPT hooks). GamesForAI: Rust backend, React + PhaserJS frontend, "LLM- and MCP-friendly" APIs.

## nosam1998 (personal): AutoHypedrop, ParticleSim, HordeFoundry, ClanWarsReborn
AutoHypedrop: Playwright/Chromium automation in Docker with Makefile (`make run`, `make push`), Discord notifier for captchas and login links, GHCR images; stale Chromium profile locks across containers cause login failures. ParticleSim: universe simulator design doc and issue backlog. HordeFoundry: game built with Opus workers and Blender-generated assets. ClanWarsReborn: PRD with Discord bot, webhooks and push notifications.

<!-- synced from the orchestration plugin (plugins/orchestration in CurtisTechSolutions/AutoMailFilter); edit there and re-run scripts/sync-to-project.sh -->
