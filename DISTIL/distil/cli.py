"""Command line. `python3 -m distil.cli <command>`.

Every command works offline against `LocalProvider`; attach a real one with
`--provider ollama|openai|anthropic` or `DISTIL_PROVIDER`.
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Distil
from .challenge import Persistence, challenge, interrogate
from .memory import Kind
from .provider import ProviderError, auto, catalogue


def _agent(args) -> Distil:
    provider = auto(args.provider) if args.provider else auto()
    return Distil(provider, home=args.home, seed=args.seed)


def cmd_providers(args) -> int:
    for p in catalogue():
        state = "available" if p.available() else "not configured"
        embed = "embeddings" if p.can_embed else "no embeddings (lexical fallback)"
        print(f"  {p.name:<10} {state:<16} {embed}")
    print(f"\n  auto -> {auto().name}")
    return 0


def cmd_ask(args) -> int:
    d = _agent(args)
    result = d.solve(" ".join(args.task), persist=not args.no_persist,
                     interrogate=not args.no_questions,
                     ask=None if args.no_ask else _prompt)
    if result.get("needs_clarification"):
        # The point of the gate: nothing was distilled, because distilling a task
        # nobody can state the objective of produces a tidy plan for the wrong
        # problem.
        print(result["frame"].render())
        print(f"\n  stopped before reasoning: {result['reason']}")
        print("\n  answer these and try again (or run without --no-ask):")
        for q in result["questions"]:
            print(q.render())
        return 1
    print(result["session"].chain.render())
    print()
    print(result["session"].tree.render())
    print()
    print(f"  solved: {result['solved']}  ({result['reason']})")
    for a in result["attempts"]:
        print(f"    attempt via {a['via']:<24} reframe={a['reframe'] or '-':<12} ok={a['ok']}")
    if result.get("boundary"):
        print("\n" + result["boundary"])
    print(f"\n  chain trace: {result['session'].trace_id}  (grade it: distil grade <id> <-1..1>)")
    return 0


def cmd_why(args) -> int:
    d = _agent(args)
    subject = " ".join(args.subject)

    def ask(question, context):
        hits = d.memory.recall(question, k=2, kinds=(Kind.ANSWER, Kind.FACT, Kind.FAILURE))
        if hits and hits[0].similarity > 0.55:
            return hits[0].trace.text
        from .provider import Message
        try:
            return d.provider.complete([Message("user", "WHY:: " + question)],
                                       temperature=0.3, max_tokens=150)
        except ProviderError:
            return ""

    chain = interrogate(subject, ask, d.memory.embedder, max_depth=args.depth)
    print(chain.render())
    print(f"\n  terminal: {chain.terminal} after {chain.depth} question(s)")
    print("\n  premise attacks, ranked by bits:")
    for q in challenge(subject, d.memory, limit=6):
        print(f"    {q.value:.3f}  {q}")
    return 0


def _prompt(questions) -> dict:
    """Put the questions to whoever is at the terminal. Blank skips one."""
    answers = {}
    print("\n  I cannot state the objective yet. A few questions:\n")
    for q in questions:
        print(f"  [{q.gap}] {q.text}")
        print(f"      ({q.unblocks})")
        try:
            reply = input("      > ").strip()
        except EOFError:
            reply = ""
        if reply:
            answers[q.gap] = reply
        print()
    return answers


def cmd_clarify(args) -> int:
    d = _agent(args)
    ask = None if args.no_ask else _prompt
    out = d.clarifier.clarify(" ".join(args.task), ask=ask, max_rounds=args.rounds)
    print(out.render())
    d.save()
    return 0 if out.actionable else 1


def cmd_mcp(args) -> int:
    d = _agent(args)
    if args.add:
        name, *command = args.add
        if not command:
            print("  usage: --add <name> <command> [args...]", file=sys.stderr)
            return 1
        d.mcp.add(name, command)
        path = d.workspace.home / "mcp.json"
        config = {}
        if path.exists():
            try:
                config = json.loads(path.read_text())
            except json.JSONDecodeError:
                config = {}
        config.setdefault("mcpServers", {})[name] = {"command": command[0],
                                                     "args": command[1:]}
        path.write_text(json.dumps(config, indent=2))
        print(f"  added {name} -> {' '.join(command)}  (saved to {path})")
    if not d.mcp.names():
        print("  no mcp servers configured")
        print(f"  add one:  distil mcp --add fs npx -y @modelcontextprotocol/server-filesystem /tmp")
        print(f"  or write {d.workspace.home / 'mcp.json'} in the usual mcpServers format")
        return 0
    report = d.mcp.discover()
    for server, tools in report["servers"].items():
        print(f"  {server}: {len(tools)} tool(s)")
        for t in tools:
            print(f"      {server}.{t}")
    for failure in report["failed"]:
        print(f"  {failure['server']}: UNAVAILABLE -- {failure['error'][:90]}")
    print(f"\n  {report['tools']} tool(s) embedded; they are now recalled alongside "
          f"locally forged ones, ungraded until used")
    d.save()
    d.mcp.close()
    return 0


def cmd_seed(args) -> int:
    from .seed import plant
    d = _agent(args)
    report = plant(d.toolsmith, d.toolbox)
    for name, score in report["planted"]:
        print(f"  planted  {name:<22} grade {score:+.2f}")
    for name, why in report["rejected"]:
        print(f"  REJECTED {name:<22} {why}")
    if report["already"]:
        print(f"  already present: {', '.join(report['already'])}")
    print(f"\n  toolbox: {', '.join(d.toolbox.names())}")
    d.save()
    return 0


def cmd_frame(args) -> int:
    d = _agent(args)
    task = " ".join(args.task)
    frame, plan = d.understand(task)
    print(frame.render())
    print()
    print(plan.render())
    if not frame.understood:
        print("\n  this frame is below the play threshold: establish the items above "
              "before acting (GREN/DESIGN.md 22.1 makes the same refusal)")
    return 0


def cmd_cases(args) -> int:
    d = _agent(args)
    if args.like:
        out = d.casebook.adapt(" ".join(args.like))
        print(f"  {out['note']}")
        if out["precedent"]:
            print(out["precedent"].render())
        for p in out["avoid"]:
            print(p.render())
        return 0
    cases = d.casebook.all()
    if not cases:
        print("  no cases on record yet")
        return 0
    for c in cases:
        mark = "worked" if c.grade > 0 else "failed"
        print(f"  [{mark}] {c.problem[:56]!r}\n      -> {c.solution[:90]}")
    return 0


def cmd_compress(args) -> int:
    d = _agent(args)
    report = d.compress(dry_run=args.dry_run)
    print(f"  {report['clusters']} cold cluster(s); "
          f"{'would compress' if args.dry_run else 'compressed'} {report['compressed']}, "
          f"folding {report['freed']} trace(s)")
    for digest in report["digests"][:5]:
        print(f"\n  --- cluster of {digest['size']} (cohesion {digest['cohesion']}, "
              f"heat {digest['heat']}) ---")
        for line in digest["text"].splitlines():
            print(f"    {line}")
    if not args.dry_run and report["compressed"]:
        print("\n  originals archived; restore with: distil restore")
    return 0


def cmd_restore(args) -> int:
    d = _agent(args)
    print(f"  restored {d.compressor.restore()} trace(s) from the archive")
    d.save()
    return 0


def cmd_distill(args) -> int:
    d = _agent(args)
    tree = d.reasoner.distill(" ".join(args.task))
    print(tree.render())
    unver = tree.unverifiable()
    if unver:
        print("\n  no check could be written for:")
        for g in unver:
            print(f"    - {g.text}")
        print("  (that list is the specification gap, not a bug)")
    return 0


def cmd_recall(args) -> int:
    d = _agent(args)
    hits = d.memory.recall(" ".join(args.query), k=args.k)
    if not hits:
        print("  nothing recalled")
        return 0
    for h in hits:
        print(f"  {h.trace.id}  {h.explain()}")
    return 0


def cmd_grade(args) -> int:
    d = _agent(args)
    try:
        out = d.ask_user_grade(args.trace_id, args.score, args.comment or "")
    except KeyError:
        print(f"  no trace {args.trace_id}", file=sys.stderr)
        return 1
    print(f"  graded {out['trace']} [{out['kind']}] -> credibility {out['credibility']}")
    return 0


def cmd_forge(args) -> int:
    d = _agent(args)
    goal = " ".join(args.goal)
    spec = d.toolsmith.forge(goal)
    if spec is None:
        print("  no tool could be written for that goal")
        return 1
    grade = d.toolsmith.validate(spec)
    kept = d.toolsmith.register(spec)
    print(grade.report())
    print(f"\n  {'registered' if kept else 'rejected'}: {spec.name}{spec.signature and ' ' + spec.signature}")
    if not kept:
        print("  (recorded as a failure so it is not re-derived)")
    d.save()
    return 0 if kept else 1


def cmd_tools(args) -> int:
    d = _agent(args)
    specs = d.toolbox.all()
    if not specs:
        print("  no tools yet -- try: distil forge 'compute the median of a list'")
        return 0
    for s in specs:
        print(f"  {s.name:<16} {s.signature:<34} solved {len(s.solved)}  [{s.built_by}]")
        for p in s.solved[:3]:
            print(f"      - {p[:80]}")
    return 0


def cmd_explore(args) -> int:
    d = _agent(args)
    print("  ideas on the table, ranked by bits per unit cost:")
    for i in d.explorer.brainstorm(args.seed, n=6):
        print(f"    {i.value(d.policy):.3f}  p={i.p_success:.2f} nov={i.novelty:.2f}  {i}")
    print(f"\n  running {args.steps} experiment(s):")
    for e in d.explore(steps=args.steps, seed=args.seed):
        print(f"    {e.summary()}")
    strat = dict(zip(d.explorer.regret.actions,
                     [round(x, 3) for x in d.explorer.regret.average_strategy()]))
    print(f"\n  regret-matched mix over idea sources: {strat}")
    return 0


def cmd_upgrade(args) -> int:
    d = _agent(args)
    out = d.upgrade(trials=args.trials)
    print(f"  objective {out['score_before']} -> {out['score_after']}")
    if out["changed"]:
        for k, (a, b) in out["changed"].items():
            print(f"    {k}: {a} -> {b}")
    else:
        print("    no mutation beat the incumbent; policy unchanged")
    return 0


def cmd_selfedit(args) -> int:
    d = _agent(args)
    out = d.self_edit(args.module, " ".join(args.instruction))
    print(f"  {'ACCEPTED' if out['ok'] else 'REJECTED'}: {out['reason']}")
    for v in out.get("violations", []):
        print(f"    violation: {v}")
    if out.get("generation"):
        print(f"    generation {out['generation']}  (+/-{out['churn']} lines)")
        print(f"    roll back with: distil rollback")
    return 0 if out["ok"] else 1


def cmd_lineage(args) -> int:
    d = _agent(args)
    gens = d.editor.lineage()
    if not gens:
        print("  no self-edits attempted")
        return 0
    for g in gens:
        print(f"  {g.id}  {'accepted' if g.accepted else 'rejected'}  {g.target:<14} "
              f"+/-{g.churn:<5} {g.reason[:60]}")
    return 0


def cmd_rollback(args) -> int:
    d = _agent(args)
    print("  " + d.editor.rollback(args.generation))
    return 0


def cmd_stats(args) -> int:
    d = _agent(args)
    print(json.dumps(d.stats(), indent=2, sort_keys=True))
    return 0


def cmd_demo(args) -> int:
    """The whole system, offline, in one command."""
    import tempfile
    from pathlib import Path
    d = Distil(auto("local"), home=args.home or tempfile.mkdtemp(), seed=7)
    print("=" * 74)
    print("0. understand the game first -- everything is a game")
    print("=" * 74)
    for task in ("win a chess endgame against a stronger opponent",
                 "negotiate a lease renewal with the landlord every year",
                 "build a csv parser that passes the test suite"):
        f, plan = d.understand(task)
        print(f"  {task}")
        print(f"      {f.players}, {f.payoff}, {f.horizon} -> {f.solution}")
        print(f"      referee: {f.referee or 'NONE -- nothing here can be self-graded'}")
        gaps = [c.action for c in plan.gaps]
        print(f"      capability gaps: {gaps or 'none'}")

    print()
    print("=" * 74)
    print("1. graded recall: the same store, ranked two ways")
    print("=" * 74)
    from .memory import Source
    from .policy import Policy
    good = d.memory.remember(Kind.ANSWER, "to merge two dicts use z = {**a, **b} which copies both")
    bad = d.memory.remember(Kind.ANSWER, "to merge two dicts use a.merge(b) which copies both")
    for _ in range(3):
        d.memory.grade(good.id, 1.0, Source.SELF)
        d.memory.grade(bad.id, -1.0, Source.SELF)
    for weight, label in ((0.0, "similarity only (a plain vector store)"),
                          (d.policy.recall_credibility_weight, "similarity x credibility x recency")):
        d.memory.policy = Policy(recall_credibility_weight=weight)
        top = d.memory.recall("merge two dicts", k=2)[0]
        print(f"  {label:<44} -> {top.trace.text[:46]!r}")
    d.memory.policy = d.policy

    print()
    print("=" * 74)
    print("2. question the task before distilling it")
    print("=" * 74)
    claim = "we must rewrite the parser in Rust because Python is always too slow"
    for q in challenge(claim, d.memory, limit=3):
        print(f"  {q.value:.3f} bits  {q}")

    print()
    print("=" * 74)
    print("3. solve: distil -> choose by game -> write a tool -> verify")
    print("=" * 74)
    r = d.solve("build a csv cleaner and compute the median of each column", interrogate=False)
    print(r["session"].chain.render())
    print(f"\n  solved={r['solved']} via {[a['via'] for a in r['attempts']]}")

    print()
    print("=" * 74)
    print("3b. it asks instead of guessing when it cannot state the objective")
    print("=" * 74)
    for task in ("build a csv parser that passes the test suite", "make the thing better"):
        gated = d.solve(task, interrogate=False)
        if gated.get("needs_clarification"):
            print(f"  {task!r}\n      GATED -- {gated['reason']}")
            print(f"      asks: {gated['questions'][0].text[:66]}")
        else:
            print(f"  {task!r}\n      proceeds -- the task states its own objective")

    print()
    print("=" * 74)
    print("4. never take no for an answer: reframe until the noes repeat")
    print("=" * 74)
    # Clear enough to proceed, impossible to satisfy: the persistence loop runs
    # and returns the boundary rather than a shrug.
    r2 = d.solve("parse a quantum waveform capture file", interrogate=False)
    print(f"  attempts: {[a['reframe'] or 'direct' for a in r2.get('attempts', [])]}")
    print(f"  stopped because: {r2['reason']}")
    print("  " + (r2.get("boundary") or "").replace("\n", "\n  "))

    print()
    print("=" * 74)
    print("5. explore: curiosity first, ranked by bits per unit cost")
    print("=" * 74)
    for i in d.explorer.brainstorm(n=4):
        print(f"  {i.value(d.policy):.3f}  {i}")
    for e in d.explore(steps=2):
        print(f"  -> {e.summary()}")

    print()
    print("=" * 74)
    print("6. invention by analogy: what worked over there, tried over here")
    print("=" * 74)
    from .memory import Source as _S
    a = d.memory.remember(Kind.FACT, "retrying a flaky network call with exponential "
                                     "backoff fixed the intermittent timeout failures")
    d.memory.grade(a.id, 1.0, _S.SELF)
    b = d.memory.remember(Kind.FACT, "caching the parsed schema removed the repeated startup cost")
    d.memory.grade(b.id, 1.0, _S.SELF)
    d.memory.remember(Kind.FAILURE, "the integration test suite fails intermittently with a timeout")
    d.memory.remember(Kind.FAILURE, "the importer re-parses the same schema on every record")
    for i in d.explorer.analogies(limit=3):
        print(f"  {i.text}")

    print()
    print("=" * 74)
    print("7. the casebook: problems, and what actually solved them")
    print("=" * 74)
    d.casebook.record("csv rows have ragged column counts",
                      "pad short rows to the header width before parsing", 1.0, via="parse_csv")
    d.casebook.record("csv rows have ragged column counts",
                      "drop any row that is not the header width", -1.0, via="naive")
    out = d.casebook.adapt("the csv export has ragged rows with inconsistent column counts")
    print(f"  {out['note']}")
    print(out["precedent"].render())
    for p in out["avoid"]:
        print(p.render())

    print()
    print("=" * 74)
    print("8. one embedding layer over local tools and MCP tools alike")
    print("=" * 74)
    from .seed import plant
    planted = plant(d.toolsmith, d.toolbox)
    print(f"  starter toolkit: {len(planted['planted'])} planted, "
          f"{len(planted['rejected'])} rejected, all graded before entry")
    import sys as _sys
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "echo_mcp_server.py"
    if fixture.exists():
        d.mcp.add("echo", [_sys.executable, str(fixture)])
        report = d.mcp.discover()
        print(f"  mcp server 'echo': {report['tools']} tool(s) embedded (ungraded)")
        print(f"  invoke local  median   -> {d.toolbox.invoke('median', [[5, 3, 1, 4]]).get('value')}")
        print(f"  invoke remote echo.add -> {d.toolbox.invoke('echo.add', kwargs={'a': 2, 'b': 40}).get('value')!r}")
        d.mcp.close()
    print(f"\n  toolbox: {', '.join(d.toolbox.names())}")
    print(f"\n  memory: {json.dumps(d.memory.stats())}")
    print(f"  cases:  {len(d.casebook.all())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distil", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", help="ollama | openai | anthropic | local")
    p.add_argument("--home", help="state directory (default $DISTIL_HOME or ~/.distil)")
    p.add_argument("--seed", type=int, default=None)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ask", help="reason about a task and act on it")
    s.add_argument("task", nargs="+")
    s.add_argument("--no-persist", action="store_true", help="stop at the first refusal")
    s.add_argument("--no-questions", action="store_true", help="skip the interrogation")
    s.add_argument("--no-ask", action="store_true",
                   help="do not prompt; print the clarifying questions and stop")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("clarify", help="ask until the first step is actionable")
    s.add_argument("task", nargs="+")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--no-ask", action="store_true", help="print the questions and stop")
    s.set_defaults(fn=cmd_clarify)

    s = sub.add_parser("mcp", help="mcp servers and their tools")
    s.add_argument("--add", nargs="+", metavar="ARG",
                   help="add a server: --add <name> <command> [args...]")
    s.set_defaults(fn=cmd_mcp)

    sub.add_parser("seed", help="plant the starter toolkit").set_defaults(fn=cmd_seed)

    s = sub.add_parser("why", help="interrogate a claim and attack its premises")
    s.add_argument("subject", nargs="+")
    s.add_argument("--depth", type=int, default=5)
    s.set_defaults(fn=cmd_why)

    s = sub.add_parser("frame", help="understand the game before playing it")
    s.add_argument("task", nargs="+")
    s.set_defaults(fn=cmd_frame)

    s = sub.add_parser("cases", help="problems and what solved them")
    s.add_argument("--like", nargs="+", help="find precedents for a problem")
    s.set_defaults(fn=cmd_cases)

    s = sub.add_parser("compress", help="consolidate cold memory into digests")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_compress)

    sub.add_parser("restore", help="bring archived traces back").set_defaults(fn=cmd_restore)

    s = sub.add_parser("distill", help="task -> goals, stopping at verifiability")
    s.add_argument("task", nargs="+")
    s.set_defaults(fn=cmd_distill)

    s = sub.add_parser("recall", help="query the memory")
    s.add_argument("query", nargs="+")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_recall)

    s = sub.add_parser("grade", help="grade a trace by hand")
    s.add_argument("trace_id")
    s.add_argument("score", type=float)
    s.add_argument("--comment")
    s.set_defaults(fn=cmd_grade)

    s = sub.add_parser("forge", help="write, verify and register a tool")
    s.add_argument("goal", nargs="+")
    s.set_defaults(fn=cmd_forge)

    s = sub.add_parser("tools", help="list registered tools")
    s.set_defaults(fn=cmd_tools)

    s = sub.add_parser("explore", help="brainstorm and run experiments")
    s.add_argument("--steps", type=int, default=3)
    s.add_argument("--seed-idea", dest="seed")
    s.set_defaults(fn=cmd_explore)

    s = sub.add_parser("upgrade", help="tune the policy against measured outcomes")
    s.add_argument("--trials", type=int, default=8)
    s.set_defaults(fn=cmd_upgrade)

    s = sub.add_parser("selfedit", help="propose, test and promote an edit to its own source")
    s.add_argument("module", help="e.g. reason.py")
    s.add_argument("instruction", nargs="+")
    s.set_defaults(fn=cmd_selfedit)

    s = sub.add_parser("lineage", help="the history of self-edits")
    s.set_defaults(fn=cmd_lineage)

    s = sub.add_parser("rollback", help="restore the package to before a generation")
    s.add_argument("generation", nargs="?")
    s.set_defaults(fn=cmd_rollback)

    sub.add_parser("providers", help="what is reachable").set_defaults(fn=cmd_providers)
    sub.add_parser("stats", help="memory and tool counts").set_defaults(fn=cmd_stats)
    sub.add_parser("demo", help="the whole system, offline").set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ProviderError as exc:
        print(f"  provider error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
