import { Fragment, useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines, yesNo } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import UploadPicker from "./UploadPicker.jsx";
import { CheckField, NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";

const MAX_PROBLEM_ROWS = 200;
const MAX_ROUND_LINES = 50;
const MAX_ATTEMPTS = 20;
const DEFAULT_TEACHER_MODEL = "gemma4";
const DEFAULT_CHATGPT_MODEL = "gpt-4o-mini";
const PROVIDER_LABELS = { ollama: "Ollama", chatgpt: "ChatGPT" };

/** "Ollama" / "ChatGPT" for a provider name. */
function providerLabel(provider) {
  return PROVIDER_LABELS[provider] || PROVIDER_LABELS.ollama;
}

/** What the tutor needs to work, per provider. */
function providerNote(provider) {
  return provider === "chatgpt"
    ? "The teacher and the judge call OpenAI with the server's OPENAI_API_KEY; sandbox runs work without one."
    : "The teacher and the judge need a running Ollama server; sandbox runs work without one.";
}

const slowNote = (provider) => `${providerLabel(provider)} is working; this can take a minute or two.`;

/** Model / sandbox settings shared by the training job, "Try a problem" and the sandbox (numbers kept as strings). */
const DEFAULT_SETTINGS = {
  phases: "both",
  rounds: "1",
  teacherProvider: "ollama",
  teacherModel: DEFAULT_TEACHER_MODEL,
  url: "",
  teacherAttempts: "3",
  modelAttempts: "4",
  strictness: "strict",
  temperature: "1.0",
  maxLength: "800",
  sandboxTimeout: "10",
  memoryMb: "256",
  judge: true,
  fallbackTeacher: true,
  replay: true,
  networkIsolation: true,
  twonrlPer: "problem",
  negEpochs: "2",
  posEpochs: "3",
  negLr: "0.5",
  posLr: "0.1",
  batchSize: "4",
};

/**
 * Server-side defaults of one provider, reported by /api/status as
 * {"ollama": {"url", "model"}, "chatgpt": {"url", "model", "configured"}}.
 * `configured` is false when the server has no OPENAI_API_KEY.
 */
function providerDefaults(status, provider) {
  const key = provider === "chatgpt" ? "chatgpt" : "ollama";
  const o = status && status[key] && typeof status[key] === "object" ? status[key] : {};
  return {
    url: typeof o.url === "string" ? o.url : "",
    model: typeof o.model === "string" ? o.model : "",
    configured: key === "ollama" ? true : Boolean(o.configured),
  };
}

/** The model a provider tutors with when the field is left blank (the server applies the same default). */
function defaultModel(status, provider) {
  if (provider !== "chatgpt") return DEFAULT_TEACHER_MODEL;
  return providerDefaults(status, "chatgpt").model || DEFAULT_CHATGPT_MODEL;
}

/** Request fields every codegen call takes: the tutor, its model and URL (blank = server default), strictness, sandbox. */
function sharedFields(s) {
  const url = s.url.trim();
  const model = s.teacherModel.trim();
  return {
    teacher_provider: s.teacherProvider,
    ...(model ? { teacher_model: model } : {}),
    ...(url ? { url } : {}),
    strictness: s.strictness,
    temperature: Math.max(0, parseNumber(s.temperature, 1)),
    max_length: Math.max(1, parseInteger(s.maxLength, 800)),
    sandbox_timeout: Math.max(0.1, parseNumber(s.sandboxTimeout, 10)),
    memory_mb: Math.max(0, parseInteger(s.memoryMb, 256)),
    network_isolation: s.networkIsolation,
  };
}

/** Problem lines of the textarea; the server ignores blank lines and # comments the same way. */
function problemLines(text) {
  return splitLines(text).filter((line) => !line.trimStart().startsWith("#"));
}

/** A problem for /api/codegen/solve: the bare prompt, or an object when tests / expected output are given. */
function problemPayload(prompt, tests, expected) {
  const problem = { prompt: prompt.trim() };
  if (tests.trim() !== "") problem.tests = tests;
  if (expected.trim() !== "") problem.expected_output = expected;
  return problem.tests === undefined && problem.expected_output === undefined ? problem.prompt : problem;
}

/** Badge class and text of an outcome: correct, runs but rejected, or did not run. */
function outcome(correct, runs) {
  if (correct) return { cls: "pass", text: "correct" };
  if (runs) return { cls: "unrated", text: "runs but rejected" };
  return { cls: "fail", text: "error" };
}

function sourceClass(source) {
  if (source === "chatgpt") return "chatgpt";
  return source === "ollama" ? "ollama" : "model";
}

/** The sandbox error of an attempt record, else its first issue. */
function firstIssue(record) {
  if (record.error) return String(record.error);
  const issues = asArray(record.issues);
  return issues.length > 0 ? String(issues[0]) : "";
}

/** "2NRL punish (neg loss 0.1234)" for records carrying a per-problem / per-round 2NRL result. */
function learnText(record) {
  if (!("action" in record)) return "";
  const parts = [];
  if (typeof record.neg_loss === "number") parts.push(`neg loss ${fmtNum(record.neg_loss, 4)}`);
  if (typeof record.pos_loss === "number") parts.push(`pos loss ${fmtNum(record.pos_loss, 4)}`);
  return `2NRL ${record.action ? String(record.action) : "none"}${parts.length ? ` (${parts.join(", ")})` : ""}`;
}

const isObject = (value) => Boolean(value) && typeof value === "object";
const tri = (value) => (value === null || value === undefined ? "–" : yesNo(value));

function IssueList({ issues, title }) {
  const list = asArray(issues).map((issue) => String(issue));
  if (list.length === 0) return null;
  return (
    <>
      {title ? <h4>{title}</h4> : null}
      <ul className="issues">
        {list.map((issue, i) => (
          <li key={i}>{issue}</li>
        ))}
      </ul>
    </>
  );
}

/** Exit status, timing and output of one sandbox run. */
function RunDetails({ run }) {
  const r = isObject(run) ? run : {};
  return (
    <>
      <h4>Run</h4>
      <dl className="kv">
        <dt>ran</dt>
        <dd>
          {tri(r.ok)}
          {r.timed_out ? " (timed out)" : ""}
        </dd>
        <dt>exit code</dt>
        <dd>{r.exit_code === null || r.exit_code === undefined ? "–" : String(r.exit_code)}</dd>
        <dt>seconds</dt>
        <dd>{fmtNum(r.seconds, 3)}</dd>
        <dt>expected output</dt>
        <dd>{r.expected_ok === null || r.expected_ok === undefined ? "not checked" : r.expected_ok ? "matched" : "differs"}</dd>
        <dt>network isolated</dt>
        <dd>{tri(r.network_isolated)}</dd>
        {r.error ? (
          <>
            <dt>error</dt>
            <dd className="error-text">{String(r.error)}</dd>
          </>
        ) : null}
      </dl>
      <h4>stdout</h4>
      <pre className="output">{String(r.stdout ?? "") || "(empty)"}</pre>
      {r.stderr ? (
        <>
          <h4>stderr</h4>
          <pre className="output">{String(r.stderr)}</pre>
        </>
      ) : null}
    </>
  );
}

/** Objective style report: syntax, PEP 8 layout, naming. */
function StyleDetails({ style }) {
  const s = isObject(style) ? style : {};
  const flags = [
    ["syntax", s.syntax_ok],
    ["PEP 8", s.pep8_ok],
    ["naming", s.naming_ok],
  ];
  return (
    <>
      <h4>Style</h4>
      <div className="job-status">
        <span className={`badge ${s.ok ? "pass" : "fail"}`}>{s.ok ? "ok" : "issues"}</span>
        {flags.map(([name, ok]) => (
          <span className="muted" key={name}>
            {name} {ok === undefined ? "–" : ok ? "✓" : "✗"}
          </span>
        ))}
      </div>
      <IssueList issues={s.issues} />
    </>
  );
}

/** Combined verdict of the sandbox, the style checker and (when used) the Ollama judge. */
function VerdictDetails({ verdict }) {
  const v = isObject(verdict) ? verdict : {};
  const o = outcome(Boolean(v.correct), Boolean(v.runs));
  return (
    <>
      <h4>Verdict</h4>
      <div className="job-status">
        <span className={`badge ${o.cls}`}>{o.text}</span>
        <span className="muted">score {fmtNum(v.score, 1)}</span>
        <span className="muted">judged by {String(v.judged_by ?? "–")}</span>
        <span className="muted">
          runs {tri(v.runs)} · task {tri(v.task)} · PEP 8 {tri(v.pep8)} · naming {tri(v.naming)}
        </span>
      </div>
      <IssueList issues={v.issues} />
      {v.critique ? <p className="critique">{String(v.critique)}</p> : null}
    </>
  );
}

/** One attempt of /api/codegen/solve: code, sandbox run, style report and verdict. */
function AttemptBlock({ attempt }) {
  const a = isObject(attempt) ? attempt : {};
  const v = isObject(a.verdict) ? a.verdict : {};
  const o = outcome(Boolean(a.correct ?? v.correct), Boolean(v.runs));
  const number = typeof a.index === "number" ? a.index + 1 : undefined;
  return (
    <div className="attempt-block">
      <div className="job-status">
        <b>attempt {fmtInt(number)}</b>
        <span className={`badge ${sourceClass(a.source)}`}>{String(a.source ?? "–")}</span>
        <span className={`badge ${o.cls}`}>{o.text}</span>
        <span className="muted">score {fmtNum(v.score, 1)}</span>
        <span className="muted">judged by {String(v.judged_by ?? "–")}</span>
        <span className="muted">{fmtNum(a.seconds, 2)} s</span>
      </div>
      <pre className="code">{String(a.code ?? "")}</pre>
      <RunDetails run={a.run} />
      <StyleDetails style={a.style} />
      <VerdictDetails verdict={a.verdict} />
    </div>
  );
}

/** One line per round record: solved counts, the model's share and the per-round 2NRL step when there is one. */
function RoundSummary({ rounds, running }) {
  if (rounds.length === 0) {
    return <p className="muted">{running ? "Waiting for the first round…" : "No rounds yet."}</p>;
  }
  const shown = rounds.slice(-MAX_ROUND_LINES);
  return (
    <>
      {rounds.length > shown.length ? (
        <p className="muted">
          Showing the last {shown.length} of {rounds.length} rounds.
        </p>
      ) : null}
      <ul className="rounds">
        {shown.map((r, i) => {
          const learned = learnText(r);
          return (
            <li key={i}>
              <b>
                round {fmtInt(r.round)} · {String(r.phase ?? "–")} phase
              </b>
              : {fmtInt(r.solved)} / {fmtInt(r.problems)} solved, {fmtInt(r.model_solved)} by the model
              {learned ? ` · ${learned}` : ""} · {fmtNum(r.seconds, 1)} s
            </li>
          );
        })}
      </ul>
    </>
  );
}

/** Problem records; a row expands (click, or the arrow) to the final code and its issues. */
function ProblemTable({ rows, total, running }) {
  const [open, setOpen] = useState(() => new Set());

  function toggle(key) {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  if (rows.length === 0) {
    return <p className="muted">{running ? "Waiting for the first problem…" : "No problem records yet."}</p>;
  }
  return (
    <>
      {total > rows.length ? (
        <p className="muted">
          Showing the last {rows.length} of {total} problem records.
        </p>
      ) : null}
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th></th>
              <th>phase</th>
              <th>round</th>
              <th>problem</th>
              <th>attempts</th>
              <th>result</th>
              <th>solved by</th>
              <th>action</th>
              <th>neg loss</th>
              <th>pos loss</th>
              <th>seconds</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ record: r, key }) => {
              const expanded = open.has(key);
              return (
                <Fragment key={key}>
                  <tr className="clickable" onClick={() => toggle(key)}>
                    <td>
                      <button
                        type="button"
                        className="link"
                        aria-expanded={expanded}
                        aria-label={expanded ? "Hide the code" : "Show the code"}
                        onClick={(event) => {
                          event.stopPropagation();
                          toggle(key);
                        }}
                      >
                        {expanded ? "▾" : "▸"}
                      </button>
                    </td>
                    <td className="text">{String(r.phase ?? "–")}</td>
                    <td>{fmtInt(r.round)}</td>
                    <td className="text">{String(r.problem ?? "–")}</td>
                    <td>{fmtInt(r.attempts)}</td>
                    <td>
                      <span className={`badge ${r.correct ? "pass" : "fail"}`}>{r.correct ? "correct" : "failed"}</span>
                    </td>
                    <td className="text">{r.solved_by ? String(r.solved_by) : "–"}</td>
                    <td className="text">{r.action ? String(r.action) : "–"}</td>
                    <td>{fmtNum(r.neg_loss, 4)}</td>
                    <td>{fmtNum(r.pos_loss, 4)}</td>
                    <td>{fmtNum(r.seconds, 2)}</td>
                  </tr>
                  {expanded ? (
                    <tr className="detail">
                      <td colSpan={11}>
                        {r.prompt ? <p className="muted">{String(r.prompt)}</p> : null}
                        <dl className="kv">
                          <dt>score</dt>
                          <dd>{fmtNum(r.score, 1)}</dd>
                          <dt>solved by the model</dt>
                          <dd>{tri(r.model_solved)}</dd>
                          <dt>bad / good programs</dt>
                          <dd>
                            {fmtInt(r.bad)} / {fmtInt(r.good)}
                          </dd>
                        </dl>
                        <h4>Final code</h4>
                        <pre className="code">{String(r.code ?? "") || "(empty)"}</pre>
                        <IssueList issues={r.issues} title="Issues" />
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** The most recent attempt records, newest first; each expands to its code and stdout. */
function AttemptFeed({ attempts, total, running }) {
  if (attempts.length === 0) {
    return <p className="muted">{running ? "Waiting for the first attempt…" : "No attempts yet."}</p>;
  }
  return (
    <>
      <p className="muted">
        Newest first{total > attempts.length ? `, the last ${attempts.length} of ${total} attempts` : ""}.
      </p>
      {attempts.map(({ record: r, key }) => {
        const o = outcome(Boolean(r.correct), Boolean(r.runs));
        const note = firstIssue(r);
        return (
          <details className="attempt" key={key}>
            <summary>
              <span className="mono">{String(r.problem ?? "–")}</span>
              <span>attempt {fmtInt(r.attempt)}</span>
              <span className={`badge ${sourceClass(r.source)}`}>{String(r.source ?? "–")}</span>
              <span className={`badge ${o.cls}`}>{o.text}</span>
              <span className="muted">
                score {fmtNum(r.score, 1)} · judged by {String(r.judged_by ?? "–")} · {String(r.phase ?? "–")} phase,
                round {fmtInt(r.round)} · {fmtNum(r.seconds, 2)} s
              </span>
              {note ? <span className="muted issue">{note}</span> : null}
            </summary>
            <h4>Code</h4>
            <pre className="code">{String(r.code ?? "") || "(empty)"}</pre>
            <h4>stdout</h4>
            <pre className="output">{String(r.stdout ?? "") || "(empty)"}</pre>
            <IssueList issues={r.issues} title="Issues" />
          </details>
        );
      })}
    </>
  );
}

/** Solve one problem without training (POST /api/codegen/solve); the attempts go to SolveResultCard. */
function SolveCard({ shared, onResult }) {
  const [prompt, setPrompt] = useState("");
  const [expected, setExpected] = useState("");
  const [tests, setTests] = useState("");
  const [source, setSource] = useState("model");
  const [attempts, setAttempts] = useState("1");
  const [judge, setJudge] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (prompt.trim() === "") {
      setError("Enter a problem to solve.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.codegenSolve({
        problem: problemPayload(prompt, tests, expected),
        source,
        attempts: Math.max(1, parseInteger(attempts, 1)),
        judge,
        ...shared,
      });
      onResult(isObject(data) ? data : {});
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const provider = shared.teacher_provider === "chatgpt" ? "chatgpt" : "ollama";
  const usesTutor = source === "teacher" || judge;

  return (
    <form className="card" onSubmit={handleSubmit}>
      <h2>Try a problem</h2>
      <p className="muted">
        Solve one problem with the model or the teacher without training on the result. The teacher, its model and
        URL, the strictness and the sandbox settings come from the card above. {providerNote(provider)}
      </p>
      <TextArea
        label="Problem"
        value={prompt}
        onChange={setPrompt}
        rows={4}
        disabled={loading}
        placeholder="Print the numbers from 1 to 10, one per line."
      />
      <TextArea
        label="Expected output"
        hint="optional; compared with stdout, surrounding whitespace ignored"
        value={expected}
        onChange={setExpected}
        rows={3}
        disabled={loading}
        placeholder={"1\n2\n3"}
      />
      <TextArea
        label="Tests"
        hint="optional Python appended to the program"
        value={tests}
        onChange={setTests}
        rows={3}
        disabled={loading}
        placeholder="assert add(1, 2) == 3"
      />
      <div className="row">
        <SelectField
          label="Source"
          value={source}
          onChange={setSource}
          disabled={loading}
          options={[
            ["model", "model (RadixCyclicNN)"],
            ["teacher", `teacher (${providerLabel(provider)})`],
          ]}
        />
        <NumberField
          label="Attempts"
          value={attempts}
          onChange={setAttempts}
          min={1}
          step={1}
          disabled={loading}
        />
      </div>
      <CheckField label={`Judge with ${providerLabel(provider)}`} checked={judge} onChange={setJudge} disabled={loading} />
      <div className="actions">
        <button type="submit" className="primary" disabled={loading}>
          {loading ? "Solving…" : "Solve"}
        </button>
        {loading ? (
          <span className="muted note">{usesTutor ? slowNote(provider) : "Generating and running in the sandbox…"}</span>
        ) : null}
      </div>
      <Alert message={error} onDismiss={() => setError(null)} />
    </form>
  );
}

/** Attempts of the last "Try a problem" request, one block each. */
function SolveResultCard({ result }) {
  const attempts = result ? asArray(result.attempts).filter(isObject) : [];
  const problem = result && isObject(result.problem) ? result.problem : {};
  const sandbox = result && isObject(result.sandbox) ? result.sandbox : {};
  return (
    <div className="card">
      <h2>Solutions</h2>
      {!result ? (
        <p className="muted">
          Enter a problem and press Solve. Each attempt shows its code, the sandbox run, the style report and the
          verdict.
        </p>
      ) : (
        <>
          <div className="chips">
            <span className="stat">
              source <b>{String(result.source ?? "–")}</b>
            </span>
            <span className="stat">
              result <b>{result.correct ? "correct" : "not solved"}</b>
            </span>
            <span className="stat">
              attempts <b>{fmtInt(attempts.length)}</b>
            </span>
            <span className="stat">
              network isolated <b>{tri(sandbox.network_isolated)}</b>
            </span>
          </div>
          {problem.prompt ? (
            <p className="muted">
              {String(problem.id ?? "problem")}: {String(problem.prompt)}
              {problem.tests ? " · with tests" : ""}
              {problem.expected_output ? " · with expected output" : ""}
            </p>
          ) : null}
          {attempts.length === 0 ? (
            <p className="muted">No attempts were returned.</p>
          ) : (
            attempts.map((attempt, i) => <AttemptBlock key={i} attempt={attempt} />)
          )}
        </>
      )}
    </div>
  );
}

/** Run a pasted program in the sandbox (POST /api/codegen/run); the report goes to RunResultCard. */
function RunCard({ shared, onResult }) {
  const [code, setCode] = useState("");
  const [expected, setExpected] = useState("");
  const [tests, setTests] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (code.trim() === "") {
      setError("Enter a program to run.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.codegenRun({
        code,
        ...(tests.trim() !== "" ? { tests } : {}),
        ...(expected.trim() !== "" ? { expected_output: expected } : {}),
        sandbox_timeout: shared.sandbox_timeout,
        memory_mb: shared.memory_mb,
        network_isolation: shared.network_isolation,
        strictness: shared.strictness,
      });
      onResult(isObject(data) ? data : {});
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="card" onSubmit={handleSubmit}>
      <h2>Run code in the sandbox</h2>
      <p className="muted">
        Runs a Python program with the sandbox settings above (timeout, memory, network isolation), checks its style
        and reports the verdict. Neither the model nor the tutor is involved.
      </p>
      <TextArea
        label="Code"
        hint="Python"
        value={code}
        onChange={setCode}
        rows={10}
        disabled={loading}
        placeholder={'def main():\n    print("hello")\n\n\nif __name__ == "__main__":\n    main()'}
      />
      <TextArea
        label="Expected output"
        hint="optional"
        value={expected}
        onChange={setExpected}
        rows={3}
        disabled={loading}
        placeholder="hello"
      />
      <TextArea
        label="Tests"
        hint="optional Python appended to the program"
        value={tests}
        onChange={setTests}
        rows={3}
        disabled={loading}
        placeholder="assert main() is None"
      />
      <div className="actions">
        <button type="submit" className="primary" disabled={loading}>
          {loading ? "Running…" : "Run"}
        </button>
        {loading ? <span className="muted note">Running in the sandbox…</span> : null}
      </div>
      <Alert message={error} onDismiss={() => setError(null)} />
    </form>
  );
}

/** Run, style report and verdict of the last sandbox request. */
function RunResultCard({ result }) {
  return (
    <div className="card">
      <h2>Sandbox result</h2>
      {!result ? (
        <p className="muted">Paste a program and press Run.</p>
      ) : (
        <>
          <RunDetails run={result.run} />
          <StyleDetails style={result.style} />
          <VerdictDetails verdict={result.verdict} />
        </>
      )}
    </div>
  );
}

/**
 * Code tab: a codegen job (teacher / model phases over problems, sandbox
 * runs, an optional LLM judge, 2NRL rewards) with its live records, a
 * single-problem solver and a sandbox runner. The tutor (a local Ollama model
 * or ChatGPT), the model and the sandbox settings of the job card are shared
 * by the solver and the runner.
 */
export default function CodeGenPanel({ status }) {
  const [problems, setProblems] = useState("");
  const [files, setFiles] = useState([]);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [formError, setFormError] = useState(null);
  const [serverHistory, setServerHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [historyFor, setHistoryFor] = useState(null);
  const [solve, setSolve] = useState(null);
  const [run, setRun] = useState(null);
  const { job, running, busy, error, start, stop, clearError } = useJob("codegen");

  const otherJobRunning = jobIsRunning(status) && !running;
  const s = settings;
  const field = (name) => (value) => setSettings((prev) => ({ ...prev, [name]: value }));
  const provider = s.teacherProvider;
  const defaults = providerDefaults(status, provider);
  // Switching the tutor also switches the model: model names do not carry from one provider to the other.
  const setProvider = (value) =>
    setSettings((prev) => ({ ...prev, teacherProvider: value, teacherModel: defaultModel(status, value) }));

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.codegenHistory();
      setServerHistory(asArray(data && data.history));
      setHistoryError(null);
      return true;
    } catch (err) {
      setHistoryError(err.message);
      return false;
    }
  }, []);

  // Load the stored history on mount and again once a job reaches a terminal state; `historyFor`
  // remembers which finished job the stored history is known to include.
  const jobId = job && job.id !== undefined ? job.id : null;
  const jobState = job ? job.state : null;
  useEffect(() => {
    if (jobState === "running") return undefined;
    let alive = true;
    loadHistory().then((ok) => {
      if (alive && ok) setHistoryFor(jobId);
    });
    return () => {
      alive = false;
    };
  }, [jobId, jobState, loadHistory]);

  async function handleStart(event) {
    event.preventDefault();
    const lines = problemLines(problems);
    if (lines.length === 0 && files.length === 0) {
      setFormError("Enter at least one problem (one per line) or select uploaded problem files.");
      return;
    }
    setFormError(null);
    await start(() =>
      api.codegenStart({
        ...(lines.length > 0 ? { problems_text: problems } : {}),
        ...(files.length > 0 ? { problem_files: files } : {}),
        phases: s.phases,
        rounds: Math.max(1, parseInteger(s.rounds, 1)),
        teacher_attempts: Math.max(1, parseInteger(s.teacherAttempts, 3)),
        model_attempts: Math.max(1, parseInteger(s.modelAttempts, 4)),
        judge: s.judge,
        fallback_teacher: s.fallbackTeacher,
        replay: s.replay,
        twonrl_per: s.twonrlPer,
        neg_epochs: Math.max(0, parseInteger(s.negEpochs, 2)),
        pos_epochs: Math.max(0, parseInteger(s.posEpochs, 3)),
        neg_lr: Math.max(0, parseNumber(s.negLr, 0.5)),
        pos_lr: Math.max(0, parseNumber(s.posLr, 0.1)),
        batch_size: Math.max(1, parseInteger(s.batchSize, 4)),
        ...sharedFields(s),
      }),
    );
  }

  // Live records while the job runs; afterwards the stored history (every run on this server),
  // once it has been reloaded for the finished job.
  const jobHistory = asArray(job && job.history);
  const showingJob = running || (jobId !== null && historyFor !== jobId);
  const history = showingJob ? jobHistory : serverHistory;
  const indexed = history.map((record, key) => ({ record, key })).filter(({ record }) => isObject(record));
  const rounds = indexed.filter(({ record }) => record.kind === "round").map(({ record }) => record);
  const problemRows = indexed.filter(({ record }) => record.kind === "problem");
  const attemptRows = indexed.filter(({ record }) => record.kind === "attempt");
  const shared = sharedFields(s);

  return (
    <>
      <form className="card wide" onSubmit={handleStart}>
        <h2>Problems and training</h2>
        <p className="muted">
          Each problem is solved by the teacher ({providerLabel(provider)}) and by the RadixCyclicNN model. Every
          program runs in a sandbox, is style-checked and optionally judged by {providerLabel(provider)}; correct
          programs reward the model and rejected ones punish it through 2NRL.
        </p>
        {provider === "chatgpt" && !defaults.configured ? (
          <p className="muted issue">
            This server has no OPENAI_API_KEY, so ChatGPT cannot tutor yet: set it (or OPENAI_API_KEY_FILE) in the
            server's environment and restart it.
          </p>
        ) : null}
        <TextArea
          label="Problems"
          hint="one per line; # starts a comment"
          value={problems}
          onChange={setProblems}
          rows={6}
          disabled={running}
          placeholder={"Print the numbers from 1 to 10, one per line.\nRead two integers from stdin and print their sum."}
        />
        <UploadPicker
          selected={files}
          onChange={setFiles}
          disabled={running}
          title="Problem files"
          hint=".txt: one problem per line; .json / .jsonl: objects with prompt, tests and expected_output. Selected files are solved in addition to the problems above."
        />
        <div className="row auto">
          <SelectField
            label="Phases"
            value={s.phases}
            onChange={field("phases")}
            disabled={running}
            options={[
              ["both", "both (teacher, then model)"],
              ["teacher", "teacher only"],
              ["model", "model only"],
            ]}
          />
          <NumberField label="Rounds" value={s.rounds} onChange={field("rounds")} min={1} step={1} disabled={running} />
          <SelectField
            label="Teacher"
            hint="who tutors"
            value={provider}
            onChange={setProvider}
            disabled={running}
            options={[
              ["ollama", "Ollama (local)"],
              ["chatgpt", "ChatGPT (OpenAI)"],
            ]}
          />
          <TextField
            label="Teacher model"
            hint={providerLabel(provider)}
            value={s.teacherModel}
            onChange={field("teacherModel")}
            placeholder={defaultModel(status, provider)}
            disabled={running}
          />
          <TextField
            label={`${providerLabel(provider)} URL`}
            hint="blank = server default"
            value={s.url}
            onChange={field("url")}
            placeholder={defaults.url || (provider === "chatgpt" ? "https://api.openai.com/v1" : "http://127.0.0.1:11434")}
            disabled={running}
          />
        </div>
        <div className="row auto">
          <NumberField
            label="Teacher attempts"
            value={s.teacherAttempts}
            onChange={field("teacherAttempts")}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Model attempts"
            value={s.modelAttempts}
            onChange={field("modelAttempts")}
            min={1}
            step={1}
            disabled={running}
          />
          <SelectField
            label="Strictness"
            value={s.strictness}
            onChange={field("strictness")}
            disabled={running}
            options={[
              ["strict", "strict (style must pass too)"],
              ["lenient", "lenient (must run and solve the task)"],
            ]}
          />
          <NumberField
            label="Temperature"
            hint="model sampling"
            value={s.temperature}
            onChange={field("temperature")}
            min={0}
            disabled={running}
          />
        </div>
        <div className="row auto">
          <NumberField
            label="Max length"
            hint="chars generated by the model"
            value={s.maxLength}
            onChange={field("maxLength")}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Sandbox timeout"
            hint="seconds"
            value={s.sandboxTimeout}
            onChange={field("sandboxTimeout")}
            min={0.1}
            disabled={running}
          />
          <NumberField
            label="Memory MB"
            hint="0 = unlimited"
            value={s.memoryMb}
            onChange={field("memoryMb")}
            min={0}
            step={1}
            disabled={running}
          />
          <SelectField
            label="2NRL per"
            value={s.twonrlPer}
            onChange={field("twonrlPer")}
            disabled={running}
            options={[
              ["problem", "problem (after every problem)"],
              ["round", "round (after each phase of a round)"],
            ]}
          />
        </div>
        <div className="checks">
          <CheckField
            label={`Judge with ${providerLabel(provider)}`}
            checked={s.judge}
            onChange={field("judge")}
            disabled={running}
          />
          <CheckField
            label="Fall back to the teacher when the model fails"
            checked={s.fallbackTeacher}
            onChange={field("fallbackTeacher")}
            disabled={running}
          />
          <CheckField
            label="Replay earlier solutions"
            checked={s.replay}
            onChange={field("replay")}
            disabled={running}
          />
          <CheckField
            label="Network isolation in the sandbox"
            checked={s.networkIsolation}
            onChange={field("networkIsolation")}
            disabled={running}
          />
        </div>
        <h3>Advanced 2NRL</h3>
        <div className="row auto">
          <NumberField
            label="Negative epochs"
            value={s.negEpochs}
            onChange={field("negEpochs")}
            min={0}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Positive epochs"
            value={s.posEpochs}
            onChange={field("posEpochs")}
            min={0}
            step={1}
            disabled={running}
          />
          <NumberField label="Negative lr" value={s.negLr} onChange={field("negLr")} min={0} disabled={running} />
          <NumberField label="Positive lr" value={s.posLr} onChange={field("posLr")} min={0} disabled={running} />
          <NumberField
            label="Batch size"
            value={s.batchSize}
            onChange={field("batchSize")}
            min={1}
            step={1}
            disabled={running}
          />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={running || busy || otherJobRunning}>
            {busy ? "Starting…" : "Start"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
          <span className="muted note">{providerNote(provider)}</span>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <Alert message={formError} onDismiss={() => setFormError(null)} />
        <Alert message={error} onDismiss={clearError} />
        <JobStatus job={job} emptyText="No codegen job in this session. Enter problems and press Start." />
      </form>

      <div className="card wide">
        <div className="toolbar">
          <h2>Results</h2>
          <button type="button" className="small" onClick={loadHistory}>
            Reload history
          </button>
        </div>
        <Alert message={historyError} onDismiss={() => setHistoryError(null)} />
        {!showingJob && history.length > 0 ? (
          <p className="muted">Showing the stored history of every codegen run on this server ({fmtInt(history.length)} records).</p>
        ) : null}
        {!running && history.length === 0 ? (
          <p className="muted">No codegen records yet. Enter problems above and press Start.</p>
        ) : null}
        <h3>Rounds</h3>
        <RoundSummary rounds={rounds} running={running} />
        <h3>Problems</h3>
        <ProblemTable rows={problemRows.slice(-MAX_PROBLEM_ROWS)} total={problemRows.length} running={running} />
        <h3>Attempts</h3>
        <AttemptFeed
          attempts={attemptRows.slice(-MAX_ATTEMPTS).reverse()}
          total={attemptRows.length}
          running={running}
        />
      </div>

      <SolveCard shared={shared} onResult={setSolve} />
      <SolveResultCard result={solve} />
      <RunCard shared={shared} onResult={setRun} />
      <RunResultCard result={run} />
    </>
  );
}
