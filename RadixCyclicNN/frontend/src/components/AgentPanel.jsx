import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines } from "../util.js";
import UploadPicker from "./UploadPicker.jsx";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import { CheckField, NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";

const MAX_ROWS = 200;

/** One record of the live log, rendered as a line whose shape depends on its kind. */
function LogLine({ record }) {
  const kind = record.kind;
  if (kind === "criteria") {
    return (
      <li>
        <b>{record.task}</b> — acceptance criteria for “{record.prompt}”:
        <ol className="criteria">
          {asArray(record.criteria).map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ol>
      </li>
    );
  }
  if (kind === "proposal") {
    return (
      <li>
        <b>{record.task}</b> chose: <i>{record.prompt}</i>{" "}
        <span className="muted">
          (frontier {fmtInt(record.frontier)}, read {fmtInt(record.visited)}) — from its own text:{" "}
          <code>{record.emission || "(nothing)"}</code>
        </span>
      </li>
    );
  }
  if (kind === "step") {
    return (
      <li>
        <span className={record.source === "model" ? "tag own" : "tag"}>{record.source}</span>{" "}
        <code>{record.tool}</code> {record.ok ? "→" : "failed:"}{" "}
        <span className="muted">{record.error || record.output}</span>
      </li>
    );
  }
  if (kind === "attempt") {
    return (
      <li>
        <b>{record.task}</b> attempt {fmtInt(record.attempt)} [{record.source}]{" "}
        <span className={record.correct ? "verdict pass" : "verdict fail"}>
          {record.correct ? "correct" : "failed"}
        </span>
        {record.score === null || record.score === undefined ? null : <> score {fmtNum(record.score, 1)}</>}
        {record.autonomy === null || record.autonomy === undefined ? null : (
          <> · own calls {Math.round(record.autonomy * 100)}%</>
        )}
        : <i>{record.answer || "(no answer)"}</i>
        {record.critique ? <div className="muted">judge: {record.critique}</div> : null}
      </li>
    );
  }
  return null;
}

/**
 * Tool use: the network browses and solves with tools while Ollama writes the
 * acceptance criteria, mediates the calls it cannot write yet, judges the
 * answer and demonstrates the task when it failed. Two modes: a list of tasks,
 * or exploration, where the network chooses every task itself.
 */
export default function AgentPanel({ status }) {
  const [mode, setMode] = useState("tasks");
  const [tasks, setTasks] = useState("");
  const [taskFiles, setTaskFiles] = useState([]);
  const [steps, setSteps] = useState("10");
  const [seedUrls, setSeedUrls] = useState("");
  const [phase, setPhase] = useState("model");
  const [rounds, setRounds] = useState("1");
  const [maxSteps, setMaxSteps] = useState("6");
  const [modelAttempts, setModelAttempts] = useState("2");
  const [mediation, setMediation] = useState("repair");
  const [criteria, setCriteria] = useState("4");
  const [temperature, setTemperature] = useState("1.0");
  const [maxLength, setMaxLength] = useState("200");
  const [judge, setJudge] = useState(true);
  const [teach, setTeach] = useState(true);
  const [strict, setStrict] = useState(true);
  const [readReward, setReadReward] = useState(false);
  const [blatantMode, setBlatantMode] = useState("fail_invert");
  const [blatantMargin, setBlatantMargin] = useState("0.5");
  const [blatantBoost, setBlatantBoost] = useState("4");
  const [negEpochs, setNegEpochs] = useState("2");
  const [posEpochs, setPosEpochs] = useState("3");
  const [negLr, setNegLr] = useState("0.5");
  const [posLr, setPosLr] = useState("0.1");
  const [agentModel, setAgentModel] = useState("");
  const [formError, setFormError] = useState(null);
  const [tools, setTools] = useState(null);
  const [toolsError, setToolsError] = useState(null);
  const [serverHistory, setServerHistory] = useState([]);
  const explore = mode === "explore";
  const { job, running, busy, error, start, stop, clearError } = useJob(explore ? "explore" : "agent");

  const otherJobRunning = jobIsRunning(status) && !running;

  const loadTools = useCallback(async () => {
    try {
      setTools(await api.tools());
      setToolsError(null);
    } catch (err) {
      setToolsError(err.message);
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.agentHistory();
      setServerHistory(asArray(data && data.history));
    } catch (err) {
      setToolsError(err.message);
    }
  }, []);

  useEffect(() => {
    loadTools();
    loadHistory();
  }, [loadTools, loadHistory]);

  const jobState = job ? job.state : null;
  useEffect(() => {
    if (jobState && jobState !== "running") loadHistory();
  }, [jobState, loadHistory]);

  function options() {
    return {
      max_steps: parseInteger(maxSteps, 6),
      model_attempts: parseInteger(modelAttempts, 2),
      mediation,
      criteria: parseInteger(criteria, 4),
      temperature: parseNumber(temperature, 1),
      max_length: parseInteger(maxLength, 200),
      judge,
      teach,
      strict,
      read_reward: readReward,
      blatant_mode: blatantMode,
      blatant_margin: parseNumber(blatantMargin, 0.5),
      blatant_boost: parseNumber(blatantBoost, 4),
      neg_epochs: parseInteger(negEpochs, 2),
      pos_epochs: parseInteger(posEpochs, 3),
      neg_lr: parseNumber(negLr, 0.5),
      pos_lr: parseNumber(posLr, 0.1),
      ...(agentModel.trim() ? { agent_model: agentModel.trim() } : {}),
    };
  }

  async function handleStart(event) {
    event.preventDefault();
    if (explore) {
      setFormError(null);
      const seeds = splitLines(seedUrls);
      await start(() =>
        api.agentExplore({ steps: parseInteger(steps, 10), ...(seeds.length ? { seed_urls: seeds } : {}), ...options() }),
      );
      return;
    }
    const lines = splitLines(tasks);
    if (lines.length === 0 && taskFiles.length === 0) {
      setFormError("Enter one task per line, or select uploaded task files.");
      return;
    }
    setFormError(null);
    await start(() =>
      api.agentStart({
        ...(lines.length > 0 ? { tasks: lines } : {}),
        ...(taskFiles.length > 0 ? { task_files: taskFiles } : {}),
        phase,
        rounds: parseInteger(rounds, 1),
        ...options(),
      }),
    );
  }

  const jobHistory = asArray(job && job.history);
  const history = running || jobHistory.length > 0 ? jobHistory : serverHistory;
  const done = history.filter((r) => r.kind === "task" || r.kind === "explore");
  const rows = done.slice(-MAX_ROWS);
  const log = history.filter((r) => r.kind !== "task" && r.kind !== "explore" && r.kind !== "round").slice(-40);
  const calls = done.reduce((sum, r) => sum + (r.calls || 0), 0);
  const own = done.reduce((sum, r) => sum + (r.own_calls || 0), 0);
  const solved = done.filter((r) => r.correct).length;

  return (
    <>
      <form className="card" onSubmit={handleStart}>
        <h2>Agent</h2>
        <p className="muted">
          The network uses tools by writing them as text — <code>{'<tool>web_fetch {"url": "…"}</tool>'}</code> — and
          reads the answer back as <code>{"<result>…</result>"}</code>. Ollama never solves the task for it unless it
          has to: it writes the acceptance criteria <em>before</em> anything is attempted, repairs the calls the
          network cannot write yet, judges the answer against those criteria, and only then demonstrates the task
          with the same real tools. Learning focuses on failure: every failed transcript is trained on the harder the
          worse it was, then the network is inverted and fine-tuned on what was right.
        </p>
        {tools ? (
          <p className="muted">
            Tools: {asArray(tools.names).map((n) => <code key={n}>{n} </code>)}
            {tools.options && tools.options.offline ? " (offline: no browsing)" : null}
          </p>
        ) : null}
        <Alert message={toolsError} onDismiss={() => setToolsError(null)} />
        <SelectField
          label="Mode"
          value={mode}
          onChange={setMode}
          disabled={running}
          options={[
            ["tasks", "tasks — solve a list of questions"],
            ["explore", "explore — the network chooses every task itself"],
          ]}
        />
        {explore ? (
          <>
            <div className="row">
              <NumberField
                label="Steps"
                hint="0 = until stopped"
                value={steps}
                onChange={setSteps}
                min={0}
                step={1}
                disabled={running}
              />
            </div>
            <TextArea
              label="Seed pages"
              hint="one URL per line; the frontier it starts from"
              value={seedUrls}
              onChange={setSeedUrls}
              rows={3}
              disabled={running}
              placeholder="https://en.wikipedia.org/wiki/Cat"
            />
          </>
        ) : (
          <>
            <TextArea
              label="Tasks"
              hint="one question per line"
              value={tasks}
              onChange={setTasks}
              rows={6}
              disabled={running}
              placeholder={"How many legs does a cat have?\nWhat is the boiling point of water in Fahrenheit?"}
            />
            <UploadPicker
              selected={taskFiles}
              onChange={setTaskFiles}
              disabled={running}
              title="Task files"
              hint="Uploaded .txt / .json / .jsonl task files (in addition to the tasks above)."
            />
            <div className="row">
              <SelectField
                label="Phase"
                value={phase}
                onChange={setPhase}
                disabled={running}
                options={[
                  ["model", "model — the network attempts first"],
                  ["teacher", "teacher — only Ollama demonstrates"],
                  ["both", "both — teacher, then the network"],
                ]}
              />
              <NumberField label="Rounds" value={rounds} onChange={setRounds} min={1} step={1} disabled={running} />
            </div>
          </>
        )}
        <div className="row">
          <NumberField
            label="Tool calls per attempt"
            value={maxSteps}
            onChange={setMaxSteps}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Attempts by the network"
            value={modelAttempts}
            onChange={setModelAttempts}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Criteria per task"
            value={criteria}
            onChange={setCriteria}
            min={1}
            max={8}
            step={1}
            disabled={running}
          />
        </div>
        <div className="row">
          <SelectField
            label="Mediation"
            hint="how much Ollama fixes"
            value={mediation}
            onChange={setMediation}
            disabled={running}
            options={[
              ["repair", "repair — only the calls the network cannot write"],
              ["always", "always — every call is Ollama's"],
              ["never", "never — broken calls run and are learned from as failures"],
            ]}
          />
          <TextField
            label="Ollama model"
            hint="blank = the server default"
            value={agentModel}
            onChange={setAgentModel}
            disabled={running}
            placeholder="llama3.2"
          />
        </div>
        <div className="row">
          <NumberField
            label="Temperature"
            value={temperature}
            onChange={setTemperature}
            min={0}
            disabled={running}
          />
          <NumberField
            label="Characters per step"
            value={maxLength}
            onChange={setMaxLength}
            min={1}
            step={1}
            disabled={running}
          />
        </div>
        <div className="row">
          <CheckField label="LLM judge" checked={judge} onChange={setJudge} disabled={running} />
          <CheckField label="Teach on failure" checked={teach} onChange={setTeach} disabled={running} />
          <CheckField label="Every criterion must be met" checked={strict} onChange={setStrict} disabled={running} />
          <CheckField
            label="Also learn the pages it reads"
            checked={readReward}
            onChange={setReadReward}
            disabled={running}
          />
        </div>
        <fieldset className="schedule">
          <legend>Failures: train on them, then invert</legend>
          <div className="row">
            <SelectField
              label="Failure handling"
              value={blatantMode}
              onChange={setBlatantMode}
              disabled={running}
              options={[
                ["fail_invert", "train on every failure, the worse the more, then invert"],
                ["activation", "local: invert the activation function (a) along failed transcripts"],
                ["state", "local: invert the trained node value (z) along failed transcripts"],
                ["none", "off: plain unweighted 2NRL"],
              ]}
            />
            <NumberField
              label="Margin"
              hint="gap 0–1 at which a failure is blatant"
              value={blatantMargin}
              onChange={setBlatantMargin}
              min={0.001}
              max={1}
              disabled={running || blatantMode === "none"}
            />
            <NumberField
              label="Max boost"
              hint="largest learning-rate multiplier"
              value={blatantBoost}
              onChange={setBlatantBoost}
              min={1}
              disabled={running || blatantMode !== "fail_invert"}
            />
          </div>
          <p className="muted">
            How badly an attempt failed is its <b>gap</b> (0–1): the share of the acceptance criteria it missed, or
            how far below the pass score the judge put it — an attempt that answered nothing has a gap of 1.{" "}
            <b>Train then invert</b>: every failed transcript is trained on with its learning rates multiplied by 1 +
            gap / margin (capped at the max boost), so the network reproduces its worst attempts on purpose; then it
            is inverted, turning what it now does confidently into what it confidently avoids, and fine-tuned on the
            transcripts that were judged correct. <b>Local</b>: no negative pass; every other node along a failed
            transcript is moved toward its negation instead, and blatant failures leave the 2NRL garbage set.
          </p>
        </fieldset>
        <div className="row">
          <NumberField label="Negative epochs" value={negEpochs} onChange={setNegEpochs} min={0} step={1} disabled={running} />
          <NumberField label="Negative lr" value={negLr} onChange={setNegLr} min={0} disabled={running} />
          <NumberField label="Positive epochs" value={posEpochs} onChange={setPosEpochs} min={0} step={1} disabled={running} />
          <NumberField label="Positive lr" value={posLr} onChange={setPosLr} min={0} disabled={running} />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={running || busy || otherJobRunning}>
            {busy ? "Starting…" : explore ? "Start exploring" : "Start solving"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <Alert message={formError} onDismiss={() => setFormError(null)} />
        <Alert message={error} onDismiss={clearError} />
      </form>

      <div className="card">
        <div className="toolbar">
          <h2>What it did</h2>
          <button type="button" className="small" onClick={loadHistory}>
            Reload history
          </button>
        </div>
        <JobStatus job={job} emptyText="No agent job in this session." />
        {done.length > 0 ? (
          <p className="muted">
            {solved}/{done.length} task(s) solved; {fmtInt(own)}/{fmtInt(calls)} tool call(s) written by the network
            itself{calls ? ` (${Math.round((own / calls) * 100)}%)` : ""}.
          </p>
        ) : null}
        {log.length > 0 ? (
          <ul className="log">
            {log.map((record, i) => (
              <LogLine key={i} record={record} />
            ))}
          </ul>
        ) : null}
        {rows.length > 0 ? (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>task</th>
                  <th>prompt</th>
                  <th>result</th>
                  <th>by</th>
                  <th>calls</th>
                  <th>own</th>
                  <th>failures</th>
                  <th>blatant</th>
                  <th>boost</th>
                  <th>2NRL</th>
                  <th>seconds</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td>{r.task}</td>
                    <td title={r.answer || undefined}>{r.prompt}</td>
                    <td className={r.correct ? "verdict pass" : "verdict fail"}>{r.correct ? "correct" : "failed"}</td>
                    <td>{r.solved_by || "–"}</td>
                    <td>{fmtInt(r.calls)}</td>
                    <td title={r.autonomy === null ? undefined : "written by the network itself"}>
                      {fmtInt(r.own_calls)}
                    </td>
                    <td>{fmtInt(r.failures)}</td>
                    <td>{fmtInt(r.blatant)}</td>
                    <td title={r.flipped ? `${fmtInt(r.flipped)} nodes changed` : undefined}>{fmtNum(r.boost_max, 2)}</td>
                    <td>{r.action || "–"}</td>
                    <td>{fmtNum(r.seconds, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </>
  );
}
