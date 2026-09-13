import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import LineChart from "./LineChart.jsx";
import { CheckField, NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";

const MAX_ROWS = 200;
const DEFAULT_PLAN_LESSONS = 3;
const PROVIDER_LABELS = { ollama: "Ollama", chatgpt: "ChatGPT" };

/** "Ollama" / "ChatGPT" for a provider name. */
function providerLabel(provider) {
  return PROVIDER_LABELS[provider] || PROVIDER_LABELS.ollama;
}

const slowNote = (provider) => `${providerLabel(provider)} is writing and marking the exercises; this can take a minute or two.`;
const planNote = (provider) => `${providerLabel(provider)} is reading the report card and planning the next lessons; this can take a minute.`;

/**
 * Server-side defaults of one teacher, reported by /api/status as
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

/** {"tutor_provider", "url", "tutor_model"} overrides: only fields that differ from the server defaults are sent. */
function overridesOf(provider, url, model, defaults) {
  const out = { tutor_provider: provider };
  const value = String(url ?? "").trim();
  if (value && value !== defaults.url) out.url = value;
  const name = String(model ?? "").trim();
  if (name) out.tutor_model = name;
  return out;
}

function markOf(record) {
  if (record.passed) return "pass";
  return record.score === null || record.score === undefined ? "unrated" : "fail";
}

/** One lesson of a dry run (/api/tutor/lesson) rendered like a history record. */
function lessonRecord(lesson, index) {
  const exercise = (lesson && lesson.exercise) || {};
  const grade = (lesson && lesson.grade) || {};
  return {
    kind: "lesson",
    round: "–",
    exercise: exercise.id || `e${index + 1}`,
    prefix: exercise.prefix || "",
    focus: exercise.focus || "",
    attempt: (lesson.attempt ?? 0) + 1,
    sentence: lesson.sentence || "",
    continuation: lesson.continuation || "",
    score: grade.score ?? null,
    grammar: grade.grammar ?? null,
    spelling: grade.spelling ?? null,
    fluency: grade.fluency ?? null,
    passed: Boolean(grade.passed),
    error: grade.error || "none",
    correction: grade.correction || "",
    changes: asArray(lesson && lesson.changes),
    comment: grade.comment || "",
    graded_by: grade.graded_by || "ollama",
  };
}

/** What the teacher changed in one sentence: the struck-out text against what replaced it. */
function Changes({ changes }) {
  const items = asArray(changes).filter((c) => c && typeof c === "object");
  if (items.length === 0) return <span className="muted">–</span>;
  return (
    <span className="changes">
      {items.map((change, i) => (
        <span key={i} className={`change ${String(change.op || "")}`} title={String(change.op || "")}>
          {change.wrong ? <del>{String(change.wrong)}</del> : null}
          {change.wrong && change.right ? " " : null}
          {change.right ? <ins>{String(change.right)}</ins> : null}
        </span>
      ))}
    </span>
  );
}

/** Marks, pass rate and the mistakes of a run, as pills and a small histogram. */
function ReportCard({ card, title }) {
  if (!card || !card.lessons) return null;
  const errors = card.errors && typeof card.errors === "object" ? Object.entries(card.errors) : [];
  const worst = errors.length ? Math.max(...errors.map(([, count]) => count)) : 0;
  return (
    <>
      <h3>{title}</h3>
      <div className="chips">
        <span className="stat">
          lessons <b>{fmtInt(card.lessons)}</b>
        </span>
        <span className="stat">
          passed{" "}
          <b>
            {fmtInt(card.passed)}
            {card.pass_rate === null || card.pass_rate === undefined ? "" : ` (${Math.round(card.pass_rate * 100)}%)`}
          </b>
        </span>
        <span className="stat">
          mean score <b>{fmtNum(card.mean_score, 2)}</b>
        </span>
        <span className="stat">
          grammar <b>{fmtNum(card.mean_grammar, 2)}</b>
        </span>
        <span className="stat">
          spelling <b>{fmtNum(card.mean_spelling, 2)}</b>
        </span>
        <span className="stat">
          fluency <b>{fmtNum(card.mean_fluency, 2)}</b>
        </span>
      </div>
      {errors.length > 0 ? (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>mistake</th>
                <th>times</th>
                <th>share of the lessons</th>
              </tr>
            </thead>
            <tbody>
              {errors.map(([name, count]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{fmtInt(count)}</td>
                  <td>
                    <span className="rating">
                      <span className="rating-bar fail" aria-hidden="true">
                        <i style={{ width: `${worst ? Math.round((count / worst) * 100) : 0}%` }} />
                      </span>
                      {Math.round((count / card.lessons) * 100)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">No mistakes were named.</p>
      )}
    </>
  );
}

/** The lessons the teacher plans from a report card: what each one drills, at which weakness, and why. */
function LessonPlanCard({ plan, onUse, onTeach, onClear, disabled, cardRef }) {
  const lessons = asArray(plan && plan.lessons);
  if (lessons.length === 0) return null;
  const weak = asArray(plan.weak);
  const upgrade = plan.upgrade && typeof plan.upgrade === "object" ? plan.upgrade : {};
  const by = plan.source === "report card" ? "the report card alone" : providerLabel(plan.source);
  return (
    // wide: the plan is a table of lessons, not a sidebar note
    <div className="card wide" ref={cardRef}>
      <div className="toolbar">
        <h2>Lesson plan</h2>
        <button type="button" className="small" onClick={onClear}>
          Clear
        </button>
      </div>
      <p className="muted">
        The next lessons, written from the report card by <b>{by}</b>, for a <b>{String(plan.level || "beginner")}</b>{" "}
        student: teach the whole batch to the brief below, or load a single lesson from the table.
      </p>
      {plan.summary ? <p>{String(plan.summary)}</p> : null}
      {upgrade.note ? (
        <p className="muted">
          <b>{String(upgrade.step || "hold")}</b> — {String(upgrade.note)}
        </p>
      ) : null}
      {plan.prompt ? (
        <>
          <h3>The brief for the next batch</h3>
          <blockquote className="brief">{String(plan.prompt)}</blockquote>
          <div className="actions">
            <button type="button" className="primary" disabled={disabled} onClick={() => onTeach(plan)}>
              Teach the next batch
            </button>
            <span className="muted">
              Loads the brief and the step up into the settings above — the exercise writer is given it with every
              round.
            </span>
          </div>
        </>
      ) : null}
      {weak.length > 0 ? (
        <div className="chips">
          {weak.map((point) => (
            <span className="stat" key={String(point.error)}>
              {String(point.error)}{" "}
              <b>
                {fmtInt(point.count)}
                {point.share === null || point.share === undefined ? "" : ` (${Math.round(point.share * 100)}%)`}
              </b>
            </span>
          ))}
        </div>
      ) : null}
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>#</th>
              <th>the point of grammar</th>
              <th>fixes</th>
              <th>topic</th>
              <th>exercises</th>
              <th>why</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {lessons.map((lesson, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td className="wrap">{String(lesson.focus || "–")}</td>
                <td>
                  <span className={`badge ${lesson.targets && lesson.targets !== "none" ? "fail" : "unrated"}`}>
                    {String(lesson.targets || "none")}
                  </span>
                </td>
                <td className="wrap">{String(lesson.topic || "–")}</td>
                <td title="sentence openings, and the correct example sentences taught beside them">
                  {fmtInt(lesson.exercises)}
                  {lesson.drills ? <small> +{fmtInt(lesson.drills)} drills</small> : null}
                </td>
                <td className="wrap">{String(lesson.why || "–")}</td>
                <td>
                  <button type="button" className="small" disabled={disabled} onClick={() => onUse(lesson)}>
                    Use this lesson
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Every graded completion: the marks, the mistake, what the network wrote and what it should have written. */
function LessonTable({ rows, total, running }) {
  if (rows.length === 0) {
    return <p className="muted">{running ? "Waiting for the first lesson…" : "No lessons yet."}</p>;
  }
  return (
    <>
      {total > rows.length ? (
        <p className="muted">
          Showing the last {rows.length} of {total} lessons.
        </p>
      ) : null}
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>round</th>
              <th>exercise</th>
              <th>score</th>
              <th>gram</th>
              <th>spell</th>
              <th>flu</th>
              <th>mark</th>
              <th>mistake</th>
              <th>the network wrote</th>
              <th>correct English</th>
              <th>what changed</th>
              <th>the teacher says</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const mark = markOf(r);
              const score = typeof r.score === "number" && Number.isFinite(r.score) ? r.score : null;
              return (
                <tr key={i}>
                  <td>{r.round === "–" ? "–" : fmtInt(r.round)}</td>
                  <td title={r.focus ? `drilling ${r.focus}` : undefined}>
                    {String(r.exercise ?? "")}
                    {r.attempt > 1 ? <small> #{fmtInt(r.attempt)}</small> : null}
                  </td>
                  <td>
                    <span className="rating">
                      <span className={`rating-bar ${mark === "pass" ? "pass" : "fail"}`} aria-hidden="true">
                        <i style={{ width: `${score === null ? 0 : Math.max(0, Math.min(100, score * 10))}%` }} />
                      </span>
                      {score === null ? "–" : fmtNum(score, 1)}
                    </span>
                  </td>
                  <td>{fmtNum(r.grammar, 1)}</td>
                  <td>{fmtNum(r.spelling, 1)}</td>
                  <td>{fmtNum(r.fluency, 1)}</td>
                  <td>
                    <span className={`badge ${mark}`}>{mark}</span>
                  </td>
                  <td>{String(r.error ?? "–")}</td>
                  <td className="wrap">{String(r.sentence ?? "")}</td>
                  <td className="wrap">{String(r.correction ?? "")}</td>
                  <td className="wrap">
                    <Changes changes={r.changes} />
                  </td>
                  <td className="wrap">{String(r.comment ?? "")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** What each round learned from its grades. */
function RoundTable({ rounds }) {
  if (rounds.length === 0) return null;
  const batched = rounds.some((r) => Number(r.batch) > 1); // only an auto run has more than one
  return (
    <>
      <h3>Rounds</h3>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              {batched ? <th>batch</th> : null}
              <th>round</th>
              <th>lessons</th>
              <th>passed</th>
              <th>mean</th>
              <th>grammar</th>
              <th>weakest</th>
              <th>2NRL</th>
              <th>corrections</th>
              <th>garbage</th>
              <th>taught</th>
              <th>weight</th>
              <th>neg loss</th>
              <th>pos loss</th>
              <th>seconds</th>
            </tr>
          </thead>
          <tbody>
            {rounds.map((r, i) => (
              <tr key={i}>
                {batched ? <td>{fmtInt(r.batch)}</td> : null}
                <td>{fmtInt(r.round)}</td>
                <td>{fmtInt(r.lessons)}</td>
                <td>{fmtInt(r.passed)}</td>
                <td>{fmtNum(r.mean_score, 2)}</td>
                <td>{fmtNum(r.mean_grammar, 2)}</td>
                <td className="wrap">{asArray(r.weakest).join(", ") || "–"}</td>
                <td>{r.action || "–"}</td>
                <td title="corrections taught from their diff: steps penalised / taught">
                  {r.corrections ? (
                    <>
                      {fmtInt(r.corrections)} <small>({fmtInt(r.penalised)}/{fmtInt(r.rewarded)})</small>
                    </>
                  ) : (
                    "–"
                  )}
                </td>
                <td>{fmtInt(r.bad)}</td>
                <td>{fmtInt(r.good)}</td>
                <td title="mean negative-phase weight: how badly the failed sentences failed">
                  {fmtNum(r.mean_weight, 2)}
                </td>
                <td>{fmtNum(r.neg_loss, 4)}</td>
                <td>{fmtNum(r.pos_loss, 4)}</td>
                <td>{fmtNum(r.seconds, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/**
 * The prediction process with nobody at the keyboard: the teacher (a local Ollama model or
 * ChatGPT) writes the prefixes, the network completes them, the teacher marks the English and
 * the grades drive 2NRL.
 */
export default function TutorPanel({ status }) {
  const [provider, setProvider] = useState("ollama");
  const defaults = providerDefaults(status, provider);
  const [url, setUrl] = useState("");
  const [model, setModel] = useState("");
  const [topic, setTopic] = useState("everyday life");
  const [focus, setFocus] = useState("");
  const [level, setLevel] = useState("beginner");
  const [words, setWords] = useState("3 to 6");
  const [brief, setBrief] = useState("");
  const [rounds, setRounds] = useState("3");
  const [exercises, setExercises] = useState("5");
  const [attempts, setAttempts] = useState("1");
  const [mode, setMode] = useState("dijkstra");
  const [length, setLength] = useState("20");
  const [maxLength, setMaxLength] = useState("80");
  const [temperature, setTemperature] = useState("1.0");
  const [threshold, setThreshold] = useState("6");
  const [grammarWeight, setGrammarWeight] = useState("0.6");
  const [drills, setDrills] = useState("0");
  const [planCount, setPlanCount] = useState(String(DEFAULT_PLAN_LESSONS));
  const [batches, setBatches] = useState("1");
  const [adapt, setAdapt] = useState(true);
  const [teachAnswer, setTeachAnswer] = useState(true);
  const [twonrlPer, setTwonrlPer] = useState("round");
  const [diffCorrections, setDiffCorrections] = useState(true);
  const [blame, setBlame] = useState(false);
  const [keepWeight, setKeepWeight] = useState("0.25");
  const [minWeight, setMinWeight] = useState("0.25");
  const [negEpochs, setNegEpochs] = useState("2");
  const [posEpochs, setPosEpochs] = useState("3");
  const [negLr, setNegLr] = useState("0.5");
  const [posLr, setPosLr] = useState("0.1");
  const [batchSize, setBatchSize] = useState("4");
  const [checkpointEvery, setCheckpointEvery] = useState("0");
  const [prefixes, setPrefixes] = useState("");
  const [preview, setPreview] = useState(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [plan, setPlan] = useState(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [planError, setPlanError] = useState(null);
  const [showPlan, setShowPlan] = useState(false);
  const planRef = useRef(null);
  const appliedBatch = useRef(0);
  const [notice, setNotice] = useState(null);
  const [formError, setFormError] = useState(null);
  const [serverHistory, setServerHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const { job, running, busy, error, start, stop, clearError } = useJob("tutor");

  const otherJobRunning = jobIsRunning(status) && !running;

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.tutorHistory();
      setServerHistory(asArray(data && data.history));
      setHistoryError(null);
    } catch (err) {
      setHistoryError(err.message);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const jobState = job ? job.state : null;
  useEffect(() => {
    if (jobState && jobState !== "running") loadHistory();
  }, [jobState, loadHistory]);

  /** The settings both the job and the dry run send. */
  function settings() {
    return {
      ...overridesOf(provider, url, model, defaults),
      topic: topic.trim(),
      ...(focus.trim() ? { focus: focus.trim() } : {}),
      level: level.trim() || "beginner",
      words: words.trim() || "3 to 6",
      brief: brief.trim(),
      exercises: parseInteger(exercises, 5),
      attempts: parseInteger(attempts, 1),
      mode,
      length: parseInteger(length, 20),
      max_length: parseInteger(maxLength, 80),
      temperature: parseNumber(temperature, 1),
      threshold: parseNumber(threshold, 6),
      grammar_weight: parseNumber(grammarWeight, 0.6),
      adapt,
      teach_answer: teachAnswer,
    };
  }

  async function handleStart(event) {
    event.preventDefault();
    if (!topic.trim()) {
      setFormError("Give the teacher a topic to write exercises about.");
      return;
    }
    setFormError(null);
    setNotice(null);
    setPlanError(null);
    appliedBatch.current = 0;
    await start(() =>
      api.tutorStart({
        ...settings(),
        rounds: parseInteger(rounds, 3),
        drills: parseInteger(drills, 0),
        plan: parseInteger(planCount, DEFAULT_PLAN_LESSONS),
        batches: Math.max(0, parseInteger(batches, 1)),
        twonrl_per: twonrlPer,
        diff_corrections: diffCorrections,
        blame,
        keep_weight: parseNumber(keepWeight, 0.25),
        min_weight: parseNumber(minWeight, 0.25),
        neg_epochs: parseInteger(negEpochs, 2),
        pos_epochs: parseInteger(posEpochs, 3),
        neg_lr: parseNumber(negLr, 0.5),
        pos_lr: parseNumber(posLr, 0.1),
        batch_size: parseInteger(batchSize, 4),
        checkpoint_every: parseInteger(checkpointEvery, 0),
      }),
    );
  }

  /** One round of exercises, completions and grades - nothing is trained and nothing is saved. */
  async function handlePreview() {
    if (!topic.trim() && splitLines(prefixes).length === 0) {
      setFormError("Give the teacher a topic, or write the prefixes yourself.");
      return;
    }
    setFormError(null);
    setPreviewBusy(true);
    try {
      const given = splitLines(prefixes);
      const data = await api.tutorLesson({
        ...settings(),
        topic: topic.trim() || "everyday life",
        ...(given.length > 0 ? { prefixes: given } : {}),
      });
      setPreview(data);
    } catch (err) {
      setFormError(err.message);
    } finally {
      setPreviewBusy(false);
    }
  }

  /** Hand a report card back to the teacher and show the lessons it plans from it. */
  async function handlePlan(card) {
    if (!card) return;
    setFormError(null);
    setNotice(null);
    setPlanError(null);
    setPlanBusy(true);
    try {
      const data = await api.tutorPlan({
        ...settings(),
        topic: topic.trim() || "everyday life",
        drills: parseInteger(drills, 0),
        count: Math.max(1, parseInteger(planCount, DEFAULT_PLAN_LESSONS)),
        report: card,
      });
      setPlan((data && data.plan) || null);
      setShowPlan(true); // the plan lands at the foot of the tab: take the reader to it
    } catch (err) {
      // 404: the API this tab is talking to is older than the tab itself
      setPlanError(
        err.status === 404
          ? "This server has no POST /api/tutor/plan: it is running a version older than this page. Restart it (or rebuild the Go server) and try again."
          : err.message,
      );
    } finally {
      setPlanBusy(false);
    }
  }

  /** Load the whole plan into the settings above: its brief, and the step up the marks earned. */
  function teachNextBatch(next) {
    const upgrade = next.upgrade && typeof next.upgrade === "object" ? next.upgrade : {};
    if (next.prompt) setBrief(String(next.prompt));
    if (next.topic) setTopic(String(next.topic));
    if (upgrade.level) setLevel(String(upgrade.level));
    if (upgrade.words) setWords(String(upgrade.words));
    if (upgrade.threshold !== null && upgrade.threshold !== undefined) setThreshold(String(upgrade.threshold));
    if (upgrade.drills !== null && upgrade.drills !== undefined) setDrills(String(upgrade.drills));
    setFocus(""); // the brief carries the points of grammar, in order
    setFormError(null);
    setNotice(
      `The next batch is loaded (${String(upgrade.step || "hold")}: ${String(upgrade.level || level)}, openings of ${String(upgrade.words || words)} words): press Start lessons to teach it.`,
    );
    if (typeof window !== "undefined" && typeof window.scrollTo === "function") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  /** Load one planned lesson into the settings above, ready to start. */
  function usePlanLesson(lesson) {
    const openings = asArray(lesson.prefixes).filter((p) => typeof p === "string" && p.trim());
    if (lesson.topic) setTopic(String(lesson.topic));
    setFocus(String(lesson.focus || ""));
    if (plan && plan.level) setLevel(String(plan.level));
    if (lesson.exercises) setExercises(String(lesson.exercises));
    if (lesson.drills !== null && lesson.drills !== undefined) setDrills(String(lesson.drills));
    if (openings.length > 0) setPrefixes(openings.join("\n"));
    setFormError(null);
    setNotice(
      `Loaded "${lesson.focus || lesson.topic || "the lesson"}" into the settings: press Start lessons to teach it.`,
    );
    if (typeof window !== "undefined" && typeof window.scrollTo === "function") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  const jobHistory = asArray(job && job.history);
  const history = running || jobHistory.length > 0 ? jobHistory : serverHistory;
  const lessons = history.filter((r) => r && r.kind === "lesson");
  const roundRecords = history.filter((r) => r && r.kind === "round");
  const reports = history.filter((r) => r && r.kind === "report");
  const card = reports.length ? reports[reports.length - 1] : null;
  const plans = history.filter((r) => r && r.kind === "plan");
  const plannedByRun = plans.length ? plans[plans.length - 1] : null;
  const started = history.filter((r) => r && r.kind === "batch");
  const startedBatch = started.length ? started[started.length - 1] : null;
  useEffect(() => {
    // an auto run applies each plan itself; the settings follow it here so the form shows what is being taught
    const number = startedBatch ? Number(startedBatch.batch) || 0 : 0;
    if (!startedBatch || number <= appliedBatch.current) return;
    appliedBatch.current = number;
    if (startedBatch.brief) setBrief(String(startedBatch.brief));
    if (startedBatch.topic) setTopic(String(startedBatch.topic));
    if (startedBatch.level) setLevel(String(startedBatch.level));
    if (startedBatch.words) setWords(String(startedBatch.words));
    if (startedBatch.threshold !== null && startedBatch.threshold !== undefined) {
      setThreshold(String(startedBatch.threshold));
    }
    if (startedBatch.drills !== null && startedBatch.drills !== undefined) setDrills(String(startedBatch.drills));
    setFocus("");
    setNotice(
      `Batch ${number} started (${String(startedBatch.step || "hold")}): ${String(startedBatch.brief || "")}`,
    );
  }, [startedBatch]);
  useEffect(() => {
    if (plannedByRun) setPlan(plannedByRun); // a run that ended with a plan of its own shows it straight away
  }, [plannedByRun]);
  useEffect(() => {
    if (!showPlan || !plan || !planRef.current) return;
    setShowPlan(false);
    if (typeof planRef.current.scrollIntoView === "function") {
      planRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [showPlan, plan]);
  const series = [
    {
      name: "mean score",
      color: "#2563eb",
      values: roundRecords.map((r, i) => ({ x: i + 1, y: r.mean_score })),
    },
    {
      name: "grammar",
      color: "#16a34a",
      dashed: true,
      values: roundRecords.map((r, i) => ({ x: i + 1, y: r.mean_grammar })),
    },
  ];
  const previewLessons = asArray(preview && preview.lessons).map(lessonRecord);

  return (
    <>
      <form className="card" onSubmit={handleStart}>
        <h2>Tutor</h2>
        <p className="muted">
          The Predict tab with nobody at the keyboard. Each round the teacher ({providerLabel(provider)}) writes
          sentence openings about the topic — one point of grammar each, and its own model answer — the network
          completes them with the prediction search, and the same model marks every sentence as an English teacher: grammar, spelling and
          fluency out of 10, the worst mistake named, one line of teaching and the sentence written out correctly.
          A correction is then taught <b>as a correction</b>: the sentence the network wrote and the teacher's
          version are aligned character by character, and only the trigram nodes they disagree on move — the step
          that wrote the wrong character is penalised, the step that writes the right one is rewarded, and the
          words both sentences share keep what they earned. Sentences with no correction to align stay 2NRL
          garbage weighted by how bad the mark was; with <b>adapt</b> on, the next round drills the mistakes this
          one made. At the end the report card goes back to the teacher, which writes the <b>lesson plan</b> that
          repairs it: one point of grammar per lesson, worst mistake first, the step up in difficulty the marks
          have earned, and the <b>brief</b> for the next batch — which the exercise writer is handed with every
          round of the run you start from it. Set <b>Batches</b> above 1 and the run does that itself: each batch
          ends with its report card, plans from it, fills the brief and the step up in below, and teaches the next
          one (0 keeps going until you press Stop).
        </p>
        <div className="row">
          <TextField
            label="Topic"
            hint="what the sentences are about"
            value={topic}
            onChange={setTopic}
            disabled={running}
            placeholder="everyday life"
          />
          <TextField
            label="Focus"
            hint="optional: one point of grammar"
            value={focus}
            onChange={setFocus}
            disabled={running}
            placeholder="past tense"
          />
          <TextField label="Level" value={level} onChange={setLevel} disabled={running} placeholder="beginner" />
          <TextField
            label="Prefix words"
            hint="how long an opening is"
            value={words}
            onChange={setWords}
            disabled={running}
            placeholder="3 to 6"
          />
        </div>
        <TextArea
          label="Brief"
          hint="what this batch is being taught to: the prompt the last report card led to"
          value={brief}
          onChange={setBrief}
          rows={2}
          disabled={running}
          placeholder="Drill subject-verb agreement and plural nouns, worst first. Keep the sentences about animals."
        />
        <div className="row auto">
          <NumberField label="Rounds" value={rounds} onChange={setRounds} min={1} step={1} disabled={running} />
          <NumberField
            label="Exercises"
            hint="per round"
            value={exercises}
            onChange={setExercises}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Attempts"
            hint="completions per exercise"
            value={attempts}
            onChange={setAttempts}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Drills"
            hint="extra correct sentences per round"
            value={drills}
            onChange={setDrills}
            min={0}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Plan"
            hint="lessons planned at the end, 0 = off"
            value={planCount}
            onChange={setPlanCount}
            min={0}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Batches"
            hint="auto run: each planned from the last, 0 = until stopped"
            value={batches}
            onChange={setBatches}
            min={0}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Checkpoint every"
            hint="rounds, 0 = off"
            value={checkpointEvery}
            onChange={setCheckpointEvery}
            min={0}
            step={1}
            disabled={running}
          />
        </div>
        {provider === "chatgpt" && !defaults.configured ? (
          <p className="muted issue">
            This server has no OPENAI_API_KEY, so ChatGPT cannot teach yet: set it (or OPENAI_API_KEY_FILE) in the
            server's environment and restart it.
          </p>
        ) : null}
        <div className="row">
          <SelectField
            label="Teacher"
            hint="who sets and marks the exercises"
            value={provider}
            onChange={setProvider}
            disabled={running}
            options={[
              ["ollama", "Ollama (local)"],
              ["chatgpt", "ChatGPT (OpenAI)"],
            ]}
          />
          <TextField
            label={`${providerLabel(provider)} URL`}
            hint="blank = the server default"
            value={url}
            onChange={setUrl}
            disabled={running}
            placeholder={defaults.url || (provider === "chatgpt" ? "https://api.openai.com/v1" : "http://127.0.0.1:11434")}
          />
          <TextField
            label="Teacher model"
            hint="blank = the server default"
            value={model}
            onChange={setModel}
            disabled={running}
            placeholder={defaults.model || (provider === "chatgpt" ? "gpt-4o-mini" : "llama3.2")}
          />
        </div>

        <h3>How the network answers</h3>
        <div className="row auto">
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            disabled={running}
            options={[
              ["dijkstra", "cheapest path (deterministic)"],
              ["beam", "beam search"],
              ["sample", "stochastic walk"],
            ]}
          />
          <NumberField label="Length" value={length} onChange={setLength} min={0} step={1} disabled={running} />
          <NumberField
            label="Max length"
            value={maxLength}
            onChange={setMaxLength}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Temperature"
            hint="sampled attempts"
            value={temperature}
            onChange={setTemperature}
            min={0}
            disabled={running}
          />
        </div>

        <h3>How it is marked</h3>
        <div className="row auto">
          <NumberField
            label="Pass mark"
            hint="out of 10"
            value={threshold}
            onChange={setThreshold}
            min={0}
            max={10}
            disabled={running}
          />
          <NumberField
            label="Grammar weight"
            hint="share of the mark"
            value={grammarWeight}
            onChange={setGrammarWeight}
            min={0}
            max={1}
            step={0.05}
            disabled={running}
          />
          <div className="checks">
            <CheckField label="Adapt to the weakest points" checked={adapt} onChange={setAdapt} disabled={running} />
            <CheckField
              label="Teach the model answer too"
              checked={teachAnswer}
              onChange={setTeachAnswer}
              disabled={running}
            />
          </div>
        </div>

        <h3>What it learns</h3>
        <div className="row auto">
          <div className="checks">
            <CheckField
              label="Teach corrections from the diff"
              checked={diffCorrections}
              onChange={setDiffCorrections}
              disabled={running}
            />
            <CheckField
              label="Teach the negative network why each sentence failed"
              checked={blame}
              onChange={setBlame}
              disabled={running}
            />
          </div>
          <NumberField
            label="Unchanged words keep"
            hint="0 = the fix alone, 1 = the whole sentence"
            value={keepWeight}
            onChange={setKeepWeight}
            min={0}
            max={1}
            step={0.05}
            disabled={running || !diffCorrections}
          />
          <SelectField
            label="2NRL"
            value={twonrlPer}
            onChange={setTwonrlPer}
            disabled={running}
            options={[
              ["round", "once per round"],
              ["lesson", "after every lesson"],
            ]}
          />
          <NumberField
            label="Least garbage weight"
            hint="a near miss; hopeless = 1"
            value={minWeight}
            onChange={setMinWeight}
            min={0}
            max={1}
            step={0.05}
            disabled={running}
          />
          <NumberField
            label="Negative epochs"
            value={negEpochs}
            onChange={setNegEpochs}
            min={0}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Positive epochs"
            value={posEpochs}
            onChange={setPosEpochs}
            min={0}
            step={1}
            disabled={running}
          />
          <NumberField label="Negative lr" value={negLr} onChange={setNegLr} min={0} disabled={running} />
          <NumberField label="Positive lr" value={posLr} onChange={setPosLr} min={0} disabled={running} />
          <NumberField
            label="Batch size"
            value={batchSize}
            onChange={setBatchSize}
            min={1}
            step={1}
            disabled={running}
          />
        </div>

        <div className="actions">
          <button type="submit" className="primary" disabled={running || busy || otherJobRunning}>
            {busy ? "Starting…" : parseInteger(batches, 1) === 1 ? "Start lessons" : "Start auto run"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
          <button type="button" disabled={running || previewBusy} onClick={handlePreview}>
            {previewBusy ? "Marking…" : "Dry run (mark, do not train)"}
          </button>
        </div>
        {previewBusy ? <p className="muted">{slowNote(provider)}</p> : null}
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <TextArea
          label="Own prefixes"
          hint="dry run only: one per line, skips the exercise writer"
          value={prefixes}
          onChange={setPrefixes}
          rows={3}
          disabled={running}
          placeholder={"the children were\nevery morning she"}
        />
        <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />
        <Alert message={formError} onDismiss={() => setFormError(null)} />
        <Alert message={error} onDismiss={clearError} />
      </form>

      {preview ? (
        <div className="card">
          <div className="toolbar">
            <h2>Dry run</h2>
            <button type="button" className="small" onClick={() => setPreview(null)}>
              Clear
            </button>
          </div>
          <p className="muted">
            Marked by <b>{String(preview.model ?? "–")}</b>
            {preview.source === "given" ? " on your own prefixes" : ""}; nothing was trained and nothing was saved.
          </p>
          <ReportCard card={preview.report} title="Report card" />
          <div className="actions">
            <button
              type="button"
              disabled={running || planBusy || !preview.report || !preview.report.lessons}
              onClick={() => handlePlan(preview.report)}
            >
              {planBusy ? "Planning…" : "Plan the next lessons"}
            </button>
            {planBusy ? <span className="muted">{planNote(provider)}</span> : null}
          </div>
          <Alert message={planError} onDismiss={() => setPlanError(null)} />
          <LessonTable rows={previewLessons} total={previewLessons.length} running={false} />
        </div>
      ) : null}

      <div className="card">
        <div className="toolbar">
          <h2>Lessons</h2>
          <button type="button" className="small" onClick={loadHistory}>
            Reload history
          </button>
        </div>
        <JobStatus job={job} emptyText="No tutor job in this session." />
        <Alert message={historyError} onDismiss={() => setHistoryError(null)} />
        {roundRecords.length > 0 ? (
          <LineChart
            series={series}
            xLabel="round (record #)"
            yLabel="mark out of 10"
            emptyText="No rounds yet."
          />
        ) : null}
        <ReportCard
          card={card}
          title={card && Number(card.batch) > 1 ? `Report card · batch ${fmtInt(card.batch)}` : "Report card"}
        />
        {card && card.lessons ? (
          <>
            <div className="actions">
              <button type="button" disabled={running || planBusy} onClick={() => handlePlan(card)}>
                {planBusy ? "Planning…" : "Plan the next lessons"}
              </button>
              <span className="muted">
                {planBusy ? planNote(provider) : null}
                {!planBusy && running ? "The lessons are still running; the plan is written from the card at the end." : null}
                {!planBusy && !running
                  ? "The teacher reads the report card and writes the syllabus that repairs it, worst mistake first."
                  : null}
              </span>
            </div>
            <Alert message={planError} onDismiss={() => setPlanError(null)} />
          </>
        ) : null}
        <RoundTable rounds={roundRecords.slice(-MAX_ROWS)} />
        <h3>Every lesson</h3>
        <LessonTable rows={lessons.slice(-MAX_ROWS)} total={lessons.length} running={running} />
      </div>

      <LessonPlanCard
        plan={plan}
        onUse={usePlanLesson}
        onTeach={teachNextBatch}
        onClear={() => {
          setPlan(null);
          setPlanError(null);
        }}
        disabled={running}
        cardRef={planRef}
      />
    </>
  );
}
