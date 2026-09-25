import { Fragment } from "react";
import { fmtInt, fmtNum, showWhitespace } from "../util.js";
import {
  isSentinel,
  learnedSays,
  questionsOf,
  stoppedSays,
  summarizeThought,
  thenSays,
  thoughtTree,
  triggerSays,
} from "../thinking.js";

/**
 * What a voice thought before backing up out of a repeat, in a transcript: the summary of the thought and,
 * one level down each, the questions it asked itself. Renders nothing without a thought.
 */
export function ThoughtLine({ thought }) {
  if (!thought || typeof thought !== "object") return null;
  const [top, ...questions] = thoughtTree(thought);
  return (
    <div className="thought-line" title="what it thought before backing up: a walk from the THINK sentinel">
      <div>
        <span aria-hidden="true">💭 </span>
        {summarizeThought(top.thought)}
      </div>
      {questions.map(({ thought: q, depth }, i) => (
        <div key={i} className="thought-question" style={{ paddingLeft: `${depth * 1.2}em` }}>
          ↳ {summarizeThought(q)}
        </div>
      ))}
    </div>
  );
}

/** The path a thought walked, THINK first: one chip per node, the cost of the step into it beside it. */
function ThoughtPath({ thought }) {
  const labels = Array.isArray(thought.labels) ? thought.labels : [];
  const ids = Array.isArray(thought.node_ids) ? thought.node_ids : [];
  const costs = Array.isArray(thought.step_costs) ? thought.step_costs : [];
  if (labels.length === 0) return null;
  return (
    <div className="chips thought-path">
      {labels.map((label, i) => {
        const cost = i > 0 ? costs[i - 1] : undefined;
        return (
          <Fragment key={i}>
            {i > 0 ? <span className="chip-arrow">→</span> : null}
            <span className={`chip${isSentinel(ids[i]) ? " sentinel" : ""}`} title={ids[i] !== undefined ? `node ${ids[i]}` : undefined}>
              {showWhitespace(label)}
              {typeof cost === "number" ? <small>{fmtNum(cost, 2)}</small> : null}
            </span>
          </Fragment>
        );
      })}
    </div>
  );
}

/**
 * A thought in full: what set it off, what it thought, the path it walked from THINK, how it stopped, what
 * it triggered and what it taught - and, nested one level deeper each, the questions it asked itself.
 */
export default function ThoughtView({ thought }) {
  if (!thought || typeof thought !== "object") return null;
  const t = thought;
  const questions = questionsOf(t);
  const learned = learnedSays(t);
  return (
    <div className={`thought${t.depth ? " question" : ""}`}>
      <p className="thought-summary">
        <b>{summarizeThought(t)}</b>
      </p>
      {t.text ? <p className="thought-text">{t.text}</p> : null}
      <dl className="kv">
        <dt>trigger</dt>
        <dd>{triggerSays(t.trigger)}</dd>
        <dt>at</dt>
        <dd>
          {Number.isInteger(t.at) && t.at >= 0 ? `node ${t.at}` : "nowhere in particular"}
          {t.about ? <> · where “{t.about}” ends</> : null}
        </dd>
        <dt>stopped</dt>
        <dd>{stoppedSays(t.stopped)}</dd>
        <dt>then</dt>
        <dd>{thenSays(t.then)}</dd>
        <dt>learned</dt>
        <dd>{learned || "nothing"}</dd>
        {t.text ? (
          <>
            <dt>cost</dt>
            <dd>
              {fmtNum(t.cost, 3)} · p {fmtNum(t.probability, 4)} · {fmtInt(t.expanded)} state(s) weighed
            </dd>
          </>
        ) : null}
      </dl>
      <ThoughtPath thought={t} />
      {questions.length ? (
        <ol className="thought-questions" aria-label="the questions it asked itself">
          {questions.map((q, i) => (
            <li key={i}>
              <ThoughtView thought={q} />
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
