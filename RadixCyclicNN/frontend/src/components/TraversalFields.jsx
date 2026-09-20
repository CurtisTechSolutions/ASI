import { NumberField, SelectField } from "./Fields.jsx";
import { parseNumber } from "../util.js";

/**
 * The traversal option: *what* a search looks for, as opposed to the mode,
 * which is how it looks for it.
 *
 * **reward** is the model's own distribution - the count / reward model's
 * probability carries `exp(reward_scale * reward)`, so the search follows the
 * rewards. **punishment** takes the rewards out of the score altogether and
 * lets the penalties price every step, so the cheapest path is the one that
 * accumulated the least punishment. The two scales say how heavily a
 * punishment counts and how heavily what the corpus did counts; a merit scale
 * of 0 is the pure form, where nothing but the punishments decides.
 */

export const TRAVERSALS = [
  ["reward", "reward (follow what was rewarded)"],
  ["punishment", "punishment (avoid what was punished)"],
];

/** The request fields for a traversal; `{}` for the default, so an old server is unaffected. */
export function traversalBody(traversal, penaltyScale, meritScale) {
  if (traversal !== "punishment") return {};
  return {
    traversal,
    penalty_scale: parseNumber(penaltyScale, 1),
    merit_scale: parseNumber(meritScale, 1),
  };
}

export default function TraversalFields({
  traversal,
  onTraversal,
  penaltyScale,
  onPenaltyScale,
  meritScale,
  onMeritScale,
}) {
  const punishing = traversal === "punishment";
  return (
    <>
      <div className="row">
        <SelectField label="Traversal" value={traversal} onChange={onTraversal} options={TRAVERSALS} />
        {punishing ? (
          <NumberField
            label="Penalty scale"
            hint="how heavily a punishment counts"
            value={penaltyScale}
            onChange={onPenaltyScale}
            min={0}
          />
        ) : null}
      </div>
      {punishing ? (
        <div className="row">
          <NumberField
            label="Merit scale"
            hint="how heavily what the corpus did counts; 0 = nothing but the punishments decides"
            value={meritScale}
            onChange={onMeritScale}
            min={0}
          />
        </div>
      ) : null}
      {punishing ? (
        <p className="muted">
          The rewards leave the score: only the punishments price a step, so the cheapest path is the one that
          accumulated the <b>least punishment</b>. On the negative network it walks the least blamed way through
          the failures instead of the likeliest one.
        </p>
      ) : null}
    </>
  );
}
