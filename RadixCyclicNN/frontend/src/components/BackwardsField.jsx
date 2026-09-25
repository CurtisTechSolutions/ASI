import { CheckField } from "./Fields.jsx";
import { useSiteSettings } from "../hooks/useSiteSettings.jsx";
import { unitName } from "../util.js";

/**
 * "Query backwards": whether Predict and Generate ask the model backwards
 * (`../backwards.js`, ../../SPEC-SearchAndTraining.md section 9).
 *
 * For a model trained with "Read every text backwards" on the Train tab: it
 * learned every text from its end to its start, so it is asked the same way -
 * the query is turned around before it is sent and the answer turned around
 * when it comes back, in the model's own units - and says what came *before*
 * the text it was given. A thumbs up or down on an answer goes to the model
 * the way it reads, backwards.
 *
 * Site-wide (`useSiteSettings`): the Settings tab, Predict and Generate show
 * one value. `compact` is the version for the action tabs - the box and a
 * line, without the explanation.
 */
export default function BackwardsField({ compact = false, status = null }) {
  const { backwards } = useSiteSettings();
  return (
    <>
      <CheckField
        label="Query backwards"
        hint={compact ? "shared with Settings; for a model trained backwards" : "for a model trained backwards"}
        checked={backwards.on}
        onChange={backwards.set}
      />
      {compact && backwards.on ? (
        <p className="muted">
          Backwards: what you type is sent turned around, {unitName(status, false)} by {unitName(status, false)},
          and every answer is turned back round - so it ends with your text, and what the model adds is what it says
          came <b>before</b>.
        </p>
      ) : null}
      {compact ? null : (
        <p className="muted">
          A model trained with <b>Read every text backwards</b> (the Train tab) learned every text from its end to
          its start, so what it continues is what came <b>before</b>. Asked the usual way round it answers
          nonsense; with this on, Predict and Generate turn the prefix around before it is sent and the answer
          around when it comes back - character by character, or word by word on a word model - so you type the
          end of a text and read what precedes it, the right way round. A thumbs up or down on an answer goes to
          the model backwards, the way it reads. Leave it off for a model trained the usual way.
        </p>
      )}
    </>
  );
}
