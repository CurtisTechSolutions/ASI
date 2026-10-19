import { MAX_TEMPERATURE, autoLabel } from "../lm.js";
import { CheckField, NumberField } from "./Fields.jsx";

/**
 * The auto-temperature switch and its top rung: a reply that is no answer - nothing, no words, a loop - is drawn
 * again 0.2 hotter, a few times a rung, until it is an answer or the top rung is passed.
 */
export function AutoTempFields({ on, onChange, top, onTop }) {
  return (
    <>
      <CheckField
        label="Auto temperature"
        checked={on}
        onChange={onChange}
        hint="a reply that is nothing, no words or a loop is drawn again, hotter"
      />
      <NumberField
        label="Up to"
        value={top}
        onChange={onTop}
        min="0"
        max="2"
        step="0.1"
        disabled={!on}
        placeholder={String(MAX_TEMPERATURE)}
        hint="the hottest it climbs to"
      />
    </>
  );
}

/** How auto-temperature found a reply, when it had to climb: a badge, the whole ladder in its title. */
export function AutoBadge({ auto }) {
  const label = autoLabel(auto);
  if (!label) return null;
  return (
    <span className={`badge tight auto${auto.found ? "" : " miss"}`} title={auto.note || label}>
      🌡 {label}
    </span>
  );
}
