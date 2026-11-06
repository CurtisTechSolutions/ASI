import { stripBars } from "../trace.js";

/**
 * A distribution over the lattice's states as a strip of bars, one a state, each scaled to the largest: the named
 * state filled in the accent, the lattice's prediction marked ▾ above its bar, accepting states underlined.
 * `abstain` (a belief's last slot) is drawn as one more bar, apart and hatched. Hover a bar for its number.
 */
export default function Strip({ values, named = -1, predicted = -1, accepting = [], abstain, large = false, label }) {
  const bars = stripBars(values, named, predicted);
  if (bars.length === 0) return <span className="muted">–</span>;
  const acc = Array.isArray(accepting) ? accepting : [];
  return (
    <span className={`strip${large ? " large" : ""}`} role="img" aria-label={label || "a distribution over the states"}>
      {bars.map((b) => (
        <span
          key={b.state}
          className={`strip-slot${b.named ? " named" : ""}${b.predicted ? " predicted" : ""}${acc.includes(b.state) ? " accepting" : ""}`}
          title={`state ${b.state}: ${b.p.toFixed(4)}${b.named ? " · named" : ""}${b.predicted ? " · the lattice's prediction" : ""}${
            acc.includes(b.state) ? " · accepting" : ""
          }`}
        >
          <span className="strip-bar" style={{ height: `${Math.max(2, Math.round(b.height * 100))}%` }} />
          {large ? <small>{b.state}</small> : null}
        </span>
      ))}
      {abstain !== undefined && abstain !== null ? (
        <span className="strip-slot abstain" title={`abstain: ${Number(abstain).toFixed(4)}`}>
          <span
            className="strip-bar"
            style={{
              height: `${Math.max(2, Math.round((Number(abstain) / Math.max(Number(abstain), ...bars.map((b) => b.p), 1e-12)) * 100))}%`,
            }}
          />
          {large ? <small>∅</small> : null}
        </span>
      ) : null}
    </span>
  );
}
