import { actionLabel, bucketTitle, seatsOf, triggerText } from "../trace.js";
import { fmtNum } from "../util.js";
import Strip from "./Strip.jsx";

/** One seat as a chip: its gram, its bucket, its bid, and the slot it played (abstain shown as such). */
function SeatChip({ seat }) {
  const abstained = seat.action === "abstain";
  return (
    <span
      className={`seat-chip${abstained ? " abstained" : ""}${seat.seated ? "" : " unseated"}`}
      title={`node ${seat.node} "${seat.label}" seated as ${seat.bucket} (${bucketTitle(seat.bucket)}): bid ${fmtNum(seat.bid, 3)}, reputation ${fmtNum(
        seat.reputation,
        3,
      )}, wealth ${fmtNum(seat.wealth, 1)}${seat.seated ? `, played ${actionLabel(seat.action)}` : ", not seated"}${
        seat.phi === null || seat.phi === undefined ? "" : `, φ ${fmtNum(seat.phi, 3)}`
      }`}
    >
      <code>{seat.label === "" ? "∅" : seat.label}</code>
      <small className={`bucket ${seat.bucket}`}>{seat.bucket}</small>
      <small>{fmtNum(seat.bid, 2)}</small>
      {seat.seated ? <b>{abstained ? "abstain" : actionLabel(seat.action)}</b> : null}
    </span>
  );
}

/**
 * The naming trace: one row per character of the string. Each row is the stage game that named a state there -
 * the seats the auction sold (each node's gram, its bucket in the window, its bid, the slot it played), the
 * aggregate over the lattice's states as a strip (the named state filled, the lattice's prediction marked ▾),
 * the margin, the clearing price, how many bid above the reserve, and what would have asked the meta-player.
 * A click on a row picks that step (the Game tab shows it in detail).
 */
export default function NamingTrace({ read, selected, onSelect }) {
  const trace = read && Array.isArray(read.trace) ? read.trace : [];
  if (trace.length === 0) {
    return <p className="muted">The empty string: no step is taken, and the verdict is the start state's.</p>;
  }
  return (
    <div className="table-wrap trace-wrap">
      <table className="data trace">
        <thead>
          <tr>
            <th>#</th>
            <th title="the character read">char</th>
            <th title="the state the lattice was in: the row's source, and the key of every seat's belief">from</th>
            <th title="the seats sold, highest bid first: gram, bucket, bid, the slot played">seats</th>
            <th title="the aggregate over the states: the named state filled, the lattice's prediction marked ▾">aggregate</th>
            <th title="the aggregate's argmax: the state the encoder names">named</th>
            <th title="the lattice's own target from the same row, and its probability">lattice</th>
            <th title="top1 - top2 of the aggregate">margin</th>
            <th title="the highest losing bid, which every seat pays">price</th>
            <th title="candidates that bid above the reserve">bidders</th>
            <th title="what would have asked the meta-player">triggers</th>
          </tr>
        </thead>
        <tbody>
          {trace.map((step, i) => {
            const { seated, unseated } = seatsOf(step);
            const why = triggerText(step);
            return (
              <tr
                key={i}
                className={`clickable${i === selected ? " selected" : ""}${step.named === step.predicted ? "" : " disagree"}`}
                onClick={() => onSelect && onSelect(i)}
              >
                <td className="mono">{i}</td>
                <td className="mono char">{step.symbol}</td>
                <td className="mono">{step.prev}</td>
                <td className="seats">
                  {seated.map((s) => (
                    <SeatChip key={`${s.node}-${s.bucket}`} seat={s} />
                  ))}
                  {seated.length === 0 ? <span className="muted">nobody seated</span> : null}
                  {unseated.length ? (
                    <small className="muted" title={unseated.map((s) => `${s.label} (${s.bucket}, bid ${fmtNum(s.bid, 3)})`).join("\n")}>
                      {" "}
                      +{unseated.length} not seated
                    </small>
                  ) : null}
                </td>
                <td>
                  <Strip values={step.aggregate} named={step.named} predicted={step.predicted} accepting={read.accepting} />
                </td>
                <td className="mono">
                  <b>{step.named}</b>
                  {Array.isArray(read.accepting) && read.accepting.includes(step.named) ? " ✓" : ""}
                </td>
                <td className="mono" title={step.named === step.predicted ? "agrees with the named state" : "disagrees with the named state"}>
                  {step.predicted} <small className="muted">p {fmtNum(step.probability, 2)}</small>
                </td>
                <td className="mono">{fmtNum(step.margin, 3)}</td>
                <td className="mono">{fmtNum(step.price, 3)}</td>
                <td className="mono">{step.bidders}</td>
                <td>{why ? <span className="badge trigger">{why}</span> : <span className="muted">–</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
