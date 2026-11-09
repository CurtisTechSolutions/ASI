import { actionLabel, bucketTitle, clampStep, fmtAggregate, fmtSigned, phiKnown, seatsOf, splitBelief, triggerText } from "../trace.js";
import { fmtNum, yesNo } from "../util.js";
import Strip from "./Strip.jsx";

/** One seat of the game as a card: who it is, what it believes, what it bid, what it played and what it earned. */
function SeatCard({ seat, named, predicted, accepting, learned }) {
  const { states, abstain } = splitBelief(seat.belief);
  return (
    <div className={`seat-card${seat.seated ? "" : " unseated"}${seat.action === "abstain" ? " abstained" : ""}`}>
      <div className="seat-head">
        <code className="gram">{seat.label === "" ? "∅" : seat.label}</code>
        <span className={`bucket ${seat.bucket}`} title={bucketTitle(seat.bucket)}>
          {seat.bucket}
        </span>
        <span className="muted">node {seat.node}</span>
      </div>
      <dl className="kv">
        <dt>bid</dt>
        <dd className="mono">{fmtNum(seat.bid, 3)}</dd>
        <dt>reputation</dt>
        <dd className="mono">{fmtNum(seat.reputation, 3)}</dd>
        <dt>wealth</dt>
        <dd className="mono">{fmtNum(seat.wealth, 1)}</dd>
        <dt>played</dt>
        <dd className="mono">
          <b>{seat.seated ? actionLabel(seat.action) : "not seated"}</b>
        </dd>
        <dt title="its Shapley value against v(C) = log P_C(label): its share of the coalition's log-likelihood gain">φ</dt>
        <dd className="mono">{seat.seated ? (learned ? fmtSigned(seat.phi, 3) : "after a learning run") : "–"}</dd>
      </dl>
      <div className="seat-belief">
        <small className="muted">belief over the states, ∅ abstain · top {fmtAggregate(states, 2)}</small>
        <Strip values={states} named={named} predicted={predicted} accepting={accepting} abstain={abstain} label={`node ${seat.node}'s belief`} />
      </div>
    </div>
  );
}

/**
 * The Game tab: one step of the last read in detail. The seats as cards (the gram, the bucket, reputation, bid,
 * the belief over the states keyed by the lattice's previous state, the slot played, and - learning - the
 * Shapley value that credited it), the auction (price, bidders, seated), the equilibrium (exploitability, the
 * best-response cycle, convergence), the aggregate, and the named state against the lattice's prediction.
 */
export default function GamePanel({ read, step, setStep, health, goToBody }) {
  const trace = read && Array.isArray(read.trace) ? read.trace : [];
  if (trace.length === 0) {
    return (
      <div className="panel">
        <div className="card wide">
          <h2>The stage game</h2>
          <p className="muted">
            Nothing read yet. Classify or run a non-empty string on the{" "}
            <button type="button" className="link" onClick={goToBody}>
              Body
            </button>{" "}
            tab, then pick a step here.
          </p>
        </div>
      </div>
    );
  }
  const i = clampStep(step, trace.length);
  const s = trace[i];
  const { seated, unseated } = seatsOf(s);
  const learned = phiKnown(read);
  const enc = health && health.encoder ? health.encoder : {};
  const phiSum = seated.reduce((t, x) => t + (Number.isFinite(x.phi) ? x.phi : 0), 0);
  return (
    <div className="panel">
      <div className="card wide">
        <div className="toolbar">
          <h2>The stage game at one step</h2>
          <div className="tabs sub symbols" role="tablist" aria-label="Step">
            <span className="muted">step</span>
            {trace.map((t, k) => (
              <button
                key={k}
                type="button"
                role="tab"
                aria-selected={k === i}
                className={k === i ? "active" : ""}
                title={`step ${k}: read ${t.symbol}, named ${t.named}`}
                onClick={() => setStep(k)}
              >
                {t.symbol}
              </button>
            ))}
          </div>
        </div>
        <p className="note">
          <code>{read.text}</code>, step {i}: from state <b>{s.prev}</b> reading <code>{s.symbol}</code>. The encoder
          names <b>{s.named}</b>
          {read.accepting && read.accepting.includes(s.named) ? " (accepting)" : ""}; the lattice&apos;s own row predicts{" "}
          <b>{s.predicted}</b> with probability {fmtNum(s.probability, 3)} - they{" "}
          {s.named === s.predicted ? "agree" : <b className="bad">disagree</b>}. {read.learned ? "A learning run: every seat was settled." : "A quiet read: nothing moved."}
        </p>
        {!learned ? (
          <p className="alert info">
            φ (each seat&apos;s Shapley value) is only known after a learning run: the label arrives with the
            oracle&apos;s verdict and the seats are settled against it. Use Run (learn) on the Body tab to see it.
          </p>
        ) : null}
      </div>

      <div className="card">
        <h3>The auction</h3>
        <dl className="kv">
          <dt>candidates</dt>
          <dd className="mono">{seated.length + unseated.length} (the window&apos;s distinct nodes and the current node&apos;s children)</dd>
          <dt>bid above the reserve</dt>
          <dd className="mono">
            {s.bidders} (reserve {fmtNum(enc.reserve, 2)})
          </dd>
          <dt>seated</dt>
          <dd className="mono">
            {seated.length} of at most {enc.seats ?? "–"}
          </dd>
          <dt>clearing price</dt>
          <dd className="mono">{fmtNum(s.price, 4)} (the highest losing bid; 0 when nobody lost)</dd>
        </dl>
      </div>

      <div className="card">
        <h3>The equilibrium</h3>
        <dl className="kv">
          <dt>exploitability</dt>
          <dd className="mono">{fmtNum(s.exploitability, 4)}</dd>
          <dt>converged</dt>
          <dd className="mono">{yesNo(s.converged)}</dd>
          <dt>best-response cycle</dt>
          <dd className="mono">{s.cycle ? `length ${s.cycle} (teaches THINK)` : "none"}</dd>
          <dt>nobody spoke</dt>
          <dd className="mono">{yesNo(s.no_bidders)}</dd>
          <dt>triggers</dt>
          <dd>{triggerText(s) || <span className="muted">none: the meta-player is not asked</span>}</dd>
          {learned ? (
            <>
              <dt>Σφ</dt>
              <dd className="mono">{fmtSigned(phiSum, 4)} = v(N) − v(∅)</dd>
            </>
          ) : null}
        </dl>
      </div>

      <div className="card wide">
        <h3>The aggregate</h3>
        <p className="note">
          P(state) ∝ eps / S + Σ reputation × profile over the seated seats. Filled: the named state ({s.named}); ▾ the
          lattice&apos;s prediction ({s.predicted}); underlined: accepting. Margin (top1 − top2) {fmtNum(s.margin, 4)}.
        </p>
        <Strip values={s.aggregate} named={s.named} predicted={s.predicted} accepting={read.accepting} large label="the aggregate" />
        <p className="mono small-note">{fmtAggregate(s.aggregate, 5, 4)}</p>
      </div>

      <div className="card wide">
        <h3>The seats</h3>
        <div className="seat-grid">
          {seated.map((x) => (
            <SeatCard key={`${x.node}-${x.bucket}`} seat={x} named={s.named} predicted={s.predicted} accepting={read.accepting} learned={learned} />
          ))}
        </div>
        {seated.length === 0 ? <p className="muted">Nobody was seated: the naming fell back to the most reputable candidate&apos;s belief.</p> : null}
        {unseated.length ? (
          <>
            <h3>Not seated</h3>
            <div className="seat-grid">
              {unseated.map((x) => (
                <SeatCard key={`${x.node}-${x.bucket}`} seat={x} named={s.named} predicted={s.predicted} accepting={read.accepting} learned={learned} />
              ))}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
