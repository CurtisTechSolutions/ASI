import { fmtInt, fmtNum } from "../util.js";

const WEIGHTING = ["bias", "seen", "recent", "net", "age", "width", "rate"];
const FEATURES = ["seen", "recent", "net", "age", "width"];

/** One edge, every field: what `GET /api/edge` answers, laid out as the record it is. */
export default function EdgeCard({ edge, symbol, onClose }) {
  if (!edge) return null;
  const when = (v) => (v === -1 || v === undefined ? "never" : fmtInt(v));
  const weighting = Array.isArray(edge.weighting) ? edge.weighting : [];
  const features = Array.isArray(edge.features) ? edge.features : [];
  return (
    <div className="card edge-card">
      <div className="toolbar">
        <h3>
          edge ({edge.source}, {symbol}, {edge.target})
        </h3>
        {onClose ? (
          <button type="button" className="link" onClick={onClose}>
            close
          </button>
        ) : null}
      </div>
      <div className="edge-grid">
        <dl className="kv">
          <dt>seen</dt>
          <dd>{fmtInt(edge.seen)}</dd>
          <dt>first seen</dt>
          <dd>{when(edge.first_seen)}</dd>
          <dt>last seen</dt>
          <dd>{when(edge.last_seen)}</dd>
          <dt>recent (trace)</dt>
          <dd>{fmtNum(edge.recent, 3)}</dd>
          <dt>age</dt>
          <dd>{fmtNum(edge.age, 3)}</dd>
        </dl>
        <dl className="kv">
          <dt>rewarded</dt>
          <dd className="ok">+{fmtNum(edge.rewarded, 3)}</dd>
          <dt>punished</dt>
          <dd className="bad">−{fmtNum(edge.punished, 3)}</dd>
          <dt>net per traversal</dt>
          <dd>{fmtNum(edge.net, 3)}</dd>
          <dt>last rewarded</dt>
          <dd>{when(edge.last_rewarded)}</dd>
          <dt>last punished</dt>
          <dd>{when(edge.last_punished)}</dd>
        </dl>
        <dl className="kv">
          <dt>width</dt>
          <dd>{fmtNum(edge.width, 3)}</dd>
          <dt>log-weight</dt>
          <dd>{fmtNum(edge.log_weight, 3)}</dd>
          <dt>weighting</dt>
          <dd className="mono">
            {WEIGHTING.map((name, i) => (
              <span key={name} className="coef" title={name}>
                {name} {fmtNum(weighting[i], 3)}
              </span>
            ))}
          </dd>
          <dt>features at last traversal</dt>
          <dd className="mono">
            {FEATURES.map((name, i) => (
              <span key={name} className="coef" title={name}>
                {name} {fmtNum(features[i], 3)}
              </span>
            ))}
          </dd>
        </dl>
      </div>
      <p className="note">
        The edge's log-weight is its own weighting applied to its features, plus the machine's stimulation times
        the log of its width. The weighting moves at every credit by rate × credit × feature; the width widens with
        use and reward, narrows with punishment, and relaxes toward 1 on the clock.
      </p>
    </div>
  );
}
