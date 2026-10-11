import { depthStyle, mean, probStyle, tokenPieces } from "../lm.js";
import { fmtNum } from "../util.js";

/**
 * Text written token by token, each tinted: by how deep the chain's walk went for it (`depths`, against
 * `maxDepth`, the number of input nodes), with the depth as a subscript when `subscripts` is on - or, when
 * `probs` are given, by the probability the model gave it (Score). Without either it is plain text.
 */
export default function TokenText({ tokens, depths, probs, unit, maxDepth = 8, subscripts = false }) {
  const pieces = tokenPieces(tokens, depths, unit);
  const p = Array.isArray(probs) ? probs : null;
  return (
    <span className="tokens">
      {pieces.map((piece, i) => {
        const lead = piece.text.match(/^\s*/)[0];
        const body = piece.text.slice(lead.length);
        const style = p ? probStyle(p[i]) : piece.depth !== null ? depthStyle(piece.depth, maxDepth) : undefined;
        const title = [
          p ? `p ${fmtNum(p[i] * 100, 2)}%` : null,
          piece.depth !== null ? `${piece.depth} node${piece.depth === 1 ? "" : "s"} deep` : null,
        ]
          .filter(Boolean)
          .join(" · ");
        const sub = subscripts && piece.depth !== null ? <sub>{piece.depth}</sub> : null;
        // a space of a model of letters is a token too: it is tinted where it stands
        if (!body) {
          return (
            <span key={i} className="tok" style={style} title={title || undefined}>
              {piece.text}
              {sub}
            </span>
          );
        }
        return (
          <span key={i}>
            {lead}
            <span className="tok" style={style} title={title || undefined}>
              {body}
              {sub}
            </span>
          </span>
        );
      })}
    </span>
  );
}

/** "mean depth 2.4 of 7" beside a reply, or nothing when the model gave no depths. */
export function DepthNote({ depths, maxDepth }) {
  const m = mean(depths);
  if (m === null) return null;
  return (
    <span className="muted">
      mean depth {fmtNum(m, 2)}
      {maxDepth ? ` of ${maxDepth}` : ""}
    </span>
  );
}
