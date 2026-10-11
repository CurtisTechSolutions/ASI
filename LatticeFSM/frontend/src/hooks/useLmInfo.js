import { useEffect, useState } from "react";
import { api } from "../api.js";

/**
 * The chosen language model's info (GET /api/{m}): fetched again whenever the model is switched or any
 * panel has moved something (`tick`). Answers `[info, setInfo, error]`; a panel that gets a fresh info back
 * from a call (most do, as `model`) sets it straight away instead of waiting for the refetch.
 */
export function useLmInfo(lm, tick) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    api
      .lm(lm)
      .then((r) => {
        if (!alive) return;
        setInfo(r && typeof r === "object" ? r : null);
        setError(null);
      })
      .catch((err) => {
        if (!alive) return;
        setInfo(null);
        setError(err.message);
      });
    return () => {
      alive = false;
    };
  }, [lm, tick]);

  return [info, setInfo, error];
}

/** The model's info line: "8 nodes of chars · confidence 0.90 · 12345 tokens trained · 80 distinct". */
export function infoLine(lm, info) {
  if (!info) return "";
  if (lm === "chain") {
    return `${info.nodes} nodes (${info.nodes - 1} input nodes and the output node) of ${info.unit}s · confidence ${Number(
      info.confidence,
    ).toFixed(2)} · ${info.tokens} tokens trained · ${info.vocab} distinct · ${info.contexts} contexts · ${
      info.credited ?? 0
    } credited`;
  }
  return `${info.order}-grams of ${info.unit}s · ${info.tokens} tokens trained · ${info.vocab} distinct · ${info.ngrams} n-grams · ${
    info.credited_edges ?? 0
  } credited edges`;
}
