/** Pure helpers behind the Nim tab: the totals `POST /api/nim/play` and `/api/nim/babble` answer, as tiles. */

/** `[key, label, kind]` of the tiles, in order; kind "rate" is a share shown as a percentage. */
export const NIM_TILES = [
  ["win_rate", "wins", "rate", "against a random opponent; chance is 0.5"],
  ["legal_first_try", "legal first try", "rate", "the decoder's first proposal accepted by the referee"],
  ["refusals_per_move", "refusals / move", "number", "proposals the referee refused before one was accepted"],
  ["optimal_rate", "optimal moves", "rate", "moves to a zero nim-sum where one existed, of our moves"],
  ["abstained_per_move", "abstained / move", "number", "the release rule declined; a random legal move stood in"],
];

/** The tiles of a report: each `{key, label, value, title}` with the value written out. */
export function nimTiles(report) {
  const totals = (report && report.totals) || {};
  const tiles = NIM_TILES.map(([key, label, kind, title]) => {
    const v = Number(totals[key]);
    const value = !Number.isFinite(v) ? "–" : kind === "rate" ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);
    return { key, label, value, title };
  });
  const bits = Number(report && report.information_bits);
  tiles.push({
    key: "information_bits",
    label: "I(state; nim-sum)",
    value: Number.isFinite(bits) ? `${bits.toFixed(3)} bits` : "–",
    title: "what the named state says about whether the position is lost for the mover",
  });
  const used = report && report.states_used;
  tiles.push({ key: "states_used", label: "states used", value: used ?? "–", title: "states the encoder ever named" });
  const agreement = Number(report && report.agreement);
  tiles.push({
    key: "agreement",
    label: "agreement",
    value: Number.isFinite(agreement) ? agreement.toFixed(2) : "–",
    title: "the forward model's prediction against the encoder's next named state: a meta-feature, not credit",
  });
  return tiles;
}

/** One game in a line: "won, moved first, 7 plies: 1/3 legal at once, 4 refused, 1 optimal". */
export function recordLine(r) {
  if (!r) return "";
  const head = `${r.won ? "won" : "lost"}, moved ${r.first ? "first" : "second"}, ${r.plies} plies`;
  return `${head}: ${r.first_try}/${r.our_moves} legal at once, ${r.refusals} refused, ${r.optimal} optimal`;
}
