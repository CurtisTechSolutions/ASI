/**
 * The dynamic window, read for display (../../SPEC-DynamicWindow.md, radixnet/window.py): the ladder of node
 * sizes a window walks - 32, 16, 8, 4 and back to 32 - as the servers compute it, so the Model settings card
 * can draw a ladder while it is being chosen, before anything is applied. What a step does to the graph is
 * the server's to say (POST /api/model/window/step); nothing here halves a node. Pure functions, checked by
 * ../test/window.test.mjs.
 */

/** Where the ladder starts and where it turns back (radixnet.window.DEFAULT_TOP / DEFAULT_FLOOR). */
export const DEFAULT_TOP = 32;
export const DEFAULT_FLOOR = 4;

/** The sizes a select offers: the binary number system, as far as a node label reasonably goes. */
export const SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

/** Whether a value is 1, 2, 4, 8, ... - a size in the binary number system. */
export function isPowerOfTwo(value) {
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
}

/** The ladder from `top` down to `floor` (radixnet.window.ladder); empty when the two are not a ladder. */
export function ladder(top, floor) {
  const t = Number(top);
  const f = Number(floor);
  if (!isPowerOfTwo(t) || !isPowerOfTwo(f) || f > t) return [];
  const out = [];
  for (let size = t; size >= f; size /= 2) out.push(size);
  return out;
}

/** The size after `size` on the ladder: half of it, or the top again from the floor; null off the ladder. */
export function nextSize(size, top, floor) {
  const rungs = ladder(top, floor);
  const at = rungs.indexOf(Number(size));
  if (at < 0) return null;
  return rungs[(at + 1) % rungs.length];
}

/** `size` kept on the ladder `top .. floor`, the way the servers keep it: above the top it is the top, below the floor the floor. */
export function clampSize(size, top, floor) {
  const rungs = ladder(top, floor);
  if (rungs.length === 0) return null;
  const n = Number(size);
  if (rungs.includes(n)) return n;
  if (!Number.isFinite(n)) return rungs[0];
  if (n > rungs[0]) return rungs[0];
  if (n < rungs[rungs.length - 1]) return rungs[rungs.length - 1];
  return rungs.find((rung) => rung <= n) ?? rungs[0];
}

/** One sentence for the window as GET /api/model/window reports it. */
export function describeWindow(window) {
  if (!window || typeof window !== "object") return "unknown";
  if (!window.on) return "off - compression is unbounded, and no node is halved";
  const rungs = Array.isArray(window.sizes) ? window.sizes.join(" → ") : `${window.top} … ${window.floor}`;
  const when = window.auto ? "stepping at the end of every training epoch" : "stepping by hand";
  return `on: ${rungs} ${window.units || "units"}, at ${window.size} (next ${window.next}), ${when}`;
}

/** What a step did, in one line (the `step` of POST /api/model/window/step). */
export function describeStep(step) {
  if (!step || typeof step !== "object") return "";
  const sizes = Array.isArray(step.sizes) ? step.sizes.join(", ") : String(step.from);
  return (
    `${step.steps} step${step.steps === 1 ? "" : "s"} at ${sizes}: ${step.merges} merge${step.merges === 1 ? "" : "s"}, ` +
    `${step.splits} split${step.splits === 1 ? "" : "s"}, nodes ${step.nodes_before} → ${step.nodes_after}, ` +
    `edges ${step.edges_before} → ${step.edges_after}; the window now stands at ${step.to}`
  );
}
