/** Background jobs (`runJob` in api.js): how one is described, in a line. */
import { fmtInt } from "./util.js";

/** "12.3 s", "2 min 05 s", "1 h 02 min" */
export function fmtElapsed(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${String(Math.floor(s % 60)).padStart(2, "0")} s`;
  return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} min`;
}

/** "job 3: body train" */
export function jobName(job) {
  if (!job) return "";
  return `job ${job.id}: ${[job.model, job.kind].filter(Boolean).join(" ")}`;
}

/**
 * What a job's two counters count: the server ticks once a string (`/api/train`: episodes of symbols) or once a
 * game (`/api/nim/...`: games of plies). `[sequences, tokens]` as words.
 */
export function jobUnits(job) {
  return job && job.model === "nim" ? ["games", "plies"] : ["episodes", "symbols"];
}

/** "job 3: body train - training even-b · 1,200 episodes · 4,000 symbols (5,200/s) · tenth 2 · 6.5 s" */
export function jobLine(job) {
  if (!job) return "";
  const p = job.progress || {};
  const [sequences, tokens] = jobUnits(job);
  const parts = [];
  const doing = [p.phase, p.label].filter(Boolean).join(" ");
  if (doing) parts.push(doing);
  if (p.sequences) parts.push(`${fmtInt(p.sequences)} ${sequences}`);
  if (p.tokens) parts.push(`${fmtInt(p.tokens)} ${tokens}${p.tokens_per_second ? ` (${fmtInt(p.tokens_per_second)}/s)` : ""}`);
  if (p.round) parts.push(`tenth ${fmtInt(p.round)}`);
  parts.push(fmtElapsed(job.elapsed));
  if (p.stopping && job.state === "running") parts.push("stopping…");
  return `${jobName(job)} - ${parts.join(" · ")}`;
}
