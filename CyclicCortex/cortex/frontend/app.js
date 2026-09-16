/* CyclicCortex front end.
 *
 * The server does every piece of thinking: each button here is a call the
 * command line already makes. What the browser adds is the part that only
 * works live -- the similarity matrix as a picture rather than a table, the
 * learning curve while it is still learning, and p_valid and grade for every
 * legal move on the board while it is your turn.
 */

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined)
    n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
}

function sv(tag, attrs, ...kids) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined)
    n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
}

const fix = (x, n = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(n);

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch (e) { throw new Error(text.slice(0, 200)); }
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function say(id, text, kind) {
  const n = $(id);
  if (!n) return;
  n.textContent = text || "";
  n.className = "msg" + (kind ? " " + kind : "");
}

/* --------------------------------------------------------------- the state */

const S = {
  info: null, models: [], modelName: "main", model: null, checkpoints: [],
  job: null, jobEvents: [], jobSince: 0, jobTimer: null,
  credit: null, creditJob: null, transferJob: null,
  match: null, sel: null, digit: null, autoTimer: null,
  hover: null,
};

/* ------------------------------------------------------------------- boot */

async function boot() {
  S.info = await api("/api/info");
  $("badge-sf").textContent = S.info.stockfish.available
    ? "stockfish: " + S.info.stockfish.path : "stockfish: not installed";
  $("badge-sf").className = "badge " + (S.info.stockfish.available ? "good" : "");
  $("foot-info").textContent = `${S.info.name} ${S.info.version} · checkpoints in ${S.info.checkpoint_dir}`;

  fillGames();
  wire();
  await refreshState();
  await loadModel();
  newMatchFromForm().catch(() => {});
  setInterval(refreshState, 2500);
}

function games() { return S.info.games.map((g) => g.name); }
function gameInfo(n) { return S.info.games.find((g) => g.name === n); }

function fillGames() {
  for (const sel of [$("job-game"), $("play-game"), $("add-game")]) {
    sel.replaceChildren(...games().map((n) => el("option", { value: n }, n)));
  }
  $("job-game").value = "chess";
  $("play-game").value = "chess";
  $("build-games").replaceChildren(...games().map((n) =>
    el("label", { class: "inline" },
      el("input", { type: "checkbox", value: n, checked: true }), n)));
  for (const s of ["play-a", "play-b"]) {
    $(s).replaceChildren(...["human", "cortex", "bot", "stockfish", "none"].map(
      (k) => el("option", { value: k }, k)));
  }
  $("play-a").value = "human"; $("play-b").value = "cortex";
}

function wire() {
  for (const b of $("tabs").querySelectorAll("button")) {
    b.onclick = () => {
      for (const o of $("tabs").querySelectorAll("button")) o.classList.toggle("on", o === b);
      for (const t of document.querySelectorAll(".tab")) t.hidden = ("tab-" + b.dataset.tab) !== t.id;
    };
  }
  $("model").onchange = () => { S.modelName = $("model").value; loadModel(); };
  $("do-build").onclick = doBuild;
  $("do-add").onclick = doAdd;
  $("do-fork").onclick = doFork;
  $("do-save").onclick = doSave;
  $("do-load").onclick = doLoad;
  $("do-job").onclick = startJob;
  $("do-stop").onclick = stopJob;
  $("job-kind").onchange = jobKindChanged;
  $("do-credit").onclick = startCredit;
  $("do-transfer").onclick = startTransfer;
  $("do-new").onclick = () => newMatchFromForm();
  $("do-step").onclick = () => matchAct("/api/match/step", { n: 1 });
  $("do-run").onclick = () => matchAct("/api/match/run", {});
  $("do-pass").onclick = () => sendMove("pass");
  $("play-game").onchange = playersChanged;
  $("play-a").onchange = playersChanged;
  $("play-b").onchange = playersChanged;
  $("heat").onchange = () => renderMatch();
  $("auto").onchange = () => { if ($("auto").checked) autoStep(); };
  for (const b of document.querySelectorAll(".chip[data-preset]")) {
    b.onclick = () => preset(b.dataset.preset);
  }
  jobKindChanged();
  playersChanged();
}

/* ------------------------------------------------------------ shared state */

async function refreshState() {
  let st;
  try { st = await api("/api/state"); } catch (e) { return; }
  S.models = st.models; S.checkpoints = st.checkpoints;
  const names = st.models.map((m) => m.name);
  if (!names.includes(S.modelName)) S.modelName = names[0] || "main";
  const sel = $("model");
  if (sel.value !== S.modelName || sel.options.length !== names.length) {
    sel.replaceChildren(...names.map((n) => el("option", { value: n }, n)));
    sel.value = S.modelName;
  }
  $("ckpt-list").replaceChildren(...st.checkpoints.map((c) =>
    el("option", { value: c.file }, `${c.file} · ${(c.bytes / 1024).toFixed(0)} kB`)));
  const running = st.jobs.filter((j) => j.status === "running");
  const badge = $("badge-jobs");
  badge.textContent = running.length
    ? `${running.length} running: ${running.map((j) => j.kind).join(", ")}` : "idle";
  badge.className = "badge" + (running.length ? " live" : "");
  renderJobList(st.jobs);
  playerOptions();
}

async function loadModel() {
  try {
    S.model = await api("/api/model?name=" + encodeURIComponent(S.modelName));
  } catch (e) { say("build-msg", e.message, "err"); return; }
  const m = S.model;
  $("badge-regions").textContent =
    `${m.stats.regions} regions · ${Object.keys(m.route_of).length} games`;
  $("badge-metric").textContent = `triangle violations: ${m.graph.violations}`;
  $("badge-metric").className = "badge " + (m.graph.violations === 0 ? "good" : "bad");
  $("metric-note").textContent =
    `Zero violations is the claim; this model measures ${m.graph.violations}.`;
  renderGraph(); renderRegions(); renderRouting();
  const missing = games().filter((g) => !(g in m.route_of));
  $("add-game").replaceChildren(...missing.map((n) => el("option", { value: n }, n)));
  $("do-add").disabled = missing.length === 0;
}

/* ================================================================ 1. cortex */

/* Where to put the games on a plane, given only the distances between them.
 *
 * Classical scaling first -- double-centre the squared distances, take the two
 * leading eigenvectors by power iteration with a spectral shift, because
 * Jaccard is a proper metric but nothing promises it embeds in a PLANE and a
 * negative eigenvalue would otherwise be chased. Then stress majorisation
 * (SMACOF) from that start, which minimises the distance error directly and,
 * on the four games here, halves it: chess and checkers land 0.45 apart
 * instead of 0.06.
 *
 * Nothing about the layout is chosen. The residual error is REPORTED under the
 * map rather than hidden -- a plane cannot hold these four distances exactly,
 * and a picture that says otherwise would be the wrong picture. */
function classicalMds(D) {
  const n = D.length;
  const d2 = D.map((r) => r.map((v) => v * v));
  const rowm = d2.map((r) => r.reduce((a, b) => a + b, 0) / n);
  const gm = rowm.reduce((a, b) => a + b, 0) / n;
  const B = d2.map((r, i) => r.map((v, j) => -0.5 * (v - rowm[i] - rowm[j] + gm)));
  const axes = [];
  for (let a = 0; a < 2; a++) {
    let v = Array.from({ length: n }, (_, i) => Math.cos((i + 1) * (a + 1) * 1.7) + 0.013 * (i + 1));
    const nr = Math.hypot(...v) || 1;
    v = v.map((x) => x / nr);
    for (let it = 0; it < 400; it++) {
      const w = B.map((row) => row.reduce((s, x, j) => s + x * v[j], 0)).map((x, i) => x + 2 * v[i]);
      const norm = Math.hypot(...w);
      if (!norm || !Number.isFinite(norm)) break;
      v = w.map((x) => x / norm);
    }
    const lam = v.reduce((s, vi, i) => s + vi * B[i].reduce((t, x, j) => t + x * v[j], 0), 0);
    axes.push([v, lam]);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i][j] -= lam * v[i] * v[j];
  }
  return Array.from({ length: n }, (_, i) =>
    axes.map(([v, lam]) => v[i] * Math.sqrt(Math.max(lam, 0))));
}

function smacof(D, X, iters = 400) {
  const n = D.length;
  for (let it = 0; it < iters; it++) {
    const Y = [];
    for (let i = 0; i < n; i++) {
      let ax = 0, ay = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1];
        const d = Math.hypot(dx, dy) || 1e-9;
        ax += X[j][0] + D[i][j] * dx / d;
        ay += X[j][1] + D[i][j] * dy / d;
      }
      Y.push([ax / (n - 1), ay / (n - 1)]);
    }
    X = Y;
  }
  return X;
}

function mds(D) {
  const n = D.length;
  if (n === 0) return [];
  if (n === 1) return [[0, 0]];
  return smacof(D, classicalMds(D));
}

function mdsError(D, P) {
  let e = 0, k = 0;
  for (let i = 0; i < D.length; i++) for (let j = i + 1; j < D.length; j++) {
    e += Math.abs(Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) - D[i][j]); k++;
  }
  return k ? e / k : 0;
}

const REGION_COLOURS = ["#6ea8fe", "#f0a35e", "#66d19e", "#c78ae0", "#e8c26a", "#7ad3e8"];
const colourOf = (i) => REGION_COLOURS[i % REGION_COLOURS.length];

function renderGraph() {
  const g = S.model.graph, names = g.names, n = names.length;
  const region = {}, params = {};
  for (const r of S.model.regions) for (const gm of r.games) {
    region[gm] = r.id; params[gm] = (r.shape && r.shape.params) || 0;
  }
  // DESIGN.md 11: a node is sized by its region's parameter count, so the
  // picture carries how much network is behind each game as well as where it sits.
  const pmax = Math.max(1, ...Object.values(params));
  const radius = (nm) => 10 + 9 * Math.sqrt((params[nm] || 0) / pmax);
  const arrived = S.model.log.length ? S.model.log[S.model.log.length - 1].game : null;

  // --- the map
  const W = 460, H = 360, pad = 58;
  const pts = mds(g.matrix);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const spanX = Math.max(...xs) - Math.min(...xs) || 1;
  const spanY = Math.max(...ys) - Math.min(...ys) || 1;
  const sx = (v) => pad + (v - Math.min(...xs)) / spanX * (W - 2 * pad);
  const sy = (v) => pad + (v - Math.min(...ys)) / spanY * (H - 2 * pad);
  const kids = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const d = g.matrix[i][j];
    kids.push(sv("line", {
      x1: sx(xs[i]), y1: sy(ys[i]), x2: sx(xs[j]), y2: sy(ys[j]),
      stroke: d <= g.tau ? "#4d5670" : "#242a35", "stroke-width": Math.max(0.6, 4 * (1 - d)),
      opacity: d <= g.tau ? 0.55 : 0.2,
    }));
    kids.push(sv("text", {
      x: (sx(xs[i]) + sx(xs[j])) / 2, y: (sy(ys[i]) + sy(ys[j])) / 2 - 3,
      fill: "#6c7484", "font-size": 9, "text-anchor": "middle", "font-family": "monospace",
    }, d.toFixed(3)));
  }
  for (let i = 0; i < n; i++) {
    const nm = names[i], rr = radius(nm);
    if (nm === arrived) {
      const ring = sv("circle", { cx: sx(xs[i]), cy: sy(ys[i]), r: rr + 4,
        fill: "none", stroke: "#e6e9ef", "stroke-width": 2, opacity: 0.8 });
      ring.append(sv("animate", { attributeName: "r", values: `${rr + 3};${rr + 13};${rr + 3}`,
        dur: "2.4s", repeatCount: "indefinite" }));
      ring.append(sv("animate", { attributeName: "opacity", values: "0.75;0;0.75",
        dur: "2.4s", repeatCount: "indefinite" }));
      kids.push(ring);
    }
    kids.push(sv("circle", { cx: sx(xs[i]), cy: sy(ys[i]), r: rr,
      fill: colourOf(region[nm] || 0), opacity: 0.92 },
      sv("title", {}, `${nm}: region ${region[nm]}, ${params[nm]} parameters`)));
    kids.push(sv("text", { x: sx(xs[i]), y: sy(ys[i]) + rr + 17, fill: "#e6e9ef",
      "font-size": 12, "text-anchor": "middle" }, nm));
    kids.push(sv("text", { x: sx(xs[i]), y: sy(ys[i]) + 4, fill: "#0b1220",
      "font-size": 11, "text-anchor": "middle", "font-weight": "700" },
      "R" + (region[nm] ?? "?")));
  }
  $("graph-map").replaceChildren(sv("svg",
    { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, kids),
    el("p", { class: "hint" },
      `Mean error between a drawn distance and the real one: ${fix(mdsError(g.matrix, pts))}. `,
      "A plane cannot hold these distances exactly; the matrix is exact and this is the fit. ",
      "A node's size is its region's parameter count; the pulsing ring is the game that arrived last."));

  // --- the matrix
  const cell = 52, left = 76, top = 40;
  const mk = [];
  for (let j = 0; j < n; j++)
    mk.push(sv("text", { x: left + j * cell + cell / 2, y: top - 10, fill: "#98a0b0",
      "font-size": 11, "text-anchor": "middle" }, names[j].slice(0, 8)));
  for (let i = 0; i < n; i++) {
    mk.push(sv("text", { x: left - 8, y: top + i * cell + cell / 2 + 4, fill: "#98a0b0",
      "font-size": 11, "text-anchor": "end" }, names[i]));
    for (let j = 0; j < n; j++) {
      const d = g.matrix[i][j];
      mk.push(sv("rect", { x: left + j * cell, y: top + i * cell, width: cell - 2,
        height: cell - 2, rx: 4, fill: heatColour(1 - d), opacity: i === j ? 1 : 0.92 }));
      mk.push(sv("text", { x: left + j * cell + cell / 2 - 1,
        y: top + i * cell + cell / 2 + 4, "font-size": 11, "text-anchor": "middle",
        "font-family": "monospace", fill: 1 - d > 0.45 ? "#0b1220" : "#e6e9ef" },
        d.toFixed(2)));
    }
  }
  $("graph-matrix").replaceChildren(sv("svg", {
    viewBox: `0 0 ${left + n * cell + 10} ${top + n * cell + 10}`,
    width: left + n * cell + 10, height: top + n * cell + 10 }, mk));
}

/* blue (far) through to orange (near) -- one scale used by the matrix and by
 * the board heat map, so the same colour means the same thing everywhere. */
function heatColour(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[15, 22, 38], [42, 74, 122], [110, 168, 254], [240, 163, 94], [255, 236, 200]];
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function renderRegions() {
  const cards = S.model.regions.map((r) => {
    const total = Math.max(1, r.vocab_width);
    const bar = el("div", { class: "slots" }, r.slots.map((s, i) =>
      el("div", { class: "slot", style: `width:${s.width / total * 100}%;background:${colourOf(i)}`,
        title: `${s.mechanic}: slots ${s.offset}..${s.offset + s.width - 1}` })));
    const key = el("div", { class: "slotkey" }, r.slots.map((s, i) =>
      el("span", {}, el("i", { style: `background:${colourOf(i)}` }),
        `${s.mechanic} ×${s.width}`)));
    const grew = r.growth.filter((x) => x.grew === "hidden").length;
    const log = r.growth.length ? el("div", { class: "slotkey growth" },
      r.growth.slice(-6).map((x) =>
        el("span", {}, `${x.grew} +${x.by} → ${x.to} at ${x.at.toLocaleString()} samples`)))
      : null;
    return el("div", { class: "card" },
      el("h4", {}, `region ${r.id}`),
      el("div", { class: "games" }, r.games.join(", ") || "empty"),
      el("dl", {},
        el("dt", {}, "network"), el("dd", {}, r.shape.ni
          ? `${r.shape.ni} → ${r.shape.nh} → 2 · ${r.shape.params} params` : "none"),
        el("dt", {}, "rule"), el("dd", {}, r.rule),
        el("dt", {}, "vocabulary"), el("dd", {}, `${r.vocab_width} inputs / ${r.slots.length} mechanics`),
        el("dt", {}, "samples seen"), el("dd", {}, r.seen.toLocaleString()),
        el("dt", {}, "running loss"), el("dd", {}, r.loss === null ? "–" : fix(r.loss, 5)),
        el("dt", {}, "hidden growths"), el("dd", {}, String(grew)),
        el("dt", {}, "reputation"), el("dd", {}, fix(r.reputation))),
      bar, key, log);
  });
  $("regions").replaceChildren(...cards);
}

function renderRouting() {
  const rows = S.model.log.map((e) => el("tr", {},
    el("td", {}, e.game),
    el("td", {}, e.action),
    el("td", { class: "num" }, "R" + e.region),
    el("td", { class: "num" }, e.action === "joined" ? fix(e.distance)
      : (e.nearest === null || e.nearest === undefined ? "–" : fix(e.nearest))),
    el("td", { class: "num" }, "+" + (e.inputs_grown ?? 0))));
  $("routing").replaceChildren(el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "game"), el("th", {}, "action"),
      el("th", { class: "num" }, "region"), el("th", { class: "num" }, "distance"),
      el("th", { class: "num" }, "inputs grown"))),
    el("tbody", {}, rows)));
}

async function doBuild() {
  const picked = [...$("build-games").querySelectorAll("input:checked")].map((i) => i.value);
  if (!picked.length) return say("build-msg", "pick at least one game", "err");
  try {
    const name = $("build-name").value.trim() || "main";
    await api("/api/model/build", { name, games: picked,
      tau_new: Number($("build-tau").value), seed: Number($("build-seed").value),
      nh: Number($("build-nh").value) });
    S.modelName = name;
    await refreshState(); await loadModel();
    say("build-msg", `built ${name}: ${picked.join(", ")}`, "ok");
  } catch (e) { say("build-msg", e.message, "err"); }
}

async function doAdd() {
  try {
    const r = await api("/api/model/add", { name: S.modelName, game: $("add-game").value,
      rehearse: $("add-rehearse").checked });
    await loadModel();
    say("build-msg", `${$("add-game").value} → region ${r.region}` +
      (r.rehearse.samples ? `, rehearsed ${r.rehearse.rehearsed.join(" + ")} over ` +
        `${r.rehearse.samples} samples` : ""), "ok");
  } catch (e) { say("build-msg", e.message, "err"); }
}

async function doFork() {
  const to = $("fork-name").value.trim();
  if (!to) return say("build-msg", "a fork needs a name", "err");
  try {
    await api("/api/model/copy", { from: S.modelName, to });
    await refreshState();
    say("build-msg", `forked ${S.modelName} → ${to}; pick it as a player to have them meet`, "ok");
  } catch (e) { say("build-msg", e.message, "err"); }
}

async function doSave() {
  try {
    const r = await api("/api/model/save", { name: S.modelName,
      file: $("ckpt-name").value.trim() || S.modelName });
    await refreshState();
    say("build-msg", `saved ${r.file}, ${(r.bytes / 1024).toFixed(1)} kB`, "ok");
  } catch (e) { say("build-msg", e.message, "err"); }
}

async function doLoad() {
  const file = $("ckpt-list").value;
  if (!file) return say("build-msg", "no checkpoint on disk yet", "err");
  try {
    await api("/api/model/load", { name: S.modelName, file });
    await loadModel();
    say("build-msg", `loaded ${file} into ${S.modelName}`, "ok");
  } catch (e) { say("build-msg", e.message, "err"); }
}

/* ================================================================= 2. train */

const JOB_N_LABEL = { train: "Episodes", selfplay: "Games", rehearse: "Episodes",
                      distill: "Positions", evaluate: "Positions", benchmark: "Matches" };
const JOB_N_DEFAULT = { train: 400, selfplay: 40, rehearse: 200, distill: 150,
                        evaluate: 40, benchmark: 6 };

function jobKindChanged() {
  const k = $("job-kind").value;
  $("job-n-label").textContent = JOB_N_LABEL[k] || "N";
  $("job-n").value = JOB_N_DEFAULT[k] || 100;
  $("job-chunk").parentElement.hidden = k === "evaluate" || k === "benchmark";
  $("job-lr").parentElement.hidden = k === "evaluate" || k === "benchmark";
  $("job-k").parentElement.hidden = k === "benchmark";
}

function opponentSpec() {
  const v = $("job-opp").value;
  if (v === "none") return { kind: "none" };
  if (v === "stockfish") return { kind: "stockfish", skill: 5, depth: 6 };
  return { kind: "bot", depth: Number(v.split(":")[1]) };
}

async function startJob() {
  const kind = $("job-kind").value;
  const n = Number($("job-n").value);
  const body = { kind, model: S.modelName, game: $("job-game").value,
    chunk: Number($("job-chunk").value), k: Number($("job-k").value),
    lr: Number($("job-lr").value), seed: Number($("job-seed").value),
    opponent: opponentSpec() };
  if (kind === "selfplay") { body.rounds = n; body.opponent_depth = 1; }
  else if (kind === "distill") body.positions = n;
  else if (kind === "evaluate") { body.n = n; body.games = [$("job-game").value]; }
  else if (kind === "benchmark") {
    body.games = n;
    body.a = { kind: "cortex", model: S.modelName };
    body.b = opponentSpec().kind === "stockfish"
      ? { kind: "stockfish", skill: 5, depth: 6 }
      : { kind: "bot", depth: opponentSpec().depth ?? 1 };
  } else body.episodes = n;
  try {
    const r = await api("/api/jobs", body);
    S.job = r.job; S.jobEvents = []; S.jobSince = 0;
    $("job-log").textContent = ""; $("job-result").textContent = "";
    say("job-msg", `${r.job.id}: ${kind} started`, "ok");
    $("do-stop").disabled = false;
    pollJob();
  } catch (e) { say("job-msg", e.message, "err"); }
}

async function stopJob() {
  if (!S.job) return;
  try { await api("/api/job/stop", { id: S.job.id }); say("job-msg", "stopping after this chunk"); }
  catch (e) { say("job-msg", e.message, "err"); }
}

async function pollJob() {
  if (!S.job) return;
  clearTimeout(S.jobTimer);
  let r;
  try { r = await api(`/api/job?id=${S.job.id}&since=${S.jobSince}`); }
  catch (e) { say("job-msg", e.message, "err"); return; }
  const j = r.job;
  S.jobSince = j.events_total;
  S.jobEvents.push(...(j.events || []));
  S.job = j;
  renderJob(j);
  if (j.status === "running" || j.status === "queued") {
    S.jobTimer = setTimeout(pollJob, 600);
  } else {
    $("do-stop").disabled = true;
    say("job-msg", `${j.id}: ${j.status}` + (j.error ? ` — ${j.error}` : ""),
      j.status === "error" ? "err" : "ok");
    loadModel();
  }
}

function renderJob(j) {
  const pct = j.progress.total ? Math.round(100 * j.progress.done / j.progress.total) : 0;
  $("job-bar").style.width = pct + "%";
  $("job-progress").textContent =
    `${j.kind} · ${j.progress.done}/${j.progress.total || "?"} · ${j.status}` +
    (j.stopping ? " (stopping)" : "");
  const lines = S.jobEvents.filter((e) => e.type === "log" || e.type === "error")
    .map((e) => `${String(e.at).padStart(7)}s  ${e.text || e.error}`);
  const metrics = S.jobEvents.filter((e) => e.type === "metric");
  for (const m of metrics.slice(-3)) {
    lines.push(`${String(m.at).padStart(7)}s  ` + Object.entries(m.values)
      .map(([k, v]) => `${k}=${typeof v === "number" ? fix(v) : v}`).join("  "));
  }
  $("job-log").textContent = lines.slice(-40).join("\n");
  if (j.result) $("job-result").textContent = JSON.stringify(j.result, null, 1).slice(0, 4000);
  renderChart(metrics);
}

function renderChart(metrics) {
  const box = $("job-chart");
  if (!metrics.length) { box.replaceChildren(el("p", { class: "hint" },
    "Metrics appear after the first chunk.")); return; }
  const keys = Object.keys(metrics[metrics.length - 1].values)
    .filter((k) => typeof metrics[metrics.length - 1].values[k] === "number");
  const W = 860, H = 300, l = 52, r = 190, t = 18, b = 38;
  const xs = metrics.map((m) => m.step);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const rate = keys.filter((k) => !["hidden", "samples", "moves_scored", "skipped",
    "a_wins", "b_wins", "drawn"].includes(k));
  const counts = keys.filter((k) => !rate.includes(k));
  const vals = (k) => metrics.map((m) => m.values[k]).filter((v) => typeof v === "number");
  const rmax = Math.max(1, ...rate.flatMap(vals).map(Math.abs));
  const cmax = Math.max(1, ...counts.flatMap(vals));
  const sx = (v) => l + (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0)) * (W - l - r);
  const sy = (v, k) => H - b - (counts.includes(k) ? v / cmax : (v / rmax)) * (H - t - b);
  const kids = [];
  for (let i = 0; i <= 4; i++) {
    const y = t + i * (H - t - b) / 4;
    kids.push(sv("line", { x1: l, y1: y, x2: W - r, y2: y, stroke: "#242a35" }));
    kids.push(sv("text", { x: l - 8, y: y + 4, fill: "#6c7484", "font-size": 10,
      "text-anchor": "end", "font-family": "monospace" },
      (rmax * (1 - i / 4)).toFixed(2)));
  }
  kids.push(sv("text", { x: (l + W - r) / 2, y: H - 5, fill: "#6c7484", "font-size": 11,
    "text-anchor": "middle" }, "episodes completed"));
  for (const v of (x0 === x1 ? [x0] : [x0, x1]))
    kids.push(sv("text", { x: sx(v), y: H - b + 14, fill: "#6c7484", "font-size": 10,
      "text-anchor": "middle", "font-family": "monospace" }, String(v)));
  keys.forEach((k, i) => {
    const pts = metrics.filter((m) => typeof m.values[k] === "number")
      .map((m) => `${sx(m.step)},${sy(m.values[k], k)}`).join(" ");
    kids.push(sv("polyline", { points: pts, fill: "none", stroke: colourOf(i),
      "stroke-width": 2, "stroke-dasharray": counts.includes(k) ? "4 3" : null }));
    metrics.forEach((m) => {
      if (typeof m.values[k] === "number")
        kids.push(sv("circle", { cx: sx(m.step), cy: sy(m.values[k], k), r: 2.5,
          fill: colourOf(i) }));
    });
    const last = metrics[metrics.length - 1].values[k];
    const shown = typeof last !== "number" ? last
      : (counts.includes(k) ? String(Math.round(last)) : fix(last));
    kids.push(sv("text", { x: W - r + 8, y: t + 14 + i * 17, fill: colourOf(i),
      "font-size": 11, "font-family": "monospace" }, `${k} ${shown}`));
  });
  box.replaceChildren(sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, kids),
    el("p", { class: "hint" },
      "Solid lines read against the left axis; dashed lines are counts, scaled to their own maximum."));
}

function renderJobList(jobs) {
  const rows = jobs.slice(0, 12).map((j) => el("tr", {},
    el("td", {}, j.id),
    el("td", {}, j.kind),
    el("td", {}, (j.params.game || j.params.games || "") + ""),
    el("td", {}, j.params.model || "main"),
    el("td", { class: "num" }, `${j.progress.done}/${j.progress.total || "?"}`),
    el("td", { class: j.status === "error" ? "neg" : (j.status === "done" ? "pos" : "") },
      j.status),
    el("td", {}, el("button", { class: "chip", onclick: () => {
      S.job = j; S.jobSince = 0; S.jobEvents = []; pollJob();
      $("do-stop").disabled = j.status !== "running";
    } }, "view"))));
  $("job-list").replaceChildren(el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "job"), el("th", {}, "kind"),
      el("th", {}, "game"), el("th", {}, "model"), el("th", { class: "num" }, "progress"),
      el("th", {}, "status"), el("th", {}, ""))),
    el("tbody", {}, rows)));
}

/* ================================================================== 3. play */

function playersChanged() {
  const g = gameInfo($("play-game").value);
  $("lab-a").textContent = g.side_names[g.sides[0]];
  $("lab-b").textContent = g.side_names[g.sides[1]];
  playerOptions();
}

function playerOptions() {
  for (const [who, optId] of [["play-a", "play-a-opt"], ["play-b", "play-b-opt"]]) {
    const kind = $(who).value, sel = $(optId);
    let opts = [];
    if (kind === "cortex") opts = S.models.map((m) => [m.name, "model " + m.name]);
    else if (kind === "bot") opts = [["1", "depth 1"], ["2", "depth 2"], ["0", "random legal"]];
    else if (kind === "stockfish") opts = [["1", "skill 1"], ["5", "skill 5"],
      ["10", "skill 10"], ["20", "skill 20"]];
    const keep = sel.value;
    sel.replaceChildren(...opts.map(([v, t]) => el("option", { value: v }, t)));
    if (opts.some(([v]) => v === keep)) sel.value = keep;
    sel.disabled = opts.length === 0;
    sel.parentElement.hidden = opts.length === 0;
  }
}

function playerSpec(who, optId) {
  const kind = $(who).value, v = $(optId).value;
  if (kind === "cortex") return { kind, model: v || "main" };
  if (kind === "bot") return { kind, depth: Number(v || 1) };
  if (kind === "stockfish") return { kind, skill: Number(v || 5), depth: 6 };
  return { kind };
}

function preset(name) {
  const map = { "human-model": ["human", "cortex"], "model-model": ["cortex", "cortex"],
                "bot-model": ["bot", "cortex"], "human-bot": ["human", "bot"] };
  const [a, b] = map[name];
  $("play-a").value = a; $("play-b").value = b;
  playersChanged();
  newMatchFromForm();
}

async function newMatchFromForm() {
  const game = $("play-game").value;
  const g = gameInfo(game);
  const players = {};
  players[g.sides[0]] = playerSpec("play-a", "play-a-opt");
  players[g.sides[1]] = playerSpec("play-b", "play-b-opt");
  try {
    const r = await api("/api/match", { game, players, seed: Number($("play-seed").value),
      spectator: S.modelName });
    S.match = r.match; S.sel = null; S.digit = null;
    say("play-msg", `${players[g.sides[0]].kind} vs ${players[g.sides[1]].kind} · ${game}`, "ok");
    renderMatch();
    autoStep();
  } catch (e) { say("play-msg", e.message, "err"); }
}

async function refreshMatch() {
  if (!S.match) return;
  try {
    const r = await api("/api/match?id=" + encodeURIComponent(S.match.id));
    S.match = r.match;
    renderMatch();
  } catch (e) { /* the match may have been rotated away; leave the board as it is */ }
}

async function matchAct(path, body) {
  if (!S.match) return;
  try {
    const r = await api(path, { id: S.match.id, ...body });
    S.match = r.match; S.sel = null;
    renderMatch();
    autoStep();
  } catch (e) { say("play-msg", e.message, "err"); }
}

async function sendMove(key) {
  if (!S.match) return;
  try {
    const r = await api("/api/match/move", { id: S.match.id, key, reply: !$("auto").checked });
    S.match = r.match; S.sel = null; S.digit = null;
    renderMatch();
    autoStep();
  } catch (e) { say("play-msg", e.message, "err"); }
}

/* Training from the board: the same job the Training tab starts, with the
 * numbers the README's runs use, so a game that begins against an untrained
 * region can become a real game without leaving the page. */
async function quickTrain(game) {
  const episodes = game === "sudoku" ? 60 : 300;
  try {
    const r = await api("/api/jobs", { kind: "train", model: S.modelName, game,
      episodes, chunk: 50, k: 12, eval_n: 30,
      opponent: game === "sudoku" ? { kind: "none" } : { kind: "bot", depth: 1 } });
    say("play-msg", `training ${game} on ${S.modelName}: ${episodes} episodes…`);
    const tick = async () => {
      const j = (await api(`/api/job?id=${r.job.id}&since=0`)).job;
      say("play-msg", `training ${game}: ${j.progress.done}/${j.progress.total} · ${j.status}`
        + (j.error ? ` — ${j.error}` : ""), j.status === "error" ? "err" : "");
      if (j.status === "running" || j.status === "queued") return void setTimeout(tick, 700);
      if (j.result && j.result.eval)
        say("play-msg", `${game}: legality ${fix(j.result.eval.legality_acc)} ` +
          `against a ${fix(j.result.eval.majority_baseline)} baseline, top pick legal ` +
          `${fix(j.result.eval.top_choice_legal)}. Start a new game to play it.`, "ok");
      await loadModel();
      refreshMatch();
    };
    tick();
  } catch (e) { say("play-msg", e.message, "err"); }
}

function autoStep() {
  clearTimeout(S.autoTimer);
  const m = S.match;
  if (!m || m.over || !$("auto").checked || m.waiting_for_human) return;
  S.autoTimer = setTimeout(() => matchAct("/api/match/step", { n: 1 }), 350);
}

/* --- the board ----------------------------------------------------------- */

/* The SOLID glyphs for both sides, coloured by side. The outline set (♔♕♖♗♘♙)
 * renders as a hollow shape on a dark board and white then looks like a ghost of
 * a piece rather than a piece. */
const GLYPH = { K: "♚", Q: "♛", R: "♜", B: "♝", N: "♞", P: "♟" };
const GO_FILES = "ABCDEFGHJ";

/* The rule score of every legal move, rescaled to 0..1 for colour, plus the
 * handful the board marks when nothing is selected. Marking all thirty legal
 * chess moves at once paints the board and says nothing; the top few say where
 * the model wants to go. */
const HEAT_TOP = 10;

function scoreMap() {
  const a = S.match && S.match.analysis;
  if (!a || !$("heat").checked) return null;
  const vals = a.moves.map((m) => m.score).filter((v) => v > -50);
  if (!vals.length) return null;
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const v = new Map();
  for (const m of a.moves) v.set(m.key, m.score > -50 ? (m.score - lo) / span : 0);
  const top = new Set(a.moves.slice(0, HEAT_TOP).map((m) => m.key));
  return { v, top };
}

function heatMark(key, heat, force) {
  if (!heat || !heat.v.has(key)) return null;
  if (!force && !heat.top.has(key)) return null;
  return heat.v.get(key);
}

function renderMatch() {
  const m = S.match;
  if (!m) return;
  const heat = scoreMap();
  const legal = new Map(m.legal.map((x) => [x.key, x]));
  const last = m.history.length ? m.history[m.history.length - 1].move : null;
  const fn = { chess: boardSquares, checkers: boardSquares, go: boardGo,
               sudoku: boardSudoku }[m.game];
  $("board").replaceChildren(fn(m, legal, heat, last));
  $("do-pass").hidden = m.game !== "go" || !m.waiting_for_human;
  $("digits").hidden = m.game !== "sudoku" || !m.waiting_for_human;
  if (m.game === "sudoku") renderDigits(m, legal);
  renderStatus(m);
  renderThink(m);
  renderMoves(m);
  $("do-step").disabled = m.over || m.waiting_for_human;
  $("do-run").disabled = m.over || m.waiting_for_human;
}

function boardSquares(m, legal, heat, last) {
  const N = m.board.w, C = 62, pad = 22, W = N * C + 2 * pad;
  const kids = [];
  const px = (x) => pad + x * C, py = (y) => pad + (N - 1 - y) * C;
  const from = S.sel;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const dark = (x + y) % 2 === 0;
    kids.push(sv("rect", { x: px(x), y: py(y), width: C, height: C,
      fill: dark ? "#2b3040" : "#3b4255" }));
  }
  if (last && last.from && last.to) for (const p of [last.from, last.to])
    kids.push(sv("rect", { x: px(p[0]), y: py(p[1]), width: C, height: C,
      fill: "#e8c26a", opacity: 0.16 }));
  // the moves available from the selected square, or from every square
  for (const [key, mv] of legal) {
    if (!mv.from) continue;
    if (from && (mv.from[0] !== from[0] || mv.from[1] !== from[1])) continue;
    const h = heatMark(key, heat, !!from);
    if (h !== null && h !== undefined) {
      // Size AND colour carry the score, so the move it wants stands out of a
      // board where every square has some legal move landing on it.
      kids.push(sv("circle", { cx: px(mv.to[0]) + C / 2, cy: py(mv.to[1]) + C / 2,
        r: C * (0.10 + 0.22 * h), fill: heatColour(h), opacity: 0.45 + 0.4 * h }));
    }
    if (from)
      kids.push(sv("circle", { cx: px(mv.to[0]) + C / 2, cy: py(mv.to[1]) + C / 2,
        r: 7, fill: "none", stroke: "#e6e9ef", "stroke-width": 2, opacity: 0.75 }));
  }
  if (from) kids.push(sv("rect", { x: px(from[0]), y: py(from[1]), width: C, height: C,
    fill: "none", stroke: "#6ea8fe", "stroke-width": 3 }));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const v = m.board.cells[y][x];
    if (!v) continue;
    if (m.game === "chess") {
      kids.push(sv("text", { x: px(x) + C / 2, y: py(y) + C / 2 + 14, "font-size": 42,
        "text-anchor": "middle", fill: v[0] === "w" ? "#f4f6fa" : "#12151c",
        stroke: v[0] === "w" ? "#0b0d12" : "#7d879b", "stroke-width": 1.1,
        "paint-order": "stroke" }, GLYPH[v[1]] || v[1]));
    } else {
      const white = v.toLowerCase() === "w";
      kids.push(sv("circle", { cx: px(x) + C / 2, cy: py(y) + C / 2, r: C * 0.34,
        fill: white ? "#e8ecf4" : "#14181f", stroke: white ? "#9aa3b5" : "#4a5266",
        "stroke-width": 2 }));
      if (v === v.toUpperCase())
        kids.push(sv("text", { x: px(x) + C / 2, y: py(y) + C / 2 + 6, "font-size": 20,
          "text-anchor": "middle", fill: white ? "#4d5670" : "#e8c26a",
          "font-weight": "700" }, "K"));
    }
  }
  for (let i = 0; i < N; i++) {
    kids.push(sv("text", { x: px(i) + C / 2, y: pad + N * C + 15, fill: "#6c7484",
      "font-size": 11, "text-anchor": "middle" }, "abcdefgh"[i]));
    kids.push(sv("text", { x: pad - 8, y: py(i) + C / 2 + 4, fill: "#6c7484",
      "font-size": 11, "text-anchor": "end" }, i + 1));
  }
  const svgn = sv("svg", { viewBox: `0 0 ${W} ${W}`, width: W, height: W }, kids);
  svgn.addEventListener("click", (ev) => {
    const p = pointAt(svgn, ev, pad, C, N, true);
    if (!p) return;
    clickSquare(m, legal, p);
  });
  return svgn;
}

function pointAt(svgn, ev, pad, C, N, flip) {
  const r = svgn.getBoundingClientRect();
  const scale = r.width / svgn.viewBox.baseVal.width;
  const x = Math.floor(((ev.clientX - r.left) / scale - pad) / C);
  const yr = Math.floor(((ev.clientY - r.top) / scale - pad) / C);
  const y = flip ? N - 1 - yr : yr;
  if (x < 0 || x >= N || y < 0 || y >= N) return null;
  return [x, y];
}

function clickSquare(m, legal, p) {
  if (!m.waiting_for_human) return;
  if (S.sel) {
    const key = `${S.sel[0]},${S.sel[1]}-${p[0]},${p[1]}`;
    if (legal.has(key)) return sendMove(key);
  }
  const any = [...legal.values()].some((mv) => mv.from &&
    mv.from[0] === p[0] && mv.from[1] === p[1]);
  S.sel = any ? p : null;
  renderMatch();
}

function boardGo(m, legal, heat, last) {
  const N = m.board.w, C = 56, pad = 34, W = (N - 1) * C + 2 * pad;
  const px = (x) => pad + x * C, py = (y) => pad + (N - 1 - y) * C;
  const kids = [sv("rect", { x: 0, y: 0, width: W, height: W, rx: 8, fill: "#3a3324" })];
  for (let i = 0; i < N; i++) {
    kids.push(sv("line", { x1: px(0), y1: py(i), x2: px(N - 1), y2: py(i),
      stroke: "#0d0b07", "stroke-width": 1.2 }));
    kids.push(sv("line", { x1: px(i), y1: py(0), x2: px(i), y2: py(N - 1),
      stroke: "#0d0b07", "stroke-width": 1.2 }));
  }
  for (const [sx, sy2] of [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]])
    kids.push(sv("circle", { cx: px(sx), cy: py(sy2), r: 3.5, fill: "#0d0b07" }));
  for (const [key, mv] of legal) {
    if (!mv.point) continue;
    const h = heatMark(key, heat);
    if (h === null || h === undefined) continue;
    // Clearly smaller than a stone (0.45C): at high scores the heat colour is a
    // pale cream, and a mark the size of a stone would read as one.
    kids.push(sv("circle", { cx: px(mv.point[0]), cy: py(mv.point[1]),
      r: C * (0.07 + 0.19 * h), fill: heatColour(h), opacity: 0.6 + 0.35 * h }));
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const v = m.board.cells[y][x];
    if (!v) continue;
    kids.push(sv("circle", { cx: px(x), cy: py(y), r: C * 0.45,
      fill: v === "b" ? "#15181e" : "#eef1f6",
      stroke: v === "b" ? "#000" : "#a9b0c0", "stroke-width": 1.5 }));
  }
  if (last && last.point)
    kids.push(sv("circle", { cx: px(last.point[0]), cy: py(last.point[1]), r: 7,
      fill: "none", stroke: "#f0a35e", "stroke-width": 2.5 }));
  for (let i = 0; i < N; i++) {
    kids.push(sv("text", { x: px(i), y: pad + (N - 1) * C + 22, fill: "#b9ae92",
      "font-size": 11, "text-anchor": "middle" }, GO_FILES[i]));
    kids.push(sv("text", { x: pad - 14, y: py(i) + 4, fill: "#b9ae92", "font-size": 11,
      "text-anchor": "end" }, i + 1));
  }
  const svgn = sv("svg", { viewBox: `0 0 ${W} ${W}`, width: W, height: W }, kids);
  svgn.addEventListener("click", (ev) => {
    if (!m.waiting_for_human) return;
    const r = svgn.getBoundingClientRect();
    const scale = r.width / W;
    const x = Math.round(((ev.clientX - r.left) / scale - pad) / C);
    const yr = Math.round(((ev.clientY - r.top) / scale - pad) / C);
    const y = N - 1 - yr;
    if (x < 0 || x >= N || y < 0 || y >= N) return;
    const key = `${x},${y}`;
    if (legal.has(key)) sendMove(key);
  });
  return svgn;
}

function boardSudoku(m, legal, heat, last) {
  const N = 9, C = 54, pad = 14, W = N * C + 2 * pad;
  const px = (x) => pad + x * C, py = (y) => pad + y * C;
  const kids = [sv("rect", { x: 0, y: 0, width: W, height: W, fill: "#20242e", rx: 6 })];
  const best = new Map();
  for (const [key, mv] of legal) {
    const h = heatMark(key, heat, true);
    if (h === null || h === undefined) continue;
    const k = mv.cell.join(",");
    if (!best.has(k) || best.get(k) < h) best.set(k, h);
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const h = best.get(`${x},${y}`);
    kids.push(sv("rect", { x: px(x), y: py(y), width: C, height: C,
      fill: h === undefined ? "#262b36" : heatColour(h),
      opacity: h === undefined ? 1 : 0.5, stroke: "#151922" }));
    const v = m.board.cells[y][x];
    if (v) kids.push(sv("text", { x: px(x) + C / 2, y: py(y) + C / 2 + 8, "font-size": 24,
      "text-anchor": "middle", "font-weight": m.board.given[y][x] ? "700" : "400",
      fill: m.board.given[y][x] ? "#e6e9ef" : "#6ea8fe" }, v));
  }
  for (let i = 0; i <= N; i += 3) {
    kids.push(sv("line", { x1: px(i), y1: pad, x2: px(i), y2: pad + N * C,
      stroke: "#8b93a5", "stroke-width": 2 }));
    kids.push(sv("line", { x1: pad, y1: py(i), x2: pad + N * C, y2: py(i),
      stroke: "#8b93a5", "stroke-width": 2 }));
  }
  if (S.sel) kids.push(sv("rect", { x: px(S.sel[0]), y: py(S.sel[1]), width: C, height: C,
    fill: "none", stroke: "#6ea8fe", "stroke-width": 3 }));
  if (last && last.cell) kids.push(sv("rect", { x: px(last.cell[0]) + 2,
    y: py(last.cell[1]) + 2, width: C - 4, height: C - 4, fill: "none",
    stroke: "#f0a35e", "stroke-width": 2 }));
  const svgn = sv("svg", { viewBox: `0 0 ${W} ${W}`, width: W, height: W }, kids);
  svgn.addEventListener("click", (ev) => {
    if (!m.waiting_for_human) return;
    const p = pointAt(svgn, ev, pad, C, N, false);
    if (!p) return;
    S.sel = p; renderMatch();
  });
  return svgn;
}

function renderDigits(m, legal) {
  const box = $("digits");
  const here = S.sel;
  const ok = new Set();
  if (here) for (const [key, mv] of legal)
    if (mv.cell[0] === here[0] && mv.cell[1] === here[1]) ok.add(mv.value);
  box.replaceChildren(
    el("span", { class: "msg" }, here
      ? `r${here[1] + 1}c${here[0] + 1}: ${ok.size} legal value${ok.size === 1 ? "" : "s"}`
      : "pick a cell"),
    ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((v) => el("button", {
      class: "secondary" + (ok.has(v) ? "" : " off"), disabled: !ok.has(v),
      onclick: () => sendMove(`${here[0]},${here[1]}=${v}`) }, String(v))));
}

function renderStatus(m) {
  const b = m.board;
  const bits = [];
  if (m.over) {
    const r = m.result;
    bits.push(el("b", { class: r.winner ? "won" : "" },
      r.winner_name ? `${r.winner_name} ${r.natural ? "wins" : "leads"}` : "drawn"),
      ` — ${r.reason}, ${r.detail}, ${r.plies} plies. `);
  } else {
    bits.push(el("b", {}, `${m.turn_name} to move`),
      ` — ${m.to_move.name}. ply ${m.plies + 1} of at most ${m.max_plies}. `);
  }
  if (m.game === "chess") bits.push(`material (white) ${b.material.w}. `,
    b.in_check ? el("b", { class: "lost" }, "in check. ") : "");
  if (m.game === "checkers") bits.push(`pieces ${b.counts.w}–${b.counts.b}. `,
    b.forced ? "a jump is available, so only jumps are legal. " : "");
  if (m.game === "go") bits.push(`area ${b.score.b}–${b.score.w}, ${b.passes} consecutive pass(es). `);
  if (m.game === "sudoku") bits.push(`${b.empties} cells empty. `);
  for (const s of m.sides) {
    const st = m.stats[s];
    if (m.players[s].kind === "cortex")
      bits.push(el("span", {}, `${m.side_names[s]} played `,
        el("b", { class: st.illegal ? "lost" : "won" },
          `${st.illegal}/${st.tried} illegal (${fix(st.illegal_rate)})`), ". "));
  }
  $("play-status").replaceChildren(...bits);
}

function renderThink(m) {
  const a = m.analysis;
  const box = $("think");
  if (!a) { box.replaceChildren(el("p", { class: "hint" },
    m.over ? "The game is over." : "No model is watching this position.")); return; }
  const untrained = a.trained ? null : el("p", { class: "hint" },
    el("b", {}, `Region ${a.region} has seen nothing. `),
    "It is ranking these moves from its random initialisation, which is not a weak " +
    "model but no model. ",
    el("button", { class: "chip", onclick: () => quickTrain(m.game) },
      `train it on ${m.game} now`));
  const head = el("div", { class: "head" },
    el("span", { class: "badge" }, `model ${a.model}`),
    el("span", { class: "badge" }, `region ${a.region}`),
    el("span", { class: "badge" }, `rule ${a.rule}`),
    el("span", { class: "badge" }, `${a.shape.ni}→${a.shape.nh}`),
    el("span", { class: "badge" }, `${a.legal_moves} legal`),
    el("span", { class: "badge " + (a.playing ? "" : "live") },
      a.playing ? "playing" : "watching"),
    el("span", { class: "badge " + (a.trained ? "good" : "bad") },
      a.trained ? `${a.seen.toLocaleString()} samples seen` : "untrained"));
  const honest = a.honest ? el("p", { class: "hint" },
    "Its own candidate sampling picks ", el("b", {}, a.honest.label), " — ",
    a.honest_legal ? el("span", { class: "pos" }, "legal")
                   : el("span", { class: "neg" }, "NOT legal, so the best legal grade is substituted"),
    ". The ranking below is over legal moves only.") : null;
  const rows = a.moves.slice(0, 14).map((r) => el("tr",
    { class: r.pick ? "pick" : "",
      onmouseenter: () => { S.hover = r.key; },
      onclick: () => { if (m.waiting_for_human && !m.over) sendMove(r.key); } },
    el("td", {}, r.label),
    el("td", { class: "num" }, fix(r.pv)),
    el("td", { class: "num" }, fix(r.grade)),
    el("td", { class: "num" }, r.score < -50 ? "—" : fix(r.score))));
  box.replaceChildren(...[head, untrained, honest,
    el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "move"),
      el("th", { class: "num" }, "p_valid"), el("th", { class: "num" }, "grade"),
      el("th", { class: "num" }, "rule score"))), el("tbody", {}, rows)),
    el("div", { class: "legendbar" }, "low",
      el("div", { class: "grad", style:
        `background:linear-gradient(90deg,${heatColour(0)},${heatColour(0.25)},${heatColour(0.5)},${heatColour(0.75)},${heatColour(1)})` }),
      "high — the board heat map is this column")].filter(Boolean));
}

function renderMoves(m) {
  const rows = m.history.slice().reverse().map((h) => el("tr", {},
    el("td", {}, h.ply),
    el("td", { class: h.by === "human" ? "human" : "" }, h.side_name),
    el("td", {}, h.label),
    el("td", { class: h.illegal ? "illegal" : "" },
      h.illegal ? `illegal pick ${h.intended ? h.intended.label : ""} → substituted` : h.by)));
  $("moves").replaceChildren(el("table", {}, el("tbody", {}, rows)));
}

/* ================================================================ 4. credit */

async function startCredit() {
  try {
    const r = await api("/api/jobs", { kind: "credit", model: S.modelName,
      states: Number($("credit-states").value), k: Number($("credit-k").value) });
    S.creditJob = r.job;
    pollCredit();
  } catch (e) { $("credit-progress").textContent = e.message; }
}

async function pollCredit() {
  if (!S.creditJob) return;
  const r = await api(`/api/job?id=${S.creditJob.id}&since=0`);
  const j = r.job;
  const pct = j.progress.total ? Math.round(100 * j.progress.done / j.progress.total) : 0;
  $("credit-bar").style.width = pct + "%";
  $("credit-progress").textContent = `${j.progress.done}/${j.progress.total || "?"} · ${j.status}`
    + (j.error ? ` — ${j.error}` : "");
  if (j.status === "running" || j.status === "queued") return void setTimeout(pollCredit, 700);
  if (j.result) { S.credit = j.result; renderCredit(); }
}

function renderCredit() {
  const c = S.credit;
  const names = Object.keys(c.shapley);
  $("coverage").replaceChildren(
    el("h3", {}, "Coverage — what a region can encode at all"),
    el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "region"),
        ...names.map((n) => el("th", { class: "num" }, n)))),
      el("tbody", {}, c.coverage.map((r) => el("tr", {},
        el("td", {}, `R${r.region} · ${r.games.join(", ")}`),
        ...names.map((n) => el("td", { class: "num " + cls(r.coverage[n]) },
          fix(r.coverage[n], 2))))))));

  const blocks = names.map((n) => {
    const s = c.shapley[n];
    const ids = Object.keys(s.phi).sort();
    const max = Math.max(0.001, ...ids.map((i) => Math.abs(s.phi[i])));
    const gutter = 150, W = 470, rowh = 28, H = ids.length * rowh + 16;
    const mid = gutter + (W - gutter - 60) / 2;     // zero sits mid-plot, not mid-card
    const arm = (W - gutter - 60) / 2;
    const kids = [sv("line", { x1: mid, y1: 4, x2: mid, y2: H - 8, stroke: "#3a4150" })];
    ids.forEach((i, k) => {
      const v = s.phi[i], w = Math.abs(v) / max * arm;
      const y = 10 + k * rowh;
      kids.push(sv("rect", { x: v >= 0 ? mid : mid - w, y, width: Math.max(w, 1.5),
        height: rowh - 14, rx: 3, fill: v >= 0 ? "#66d19e" : "#f2777a" }));
      kids.push(sv("text", { x: 2, y: y + 11, fill: "#98a0b0", "font-size": 11 },
        `R${i} ${(c.coverage.find((r) => String(r.region) === i) || { games: [] }).games.join(", ")}`));
      kids.push(sv("text", { x: (v >= 0 ? mid + w + 6 : mid - w - 6), y: y + 11,
        fill: "#e6e9ef", "font-size": 11,
        "text-anchor": v >= 0 ? "start" : "end", "font-family": "monospace" },
        (v >= 0 ? "+" : "") + v.toFixed(3)));
    });
    return el("div", { class: "card" },
      el("h4", {}, n),
      el("div", { class: "games" },
        `${s.method} · ${s.samples} samples · efficiency error ${s.efficiency_error.toExponential(1)}`),
      sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, kids),
      el("p", { class: "hint" }, `auction: seats ${JSON.stringify(s.auction.seats)}`,
        ` at price ${fix(s.auction.price)} from ${s.auction.bidders} bidders`,
        `, bids ${JSON.stringify(s.auction.bids)}`));
  });
  $("shapley").replaceChildren(
    el("h3", {}, "Shapley over regions — each game's own region should earn the most"),
    el("div", { class: "cards wide" }, blocks));
}

const cls = (v) => v > 0.001 ? "pos" : (v < -0.001 ? "neg" : "zero");

/* ------------------------------------------------------ cross-region transfer */

async function startTransfer() {
  try {
    const r = await api("/api/jobs", { kind: "transfer", model: S.modelName,
      states: Number($("tr-states").value), k: Number($("tr-k").value) });
    S.transferJob = r.job;
    pollTransfer();
  } catch (e) { $("tr-progress").textContent = e.message; }
}

async function pollTransfer() {
  if (!S.transferJob) return;
  const j = (await api(`/api/job?id=${S.transferJob.id}&since=0`)).job;
  const pct = j.progress.total ? Math.round(100 * j.progress.done / j.progress.total) : 0;
  $("tr-bar").style.width = pct + "%";
  $("tr-progress").textContent = `${j.progress.done}/${j.progress.total || "?"} · ${j.status}`
    + (j.error ? ` — ${j.error}` : "");
  if (j.status === "running" || j.status === "queued") return void setTimeout(pollTransfer, 700);
  if (j.result) renderTransfer(j.result);
}

function renderTransfer(res) {
  const rows = res.rows.map((r) => el("tr", {},
    el("td", {}, `${r.game} · R${r.own_region}`),
    el("td", { class: "num" }, fix(r.own)),
    el("td", { class: "num" }, fix(r.auction_top2)),
    el("td", { class: "num" }, fix(r.all)),
    el("td", { class: "num zero" }, fix(r.baseline)),
    el("td", { class: "num " + cls(r.gain) }, (r.gain > 0 ? "+" : "") + fix(r.gain))));
  const best = res.rows.filter((r) => r.gain > 0.001);
  $("transfer").replaceChildren(
    el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "game · own region"),
      el("th", { class: "num" }, "own region"), el("th", { class: "num" }, "auction top-2"),
      el("th", { class: "num" }, "all regions"), el("th", { class: "num" }, "baseline"),
      el("th", { class: "num" }, "best gain"))), el("tbody", {}, rows)),
    el("p", { class: "hint" }, best.length
      ? `${best.map((r) => r.game).join(", ")} gains from a region that does not own it — ` +
        "one region's learning reaching another's game, which is the only thing an ensemble can do."
      : "Every column is flat: on this model nothing transfers between regions, and the graph is " +
        "earning its place by placing games rather than by regions teaching each other."));
}

boot().catch((e) => {
  document.body.prepend(el("p", { class: "msg err", style: "padding:20px" },
    "could not start: " + e.message));
});
