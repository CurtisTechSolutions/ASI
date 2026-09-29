// LatentRadixPair frontend: plain JS against the JSON API served by `latentpair serve`.
'use strict';

const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { error: text }; }
  if (!res.ok || data.error) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
function fmt(x, digits = 3) { return typeof x === 'number' ? x.toFixed(digits) : String(x); }
function unitLabel(u) { return u === ' ' ? '\u2423' : u; }
function showError(el, err) { el.innerHTML = `<span class="err">${esc(err.message || err)}</span>`; }
function busy(button, on) { button.disabled = on; }
function splitTexts(raw) { return raw.split(/\n\s*\n/).map((t) => t.trim()).filter((t) => t.length); }
async function readFiles(input) {
  const out = [];
  for (const f of input.files || []) out.push(await f.text());
  return out;
}

// ---- charts ----------------------------------------------------------------------------------------

// A ranked list of horizontal bars: one hue for a magnitude, a label and the value on every row (the list
// is its own table), a hover tooltip.
function barList(container, items, { max = null, valueText = (v) => fmt(v), negative = () => false } = {}) {
  container.innerHTML = '';
  if (!items.length) return;
  const top = max ?? Math.max(...items.map((d) => Math.abs(d.value)), 1e-9);
  const rowH = 22, labelW = 120, valueW = 70, width = 700, barW = width - labelW - valueW - 8;
  const height = items.length * rowH + 4;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'img');
  const tip = document.createElement('div');
  tip.className = 'tooltip';
  items.forEach((d, i) => {
    const y = i * rowH + 2;
    const w = Math.max(2, (Math.abs(d.value) / top) * barW);
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.innerHTML = `
      <rect class="hit" x="0" y="${y}" width="${width}" height="${rowH}"></rect>
      <text class="bar-label" x="0" y="${y + 15}">${esc(d.label)}</text>
      <rect class="bar${negative(d) ? ' neg' : ''}" x="${labelW}" y="${y + 4}" width="${w}" height="${rowH - 8}" rx="4" ry="4"></rect>
      <rect class="bar${negative(d) ? ' neg' : ''}" x="${labelW}" y="${y + 4}" width="${Math.min(4, w)}" height="${rowH - 8}"></rect>
      <text class="bar-value" x="${labelW + w + 6}" y="${y + 15}">${esc(valueText(d.value, d))}</text>`;
    g.addEventListener('mousemove', (ev) => {
      tip.style.display = 'block';
      tip.textContent = d.tip || `${d.label}: ${valueText(d.value, d)}`;
      const r = container.getBoundingClientRect();
      tip.style.left = `${ev.clientX - r.left + 12}px`;
      tip.style.top = `${ev.clientY - r.top - 28}px`;
    });
    g.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
    svg.appendChild(g);
  });
  container.appendChild(svg);
  container.appendChild(tip);
}

// A line chart of one or two series over steps, with a legend, direct labels at the line ends and a
// crosshair tooltip.
function lineChart(container, series, { yLabel = 'bits' } = {}) {
  container.innerHTML = '';
  const all = series.flatMap((s) => s.points);
  if (all.length < 2) return;
  const width = 700, height = 220, left = 46, right = 90, topPad = 12, bottom = 28;
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys) * 0.95, y1 = Math.max(...ys) * 1.05;
  const sx = (x) => left + ((x - x0) / Math.max(x1 - x0, 1)) * (width - left - right);
  const sy = (y) => topPad + (1 - (y - y0) / Math.max(y1 - y0, 1e-9)) * (height - topPad - bottom);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  let html = '';
  for (let i = 0; i <= 4; i++) {
    const y = y0 + ((y1 - y0) * i) / 4;
    html += `<line class="axis" x1="${left}" x2="${width - right}" y1="${sy(y)}" y2="${sy(y)}"></line>
             <text class="tick" x="${left - 6}" y="${sy(y) + 4}" text-anchor="end">${y.toFixed(2)}</text>`;
  }
  for (let i = 0; i <= 4; i++) {
    const x = x0 + ((x1 - x0) * i) / 4;
    html += `<text class="tick" x="${sx(x)}" y="${height - 8}" text-anchor="middle">${Math.round(x)}</text>`;
  }
  html += `<text class="tick" x="${left - 6}" y="${topPad - 2}" text-anchor="end">${esc(yLabel)}</text>`;
  series.forEach((s, k) => {
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
    const last = s.points[s.points.length - 1];
    html += `<path class="line s${k + 1}" d="${d}"></path>
             <text class="bar-label" x="${sx(last.x) + 6}" y="${sy(last.y) + 4}">${esc(s.name)} ${last.y.toFixed(2)}</text>`;
  });
  html += `<line class="crosshair" id="cross" x1="0" x2="0" y1="${topPad}" y2="${height - bottom}" style="display:none"></line>`;
  svg.innerHTML = html;
  const tip = document.createElement('div');
  tip.className = 'tooltip';
  svg.addEventListener('mousemove', (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * width;
    const x = x0 + ((px - left) / (width - left - right)) * (x1 - x0);
    let best = null;
    for (const s of series) for (const p of s.points) if (!best || Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
    if (!best) return;
    const cross = svg.querySelector('#cross');
    cross.style.display = 'block';
    cross.setAttribute('x1', sx(best.x));
    cross.setAttribute('x2', sx(best.x));
    tip.style.display = 'block';
    tip.innerHTML = `step ${best.x}<br>` + series.map((s) => {
      const p = s.points.find((q) => q.x === best.x);
      return p ? `${esc(s.name)}: ${p.y.toFixed(3)}` : '';
    }).filter(Boolean).join('<br>');
    const cr = container.getBoundingClientRect();
    tip.style.left = `${ev.clientX - cr.left + 12}px`;
    tip.style.top = `${ev.clientY - cr.top - 40}px`;
  });
  svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; svg.querySelector('#cross').style.display = 'none'; });
  const legend = document.createElement('div');
  legend.className = 'legend';
  legend.innerHTML = series.map((s, k) => `<span><span class="swatch" style="background:${k ? 'var(--series-2)' : 'var(--accent)'}"></span>${esc(s.name)}</span>`).join('');
  container.appendChild(legend);
  container.appendChild(svg);
  container.appendChild(tip);
}

// ---- model summary ----------------------------------------------------------------------------------

async function refreshInfo() {
  try {
    const info = await api('/api/info');
    const t = info.tokenizer;
    $('summary').textContent =
      `${info.model_path} · code ${t.levels} (${t.code_bits.toFixed(0)} bits, window ${t.window}, ${t.steps} steps) · ` +
      `${info.nodes.toLocaleString()} nodes × 257 · read ${info.read.units.toLocaleString()} units in ${info.read.texts} texts · ` +
      `${info.judged.texts} verdicts · outcomes ${info.settings.outcomes}` + (info.training ? ` · training: ${info.training}` : '');
    $('j-outcomes').placeholder = String(info.settings.outcomes);
  } catch (err) {
    $('summary').textContent = `no model: ${err.message}`;
  }
}

// ---- predict ----------------------------------------------------------------------------------------

$('p-run').addEventListener('click', async () => {
  busy($('p-run'), true);
  try {
    const r = await api('/api/predict', {
      prefix: $('p-prefix').value, length: +$('p-length').value, mode: $('p-mode').value, traversal: $('p-traversal').value,
      backoff: $('p-backoff').value, temperature: +$('p-temperature').value, to_end: $('p-toend').checked,
    });
    const gen = r.units.map((u) => r.unit_names[r.units.indexOf(u)]);
    $('p-out').innerHTML = `${esc($('p-prefix').value)}<span class="gen">${esc(r.text)}</span>${r.reached_end ? ' ⟨end⟩' : ''}\n` +
      `${r.units.length} units, ${fmt(r.cost)} nats, ${fmt(r.bits_per_unit, 2)} bits/unit, first-step peak ${fmt(r.peak)}, mode ${r.mode}`;
    barList($('p-chart'), r.units.map((u, i) => ({ label: unitLabel(r.unit_names[i]), value: r.step_costs[i] / Math.LN2, tip: `${unitLabel(r.unit_names[i])}: ${(r.step_costs[i] / Math.LN2).toFixed(2)} bits` })),
      { valueText: (v) => `${v.toFixed(2)} bits` });
    void gen;
    if (r.mode === 'sample') refreshInfo();
  } catch (err) { showError($('p-out'), err); }
  busy($('p-run'), false);
});

// ---- fold -------------------------------------------------------------------------------------------

async function runFold() {
  busy($('f-run'), true);
  try {
    const r = await api('/api/fold', { prefix: $('f-prefix').value, traversal: $('f-traversal').value, backoff: $('f-backoff').value, top: 12 });
    let html = `code [${r.code.join(' ')}] · entropy ${fmt(r.entropy_bits, 2)} bits\n<table><tr><th>level</th><th>node</th><th class="num">seen</th><th class="num">own</th><th>decodes to</th></tr>`;
    for (const l of r.levels) html += `<tr><td>${l.level}</td><td>${l.node}</td><td class="num">${l.seen}</td><td class="num">${fmt(l.own)}</td><td>${l.level ? esc(JSON.stringify(l.decode)) : '(root: every context)'}</td></tr>`;
    $('f-levels').innerHTML = html + '</table>';
    barList($('f-chart'), r.top.map((d) => ({ label: unitLabel(d.unit), value: d.p, tip: `${unitLabel(d.unit)}: ${(100 * d.p).toFixed(2)}%` })), { max: 1, valueText: (v) => `${(100 * v).toFixed(2)}%` });
  } catch (err) { showError($('f-levels'), err); }
  busy($('f-run'), false);
}
$('f-run').addEventListener('click', runFold);

// ---- judge ------------------------------------------------------------------------------------------

$('j-kind').addEventListener('change', () => {
  const two = $('j-kind').value === 'two_nrl';
  $('j-bad-label').classList.toggle('hidden', !two);
  $('j-text-label').firstChild.textContent = two ? 'Right outcome ' : 'Outcome ';
});
for (const b of document.querySelectorAll('.presets button')) b.addEventListener('click', () => { $('j-outcomes').value = b.dataset.outcomes; });
$('j-run').addEventListener('click', async () => {
  busy($('j-run'), true);
  try {
    const r = await api('/api/judge', {
      kind: $('j-kind').value, text: $('j-text').value, bad: $('j-bad').value, prefix: $('j-prefix').value,
      strength: +$('j-strength').value, outcomes: +$('j-outcomes').value, read: $('j-read').checked,
    });
    const rec = r.record;
    $('j-out').textContent = `${rec.call}: ${rec.cells ?? (rec.rewarded + rec.punished)} cells, strength ${rec.strength}, outcomes ${rec.outcomes}, ` +
      `${fmt(rec.amount, 4)} per rung${rec.prefix ? `, prefix ${JSON.stringify(rec.prefix)}` : ''}`;
    const rows = [];
    for (const c of r.changes) {
      rows.push({ label: `${unitLabel(c.unit)} before`, value: c.before, tip: `P(${unitLabel(c.unit)}) before: ${(100 * c.before).toFixed(3)}%` });
      rows.push({ label: `${unitLabel(c.unit)} after`, value: c.after, tip: `P(${unitLabel(c.unit)}) after: ${(100 * c.after).toFixed(3)}%` });
    }
    barList($('j-chart'), rows, { max: Math.max(...rows.map((d) => d.value), 1e-9), valueText: (v) => `${(100 * v).toFixed(3)}%` });
    refreshInfo();
  } catch (err) { showError($('j-out'), err); }
  busy($('j-run'), false);
});

// ---- read -------------------------------------------------------------------------------------------

$('r-run').addEventListener('click', async () => {
  busy($('r-run'), true);
  try {
    const texts = [...splitTexts($('r-texts').value), ...(await readFiles($('r-files')))];
    const r = await api('/api/read', { texts });
    $('r-out').textContent = `read ${r.texts} texts, ${r.units} units`;
    refreshInfo();
  } catch (err) { showError($('r-out'), err); }
  busy($('r-run'), false);
});

// ---- score ------------------------------------------------------------------------------------------

$('s-run').addEventListener('click', async () => {
  busy($('s-run'), true);
  try {
    const r = await api('/api/score', { text: $('s-text').value, traversal: $('s-traversal').value, backoff: '' });
    $('s-out').textContent = `${fmt(r.bits)} bits/unit over ${r.units} units, mean reward ${fmt(r.mean_reward)}, worst penalty ${fmt(r.worst_penalty)}`;
    barList($('s-chart'), r.per_unit.map((u) => ({ label: unitLabel(u.unit), value: u.bits, tip: `${unitLabel(u.unit)}: ${u.bits.toFixed(2)} bits, reward ${u.reward.toFixed(3)}, penalty ${u.penalty.toFixed(3)}` })),
      { valueText: (v) => `${v.toFixed(2)} bits` });
  } catch (err) { showError($('s-out'), err); }
  busy($('s-run'), false);
});

// ---- tokenizer --------------------------------------------------------------------------------------

let pollTimer = null;
function renderProgress(p) {
  const lines = p.stats.map((s) => `step ${String(s.step).padStart(6)}  recon ${s.loss_bits.toFixed(3)} bits  next ${s.next_bits.map((v) => v.toFixed(2)).join('/')}  ` +
    `exact ${s.accuracy.map((v) => v.toFixed(2)).join('/')}  tail ${s.tail.map((v) => v.toFixed(2)).join('/')}  ${s.seconds.toFixed(0)}s`);
  const status = p.error ? `<span class="err">${esc(p.error)}</span>` : p.running ? 'training…' : p.done ? 'done: the server now holds the new model' : 'idle';
  $('t-out').innerHTML = `${status}\n${esc(lines.join('\n'))}`;
  if (p.stats.length >= 2) {
    lineChart($('t-chart'), [
      { name: 'reconstruction', points: p.stats.map((s) => ({ x: s.step, y: s.loss_bits })) },
      { name: 'next byte', points: p.stats.map((s) => ({ x: s.step, y: s.next_bits[s.next_bits.length - 1] })) },
    ]);
  }
}
async function poll() {
  try {
    const p = await api('/api/tokenizer/progress');
    renderProgress(p);
    if (p.running) { pollTimer = setTimeout(poll, 1000); } else { pollTimer = null; busy($('t-run'), false); refreshInfo(); }
  } catch (err) { showError($('t-out'), err); busy($('t-run'), false); }
}
$('t-run').addEventListener('click', async () => {
  const texts = [...splitTexts($('t-texts').value), ...(await readFiles($('t-files')))];
  if (!texts.length) { $('t-out').innerHTML = '<span class="err">paste some text or choose files first</span>'; return; }
  if (!confirm('Train a new tokenizer and replace the model on the server?')) return;
  busy($('t-run'), true);
  try {
    await api('/api/tokenizer/train', {
      texts, steps: +$('t-steps').value, batch: +$('t-batch').value, lr: +$('t-lr').value, window: +$('t-window').value,
      levels: $('t-levels').value, recency: +$('t-recency').value, predict: +$('t-predict').value, read: $('t-read').checked,
    });
    if (!pollTimer) poll();
  } catch (err) { showError($('t-out'), err); busy($('t-run'), false); }
});
$('t-classes').addEventListener('click', async () => {
  busy($('t-classes'), true);
  try {
    const texts = [...splitTexts($('t-texts').value), ...(await readFiles($('t-files'))), ...splitTexts($('r-texts').value)];
    const r = await api('/api/classes', { texts });
    let html = `first symbol over ${r.total} positions (radix ${r.radix})\n<table><tr><th>class</th><th class="num">share</th><th>prototype</th><th>example</th></tr>`;
    for (const c of r.classes) html += `<tr><td>${c.symbol}</td><td class="num">${(100 * c.share).toFixed(2)}%</td><td>${esc(JSON.stringify(c.prototype))}</td><td>${esc(JSON.stringify(c.example))}</td></tr>`;
    $('t-classes-out').innerHTML = html + '</table>';
  } catch (err) { showError($('t-classes-out'), err); }
  busy($('t-classes'), false);
});

// ---- header -----------------------------------------------------------------------------------------

$('save').addEventListener('click', async () => {
  try { const r = await api('/api/save', {}); $('summary').textContent = `saved ${r.saved}`; setTimeout(refreshInfo, 1500); }
  catch (err) { $('summary').textContent = err.message; }
});
$('theme').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme === 'dark' || (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  root.dataset.theme = dark ? 'light' : 'dark';
});

refreshInfo();
api('/api/tokenizer/progress').then((p) => { if (p.running) { busy($('t-run'), true); poll(); } else if (p.stats.length) renderProgress(p); }).catch(() => {});
runFold();
