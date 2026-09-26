// The drawing half of the live page (IKA-332): plain, on purpose. It takes what
// live-data.js hands it and writes it into the page's elements by id; the look is
// live.css's. A designed page can replace this file and keep live-data.js.
"use strict";
const READ_JA = { deep: "読んだ", leaf: "葉の値", refused: "読めない（葉の値）" };
const state = {
  names: ["側0", "側1"], personSide: 1, board: null, prompt: null, select: null, picked: [],
  decision: -1, series: [], last: null, plan: null, answered: true, sheets: null,
  open: new Set(["p0"]),
};
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (p) => (100 * p).toFixed(p >= 0.1 ? 0 : 1) + "%";
const val = (v) => v.toFixed(3);
const delta = (d) => `<span class="${d >= 0 ? "up" : "down"}">${d >= 0 ? "+" : ""}${d.toFixed(3)}</span>`;

// ------------------------------------------------------------------ events
function setStatus(text, cls) { const el = $("status"); el.textContent = text; el.className = cls || ""; }
function log(html) { const d = document.createElement("div"); d.innerHTML = html; $("log").prepend(d); }

function onEvent(e) {
  switch (e.type) {
    case "sheets":
      state.names = e.names; state.personSide = e.personSide; state.sheets = e;
      renderSheets(); log(`対局開始: AI は ${esc(e.agent)}、1 手 ${e.seconds} 秒（${e.clock === "wall" ? "壁時計" : "数えの時計"}、${e.cores} コア）`);
      break;
    case "select":
      state.select = e; state.picked = []; renderInput(); setStatus("選出を選んでください", "turn");
      break;
    case "board":
      state.board = e; renderBoard();
      if (state.select) { state.select = null; renderInput(); }
      break;
    case "think":
      state.decision = e.decision; state.series = []; state.plan = e; state.answered = false;
      state.prompt = null; renderInput();
      setStatus(`AI が考え中（ターン ${e.turn}）`, "think"); renderPlan(); applyHide();
      break;
    case "answer":
      setStatus(`AI は答えを出した（${e.seconds.toFixed(2)} 秒）。あなたの番`, "turn");
      break;
    case "prompt":
      state.prompt = e; state.answered = false; renderInput(); applyHide();
      if (e.kind !== "move") setStatus(e.heading, "turn");
      break;
    case "turn":
      if (state.prompt) { state.prompt = null; state.answered = true; renderInput(); applyHide(); }
      log(`ターン ${e.turn}: <span style="color:var(--ai)">AI</span> ${esc(e.agent)} ／ <span style="color:var(--you)">あなた</span> ${esc(e.person)}`);
      break;
    case "end": {
      const mine = e.outcome === null ? null : (e.outcome > 0.5) === (e.personSide === 0);
      const text = mine === null ? `打ち切り（${e.reason}）` : mine ? "あなたの勝ち" : "あなたの負け";
      setStatus(`終局: ${text}（${e.turns} ターン）`); log(`<b>終局: ${text}</b>`);
      state.prompt = null; state.select = null; state.answered = true; renderInput(); applyHide();
      break;
    }
    case "error": log(`<span class="error">${esc(e.message)}</span>`); break;
  }
}

function onStep(s) {
  if (s.decision !== state.decision) { state.decision = s.decision; state.series = []; }
  state.series.push([s.ms, s.value, s.cells, s.kind]);
  state.last = s;
  renderCounters(s); renderChart(); renderBars(s); renderPv(s);
}

// ------------------------------------------------------------------ rendering
function hpClass(p) { return p > 50 ? "" : p > 20 ? "mid" : "low"; }
function monHtml(m, exact) {
  if (!m) return "";
  const boosts = Object.entries(m.boosts || {}).map(([k, v]) => `${k}${v > 0 ? "+" : ""}${v}`).join(" ");
  const hp = exact && m.hp ? `${m.hp[0]}/${m.hp[1]}` : `${m.percent}%`;
  const extras = [m.status, m.item ? "@" + m.item : null, boosts || null, (m.volatiles || []).join(", ") || null].filter(Boolean);
  const moves = exact && m.moves ? `<div class="extra">${m.moves.map(([n, pp, mx]) => `${esc(n)} ${pp}/${mx}`).join("、")}</div>` : "";
  return `<div class="mon ${m.fainted ? "fainted" : ""}"><div><span class="name">${esc(m.species)}</span>
    ${m.fainted ? "ひんし" : ""}<div class="extra">${esc(extras.join(" ・ "))}</div></div>
    <div><div class="num">${hp}</div><div class="hpbar"><div class="${hpClass(m.percent)}" style="width:${m.fainted ? 0 : m.percent}%"></div></div></div>${moves}</div>`;
}
function renderBoard() {
  const b = state.board;
  if (!b) return;
  const order = [1 - state.personSide, state.personSide];
  let html = `<div class="chips"><span>ターン ${b.turn}</span>${b.field.map((f) => `<span>${esc(f)}</span>`).join("")}
    ${b.seconds != null ? `<span>AI の持ち時間 ${b.seconds} 秒/手</span>` : ""}</div>`;
  for (const i of order) {
    const side = b.sides[i];
    const exact = i === b.viewer;
    html += `<div class="side ${exact ? "you" : "ai"}"><div class="who">${esc(side.name)}${side.conditions.length ? ` <span class="chips">${side.conditions.map((c) => `<span>${esc(c)}</span>`).join("")}</span>` : ""}</div>`;
    html += side.active.map((m) => (m ? `<div>${monHtml(m, exact)}</div>` : "")).join("");
    if (side.bench.length || side.hidden) {
      html += `<div class="note">控え（裏）: ${side.bench.map((m) => `${esc(m.species)} ${m.fainted ? "ひんし" : exact && m.hp ? m.hp[0] + "/" + m.hp[1] : m.percent + "%"}`).join("、")}${side.hidden ? `${side.bench.length ? "、" : ""}まだ見ていない ${side.hidden} 体` : ""}</div>`;
    }
    html += "</div>";
  }
  $("board").innerHTML = html;
}
function renderPlan() {
  const p = state.plan;
  if (!p) return;
  const plan = p.plan || {};
  $("plan").textContent = `ターン ${p.turn}: 予算 ${p.seconds} 秒、幅 ${plan.width}、深さ 1 の予測 ${Math.round(plan.predictedMs)} ms、深化に ${Math.round(plan.deepenMs)} ms、メニューに ${Math.round(p.menuMs)} ms` +
    (p.exact ? "（裏は尽きている）" : `（あなたの裏の決定化 ${p.classes} 通り）`);
}
function renderCounters(s) {
  const support = s.ourP.filter((p) => p > 1e-9).length;
  const items = [
    ["経過", `${(s.ms / 1000).toFixed(2)} s`], ["歩", s.step], ["読んだセル", s.cells.toLocaleString()],
    ["深化したセル", s.expanded], ["段（深さ）", s.depth], ["埋め", s.fills], ["精緻化", s.refines],
    ["幅（AI×あなた）", `${s.ours.length}×${s.theirs.length}`], ["台の大きさ", support],
    ["深く読んだ重み", pct(s.read)], ["裏の決定化", s.classes.length || "なし"],
    ["入れ/抜き", `${s.widened}/${s.swapped}`],
  ];
  $("counters").innerHTML = items.map(([k, v]) => `<div><b>${v}</b><small>${k}</small></div>`).join("");
  $("valuenote").innerHTML = `今の値 ${val(s.value)}（${s.kind === "done" ? "この手の答え" : "途中"}）。LP の双対ギャップ ${s.gap.toExponential(1)}。` +
    `「深く読んだ重み」は均衡の組の確率のうち、葉より深く読んだセルの分（収束の目安）。`;
}
function renderChart() {
  const c = $("chart"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const pts = state.series;
  if (!pts.length) return;
  const maxMs = Math.max(pts[pts.length - 1][0], (state.plan ? state.plan.seconds * 1000 : 1), 1);
  let lo = Math.min(...pts.map((p) => p[1])), hi = Math.max(...pts.map((p) => p[1]));
  const pad = Math.max(0.01, (hi - lo) * 0.15); lo = Math.max(0, lo - pad); hi = Math.min(1, hi + pad);
  const X = (ms) => 40 + (w - 50) * ms / maxMs, Y = (v) => 10 + (h - 30) * (1 - (v - lo) / (hi - lo || 1));
  g.strokeStyle = "#2e323c"; g.fillStyle = "#9aa0ab"; g.font = "11px sans-serif"; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    g.beginPath(); g.moveTo(40, y); g.lineTo(w - 10, y); g.stroke(); g.fillText(v.toFixed(3), 2, y + 4);
  }
  g.fillText(`${(maxMs / 1000).toFixed(1)} s`, w - 40, h - 4); g.fillText("0", 40, h - 4);
  g.strokeStyle = "#e8a33d"; g.lineWidth = 2; g.beginPath();
  pts.forEach(([ms, v], i) => (i ? g.lineTo(X(ms), Y(v)) : g.moveTo(X(ms), Y(v)))); g.stroke();
  g.fillStyle = "#e8a33d";
  const [lm, lv] = pts[pts.length - 1]; g.beginPath(); g.arc(X(lm), Y(lv), 3, 0, 7); g.fill();
}
function bars(labels, p, loss, keep) {
  const idx = labels.map((_, i) => i);
  const live = idx.filter((i) => p[i] > 1e-4).sort((a, b) => p[b] - p[a] || a - b);
  const rest = idx.filter((i) => p[i] <= 1e-4).sort((a, b) => loss[a] - loss[b] || a - b).slice(0, keep);
  const row = (i) => `<div class="row"><div class="label" title="${esc(labels[i])}">${esc(labels[i])}</div>
    <div class="track"><div class="fill" style="width:${(100 * p[i]).toFixed(1)}%"></div></div>
    <div class="num">${pct(p[i])}</div><div class="num loss">${loss[i] > 0 ? "−" + loss[i].toFixed(3) : "0"}</div></div>`;
  let html = live.map(row).join("");
  if (rest.length) html += `<div class="note">使わない手のうち損の小さいもの（全 ${labels.length} 手）</div>` + rest.map(row).join("");
  return html;
}
function renderBars(s) {
  $("ours").innerHTML = bars(s.ours, s.ourP, s.ourLoss, 3);
  $("theirs").innerHTML = bars(s.theirs, s.theirP, s.theirLoss, 2) +
    (s.classes.length ? `<div class="note">裏の決定化ごとの平均（信念の重みで）</div>` : "");
  if (!s.classes.length) { $("classes").innerHTML = ""; return; }
  const rows = s.classes.map((c, k) => {
    const order = c.p.map((p, i) => [p, i]).filter(([p]) => p > 1e-4).sort((a, b) => b[0] - a[0]).slice(0, 3);
    return `<tr><td class="num">${pct(c.weight)}</td><td>${esc(c.bench)}</td><td class="num">${val(c.value)}</td>
      <td>${order.map(([p, i]) => `${esc(s.theirs[i])} ${pct(p)}`).join("<br>")}</td></tr>`;
  }).join("");
  $("classes").innerHTML = `<table><tr><th>信念の重み</th><th>あなたの裏の決定化</th><th>値</th><th>その裏なら（上位）</th></tr>${rows}</table>`;
}
function pairHtml(pair, key, parentValue, s) {
  const cls = pair.klass >= 0 && s.classes[pair.klass] ? ` <span class="badge">裏: ${esc(s.classes[pair.klass].bench)}</span>` : "";
  const head = `${pct(pair.p)} <span class="ai">AI: ${esc(pair.ours)}</span> × <span class="you">あなた: ${esc(pair.theirs)}</span>${cls}
    → ${val(pair.value)} (${delta(pair.value - parentValue)}) <span class="badge ${pair.read}">${READ_JA[pair.read]}</span>`;
  if (!pair.branches.length) return `<div class="leafrow">${head}</div>`;
  const inner = pair.branches.map((b, i) => branchHtml(b, `${key}.${i}`, pair.value, s)).join("");
  return `<details data-key="${key}"${state.open.has(key) ? " open" : ""}><summary>${head}</summary>${inner}</details>`;
}
function branchHtml(b, key, parentValue, s) {
  const head = `<span class="chance">偶然 ${pct(b.weight)}</span>: ${esc(b.what)} → ${val(b.value)} (${delta(b.value - parentValue)})${b.ended ? ' <span class="badge">決着（葉の値）</span>' : ""}`;
  if (!b.node) return `<div class="leafrow">${head}</div>`;
  const n = b.node;
  const tops = `<div class="leafrow note">次のターン（値 ${val(n.value)}）: <span class="ai">AI ${n.ours.map(([l, p]) => `${esc(l)} ${pct(p)}`).join("、") || "-"}</span> ／ <span class="you">あなた ${n.theirs.map(([l, p]) => `${esc(l)} ${pct(p)}`).join("、") || "-"}</span>${b.more ? "（この先も読んだが省略）" : ""}</div>`;
  const inner = tops + n.pairs.map((p, i) => pairHtml(p, `${key}.${i}`, n.value, s)).join("");
  return `<details data-key="${key}"${state.open.has(key) ? " open" : ""}><summary>${head}</summary>${inner}</details>`;
}
function renderPv(s) {
  $("pv").innerHTML = s.pv.length ? s.pv.map((p, i) => pairHtml(p, `p${i}`, s.value, s)).join("") : '<span class="note">まだ組が無い</span>';
}
$("pv").addEventListener("toggle", (e) => {
  const key = e.target.dataset && e.target.dataset.key;
  if (!key) return;
  if (e.target.open) state.open.add(key); else state.open.delete(key);
}, true);

function renderSheets() {
  const e = state.sheets;
  if (!e) return;
  const order = [1 - e.personSide, e.personSide];
  $("sheets").innerHTML = order.map((i) => `<div class="sheet"><b style="color:var(${i === e.personSide ? "--you" : "--ai"})">${esc(e.names[i])}</b>
    ${e.teams[i].map((m, n) => `<div>${n + 1}. ${esc(m.species)} @ ${esc(m.item)} / ${esc(m.ability)} / ${esc(m.nature)} / SP ${m.sp.join("-")}<br><span class="note">${m.moves.map(esc).join(" / ")}</span></div>`).join("")}</div>`).join("");
}

// ------------------------------------------------------------------ input
function send(line) {
  LiveData.send(line);
  state.prompt = null; state.select = null; state.answered = true;
  $("input").innerHTML = '<span class="note">送りました。AI の番を待っています</span>'; applyHide();
}
function renderInput() {
  const box = $("input");
  if (state.select && state.sheets) {
    const team = state.sheets.teams[state.personSide];
    box.innerHTML = `<div class="note">${state.select.size} 体を順に押す（先の 2 体が先発）</div>` +
      team.map((m, i) => `<button data-i="${i}" class="${state.picked.includes(i) ? "picked" : ""}">${state.picked.includes(i) ? state.picked.indexOf(i) + 1 + ". " : ""}${esc(m.species)}</button>`).join("") +
      `<div><button class="primary" id="go" ${state.picked.length === state.select.size ? "" : "disabled"}>選出を送る</button> <button id="clear">やり直す</button></div>`;
    box.querySelectorAll("button[data-i]").forEach((b) => b.onclick = () => {
      const i = +b.dataset.i;
      if (!state.picked.includes(i) && state.picked.length < state.select.size) state.picked.push(i);
      renderInput();
    });
    $("go").onclick = () => send(state.picked.map((i) => i + 1).join(" "));
    $("clear").onclick = () => { state.picked = []; renderInput(); };
    return;
  }
  const p = state.prompt;
  if (!p) { box.innerHTML = '<span class="note">AI の番を待っています</span>'; return; }
  const slots = p.slots[0].length;
  const chosen = [];
  const draw = () => {
    let html = `<div><b>${esc(p.heading)}</b>（ターン ${p.turn}、合法手 ${p.choices.length}）</div>`;
    for (let k = 0; k < slots; k++) {
      const fits = p.slots.filter((a) => chosen.slice(0, k).every((c, j) => a[j][0] === c));
      const opts = [];
      fits.forEach((a) => { if (!opts.some((o) => o[0] === a[k][0])) opts.push(a[k]); });
      if (chosen[k] === undefined || !opts.some((o) => o[0] === chosen[k])) chosen[k] = opts[0][0];
      html += `<div class="slot"><label>${k + 1} 体目</label><select data-k="${k}">${opts.map(([c, l]) => `<option value="${esc(c)}" ${c === chosen[k] ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></div>`;
    }
    html += `<button class="primary" id="go">この手を送る</button>`;
    box.innerHTML = html;
    box.querySelectorAll("select").forEach((sel) => sel.onchange = () => {
      const k = +sel.dataset.k; chosen[k] = sel.value; chosen.length = k + 1; draw();
    });
    $("go").onclick = () => send(chosen.join(", "));
  };
  draw();
}
function applyHide() {
  const hide = $("hide").checked && !state.answered;
  document.body.classList.toggle("hidden-thinking", hide);
}
$("hide").onchange = applyHide;

// ------------------------------------------------------------------ connection
window.addEventListener("resize", renderChart);
LiveData.connect({
  onEvent, onStep,
  onOpen: () => setStatus("接続した"),
  onClose: () => setStatus("切れた（再読み込みで繋ぎ直す）"),
});
