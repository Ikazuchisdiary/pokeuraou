// The drawing half of the live page (IKA-332, designed). It takes what live-data.js hands it
// (LiveData.connect({onEvent, onStep, onOpen, onClose}), LiveData.send(line)) and draws the
// court, the person's input and the AI's reading. The look is live.css's.
"use strict";
(function () {
// ------------------------------------------------------------------ settings
// Sprite source: "{id}" becomes the Showdown sprite id (toID of the species plus "-" + forme,
// e.g. "urshifu-rapidstrike"). Set <meta name="sprite-url" content="..."> (a URL or a local
// folder the server serves) or window.POKEURAOU_SPRITE_URL; an empty string turns images off.
const metaSprite = document.querySelector('meta[name="sprite-url"]');
const SPRITE_URL = window.POKEURAOU_SPRITE_URL != null ? window.POKEURAOU_SPRITE_URL
  : metaSprite ? metaSprite.getAttribute("content")
  : "https://play.pokemonshowdown.com/sprites/gen5/{id}.png";
const LIVE_MIN = 0.005;      // a mixture weight below this counts as "not played"
const REORDER_MS = 1000;     // bars keep their order at least this long (motion without shuffling)
const PV_EVERY_MS = 400;
const CHART_DECISIONS = 8;   // the value chart shows the last this many decisions     // the open PV tree is redrawn at most this often
const TYPE_COLORS = {
  normal: "#9fa19f", fire: "#e62829", water: "#2980ef", electric: "#fac000", grass: "#3fa129", ice: "#3dcef3",
  fighting: "#ff8000", poison: "#9141cb", ground: "#915121", flying: "#81b9ef", psychic: "#ef4179", bug: "#91a119",
  rock: "#afa981", ghost: "#704170", dragon: "#5060e1", dark: "#624d4e", steel: "#60a1b8", fairy: "#ef70ef",
};
const READ_JA = { deep: "読んだ", leaf: "葉の値", refused: "読めない" };

// ------------------------------------------------------------------ state
const S = {
  sheets: null, names: ["AI", "あなた"], personSide: 1, agentSide: 0,
  board: null, prevHp: new Map(), prompt: null, select: null, picked: [], chosen: [],
  think: null, decision: -1, last: null, history: [], answered: true,
  info: new Map(), order: { ours: { keys: [], at: 0 }, theirs: { keys: [], at: 0 } },
  open: new Set(["p0"]), pvAt: 0, pvTimer: 0, openMon: new Set(),
};
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => (p >= 0.1 || p === 0 ? (100 * p).toFixed(0) : (100 * p).toFixed(1)) + "%";
const v3 = (v) => v.toFixed(3);
const dlt = (d) => `<span class="n ${d >= 0 ? "up" : "down"}">${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(3)}</span>`;
const aiSide = () => 1 - S.personSide;

// ------------------------------------------------------------------ sprites
// The page needs a sprite id and types per species; they come from the mon itself (id, types)
// or from the team sheets, looked up by the Japanese name.
const badSprite = new Set();
function learn(m) {
  if (!m || !m.species) return;
  const old = S.info.get(m.species) || {};
  S.info.set(m.species, { id: m.id || m.sprite || old.id || null, types: m.types || old.types || [] });
}
function infoOf(name) { return S.info.get(name) || { id: null, types: [] }; }
function spriteUrl(id) { return SPRITE_URL && id ? SPRITE_URL.replace("{id}", encodeURIComponent(id)) : null; }
function art(name, size) {
  const inf = infoOf(name);
  const t1 = TYPE_COLORS[inf.types[0]] || "", t2 = TYPE_COLORS[inf.types[1]] || "";
  const style = `${t1 ? `--t1:${t1};` : ""}${t2 ? `--t2:${t2};` : ""}`;
  const url = spriteUrl(inf.id);
  const fb = `<span class="fb-full">${esc(name)}</span><span class="fb-short">${esc(String(name).slice(0, 1))}</span>`;
  if (!url || badSprite.has(url)) return `<span class="art s-${size} fb" style="${style}" title="${esc(name)}">${fb}</span>`;
  return `<span class="art s-${size}" style="${style}" title="${esc(name)}">${fb}<img src="${esc(url)}" alt="${esc(name)}" decoding="async"></span>`;
}
document.addEventListener("error", (e) => {
  const img = e.target;
  if (img.tagName === "IMG" && img.parentElement && img.parentElement.classList.contains("art")) {
    badSprite.add(img.getAttribute("src")); img.parentElement.classList.add("fb");
  }
}, true);
document.addEventListener("load", (e) => {
  const img = e.target;
  if (img.tagName === "IMG" && img.parentElement && img.parentElement.classList.contains("art")) img.parentElement.classList.add("ok");
}, true);

// An action's label comes as one part per slot (live-data.js splits it): slot k is the side's
// k-th active Pokemon, so each part gets its user's icon.
function splitAct(label, side) {
  const act = S.board && S.board.sides[side] ? S.board.sides[side].active : [];
  const parts = Array.isArray(label) ? label : label.split(LiveData.SLOT_SEPARATOR);
  if (parts.length === act.length) return parts.map((t, k) => [act[k] ? act[k].species : null, t]);
  return [[null, parts.join(LiveData.SLOT_SEPARATOR)]];
}
function actHtml(label, side, size) {
  return splitAct(label, side).map(([who, t]) =>
    `<span class="part">${who ? art(who, size || "xs") : ""}<span>${esc(t)}</span></span>`).join("");
}
const benchText = (bench) => (bench && bench.length ? bench.join(" / ") : "–");

// ------------------------------------------------------------------ events
function setStatus(text, cls) { const el = $("status"); el.textContent = text; el.className = "status " + (cls || ""); }
function log(turn, html, cls) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${turn != null ? "T" + turn : "·"}</span><div class="${cls || ""}">${html}</div>`;
  $("log").prepend(li);
}
function toast(text) {
  const t = document.createElement("div"); t.className = "toast"; t.textContent = text;
  document.body.appendChild(t); setTimeout(() => t.remove(), 4000);
}

function onEvent(e) {
  switch (e.type) {
    case "sheets":
      S.sheets = e; S.names = e.names; S.personSide = e.personSide; S.agentSide = e.agentSide;
      e.teams.forEach((team) => team.forEach(learn));
      renderSheets();
      log(null, `対局開始。AI は <b>${esc(e.agent)}</b>、1 手 ${e.seconds} 秒（${e.clock === "wall" ? "壁時計" : "数えの時計"}・${e.cores} コア）`);
      break;
    case "select":
      S.select = e; S.picked = []; renderInput(); setStatus("選出を選んでください", "turn");
      break;
    case "board":
      e.sides.forEach((sd) => [...sd.active, ...sd.bench].forEach(learn));
      S.board = e; renderBoard();
      if (S.select) { S.select = null; renderInput(); }
      break;
    case "think":
      S.sent = false; S.think = e; S.decision = e.decision; S.answered = false; S.prompt = null;
      S.history.push({ decision: e.decision, turn: e.turn, pts: [], done: false });
      renderInput(); renderClock(0, false);
      setStatus(`AI が考えています（ターン ${e.turn}）`, "think"); applyHide();
      break;
    case "answer":
      setStatus(`AI は手を決めました（${e.seconds.toFixed(1)} 秒）`, "turn");
      renderClock(e.seconds * 1000, true);
      break;
    case "prompt":
      S.prompt = e; S.chosen = []; S.answered = false; renderInput(); applyHide();
      setStatus(e.kind === "move" ? "あなたの番" : e.heading, "turn");
      break;
    case "turn":
      S.sent = false;
      if (S.prompt) { S.prompt = null; S.answered = true; applyHide(); }
      renderInput();
      log(e.turn, `<span class="ai-c">AI</span> ${esc((e.agentSlots || [e.agent]).join(" ／ "))}<br><span class="you-c">あなた</span> ${esc((e.personSlots || [e.person]).join(" ／ "))}` +
        (e.offMenu ? ` <span class="badge" title="あなたの手は AI の候補集合の外でした">候補集合の外</span>` : "") +
        ((e.changes || []).length ? `<br><span class="dim">${e.changes.map((c) =>
          `${esc(c.species)}${c.entered ? " 登場" : ""}${c.from !== c.to ? ` ${c.from}→${c.to}%` : ""}${c.fainted ? " ひんし" : ""}${c.status ? " " + esc(c.status) : ""}`).join("・")}</span>` : ""));
      break;
    case "end": {
      const mine = e.outcome === null ? null : (e.outcome > 0.5) === (e.personSide === 0);
      const head = mine === null ? "打ち切り" : mine ? "あなたの勝ち" : "AI の勝ち";
      setStatus(`終局: ${head}`, "end");
      const r = $("result"); r.hidden = false;
      r.innerHTML = `<div><b>${head}</b><span>${e.turns} ターン${e.reason ? "・" + esc(e.reason) : ""}</span></div>`;
      log(e.turns, `<b>終局: ${head}</b>`);
      S.prompt = null; S.select = null; S.answered = true; renderInput(); applyHide();
      break;
    }
    case "error": log(null, esc(e.message), "err"); toast(e.message); break;
  }
}

function onStep(s) {
  if (s.decision !== S.decision || !S.history.length) {
    S.decision = s.decision; S.history.push({ decision: s.decision, turn: s.turn, pts: [], done: false });
  }
  const h = S.history[S.history.length - 1];
  h.pts.push([Math.min(1, share(s)), s.value]);
  h.done = s.kind === "done";
  S.last = s;
  renderClock(s.ms, s.kind === "done", s);
  renderBalance(s); renderStrip(s); drawChart();
  renderMix("ours", s.ours, s.ourP, s.ourLoss, aiSide(), s.kind === "done", s.oursSlots);
  renderMix("theirs", s.theirs, s.theirP, s.theirLoss, S.personSide, s.kind === "done", s.theirsSlots);
  renderClasses(s);
  if ($("countBox").open) renderCounters(s);
  schedulePv(s.kind === "done");
}

// ------------------------------------------------------------------ the court
function hpCls(p) { return p > 50 ? "" : p > 20 ? "mid" : "low"; }
function monCard(m, mine, key) {
  if (!m) return `<div class="mon empty">空き</div>`;
  const prev = S.prevHp.has(key) && S.prevHp.get(key)[0] === m.species ? S.prevHp.get(key)[1] : m.percent;
  const hpNow = m.fainted ? 0 : m.percent;
  const boosts = Object.entries(m.boosts || {}).filter(([, v]) => v)
    .map(([k, v]) => `<span class="boost ${v > 0 ? "up" : "down"}">${esc((S.sheets && S.sheets.statNames && S.sheets.statNames[k]) || k)}${v > 0 ? "+" : "−"}${Math.abs(v)}</span>`).join("");
  const exact = mine && m.hp ? `<span class="n">${m.hp[0]}/${m.hp[1]}</span>` : "";
  const status = m.fainted ? `<span class="st">ひんし</span>` : m.status ? `<span class="st st-${esc(m.status)}">${esc(m.status)}</span>` : "";
  const more = [];
  if (m.item) more.push(`<div>持ち物 ${esc(m.item)}</div>`);
  if ((m.volatiles || []).length) more.push(`<div>${m.volatiles.map(esc).join("・")}</div>`);
  if (mine && m.moves) more.push(`<table>${m.moves.map(([n, pp, mx]) => `<tr><td>${esc(n)}</td><td class="n">${pp}/${mx}</td></tr>`).join("")}</table>`);
  const drop = prev - hpNow;
  return `<article class="mon${m.fainted ? " fainted" : ""}${S.openMon.has(key) ? " open" : ""}" data-key="${key}" title="押すと詳細">
    ${art(m.species, "lg")}
    <div style="min-width:0">
      <div class="mon-name">${esc(m.species)} ${status}</div>
      <div class="hp"><div class="hp-track"><i class="${hpCls(prev)}" style="width:${prev}%" data-to="${hpNow}"></i></div>
        <b class="hp-num n">${hpNow}<small>%</small></b></div>
      <div class="mon-sub">${exact}${boosts}${(m.volatiles || []).length ? `<span>${m.volatiles.map(esc).join("・")}</span>` : ""}</div>
    </div>
    ${drop > 0 ? `<span class="hp-delta n">−${drop}%</span>` : ""}
    <div class="mon-more">${more.join("") || "詳細なし"}</div>
  </article>`;
}
function benchHtml(side, mine) {
  const minis = side.bench.map((m) => `<span class="mini${m.fainted ? " fainted" : ""}">${art(m.species, "sm")}<span>${esc(m.species)}</span>
    <span class="n dim">${m.fainted ? "ひんし" : mine && m.hp ? m.hp[0] + "/" + m.hp[1] : m.percent + "%"}</span></span>`);
  for (let i = 0; i < side.hidden; i++) minis.push(`<span class="facedown" title="まだ見ていない裏">?</span>`);
  if (!minis.length) return "";
  return `<div class="bench"><span class="lbl">裏</span>${minis.join("")}</div>`;
}
function renderBoard() {
  const b = S.board;
  if (!b) return;
  $("result").hidden = !b.ended || $("result").innerHTML === "";
  const top = aiSide(), bottom = S.personSide;
  const draw = (i, where) => {
    const sd = b.sides[i], mine = i === b.viewer;
    const head = `<div class="side-head"><span class="who ${mine ? "you" : "ai"}">${esc(sd.name)}</span>
      ${sd.conditions.map((c) => `<span class="chip">${esc(c)}</span>`).join("")}</div>`;
    const actives = `<div class="actives">${sd.active.map((m, k) => monCard(m, mine, `${i}.${k}`)).join("")}</div>`;
    const bench = benchHtml(sd, mine);
    $(where).innerHTML = where === "sideTop" ? head + bench + actives : actives + bench + head;
  };
  draw(top, "sideTop"); draw(bottom, "sideBottom");
  $("fieldbar").innerHTML = `<div class="turnno"><span>ターン</span><b>${b.turn}</b></div>
    <div class="chips">${b.field.map((f) => `<span class="chip">${esc(f)}</span>`).join("")}</div>
    ${b.seconds != null ? `<span class="clock">AI 1 手 <span class="n">${b.seconds}</span> 秒</span>` : ""}`;
  // HP bars slide from the last board's value to this one.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".hp-track i[data-to]").forEach((el) => {
      const to = +el.dataset.to; el.style.width = to + "%"; el.className = hpCls(to);
    });
  }));
  S.prevHp.clear();
  b.sides.forEach((sd, i) => sd.active.forEach((m, k) => m && S.prevHp.set(`${i}.${k}`, [m.species, m.fainted ? 0 : m.percent])));
}
$("stage").addEventListener("click", (e) => {
  const card = e.target.closest(".mon[data-key]");
  if (!card) return;
  const k = card.dataset.key;
  if (S.openMon.has(k)) S.openMon.delete(k); else S.openMon.add(k);
  card.classList.toggle("open");
});

// ------------------------------------------------------------------ the reading
// How far through its budget a step is: the wall clock's seconds, or on the count clock the
// deepening's spent share of its budget (cells at measured prices; seconds do not bound it).
function share(s) {
  if (S.think && S.think.clock === "count") return s && s.budget ? s.spent / s.budget : 1;
  const budget = S.think && S.think.seconds ? S.think.seconds * 1000 : Math.max(s ? s.ms : 1, 1);
  return (s ? s.ms : 0) / budget;
}
function renderClock(ms, done, s) {
  const budget = S.think && S.think.seconds ? S.think.seconds * 1000 : 0;
  const bar = $("clockbar");
  bar.classList.toggle("done", !!done);
  const count = S.think && S.think.clock === "count";
  const f = count ? (s ? share(s) : done ? 1 : 0) : budget ? ms / budget : 0;
  bar.firstElementChild.style.width = Math.min(100, 100 * f) + "%";
  $("clocktext").textContent = !budget ? "–" : count
    ? `${(ms / 1000).toFixed(1)} 秒・数えの計算予算 ${Math.round(100 * Math.min(f, 9.99))}%${done ? "・答え" : ""}`
    : `${(ms / 1000).toFixed(1)} / ${(budget / 1000).toFixed(1)} 秒${done ? "・答え" : ""}`;
}
function renderBalance(s) {
  $("balAi").textContent = (100 * s.value).toFixed(1);
  $("balYou").textContent = (100 * (1 - s.value)).toFixed(1);
  $("balBar").style.width = (100 * s.value).toFixed(2) + "%";
}
function renderStrip(s) {
  $("strip").innerHTML = [
    [s.cells.toLocaleString("ja-JP"), "セル"], [s.depth, "段"], [`${s.ours.length}×${s.theirs.length}`, "幅"],
    [pct(s.read), "深く読んだ重み"], [v3(s.value), "値"],
  ].map(([v, k]) => `<span><b>${v}</b>${k}</span>`).join("");
}
function renderCounters(s) {
  const support = s.ourP.filter((p) => p > 1e-9).length;
  const items = [
    ["経過", `${(s.ms / 1000).toFixed(2)} s`], ["深化のステップ", s.step], ["読んだセル", s.cells.toLocaleString("ja-JP")],
    ["深化したセル", s.expanded], ["段（深さ）", s.depth], ["埋め", s.fills], ["精緻化", s.refines],
    ["読めなかった", s.refused], ["幅（AI×あなた）", `${s.ours.length}×${s.theirs.length}`], ["サポートの大きさ", support],
    ["深く読んだ重み", pct(s.read)], ["裏の決定化", s.classes.length || "なし"], ["計算予算", `${Math.round(s.spent)} / ${s.budget}`],
    ["双対ギャップ", s.gap.toExponential(1)],
  ];
  $("counters").innerHTML = items.map(([k, v]) => `<div><b class="n">${v}</b><small>${k}</small></div>`).join("");
  const t = S.think;
  if (t) {
    const p = t.plan || {};
    $("plan").textContent = `ターン ${t.turn}: 計算予算 ${t.seconds} 秒、候補集合の幅 ${p.width}、深さ 1 の予測 ${Math.round(p.predictedMs || 0)} ms、` +
      `深化に ${Math.round(p.deepenMs || 0)} ms、候補集合づくりに ${Math.round(t.menuMs || 0)} ms。` +
      (t.exact ? "あなたの裏は尽きている。" : `あなたの裏の決定化 ${t.classCount} 通り。`) +
      "「深く読んだ重み」は均衡の組の確率のうち、葉より深く読んだ分（収束の目安）。";
  }
}

// Bars keep their places for REORDER_MS so a converging mixture reads as growing and shrinking
// bars, not as rows jumping about.
function renderMix(key, labels, p, loss, side, final, slots) {
  const el = $(key), ord = S.order[key], now = performance.now();
  const idx = labels.map((_, i) => i);
  const sameMenu = el._labels && el._labels.length === labels.length && el._labels.every((l, i) => l === labels[i]);
  if (!sameMenu || final || now - ord.at > REORDER_MS) {
    const live = idx.filter((i) => p[i] >= LIVE_MIN).sort((a, b) => p[b] - p[a] || a - b);
    const rest = idx.filter((i) => p[i] < LIVE_MIN).sort((a, b) => loss[a] - loss[b] || a - b);
    ord.keys = [live, rest]; ord.at = now;
  }
  if (!sameMenu) { el.innerHTML = ""; el._rows = new Map(); el._labels = labels.slice(); el._more = null; }
  const [live, rest] = ord.keys;
  const row = (i) => {
    let r = el._rows.get(labels[i]);
    if (!r) {
      r = document.createElement("div"); r.className = "mrow";
      r.innerHTML = `<div class="act" title="${esc(labels[i])}">${actHtml(slots ? slots[i] : labels[i], side)}</div><div class="track"><i></i></div><b class="pct n"></b><span class="loss n"></span>`;
      r._fill = r.children[1].firstChild; r._pct = r.children[2]; r._loss = r.children[3];
      el._rows.set(labels[i], r);
    }
    r.classList.toggle("zero", p[i] < LIVE_MIN);
    r._fill.style.width = (100 * p[i]).toFixed(1) + "%";
    r._pct.textContent = pct(p[i]);
    r._loss.textContent = loss[i] >= 0.0005 ? "−" + loss[i].toFixed(3) : "0";
    return r;
  };
  live.forEach((i) => el.appendChild(row(i)));
  if (rest.length) {
    if (!el._more) {
      el._more = document.createElement("details"); el._more.className = "mix-more";
      el._more.innerHTML = "<summary></summary><div class=\"mix\"></div>";
    }
    el._more.firstChild.textContent = `打たない手 ${rest.length}（損の小さい順）`;
    const box = el._more.lastChild;
    rest.forEach((i) => box.appendChild(row(i)));
    el.appendChild(el._more);
  } else if (el._more) el._more.remove();
}

function renderClasses(s) {
  const box = $("beliefBox");
  if (!s.classes.length) {
    $("beliefSum").textContent = S.think && S.think.exact ? "裏は尽きている" : "なし";
    $("classes").innerHTML = ""; return;
  }
  const order = s.classes.map((c, k) => k).sort((a, b) => s.classes[b].weight - s.classes[a].weight);
  const top = s.classes[order[0]];
  $("beliefSum").textContent = `最有力 ${benchText(top.bench)}（${pct(top.weight)}）・${s.classes.length} 通り`;
  if (!box.open) return;
  const el = $("classes");
  const key = s.classes.map((c) => benchText(c.bench)).join("|");
  if (el._key !== key) { el._key = key; el.innerHTML = ""; el._rows = new Map(); }
  order.forEach((k) => {
    const c = s.classes[k];
    let r = el._rows.get(benchText(c.bench));
    if (!r) {
      r = document.createElement("div");
      r.innerHTML = `<div class="crow"><span class="cpair">${c.bench.map((n) => art(n, "sm")).join("")}<span>${esc(benchText(c.bench))}</span></span>
        <div class="track"><i></i></div><b class="pct n"></b></div><div class="cnote"></div>`;
      el._rows.set(benchText(c.bench), r);
    }
    r.querySelector(".track i").style.width = (100 * c.weight).toFixed(1) + "%";
    r.querySelector(".pct").textContent = pct(c.weight);
    const tops = c.p.map((q, i) => [q, i]).filter(([q]) => q >= LIVE_MIN).sort((a, b) => b[0] - a[0]).slice(0, 2);
    r.querySelector(".cnote").innerHTML = `この裏での値 <span class="n">${v3(c.value)}</span>・あなたは ` +
      tops.map(([q, i]) => `${esc(s.theirs[i])} <span class="n">${pct(q)}</span>`).join("、");
    el.appendChild(r);
  });
}

// ------------------------------------------------------------------ the PV tree
function schedulePv(force) {
  const s = S.last;
  if (!s) return;
  $("pvSum").textContent = s.pv.length ? `重い組 ${s.pv.length}・最も重い ${pct(s.pv[0].p)}` : "まだ組が無い";
  if (!$("pvBox").open) return;
  const now = performance.now();
  clearTimeout(S.pvTimer);
  if (force || now - S.pvAt >= PV_EVERY_MS) { S.pvAt = now; renderPv(s); }
  else S.pvTimer = setTimeout(() => { S.pvAt = performance.now(); renderPv(S.last); }, PV_EVERY_MS - (now - S.pvAt));
}
function pairHtml(pair, key, parentValue, s, root) {
  const cls = pair.klass >= 0 && s.classes[pair.klass] ? ` <span class="badge cls">裏 ${esc(benchText(s.classes[pair.klass].bench))}</span>` : "";
  // Icons only at the root: below it the actives may have changed and the data does not say how.
  const ours = root ? actHtml(pair.oursSlots || pair.ours, aiSide()) : `<span class="part"><span>${esc(pair.ours)}</span></span>`;
  const theirs = root ? actHtml(pair.theirsSlots || pair.theirs, S.personSide) : `<span class="part"><span>${esc(pair.theirs)}</span></span>`;
  const head = `<span class="pp n">${pct(pair.p)}</span><span class="pvmain"><span class="pvacts"><span class="act a">${ours}</span><span class="x">×</span><span class="act y">${theirs}</span></span>
    <span class="pvmeta"><span class="n">${v3(pair.value)}</span> ${dlt(pair.value - parentValue)} <span class="badge ${pair.read}">${READ_JA[pair.read] || pair.read}</span>${cls}</span></span>`;
  if (!pair.branches.length) return `<div class="leaf">${head}</div>`;
  const inner = pair.branches.map((b, i) => branchHtml(b, `${key}.${i}`, pair.value, s)).join("");
  return `<details data-key="${key}"${S.open.has(key) ? " open" : ""}><summary>${head}</summary>${inner}</details>`;
}
function branchHtml(b, key, parentValue, s) {
  const head = `<span class="pp c n">${pct(b.weight)}</span><span class="pvmain"><span><span class="c">偶然</span> ${esc(b.what)}</span>
    <span class="pvmeta"><span class="n">${v3(b.value)}</span> ${dlt(b.value - parentValue)}${b.ended ? ' <span class="badge">決着</span>' : ""}</span></span>`;
  if (!b.node) return `<div class="leaf">${head}</div>`;
  const n = b.node;
  const tops = `<div class="nodeline">次のターン 値 <span class="n">${v3(n.value)}</span>・<span class="a">AI ${n.ours.map(([l, p]) => `${esc(l)} ${pct(p)}`).join("、") || "–"}</span>・<span class="y">あなた ${n.theirs.map(([l, p]) => `${esc(l)} ${pct(p)}`).join("、") || "–"}</span>${b.more ? "・この先も読んだが省略" : ""}</div>`;
  const inner = tops + n.pairs.map((p, i) => pairHtml(p, `${key}.${i}`, n.value, s, false)).join("");
  return `<details data-key="${key}"${S.open.has(key) ? " open" : ""}><summary>${head}</summary>${inner}</details>`;
}
function renderPv(s) {
  $("pv").innerHTML = s.pv.length
    ? `<p class="note">重い組（AI の手 × あなたの手）。読んだ組は偶然の分岐と次のターンへ開く。値は AI から見た値、括弧は親との差。</p>` +
      s.pv.map((p, i) => pairHtml(p, `p${i}`, s.value, s, true)).join("")
    : '<span class="note">まだ組が無い</span>';
}
$("pv").addEventListener("toggle", (e) => {
  const key = e.target.dataset && e.target.dataset.key;
  if (!key) return;
  if (e.target.open) S.open.add(key); else S.open.delete(key);
}, true);
$("pvBox").addEventListener("toggle", () => S.last && $("pvBox").open && renderPv(S.last));
$("beliefBox").addEventListener("toggle", () => S.last && renderClasses(S.last));
$("countBox").addEventListener("toggle", () => S.last && $("countBox").open && renderCounters(S.last));

// ------------------------------------------------------------------ the value chart
// One band per decision; inside a band the line is the value while the AI thinks (x = share of
// the budget). The area between the line and 0.5 is tinted for whoever it favours.
function drawChart() {
  const c = $("chart"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w) return;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  const H = S.history.filter((d) => d.pts.length).slice(-CHART_DECISIONS);
  if (!H.length) return;
  const all = H.flatMap((d) => d.pts.map((p) => p[1]));
  let lo = Math.min(0.5, ...all), hi = Math.max(0.5, ...all);
  const span = Math.max(0.1, hi - lo); lo = Math.max(0, lo - span * 0.15); hi = Math.min(1, hi + span * 0.15);
  const L = 34, R = 8, T = 6, B = 18, n = H.length, bw = (w - L - R) / n;
  const X = (d, f) => L + bw * (d + 0.06 + 0.88 * f), Y = (v) => T + (h - T - B) * (1 - (v - lo) / (hi - lo));
  g.font = `11px ${col("--font-num") || "monospace"}`; g.textBaseline = "middle";
  H.forEach((d, i) => {
    if (i % 2 === 0) { g.fillStyle = col("--surface-2"); g.fillRect(L + bw * i, T, bw, h - T - B); }
    g.fillStyle = col("--faint"); g.textAlign = "center"; g.fillText(`T${d.turn}`, L + bw * (i + 0.5), h - 8);
  });
  g.textAlign = "right";
  const ticks = [lo, hi]; if (Math.abs(Y(0.5) - Y(lo)) > 14 && Math.abs(Y(0.5) - Y(hi)) > 14) ticks.push(0.5);
  ticks.forEach((v) => { g.fillStyle = col("--faint"); g.fillText(v.toFixed(2), L - 4, Y(v)); });
  g.strokeStyle = col("--line"); g.lineWidth = 1; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(L, Y(0.5)); g.lineTo(w - R, Y(0.5)); g.stroke(); g.setLineDash([]);
  const path = () => {
    g.beginPath();
    H.forEach((d, i) => d.pts.forEach(([f, v], j) => (i || j ? g.lineTo(X(i, f), Y(v)) : g.moveTo(X(i, f), Y(v)))));
  };
  const lastD = H.length - 1, lastP = H[lastD].pts[H[lastD].pts.length - 1];
  const area = (tint, above) => {
    g.save(); g.beginPath();
    if (above) g.rect(0, 0, w, Y(0.5)); else g.rect(0, Y(0.5), w, h);
    g.clip(); path();
    g.lineTo(X(lastD, lastP[0]), Y(0.5)); g.lineTo(X(0, H[0].pts[0][0]), Y(0.5)); g.closePath();
    g.globalAlpha = 0.16; g.fillStyle = tint; g.fill(); g.restore();
  };
  area(col("--ai"), true); area(col("--you"), false);
  path(); g.strokeStyle = col("--ink"); g.globalAlpha = 0.85; g.lineWidth = 1.6; g.lineJoin = "round"; g.stroke(); g.globalAlpha = 1;
  g.fillStyle = lastP[1] >= 0.5 ? col("--ai") : col("--you");
  g.beginPath(); g.arc(X(lastD, lastP[0]), Y(lastP[1]), 4, 0, 7); g.fill();
}
window.addEventListener("resize", drawChart);

// ------------------------------------------------------------------ the person's input
function send(line) {
  LiveData.send(line);
  S.prompt = null; S.select = null; S.answered = true; S.sent = true;
  renderInput(); applyHide();
}
function renderInput() {
  const box = $("input");
  $("inputBox").classList.toggle("active", !!(S.select || S.prompt));
  if (S.select && S.sheets) {
    const team = S.sheets.teams[S.personSide], foe = S.sheets.teams[aiSide()];
    const size = S.select.size;
    box.innerHTML = `<div class="ask">選出（${size} 体を順に押す。先の 2 体が先発）</div>
      <div class="foeteam"><span class="note">相手</span>${foe.map((m) => art(m.species, "sm")).join("")}</div>
      <div class="pickgrid">${team.map((m, i) => {
        const at = S.picked.indexOf(i);
        return `<button type="button" class="pick" data-i="${i}" aria-pressed="${at >= 0}">${at >= 0 ? `<span class="ord">${at + 1}</span>` : ""}${at >= 0 && at < 2 ? '<span class="lead">先発</span>' : ""}
          ${art(m.species, "md")}<b>${esc(m.species)}</b><small>${esc(m.item)}</small></button>`;
      }).join("")}</div>
      <div class="sendbar"><button type="button" class="ghost" id="clear">やり直す</button>
        <button type="button" class="primary" id="go" ${S.picked.length === size ? "" : "disabled"}>この選出で始める</button></div>`;
    box.querySelectorAll(".pick").forEach((b) => b.onclick = () => {
      const i = +b.dataset.i, at = S.picked.indexOf(i);
      if (at >= 0) S.picked.splice(at, 1); else if (S.picked.length < size) S.picked.push(i);
      renderInput();
    });
    $("go").onclick = () => send(S.picked.map((i) => i + 1).join(" "));
    $("clear").onclick = () => { S.picked = []; renderInput(); };
    return;
  }
  const p = S.prompt;
  if (!p) {
    box.innerHTML = S.sent ? '<span class="note">送りました。ターンの結果を待っています</span>' : '<span class="note">AI の番を待っています</span>';
    return;
  }
  const nSlots = p.slots.length ? p.slots[0].length : 0;
  const act = p.actives || (S.board ? S.board.sides[S.personSide].active : []);
  const fits = (k) => p.slots.filter((a) => S.chosen.slice(0, k).every((c, j) => a[j][0] === c));
  let html = `<div class="ask">${esc(p.heading)}</div><div class="note">ターン ${p.turn}・合法手 ${p.choices.length}</div>`;
  for (let k = 0; k < nSlots; k++) {
    const opts = [];
    fits(k).forEach((a) => { if (!opts.some((o) => o[0] === a[k][0])) opts.push(a[k]); });
    if (opts.length === 1) S.chosen[k] = opts[0][0];
    if (S.chosen[k] !== undefined && !opts.some((o) => o[0] === S.chosen[k])) S.chosen.length = k;
    const who = nSlots === act.length && act[k] ? act[k].species : null;
    html += `<div class="slot"><div class="slot-head">${who ? art(who, "sm") + `<b>${esc(who)}</b>` : `<b>${nSlots > 1 ? `${k + 1} 体目` : "選ぶ"}</b>`}</div>
      <div class="opts">${opts.map(([c, l]) => `<button type="button" class="opt" data-k="${k}" data-c="${esc(c)}" aria-pressed="${S.chosen[k] === c}">${esc(l)}</button>`).join("")}</div></div>`;
    if (S.chosen[k] === undefined) { for (let j = k + 1; j < nSlots; j++) html += `<div class="slot"><div class="slot-head dim">${j + 1} 体目は上を選んでから</div></div>`; break; }
  }
  const ready = S.chosen.length === nSlots && S.chosen.every((c) => c !== undefined);
  const labels = ready ? p.slots.find((a) => a.every((x, j) => x[0] === S.chosen[j])) : null;
  html += `<div class="sendbar"><span class="summary">${labels ? esc(labels.map((x) => x[1]).join(" ／ ")) : "手を選んでください"}</span>
    <button type="button" class="primary" id="go" ${labels ? "" : "disabled"}>この手で決定</button></div>`;
  box.innerHTML = html;
  box.querySelectorAll(".opt").forEach((b) => b.onclick = () => {
    const k = +b.dataset.k; S.chosen[k] = b.dataset.c; S.chosen.length = k + 1; renderInput();
  });
  $("go").onclick = () => { if (labels) send(S.chosen.join(", ")); };
}
function applyHide() { document.body.classList.toggle("hidden-thinking", $("hide").checked && !S.answered); }
$("hide").onchange = applyHide;

// ------------------------------------------------------------------ sheets
function renderSheets() {
  const e = S.sheets;
  if (!e) return;
  const order = [aiSide(), e.personSide];
  $("sheets").innerHTML = order.map((i) => `<div><div class="who ${i === e.personSide ? "you" : "ai"}">${esc(e.names[i])}</div>
    ${e.teams[i].map((m) => `<div class="sheet-mon">${art(m.species, "sm")}<div><b>${esc(m.species)}</b> <span class="dim" style="display:inline">@ ${esc(m.item)}</span>
      <span class="dim">${esc(m.ability)}・${esc(m.nature)}・<span class="n">${m.sp.join("-")}</span></span>
      <span class="dim">${m.moves.map(esc).join(" / ")}</span></div></div>`).join("")}</div>`).join("");
}

// ------------------------------------------------------------------ theme
const THEMES = ["auto", "light", "dark"], THEME_JA = { auto: "自動", light: "ライト", dark: "ダーク" };
let theme = "auto";
try { theme = localStorage.getItem("pokeuraou-theme") || "auto"; } catch (_) { /* storage may be blocked */ }
function applyTheme() {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  $("theme").textContent = "テーマ: " + THEME_JA[theme];
  drawChart();
}
$("theme").onclick = () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % 3];
  try { localStorage.setItem("pokeuraou-theme", theme); } catch (_) { /* ignore */ }
  applyTheme();
};
if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawChart);
applyTheme();

// ------------------------------------------------------------------ connection
LiveData.connect({
  onEvent, onStep,
  onOpen: () => setStatus("接続した"),
  onClose: () => setStatus("切れた（再読み込みで繋ぎ直す）"),
});
})();
