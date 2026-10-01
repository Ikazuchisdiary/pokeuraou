"use strict";
// 局面の検討の画面（IKA-408）。1 ターン = 1 つのフォーム。サーバ（rankedboard.BoardApp）の JSON を描き、
// 入力をそのまま送るだけで、型の絞り込みも配分の絞り込みもサーバがする。
(function () {
const TYPE_COLORS = {
  normal: "#9fa19f", fire: "#e62829", water: "#2980ef", electric: "#fac000", grass: "#3fa129", ice: "#3dcef3",
  fighting: "#ff8000", poison: "#9141cb", ground: "#915121", flying: "#81b9ef", psychic: "#ef4179", bug: "#91a119",
  rock: "#afa981", ghost: "#704170", dragon: "#5060e1", dark: "#624d4e", steel: "#60a1b8", fairy: "#ef70ef",
};
const BOOST_NAMES = { atk: "攻撃", def: "防御", spa: "特攻", spd: "特防", spe: "素早さ", accuracy: "命中", evasion: "回避" };
const metaSprite = document.querySelector('meta[name="sprite-url"]');
const SPRITE_URL = metaSprite ? metaSprite.getAttribute("content") : "";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nameHtml = (n) => {
  const [base, ...rest] = String(n == null ? "" : n).split(" (");
  return rest.length ? `${esc(base)}<small class="forme">(${esc(rest.join(" (").replace(/\)$/, ""))})</small>` : esc(base);
};
const pct = (x, d) => (x * 100).toFixed(d == null ? 0 : d) + "%";
const clone = (x) => JSON.parse(JSON.stringify(x));

let META = null, STATE = null, BOARD = null, CUR = 0, FORM = null;
let poll = null, saveTimer = null, saveSeq = 0, pickM = [], pickO = {};
const QUERY = new URLSearchParams(location.search);
const itemByName = new Map(), moveByName = new Map();

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "エラー");
  return data;
}

// ------------------------------------------------------------------ theme and gear
let theme = "auto";
try { theme = localStorage.getItem("pokeuraou-theme") || "auto"; } catch (_) { /* storage may be blocked */ }
if (QUERY.get("theme")) theme = QUERY.get("theme");
function applyTheme() {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  document.querySelectorAll("#themeSeg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.t === theme)));
}
document.querySelectorAll("#themeSeg button").forEach((b) => b.onclick = () => {
  theme = b.dataset.t;
  try { localStorage.setItem("pokeuraou-theme", theme); } catch (_) { /* ignore */ }
  applyTheme();
});
applyTheme();
function gear(open) { $("settings").hidden = !open; $("settingsBackdrop").hidden = !open; $("gear").setAttribute("aria-expanded", String(open)); }
$("gear").onclick = () => gear($("settings").hidden);
$("settingsClose").onclick = () => gear(false);
$("settingsBackdrop").onclick = () => gear(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") gear(false); });

// ------------------------------------------------------------------ pieces
function art(sp, size) {
  const t1 = TYPE_COLORS[sp.types[0]] || "", t2 = TYPE_COLORS[sp.types[1]] || "";
  const style = `${t1 ? `--t1:${t1};` : ""}${t2 ? `--t2:${t2};` : ""}`;
  const fb = `<span class="fb-full">${esc(sp.name)}</span><span class="fb-short">${esc(String(sp.name).slice(0, 1))}</span>`;
  const url = SPRITE_URL ? SPRITE_URL.replace("{id}", encodeURIComponent(sp.sprite)) : "";
  if (!url) return `<span class="art s-${size} fb" style="${style}" title="${esc(sp.name)}">${fb}</span>`;
  return `<span class="art s-${size}" style="${style}" title="${esc(sp.name)}">${fb}<img src="${esc(url)}" alt="" decoding="async"></span>`;
}
document.addEventListener("error", (e) => {
  const img = e.target;
  if (img.tagName === "IMG" && img.parentElement && img.parentElement.classList.contains("art")) img.parentElement.classList.add("fb");
}, true);
document.addEventListener("load", (e) => {
  const img = e.target;
  if (img.tagName === "IMG" && img.parentElement && img.parentElement.classList.contains("art")) img.parentElement.classList.add("ok");
}, true);

const speciesView = (sid) => (BOARD && BOARD.opp[sid] ? BOARD.opp[sid].species : { id: sid, name: sid, sprite: sid, types: [] });
const mineView = (i) => BOARD.mine[i].view;
const mineName = (i) => mineView(i).species.name;
const oppName = (sid) => speciesView(sid).name;

// ------------------------------------------------------------------ setup (who is out)
function ready1a() {
  return STATE && STATE.mine && (STATE.opponent || []).length === 6 && STATE.opponent.every((v) => v && v.moves.length) && !(STATE.teamProblems || []).length;
}
function renderSetup() {
  const ok = ready1a();
  $("setupNeed").hidden = ok;
  $("setupForm").hidden = !ok;
  if (!ok) return;
  $("pickMine").innerHTML = STATE.mine.map((v, i) => {
    const n = pickM.indexOf(i);
    const tag = n < 0 ? "" : `<b class="ps-no">${n < 2 ? "先発" : "裏"} ${n < 2 ? n + 1 : n - 1}</b>`;
    return `<button type="button" class="ps-chip${n >= 0 ? " on" : ""}" data-pm="${i}" aria-pressed="${n >= 0}">${art(v.species, "xs")}<span>${nameHtml(v.species.name)}</span>${tag}</button>`;
  }).join("");
  $("pickOpp").innerHTML = STATE.opponent.map((v) => {
    const id = v.species.id, role = pickO[id] || "";
    const tag = role === "lead" ? `<b class="ps-no">先発</b>` : role === "seen" ? `<b class="ps-no alt">見えた</b>` : "";
    return `<button type="button" class="ps-chip${role ? " on" : ""}" data-po="${esc(id)}" aria-pressed="${!!role}">${art(v.species, "xs")}<span>${nameHtml(v.species.name)}</span>${tag}</button>`;
  }).join("");
  const leads = Object.values(pickO).filter((r) => r === "lead").length;
  const seen = Object.values(pickO).length;
  const msgs = [];
  if (pickM.length !== 4) msgs.push(`自分の選出を 4 体にしてください（今 ${pickM.length} 体）`);
  if (leads !== 2) msgs.push(`相手の先発を 2 体にしてください（今 ${leads} 体）`);
  if (seen > 4) msgs.push("相手の見えた体は先発をふくめて 4 体までです");
  $("setupMsg").innerHTML = msgs.map((m) => `<li class="warn">${esc(m)}</li>`).join("");
  $("startBoard").disabled = !!msgs.length;
}
document.addEventListener("click", (e) => {
  const m = e.target.closest("[data-pm]"), o = e.target.closest("[data-po]");
  if (m) {
    const i = Number(m.dataset.pm), at = pickM.indexOf(i);
    if (at >= 0) pickM.splice(at, 1); else if (pickM.length < 4) pickM.push(i);
    renderSetup();
  } else if (o) {
    const id = o.dataset.po, role = pickO[id] || "";
    const leads = Object.values(pickO).filter((r) => r === "lead").length;
    // none -> lead (while fewer than two) -> seen -> none
    if (!role) pickO[id] = leads < 2 ? "lead" : "seen";
    else if (role === "lead") pickO[id] = "seen";
    else delete pickO[id];
    renderSetup();
  }
});
$("startBoard").onclick = async () => {
  $("startBoard").disabled = true;
  try {
    const leads = Object.keys(pickO).filter((k) => pickO[k] === "lead");
    const seen = Object.keys(pickO).filter((k) => pickO[k] === "seen");
    BOARD = await api("/api/board/start", { brought: pickM, leads, seen });
    CUR = 0; enterTurn(); renderAll();
  } catch (e) { $("setupMsg").innerHTML = `<li>${esc(e.message)}</li>`; }
  $("startBoard").disabled = false;
};
$("resetBoard").onclick = async () => {
  gear(false); BOARD = await api("/api/board/reset", {}); pickM = []; pickO = {}; CUR = 0; FORM = null; renderAll();
};

// ------------------------------------------------------------------ the turn
function enterTurn() {
  FORM = clone(BOARD.turns[CUR].form);
  const p = prevForm();
  if (!p) { FOLD = { mineBench: true, field: true, oppBench: false }; return; }
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const bench = (f, side) => (side === "m" ? BOARD.brought.filter((i) => !f.mineActive.includes(i)).map((i) => f.mine[String(i)])
    : Object.keys(f.theirs).filter((s) => !f.theirActive.includes(s)).map((s) => f.theirs[s]));
  FOLD = { mineBench: !same(bench(FORM, "m"), bench(p, "m")), oppBench: !same(bench(FORM, "t"), bench(p, "t")),
    field: !same([FORM.field, FORM.sides], [p.field, p.sides]) };
}
let FOLD = { mineBench: true, field: true, oppBench: false };
document.addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.id === "mineBenchBox") FOLD.mineBench = d.open;
  else if (d.id === "oppBenchBox") FOLD.oppBench = d.open;
  else if (d.id === "fieldBox") FOLD.field = d.open;
}, true);
const prevForm = () => (CUR > 0 ? BOARD.turns[CUR - 1].form : null);
function changed(side, key, f, value) {
  const p = prevForm();
  if (!p) return "";
  const box = side === "m" ? p.mine[String(key)] : p.theirs[key];
  if (!box) return " chg";   // a Pokemon that was not there last turn
  return JSON.stringify(box[f]) === JSON.stringify(value) ? "" : " chg";
}
function boostSummary(b) {
  const parts = Object.keys(b).filter((k) => b[k]).map((k) => `${BOOST_NAMES[k]}${b[k] > 0 ? "+" : ""}${b[k]}`);
  return parts.length ? parts.join(" ") : "なし";
}
function statusSelect(side, key, value) {
  const o = BOARD.options.status;
  return `<label class="ps-f${changed(side, key, "status", value)}"><span>状態異常</span><select data-side="${side}" data-key="${esc(key)}" data-f="status">
    <option value="">なし</option>${o.map((s) => `<option value="${s.id}" ${s.id === value ? "selected" : ""}>${esc(s.name)}</option>`).join("")}</select></label>`;
}
function boostBox(side, key, boosts) {
  const cells = Object.keys(BOOST_NAMES).map((k) =>
    `<label class="ps-boost"><span>${BOOST_NAMES[k]}</span><input type="number" min="-6" max="6" inputmode="numeric" data-side="${side}" data-key="${esc(key)}" data-f="boost" data-b="${k}" value="${boosts[k] || 0}"></label>`).join("");
  return `<details class="ps-boosts${changed(side, key, "boosts", boosts)}"><summary>能力変化 <b>${esc(boostSummary(boosts))}</b></summary><div class="ps-bgrid">${cells}</div></details>`;
}
function check(side, key, f, label, value) {
  return `<label class="ps-check${changed(side, key, f, value)}"><input type="checkbox" data-side="${side}" data-key="${esc(key)}" data-f="${f}" ${value ? "checked" : ""}> ${label}</label>`;
}
function slotSelect(side, slot, current, options) {
  const opts = options.map((o) => `<option value="${esc(o.id)}" ${String(o.id) === String(current) ? "selected" : ""}>${esc(o.name)}</option>`).join("");
  return `<select class="ps-slot" data-slot="${side}${slot}" aria-label="場の ${slot + 1} 体目">${current == null ? `<option value="" selected>（空き）</option>` : ""}${opts}</select>`;
}

function mineCard(i, slot) {
  const st = FORM.mine[String(i)], v = mineView(i), max = BOARD.mineMaxHp[String(i)];
  const active = slot != null;
  const opts = BOARD.brought.filter((j) => FORM.mine[String(j)].hp > 0 && (j === i || !FORM.mineActive.includes(j))).map((j) => ({ id: j, name: mineName(j) }));
  const head = active ? slotSelect("m", slot, i, opts) : `<span class="ps-nm">${nameHtml(v.species.name)}</span>`;
  return `<article class="ps-mon mine${active ? " active" : ""}${st.hp === 0 ? " down" : ""}">
    <div class="ps-head">${art(v.species, "sm")}<div class="ps-who">${head}<span class="ps-line">${esc(v.item ? v.item.name : "持ち物なし")} ・ ${esc(v.ability ? v.ability.name : "")}</span></div></div>
    <div class="ps-row">
      <label class="ps-f hp${changed("m", i, "hp", st.hp)}"><span>HP</span><input type="number" min="0" max="${max}" inputmode="numeric" data-side="m" data-key="${i}" data-f="hp" value="${st.hp}"><em>/ ${max}</em></label>
      <div class="ps-hpbar" aria-hidden="true"><i style="width:${Math.round(100 * st.hp / max)}%"></i></div>
    </div>
    <div class="ps-row">${statusSelect("m", i, st.status)}</div>
    ${boostBox("m", i, st.boosts)}
    <div class="ps-row ps-checks">${check("m", i, "mega", "メガシンカ済み", st.mega)}${check("m", i, "itemGone", "持ち物を使った", st.itemGone)}</div>
  </article>`;
}

function estimateHtml(sid) {
  const v = BOARD.opp[sid];
  if (!v) return "";
  const r = v.refine || {};
  const lines = [];
  lines.push(`<b>推定</b> ${esc(v.ability ? v.ability.name : "—")} ・ ${esc(v.item ? v.item.name : "持ち物なし")} ・ ${esc(v.nature.name)}`);
  lines.push(`技 ${v.moves.map((m) => esc(m.name)).join("・")}`);
  const sp = v.sp.map((x, i) => (x ? `${META.statNames[i]} ${x}` : "")).filter(Boolean).join("・");
  lines.push(`配分 ${esc(sp || "すべて 0")}${v.spProvisional ? "（仮の配分）" : ""}`);
  const extra = [];
  if (r.members) extra.push(`大会の型 ${r.members[0]} 体 → 見えたものに合う ${r.members[1]} 体`);
  (r.notes || []).forEach((n) => extra.push(n));
  BOARD.spread.filter((s) => s.id === sid).forEach((s) => extra.push(`${esc(META.statNames[META.stats.indexOf(s.stat)])}の配分 ${s.before} → ${s.after}（${s.low}〜${s.high} が可能）`));
  const alts = (v.alternatives || []).length > 1
    ? `<div class="rk-alts" role="group" aria-label="残っている型の候補">${v.alternatives.map((a) =>
      `<button type="button" class="rk-alt" data-choose="${esc(sid)}" data-a="${a.index}" aria-pressed="${String(v.chosen === a.index)}">${esc(a.label)}（${a.count} 体）<small>${esc(a.moves.join("・"))}</small></button>`).join("")}</div>` : "";
  return `<div class="ps-est" data-est="${esc(sid)}">${lines.map((l) => `<div>${l}</div>`).join("")}${extra.length ? `<ul>${extra.map((x) => `<li>${x}</li>`).join("")}</ul>` : ""}${alts}</div>`;
}

function hpHint(sid) {
  const h = BOARD.turns[CUR].oppHp[sid];
  if (!h) return "";
  const range = h.low === h.high ? `${h.low}` : `${h.low}〜${h.high}`;
  return `おおよそ HP ${h.hp} / ${h.max}（${range}）`;
}

function oppCard(sid, slot) {
  const st = FORM.theirs[sid], v = speciesView(sid);
  const active = slot != null;
  const opts = Object.keys(FORM.theirs).filter((k) => !FORM.theirs[k].fainted && (k === sid || !FORM.theirActive.includes(k))).map((k) => ({ id: k, name: oppName(k) }));
  const head = active ? slotSelect("t", slot, sid, opts) : `<span class="ps-nm">${nameHtml(v.name)}</span>`;
  const info = BOARD.opp[sid];
  const colour = st.pct === 20 || st.pct === 50
    ? `<label class="ps-f"><span>色</span><select data-side="t" data-key="${esc(sid)}" data-f="colour"><option value="">分からない</option>${(st.pct === 20 ? [["r", "赤（20%以下）"], ["y", "黄（20%より上）"]] : [["y", "黄（50%以下）"], ["g", "緑（50%より上）"]]).map(([c, n]) => `<option value="${c}" ${st.colour === c ? "selected" : ""}>${n}</option>`).join("")}</select></label>` : "";
  const moveChips = st.moves.map((m) => `<button type="button" class="ps-x" data-rm-move="${esc(sid)}" data-m="${esc(m)}" title="取り消す">${esc(moveName(m))} ×</button>`).join("");
  const fieldMoves = (info.fieldMoves || []).filter((m) => !st.moves.includes(m.id));
  const addMove = st.moves.length >= 4 ? "" : `<select class="ps-add" data-add-move="${esc(sid)}" aria-label="使った技を足す"><option value="">＋ 使った技</option>${fieldMoves.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}（${m.count}）</option>`).join("")}<option value="__other">ほかの技…</option></select>`;
  const items = (info.fieldItems || []);
  const itemOpts = `<option value="">見えていない</option>${items.map((i) => `<option value="${esc(i.id)}" ${st.item === i.id ? "selected" : ""}>${esc(i.name)}（${i.count}）</option>`).join("")}${st.item && !items.some((i) => i.id === st.item) ? `<option value="${esc(st.item)}" selected>${esc(itemName(st.item))}</option>` : ""}<option value="__other">ほかの持ち物…</option>`;
  const abil = (info.abilityChoices || []).map((a) => `<option value="${esc(a.id)}" ${st.ability === a.id ? "selected" : ""}>${esc(a.name)}</option>`).join("");
  return `<article class="ps-mon opp${active ? " active" : ""}${st.fainted ? " down" : ""}" data-card="${esc(sid)}">
    <div class="ps-head">${art(v, "sm")}<div class="ps-who">${head}<span class="ps-line">${(info.species.types || []).map((t) => `<i class="rk-type" style="--c:${TYPE_COLORS[t] || ""}"></i>`).join("")}</span></div></div>
    <div class="ps-row">
      <label class="ps-f hp${changed("t", sid, "pct", st.pct)}"><span>HP</span><input type="number" min="1" max="100" inputmode="numeric" data-side="t" data-key="${esc(sid)}" data-f="pct" value="${st.pct}"><em>%</em></label>
      ${colour}
      <div class="ps-hpbar" aria-hidden="true"><i style="width:${st.fainted ? 0 : st.pct}%"></i></div>
    </div>
    <div class="ps-hint" data-hphint="${esc(sid)}">${esc(hpHint(sid))}</div>
    <div class="ps-row">${statusSelect("t", sid, st.status)}${check("t", sid, "fainted", "倒れた", st.fainted)}</div>
    ${boostBox("t", sid, st.boosts)}
    <div class="ps-seen">
      <h4>見えた情報 <small>見えたものだけ入れてください。相手の型の候補が自動で絞られます</small></h4>
      <div class="ps-chips${changed("t", sid, "moves", st.moves)}"><span class="lab">使った技</span>${moveChips}${addMove}</div>
      <div class="ps-row"><label class="ps-f${changed("t", sid, "item", st.item)}"><span>持ち物</span><select data-side="t" data-key="${esc(sid)}" data-f="item">${itemOpts}</select></label>
        ${st.item ? check("t", sid, "itemGone", "使った・落とされた", st.itemGone) : ""}</div>
      <div class="ps-row"><label class="ps-f${changed("t", sid, "ability", st.ability)}"><span>発動した特性</span><select data-side="t" data-key="${esc(sid)}" data-f="ability"><option value="">見えていない</option>${abil}</select></label>
        ${check("t", sid, "mega", "メガシンカ済み", st.mega)}</div>
    </div>
    ${estimateHtml(sid)}
  </article>`;
}
const moveName = (id) => (META.moves.find((m) => m.id === id) || { name: id }).name;
const itemName = (id) => (META.items.find((m) => m.id === id) || { name: id }).name;

// ------------------------------------------------------------------ field state
function fieldHtml() {
  const f = FORM.field, o = BOARD.options;
  const turns = (id, val, extra) => `<select class="ps-turns-sel" data-field="${id}" aria-label="残りターン">${[1, 2, 3, 4, 5, 6, 7, 8].map((n) => `<option value="${n}" ${n === val ? "selected" : ""}>残り ${n}</option>`).join("")}${extra || ""}</select>`;
  const p = prevForm();
  const fchg = (k) => (p && JSON.stringify(p.field[k]) !== JSON.stringify(f[k]) ? " chg" : "");
  const sideBox = (idx, title) => {
    const s = FORM.sides[idx], ps = p ? p.sides[idx] : null;
    const timed = o.timed.map((c) => {
      const on = s[c.id] != null, d = !ps ? "" : ((ps[c.id] != null) !== on ? " chg" : "");
      return `<span class="ps-cond${on ? " on" : ""}${d}"><button type="button" data-cond="${idx}:${c.id}" aria-pressed="${on}">${esc(c.name)}</button>${on ? `<select data-condturns="${idx}:${c.id}" aria-label="${esc(c.name)} の残りターン">${[1, 2, 3, 4, 5, 6, 7, 8].map((n) => `<option value="${n}" ${n === s[c.id] ? "selected" : ""}>残り ${n}</option>`).join("")}</select>` : ""}</span>`;
    }).join("");
    const layered = o.layered.map((c) => {
      const n = s[c.id] || 0, d = !ps ? "" : ((ps[c.id] || 0) !== n ? " chg" : "");
      return `<span class="ps-cond${n ? " on" : ""}${d}"><button type="button" data-layer="${idx}:${c.id}:${c.max}" aria-pressed="${n > 0}">${esc(c.name)}${c.max > 1 ? ` ${n}/${c.max}` : ""}</button></span>`;
    }).join("");
    return `<div class="ps-side"><h4>${title}</h4><div class="ps-conds">${timed}${layered}</div></div>`;
  };
  return `<div class="ps-fgrid">
    <label class="ps-f${fchg("weather")}${fchg("weatherTurns")}"><span>天気</span><select data-field="weather"><option value="">なし</option>${o.weather.map((w) => `<option value="${w.id}" ${f.weather === w.id ? "selected" : ""}>${esc(w.name)}</option>`).join("")}</select></label>
    ${f.weather ? turns("weatherTurns", f.weatherTurns) : ""}
    <label class="ps-f${fchg("terrain")}${fchg("terrainTurns")}"><span>フィールド</span><select data-field="terrain"><option value="">なし</option>${o.terrain.map((w) => `<option value="${w.id}" ${f.terrain === w.id ? "selected" : ""}>${esc(w.name)}</option>`).join("")}</select></label>
    ${f.terrain ? turns("terrainTurns", f.terrainTurns) : ""}
    <label class="ps-f${fchg("trickRoom")}"><span>トリックルーム</span><select data-field="trickRoom"><option value="0">なし</option>${[1, 2, 3, 4, 5].map((n) => `<option value="${n}" ${f.trickRoom === n ? "selected" : ""}>残り ${n}</option>`).join("")}</select></label>
  </div>
  <div class="ps-sides">${sideBox(1, "相手側")}${sideBox(0, "自分側")}</div>`;
}

// ------------------------------------------------------------------ events
function eventsHtml() {
  if (CUR === 0) return `<p class="note">ターン 1 の始めの局面です。次のターンから、前のターンに起きたことを入れられます。</p>`;
  const p = prevForm(), ev = FORM.events;
  const refName = (r) => (r.startsWith("m:") ? `自分の${mineName(Number(r.slice(2)))}` : `相手の${oppName(r.slice(2))}`);
  const actives = [...p.mineActive.filter((x) => x != null).map((i) => ({ id: `m:${i}`, name: `自分の${mineName(i)}` })),
    ...p.theirActive.filter((x) => x).map((s) => ({ id: `t:${s}`, name: `相手の${oppName(s)}` }))];
  const opts = (sel) => `<option value="">選ぶ</option>${actives.map((a) => `<option value="${esc(a.id)}" ${a.id === sel ? "selected" : ""}>${esc(a.name)}</option>`).join("")}`;
  const orderList = ev.order.map((o, i) => `<li>${esc(refName(o.first))} が先、${esc(refName(o.second))} が後${o.firstMove || o.secondMove ? `（${esc(o.firstMove ? moveName(o.firstMove) : "?")} / ${esc(o.secondMove ? moveName(o.secondMove) : "?")}）` : ""} <button type="button" class="ps-x" data-del-order="${i}">消す</button></li>`).join("");
  const dmgList = ev.damage.map((d, i) => `<li>相手の${esc(oppName(d.attacker))} の ${esc(moveName(d.move))} で、自分の${esc(mineName(d.target))} が <b>${d.amount}</b> ダメージ${d.crit ? "（急所）" : ""} <button type="button" class="ps-x" data-del-dmg="${i}">消す</button></li>`).join("");
  const foes = p.theirActive.filter((x) => x), mine = p.mineActive.filter((x) => x != null);
  const dropHint = mine.map((i) => { const was = p.mine[String(i)].hp, now = FORM.mine[String(i)] ? FORM.mine[String(i)].hp : was; return was !== now ? `${mineName(i)} の HP は ${was} → ${now}（${was - now} 減）` : ""; }).filter(Boolean).join(" ／ ");
  const results = [...(BOARD.spread.map((s) => `${esc(s.species)} の${esc(META.statNames[META.stats.indexOf(s.stat)])}の配分を ${s.before} → ${s.after} にしました（${s.low}〜${s.high} が可能。実数値 ${s.statLow}〜${s.statHigh}）<small>${s.because.map(esc).join("、")}</small>`)),
    ...BOARD.lines.map(esc)];
  return `<p class="note">ターン ${p.turn} の始めの局面で計算します。途中で能力変化や天気が変わったときは、外れることがあります。</p>
    <div class="ps-ev">
      <h4>どちらが先に動いたか</h4>
      <div class="ps-evrow"><select id="ordA" aria-label="先に動いた体">${opts("")}</select><span>が先、</span><select id="ordB" aria-label="後に動いた体">${opts("")}</select><span>が後</span>
        <button type="button" class="ghost small" id="addOrder">この順を記録する</button></div>
      <details class="ps-more"><summary>先制技のときは、使った技も</summary><div class="ps-evrow"><input id="ordMA" list="moveList" placeholder="先の体の技" autocomplete="off"><input id="ordMB" list="moveList" placeholder="後の体の技" autocomplete="off"></div></details>
      <ul class="ps-evlist">${orderList}</ul>
      <h4>受けたダメージ</h4>
      <div class="ps-evrow"><span>相手の</span><select id="dmgA" aria-label="攻撃した相手">${foes.map((s) => `<option value="${esc(s)}">${esc(oppName(s))}</option>`).join("")}</select>
        <input id="dmgM" list="moveList" placeholder="技" autocomplete="off" aria-label="技"><span>で、自分の</span>
        <select id="dmgT" aria-label="受けた自分の体">${mine.map((i) => `<option value="${i}">${esc(mineName(i))}</option>`).join("")}</select>
        <input id="dmgN" type="number" min="1" inputmode="numeric" placeholder="ダメージ" aria-label="ダメージ"><label class="ps-check"><input type="checkbox" id="dmgC"> 急所</label>
        <button type="button" class="ghost small" id="addDmg">このダメージを記録する</button></div>
      ${dropHint ? `<div class="ps-hint">${esc(dropHint)}（1 回の攻撃だけで減ったときは、この数がダメージです）</div>` : ""}
      <ul class="ps-evlist">${dmgList}</ul>
      <div class="ps-evres" id="evRes">${results.length ? `<ul>${results.map((r) => `<li>${r}</li>`).join("")}</ul>` : ""}</div>
    </div>`;
}

// ------------------------------------------------------------------ result
function actionRows(rows) {
  const shown = rows.filter((r, i) => i === 0 || r.p >= 0.005);
  return shown.map((r) => `<div class="ps-act"><span class="t">${esc(r.text)}</span><span class="p">${pct(r.p, r.p < 0.1 ? 1 : 0)}</span><div class="pbar"><i style="width:${Math.round(r.p * 100)}%"></i></div></div>`).join("");
}
function renderResult(res) {
  const o = res.ours, t = res.theirs;
  const cols = o.slots.map((s) => `<div><h3>${esc(s.name || `${s.slot + 1} 体目`)} の手</h3>${actionRows(s.actions)}</div>`).join("");
  const theirs = t.slots.map((s) => `<div><h3>相手の${esc(s.name)} の手</h3>${actionRows(s.actions)}</div>`).join("");
  $("result").hidden = false;
  $("result").innerHTML = `
    <div class="rk-value"><b class="n">${pct(res.value, 1)}</b><span>あなたの勝率の見積もり（ターン ${res.turn}。相手の型の推定が当たっているとしたとき。読んだ時間 ${Math.round(res.seconds)} 秒）</span></div>
    <p class="note">％は、その手を選ぶ確率です。混ぜて選ぶ手は、確率どおりに選びます。</p>
    <div class="rk-cols">${cols}</div>
    <details class="ps-more"><summary>2 体の手の組み合わせ（上位）</summary>${actionRows(o.joint)}</details>
    <details class="ps-more"><summary>相手が打ちそうな手（この読みの中での相手の混合）</summary><div class="rk-cols">${theirs}</div></details>
    ${res.notes.length ? `<ul class="rk-msgs">${res.notes.map((n) => `<li class="warn">${esc(n)}</li>`).join("")}</ul>` : ""}
    <ul class="rk-msgs"><li class="warn"><b>相手の型の推定が外れると、この結果は変わります。</b></li></ul>`;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ------------------------------------------------------------------ render
function renderAll() {
  $("setupBox").hidden = !!(BOARD && BOARD.started);
  $("boardBox").hidden = !(BOARD && BOARD.started);
  if (!(BOARD && BOARD.started)) { renderSetup(); return; }
  const turns = BOARD.turns;
  $("turnBar").innerHTML = turns.map((t, i) => `<button type="button" class="ps-tab${i === CUR ? " on" : ""}" data-turn="${i}" aria-current="${i === CUR}">ターン ${t.form.turn}</button>`).join("") +
    (turns.length > 1 && CUR === turns.length - 1 ? `<button type="button" class="ghost small" id="dropTurn">このターンを消す</button>` : "");
  $("diffNote").textContent = CUR === 0 ? "ターン 1 の始めの局面です。先発の特性（いかくなど）は自動で入れました。違うところだけ直してください。"
    : "前のターンの入力を引き継いでいます。変わったところだけ直してください（直した所に色が付きます）。";
  const f = FORM;
  $("oppActive").innerHTML = [0, 1].map((k) => (f.theirActive[k] ? oppCard(f.theirActive[k], k) : emptySlot("t", k))).join("");
  const oppBench = Object.keys(f.theirs).filter((s) => !f.theirActive.includes(s));
  $("oppBench").innerHTML = oppBench.map((s) => oppCard(s, null)).join("");
  $("oppBenchSum").textContent = `相手の裏（見えた体 ${oppBench.length} 体）`;
  const unseen = BOARD.oppSix.filter((s) => !f.theirs[s.id]);
  $("oppUnseen").innerHTML = unseen.length ? `<span class="lab">まだ見えていない</span>${unseen.map((s) => `<button type="button" class="ps-x" data-appear="${esc(s.id)}">${esc(s.name)} が出てきた</button>`).join("")}` : "";
  $("mineActive").innerHTML = [0, 1].map((k) => (f.mineActive[k] != null ? mineCard(f.mineActive[k], k) : emptySlot("m", k))).join("");
  const mineBench = BOARD.brought.filter((i) => !f.mineActive.includes(i));
  $("mineBench").innerHTML = mineBench.map((i) => mineCard(i, null)).join("");
  $("mineBenchSum").textContent = `自分の裏（${mineBench.length} 体）`;
  $("fieldState").innerHTML = fieldHtml();
  $("fieldBox").open = FOLD.field; $("mineBenchBox").open = FOLD.mineBench; $("oppBenchBox").open = FOLD.oppBench;
  $("events").innerHTML = eventsHtml();
  renderDerived();
  renderRead();
}
function emptySlot(side, slot) {
  const pool = side === "m" ? BOARD.brought.filter((j) => FORM.mine[String(j)].hp > 0 && !FORM.mineActive.includes(j)).map((j) => ({ id: j, name: mineName(j) }))
    : Object.keys(FORM.theirs).filter((k) => !FORM.theirs[k].fainted && !FORM.theirActive.includes(k)).map((k) => ({ id: k, name: oppName(k) }));
  return `<article class="ps-mon empty"><div class="ps-head"><div class="ps-who">${slotSelect(side, slot, null, pool)}<span class="ps-line">場が空いています</span></div></div></article>`;
}
function renderDerived() {
  const t = BOARD.turns[CUR];
  const msgs = [...t.problems.map((p) => `<li>${esc(p)}</li>`), ...t.notes.map((n) => `<li class="warn">${esc(n)}</li>`)];
  if (BOARD.lines.length) BOARD.lines.forEach((l) => msgs.push(`<li class="warn">${esc(l)}</li>`));
  $("turnMsg").innerHTML = msgs.join("");
  document.querySelectorAll("[data-est]").forEach((el) => { el.outerHTML = estimateHtml(el.dataset.est); });
  document.querySelectorAll("[data-hphint]").forEach((el) => { el.textContent = hpHint(el.dataset.hphint); });
  const res = $("evRes");
  if (res && CUR > 0) {
    const results = [...BOARD.spread.map((s) => `${esc(s.species)} の${esc(META.statNames[META.stats.indexOf(s.stat)])}の配分を ${s.before} → ${s.after} にしました（${s.low}〜${s.high} が可能。実数値 ${s.statLow}〜${s.statHigh}）<small>${s.because.map(esc).join("、")}</small>`), ...BOARD.lines.map(esc)];
    res.innerHTML = results.length ? `<ul>${results.map((r) => `<li>${r}</li>`).join("")}</ul>` : "";
  }
  const running = BOARD.read && BOARD.read.state === "running";
  $("readTurn").disabled = !t.readable || running;
  $("readTurn").textContent = running ? "読んでいます…" : `このターンを読む（${Math.round(BOARD.seconds || 40)} 秒）`;
  $("nextTurn").disabled = running;
  $("barSummary").textContent = running ? "" : (!BOARD.reader ? "この画面では読めません" : (t.problems.length ? t.problems[0] : (t.readable ? (CUR < BOARD.turns.length - 1 ? "過去のターンも読めます" : "入れ終えたら読みます。次のターンへ進むと、いまの入力を引き継ぎます") : "場に出ている体を入れてください")));
}
function renderRead() {
  const job = BOARD.read || { state: "idle" };
  $("readNote").textContent = `「このターンを読む」を押すと、ここに出ます（${Math.round(BOARD.seconds || 40)} 秒ほど）。入力を変えると前の読みは消えます。`;
  $("result").hidden = true; $("readError").hidden = true;
  if (job.state === "running") watch();
  else if (job.state === "done" && job.turn === CUR) renderResult(job.result);
  else if (job.state === "error") { $("readError").hidden = false; $("readError").textContent = job.error; }
}

// ------------------------------------------------------------------ edits
function setValue(el) {
  const side = el.dataset.side, key = el.dataset.key, f = el.dataset.f;
  const box = side === "m" ? FORM.mine[key] : FORM.theirs[key];
  let v;
  if (el.type === "checkbox") v = el.checked;
  else if (el.type === "number") v = el.value === "" ? "" : Number(el.value);
  else v = el.value;
  if (f === "boost") { if (v) box.boosts[el.dataset.b] = v; else delete box.boosts[el.dataset.b]; }
  else if (f === "status" || f === "colour" || f === "ability") box[f] = v || null;
  else if (f === "item") {
    if (v === "__other") { const name = prompt("持ち物の名前"); const id = name ? itemByName.get(name.trim().toLowerCase()) : null; box.item = id || null; if (name && !id) alert(`持ち物 ${name} がわかりません`); structural = true; }
    else { box.item = v || null; if (!v) box.itemGone = false; structural = true; }
  }
  else if (f === "pct") { box.pct = v === "" ? 100 : v; if (v === 20 || v === 50 || box.colour) structural = true; if (v !== 20 && v !== 50) box.colour = null; }
  else if (f === "hp") box.hp = v === "" ? 0 : v;
  else { box[f] = v; if (f === "fainted" || f === "itemGone") structural = true; }
  if (f === "hp" && box.hp === 0) structural = true;
  if (f === "boost") el.closest("details").querySelector("summary b").textContent = boostSummary(box.boosts);
}
let structural = false;
function schedule(now) {
  clearTimeout(saveTimer);
  const go = async () => {
    const seq = ++saveSeq, rebuild = structural; structural = false;
    try {
      const resp = await api("/api/board/save", { index: CUR, form: FORM });
      if (seq !== saveSeq) return;
      resp.read = BOARD.read && BOARD.read.state === "running" ? BOARD.read : resp.read;
      BOARD = resp; FORM = clone(BOARD.turns[CUR].form);
      if (rebuild) renderAll(); else { renderDerived(); markRead(); }
    } catch (e) { if (seq === saveSeq) { $("turnMsg").innerHTML = `<li>${esc(e.message)}</li>`; $("readTurn").disabled = true; } }
  };
  if (now) go(); else saveTimer = setTimeout(go, 500);
}
function markRead() { $("result").hidden = true; }
document.addEventListener("input", (e) => {
  const el = e.target;
  if (el.dataset && el.dataset.f && el.type === "number") { setValue(el); schedule(false); }
});
document.addEventListener("change", (e) => {
  const el = e.target;
  if (el.dataset && el.dataset.f && el.type !== "number") { setValue(el); schedule(true); }
  else if (el.dataset && el.dataset.slot) {
    const side = el.dataset.slot[0], slot = Number(el.dataset.slot[1]);
    if (side === "m") FORM.mineActive[slot] = el.value === "" ? null : Number(el.value);
    else FORM.theirActive[slot] = el.value || null;
    structural = true; schedule(true);
  } else if (el.dataset && el.dataset.field) {
    const k = el.dataset.field; let v = el.value;
    if (k === "weather" || k === "terrain") v = v || null; else v = Number(v);
    FORM.field[k] = v; structural = true; schedule(true);
  } else if (el.dataset && el.dataset.condturns) {
    const [idx, id] = el.dataset.condturns.split(":"); FORM.sides[Number(idx)][id] = Number(el.value); schedule(true);
  } else if (el.dataset && el.dataset.addMove) {
    const sid = el.dataset.addMove; let id = el.value;
    if (id === "__other") { const name = prompt("使った技の名前"); id = name ? moveByName.get(name.trim().toLowerCase()) : null; if (name && !id) alert(`技 ${name} がわかりません`); }
    if (id) { const mv = FORM.theirs[sid].moves; if (!mv.includes(id)) mv.push(id); structural = true; schedule(true); } else el.value = "";
  }
});
document.addEventListener("click", async (e) => {
  const t = e.target.closest("button"); if (!t) return;
  if (t.dataset.turn != null) { CUR = Number(t.dataset.turn); enterTurn(); renderAll(); }
  else if (t.dataset.choose) {
    clearTimeout(saveTimer);
    try {
      BOARD = await api("/api/board/save", { index: CUR, form: FORM });
      BOARD = await api("/api/board/choose", { species: t.dataset.choose, alternative: Number(t.dataset.a) });
      FORM = clone(BOARD.turns[CUR].form); renderAll();
    } catch (err) { $("turnMsg").innerHTML = `<li>${esc(err.message)}</li>`; }
  }
  else if (t.id === "dropTurn") { if (confirm("最後のターンの入力を消します。")) { BOARD = await api("/api/board/drop", {}); CUR = BOARD.turns.length - 1; enterTurn(); renderAll(); } }
  else if (t.dataset.rmMove) { const mv = FORM.theirs[t.dataset.rmMove].moves; mv.splice(mv.indexOf(t.dataset.m), 1); structural = true; schedule(true); }
  else if (t.dataset.appear) {
    FORM.theirs[t.dataset.appear] = { pct: 100, colour: null, status: null, boosts: {}, mega: false, fainted: false, moves: [], item: null, itemGone: false, ability: null };
    structural = true; schedule(true);
  } else if (t.dataset.cond) {
    const [idx, id] = t.dataset.cond.split(":"), s = FORM.sides[Number(idx)];
    if (s[id] != null) delete s[id]; else s[id] = (BOARD.options.timed.find((c) => c.id === id) || { turns: 5 }).turns;
    structural = true; schedule(true);
  } else if (t.dataset.layer) {
    const [idx, id, max] = t.dataset.layer.split(":"), s = FORM.sides[Number(idx)];
    const n = ((s[id] || 0) + 1) % (Number(max) + 1);
    if (n) s[id] = n; else delete s[id];
    structural = true; schedule(true);
  } else if (t.id === "addOrder") {
    const a = $("ordA").value, b = $("ordB").value;
    if (!a || !b || a === b) { $("evRes").innerHTML = `<ul><li>先に動いた体と後に動いた体を、別々に選んでください</li></ul>`; return; }
    const mv = (id) => { const x = $(id).value.trim(); return x ? (moveByName.get(x.toLowerCase()) || null) : null; };
    FORM.events.order.push({ first: a, second: b, firstMove: mv("ordMA"), secondMove: mv("ordMB") });
    structural = true; schedule(true);
  } else if (t.id === "addDmg") {
    const mvName = $("dmgM").value.trim(), move = mvName ? moveByName.get(mvName.toLowerCase()) : null, n = Number($("dmgN").value);
    if (!move) { $("evRes").innerHTML = `<ul><li>技の名前を選んでください</li></ul>`; return; }
    if (!(n >= 1)) { $("evRes").innerHTML = `<ul><li>ダメージの数を入れてください</li></ul>`; return; }
    FORM.events.damage.push({ attacker: $("dmgA").value, move, target: Number($("dmgT").value), amount: n, crit: $("dmgC").checked });
    structural = true; schedule(true);
  } else if (t.dataset.delOrder != null) { FORM.events.order.splice(Number(t.dataset.delOrder), 1); structural = true; schedule(true); }
  else if (t.dataset.delDmg != null) { FORM.events.damage.splice(Number(t.dataset.delDmg), 1); structural = true; schedule(true); }
});
$("nextTurn").onclick = async () => {
  clearTimeout(saveTimer); schedule(true);
  await new Promise((r) => setTimeout(r, 50));
  try { BOARD = await api("/api/board/next", {}); CUR = BOARD.turns.length - 1; enterTurn(); renderAll(); window.scrollTo({ top: 0 }); }
  catch (e) { $("turnMsg").innerHTML = `<li>${esc(e.message)}</li>`; }
};
$("readTurn").onclick = async () => {
  $("readError").hidden = true; $("result").hidden = true;
  try { BOARD.read = await api("/api/board/read", { index: CUR }); }
  catch (e) { $("readError").hidden = false; $("readError").textContent = e.message; $("readBox").scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  $("readBox").scrollIntoView({ behavior: "smooth", block: "start" });
  watch();
};
function watch() {
  $("progress").hidden = false; renderDerived();
  if (poll) clearInterval(poll);
  const tick = async () => {
    const job = await api("/api/board/job"); BOARD.read = job;
    if (job.state === "running") {
      const budget = job.seconds || 40;
      $("pbar").style.width = Math.min(100, 100 * job.elapsed / budget) + "%";
      $("ptext").textContent = `読んでいます… ${Math.round(job.elapsed)} / ${Math.round(budget)} 秒`;
    } else {
      clearInterval(poll); poll = null; $("progress").hidden = true; renderDerived();
      if (job.state === "done") renderResult(job.result);
      if (job.state === "error") { $("readError").hidden = false; $("readError").textContent = job.error; }
    }
  };
  poll = setInterval(tick, 700); tick();
}

// ------------------------------------------------------------------ start
(async function init() {
  META = await api("/api/meta");
  STATE = await api("/api/state");
  BOARD = await api("/api/board");
  $("moveList").innerHTML = META.moves.map((m) => `<option value="${esc(m.name)}">`).join("");
  $("itemList").innerHTML = META.items.map((i) => `<option value="${esc(i.name)}">`).join("");
  META.items.forEach((i) => { itemByName.set(i.name.toLowerCase(), i.id); itemByName.set(i.id, i.id); });
  META.moves.forEach((m) => { moveByName.set(m.name.toLowerCase(), m.id); moveByName.set(m.id, m.id); });
  $("aboutEvent").textContent = META.event;
  $("aboutRead").textContent = (META.model ? `評価モデル: ${META.model}。` : "") + `局面の読みは 1 手 ${Math.round(META.readSeconds || 40)} 秒です。`;
  if (BOARD.started) { CUR = Math.min(Number(QUERY.get("turn") || BOARD.turns.length) - 1, BOARD.turns.length - 1); CUR = Math.max(CUR, 0); enterTurn(); }
  renderAll();
})().catch((e) => { document.body.insertAdjacentHTML("beforeend", `<p class="rk-error">${esc(e.message)}</p>`); });
})();
