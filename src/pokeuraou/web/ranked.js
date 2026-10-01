"use strict";
// 選出の検討の画面（IKA-407）。サーバ（rankedweb.RankedApp）の JSON を描くだけで、推定の中身は持たない。
(function () {
const TYPE_COLORS = {
  normal: "#9fa19f", fire: "#e62829", water: "#2980ef", electric: "#fac000", grass: "#3fa129", ice: "#3dcef3",
  fighting: "#ff8000", poison: "#9141cb", ground: "#915121", flying: "#81b9ef", psychic: "#ef4179", bug: "#91a119",
  rock: "#afa981", ghost: "#704170", dragon: "#5060e1", dark: "#624d4e", steel: "#60a1b8", fairy: "#ef70ef",
};
const metaSprite = document.querySelector('meta[name="sprite-url"]');
const SPRITE_URL = metaSprite ? metaSprite.getAttribute("content") : "";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// "Floette (Eternal)": the localiser writes base (forme) -- the forme in small type, as the game page does
const nameHtml = (n) => {
  const [base, ...rest] = String(n == null ? "" : n).split(" (");
  return rest.length ? `${esc(base)}<small class="forme">(${esc(rest.join(" (").replace(/\)$/, ""))})</small>` : esc(base);
};
const pct = (x, d) => (x * 100).toFixed(d == null ? 0 : d) + "%";

let META = null;
let STATE = { mine: null, opponent: [], teamProblems: [] };
let editing = new Set();
let poll = null;
let itemByName = new Map(), moveByName = new Map();

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "エラー");
  return data;
}

// ------------------------------------------------------------------ theme and gear
let theme = "auto";
try { theme = localStorage.getItem("pokeuraou-theme") || "auto"; } catch (_) { /* storage may be blocked */ }
// ?theme=dark and ?edit=0,2 open the page in a given state (for a screenshot)
const QUERY = new URLSearchParams(location.search);
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
function gear(open) {
  $("settings").hidden = !open; $("settingsBackdrop").hidden = !open;
  $("gear").setAttribute("aria-expanded", String(open));
}
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

function typeDots(sp) { return `<span class="rk-types">${sp.types.map((t) => `<i class="rk-type" style="--c:${TYPE_COLORS[t] || ""}" title="${esc(t)}"></i>`).join("")}</span>`; }

function spBars(v) {
  const max = META.spMax;
  return `<div class="rk-sp" role="img" aria-label="配分 ${esc(v.sp.join("-"))}">${v.sp.map((x, i) =>
    `<div><span class="bar"><i style="height:${Math.round(100 * x / max)}%"></i></span><b>${x}</b>${esc(META.statNames[i])}</div>`).join("")}</div>
    <div class="rk-sptotal">合計 ${v.spTotal} / ${META.spLimit}</div>`;
}

function unlearnedTag(v) {
  if (!v.unlearned || !v.unlearned.length) return "";
  const kinds = { species: "種族", move: "技", item: "持ち物", ability: "特性" };
  const list = v.unlearned.map((u) => `${kinds[u.kind]} ${u.name}`).join("、");
  return `<span class="rk-tag nolearn" title="評価モデルの学習データに出ていない id です。この行の値は雑音で、読みの答えは当てになりません">評価モデルが学んでいない: ${esc(list)}</span>`;
}

function renderSet(v, role, index) {
  const unlearnedMoves = new Set((v.unlearned || []).filter((u) => u.kind === "move").map((u) => u.id));
  const moves = v.moves.map((m) => `<li class="${unlearnedMoves.has(m.id) ? "nolearn" : ""}">${esc(m.name)}</li>`).join("");
  let tags = "", alts = "", edit = "", editor = "", prov = "", thin = "";
  if (role === "opp") {
    const t = [];
    prov = `<div class="rk-prov"><div><span class="k">推定した型</span><span class="v">${esc(v.kindLabel)}</span></div>
      <div><span class="k">型の根拠</span><span class="v">${esc(v.basisLabel)}</span></div>
      <div><span class="k">配分の出どころ</span><span class="v">${v.spProvisional ? "<b>仮の配分</b>：" : ""}${esc(v.spLabel)}</span></div></div>`;
    if (v.thin && v.thin.length) thin = `<div class="rk-thin"><b>根拠が薄い推定です</b>${v.thin.map((x) => `<span>${esc(x)}</span>`).join("")}</div>`;
    if (v.overridden.length) t.push(`<span class="rk-tag edit">上書き：${esc(v.overridden.map((k) => ({ ability: "特性", item: "持ち物", nature: "性格", moves: "技", sp: "配分" }[k] || k)).join("・"))}</span>`);
    t.push(unlearnedTag(v));
    (v.notes || []).forEach((n) => t.push(`<span class="rk-tag">${esc(n)}</span>`));
    tags = t.length ? `<div class="rk-tags">${t.join("")}</div>` : "";
    if (v.alternatives && v.alternatives.length > 1) {
      alts = `<div class="rk-alts" role="group" aria-label="別の型">${v.alternatives.map((a) =>
        `<button type="button" class="rk-alt" data-i="${index}" data-a="${a.index}" aria-pressed="${String(v.chosen === a.index)}">${esc(a.label)}（${a.count} 体）<small>${esc(a.moves.join("・"))}</small></button>`).join("")}</div>`;
    }
    edit = `<button type="button" class="ghost small rk-edit" data-edit="${index}">${editing.has(index) ? "閉じる" : "編集"}</button>`;
    if (editing.has(index)) editor = renderEditor(v, index);
  } else {
    const u = unlearnedTag(v);
    if (u) tags = `<div class="rk-tags">${u}</div>`;
  }
  return `<article class="rk-set ${role}${thin ? " thin" : ""}">
    ${thin}
    <div class="rk-head">${art(v.species, "md")}<div class="rk-name">${nameHtml(v.species.name)}${typeDots(v.species)}</div></div>
    <div class="rk-line"><span>特性 <b>${esc(v.ability ? v.ability.name : "—")}</b></span><span>持ち物 <b>${esc(v.item ? v.item.name : "なし")}</b></span><span>性格 <b>${esc(v.nature.name)}</b></span></div>
    <ul class="rk-moves">${moves}</ul>
    ${spBars(v)}
    ${prov}${tags}${alts}${edit}${editor}
  </article>`;
}
function renderEditor(v, index) {
  const abil = v.abilityChoices.map((a) => `<option value="${esc(a.id)}" ${v.ability && v.ability.id === a.id ? "selected" : ""}>${esc(a.name)}</option>`).join("");
  const nat = META.natures.map((n) => `<option value="${esc(n.id)}" ${v.nature.id === n.id ? "selected" : ""}>${esc(n.name)}</option>`).join("");
  const moves = [0, 1, 2, 3].map((k) => `<label class="field">技 ${k + 1}<input data-f="move" data-k="${k}" list="moveList" value="${esc(v.moves[k] ? v.moves[k].name : "")}" autocomplete="off"></label>`);
  const sps = v.sp.map((x, i) => `<label class="field">${esc(META.statNames[i])}<input data-f="sp" data-k="${i}" type="number" min="0" max="${META.spMax}" inputmode="numeric" value="${x}"></label>`).join("");
  return `<div class="rk-ed" data-ed="${index}">
    <div class="row">
      <label class="field">特性<select data-f="ability">${abil}</select></label>
      <label class="field">性格<select data-f="nature">${nat}</select></label>
    </div>
    <label class="field">持ち物（空で「なし」）<input data-f="item" list="itemList" value="${esc(v.item ? v.item.name : "")}" autocomplete="off"></label>
    <div class="row">${moves[0]}${moves[1]}</div>
    <div class="row">${moves[2]}${moves[3]}</div>
    <div class="sprow">${sps}</div>
    <div class="sptot" data-sptot>合計 ${v.spTotal} / ${META.spLimit}</div>
    <div class="rk-error" data-ederr hidden></div>
    <div class="sendbar"><span></span><button type="button" class="primary small" data-save="${index}">この型にする</button></div>
  </div>`;
}

// ------------------------------------------------------------------ render
function renderMine(resp) {
  const msgs = [];
  if (resp) {
    (resp.problems || []).forEach((p) => msgs.push(`<li>${esc(p)}</li>`));
    (resp.warnings || []).forEach((p) => msgs.push(`<li class="warn">${esc(p)}</li>`));
  }
  $("mineMsg").innerHTML = msgs.join("");
  const team = resp ? resp.team : STATE.mine;
  $("mineSets").innerHTML = team ? team.map((v, i) => renderSet(v, "mine", i)).join("") : "";
  $("mineSummary").textContent = "";
  // loaded: the six fold into one row (the paste and the cards are one tap away)
  $("mineEdit").hidden = !!team;
  $("mineBrief").hidden = !team;
  $("mineFold").hidden = !team;
  $("mineChips").innerHTML = team ? team.map((v) => `<span class="rk-chip">${art(v.species, "xs")}${nameHtml(v.species.name)}</span>`).join("") : "";
  refreshSolve();
}

function renderOpp() {
  const sets = STATE.opponent || [];
  const inputs = document.querySelectorAll("#speciesInputs input");
  if (sets.length && Array.from(inputs).every((i) => !i.value)) sets.forEach((v, i) => { if (v && inputs[i]) inputs[i].value = v.species.name; });
  $("oppSets").innerHTML = sets.map((v, i) => (v ? renderSet(v, "opp", i) : "")).join("");
  $("oppMsg").innerHTML = (STATE.teamProblems || []).map((p) => `<li>${esc(p)}</li>`).join("");
  const note = $("estNote");
  note.hidden = !sets.length;
  $("estEvent").textContent = META.event;
  $("estSpread").textContent = META.spreadNote;
  refreshSolve();
}

function refreshSolve() {
  const ready = STATE.mine && (STATE.opponent || []).length === 6 && STATE.opponent.every((v) => v && v.moves.length) && !(STATE.teamProblems || []).length;
  const running = STATE.job && STATE.job.state === "running";
  $("solve").disabled = !ready || running;
  let why = "";
  if (!STATE.mine) why = "自分の構築を読み込んでください";
  else if ((STATE.opponent || []).length !== 6) why = "相手の 6 種族を入れて「型を推定する」を押してください";
  else if ((STATE.teamProblems || []).length) why = "相手の型の問題を直してください";
  else if (!ready) why = "技が入っていない相手がいます";
  $("readSummary").textContent = running ? "" : why;
}

function renderResult(r) {
  const mine = r.mine, theirs = r.theirs;
  const row = (s, cls) => `<div class="rk-selrow ${cls}"><div class="who"><span class="lab">先発</span><b>${s.leads.map(nameHtml).join("・")}</b><span class="sep">／</span><span class="lab">裏</span>${s.back.map(nameHtml).join("・")}</div><span class="pct" title="この選出を選ぶ確率">${pct(s.p, s.p < 0.1 ? 1 : 0)}</span><div class="pbar"><i style="width:${Math.round(s.p * 100)}%"></i></div></div>`;
  const bring = (side) => `<ul class="rk-bring"><li class="head"><span>体</span><span class="n">選ぶ確率</span><span class="n">先発の確率</span></li>${side.members.map((m) =>
    `<li><span>${nameHtml(m.species)}</span><span class="n">${pct(m.bring)}</span><span class="n">${pct(m.lead)}</span></li>`).join("")}</ul>`;
  const hints = [];
  const prov = r.estimated.filter((e) => e.provisional).map((e) => e.species);
  if (prov.length) hints.push(`仮の配分で読んでいます：${prov.join("、")}`);
  const part = r.estimated.filter((e) => e.kind.startsWith("部分ごと")).map((e) => e.species);
  if (part.length) hints.push(`最頻の型が少数なので、部分ごとの最頻で読んでいます：${part.join("、")}`);
  const edited = r.estimated.filter((e) => e.overridden.length).map((e) => e.species);
  if (edited.length) hints.push(`あなたが上書きした型：${edited.join("、")}`);
  const un = [];
  r.estimated.forEach((e) => { if (e.unlearned.length) un.push(`相手 ${e.species}：${e.unlearned.map((u) => u.name).join("・")}`); });
  r.mineUnlearned.forEach((e) => { if (e.unlearned.length) un.push(`自分 ${e.species}：${e.unlearned.map((u) => u.name).join("・")}`); });
  if (un.length) hints.push(`評価モデルが学んでいない種族・技（その行の値は雑音）：${un.join(" ／ ")}`);
  $("result").hidden = false;
  $("result").innerHTML = `
    <div class="rk-value"><b class="n">${pct(r.value, 1)}</b><span>あなたの勝率の見積もり（相手の型の推定が当たっているとしたとき。読んだ時間 ${Math.round(r.elapsed || r.seconds)} 秒）</span></div>
    <div class="rk-cols">
      <div><h3>おすすめの選出 <small>％は、その選出を選ぶ確率（混ぜて選びます）</small></h3>${mine.selections.map((s) => row(s, "mine")).join("")}<h3>体ごとの確率</h3>${bring(mine)}</div>
      <div><h3>相手が選びそうな選出 <small>推定した型での最善。％は、その選出を選ぶ確率</small></h3>${theirs.selections.map((s) => row(s, "opp")).join("")}<h3>体ごとの確率</h3>${bring(theirs)}</div>
    </div>
    <div class="rk-hints rk-msgs">${hints.map((h) => `<li class="warn">${esc(h)}</li>`).join("")}<li class="warn"><b>相手の型の推定が外れると、この結果は変わります。</b></li></div>`;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ------------------------------------------------------------------ actions
async function loadMine() {
  $("loadMine").disabled = true;
  try { const resp = await api("/api/mine", { text: $("paste").value }); STATE.mine = resp.team; $("result").hidden = true; renderMine(resp); }
  catch (e) { $("mineMsg").innerHTML = `<li>${esc(e.message)}</li>`; }
  $("loadMine").disabled = false;
}
async function fillOpp() {
  const names = Array.from(document.querySelectorAll("#speciesInputs input")).map((i) => i.value.trim()).filter(Boolean);
  $("fillOpp").disabled = true;
  try { editing = new Set(); STATE = Object.assign(STATE, await api("/api/opponent", { species: names })); $("result").hidden = true; renderOpp(); }
  catch (e) { $("oppMsg").innerHTML = `<li>${esc(e.message)}</li>`; }
  $("fillOpp").disabled = false;
}
function resolve(map, text) { const k = text.trim().toLowerCase(); return k ? (map.get(k) || null) : ""; }

async function save(index) {
  const root = document.querySelector(`.rk-ed[data-ed="${index}"]`), v = STATE.opponent[index];
  const err = root.querySelector("[data-ederr]");
  const val = (f) => root.querySelector(`[data-f="${f}"]`).value;
  const patch = {};
  if (v.ability && val("ability") !== v.ability.id) patch.ability = val("ability");
  if (val("nature") !== v.nature.id) patch.nature = val("nature");
  const item = resolve(itemByName, val("item"));
  if (item === null) { err.hidden = false; err.textContent = `持ち物 ${val("item")} がわかりません`; return; }
  if (item !== (v.item ? v.item.id : "")) patch.item = item;
  const moves = [];
  for (const el of root.querySelectorAll('[data-f="move"]')) {
    const id = resolve(moveByName, el.value);
    if (id === null) { err.hidden = false; err.textContent = `技 ${el.value} がわかりません`; return; }
    if (id) moves.push(id);
  }
  if (moves.join("|") !== v.moves.map((m) => m.id).join("|")) patch.moves = moves;
  const sp = {}; let changed = false;
  root.querySelectorAll('[data-f="sp"]').forEach((el) => { const x = Number(el.value || 0); sp[META.stats[Number(el.dataset.k)]] = x; if (x !== v.sp[Number(el.dataset.k)]) changed = true; });
  if (changed) patch.sp = sp;
  if (!Object.keys(patch).length) { editing.delete(index); renderOpp(); return; }
  try { STATE = Object.assign(STATE, await api("/api/override", { index, patch })); $("result").hidden = true; editing.delete(index); renderOpp(); }
  catch (e) { err.hidden = false; err.textContent = e.message; }
}

document.addEventListener("click", async (e) => {
  const t = e.target.closest("button"); if (!t) return;
  if (t.dataset.edit != null) { const i = Number(t.dataset.edit); editing.has(i) ? editing.delete(i) : editing.add(i); renderOpp(); }
  else if (t.dataset.save != null) save(Number(t.dataset.save));
  else if (t.classList.contains("rk-alt")) {
    try { STATE = Object.assign(STATE, await api("/api/choose", { index: Number(t.dataset.i), alternative: Number(t.dataset.a) })); $("result").hidden = true; renderOpp(); }
    catch (err) { $("oppMsg").innerHTML = `<li>${esc(err.message)}</li>`; }
  }
});
document.addEventListener("input", (e) => {
  const el = e.target;
  if (el.dataset && el.dataset.f === "sp") {
    const root = el.closest(".rk-ed"); const tot = Array.from(root.querySelectorAll('[data-f="sp"]')).reduce((a, x) => a + Number(x.value || 0), 0);
    const box = root.querySelector("[data-sptot]"); box.textContent = `合計 ${tot} / ${META.spLimit}`; box.classList.toggle("over", tot > META.spLimit);
  }
});

async function solve() {
  $("readError").hidden = true; $("result").hidden = true;
  try { STATE.job = await api("/api/solve", {}); } catch (e) { $("readError").hidden = false; $("readError").textContent = e.message; return; }
  watch();
}
function watch() {
  $("progress").hidden = false; refreshSolve();
  if (poll) clearInterval(poll);
  const tick = async () => {
    const job = await api("/api/job"); STATE.job = job;
    if (job.state === "running") {
      const budget = job.seconds || 90;
      $("pbar").style.width = Math.min(100, 100 * job.elapsed / budget) + "%";
      $("ptext").textContent = `読んでいます… ${Math.round(job.elapsed)} / ${Math.round(budget)} 秒`;
    } else {
      clearInterval(poll); poll = null; $("progress").hidden = true; refreshSolve();
      if (job.state === "done") renderResult(job.result);
      if (job.state === "error") { $("readError").hidden = false; $("readError").textContent = job.error; }
    }
  };
  poll = setInterval(tick, 700); tick();
}

$("loadMine").onclick = loadMine;
$("mineRedo").onclick = () => { $("mineEdit").hidden = false; $("mineBrief").hidden = true; $("paste").focus(); };
$("fillOpp").onclick = fillOpp;
$("solve").onclick = solve;

// ------------------------------------------------------------------ start
(async function init() {
  META = await api("/api/meta");
  $("speciesList").innerHTML = META.species.map((s) => `<option value="${esc(s.name)}">${esc(s.en)}${s.inField ? "" : "（大会データに無い）"}</option>`).join("");
  $("itemList").innerHTML = META.items.map((i) => `<option value="${esc(i.name)}">`).join("");
  $("moveList").innerHTML = META.moves.map((m) => `<option value="${esc(m.name)}">`).join("");
  META.items.forEach((i) => { itemByName.set(i.name.toLowerCase(), i.id); itemByName.set(i.id, i.id); });
  META.moves.forEach((m) => { moveByName.set(m.name.toLowerCase(), m.id); moveByName.set(m.id, m.id); });
  $("speciesInputs").innerHTML = [1, 2, 3, 4, 5, 6].map((n) => `<input list="speciesList" aria-label="相手の ${n} 体目" placeholder="${n} 体目" autocomplete="off">`).join("");
  $("aboutEvent").textContent = META.event;
  $("aboutModel").textContent = (META.model ? `評価モデル: ${META.model}。` : "") + (META.learned ? "学んでいない種族・技は、学習の局に出た id の一覧から判定します。" : "");
  $("readNote").textContent = `下の「選出を求める」を押すと、ここに出ます（${META.seconds ? Math.round(META.seconds) : 90} 秒ほどかかります）。型を直してから、もう一度求められます。`;
  const state = await api("/api/state");
  STATE = state;
  (QUERY.get("edit") || "").split(",").filter(Boolean).forEach((i) => editing.add(Number(i)));
  if (state.mine) renderMine(null);
  renderOpp();
  if (state.job && state.job.state === "running") watch();
  else if (state.job && state.job.state === "done") renderResult(state.job.result);
})().catch((e) => { document.body.insertAdjacentHTML("beforeend", `<p class="rk-error">${esc(e.message)}</p>`); });
})();
