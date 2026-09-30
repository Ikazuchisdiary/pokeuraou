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
const PV_EVERY_MS = 400;     // the open PV tree is redrawn at most this often
const CHART_DECISIONS = 8;   // the value chart shows the last this many decisions
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
  analysis: false, catalogue: null, status: null, running: false, analysisState: null,
};
const $ = (id) => document.getElementById(id);
// A species name with its forme ("ウインディ (Hisui)", "Floette (Eternal)": the localiser writes
// base (forme)): the forme in small type, the same as the game page (tools/game_page.py name_html).
const nameHtml = (n) => {
  const [base, ...rest] = String(n == null ? "" : n).split(" (");
  return rest.length ? `${esc(base)}<small class="forme">(${esc(rest.join(" (").replace(/\)$/, ""))})</small>` : esc(base);
};
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => (p >= 0.1 || p === 0 ? (100 * p).toFixed(0) : (100 * p).toFixed(1)) + "%";
const v3 = (v) => v.toFixed(3);
const dlt = (d) => (Math.abs(d) < 0.0005 ? '<span class="n zero">±0.000</span>'
  : `<span class="n ${d > 0 ? "up" : "down"}">${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(3)}</span>`);
const aiSide = () => 1 - S.personSide;

// ------------------------------------------------------------------ the two pages (IKA-349)
// The game page and the analysis page are two servers (tools/play.py starts both). Each links to
// the other: the address comes from the page's meta (the server writes it), else the launcher's
// ports on this host.
const metaUrl = (name, port) => {
  const m = document.querySelector(`meta[name="${name}"]`);
  return (m && m.getAttribute("content")) || `${location.protocol}//${location.hostname}:${port}/`;
};
const ANALYSIS_URL = metaUrl("analysis-url", 8337), GAME_URL = metaUrl("game-url", 8332);
const withQuery = (url, q) => url + (url.includes("?") ? "&" : "?") + new URLSearchParams(q).toString();
// How the game page shows the AI's reading: show / until the person's move is sent / off.
const READ_MODES = ["show", "until", "off"];
let readMode = "until";
try { readMode = localStorage.getItem("pokeuraou-read") || "until"; } catch (_) { /* storage may be blocked */ }
if (!READ_MODES.includes(readMode)) readMode = "until";

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
function art(name, size, title, letters) {
  const inf = infoOf(name);
  const t1 = TYPE_COLORS[inf.types[0]] || "", t2 = TYPE_COLORS[inf.types[1]] || "";
  const style = `${t1 ? `--t1:${t1};` : ""}${t2 ? `--t2:${t2};` : ""}`;
  const url = spriteUrl(inf.id);
  const fb = `<span class="fb-full">${esc(name)}</span><span class="fb-short">${esc(String(name).slice(0, letters || 1))}</span>`;
  const tip = esc(title || name);
  if (!url || badSprite.has(url)) return `<span class="art s-${size} fb" style="${style}" title="${tip}">${fb}</span>`;
  return `<span class="art s-${size}" style="${style}" title="${tip}">${fb}<img src="${esc(url)}" alt="${esc(name)}" decoding="async"></span>`;
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

// An action's label comes as its parts, each [slot, text] with the active slot whose action it
// is (IKA-345: the data says so; nothing is split or guessed). `who` names the Pokemon in each
// active slot where the action is taken: the board's at the root, a node's field below it.
function boardWho(side) {
  const sd = S.board && S.board.sides[side];
  return sd ? sd.active.map((m) => (m ? m.species : null)) : [];
}
function fieldWho(row) { return (row || []).map((m) => (m ? m.name : null)); }
// A part is [slot, text, verb, target side (-1: none), target name, target sprite, mega]
// (IKA-345): the move's name, then its target as the target's icon -- ringed, and its arrow
// coloured, in the target's side's colour -- then the Mega Evolution mark. A part that is only
// [slot, text] (an older record) is its text.
const MEGA = '<span class="mega-mark" role="img" aria-label="メガシンカ" title="メガシンカ"></span>';
function sideName(side) { return (S.names && S.names[side]) || (side === aiSide() ? "AI" : "あなた"); }
function partBody(p, chunked) {
  const [, t, verb, tside, tname, tsprite, , swap] = p;
  if (verb === undefined) {
    return chunked
      ? `<span class="chunks">${String(t).split(/(?= → | \+ )/).map((c) => `<span class="ck">${esc(c.trim())}</span>`).join("")}</span>`
      : `<span>${esc(t)}</span>`;
  }
  if (swap && tname) {
    // A switch: the swap mark and the Pokemon coming in.
    learn({ species: tname, id: tsprite });
    const tip = `交代: ${tname}`;
    return `<span class="pbody"><span class="swap ${tside === aiSide() ? "a" : "y"}" role="img" aria-label="${esc(tip)}" title="${esc(tip)}"><span class="swapmark" aria-hidden="true">⇄</span>${art(tname, "xs", tip, 2)}</span></span>`;
  }
  let h = `<span class="verb">${esc(verb)}</span>`;
  if (tside >= 0 && tname) {
    learn({ species: tname, id: tsprite });
    const tip = `${tname}（${sideName(tside)}の側）`;
    // Two letters on a name card: one would not tell ガオガエン from ガブリアス.
    h += `<span class="tgt ${tside === aiSide() ? "a" : "y"}" aria-label="→ ${esc(tip)}"><span class="arrow" aria-hidden="true">→</span>${art(tname, "xs", tip, 2)}</span>`;
  }
  return `<span class="pbody">${h}</span>`;
}
function actHtml(parts, who, size, chunked) {
  return (parts || []).map((p) => {
    const slot = p[0], t = p[1];
    const name = slot >= 0 && who ? who[slot] : null;
    // Mega Evolution is the user's: its mark sits on the user's icon (or after the move
    // where there is no icon). A slot that cannot act (its Pokemon fainted) passes, quietly.
    const mega = p.length >= 7 && p[6];
    const user = name ? `<span class="user">${art(name, size || "xs")}${mega ? MEGA : ""}</span>` : "";
    const body = partBody(p, chunked) + (mega && !name ? MEGA : "");
    return `<span class="part${t === "行動なし" || t === "pass" ? " idle" : ""}">${user}${body}</span>`;
  }).join("");
}
const benchText = (bench) => (bench && bench.length ? bench.join(" / ") : "–");

// ------------------------------------------------------------------ events
// `short` (optional) is what a phone's header shows; the whole text stays in the title.
function setStatus(text, cls, short) {
  const el = $("status");
  el.innerHTML = short ? `<span class="long">${esc(text)}</span><span class="short">${esc(short)}</span>` : esc(text);
  el.title = text; el.className = "status " + (cls || "");
}
// The AI reads the selection while the person picks (IKA-392): the status counts the AI's
// seconds down until the AI is done (`selected`, or the first board).
// The seconds go into the words only (the class, title of the dot and the short text stay).
function statusText(text) {
  const el = $("status"), long = el.querySelector(".long");
  if (long) long.textContent = text; else el.textContent = text;
  el.title = text;
}
function selRest() {
  const left = Math.ceil((S.selEnd - Date.now()) / 1000);
  return !S.selSeconds ? "" : left > 0 ? `残り ${left} 秒` : "まもなく終わります";
}
// The same seconds in the input box (a phone's status pill has no room for them).
function selBox() {
  const rest = selRest(), a = $("selAi"), w = $("selRest");
  if (a) a.textContent = `AI も選出を読んでいます${rest ? "・" + rest : ""}`;
  if (w) w.textContent = rest ? `（${rest}）` : "";
}
function selWords() {
  const rest = selRest();
  if (S.sent) return `AI の選出を待っています${rest ? `（${rest}）` : ""}`;
  return `${S.select ? S.select.size : 4} 体を選んでください（AI も選出を読んでいます${rest ? "・" + rest : ""}）`;
}
function stopSelecting() { clearInterval(S.selTimer); S.selTimer = null; S.selecting = false; }
function log(turn, html, cls) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${turn === "" ? "" : turn != null ? "T" + turn : "·"}</span><div class="${cls || ""}">${html}</div>`;
  $("log").prepend(li);
}
function toast(text) {
  const t = document.createElement("div"); t.className = "toast"; t.textContent = text;
  document.body.appendChild(t); setTimeout(() => t.remove(), 4000);
}

function onEvent(e) {
  switch (e.type) {
    case "session":
      S.gamesLeft = e.gamesLeft; S.games = e.games; S.gameNo = e.game; S.gameIndex = e.gameIndex;
      if (S.ended) {
        // The next game starts at once (IKA-356): the last game's card stays over its board
        // until the new game's first board, so that its result can be read and analysed (its
        // button quieter: the selection is what to do now); the AI's reading of the last game
        // is cleared, and the log marks where the last game ends.
        const more = $("result").querySelector(".more");
        if (more) more.textContent = `${e.game + 1} 局目（全 ${e.games} 局）が始まりました。選出を選んでください`;
        const go = $("result").querySelector(".btnlink");
        if (go) { go.classList.add("quiet"); go.textContent = `${e.game} 局目を分析する`; }
        $("result").classList.add("between");
        if (S.lastEnd) log("", `${e.game} 局目（${S.lastEnd.head}・${S.lastEnd.turns} ターン）`, "gamehead");
        resetReading();
      } else {
        $("result").hidden = true; $("result").innerHTML = "";
      }
      S.ended = false; document.body.classList.remove("ended"); updateLinks();
      break;
    case "sheets":
      S.sheets = e; S.names = e.names; S.personSide = e.personSide; S.agentSide = e.agentSide;
      e.teams.forEach((team) => team.forEach(learn));
      renderSheets();
      if (!e.analysis) log(null, `対局開始${S.games > 1 ? `（${S.gameNo + 1} / ${S.games} 局目）` : ""}。AI は <b>${esc(e.agent)}</b>、1 手 ${e.seconds} 秒（${e.clock === "wall" ? "実時間" : "ノード時間"}・${e.cores} コア）`, S.gameNo > 0 ? "newgame" : "");
      break;
    case "catalogue":
      enterAnalysis(); S.catalogue = e; learnReads(e); renderPicker(true); openFromUrl(); renderTimeline();
      break;
    case "analysis":
      enterAnalysis(); onAnalysis(e);
      break;
    case "selecting": {
      // IKA-392: the AI reads the selection for e.seconds while the person picks (`select`
      // follows at once); what the status says depends on whether the person has picked.
      stopSelecting();
      S.selecting = true;
      S.selEnd = Date.now() + (e.seconds || 0) * 1000; S.selSeconds = e.seconds || 0;
      // Only the words change after the first draw: the status dot's pulse is not restarted.
      S.selTimer = setInterval(() => { statusText(selWords()); selBox(); }, 1000);
      break;
    }
    case "selected":
      // The AI has its four (not shown: it is public when both have chosen, at the first board).
      if (S.selecting) {
        stopSelecting();
        if (S.select) setStatus(`${S.select.size} 体を選んでください`, "turn");
        else setStatus("AI の選出が決まりました。まもなく始まります", "think");
      }
      break;
    case "select":
      S.select = e; S.picked = []; S.sent = false; renderInput();
      if (S.selecting) setStatus(selWords(), "turn", "選出の番");
      else setStatus(`${e.size} 体を選んでください`, "turn");
      if ($("result").classList.contains("between")) {
        const more = $("result").querySelector(".more");
        if (more && S.games) more.textContent = `${S.gameNo + 1} 局目（全 ${S.games} 局）が始まりました。${e.size} 体を選んでください`;
      }
      break;
    case "board":
      stopSelecting();
      e.sides.forEach((sd) => [...sd.active, ...sd.bench].forEach(learn));
      S.board = e; renderBoard();
      if (S.select) { S.select = null; renderInput(); }
      break;
    case "think":
      S.sent = false; S.think = e; S.decision = e.decision; S.answered = false; S.prompt = null;
      // IKA-344: with ponder the person is asked while the AI reads, until they choose.
      S.pondering = !!e.ponder;
      S.history.push({ decision: e.decision, turn: e.turn, pts: [], done: false, width: S.analysisState && S.analysisState.width });
      if (!S.analysis) { renderInput(); setStatus(`AI が考えています（ターン ${e.turn}）`, "think"); applyHide(); }
      renderClock(0, false);
      break;
    case "answer":
      S.pondering = false;
      if (!S.analysis) {
        if (e.memoryStop) {
          // IKA-355: the memory watch stopped this move's reading; it played what it had.
          setStatus(`AI は手を決めました（${e.seconds.toFixed(1)} 秒・メモリが少ないので読みを止めました）`, "warn",
            `メモリが少ないので読みを止めました`);
          const budget = S.sheets && S.sheets.seconds ? `持ち時間 ${S.sheets.seconds} 秒のうち ` : "";
          log(e.turn, `メモリが少ないので、この手の読みを止めました（${budget}${e.seconds.toFixed(1)} 秒）` +
            (typeof e.memoryStop === "string" ? `: ${esc(e.memoryStop)}` : ""), "err");
        } else {
          setStatus(`AI は手を決めました（${e.seconds.toFixed(1)} 秒）`, "turn");
        }
      }
      renderClock(e.seconds * 1000, true);
      break;
    case "memory":
      // IKA-355: the memory watch's state. Low from other work: the AI reads on.
      if (e.state === "stop") {
        log(null, `メモリが少ないので読みを止めます: ${esc(e.why)}`, "err");
        toast(`メモリが少ないので読みを止めます: ${e.why}`);
      } else if (e.state === "low") {
        log(null, `ほかの処理でメモリの空きが少なくなっています。AI は読みを続けます（空きが ${e.hardFreeGb} GB を切ったら止めます）: ${esc(e.why)}`, "warn");
      } else {
        log(null, "メモリの空きが戻りました", "dim");
      }
      break;
    case "prompt":
      S.prompt = e; S.chosen = []; S.mega = -1; S.answered = false; renderInput(); applyHide();
      setStatus(e.kind === "move" ? (S.pondering ? "あなたの番（AI は読み続けています）" : "あなたの番") : e.heading, "turn");
      break;
    case "turn":
      S.sent = false;
      if (S.prompt) { S.prompt = null; S.answered = true; applyHide(); }
      renderInput();
      log(e.turn, `<a class="alink" data-turn="${e.turn}"${S.gameIndex != null ? ` data-game="${S.gameIndex}"` : ""} target="pokeuraou-analysis">分析</a><span class="ai-c">AI</span> <span class="logacts">${actHtml(e.agentParts || [[-1, e.agent]], boardWho(aiSide()))}</span><br><span class="you-c">あなた</span> <span class="logacts">${actHtml(e.personParts || [[-1, e.person]], boardWho(S.personSide))}</span>` +
        (e.offMenu ? ` <span class="badge" title="あなたの手は AI の候補集合の外でした">候補集合の外</span>` : "") +
        ((e.changes || []).length ? `<br><span class="dim">${e.changes.map((c) =>
          `${nameHtml(c.species)}${c.entered ? " 登場" : ""}${c.from !== c.to ? ` ${c.from}→${c.to}%` : ""}${c.fainted ? " ひんし" : ""}${c.status ? " " + esc(c.status) : ""}`).join("・")}</span>` : ""));
      updateLinks();
      break;
    case "end": {
      stopSelecting();
      const mine = e.outcome === null ? null : (e.outcome > 0.5) === (e.personSide === 0);
      const head = mine === null ? "打ち切り" : mine ? "あなたの勝ち" : "AI の勝ち";
      setStatus(`終局: ${head}`, "end");
      const r = $("result"); r.hidden = false; r.classList.remove("between");
      // Why the game ended, in words (selfplay/humanplay's end reasons).
      const why = e.reason === "turn-cap" ? "ターンの上限で打ち切り"
        : e.reason === "unresolved" ? "解決できない局面で打ち切り"
        : e.reason === "draw" ? "両者の最後の体が同時にひんし（引き分け）"
        : e.reason === "wipeout" || mine !== null ? (mine ? "AI の 4 体がひんし" : "あなたの 4 体がひんし")
        : e.reason ? esc(e.reason) : "";
      const more = S.gamesLeft > 0 ? "次の局を待っています" : "続けて打つには tools/play.py を起動し直してください";
      const count = S.games > 1 ? `<small class="gameno">${S.gameNo + 1} / ${S.games} 局目</small>` : "";
      r.innerHTML = `<div class="endcard">${count}<b class="${mine === null ? "" : mine ? "you-c" : "ai-c"}">${head}</b><span>${e.turns} ターン${why ? "・" + why : ""}</span>
        <a class="primary btnlink" href="${esc(withQuery(ANALYSIS_URL, gameQuery({ open: "last" })))}" target="pokeuraou-analysis">この対局を分析する</a>
        <small class="more">${more}</small></div>`;
      log(e.turns, `<b>終局: ${head}</b>`);
      // This game's turns can be analysed from now on, also while the next game is played.
      document.querySelectorAll("#log .alink:not([data-ended])").forEach((a) => { a.dataset.ended = "1"; });
      S.lastEnd = { head, turns: e.turns };
      S.ended = true; document.body.classList.add("ended");
      S.prompt = null; S.select = null; S.answered = true; renderInput(); applyHide();
      break;
    }
    case "error": log(null, esc(e.message), "err"); toast(e.message); break;
  }
}

function onStep(s) {
  document.body.classList.remove("noread");
  if (s.decision !== S.decision || !S.history.length) {
    S.decision = s.decision;
    S.history.push({ decision: s.decision, turn: s.turn, pts: [], done: false, width: S.analysisState && S.analysisState.width });
  }
  const h = S.history[S.history.length - 1];
  h.pts.push([Math.min(1, share(s)), s.value, s.ms]);
  h.done = s.kind === "done";
  S.last = s;
  renderClock(s.ms, s.kind === "done", s);
  renderBalance(s); renderStrip(s); drawChart();
  renderMix("ours", s.ours, s.ourP, s.ourLoss, aiSide(), s.kind === "done", s.oursParts);
  renderMix("theirs", s.theirs, s.theirP, s.theirLoss, S.personSide, s.kind === "done", s.theirsParts);
  if (S.analysis) renderPlayed(s);
  renderClasses(s);
  if ($("countBox").open) renderCounters(s);
  schedulePv(s.kind === "done");
}

// ------------------------------------------------------------------ the court
function hpCls(p) { return p > 50 ? "" : p > 20 ? "mid" : "low"; }
// Volatile conditions (Showdown's ids) in Japanese; an id not listed is shown as it is.
const VOLATILE_JA = {
  stall: "まもる連続", protect: "まもる", substitute: "みがわり", confusion: "こんらん", taunt: "ちょうはつ",
  encore: "アンコール", leechseed: "やどりぎのタネ", flinch: "ひるみ", helpinghand: "てだすけ", followme: "このゆびとまれ",
  ragepowder: "いかりのこな", focusenergy: "きあいだめ", charge: "じゅうでん", disable: "かなしばり", torment: "いちゃもん",
  yawn: "あくび", perish1: "ほろびのうた 1", perish2: "ほろびのうた 2", perish3: "ほろびのうた 3", partiallytrapped: "バインド",
  lockedmove: "あばれる", mustrecharge: "反動で動けない", twoturnmove: "ため", roost: "はねやすめ", magnetrise: "でんじふゆう",
  attract: "メロメロ", throatchop: "じごくづき", glaiverush: "きょじゅうざん", saltcure: "しおづけ", syrupbomb: "みずあめ",
  healblock: "かいふくふうじ", imprison: "ふういん", gastroacid: "いえき", smackdown: "うちおとす", destinybond: "みちづれ",
  endure: "こらえる", wideguard: "ワイドガード", quickguard: "ファストガード", spotlight: "スポットライト",
  dynamax: "ダイマックス", laserfocus: "とぎすます", curse: "のろい", nightmare: "あくむ", powder: "ふんじん",
};
const volJa = (v) => VOLATILE_JA[v] || v;
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
  if ((m.volatiles || []).length) more.push(`<div>${m.volatiles.map((v) => esc(volJa(v))).join("・")}</div>`);
  if (mine && m.moves) more.push(`<table>${m.moves.map(([n, pp, mx]) => `<tr><td>${esc(n)}</td><td class="n">${pp}/${mx}</td></tr>`).join("")}</table>`);
  const drop = prev - hpNow;
  return `<article class="mon${m.fainted ? " fainted" : ""}${S.openMon.has(key) ? " open" : ""}" data-key="${key}" title="押すと詳細">
    ${art(m.species, "lg")}
    <div style="min-width:0">
      <div class="mon-name"><span>${nameHtml(m.species)}</span> ${status}</div>
      <div class="hp"><div class="hp-track"><i class="${hpCls(prev)}" style="width:${prev}%" data-to="${hpNow}"></i></div>
        <b class="hp-num n">${hpNow}<small>%</small></b></div>
      <div class="mon-sub">${exact}${boosts}${(m.volatiles || []).length ? `<span>${m.volatiles.map((v) => esc(volJa(v))).join("・")}</span>` : ""}</div>
    </div>
    ${drop > 0 ? `<span class="hp-delta n">−${drop}%</span>` : ""}
    <div class="mon-more">${more.join("") || "詳細なし"}</div>
  </article>`;
}
function benchHtml(side, mine) {
  const minis = side.bench.map((m) => `<span class="mini${m.fainted ? " fainted" : ""}">${art(m.species, "sm")}<span>${nameHtml(m.species)}</span>
    <span class="n dim">${m.fainted ? "ひんし" : mine && m.hp ? m.hp[0] + "/" + m.hp[1] : m.percent + "%"}</span></span>`);
  for (let i = 0; i < side.hidden; i++) minis.push(`<span class="facedown" title="まだ見ていない裏">?</span>`);
  if (!minis.length) return "";
  return `<div class="bench"><span class="lbl">裏</span>${minis.join("")}</div>`;
}
function renderBoard() {
  const b = S.board;
  if (!b) return;
  $("result").hidden = !b.ended || $("result").innerHTML === "";
  if ($("result").hidden) $("result").classList.remove("between");
  const top = aiSide(), bottom = S.personSide;
  const draw = (i, where) => {
    const sd = b.sides[i], mine = i === b.viewer;
    const head = `<div class="side-head"><span class="who ${i === aiSide() ? "ai" : "you"}">${esc(sd.name)}</span>
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
  if (S.analysis) {
    $("clockbar").classList.toggle("done", !!done);
    $("clocktext").textContent = `経過 ${fmtTime(ms / 1000)}${s ? `・深化のステップ ${s.step.toLocaleString("ja-JP")}` : ""}${done ? "・答え" : ""}`;
    return;
  }
  const budget = S.think && S.think.seconds ? S.think.seconds * 1000 : 0;
  const bar = $("clockbar");
  bar.classList.toggle("done", !!done);
  const count = S.think && S.think.clock === "count";
  // The count clock is bounded by cells, not seconds: its band is the budget spent (IKA-345),
  // full and in the warning colour past 100%, the seconds only a footnote.
  const f = count ? (s ? share(s) : done ? 1 : 0) : budget ? ms / budget : 0;
  bar.classList.toggle("count", !!count);
  bar.classList.toggle("over", !!count && f > 1.0005);
  bar.firstElementChild.style.width = Math.min(100, 100 * f) + "%";
  if (count) {
    $("clocktext").innerHTML = `計算予算 ${Math.round(100 * Math.min(f, 9.99))}%<span class="sub">${(ms / 1000).toFixed(1)} 秒${done ? "・答え" : ""}</span>`;
  } else {
    $("clocktext").textContent = !budget ? "–" : `${(ms / 1000).toFixed(1)} / ${(budget / 1000).toFixed(1)} 秒${done ? "・答え" : ""}`;
  }
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
    ["読めなかった", s.refused],
    [S.analysis ? "幅（検討する側×相手）" : "幅（AI×あなた）", `${s.ours.length}×${s.theirs.length}`], ["サポートの大きさ", support],
    ["深く読んだ重み", pct(s.read)], ["裏の決定化", s.classes.length || "なし"],
    ["計算予算", S.analysis ? "なし（止めるまで）" : `${Math.round(s.spent)} / ${s.budget}`],
    ["双対ギャップ", s.gap.toExponential(1)],
  ];
  $("counters").innerHTML = items.map(([k, v]) => `<div><b class="n">${v}</b><small>${k}</small></div>`).join("");
  const t = S.think;
  if (t && S.analysis) {
    const p = t.plan || {};
    $("plan").textContent = `ターン ${t.turn}: 計算予算なしで止めるまで読む。幅 ${p.width}・最善応答オラクル ${p.oracle}・深化の深さの上限 ${p.guard} 段、` +
      `候補集合を作るのに ${Math.round(t.menuMs || 0)} ms。` + (t.exact ? "相手の裏は尽きている。" : `相手の裏の決定化 ${t.classCount} 通り。`) +
      "「深く読んだ重み」は均衡の組の確率のうち、葉より深く読んだ分（収束の目安）。";
  } else if (t) {
    const p = t.plan || {};
    $("plan").textContent = `ターン ${t.turn}: 計算予算 ${t.seconds} 秒、候補集合の幅 ${p.width}、深さ 1 の予測 ${Math.round(p.predictedMs || 0)} ms、` +
      `深化に ${Math.round(p.deepenMs || 0)} ms、候補集合づくりに ${Math.round(t.menuMs || 0)} ms。` +
      (t.exact ? "あなたの裏は尽きている。" : `あなたの裏の決定化 ${t.classCount} 通り。`) +
      "「深く読んだ重み」は均衡の組の確率のうち、葉より深く読んだ分（収束の目安）。";
  }
}

// Bars keep their places for REORDER_MS so a converging mixture reads as growing and shrinking
// bars, not as rows jumping about.
function renderMix(key, labels, p, loss, side, final, parts) {
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
      r.innerHTML = `<div class="act" title="${esc(labels[i])}">${actHtml(parts ? parts[i] : [[-1, labels[i]]], boardWho(side))}</div><div class="track"><i></i></div><b class="pct n"></b><span class="loss n"></span>`;
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
  if (s.classes.length === 1 && !s.classes[0].bench.filter((n) => n && n !== "-" && n !== "–").length) {
    $("beliefSum").textContent = "相手の裏は残っていない";
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
// `who` is {a, y}: the Pokemon in each active slot of the AI's side and the person's where the
// pair is played -- the board's at the root, the node's field below it (IKA-345).
function pairHtml(pair, key, parentValue, s, who) {
  // The completion's badge only where there is a choice of completions and it names someone.
  const bench = pair.klass >= 0 && s.classes.length > 1 && s.classes[pair.klass]
    ? s.classes[pair.klass].bench.filter((n) => n && n !== "-" && n !== "–") : [];
  const cls = bench.length ? ` <span class="badge cls">裏 ${esc(benchText(bench))}</span>` : "";
  const ours = actHtml(pair.oursParts, who.a, "xs", true);
  const theirs = actHtml(pair.theirsParts, who.y, "xs", true);
  const head = `<span class="pp n">${pct(pair.p)}</span><span class="pvmain"><span class="pvacts"><span class="act a">${ours}</span><span class="x">×</span><span class="act y">${theirs}</span></span>
    <span class="pvmeta"><span class="n">${v3(pair.value)}</span> ${dlt(pair.value - parentValue)} <span class="badge r-${pair.read}">${READ_JA[pair.read] || pair.read}</span>${cls}</span></span>`;
  if (!pair.branches.length) return `<div class="leaf">${head}</div>`;
  const inner = pair.branches.map((b, i) => branchHtml(b, `${key}.${i}`, pair.value, s, pair.branches.length)).join("");
  return `<details data-key="${key}"${S.open.has(key) ? " open" : ""}><summary>${head}</summary>${inner}</details>`;
}
// A chance branch: its weight, its draws (causes: purple where the draw went the rare way,
// quiet where it went the ordinary way its sibling did not), then what it did on each side.
function causeHtml(b, siblings, ended) {
  const end = ended ? '<span class="cause end">決着</span>' : "";
  if (siblings < 2) return `<span class="cause-row"><span class="cause plain">分岐なし</span>${end}</span>`;
  const hot = b.causes.filter((c) => !c.plain), calm = b.causes.filter((c) => c.plain);
  const chips = hot.map((c) => `<span class="cause" title="${esc(c.title)}"><b>${esc(c.head)}</b>${esc(c.body)}</span>`);
  if (calm.length) {
    chips.push(`<span class="cause plain" title="${esc(calm.map((c) => c.title || `${c.head} ${c.body}`).join("\n"))}">${esc(calm.map((c) => (c.head === "命中" ? "命中" : c.head)).filter((h, i, a) => a.indexOf(h) === i).join("・"))}</span>`);
  }
  if (end) chips.push(end);
  return chips.length ? `<span class="cause-row">${chips.join("")}</span>` : "";
}
function changeHtml(c) {
  learn({ species: c.name, id: c.sprite });
  const bits = [];
  if (c.entered) bits.push("<span>登場</span>");
  const d = c.to - c.from;
  if (d && !c.fainted) bits.push(`<span class="n ${d < 0 ? "down" : "up"}">${d < 0 ? "−" : "+"}${Math.abs(d)}%</span>`);
  if (c.status) bits.push(`<span class="st st-${esc(c.status)}">${esc(c.status)}</span>`);
  if (c.fainted) bits.push('<span class="ko">ひんし</span>');
  const tip = `${c.name} ${c.from}% → ${c.fainted ? "ひんし" : c.to + "%"}`;
  return `<span class="hit" title="${esc(tip)}">${art(c.name, "xs", tip)}${bits.join("")}</span>`;
}
function resultHtml(b) {
  if (!b.changes[0].length && !b.changes[1].length) return '<span class="res"><span class="rl0">変化なし</span></span>';
  const rows = [["", b.changes[0]], [" y", b.changes[1]]];
  return `<span class="res">${rows.map(([cls, list]) => list.length
    ? `<span class="rl${cls}">${list.map(changeHtml).join("")}</span>`
    : `<span class="rl${cls} none">変化なし</span>`).join("")}</span>`;
}
function fieldHtml(n) {
  const row = (list, cls, name) => `<span class="who ${cls}">${esc(name)}</span><span class="mons">${(list || []).map((m) => {
    if (!m) return '<span class="fm empty"></span>';
    learn({ species: m.name, id: m.sprite });
    const tip = `${m.name} ${m.fainted ? "ひんし" : m.percent + "%"}${m.new ? "・この分岐で登場" : ""}`;
    return `<span class="fm${m.new ? " new" : ""}${m.fainted ? " ko" : ""}" title="${esc(tip)}">${art(m.name, "xs", tip)}<span class="hpb"><i class="${hpCls(m.percent) === "low" ? "b" : hpCls(m.percent) === "mid" ? "w" : ""}" style="width:${m.fainted ? 0 : m.percent}%"></i></span></span>`;
  }).join("")}</span>`;
  const names = S.analysis ? ["検討する側", "相手"] : ["AI", "あなた"];
  // A record kept before IKA-345 has no field: the value alone.
  const rows = n.field[0].length || n.field[1].length ? row(n.field[0], "a", names[0]) + row(n.field[1], "y", names[1]) : "";
  return `<div class="nodefield">${rows}` +
    `<span class="val">次のターン 値 <span class="n">${v3(n.value)}</span>${n.more ? "・この先も読んだが省略" : ""}</span></div>`;
}
function topsHtml(n, who) {
  const line = (list, cls, w) => list.length
    ? `<span class="act ${cls}">${list.map(([, p, parts]) => `<span class="topline"><span class="tparts">${actHtml(parts, w, "xs", true)}</span><span class="n">${pct(p)}</span></span>`).join("")}</span>` : "";
  return `<div class="nodetops">${line(n.ours, "a", who.a)}${line(n.theirs, "y", who.y)}</div>`;
}
function branchHtml(b, key, parentValue, s, siblings) {
  const chance = siblings > 1;
  // A record kept before IKA-345 has the branch's text only (no draws, no per-side changes).
  const body = b.textOnly
    ? `${b.ended ? '<span class="cause-row"><span class="cause end">決着</span></span>' : ""}<span class="res"><span class="old" title="IKA-345 より前の記録: 分岐の文だけが残っている">${esc(b.what)}</span></span>`
    : `${causeHtml(b, siblings, b.ended)}
    ${b.ended ? "" : resultHtml(b)}`;
  const head = `<span class="pp${chance ? " c" : ""} n">${pct(b.weight)}</span><span class="pvmain">
    ${body}
    <span class="pvmeta"><span class="n">${v3(b.value)}</span> ${dlt(b.value - parentValue)}</span></span>`;
  if (!b.node) return `<div class="leaf">${head}</div>`;
  const n = b.node;
  const who = { a: fieldWho(n.field[0]), y: fieldWho(n.field[1]) };
  const inner = fieldHtml({ ...n, more: b.more }) + topsHtml(n, who) + n.pairs.map((p, i) => pairHtml(p, `${key}.${i}`, n.value, s, who)).join("");
  return `<details data-key="${key}"${S.open.has(key) ? " open" : ""}><summary>${head}</summary>${inner}</details>`;
}
// A new game (IKA-356, --games 2 or more): the AI's reading, its value chart and the PV's open
// rows were the last game's.
function resetReading() {
  S.history = []; S.last = null; S.decision = -1; S.think = null;
  S.order = { ours: { keys: [], at: 0 }, theirs: { keys: [], at: 0 } };
  S.open = new Set(["p0"]);
  for (const id of ["ours", "theirs", "classes", "pv", "strip", "counters", "plan"]) $(id).innerHTML = "";
  $("balAi").textContent = "–"; $("balYou").textContent = "–"; $("balBar").style.width = "50%";
  $("clocktext").textContent = "–"; $("clockbar").querySelector("i").style.width = "0";
  $("pvSum").textContent = "確率の付いた木"; $("beliefSum").textContent = "–";
  document.body.classList.add("noread");
  drawChart();
}
function renderPv(s) {
  $("pv").innerHTML = s.pv.length
    ? (S.analysis
      ? `<p class="note">重い組（検討する側の手 × 相手の手）。読んだ組は偶然の分岐と次のターンへ開く。値は検討する側から見た値、括弧は親との差。</p>`
      : `<p class="note">重い組（AI の手 × あなたの手）。読んだ組は偶然の分岐と次のターンへ開く。値は AI から見た値、括弧は親との差。</p>`) +
      s.pv.map((p, i) => pairHtml(p, `p${i}`, s.value, s, { a: boardWho(aiSide()), y: boardWho(S.personSide) })).join("")
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
// A band's label under the value chart: its turn; on the analysis page a turn read again is "N 回目".
function bandLabel(H, i) {
  const d = H[i];
  if (!S.analysis) return `T${d.turn}`;
  const same = H.filter((x) => x.turn === d.turn);
  return same.length > 1 ? `T${d.turn} ${same.indexOf(d) + 1} 回目` : `T${d.turn}`;
}
function drawChart() {
  const c = $("chart"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w) return;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  let H = S.history.filter((d) => d.pts.length).slice(-CHART_DECISIONS);
  if (!H.length) return;
  if (S.analysis) {
    // No budget: a read's x is its time over the longest time it has reached.
    H = H.map((d) => {
      const end = Math.max(1, ...d.pts.map((p) => p[2] || 0));
      return { ...d, pts: d.pts.map((p) => [(p[2] || 0) / end, p[1], p[2]]) };
    });
  }
  const all = H.flatMap((d) => d.pts.map((p) => p[1]));
  let lo = Math.min(0.5, ...all), hi = Math.max(0.5, ...all);
  const span = Math.max(0.1, hi - lo); lo = Math.max(0, lo - span * 0.15); hi = Math.min(1, hi + span * 0.15);
  const L = 34, R = 8, T = 6, B = 18, n = H.length, bw = (w - L - R) / n;
  const X = (d, f) => L + bw * (d + 0.06 + 0.88 * f), Y = (v) => T + (h - T - B) * (1 - (v - lo) / (hi - lo));
  g.font = `11px ${col("--font-num") || "monospace"}`; g.textBaseline = "middle";
  H.forEach((d, i) => {
    if (i % 2 === 0) { g.fillStyle = col("--surface-2"); g.fillRect(L + bw * i, T, bw, h - T - B); }
    g.fillStyle = col("--faint"); g.textAlign = "center"; g.fillText(bandLabel(H, i), L + bw * (i + 0.5), h - 8);
  });
  g.textAlign = "right";
  const ticks = [lo, hi]; if (Math.abs(Y(0.5) - Y(lo)) > 14 && Math.abs(Y(0.5) - Y(hi)) > 14) ticks.push(0.5);
  ticks.forEach((v) => { g.fillStyle = col("--faint"); g.fillText(v.toFixed(2), L - 4, Y(v)); });
  g.strokeStyle = col("--line"); g.lineWidth = 1; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(L, Y(0.5)); g.lineTo(w - R, Y(0.5)); g.stroke(); g.setLineDash([]);
  const path = () => {
    g.beginPath();
    // On the analysis page each band is a reading of its own: its line does not join the next.
    H.forEach((d, i) => d.pts.forEach(([f, v], j) => ((i && !(S.analysis && j === 0)) || j ? g.lineTo(X(i, f), Y(v)) : g.moveTo(X(i, f), Y(v)))));
  };
  const lastD = H.length - 1, lastP = H[lastD].pts[H[lastD].pts.length - 1];
  const area = (tint, above) => {
    g.save(); g.beginPath();
    if (above) g.rect(0, 0, w, Y(0.5)); else g.rect(0, Y(0.5), w, h);
    g.clip(); path();
    // The fill closes under the last line only: on the analysis page that is the last reading.
    const from = S.analysis ? lastD : 0;
    g.lineTo(X(lastD, lastP[0]), Y(0.5)); g.lineTo(X(from, H[from].pts[0][0]), Y(0.5)); g.closePath();
    g.globalAlpha = 0.16; g.fillStyle = tint; g.fill(); g.restore();
  };
  area(col("--ai"), true); area(col("--you"), false);
  path(); g.strokeStyle = col("--ink"); g.globalAlpha = 0.85; g.lineWidth = 1.6; g.lineJoin = "round"; g.stroke(); g.globalAlpha = 1;
  g.fillStyle = lastP[1] >= 0.5 ? col("--ai") : col("--you");
  g.beginPath(); g.arc(X(lastD, lastP[0]), Y(lastP[1]), 4, 0, 7); g.fill();
  // A reading kept with its last value only (a page that came in late): a grey dot there.
  if (S.analysis) H.forEach((d, i) => {
    if (i === lastD || d.pts.length !== 1) return;
    g.fillStyle = col("--faint"); g.beginPath(); g.arc(X(i, d.pts[0][0]), Y(d.pts[0][1]), 4, 0, 7); g.fill();
  });
}
window.addEventListener("resize", drawChart);
window.addEventListener("resize", () => renderTimeline());

// ------------------------------------------------------------------ the person's input
function send(line) {
  LiveData.send(line);
  S.prompt = null; S.select = null; S.answered = true; S.sent = true;
  renderInput(); applyHide();
  if (S.selecting) setStatus(selWords(), "think", "AI を待つ");
  else if (S.pondering) setStatus("AI が読みを止めて手を引くのを待っています", "think");
}
function renderInput() {
  const box = $("input");
  $("inputBox").classList.toggle("active", !!(S.select || S.prompt));
  if (S.select && S.sheets) {
    const team = S.sheets.teams[S.personSide], foe = S.sheets.teams[aiSide()];
    const size = S.select.size;
    box.innerHTML = `<div class="ask">選出（${size} 体を順に押す。先の 2 体が先発）</div>${S.selecting ? '<div class="note" id="selAi"></div>' : ""}
      <div class="foeteam"><span class="note">相手</span>${foe.map((m) => art(m.species, "sm")).join("")}</div>
      <div class="pickgrid">${team.map((m, i) => {
        const at = S.picked.indexOf(i);
        return `<button type="button" class="pick" data-i="${i}" aria-pressed="${at >= 0}">${at >= 0 ? `<span class="ord">${at + 1}</span>` : ""}${at >= 0 && at < 2 ? '<span class="lead">先発</span>' : ""}
          ${art(m.species, "md")}<b>${nameHtml(m.species)}</b><small>${esc(m.item)}</small></button>`;
      }).join("")}</div>
      <div class="sendbar"><button type="button" class="ghost" id="clear">やり直す</button>
        <button type="button" class="primary" id="go" ${S.picked.length === size ? "" : "disabled"}>この選出で始める</button></div>`;
    selBox();
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
    box.innerHTML = S.ended ? `<span class="note">${S.gamesLeft > 0 ? "次の局を待っています" : "対局は終わりました"}</span>`
      : S.sent && S.selecting ? '<span class="note">選出を送りました。AI の選出を待っています<span id="selRest"></span></span>'
      : S.sent ? '<span class="note">送りました。ターンの結果を待っています</span>' : '<span class="note">AI の番を待っています</span>';
    selBox();
    return;
  }
  const nSlots = p.slots.length ? p.slots[0].length : 0;
  const act = p.actives || (S.board ? S.board.sides[S.personSide].active : []);
  act.forEach((m) => m && learn(m));
  // Each move once; Mega Evolution is the toggle on the slot's heading (choice.js turns the
  // pair back into one of the legal actions). One Pokemon a game Mega Evolves, so with the
  // toggle on for one slot the other's is off and cannot be pressed.
  const C = window.LiveChoice;
  const megas = C.megaSlots(p.slots);
  if (S.mega >= 0 && !megas.includes(S.mega)) S.mega = -1;
  let html = `<div class="ask">${esc(p.heading)}</div><div class="note">ターン ${p.turn}・合法手 ${p.choices.length}</div>`;
  for (let k = 0; k < nSlots; k++) {
    const opts = C.options(p.slots, k, S.chosen);
    if (opts.length === 1) S.chosen[k] = opts[0][0];
    if (S.chosen[k] !== undefined && !opts.some((o) => o[0] === S.chosen[k])) S.chosen.length = k;
    const who = nSlots === act.length && act[k] ? act[k].species : null;
    const toggle = megas.includes(k)
      ? `<button type="button" class="megatoggle" data-k="${k}" aria-pressed="${S.mega === k}" ${S.mega >= 0 && S.mega !== k ? "disabled" : ""}
          title="${S.mega >= 0 && S.mega !== k ? "メガシンカは 1 対戦に 1 回（もう 1 体に入れている）" : "この体をメガシンカさせる"}">${MEGA}<span>メガシンカ</span></button>`
      : "";
    html += `<div class="slot"><div class="slot-head">${who ? art(who, "sm") + `<b>${esc(who)}</b>` : `<b>${nSlots > 1 ? `${k + 1} 体目` : "選ぶ"}</b>`}${toggle}</div>
      <div class="opts">${opts.map(([c, l]) => `<button type="button" class="opt" data-k="${k}" data-c="${esc(c)}" aria-pressed="${S.chosen[k] === c}">${esc(l)}</button>`).join("")}</div></div>`;
    if (S.chosen[k] === undefined) { for (let j = k + 1; j < nSlots; j++) html += `<div class="slot"><div class="slot-head dim">${j + 1} 体目は上を選んでから</div></div>`; break; }
  }
  const ready = S.chosen.length === nSlots && S.chosen.every((c) => c !== undefined);
  const index = ready ? C.resolve(p.slots, S.chosen, S.mega) : -1;
  const labels = index >= 0 ? p.slots[index] : null;
  const whoIn = act.map((m) => (m ? m.species : null));
  const summary = labels
    ? `<span class="act">${actHtml(labels.map((x, k) => (x[2] ? x[2] : [k, x[1]])), whoIn)}</span>`
    : ready && S.mega >= 0 ? "この手ではメガシンカできない" : "手を選んでください";
  html += `<div class="sendbar"><span class="summary">${summary}</span>
    <button type="button" class="primary" id="go" ${labels ? "" : "disabled"}>この手で決定</button></div>`;
  box.innerHTML = html;
  box.querySelectorAll(".opt").forEach((b) => b.onclick = () => {
    const k = +b.dataset.k; S.chosen[k] = b.dataset.c; S.chosen.length = k + 1; renderInput();
  });
  box.querySelectorAll(".megatoggle").forEach((b) => b.onclick = () => {
    const k = +b.dataset.k; S.mega = S.mega === k ? -1 : k; renderInput();
  });
  $("go").onclick = () => { if (labels) send(C.line(p.slots, index)); };
}
function applyHide() {
  const game = !S.analysis;
  document.body.classList.toggle("hidden-thinking", game && readMode === "until" && !S.answered);
  document.body.classList.toggle("read-off", game && readMode === "off");
  $("hiddenBar").hidden = !(game && readMode === "off");
  document.querySelectorAll("#readSeg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === readMode)));
  $("readHelp").textContent = READ_HELP[readMode] || "";
  updateLinks();
  drawChart();
}
function setReadMode(mode) {
  readMode = mode;
  try { localStorage.setItem("pokeuraou-read", mode); } catch (_) { /* ignore */ }
  applyHide();
}
document.querySelectorAll("#readSeg button").forEach((b) => b.onclick = () => setReadMode(b.dataset.v));
$("showRead").onclick = () => setReadMode("show");
// The game page's ways into the analysis page follow the reading mode: a page that hides the
// reading does not show it in the other tab either. During a game it opens the move in hand
// (「この局面を検討」); after it, the game just recorded (and each turn of it from the log).
function analysisOpen() {
  if (S.ended) return [true, ""];
  if (readMode === "off") return [false, "AI の読みを隠しているので、対局が終わると開けます"];
  if (readMode === "until" && !S.answered) return [false, "手を送ると開けます"];
  return [true, "この局面を検討（読んでいる間は AI の手の時間とコアを取り合います）"];
}
function setLink(a, ok, href, title) {
  if (ok) { a.href = href; a.removeAttribute("aria-disabled"); } else { a.removeAttribute("href"); a.setAttribute("aria-disabled", "true"); }
  a.title = title;
}
function updateLinks() {
  if (S.analysis) {
    setLink($("modeGame"), true, GAME_URL, "対戦の画面（別のタブ）");
    $("modeAnalysis").removeAttribute("href");
    return;
  }
  $("modeGame").removeAttribute("href");
  const [ok, why] = analysisOpen();
  setLink($("modeAnalysis"), ok, withQuery(ANALYSIS_URL, S.ended ? gameQuery({ open: "last" }) : { open: "current" }), ok && S.ended ? "この対局を分析する（別のタブ）" : why);
  // A turn's link opens its own game (IKA-356: with --games 2 or more, the last game in the
  // record is not always the one the turn was played in).
  document.querySelectorAll("#log .alink").forEach((a) => {
    const done = !!a.dataset.ended;
    const q = { open: "last", ...(a.dataset.game != null ? { game: a.dataset.game } : {}), turn: a.dataset.turn };
    setLink(a, done, withQuery(ANALYSIS_URL, q), done ? `ターン ${a.dataset.turn} を分析する` : "対局が終わると開けます");
  });
}
// The analysis page's query for the game being played: which game of the record it is.
function gameQuery(q) { return S.gameIndex != null ? { ...q, game: S.gameIndex } : q; }

// ------------------------------------------------------------------ sheets
function renderSheets() {
  const e = S.sheets;
  if (!e) return;
  const order = [aiSide(), e.personSide];
  $("sheets").innerHTML = order.map((i) => `<div><div class="who ${i === e.personSide ? "you" : "ai"}">${esc(e.names[i])}</div>
    ${e.teams[i].map((m) => `<div class="sheet-mon">${art(m.species, "sm")}<div><b>${nameHtml(m.species)}</b> <span class="dim" style="display:inline">@ ${esc(m.item)}</span>
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
  document.querySelectorAll("#themeSeg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.t === theme)));
  drawChart();
}
document.querySelectorAll("#themeSeg button").forEach((b) => b.onclick = () => {
  theme = b.dataset.t;
  try { localStorage.setItem("pokeuraou-theme", theme); } catch (_) { /* ignore */ }
  applyTheme();
});

// ------------------------------------------------------------------ the settings (IKA-349): behind the gear
// Everything a person need not think about: the theme, how the game page shows the reading, the
// analysis page's reading settings and its gauges. Values act at once and stay in localStorage.
function openSettings(open) {
  $("settings").hidden = !open; $("settingsBackdrop").hidden = !open;
  $("gear").setAttribute("aria-expanded", String(open));
  if (open) { const f = $("settings").querySelector("button[aria-pressed=true], button, select, input"); if (f) f.focus(); }
  else $("gear").focus();
}
$("gear").onclick = () => openSettings($("settings").hidden);
$("settingsClose").onclick = () => openSettings(false);
$("settingsBackdrop").onclick = () => openSettings(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("settings").hidden) openSettings(false); });
document.addEventListener("pointerdown", (e) => {
  if ($("settings").hidden || window.matchMedia("(max-width: 560px)").matches) return;
  if (!$("settings").contains(e.target) && !$("gear").contains(e.target)) openSettings(false);
});
const READ_HELP = {
  show: "AI の混合戦略・形勢・読み筋をいつも見せます。",
  until: "あなたが手を送るまで、AI の読みをぼかします。",
  off: "AI の読みの欄を消し、場を広く使います。対局のあと、分析の画面で見られます。",
};
if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawChart);
applyTheme();
applyHide();

// ------------------------------------------------------------------ connection

// ------------------------------------------------------------------ the analysis mode (IKA-337)
// The page reads a chosen position with no budget until 止める. The server says so by sending
// "catalogue" (the games it can read); from then on the input panel is the position picker,
// the clock is the elapsed time, and the status frames (a few a second) carry the steps, the
// lines at the depth guard, the tree's size and the memory against its limits.
function enterAnalysis() {
  if (S.analysis) return;
  S.analysis = true;
  document.body.classList.add("analysis");
  $("inputBox").hidden = true;
  $("timelineBox").hidden = false; $("playedBox").hidden = false; $("pvBox").open = true;
  $("readGroup").hidden = true; $("engineGroup").hidden = false; $("stopTop").hidden = false; $("goTop").hidden = false;
  $("logBox").open = false;
  $("modeGame").removeAttribute("aria-current"); $("modeAnalysis").setAttribute("aria-current", "page");
  applyHide();
  document.querySelector(".brand span").textContent = "検討";
  document.title = "pokeuraou 検討";
  $("readingTitle").textContent = "読み";
  $("oursTitle").textContent = "検討する側の混合戦略";
  $("theirsTitle").textContent = "相手の手の読み";
  $("beliefTitle").textContent = "相手の裏の読み";
  $("balAiName").textContent = "検討する側"; $("balYouName").textContent = "相手";
  $("balLabel").textContent = "形勢（検討する側から見た値）";
  $("input").innerHTML = "";
  setStatus("局面を選んでください");
  $("sideBottom").innerHTML = `<div class="note">ターンを選んで「この局面を読む」と、その局面が出ます</div>`;
  setRunning(false);
}
function fillSelect(el, options, keep) {
  const was = keep ? el.value : null;
  el.innerHTML = options.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
  if (was !== null && options.some(([v]) => String(v) === was)) el.value = was;
}
function pickedGame() {
  const src = S.catalogue && S.catalogue.sources[+$("aSource").value];
  return src ? src.games[+$("aGame").value] : null;
}
function renderPicker(keep) {
  const c = S.catalogue;
  if (!c) return;
  fillSelect($("aSource"), c.sources.map((src) => [src.id, `${src.name}（${src.games.length} 局）`]), keep);
  const src = c.sources[+$("aSource").value];
  fillSelect($("aGame"), src ? src.games.map((g) => [g.index, g.label]) : [], keep);
  renderTurns(keep);
  const st = c.settings;
  if (!keep || !$("aWidth").value) $("aWidth").value = st.width;
  if (!keep || !$("aGuard").value) $("aGuard").value = st.guard;
  $("aSettings").textContent = `最善応答オラクル ${st.oracle}・${st.threads} コア・評価モデル ${st.agent}` +
    (st.maxSteps != null ? `・${st.maxSteps} ステップで止める` : "") + (st.maxSeconds != null ? `・${st.maxSeconds} 秒で止める` : "");
  const problem = src && src.problem ? `読めません: ${src.problem}` : src && !src.games.length ? (src.current ? "進行中の局はまだありません（play_human --current-out）" : "局がありません") : "";
  $("aNote").textContent = problem;
}
function renderTurns(keep) {
  const g = pickedGame();
  fillSelect($("aTurn"), g ? g.decisions.map((d) => [d.index, `ターン ${d.turn}`]) : [], keep);
  if (g && !keep) $("aSide").value = String(g.side);
  $("aGameNote").textContent = g ? [g.open ? "裏は全公開で読みます（記録にシートが無い）" : "", g.note || ""].filter(Boolean).join("・") : "";
  $("aGo").disabled = !g || !g.decisions.length;
  stepState();
}
// One turn back or forward: the picker moves and the position is read again, as when a turn
// is picked from the list. The ends cannot be passed.
function stepState() {
  const sel = $("aTurn"), n = sel.options.length, i = sel.selectedIndex;
  $("aPrev").disabled = $("tPrev").disabled = !n || i <= 0;
  $("aNext").disabled = $("tNext").disabled = !n || i < 0 || i >= n - 1;
  renderTimeline();
}
function stepTurn(d) {
  const sel = $("aTurn"), i = sel.selectedIndex + d;
  if (i < 0 || i >= sel.options.length) return;
  sel.selectedIndex = i; stepState(); startRead();
}
$("tPrev").onclick = () => stepTurn(-1);
$("tNext").onclick = () => stepTurn(1);
$("aPrev").onclick = () => stepTurn(-1);
$("aNext").onclick = () => stepTurn(1);
$("aTurn").addEventListener("change", stepState);
document.addEventListener("keydown", (e) => {
  if (!S.analysis || e.altKey || e.ctrlKey || e.metaKey || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
  const el = document.activeElement;
  if (el && (el.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName))) return;
  e.preventDefault(); stepTurn(e.key === "ArrowLeft" ? -1 : 1);
});
$("aSource").onchange = () => { fillSelect($("aGame"), []); renderPicker(true); renderTurns(false); };
$("aGame").onchange = () => renderTurns(false);
$("aSide").addEventListener("change", renderTimeline);
$("aRefresh").onclick = () => LiveData.command({ cmd: "refresh" });
// ?open=current (the game in progress, its move in hand) or ?open=last (the last game recorded,
// after the list is read again), and &turn=T: the page picks it and reads it (IKA-349).
let pendingOpen = null;
// &game=G (IKA-356) names the game by its index in the record (the last such game: an index
// starts again at 0 each time the game page is started).
try { const q = new URLSearchParams(location.search); if (q.get("open")) pendingOpen = { open: q.get("open"), turn: q.get("turn"), game: q.get("game") }; } catch (_) { /* ignore */ }
function openFromUrl() {
  if (!pendingOpen || !S.catalogue) return;
  const p = pendingOpen, c = S.catalogue;
  if (p.open === "last" && !p.refreshed) { p.refreshed = true; LiveData.command({ cmd: "refresh" }); return; }
  pendingOpen = null;
  try { history.replaceState(null, "", location.pathname); } catch (_) { /* ignore */ }
  const named = (s) => (p.game != null && p.open !== "current" ? s.games.map((g) => g.gameIndex).lastIndexOf(+p.game) : -1);
  const src = c.sources.find((s) => !s.current && named(s) >= 0)
    || c.sources.find((s) => (p.open === "current" ? s.current : !s.current) && s.games.length);
  if (!src) { setStatus(p.open === "current" ? "進行中の局はまだありません" : "記録に局がありません"); return; }
  $("aSource").value = String(src.id); renderPicker(true);
  const gi = p.open === "current" ? 0 : named(src) >= 0 ? named(src) : src.games.length - 1;
  $("aGame").value = String(src.games[gi].index); renderTurns(false);
  const g = pickedGame(), want = p.turn != null ? +p.turn : null;
  const d = want == null ? g.decisions[g.decisions.length - 1] : (g.decisions.find((x) => x.turn === want) || g.decisions[0]);
  if (d) $("aTurn").value = String(d.index);
  stepState(); startRead();
}
function startRead() {
  const g = pickedGame();
  if (!g) return;
  LiveData.command({
    cmd: "analyze", source: +$("aSource").value, game: +$("aGame").value, decision: +$("aTurn").value,
    side: +$("aSide").value, width: +$("aWidth").value || null, guard: +$("aGuard").value || null,
  });
  setStatus("読み始めます…", "think");
  renderTimeline();
}

function stopRead() { LiveData.command({ cmd: "stop" }); }
$("aGo").onclick = startRead;
$("goTop").onclick = () => startRead();
$("aReread").onclick = () => { openSettings(false); startRead(); };
// A wider width for the position being read widens it without stopping (IKA-354): the server
// keeps the tree and adds the new actions to the root. The button says which it will do.
function widensRunning() {
  const st = S.analysisState;
  if (!S.running || !st || st.source == null) return false;
  const guard = +$("aGuard").value || st.guard;
  return st.source === +$("aSource").value && st.game === +$("aGame").value &&
    st.decision === +$("aTurn").value && st.side === +$("aSide").value && guard === st.guard &&
    +$("aWidth").value > st.width;
}
function rereadLabel() {
  $("aReread").textContent = widensRunning() ? "幅を広げる（読みは続ける）" : "この設定で読み直す";
}
for (const id of ["aWidth", "aGuard", "aSide", "aTurn", "aGame", "aSource"]) $(id).addEventListener("input", rereadLabel);
for (const id of ["aSide", "aTurn", "aGame", "aSource"]) $(id).addEventListener("change", rereadLabel);
$("aStop").onclick = stopRead;
$("stopTop").onclick = stopRead;
function setRunning(on) {
  S.running = on;
  if (typeof rereadLabel === "function") rereadLabel();
  $("aStop").disabled = !on; $("stopTop").disabled = !on; $("stopTop").hidden = S.analysis && !on; $("goTop").hidden = !S.analysis || on;
  $("clockbar").classList.toggle("endless", on);
}
function onAnalysis(e) {
  S.analysisState = e;
  if (e.state === "running" && e.grown) {
    // The read goes on at the wider width (IKA-354): nothing restarts.
    $("aWidth").value = e.width;
    log(e.turn, `幅を ${e.width} に広げた（候補集合 ${e.menu[0]}×${e.menu[1]}）。読んだ木はそのまま読み続ける`);
    rereadLabel();
    return;
  }
  if (e.state === "running") {
    S.played = e.played || [];
    setRunning(true);
    // The picker shows what is being read (a read started from the command line, or another page).
    if (e.source != null && S.catalogue) {
      $("aSource").value = String(e.source); renderPicker(true);
      $("aGame").value = String(e.game); renderTurns(true);
      $("aTurn").value = String(e.decision); $("aSide").value = String(e.side); stepState();
      $("aWidth").value = e.width; $("aGuard").value = e.guard;
    }
    setStatus(`読んでいます（ターン ${e.turn}・側 ${e.side}）`, "think", `読んでいます T${e.turn}`);
    $("aNote").textContent = "";
    renderNotes(e);
    log(e.turn, `検討を始めた: 側 ${e.side} から、幅 ${e.width}（候補集合 ${e.menu[0]}×${e.menu[1]}）・最善応答オラクル ${esc(e.oracle)}・深化の深さの上限 ${e.guard} 段` +
      (e.exact ? "・裏は尽きている" : `・相手の裏の決定化 ${e.classCount} 通り`));
  } else {
    setRunning(false);
    setStatus(`${e.stopText}（${(e.seconds || 0).toFixed(1)} 秒・深化のステップ ${(e.steps || 0).toLocaleString("ja-JP")}）`, e.stop === "memory" ? "warn" : "end",
      `止めた T${e.turn}・${(e.steps || 0).toLocaleString("ja-JP")} ステップ`);
    $("aNote").textContent = e.why || "";
    renderNotes(e);
    // The read's value (side 0's), kept for the value over the game. A done event sent again to
    // a page loaded late may be of an earlier position of the game in progress: only one whose
    // turn is the listed decision's counts (the catalogue has the reads too, IKA-356).
    if (e.source != null) {
      const v0 = e.value0 != null ? e.value0 : S.last ? (e.side === 0 ? S.last.value : 1 - S.last.value) : null;
      const d = catalogueDecision(e.source, e.game, e.decision);
      if (v0 != null && (!d || d.turn === e.turn)) {
        S.readValues.set(`${e.source}.${e.game}.${e.decision}`, { value0: v0, side: e.side, steps: e.steps, seconds: e.seconds, width: e.width });
      }
    }
    renderTimeline();
    log(e.turn, `${esc(e.stopText)}${e.why ? `: ${esc(e.why)}` : ""}。深化のステップ ${(e.steps || 0).toLocaleString("ja-JP")}・${(e.seconds || 0).toFixed(1)} 秒・深さの上限に当たった読み筋 ${e.guardLines}・木のノード ${(e.nodes || 0).toLocaleString("ja-JP")}`,
      e.stop === "memory" ? "err" : "");
    if (e.stop === "memory") toast(`メモリの上限の手前で止めました: ${e.why}`);
  }
}
const gb = (x) => (Number.isFinite(x) ? x.toFixed(1) : "–");
// The port's notes, in Japanese (the server translates them, notes_ja.py); the raw ids in title.
function renderNotes(e) {
  const ja = e.notes || [], raw = e.notesRaw || [];
  $("aNotes").innerHTML = ja.map((n) => `<li title="${esc(raw.join("\n"))}">${esc(noteJa(n))}</li>`).join("");
}

// ------------------------------------------------------------------ the analysis page (IKA-349)
// The reads of each position (source.game.decision -> {value0, side, steps, seconds, width}). The
// server keeps them and lists them in the catalogue (IKA-356), so a page loaded again has them.
S.readValues = new Map();
function learnReads(c) {
  S.readValues = new Map();
  c.sources.forEach((src) => src.games.forEach((g) => g.decisions.forEach((d) => {
    if (d.read) S.readValues.set(`${src.id}.${g.index}.${d.index}`, d.read);
  })));
}
function catalogueDecision(source, game, decision) {
  const src = S.catalogue && S.catalogue.sources[source];
  const g = src && src.games[game];
  return g ? g.decisions.find((x) => x.index === decision) || null : null;
}
// The game's value over all its turns: the record's (grey, dashed across turns with none) and
// the ones read on this page (the reading side's colour), from the reading side.
function renderTimeline() {
  if (!S.analysis) return;
  const g = pickedGame(), src = +$("aSource").value, gi = +$("aGame").value, side = +$("aSide").value || 0;
  const svg = $("tSvg"), turns = $("tTurns");
  if (!g || !g.decisions.length) { svg.innerHTML = ""; turns.innerHTML = ""; $("tTurn").textContent = "–"; $("tLabel").textContent = ""; return; }
  const cur = +$("aTurn").value;
  const d = g.decisions.find((x) => x.index === cur) || g.decisions[0];
  $("tTurn").textContent = `T${d.turn}`;
  $("tLabel").textContent = `側 ${side} から読む`;
  const mine = (v0) => (v0 == null ? null : side === 0 ? v0 : 1 - v0);
  const rec = g.decisions.map((x) => mine(x.value));
  const info = g.decisions.map((x) => S.readValues.get(`${src}.${gi}.${x.index}`) || null);
  const read = info.map((r) => (r ? mine(r.value0) : null));
  const all = [...rec, ...read].filter((v) => v != null);
  let lo = Math.min(0.4, ...all), hi = Math.max(0.6, ...all);
  const pad = (hi - lo) * 0.1; lo = Math.max(0, lo - pad); hi = Math.min(1, hi + pad);
  const W = Math.max(200, svg.clientWidth || 1000), Hh = 120, n = g.decisions.length, L = 36, R = 8;
  const X = (i) => L + (n === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (n - 1)), Y = (v) => 8 + (Hh - 16) * (1 - (v - lo) / (hi - lo));
  const ci = g.decisions.indexOf(d);
  let s = `<rect class="band" x="${X(ci) - (W - L - R) / Math.max(1, n - 1) / 2}" y="0" width="${(W - L - R) / Math.max(1, n - 1)}" height="${Hh}"/>`;
  s += `<line class="mid" x1="${L}" x2="${W - R}" y1="${Y(0.5)}" y2="${Y(0.5)}"/><text class="ax" x="${L - 4}" y="${Y(0.5) + 4}">0.50</text>`;
  s += `<text class="ax" x="${L - 4}" y="${Y(hi) + 10}">${hi.toFixed(2)}</text><text class="ax" x="${L - 4}" y="${Y(lo)}">${lo.toFixed(2)}</text>`;
  // record line: solid between neighbours that both have a value, dashed across the gaps
  const idx = rec.map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0);
  for (let k = 1; k < idx.length; k++) {
    const a = idx[k - 1], b = idx[k];
    s += `<line class="${b - a > 1 ? "rec gap" : "rec"}" x1="${X(a)}" y1="${Y(rec[a])}" x2="${X(b)}" y2="${Y(rec[b])}"/>`;
  }
  idx.forEach((i) => { s += `<circle class="recpt" cx="${X(i)}" cy="${Y(rec[i])}" r="3"/>`; });
  const ridx = read.map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0);
  if (ridx.length > 1) s += `<polyline class="readl" points="${ridx.map((i) => `${X(i)},${Y(read[i])}`).join(" ")}"/>`;
  const how = (r) => [r.width ? `幅 ${r.width}` : "", r.steps ? `深化のステップ ${r.steps.toLocaleString("ja-JP")}` : "",
    r.seconds != null ? `${(+r.seconds).toFixed(1)} 秒` : ""].filter(Boolean).join("・");
  ridx.forEach((i) => { s += `<circle class="readpt" cx="${X(i)}" cy="${Y(read[i])}" r="5"><title>T${g.decisions[i].turn} 読んだ値 ${read[i].toFixed(3)}${how(info[i]) ? `（${how(info[i])}）` : ""}</title></circle>`; });
  svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
  svg.innerHTML = s;
  turns.innerHTML = g.decisions.map((x, i) => `<button type="button" data-i="${x.index}" aria-current="${x.index === d.index}"${read[i] != null ? ' class="read" title="読んだターン"' : ""}>T${x.turn}</button>`).join("");
  turns.querySelectorAll("button").forEach((b) => b.onclick = () => { $("aTurn").value = b.dataset.i; stepState(); startRead(); });
  const on = turns.querySelector('[aria-current="true"]');
  if (on && turns.scrollWidth > turns.clientWidth) turns.scrollLeft = on.offsetLeft - turns.clientWidth / 2;
}
// The actions the record played this turn, each with its probability and loss in the read
// equilibrium (the reading side's in its mixture, the other side's in the read of it).
function renderPlayed(s) {
  const box = $("played");
  if (!S.played || !S.played.length) { box.innerHTML = '<span class="note">この記録にはこのターンの手がありません</span>'; return; }
  const me = +(S.analysisState && S.analysisState.side) || 0;
  box.innerHTML = S.played.map((p) => {
    const mine = p.side === me;
    const labels = mine ? s.ours : s.theirs, P = mine ? s.ourP : s.theirP, loss = mine ? s.ourLoss : s.theirLoss;
    const i = labels.indexOf(p.text);
    let n;
    if (i < 0) n = '<span class="badge">候補集合の外</span>';
    else {
      const l = loss[i], bad = l >= 0.02;
      n = `確率 <b>${pct(P[i])}</b>・損 <span class="${bad ? "down" : ""}">${l >= 0.0005 ? "−" + l.toFixed(3) : "0"}</span>${P[i] < LIVE_MIN ? ' <span class="badge">打たない手</span>' : ""}`;
    }
    const who = mine ? "検討する側" : "相手";
    return `<div class="row"><span class="who ${mine ? "a" : "y"}">${who}</span><span class="act">${actHtml(p.parts, boardWho(p.side))}</span><span class="n">${n}</span></div>`;
  }).join("");
}
// The notes the engine writes (English) that the page can say in Japanese; others as they are.
const NOTES_JA = [
  [/^depth-2 kept the (\d+) likeliest branches of a refined cell$/, "選択的延長のセルで、確率の高い分岐 $1 つだけを深さ 2 で読んだ"],
];
function noteJa(note) {
  for (const [re, ja] of NOTES_JA) if (re.test(note)) return note.replace(re, ja);
  return note;
}
function meter(label, value, share, warn, note) {
  const f = Number.isFinite(share) ? Math.max(0, Math.min(1, share)) : 0;
  return `<div class="meter${warn ? " warn" : ""}"><div class="mhead"><small>${label}</small><b class="n">${value}</b></div>` +
    `<div class="mtrack"><i style="width:${(100 * f).toFixed(1)}%"></i></div>${note ? `<small class="mnote">${note}</small>` : ""}</div>`;
}
function onStatus(st) {
  enterAnalysis();
  S.status = st;
  if (st.state === "running" && !S.running) setRunning(true);
  const cells = [
    `<div class="stat"><small>経過</small><b class="n">${fmtTime(st.elapsed)}</b></div>`,
    `<div class="stat"><small>深化のステップ</small><b class="n">${st.steps.toLocaleString("ja-JP")}</b></div>`,
    `<div class="stat${st.guardLines ? " hot" : ""}"><small>深さの上限（${st.guard} 段）に当たった読み筋</small><b class="n">${st.guardLines.toLocaleString("ja-JP")}</b></div>`,
    `<div class="stat"><small>木のノード・段</small><b class="n">${st.nodes.toLocaleString("ja-JP")}・${st.depth}</b></div>`,
  ];
  const lim = (x) => (x > 0 ? x : NaN);
  const meters = [
    meter("このプロセスと補助のメモリ", `${gb(st.rss)} GB`, st.rss / lim(st.rssLimit), st.rssLimit > 0 && st.rss >= 0.9 * st.rssLimit,
      st.rssLimit > 0 ? `上限 ${st.rssLimit} GB` : "上限なし"),
    meter("ホストの空き", `${gb(st.free)} GB`, st.freeFloor > 0 && Number.isFinite(st.free) ? st.freeFloor / st.free : NaN,
      st.freeFloor > 0 && st.free <= st.freeFloor / 0.9, st.freeFloor > 0 ? `下限 ${st.freeFloor} GB` : "下限なし"),
    meter("GPU のメモリ（カード全体）", `${gb(st.gpu)} / ${gb(st.gpuTotal)} GB`, st.gpu / lim(st.gpuLimit),
      st.gpuLimit > 0 && st.gpu >= 0.9 * st.gpuLimit, st.gpuLimit > 0 ? `上限 ${st.gpuLimit} GB` : "見張りなし"),
  ];
  $("aMeters").innerHTML = `<div class="stats">${cells.join("")}</div>${meters.join("")}`;
  $("meterSum").textContent = `経過 ${fmtTime(st.elapsed)}・ステップ ${st.steps.toLocaleString("ja-JP")}・ノード ${st.nodes.toLocaleString("ja-JP")}${st.warn ? "・上限に近い" : ""}`;
  $("aMeters").classList.toggle("warn", st.warn);
  if (st.state === "running") $("clocktext").textContent = `経過 ${fmtTime(st.elapsed)}・深化のステップ ${st.steps.toLocaleString("ja-JP")}`;
}
function fmtTime(sec) {
  if (!Number.isFinite(sec)) return "–";
  if (sec < 60) return `${sec.toFixed(1)} 秒`;
  const m = Math.floor(sec / 60), r = Math.floor(sec - 60 * m);
  return `${m} 分 ${String(r).padStart(2, "0")} 秒`;
}

LiveData.connect({
  onEvent, onStep, onStatus,
  onOpen: () => setStatus("接続した"),
  onClose: () => { stopSelecting(); setStatus("切れた（再読み込みで繋ぎ直す）"); },
});
})();
