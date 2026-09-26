// The data half of the live page (IKA-332): the socket, the frames, the table of strings.
// It knows nothing of the page's look: it turns frames into plain objects and hands them
// to the handlers it is given. The shapes are the table in records/IKA-332.md.
//
//   LiveData.connect({ onEvent(e), onStep(s), onStatus(st), onOpen(), onClose() })
//   LiveData.send(line)          // the person's answer, as a script line
//   LiveData.command(obj)        // the analysis mode's requests: {cmd: "analyze"|"stop"|"refresh", ...}
"use strict";
(function () {
const STRINGS = 1, EVENT = 2, STEP = 3, STATUS = 4, LABELS = 5;
const NONE = 0xFFFFFFFF;
// The analysis mode's status frame (IKA-337), after the type byte: liveview._STATUS.
const STATUS_FIELDS = [
  ["state", "u8"], ["flags", "u8"], ["elapsed", "f64"], ["steps", "u32"], ["guardLines", "u32"],
  ["nodes", "u32"], ["guard", "u16"], ["depth", "u16"], ["threads", "u16"], ["rss", "f32"],
  ["rssLimit", "f32"], ["free", "f32"], ["freeFloor", "f32"], ["gpu", "f32"], ["gpuTotal", "f32"],
  ["gpuLimit", "f32"],
];
const STATES = ["idle", "running", "done"];
function decodeStatus(buf) {
  const dv = new DataView(buf);
  let at = 1;
  const st = {};
  for (const [name, type] of STATUS_FIELDS) {
    if (type === "u8") { st[name] = dv.getUint8(at); at += 1; }
    else if (type === "u16") { st[name] = dv.getUint16(at, true); at += 2; }
    else if (type === "u32") { st[name] = dv.getUint32(at, true); at += 4; }
    else if (type === "f32") { st[name] = dv.getFloat32(at, true); at += 4; }
    else { st[name] = dv.getFloat64(at, true); at += 8; }
  }
  st.state = STATES[st.state] || "idle";
  st.warn = !!(st.flags & 1);
  return st;
}
const KINDS = ["start", "refine", "refused", "widen", "done"];
const READS = ["leaf", "deep", "refused"];
// An action's label is an id into the table of labels (IKA-345): its parts, each [slot, text]
// with the active slot whose action it is. The text is the parts joined by this.
const SLOT_SEPARATOR = " ／ ";
const strings = [];
const labels = [];
const labelText = (parts) => parts.map((p) => p[1]).join(SLOT_SEPARATOR);
let socket = null;

function decodeStep(buf) {
  const dv = new DataView(buf);
  let at = 1;
  const u8 = () => dv.getUint8(at++);
  const u16 = () => { const v = dv.getUint16(at, true); at += 2; return v; };
  const i16 = () => { const v = dv.getInt16(at, true); at += 2; return v; };
  const u32 = () => { const v = dv.getUint32(at, true); at += 4; return v; };
  const f32 = () => { const v = dv.getFloat32(at, true); at += 4; return v; };
  const f64 = () => { const v = dv.getFloat64(at, true); at += 8; return v; };
  const s = {};
  s.kind = KINDS[u8()]; s.depth = u16(); s.decision = u32(); s.turn = u32(); s.step = u32();
  s.ms = f32(); s.budget = u32(); s.spent = f32(); s.cells = u32(); s.fills = u32(); s.refines = u32();
  s.expanded = u32(); s.refused = u32(); s.probed = u32(); s.widened = u16(); s.swapped = u16();
  s.value0 = f64(); s.value = f64(); s.read = f32(); s.gap = f32();
  const nO = u16(), nT = u16(), nC = u16();
  const arr = (n, f) => { const a = []; for (let i = 0; i < n; i++) a.push(f()); return a; };
  const lab = () => labels[u32()];
  const str = () => { const i = u32(); return i === NONE ? null : strings[i]; };
  const split = (t) => t.split(SLOT_SEPARATOR).filter((x) => x !== "");
  s.oursParts = arr(nO, lab); s.ourP = arr(nO, f64); s.ourLoss = arr(nO, f32);
  s.theirsParts = arr(nT, lab); s.theirP = arr(nT, f32); s.theirLoss = arr(nT, f32);
  s.ours = s.oursParts.map(labelText); s.theirs = s.theirsParts.map(labelText);
  s.classes = arr(nC, () => ({
    weight: f32(), value: f32(), bench: split(strings[u32()]), benchIds: split(strings[u32()]),
    p: arr(nT, f32),
  }));
  const top = () => arr(u8(), () => { const parts = lab(); return [labelText(parts), f32(), parts]; });
  const branch = () => {
    const b = { weight: f32(), value: f32(), what: strings[u32()] };
    const flags = u8();
    b.ended = !!(flags & 1); b.more = !!(flags & 4);
    b.causes = arr(u8(), () => ({ plain: !!u8(), head: strings[u32()], body: strings[u32()], title: strings[u32()] }));
    b.changes = arr(2, () => arr(u8(), () => {
      const c = { name: strings[u32()], sprite: strings[u32()], from: u8(), to: u8() };
      const f = u8(); c.entered = !!(f & 1); c.fainted = !!(f & 2); c.status = str();
      return c;
    }));
    if (flags & 2) {
      const value = f32();
      const field = arr(2, () => arr(u8(), () => {
        const m = { name: str(), sprite: str(), percent: u8() };
        const f = u8();
        return f & 4 ? null : { ...m, fainted: !!(f & 1), new: !!(f & 2) };
      }));
      b.node = { value, field, ours: top(), theirs: top(), pairs: pairs() };
    }
    return b;
  };
  const pairs = () => arr(u8(), () => {
    const oursParts = lab(), theirsParts = lab();
    const p = { oursParts, theirsParts, ours: labelText(oursParts), theirs: labelText(theirsParts),
      klass: i16(), read: READS[u8()], p: f32(), value: f32() };
    p.branches = arr(u8(), branch);
    return p;
  });
  s.pv = pairs();
  return s;
}

function onFrame(buf, handlers) {
  const type = new Uint8Array(buf, 0, 1)[0];
  if (type === STRINGS) {
    const dv = new DataView(buf);
    const first = dv.getUint32(1, true), count = dv.getUint16(5, true);
    let at = 7;
    const dec = new TextDecoder();
    for (let i = 0; i < count; i++) {
      const n = dv.getUint16(at, true); at += 2;
      strings[first + i] = dec.decode(new Uint8Array(buf, at, n)); at += n;
    }
  } else if (type === LABELS) {
    const dv = new DataView(buf);
    const first = dv.getUint32(1, true), count = dv.getUint16(5, true);
    let at = 7;
    for (let i = 0; i < count; i++) {
      const n = dv.getUint8(at); at += 1;
      const parts = [];
      // [slot, text, verb, target side (-1: none), target name, target sprite, mega, switch]
      for (let k = 0; k < n; k++) {
        const tn = dv.getUint32(at + 10, true), ts = dv.getUint32(at + 14, true);
        parts.push([dv.getInt8(at), strings[dv.getUint32(at + 1, true)], strings[dv.getUint32(at + 5, true)],
          dv.getInt8(at + 9), tn === NONE ? "" : strings[tn], ts === NONE ? "" : strings[ts],
          !!(dv.getUint8(at + 18) & 1), !!(dv.getUint8(at + 18) & 2)]);
        at += 19;
      }
      labels[first + i] = parts;
    }
  } else if (type === EVENT) {
    handlers.onEvent(JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 1))));
  } else if (type === STEP) {
    handlers.onStep(decodeStep(buf));
  } else if (type === STATUS) {
    if (handlers.onStatus) handlers.onStatus(decodeStatus(buf));
  }
}

function connect(handlers) {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => handlers.onOpen && handlers.onOpen();
  socket.onmessage = (m) => onFrame(m.data, handlers);
  socket.onclose = () => handlers.onClose && handlers.onClose();
}

function send(line) {
  if (socket && socket.readyState === 1) socket.send(JSON.stringify({ line }));
}

function command(message) {
  if (socket && socket.readyState === 1) socket.send(JSON.stringify(message));
}

window.LiveData = { connect, send, command, decodeStep, decodeStatus, strings, labels, SLOT_SEPARATOR };
})();
