// The data half of the live page (IKA-332): the socket, the frames, the table of strings.
// It knows nothing of the page's look: it turns frames into plain objects and hands them
// to the handlers it is given. The shapes are the table in records/IKA-332.md.
//
//   LiveData.connect({ onEvent(e), onStep(s), onOpen(), onClose() })
//   LiveData.send(line)          // the person's answer, as a script line
"use strict";
(function () {
const STRINGS = 1, EVENT = 2, STEP = 3;
const KINDS = ["start", "refine", "refused", "widen", "done"];
const READS = ["leaf", "deep", "refused"];
// An action's label is its slots' parts joined by this (slot k = the side's k-th active).
const SLOT_SEPARATOR = " ／ ";
const parts = (label) => label.split(SLOT_SEPARATOR).filter((x) => x !== "");
const strings = [];
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
  const ids = (n) => { const a = []; for (let i = 0; i < n; i++) a.push(strings[u32()]); return a; };
  const arr = (n, f) => { const a = []; for (let i = 0; i < n; i++) a.push(f()); return a; };
  s.ours = ids(nO); s.ourP = arr(nO, f64); s.ourLoss = arr(nO, f32);
  s.theirs = ids(nT); s.theirP = arr(nT, f32); s.theirLoss = arr(nT, f32);
  s.oursSlots = s.ours.map(parts); s.theirsSlots = s.theirs.map(parts);
  s.classes = arr(nC, () => ({
    weight: f32(), value: f32(), bench: parts(strings[u32()]), benchIds: parts(strings[u32()]),
    p: arr(nT, f32),
  }));
  const top = () => arr(u8(), () => [strings[u32()], f32()]);
  const branch = () => {
    const b = { weight: f32(), value: f32(), what: strings[u32()] };
    const flags = u8();
    b.ended = !!(flags & 1); b.more = !!(flags & 4);
    if (flags & 2) b.node = { value: f32(), ours: top(), theirs: top(), pairs: pairs() };
    return b;
  };
  const pairs = () => arr(u8(), () => {
    const p = { ours: strings[u32()], theirs: strings[u32()], klass: i16(), read: READS[u8()], p: f32(), value: f32() };
    p.oursSlots = parts(p.ours); p.theirsSlots = parts(p.theirs);
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
  } else if (type === EVENT) {
    handlers.onEvent(JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 1))));
  } else if (type === STEP) {
    handlers.onStep(decodeStep(buf));
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

window.LiveData = { connect, send, decodeStep, strings, SLOT_SEPARATOR };
})();
