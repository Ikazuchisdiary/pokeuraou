// The person's move input with Mega Evolution apart from the moves (IKA-345): each slot lists
// its actions once (a move, not the move and the move + Mega), a toggle on the slot says
// "Mega Evolve", and the pair is turned back into one of the prompt's legal actions just
// before it is sent. A pair that is not one of them cannot be sent.
//
//   slots: the prompt's `slots` -- per legal action, per active slot, [choice, words, part].
//   A Mega Evolution is the choice with " mega" at the end (actions.MoveAction.to_choice).
"use strict";
(function (root) {
const MEGA = " mega";
const isMega = (choice) => choice.endsWith(MEGA);
const base = (choice) => (isMega(choice) ? choice.slice(0, -MEGA.length) : choice);

// The slots whose Pokemon can Mega Evolve this turn (some legal action does it there).
function megaSlots(slots) {
  const out = [];
  (slots || []).forEach((a) => a.forEach((x, k) => { if (isMega(x[0]) && !out.includes(k)) out.push(k); }));
  return out.sort((a, b) => a - b);
}

// Slot k's options once each, given the base choices of the slots before it: [base, words,
// part], the words and part of the plain move (the Mega variant's with its " + メガ" and flag
// taken off where only that one exists).
function options(slots, k, chosen) {
  const out = [];
  for (const a of slots || []) {
    if (!chosen.slice(0, k).every((c, j) => base(a[j][0]) === c)) continue;
    const x = a[k], b = base(x[0]);
    const at = out.findIndex((o) => o[0] === b);
    if (isMega(x[0])) {
      if (at < 0) {
        const part = x[2] ? [...x[2]] : null;
        if (part) part[6] = false;
        out.push([b, String(x[1]).replace(/ \+ (メガ|Mega)$/, ""), part]);
      }
    } else if (at < 0) out.push([b, x[1], x[2] || null]);
    else out[at] = [b, x[1], x[2] || null];
  }
  return out;
}

// The legal action (its index in `slots`) that is these base choices with slot `mega` (-1:
// none) Mega Evolving; -1 when there is none.
function resolve(slots, chosen, mega) {
  if (!slots || !slots.length || chosen.length !== slots[0].length || chosen.some((c) => c === undefined)) return -1;
  return slots.findIndex((a) => a.every((x, j) => x[0] === (j === mega ? chosen[j] + MEGA : chosen[j])));
}

// The line the page sends for that action (the choices joined, as the person's script reads them).
function line(slots, index) {
  return index < 0 ? null : slots[index].map((x) => x[0]).join(", ");
}

const api = { MEGA, isMega, base, megaSlots, options, resolve, line };
root.LiveChoice = api;
if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
