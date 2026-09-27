"""The port's notes (``unmodelled``) as a person reads them (IKA-349).

The port says what it did not model in short English ids -- ``attacker.ability:magicbounce``,
``damaging move: knockoff``, ``thaw roll (1 in 4; ...)``. The analysis page shows them to a
person, so each is said in Japanese: the ability, item and move by their names from
Showdown's own text (`names.Localiser`, generated from ``vendor/data/text``, never written
by hand) and what the port did instead ("近似"). A note this table does not know is kept as
it is, with its ids named, rather than guessed at. The notes written to ``--out`` stay the
port's own: this is for the page only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

#: Fixed notes, in the port's words (`rust/src/*.rs` `report(...)`), and what they mean.
FIXED = {
    "ability: liquidooze (drain hurts instead of healing, not applied)":
        "近似: ヘドロえき（吸い取る技で逆にダメージ）を入れていない",
    "randomNormal target (the first foe; not branched)":
        "近似: 対象が無作為の技は、最初の相手に当たるとして読んだ（分岐にしていない）",
    "rampage length (the two-turn one of 2-or-3; not branched)":
        "近似: あばれる系の長さは 2 ターンとして読んだ（分岐にしていない）",
    "rampage length (not on the position; read as its last turn)":
        "近似: あばれる系の残りが局面に無いので、最後のターンとして読んだ",
    "confusion length (the shortest of 2-to-5; not branched)":
        "近似: こんらんの長さは最短として読んだ（分岐にしていない）",
    "thaw roll (1 in 4; the cured state is not branched)":
        "近似: こおりが解ける 1/4 を分岐にしていない",
    "cursedbody (30% disable, not branched)":
        "近似: のろわれボディ（3 割のかなしばり）を分岐にしていない",
    "ability: poisontouch (30% poison not branched)":
        "近似: どくしゅ（3 割のどく）を分岐にしていない",
    "secondary on a multi-hit move (applied after the last hit)":
        "近似: 連続技の追加効果は最後の 1 発の後に 1 回として読んだ",
    "residual speed tie (Showdown breaks it at random)":
        "近似: ターン終わりの処理の同速を分岐にしていない（Showdown は無作為に決める）",
    "ability: moody (end-of-turn +2/-1 not applied)":
        "近似: ムラっけ（ターン終わりの能力変化）を入れていない",
    "forced switch (the first on the bench; not branched)":
        "近似: 強制交代で出る体は控えの最初の体として読んだ（分岐にしていない）",
    "trace found nothing to copy (it keeps seeking; not modelled)":
        "近似: トレースが写す特性を見つけられない時の続きを入れていない",
    "transform: Illusion (assumed not up)":
        "近似: へんしんの相手のイリュージョンは解けているとして読んだ",
}

_ROLE = {"attacker": "攻撃側", "defender": "防御側"}
_EFFECT = re.compile(r"^(attacker|defender)\.(ability|item):(\S+)$")
_DEPTH2 = re.compile(r"^depth-2 kept the (\d+) likeliest branches of a refined cell$")
_ID_NOTE = re.compile(
    r"^(damaging move|status move|contact ability|on-hit ability|berry eaten by another): (\S+)$"
)
_BOOST = re.compile(r"^boost:(\S+)$")
_WORD = re.compile(r"[a-z][a-z0-9]+")


def _name(loc: Any, kind: str, ident: str) -> str:  # noqa: ANN401 - names.Localiser
    if loc is None:
        return ident
    lookup = getattr(loc, kind, None)
    got = lookup(ident) if lookup is not None else None
    return got if got and got != ident else ident


def _any_name(loc: Any, ident: str) -> str:  # noqa: ANN401
    for kind in ("ability", "move", "item", "species"):
        got = _name(loc, kind, ident)
        if got != ident:
            return got
    return ident


def note_ja(loc: Any, note: str) -> str:  # noqa: ANN401
    """One note in Japanese (the module's docstring); an unknown one with its ids named."""
    if note in FIXED:
        return FIXED[note]
    m = _DEPTH2.match(note)
    if m:
        return f"選択的延長のセルで、確率の高い分岐 {m.group(1)} つだけを深さ 2 で読んだ"
    m = _EFFECT.match(note)
    if m:
        role, kind, ident = m.groups()
        what = "特性" if kind == "ability" else "持ち物"
        return f"近似: {_ROLE[role]}の{what} {_name(loc, kind, ident)} をダメージ計算に入れていない"
    m = _ID_NOTE.match(note)
    if m:
        kind, ident = m.groups()
        if kind == "damaging move":
            return f"近似: 攻撃技 {_name(loc, 'move', ident)} の効果の一部を入れていない"
        if kind == "status move":
            return f"近似: 変化技 {_name(loc, 'move', ident)} の効果を入れていない"
        if kind == "berry eaten by another":
            return f"近似: 他の体が {_name(loc, 'item', ident)} を食べた時の効果を入れていない"
        when = "接触" if kind.startswith("contact") else "攻撃を受けた時"
        return f"近似: 特性 {_name(loc, 'ability', ident)}（{when}の効果）を入れていない"
    m = _BOOST.match(note)
    if m:
        stat = loc.stat(m.group(1), short=True) if loc is not None and hasattr(loc, "stat") else m.group(1)
        return f"近似: 能力変化（{stat}）の一部を入れていない"
    return "注記: " + _WORD.sub(lambda w: _any_name(loc, w.group(0)), note)


def notes_ja(loc: Any, notes: Iterable[str]) -> list[str]:  # noqa: ANN401
    """The notes in Japanese, in order, a note that reads the same as another said once. The
    attacker's and the defender's note on one ability become one: "特性 X をダメージ計算に入れていない"."""
    notes = list(notes)
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for note in notes:
        m = _EFFECT.match(note)
        if m:
            _role, kind, ident = m.groups()
            if (kind, ident) in seen:
                continue
            seen.add((kind, ident))
            roles = [r for r in ("attacker", "defender") if f"{r}.{kind}:{ident}" in notes]
            what = "特性" if kind == "ability" else "持ち物"
            who = "・".join(_ROLE[r] for r in roles)
            text = f"近似: {who}の{what} {_name(loc, kind, ident)} をダメージ計算に入れていない"
        else:
            text = note_ja(loc, note)
        if text not in out:
            out.append(text)
    return out


__all__ = ["FIXED", "note_ja", "notes_ja"]
