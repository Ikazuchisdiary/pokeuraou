"""Imports the Match Up Web sheet's Regulation M-C teams from pokepast.es, SP included.

The sheet is one player's list of the teams worth preparing for -- a **prep list, not the
field**. IKA-79 found the field itself on Reportworm (the Baltimore Regional, 1,067 entries
valid in M-C), so the pool's primary source is the tournament. What this file adds is the
one thing a tournament sheet structurally lacks, the SP spread: Champions open team sheets
reveal the nature and blank the investment, while a pokepaste is the player's own export,
and in the Champions format the field it labels ``EVs:`` holds the SP.

Six steps, each checkable on its own:

1. the sheet's CSV export (one tab, named by its gid) -> the paste URL on each team's row
2. each paste's ``/raw`` text, cached byte for byte under ``data/pool/regmc-matchupweb/``
3. the Showdown export blocks, parsed
4. mega listings normalised to base species + stone. A member written as the mega forme
   (``Salamence-Mega @ Salamencite``) is identified by the stone it holds, so the lookup is
   the regulation's own mega map read backwards, and a forme whose stone does not produce
   it is a problem rather than a guess
5. every field validated against the M-C dump the way ``teams.load_roster`` does: team-legal
   species, the *pre-mega* ability, legal item, nature and moves, the SP caps, and the
   species and item clauses
6. ``data/pool/regmc-matchupweb.json``, with the sheet URL, gid, row, paste id and fetch
   time on every team

A team with any problem is excluded whole and listed with its reasons, never repaired. A
missing field is not filled in: ``teams.py`` keeps "making a set up" out as a fourth
source, and a plausible nature invented here would come back out as a win probability.

The one exception is a **transcription**, not a repair (IKA-138). A field a paste leaves
out may be taken from the same team's *published team sheet* -- a tournament's open sheet
for the same player's same six -- when the user has approved that one field, and it is
listed in ``SUPPLEMENTS`` with the event, player, placing, standings file and the position
of that value in it. That does not reopen the fourth source: nothing is chosen as
plausible, the value is the one the player registered, and it is traceable to a named
record. So each supplement is checked, not trusted: when the standings file is on disk
the value at its position must be the one written here, the event and player must be the
ones named, and the sheet's individual must be the paste's (species, item, ability,
moves), or the run stops. The team then carries ``supplied`` -- which member, which field,
which value, from where -- and the member names the field in its own ``supplied``, so a
reader can tell a paste field from a transcribed one. A supplement for a field the paste
does write stops the run: it would be overriding the export, not filling it.

The sheet is live. Between the sizing probe and the first import its author put a
"Your Team" row above the opponents, moving every row down four and renaming one team, with
the 65 pastes unchanged. So the output names the sheet bytes it read (hash and fetch time)
beside each row number, and that own-side row is never read as an opponent: a paste link put
there would be the author's own team.

Fetching is curl in a subprocess on purpose. This machine's Python trust store refuses
pokepast.es (``CERTIFICATE_VERIFY_FAILED: certificate has expired``) while curl's store is
current, and turning verification off would make every later fetch unauthenticated. Network
requests are spaced ``--delay`` seconds apart.

The fetch times live beside the cached bytes (``fetch-log.json``) and the output copies them
from there, so a rerun from the cache writes the same bytes; the log's hashes are checked on
every read, so an edited cache stops the run instead of being published. ``--refetch``
takes the sheet and every paste again.

Nothing fetched is committed. ``data/`` is gitignored, the standings' handling: the teams
are other people's published work. What is committed is this importer and the sheet
reference below.

    uv run python tools/fetch_pastes.py             # the cache where present, fetch the rest
    uv run python tools/fetch_pastes.py --refetch   # take the sheet and every paste again
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.regulation import Regulation, load_regulation, regulation_dir, to_id  # noqa: E402

REGULATION = "gen9championsvgc2026regmc"
POOL_ID = "regmc-matchupweb"
#: The sheet reference: the one part of the source that is committed. Three tabs (Team 1/2/3)
#: share the same 65-team opponent list; their own columns are the author's side of the
#: prep, so one tab is enough.
SHEET_ID = "1JZJg4-bCW4nv3XsZBt72TApmoj4rJluX73MHw0Xmxj0"
SHEET_GID = "919829702"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/"
SHEET_EXPORT = f"{SHEET_URL}export?format=csv&gid={SHEET_GID}"
PASTE_URL = re.compile(r"https?://pokepast\.es/([0-9a-f]{8,})")
USER_AGENT = "pokeuraou/0.0 (Pokemon Champions doubles solver; personal research; one-off manual fetch)"

#: Showdown's export labels -> this repository's stat ids.
STAT_LABELS = {"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}
#: Lines a paste may carry that say nothing the format keeps: cosmetics, and fields that
#: Champions fixes (IVs are 31) or disables (Terastallization). Kept as notes, not dropped.
IGNORED_PREFIXES = ("Shiny:", "Tera Type:", "Happiness:", "Pokeball:", "IVs:", "Gigantamax:")

CHARACTER = (
    "prep list であって field ではない。Match Up Web のシートは1人のプレイヤーが『対策すべき相手』"
    "として選んだ一覧（要約の側）で、ここから一様に引くのは作者が気にした構築を等確率で引くという"
    "意味であり、使用率ではない。"
)
ROLE = (
    "配分（SP）の出典。M-C の環境プールの一次資料は大会の実エントリー（IKA-79: Baltimore Regional "
    "2026-09-19〜20、1,082人中 1,067本が M-C に完全適合。tools/fetch_standings.py 2027 baltimore）。"
    "大会のオープンシートは性格を見せて SP を伏せるので、実配分はこちらにしか無い。"
)
NOTES = {
    "sp": "paste の `EVs:` 欄に書かれている数字は SP（Champions の形式。合計 66・各 32 まで）",
    "normalisation": (
        "メガ形で書かれた個体（例 `Salamence-Mega @ Salamencite`）は、持っている石から素の種族に戻す"
        "（M-C ダンプの megaMap を逆に引く）。`writtenAs` が paste の表記"
    ),
    "exclusion": (
        "問題が1つでもある構築は丸ごと除外し、理由を excluded に残す。欠けた欄は推測しない"
        "（teams.py: 4つめの出典を作らない）"
    ),
    "names": "種族・特性・道具・技は M-C ダンプの表記（to_id で paste の表記と一致したもの）",
    "supplied": (
        "paste に無い欄を、同じ構築の公開シート（大会のオープンシート）から転記したものは、構築の "
        "supplied に個体・欄・値・出典（大会・プレイヤー・順位・standings のファイルとその中の位置）を、"
        "個体の supplied に欄名を残す。推測ではなく転記で、ユーザが欄ごとに承認したものだけ（IKA-138）"
    ),
}


# -- transcriptions from a published team sheet (IKA-138) -------------------------------


@dataclass(frozen=True, slots=True)
class Supplement:
    """One field a paste leaves out, transcribed from the same team's open team sheet.

    ``player_code`` and ``team_index`` locate the value in a Reportworm standings file
    (``tools/fetch_standings.py``): ``standings[player_code].team[team_index][field]``.
    """

    paste_id: str
    member: int  # index in the paste
    written_as: str  # the paste's species, so a reordered paste cannot shift the target
    field: str  # a PasteMon field the paste left empty
    value: str
    event: str  # the standings file's event name
    player: str
    place: int
    standings_file: str  # under data/standings/
    player_code: str
    team_index: int
    approved: str

    @property
    def pointer(self) -> str:
        return f"standings.{self.player_code}.team[{self.team_index}].{self.field}"

    def record(self, checked: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "member": self.member,
            "writtenAs": self.written_as,
            "field": self.field,
            "value": self.value,
            "source": {
                "kind": "open team sheet",
                "event": self.event,
                "player": self.player,
                "place": self.place,
                "standings": f"data/standings/{self.standings_file}",
                "pointer": self.pointer,
                "tool": "tools/fetch_standings.py",
                "approved": self.approved,
                "checkedAgainst": checked,
            },
        }


#: The fields a supplement may fill: the ones an open team sheet shows. SP is not one of
#: them -- the sheet blanks it, which is why this pool exists.
SUPPLIABLE = frozenset({"nature", "ability", "item"})

SUPPLEMENTS: tuple[Supplement, ...] = (
    # "Wolfe's Mence + Garde": the paste's Salamence (19 Atk / 15 SpA / 32 Spe) has no
    # nature line. Wolfe Glick's Baltimore sheet has the same six with every other field
    # equal, and its Salamence is Naive. Approved by the user on 9/23 ("Naive でおｋ").
    Supplement(
        paste_id="9de5ee0a9fd58d26",
        member=0,
        written_as="Salamence-Mega",
        field="nature",
        value="Naive",
        event="Baltimore Regional",
        player="Wolfe Glick",
        place=15,
        standings_file="2027-baltimore.json.gz",
        player_code="wolfe-glick",
        team_index=3,
        approved="ユーザ承認 2026-09-23（IKA-138）",
    ),
)


def check_supplement(
    reg: Regulation, supplement: Supplement, mon: PasteMon, standings_dir: Path | None
) -> dict[str, Any] | None:
    """Stop the run unless the supplement fills an empty field of the individual its sheet
    shows. Returns what it was checked against, or ``None`` when the file is not on disk."""
    where = f"supplement {supplement.paste_id}[{supplement.member}] {supplement.field}"
    if supplement.field not in SUPPLIABLE:
        raise SystemExit(f"{where}: {supplement.field!r} is not a field an open sheet shows")
    if to_id(mon.species) != to_id(supplement.written_as):
        raise SystemExit(f"{where}: the paste has {mon.species!r} there, not {supplement.written_as!r}")
    if getattr(mon, supplement.field) is not None:
        raise SystemExit(f"{where}: the paste writes {getattr(mon, supplement.field)!r}; not overriding it")
    if standings_dir is None:
        return None
    path = standings_dir / supplement.standings_file
    if not path.exists():
        return None
    packed = path.read_bytes()
    raw = gzip.decompress(packed)
    doc = json.loads(raw)
    if doc["event"]["name"] != supplement.event:
        raise SystemExit(f"{where}: {path.name} is {doc['event']['name']!r}, not {supplement.event!r}")
    entry = doc["standings"].get(supplement.player_code)
    if entry is None or entry["name"] != supplement.player or entry["place"] != supplement.place:
        raise SystemExit(
            f"{where}: {supplement.player_code} is not {supplement.player} at {supplement.place}"
        )
    shown = entry["team"][supplement.team_index]
    if shown.get(supplement.field) != supplement.value:
        raise SystemExit(f"{where}: the sheet has {shown.get(supplement.field)!r}, not {supplement.value!r}")
    base = base_of_mega(reg).get((to_id(mon.species), to_id(mon.item or "")), to_id(mon.species))
    same = {
        "species": (to_id(shown["name"]), base),
        "item": (to_id(shown["item"]), to_id(mon.item or "")),
        "ability": (to_id(shown["ability"]), to_id(mon.ability or "")),
        "moves": (sorted(to_id(m["name"]) for m in shown["moves"]), sorted(map(to_id, mon.moves))),
    }
    differ = [name for name, (a, b) in same.items() if name != supplement.field and a != b]
    if differ:
        raise SystemExit(f"{where}: the sheet's individual differs from the paste's in {', '.join(differ)}")
    return {"sha256": sha256(packed), "matched": sorted(n for n in same if n != supplement.field)}


# -- the sheet -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SheetEntry:
    row: int  # 1-based spreadsheet row, so a reader can find it in the sheet
    name: str
    paste_id: str


#: The template's own-side row. It sits above the opponents and holds a placeholder today
#: ("Your PokePaste"); a paste link put there would be the author's team, not an opponent.
OWN_ROW_NAMES = frozenset({"your team"})


def parse_sheet(text: str) -> tuple[list[SheetEntry], list[SheetEntry]]:
    """The opponents -- each row with a paste link in the ``Pokepastes`` column, in sheet
    order -- and, apart, any own-side row that carries a link.

    The column is found by its header rather than by position, and the CSV is read as CSV:
    the header cells span lines, so splitting on newlines first would misnumber the rows.
    """
    rows = list(csv.reader(io.StringIO(text, newline="")))
    column = next(
        (
            (r, c)
            for r, row in enumerate(rows)
            for c, cell in enumerate(row)
            if cell.strip().lower().startswith("pokepastes")
        ),
        None,
    )
    if column is None:
        raise SystemExit("sheet: no 'Pokepastes' column header")
    header_row, col = column
    entries: list[SheetEntry] = []
    own: list[SheetEntry] = []
    seen: dict[str, int] = {}
    for r, row in enumerate(rows[header_row + 1 :], start=header_row + 2):
        if len(row) <= col:
            continue
        match = PASTE_URL.search(row[col])
        if match is None:
            continue
        paste_id = match.group(1)
        if paste_id in seen:
            raise SystemExit(f"sheet: paste {paste_id} on rows {seen[paste_id]} and {r}")
        seen[paste_id] = r
        entry = SheetEntry(row=r, name=row[0].strip(), paste_id=paste_id)
        (own if entry.name.lower() in OWN_ROW_NAMES else entries).append(entry)
    return entries, own


# -- one paste ---------------------------------------------------------------------------


@dataclass(slots=True)
class PasteMon:
    """One Showdown export block, as the paste wrote it."""

    species: str  # nickname and gender removed, otherwise verbatim
    nickname: str | None = None
    gender: str | None = None
    item: str | None = None
    ability: str | None = None
    nature: str | None = None
    level: int | None = None
    sp: dict[str, int] = field(default_factory=dict)
    moves: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    #: (field, reason) for what the text itself does not let us read.
    problems: list[tuple[str, str]] = field(default_factory=list)


def parse_head(line: str) -> tuple[str, str | None, str | None, str | None]:
    """``Nickname (Species) (M) @ Item`` -> species, nickname, gender, item."""
    head, item = line, None
    if " @ " in head:
        head, item = head.split(" @ ", 1)
        item = item.strip() or None
    head = head.strip()
    gender = None
    marked = re.search(r"\s*\(([MF])\)$", head)
    if marked:
        gender = marked.group(1)
        head = head[: marked.start()].strip()
    nickname = None
    named = re.fullmatch(r"(.*\S)\s+\(([^()]+)\)", head)
    if named:
        nickname, head = named.group(1), named.group(2).strip()
    return head, nickname, gender, item


def _read_sp(spec: str, mon: PasteMon) -> None:
    for part in spec.split("/"):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s+(\w+)", part)
        if match is None or match.group(2) not in STAT_LABELS:
            mon.problems.append(("sp", f"cannot read {part!r}"))
            continue
        stat = STAT_LABELS[match.group(2)]
        if stat in mon.sp:
            mon.problems.append(("sp", f"{match.group(2)} written twice"))
        mon.sp[stat] = int(match.group(1))


def parse_paste(text: str) -> list[PasteMon]:
    """The Pokemon of one paste, in the paste's order. Line endings do not matter."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    mons: list[PasteMon] = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        species, nickname, gender, item = parse_head(lines[0])
        mon = PasteMon(species=species, nickname=nickname, gender=gender, item=item)
        for ln in lines[1:]:
            if ln.startswith("- "):
                mon.moves.append(ln[2:].strip())
            elif ln.startswith("Ability:"):
                mon.ability = ln.split(":", 1)[1].strip() or None
            elif ln.startswith("EVs:"):
                _read_sp(ln.split(":", 1)[1], mon)
            elif ln.startswith("Level:"):
                value = ln.split(":", 1)[1].strip()
                if value.isdigit():
                    mon.level = int(value)
                else:
                    mon.problems.append(("line", f"cannot read {ln!r}"))
            elif re.fullmatch(r"[A-Za-z]+ Nature", ln):
                mon.nature = ln.split(" ", 1)[0]
            elif ln.startswith(IGNORED_PREFIXES):
                mon.ignored.append(ln)
            else:
                mon.problems.append(("line", f"cannot read {ln!r}"))
        mons.append(mon)
    return mons


def base_of_mega(reg: Regulation) -> dict[tuple[str, str], str]:
    """(mega forme id, stone id) -> base species id: the regulation's mega map, backwards."""
    return {(mega, stone): base for (base, stone), mega in reg.mega_by_species.items()}


def normalise(reg: Regulation, mon: PasteMon) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """One member in the roster's schema, plus what is wrong with it as (field, reason).

    The member dict uses ``teams.load_roster``'s field names and the dump's spellings, so a
    reader of recorded teams reads these too. ``writtenAs`` keeps the paste's own species.
    """
    problems = list(mon.problems)
    written_id = to_id(mon.species)
    item_id = to_id(mon.item) if mon.item else None
    species = reg.species.get(written_id)
    if species is not None and species.is_mega:
        base = base_of_mega(reg).get((written_id, item_id or ""))
        species = reg.species.get(base) if base is not None else None
        if species is None:
            problems.append(("species", f"{mon.species} is not what {mon.item or 'no item'} makes"))
    elif species is None:
        problems.append(("species", f"{mon.species!r} is not in {reg.meta.format_id}"))
    if species is not None and not species.team_legal:
        problems.append(("species", f"{species.name} is not team-legal in {reg.meta.format_id}"))
        species = None

    if mon.ability is None:
        problems.append(("ability", "no 'Ability:' line"))
    elif species is not None and to_id(mon.ability) not in {to_id(a) for a in species.abilities}:
        problems.append(
            (
                "ability",
                f"{mon.ability!r} is not an ability of {species.name} (a sheet shows the pre-mega one)",
            )
        )
    if item_id is not None and item_id not in reg.items:
        problems.append(("item", f"{mon.item!r} is not legal"))
    if mon.nature is None:
        problems.append(("nature", "no '<Nature> Nature' line"))
    elif mon.nature not in reg.natures:
        problems.append(("nature", f"{mon.nature!r} is unknown"))
    move_ids = [to_id(m) for m in mon.moves]
    if not 1 <= len(move_ids) <= reg.meta.max_move_count:
        problems.append(("moves", f"{len(move_ids)} moves"))
    if len(set(move_ids)) != len(move_ids):
        problems.append(("moves", "a move written twice"))
    problems += [("moves", f"{m!r} is not legal") for m in mon.moves if to_id(m) not in reg.moves]
    total = sum(mon.sp.values())
    if total > reg.meta.sp_limit:
        problems.append(("sp", f"total {total} over {reg.meta.sp_limit}"))
    over = [s for s, v in mon.sp.items() if v > reg.meta.sp_per_stat_max]
    if over:
        problems.append(("sp", f"{', '.join(over)} over {reg.meta.sp_per_stat_max}"))

    notes = list(mon.ignored)
    if mon.level is not None and mon.level != reg.meta.level:
        notes.append(f"Level: {mon.level}（形式は {reg.meta.level} 固定なので使わない）")
    if mon.nickname:
        notes.append(f"nickname {mon.nickname!r}")

    member: dict[str, Any] = {
        "species": species.name if species is not None else mon.species,
        "writtenAs": mon.species,
        "ability": _name(reg.abilities, mon.ability),
        "item": _name(reg.items, mon.item),
        "nature": mon.nature,
    }
    if mon.gender:
        member["gender"] = mon.gender
    member["moves"] = [_name(reg.moves, m) for m in mon.moves]
    member["sp"] = {stat: mon.sp[stat] for stat in STAT_LABELS.values() if mon.sp.get(stat)}
    if notes:
        member["notes"] = notes
    return member, problems


def _name(table: dict[str, Any], written: str | None) -> str | None:
    """The dump's spelling when it knows the thing, the paste's otherwise."""
    if written is None:
        return None
    entry = table.get(to_id(written))
    return entry.name if entry is not None else written


def check_team(reg: Regulation, members: list[dict[str, Any]]) -> list[str]:
    """What the roster loader refuses at team level: size and the two clauses."""
    problems: list[str] = []
    if len(members) != reg.meta.team_size:
        problems.append(f"{len(members)} Pokemon, this regulation brings {reg.meta.team_size}")
    bases = [
        reg.species[to_id(m["species"])].base_species for m in members if to_id(m["species"]) in reg.species
    ]
    doubled = sorted({b for b in bases if bases.count(b) > 1})
    if doubled:
        problems.append(f"Species Clause: {', '.join(doubled)}")
    items = [to_id(m["item"]) for m in members if m["item"]]
    if reg.meta.item_clause is not None and len(set(items)) != len(items):
        twice = sorted({i for i in items if items.count(i) > 1})
        problems.append(f"Item Clause: {', '.join(twice)}")
    return problems


def build_team(
    reg: Regulation,
    text: str,
    supplements: tuple[Supplement, ...] = (),
    standings_dir: Path | None = None,
    supplied: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A paste's members, and its problems as ``{member, writtenAs, field, reason}``.

    ``supplements`` are this paste's transcriptions; each is checked, applied to the empty
    field, and appended as its record to ``supplied``.
    """
    members: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    mons = parse_paste(text)
    for supplement in supplements:
        if not 0 <= supplement.member < len(mons):
            raise SystemExit(f"supplement {supplement.paste_id}: no member {supplement.member}")
    for index, mon in enumerate(mons):
        filled: list[str] = []
        for supplement in (s for s in supplements if s.member == index):
            checked = check_supplement(reg, supplement, mon, standings_dir)
            setattr(mon, supplement.field, supplement.value)
            filled.append(supplement.field)
            if supplied is not None:
                supplied.append(supplement.record(checked))
        member, bad = normalise(reg, mon)
        if filled:
            member["supplied"] = filled
        members.append(member)
        problems += [{"member": index, "writtenAs": mon.species, "field": f, "reason": r} for f, r in bad]
    problems += [
        {"member": None, "writtenAs": None, "field": "team", "reason": r} for r in check_team(reg, members)
    ]
    return members, problems


# -- fetching and the cache ---------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def curl_get(url: str, dest: Path) -> dict[str, Any]:
    """One GET with curl (verification on), written atomically. Returns the log record."""
    curl = shutil.which("curl")
    if curl is None:
        raise SystemExit("curl is not on PATH; this importer does not fall back to unverified TLS")
    part = dest.with_name(dest.name + ".part")
    done = subprocess.run(  # noqa: S603
        [curl, "-sS", "--fail", "-L", "--max-time", "60", "-A", USER_AGENT, "-o", str(part), url],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        part.unlink(missing_ok=True)
        raise SystemExit(f"{url}: curl failed: {done.stderr.decode(errors='replace').strip()}")
    part.replace(dest)
    data = dest.read_bytes()
    return {
        "url": url,
        "fetchedAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes": len(data),
        "sha256": sha256(data),
    }


class Cache:
    """``sheet.csv``, ``pastes/<id>.txt`` and the log that says when each was taken."""

    def __init__(self, root: Path, *, refetch: bool, delay: float) -> None:
        self.root = root
        self.refetch = refetch
        self.delay = delay
        self.log_path = root / "fetch-log.json"
        self.log: dict[str, dict[str, Any]] = (
            json.loads(self.log_path.read_bytes()) if self.log_path.exists() else {}
        )
        self.fetched = 0
        self._last = 0.0

    def get(self, key: str, url: str) -> tuple[bytes, dict[str, Any]]:
        path = self.root / key
        if self.refetch or not path.exists() or key not in self.log:
            path.parent.mkdir(parents=True, exist_ok=True)
            wait = self.delay - (time.monotonic() - self._last)
            if self.fetched and wait > 0:
                time.sleep(wait)
            print(f"  fetching {url}")
            self.log[key] = curl_get(url, path)
            self._last = time.monotonic()
            self.fetched += 1
            self._write_log()
        data = path.read_bytes()
        record = self.log[key]
        if sha256(data) != record["sha256"]:
            raise SystemExit(
                f"{path}: the cached bytes are not the ones fetched at {record['fetchedAt']}; "
                "rerun with --refetch rather than publishing an edited cache"
            )
        return data, record

    def _write_log(self) -> None:
        body = json.dumps(self.log, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self.log_path.write_bytes(body.encode("utf-8"))


# -- the pool file ------------------------------------------------------------------------


def build_pool(
    reg: Regulation,
    sheet: tuple[bytes, dict[str, Any]],
    pastes: list[tuple[SheetEntry, bytes, dict[str, Any]]],
    own_rows: list[SheetEntry] | None = None,
    supplements: tuple[Supplement, ...] = SUPPLEMENTS,
    standings_dir: Path | None = None,
) -> dict[str, Any]:
    """The output document. A pure function of its inputs, so the cache reproduces it.

    ``standings_dir`` is where the supplements' standings files are checked; each record
    says whether its file was read (``checkedAgainst``) or absent (``null``).
    """
    dump_path = regulation_dir() / f"{reg.meta.format_id}.json"
    dump_bytes = dump_path.read_bytes()
    dump_meta = json.loads(dump_bytes)["meta"]
    sheet_bytes, sheet_record = sheet

    def written_as_mega(member: dict[str, Any]) -> bool:
        species = reg.species.get(to_id(member["writtenAs"]))
        return species is not None and species.is_mega

    teams: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for entry, raw, record in pastes:
        text = raw.decode("utf-8")
        supplied: list[dict[str, Any]] = []
        mine = tuple(s for s in supplements if s.paste_id == entry.paste_id)
        members, problems = build_team(reg, text, mine, standings_dir, supplied)
        source = {
            "sheet": SHEET_URL,
            "gid": SHEET_GID,
            "sheetRow": entry.row,
            "paste": f"https://pokepast.es/{entry.paste_id}",
            "pasteId": entry.paste_id,
            "fetchedAt": record["fetchedAt"],
            "sha256": record["sha256"],
        }
        team: dict[str, Any] = {"id": entry.paste_id, "name": entry.name, "source": source}
        if supplied:
            team["supplied"] = supplied
        team["team"] = members
        if problems:
            excluded.append({**team, "problems": problems})
        else:
            teams.append(team)

    listed = [m for t in teams + excluded for m in t["team"]]
    kept = [m for t in teams for m in t["team"]]
    totals = [sum(m["sp"].values()) for m in kept]
    return {
        "id": POOL_ID,
        "name": "Match Up Web シートの Reg M-C 構築（pokepaste、SP つき）",
        "regulation": reg.meta.format_id,
        "character": CHARACTER,
        "role": ROLE,
        "notes": NOTES,
        "source": {
            "sheet": SHEET_URL,
            "gid": SHEET_GID,
            "export": SHEET_EXPORT,
            "fetchedAt": sheet_record["fetchedAt"],
            "sha256": sha256(sheet_bytes),
            "tool": "tools/fetch_pastes.py",
        },
        "validatedAgainst": {
            "formatId": dump_meta["formatId"],
            "showdownCommit": dump_meta["showdownCommit"],
            "generatedAt": dump_meta["generatedAt"],
            "sha256": sha256(dump_bytes),
        },
        "counts": {
            "teams": {"listed": len(pastes), "kept": len(teams), "excluded": len(excluded)},
            "members": {"listed": len(listed), "kept": len(kept)},
            "writtenAsMega": {
                "listed": sum(map(written_as_mega, listed)),
                "kept": sum(map(written_as_mega, kept)),
            },
            "spTotalKept": [min(totals), max(totals)] if totals else None,
            "suppliedFields": sum(len(t.get("supplied", [])) for t in teams + excluded),
        },
        "teams": teams,
        "excluded": excluded,
        #: Own-side rows that carried a paste link: never read as opponents.
        "ownRowsSkipped": [
            {"sheetRow": e.row, "name": e.name, "paste": f"https://pokepast.es/{e.paste_id}"}
            for e in own_rows or []
        ],
    }


def encode(pool: dict[str, Any]) -> bytes:
    """UTF-8, LF, a trailing newline: the bytes the reproducibility check compares."""
    return (json.dumps(pool, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refetch", action="store_true", help="take the sheet and every paste again")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "pool" / POOL_ID)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pool" / f"{POOL_ID}.json")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between network requests")
    ap.add_argument(
        "--standings",
        type=Path,
        default=ROOT / "data" / "standings",
        help="where the supplements' standings files are read to check them (never fetched here)",
    )
    args = ap.parse_args()

    reg = load_regulation(REGULATION)
    cache = Cache(args.cache, refetch=args.refetch, delay=args.delay)
    sheet = cache.get("sheet.csv", SHEET_EXPORT)
    entries, own_rows = parse_sheet(sheet[0].decode("utf-8"))
    pastes = []
    for entry in entries:
        raw, record = cache.get(f"pastes/{entry.paste_id}.txt", f"https://pokepast.es/{entry.paste_id}/raw")
        pastes.append((entry, raw, record))

    pool = build_pool(reg, sheet, pastes, own_rows, SUPPLEMENTS, args.standings)
    body = encode(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)

    counts = pool["counts"]
    teams, members, mega = counts["teams"], counts["members"], counts["writtenAsMega"]
    print(
        f"sheet {SHEET_URL} gid {SHEET_GID}: {teams['listed']} teams listed; "
        f"{cache.fetched} fetched now, {len(entries) + 1 - cache.fetched} from the cache"
    )
    print(
        f"teams kept {teams['kept']}, excluded {teams['excluded']}; members {members['listed']} "
        f"listed, {members['kept']} kept; written as the mega forme {mega['listed']} listed, "
        f"{mega['kept']} kept; SP totals of the kept {counts['spTotalKept']}"
    )
    for team in pool["teams"] + pool["excluded"]:
        for s in team.get("supplied", []):
            src = s["source"]
            checked = src["checkedAgainst"]
            print(
                f"  supplied {team['name']!r} {s['writtenAs']} {s['field']} = {s['value']!r} from "
                f"{src['event']} {src['player']} (#{src['place']}) {src['standings']} {src['pointer']}; "
                + (f"checked, file sha256 {checked['sha256']}" if checked else "file absent, NOT checked")
            )
    listed_ids = {e.paste_id for e in entries}
    for s in SUPPLEMENTS:
        if s.paste_id not in listed_ids:
            print(f"  supplement for paste {s.paste_id} matches no listed team: unused")
    for team in pool["excluded"]:
        reasons = "; ".join(
            f"{p['writtenAs'] or 'team'} {p['field']}: {p['reason']}" for p in team["problems"]
        )
        print(f"  excluded {team['name']!r} ({team['source']['paste']}): {reasons}")
    noted = [
        (t["name"], m["writtenAs"], n) for t in pool["teams"] for m in t["team"] for n in m.get("notes", [])
    ]
    for name, species, note in noted:
        print(f"  note {name!r} {species}: {note}")
    for row in pool["ownRowsSkipped"]:
        print(
            f"  own-side row {row['sheetRow']} {row['name']!r} has {row['paste']}: not an opponent, skipped"
        )
    print(f"-> {args.out} ({len(body):,} bytes, sha256 {sha256(body)})")


if __name__ == "__main__":
    main()
