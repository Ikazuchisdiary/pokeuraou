"""What does the solved book believe about the counters a player expects to exist?

The equilibrium brings Charizard + Garchomp against almost the whole field, and there is a
mechanical reason to expect that: Charizard is Flying, so Garchomp's Earthquake does not
touch it, and the pair fires two spread moves a turn -- sun-boosted Heat Wave and
Earthquake or Rock Slide -- at no cost to itself. The hypothesis in
``configs/knowledge/*-counters.json`` is that many compositions answer exactly that, and
that a value function which has not learned the answers yet would over-rate the pair.

This tool measures the *model's* side of that, which needs no games and is therefore worth
doing first. Three questions, and the third is the sharp one:

- **does the model discount teams that carry a cited answer?** Group the 394 equilibrium
  values by whether a team matches, and compare. A discount means the model has noticed
  something; no discount is consistent with the hypothesis and proves nothing on its own.
- **does the model stop bringing the pair against them?** Our equilibrium selection is the
  model's own recommendation, so the share of it that is still Charizard + Garchomp says
  whether the answer changes its advice.
- **does the model's opponent bring the answer?** ``solve_bayesian`` returns the
  opponent's equilibrium selection per spread class. If the model believed Aerodactyl +
  Floette beat this pair, its own opponent model would *select* Aerodactyl and Floette when
  the sheet has them. This is an internal-consistency test with no games in it and no
  appeal to the model's calibration: it asks whether the belief the values imply is the
  belief the strategies act on.

Nothing here can confirm that a counter works. The book is the model's opinion, and the
question "is the model right" is only answerable by play -- ``tools/book_check.py`` and
``tools/cycle_match.py`` are that shape. What this tool can do is separate "the model has
noticed and disagrees" from "the model has never seen it", which are different problems
with different fixes.

    uv run --group learn python tools/counterplay.py --text /c/tmp/counters.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.names import localiser
from pokeuraou.regulation import to_id
from pokeuraou.selection_book import SelectionBook, key_for_team, selection_dir
from pokeuraou.standings import (
    TournamentTeam,
    find_cached_standings,
    load_standings,
)
from pokeuraou.teams import all_selections, load_roster


def knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "knowledge"


def matches(team: TournamentTeam, hypothesis: dict) -> bool:
    """Whether a team sheet carries what a hypothesis names.

    Species and moves are both read off the sheet, which is public -- so a grouping built
    from them is a grouping a player could make before selecting, not hindsight.
    """
    species = {to_id(m.species) for m in team.members}
    moves = {to_id(x) for m in team.members for x in m.moves}
    wanted_species = {to_id(s) for s in hypothesis.get("species", [])}
    wanted_moves = {to_id(x) for x in hypothesis.get("moves", [])}
    if wanted_species and not wanted_species <= species:
        return False
    if wanted_moves and not wanted_moves <= moves:
        return False
    return bool(wanted_species or wanted_moves)


def brought_share(
    book: SelectionBook, team: TournamentTeam, ids: set[str], selections: list[tuple[int, ...]]
) -> float | None:
    """How much of the opponent's equilibrium mass brings *all* of ``ids``.

    Averaged over spread classes with the class weights, because the opponent picks per
    class -- they know their own investment -- and the marginal is what we would observe.
    """
    entry = book.get(team)
    if entry is None:
        return None
    sheet = [to_id(m.species) for m in team.members]
    if not ids <= set(sheet):
        return None
    wanted = {i for i, species in enumerate(sheet) if species in ids}
    weights = np.asarray(entry.class_weights, dtype=np.float64)
    weights = weights / weights.sum()
    total = 0.0
    for weight, strategy in zip(weights, entry.their_strategies, strict=True):
        strategy = np.asarray(strategy, dtype=np.float64)
        strategy = strategy / strategy.sum()
        mass = sum(
            float(strategy[j])
            for j, selection in enumerate(selections)
            if wanted <= set(selection)
        )
        total += float(weight) * mass
    return total


def describe(values: np.ndarray) -> str:
    if values.size == 0:
        return "該当なし"
    if values.size == 1:
        return f"{values[0] * 100:.1f}%（1 チーム）"
    half = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    return f"{values.mean() * 100:.1f}% ±{half * 100:.1f}（{values.size} チーム）"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", default="value-gen2")
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--knowledge", type=Path, default=None)
    ap.add_argument("--text", type=Path, default=None)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    book_path = args.book or (selection_dir() / f"{args.roster}-{args.model}.jsonl.gz")
    book = SelectionBook.read(book_path)
    book.require_roster(args.roster)
    standings = load_standings(find_cached_standings(), reg)
    path = args.knowledge or (knowledge_dir() / f"{args.roster}-counters.json")
    knowledge = json.loads(path.read_text(encoding="utf-8"))
    loc = localiser(reg, "ja")
    selections = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    def species_name(species_id: str) -> str:
        return loc.species(species_id) if loc else species_id

    def our_name(index: int) -> str:
        return species_name(roster.sets[index].species)

    covered = [t for t in standings.teams if key_for_team(t) in book.entries]
    values = {t.player: book.entries[key_for_team(t)].value for t in covered}
    top_selection = {
        t.player: selections[int(np.argmax(book.entries[key_for_team(t)].our_strategy))]
        for t in covered
    }
    # The pair the hypothesis is about, as party indices, so "did the advice change" is a
    # question about this specific pair rather than about any change at all.
    pair = {
        i
        for i, entry in enumerate(roster.sets)
        if to_id(entry.species) in {"charizard", "garchomp"}
    }

    out: list[str] = []
    out.append(f"■ 仮説の検証（モデル側のみ）  本: {book_path.name}、{len(covered)} チーム")
    out.append(f"  前提: {knowledge.get('premise', '')}")
    out.append(f"  範囲: {knowledge.get('scope', '')}")
    everything = np.array([values[t.player] for t in covered])
    out.append(f"  全体の均衡値 {describe(everything)}")
    leads = sum(
        1 for t in covered if pair <= set(top_selection[t.player][:2])
    )
    out.append(
        f"  リザードン+ガブリアス先発を最上位に置いたチーム: {leads}/{len(covered)}"
        f"（{leads / len(covered) * 100:.0f}%）"
    )
    out.append("")

    for hypothesis in knowledge.get("hypotheses", []):
        inside = [t for t in covered if matches(t, hypothesis)]
        outside = [t for t in covered if not matches(t, hypothesis)]
        out.append(f"■ {hypothesis['id']}")
        out.append(f"  {hypothesis.get('note', '')}")
        if not inside:
            out.append("  → 該当するチームが環境に無い。この仮説はこの本では検証できない")
            out.append("")
            continue
        a = np.array([values[t.player] for t in inside])
        b = np.array([values[t.player] for t in outside])
        out.append(f"  該当 {describe(a)}")
        out.append(f"  非該当 {describe(b)}")
        gap = (a.mean() - b.mean()) * 100
        # Welch, because the two groups differ wildly in size and there is no reason to
        # assume equal spread.
        se = float(
            np.sqrt(
                a.var(ddof=1) / a.size + b.var(ddof=1) / b.size
                if a.size > 1 and b.size > 1
                else np.nan
            )
        )
        interval = (
            f" [{gap - 1.96 * se * 100:+.1f}, {gap + 1.96 * se * 100:+.1f}]"
            if np.isfinite(se)
            else ""
        )
        out.append(f"  差 {gap:+.1f} ポイント{interval}（負なら「モデルは警戒している」）")
        still = sum(1 for t in inside if pair <= set(top_selection[t.player][:2]))
        out.append(
            f"  それでもリザードン+ガブリアス先発: {still}/{len(inside)}"
            f"（{still / len(inside) * 100:.0f}%）"
        )
        # The internal-consistency test: does the model's own opponent bring the answer?
        ids = {to_id(s) for s in hypothesis.get("species", [])}
        if ids:
            shares = [
                s
                for s in (brought_share(book, t, ids, selections) for t in inside)
                if s is not None
            ]
            if shares:
                arr = np.array(shares)
                # The uniform baseline depends on how many members the hypothesis names:
                # a single species is brought by 60 of the 90 selections, a pair by 36.
                probe = set(range(len(ids)))
                base = sum(
                    1 for s in selections if probe <= set(s)
                ) / len(selections)
                out.append(
                    f"  相手の均衡選出がその{len(ids)}匹を"
                    f"{'揃えて' if len(ids) > 1 else ''}出す割合: {arr.mean() * 100:.1f}%"
                    f"（{arr.size} チーム、一様なら {base * 100:.1f}% 相当）"
                )
                out.append(
                    "  → 低ければ、モデルは値でも戦略でもその対策を評価していない。"
                    "「気づいて否定している」のではなく「見たことがない」側の証拠"
                )
        out.append("")

    order = sorted(covered, key=lambda t: values[t.player])
    out.append(f"■ モデルが最も苦しいと言うチーム（下位 {args.top}）")
    for team in order[: args.top]:
        out.append(
            f"  {values[team.player] * 100:5.1f}%  {team.place:>3}位 {team.player[:20]:<20} "
            f"{' / '.join(species_name(s) for s in team.species)}"
        )
    out.append(f"■ 最も楽だと言うチーム（上位 {args.top}）")
    for team in order[-args.top :][::-1]:
        out.append(
            f"  {values[team.player] * 100:5.1f}%  {team.place:>3}位 {team.player[:20]:<20} "
            f"{' / '.join(species_name(s) for s in team.species)}"
        )
    out.append("")

    # What the model thinks it will face: the opponent's equilibrium marginal over species,
    # pooled across the field, against the base rate of the same species on the sheets.
    faced: dict[str, float] = {}
    present: dict[str, float] = {}
    for team in covered:
        entry = book.entries[key_for_team(team)]
        sheet = [to_id(m.species) for m in team.members]
        weights = np.asarray(entry.class_weights, dtype=np.float64)
        weights = weights / weights.sum()
        for species in set(sheet):
            present[species] = present.get(species, 0.0) + 1.0
        for weight, strategy in zip(weights, entry.their_strategies, strict=True):
            strategy = np.asarray(strategy, dtype=np.float64)
            strategy = strategy / strategy.sum()
            for j, selection in enumerate(strategy):
                if selection <= 0:
                    continue
                for index in selections[j]:
                    species = sheet[index]
                    faced[species] = faced.get(species, 0.0) + float(weight) * float(selection)
    # A species on a sheet is brought with probability 4/6 under a uniform draw, so the
    # ratio against that is what says the model singles it out.
    rows = [
        (species, faced.get(species, 0.0) / (count * 4.0 / 6.0), count)
        for species, count in present.items()
        if count >= 10
    ]
    rows.sort(key=lambda row: -row[1])
    out.append("■ モデルが「出してくる」と考えているポケモン（一様選出比、シート10枚以上）")
    for species, ratio, count in rows[:12]:
        out.append(f"  {ratio:5.2f}x  {species_name(species):<12} シート {int(count)} 枚")
    out.append(
        "  1.00x は一様選出と同じ。1.5x なら「このチームならこれを出す」とモデルが言っている"
    )

    text = "\n".join(out)
    print(text, file=sys.stderr)
    if args.text is not None:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text + "\n", encoding="utf-8")
        print(f"（読める形: {args.text}）", file=sys.stderr)


if __name__ == "__main__":
    main()
