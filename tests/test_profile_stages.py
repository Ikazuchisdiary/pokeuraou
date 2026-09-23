"""The measuring tool against the run it measures (IKA-98).

Two ways `tools/profile_stages.py` had fallen behind generation, each of which runs to the
end and prints a table: it passed defaults of its own (width 24, 8 workers) where shipping
ran 12 and 24, and it had no way to pass a book or a seed. And a worktree run whose
workers imported the main checkout's src would have timed the wrong code without a word.
These pin the command it builds and the check it makes afterwards, with a control that
breaks each named shape.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "profile_stages.py"


@pytest.fixture(scope="module")
def tool() -> Any:
    spec = importlib.util.spec_from_file_location("_profile_stages_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace(**given: Any) -> argparse.Namespace:
    base = {
        "workload": "generation", "out": Path("out"), "games": 24, "first_game": None,
        "workers": None, "seed": None, "roster": None, "value": "v.pt", "limit": None,
        "selection_book": None, "uniform_selection": False, "force_lead": None,
        "device": "cuda", "no_bridge": False, "served": False, "servers": 2,
        "hide_bench": False, "rest": [],
    }
    base.update(given)
    return argparse.Namespace(**base)


def test_an_unset_option_is_left_to_the_driver(tool: Any) -> None:
    command = tool.build(namespace())
    for flag in ("--limit", "--workers", "--seed", "--selection-book", "--first-game",
                 "--roster", "--force-lead", "--uniform-selection"):
        assert flag not in command


def test_the_shipping_command_line_goes_through(tool: Any) -> None:
    command = tool.build(namespace(
        games=600, seed=6601, served=True, servers=2, workers=24, limit=12,
        selection_book=Path("book.jsonl.gz"), hide_bench=True, rest=["--rank-leaf"],
    ))
    joined = " ".join(command)
    for part in ("--games 600", "--seed 6601", "--served --servers 2", "--workers 24",
                 "--limit 12", "--selection-book book.jsonl.gz", "--hide-bench"):
        assert part in joined
    # The tail is the workers', after generate_queue's own `--`.
    assert command[-2:] == ["--", "--rank-leaf"]


def test_process_role_reads_the_command_line(tool: Any) -> None:
    assert tool.process_role("python.exe", ["python", "C:\\x\\tools\\inference_server.py"]) \
        == "inference_server.py"
    assert tool.process_role("python.exe", ["python", "/x/tools/selfplay.py", "--queue"]) \
        == "selfplay.py"
    assert tool.process_role("pokeuraou-damage.exe", []) == "pokeuraou-damage.exe"
    assert tool.process_role("python.exe", ["python", "-c", "pass"]) == "python.exe"


def worker(tool: Any, **overrides: Any) -> dict[str, Any]:
    argv = ["tools/selfplay.py", "--queue", "q", "--seed", "6601", "--limit", "12",
            "--roster", "r", "--hide-bench", "--out", "missing.jsonl",
            "--selection-book", "book.jsonl.gz", "--rank-leaf"]
    report = {
        "pid": 1, "argv": argv, "source": str(tool.ROOT / "src" / "pokeuraou"),
        "elapsed": 10.0, "stages": {"lp": {"wall": 4.0, "cpu": 4.0, "calls": 10}},
        "decisions": {"move": {"n": 100, "wall": 9.0, "stages": {}, "counts": {}}},
    }
    report.update(overrides)
    return report


def test_delivery_passes_the_run_that_was_asked_for(tool: Any) -> None:
    args = namespace(seed=6601, limit=12, selection_book=Path("book.jsonl.gz"),
                     hide_bench=True, rest=["--rank-leaf"])
    found = tool.delivery([worker(tool), worker(tool)], args)
    assert found["problems"] == []
    assert found["seen"]["--limit"] == {"12": 2}


@pytest.mark.parametrize(
    ("change", "said"),
    [
        # Each control breaks one of the shapes the check is for.
        ({"limit": 24}, "--limit"),
        ({"seed": 1}, "--seed"),
        ({"selection_book": Path("other.jsonl.gz")}, "--selection-book"),
        ({"hide_bench": False}, "--hide-bench"),
        ({"rest": ["--rank-leaf", "--mirror-share", "0.1"]}, "from the tail"),
    ],
)
def test_delivery_names_a_setting_that_did_not_arrive(
    tool: Any, change: dict[str, Any], said: str
) -> None:
    asked = {"seed": 6601, "limit": 12, "selection_book": Path("book.jsonl.gz"),
             "hide_bench": True, "rest": ["--rank-leaf"]}
    asked.update(change)
    found = tool.delivery([worker(tool)], namespace(**asked))
    assert any(said in problem for problem in found["problems"])


def test_delivery_names_the_wrong_checkout(tool: Any) -> None:
    elsewhere = worker(tool, source="C:/elsewhere/src/pokeuraou")
    found = tool.delivery([elsewhere], namespace(hide_bench=True))
    assert any("elsewhere" in problem for problem in found["problems"])
    # A report from before `source` existed is not a mismatch; it says it was not recorded.
    old = worker(tool)
    del old["source"]
    found = tool.delivery([old], namespace(hide_bench=True))
    assert found["problems"] == []
    assert found["sources"] == {"(not recorded)": 1}


def test_rest_fit_separates_the_fixed_part_from_the_per_decision_part(tool: Any) -> None:
    """rest = 2 s + 5 ms x decisions, exactly, across three workers."""
    reports = []
    for decisions in (100, 200, 400):
        rest = 2.0 + 0.005 * decisions
        reports.append(worker(
            tool,
            elapsed=10.0 + rest,
            stages={"lp": {"wall": 10.0, "cpu": 10.0, "calls": 1},
                    "rust.fill@rank": {"wall": 99.0, "cpu": 0.0, "calls": 1}},
            decisions={"move": {"n": decisions, "wall": 1.0, "stages": {}, "counts": {}},
                       "between": {"n": 7, "wall": 1.0, "stages": {}, "counts": {}}},
        ))
    found = tool.rest_fit(reports)
    assert found["decisions"] == 700  # `between` is not a decision
    assert found["fit"]["intercept"] == pytest.approx(2.0)
    assert found["fit"]["slope"] == pytest.approx(0.005)
    assert found["fit"]["steady"] == pytest.approx(0.005 * 700 / found["elapsed"])


def test_rest_fit_falls_back_to_the_games_file(tool: Any, tmp_path: Path) -> None:
    """A report from before IKA-98 has no decisions; the games file it names has them."""
    reports = []
    for index, decisions in enumerate((3, 5)):
        games = tmp_path / f"games-worker{index}.jsonl"
        games.write_text(json.dumps({"decisions": [{}] * decisions}) + "\n", encoding="utf-8")
        report = worker(tool)
        del report["decisions"]
        report["argv"] = ["tools/selfplay.py", "--out", str(games)]
        reports.append(report)
    found = tool.rest_fit(reports)
    assert found["decisions"] == 8
    assert found["timed"] is None


def test_the_per_decision_table_sums_workers(tool: Any) -> None:
    one = {"move.hidden": {"n": 2, "wall": 1.0, "stages": {"lp": [0.1, 4]},
                           "counts": {"forward.passes": 20}}}
    found = tool.per_decision([worker(tool, decisions=one), worker(tool, decisions=one)])
    assert found["move.hidden"]["n"] == 4
    assert found["move.hidden"]["stages"]["lp"] == [pytest.approx(0.2), 8]
    assert found["move.hidden"]["counts"]["forward.passes"] == 40


def test_spin_is_server_cpu_over_held(tool: Any) -> None:
    tree = {"by_role": {"inference_server.py": 30.0, "selfplay.py": 100.0}}
    servers = {"stages": {"server.held": {"wall": 40.0, "cpu": 0.0, "calls": 9}}}
    assert tool.spin(tree, servers)["ratio"] == pytest.approx(0.75)
    assert tool.spin({"by_name": {}}, servers) is None
    # A server's CPU at `timing.ready()` is its start, not its waiting.
    started = [{"argv": ["tools/inference_server.py"], "startup_process_cpu": 10.0},
               {"argv": ["tools/selfplay.py"], "startup_process_cpu": 99.0}]
    assert tool.spin(tree, servers, started)["ratio"] == pytest.approx(0.5)


def lumpy_workers(tool: Any) -> list[dict[str, Any]]:
    """The shape of IKA-98's shipping measurement, in miniature (IKA-149).

    Every worker lives 100 s, as the queue makes them. Each spends a lump of untimed
    seconds in its self-switch decisions and the rest of its clock deciding moves at
    0.2 s a decision, all of it on a row. So the more lump, the fewer decisions: the rest
    falls as decisions grow, and no stage is counted twice.
    """
    reports = []
    for lump in (2.0, 10.0, 20.0, 40.0):
        moves = int((100.0 - 1.0 - lump) / 0.2)
        timed = 0.2 * moves
        reports.append(worker(
            tool,
            elapsed=100.0,
            stages={"startup": {"wall": 1.0, "cpu": 1.0, "calls": 1},
                    "branch": {"wall": timed, "cpu": timed, "calls": moves},
                    "rust.fill@rank": {"wall": 50.0, "cpu": 0.0, "calls": 1}},
            decisions={
                "move.hidden": {"n": moves, "wall": timed, "stages": {"branch": [timed, moves]},
                                "counts": {}},
                "selfswitch": {"n": 10, "wall": lump,
                               "stages": {"rust.fill@rank": [50.0, 1]}, "counts": {}},
                "between": {"n": 5, "wall": 99.0 - timed - lump, "stages": {}, "counts": {}},
            },
        ))
    return reports


def test_a_falling_rest_warns_and_is_not_called_a_double_count(
    tool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    found = tool.rest_fit(lumpy_workers(tool))
    assert found["fit"]["steady"] < -tool.UNNAMED_LIMIT
    tool.print_rest_fit(found)
    printed = capsys.readouterr().out
    assert "below -5%" in printed
    assert "counted twice" not in printed
    assert found["negative"] == 0 and found["least_rest"] > 0


def test_a_rising_rest_still_warns_and_a_flat_one_does_not(
    tool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    def fitted(per_decision: float) -> dict[str, Any]:
        reports = []
        for decisions in (100, 200, 400):
            rest = 1.0 + per_decision * decisions
            reports.append(worker(
                tool, elapsed=40.0 + rest,
                stages={"lp": {"wall": 40.0, "cpu": 40.0, "calls": 1}},
                decisions={"move": {"n": decisions, "wall": 1.0, "stages": {}, "counts": {}}},
            ))
        return tool.rest_fit(reports)

    tool.print_rest_fit(fitted(0.01))
    assert "over 5%" in capsys.readouterr().out
    tool.print_rest_fit(fitted(0.0001))
    assert "**" not in capsys.readouterr().out


def test_a_negative_rest_is_named_as_a_double_count(
    tool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The control that breaks the named shape: rows adding to more than the clock."""
    reports = []
    for decisions in (100, 200):
        reports.append(worker(
            tool, elapsed=10.0,
            stages={"lp": {"wall": 8.0, "cpu": 8.0, "calls": 1},
                    "belief": {"wall": 3.0, "cpu": 3.0, "calls": 1}},
            decisions={"move": {"n": decisions, "wall": 9.0, "stages": {}, "counts": {}}},
        ))
    found = tool.rest_fit(reports)
    tool.print_rest_fit(found)
    assert "counted twice" in capsys.readouterr().out
    assert found["negative"] == 2
    assert found["least_rest"] == pytest.approx(-1.0)


def test_the_rest_by_kind_names_where_the_lump_is(tool: Any) -> None:
    found = tool.rest_by_kind(lumpy_workers(tool))
    kinds = found["kinds"]
    # All of each worker's lump, and nothing of the moves, whose clock is all on a row.
    assert kinds["selfswitch"]["rest"] == pytest.approx(2.0 + 10.0 + 20.0 + 40.0)
    assert kinds["move.hidden"]["rest"] == pytest.approx(0.0)
    assert kinds["selfswitch"]["n"] == 40
    # Borrowed rows charged in a stretch are not taken off its rest.
    assert found["rest"] == pytest.approx(sum(row["rest"] for row in kinds.values()))
    assert found["outside"] == pytest.approx(0.0, abs=1e-9)
    # A report from before IKA-98 has no stretches to split.
    old = worker(tool)
    del old["decisions"]
    assert tool.rest_by_kind([old]) is None
