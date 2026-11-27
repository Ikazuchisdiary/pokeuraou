"""IKA-465: `tools/ci_torch_tests.py`, the list of files CI's `learn` job runs with torch.

`learn` no longer runs the whole suite; it runs the test files that name torch, and `suite`
checks that every test it skipped for torch is in one of them. These hold that check to
the shapes it names: a torch skip in a listed file passes, one in an unlisted file fails,
and a report with no torch skip at all fails too (a suite that had torch is not `suite`).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ci_torch_tests  # noqa: E402

REASON = "could not import 'torch': No module named 'torch'"


def _junit(path: Path, cases: list[tuple[str, str, str | None]]) -> Path:
    """(classname, name, skip message or None) per case, as pytest's junit writes them."""
    rows = []
    for classname, name, skip in cases:
        body = f'<skipped type="pytest.skip" message="{skip}" />' if skip else ""
        rows.append(f'<testcase classname="{classname}" name="{name}" time="0.0">{body}</testcase>')
    skipped = sum(1 for *_, s in cases if s)
    xml = (
        f'<testsuites><testsuite name="pytest" tests="{len(cases)}" failures="0" errors="0" '
        f'skipped="{skipped}">{"".join(rows)}</testsuite></testsuites>'
    )
    path.write_bytes(xml.encode("utf-8"))
    return path


def _tree(tmp_path: Path) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_net.py").write_bytes(b'torch = pytest.importorskip("torch")\n')
    (tests / "test_plain.py").write_bytes(b"def test_x():\n    pass\n")
    return tmp_path


def test_the_list_is_the_files_that_name_torch(tmp_path: Path) -> None:
    assert ci_torch_tests.torch_files(_tree(tmp_path)) == ["tests/test_net.py"]


def test_the_module_of_each_skip_shape() -> None:
    assert ci_torch_tests.module_file("tests.test_net") == "tests/test_net.py"
    assert ci_torch_tests.module_file("tests.test_net::test_a[x]") == "tests/test_net.py"
    assert ci_torch_tests.module_file("tests.test_net.TestA::test_a") == "tests/test_net.py"


def test_a_torch_skip_in_a_listed_file_passes(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    junit = _junit(tmp_path / "r.xml", [("tests.test_net", "test_a", REASON),
                                         ("tests.test_plain", "test_x", None)])
    assert ci_torch_tests.check(junit, root) == 0


def test_a_torch_skip_in_a_file_that_does_not_name_torch_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    junit = _junit(tmp_path / "r.xml", [("tests.test_net", "test_a", REASON),
                                         ("tests.test_plain", "test_x", REASON)])
    assert ci_torch_tests.check(junit, root) == 1


def test_a_report_without_a_torch_skip_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    junit = _junit(tmp_path / "r.xml", [("tests.test_net", "test_a", None)])
    assert ci_torch_tests.check(junit, root) == 1
