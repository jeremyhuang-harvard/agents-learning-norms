"""The saved data must reproduce the numbers we report."""

import importlib.util
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_results", ROOT / "analysis" / "make_results.py")
assert spec is not None and spec.loader is not None
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)


def by_condition(rows, key="arena"):
    return {r[key]: r for r in rows}


def test_headline_memory_table():
    rows = mr.table_memory()
    baseline = rows[0]
    assert (baseline["n"], baseline["mean_seller_share"], baseline["focal_share"]) == (24, 0.409, 0.083)
    arenas = by_condition(rows[1:])
    assert {a: arenas[a]["focal_share"] for a in mr.ARENA_ORDER} == {
        "10/0": 0.192, "10/0f": 0.283, "5/5": 0.25, "0/10": 0.294, "0/10w": 0.175}


def test_playbook_keeps_performance_but_not_ordering():
    rows = mr.table_arms()
    pooled = next(r for r in rows if r["condition"] == "pooled: original four arenas")
    assert (pooled["A_mean"], pooled["B_mean"]) == (0.468, 0.478)
    arenas = by_condition(rows[:5])
    assert round(arenas["0/10w"]["B_focal"], 2) == 0.32


def test_memory_effect_vanishes_without_any_range():
    blind = [r for r in mr.table_range() if r["cell"].startswith("range removed from memory")]
    assert len(blind) == 4
    assert {round(r["focal_share"], 2) for r in blind} == {0.10}


def test_committed_results_match_a_fresh_build():
    memory, arms, rng = mr.table_memory(), mr.table_arms(), mr.table_range()
    fresh = mr.markdown(memory, arms, rng, mr.table_interior(), mr.table_models(), mr.table_bank())
    assert fresh == (ROOT / "results" / "RESULTS.md").read_text()
    assert mr.svg_focal(memory) == (ROOT / "results" / "focal_by_memory.svg").read_text()


def test_no_credentials_or_machine_paths_in_tracked_files():
    pattern = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}|/Users/[A-Za-z]|/home/[a-z]+/|BEGIN [A-Z ]*PRIVATE KEY")
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    for name in tracked:
        path = ROOT / name
        if path.suffix == ".png" or path.name == "test_results.py":
            continue
        assert not pattern.search(path.read_text(errors="ignore")), name
