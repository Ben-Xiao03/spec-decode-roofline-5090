"""Consistency checks for the committed result summaries and K tables."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"

GRID_CSVS = sorted(SUMMARY.glob("e1_*.csv")) + sorted(SUMMARY.glob("e2_*.csv"))
KEY = ["setting", "context", "concurrency"]
REQUIRED_COLUMNS = KEY + ["reps", "mean_tpot_ms"]


def test_grid_csvs_exist():
    """At least one E1 and one E2 summary CSV exist."""
    names = [p.name for p in GRID_CSVS]
    assert any(n.startswith("e1_") for n in names)
    assert any(n.startswith("e2_") for n in names)