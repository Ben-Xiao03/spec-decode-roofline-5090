"""Consistency checks for the committed result summaries and K tables."""

from pathlib import Path

import pandas as pd
import pytest

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


@pytest.fixture(params=GRID_CSVS, ids=lambda p: p.name)
def grid(request):
    return pd.read_csv(request.param)


def test_grid_has_required_columns(grid):
    """Each E1/E2 summary has the required columns and is non-empty."""
    missing = [c for c in REQUIRED_COLUMNS if c not in grid.columns]
    assert not missing, f"missing columns: {missing}"
    assert len(grid) > 0


def test_grid_has_at_least_three_reps(grid):
    """Every grid point has at least 3 reps."""
    assert (grid["reps"] >= 3).all(), grid.loc[grid["reps"] < 3, KEY + ["reps"]]