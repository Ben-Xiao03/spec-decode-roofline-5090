"""Consistency checks for the committed result summaries and K tables."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
CONFIGS = ROOT / "configs"

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


@pytest.mark.parametrize("metric", ["mean_tpot_ms", "output_throughput"])
def test_grid_mean_inside_ci(grid, metric):
    """Where present, lo <= mean <= hi for mean TPOT and output throughput."""
    lo, hi = f"{metric}_lo", f"{metric}_hi"
    if metric not in grid.columns:
        pytest.skip(f"{metric} not in this summary")
    assert lo in grid.columns and hi in grid.columns
    rows = grid.dropna(subset=[metric, lo, hi])
    tol = 1e-9 * rows[metric].abs().clip(lower=1.0)
    bad = rows[(rows[lo] > rows[metric] + tol) | (rows[metric] > rows[hi] + tol)]
    assert bad.empty, bad[KEY + [lo, metric, hi]]


def test_grid_has_no_duplicate_points(grid):
    """No (setting, context, concurrency) row appears twice."""
    dup = grid[grid.duplicated(KEY, keep=False)]
    assert dup.empty, dup[KEY]


@pytest.fixture(scope="module")
def goodput():
    return json.loads((SUMMARY / "e7_goodput_main.json").read_text())


def test_goodput_slo_matches_config(goodput):
    """The SLO in e7_goodput_main.json matches configs/slo.yaml."""
    cfg = yaml.safe_load((CONFIGS / "slo.yaml").read_text())
    assert goodput["slo"] == {"ttft_ms": cfg["ttft_ms"], "tpot_ms": cfg["tpot_ms"]}