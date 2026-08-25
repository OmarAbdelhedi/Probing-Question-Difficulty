"""Tests for src/probe_generalization.py (Tier 2g): train on the extreme
classes, check that held-out middles interpolate on planted ordinal data."""

import json

import numpy as np
import pytest

from src.probe_generalization import interpolation_by_layer, run as tier2g_run


def test_middle_interpolates_on_planted_ordinal_data(planted_xy):
    X, y = planted_xy([("2hop", 0.0), ("3hop", 3.0), ("4hop", 6.0)],
                      per_class=20, n_layers=2)
    res = interpolation_by_layer(X, y, ["2hop", "3hop", "4hop"], 5,
                                 np.random.default_rng(0))
    assert not res["skipped"]
    assert res["low"] == "2hop" and res["high"] == "4hop" and res["middles"] == ["3hop"]
    assert all(a > 0.9 for a in res["auc_2v4"])          # extremes separate
    assert all(0.25 < p < 0.75 for p in res["interp"]["3hop"])  # ~0.5 by construction


def test_two_middles_keep_their_ordinal_order(planted_xy):
    X, y = planted_xy([("2hop", 0.0), ("3hop", 2.0), ("4hop", 4.0), ("5hop", 6.0)],
                      per_class=20, n_layers=1)
    res = interpolation_by_layer(X, y, ["2hop", "3hop", "4hop", "5hop"], 5,
                                 np.random.default_rng(0))
    p3 = res["interp"]["3hop"][0]
    p4 = res["interp"]["4hop"][0]
    assert 0.0 < p3 < p4 < 1.0  # planted at 1/3 and 2/3 of the axis


def test_needs_three_classes(planted_xy):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=5, n_layers=1)
    assert interpolation_by_layer(X, y, ["easy", "hard"], 5,
                                  np.random.default_rng(0)) is None


def test_tiny_extremes_are_skipped(planted_xy):
    X, y = planted_xy([("2hop", 0.0), ("3hop", 1.5), ("4hop", 3.0)],
                      per_class=4, n_layers=1)
    keep = np.where(y == "2hop")[0][:1].tolist() + np.where(y != "2hop")[0].tolist()
    res = interpolation_by_layer(X[keep], y[keep], ["2hop", "3hop", "4hop"], 5,
                                 np.random.default_rng(0))
    assert res["skipped"] and "too small" in res["reason"]


def test_tier2g_run_end_to_end(planted_xy, make_run, tmp_path):
    X, y = planted_xy([("2hop", 0.0), ("3hop", 3.0), ("4hop", 6.0)],
                      per_class=15, n_layers=2)
    run_dir = make_run("synth2g", X, y)
    figs = tmp_path / "figs"
    tier2g_run(run_dir, plots_dir=str(figs), folds=4, seed=0)

    assert (figs / "tier2g_generalization_by_layer.png").is_file()
    with open(figs / "tier2g_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["low"] == "2hop" and summary["high"] == "4hop"
    assert summary["best_auc_2v4"] > 0.9
    assert 0.0 < summary["mean_position_informative"]["3hop"] < 1.0


def test_tier2g_run_skips_binary_runs(planted_xy, make_run, tmp_path, capsys):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=5, n_layers=1)
    run_dir = make_run("synth2g_bin", X, y)
    tier2g_run(run_dir, plots_dir=str(tmp_path / "figs"))
    assert "SKIP: need >=3 ordered classes" in capsys.readouterr().out
