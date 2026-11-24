"""Tests for the surface features (src/surface.py) and Tier 2b
(src/probe_controls.py): the feature matrix, the tokenizer-free path, and
end-to-end control runs (small permutation / bootstrap counts) on fabricated runs."""

import json

import numpy as np
import pytest

from src.probe_controls import run as tier2b_run
from src.surface import SURFACE_FEATURES, surface_matrix, try_load_tokenizer

Q0 = "Who was Ada Lovelace born in 1815, and why?"
Q1 = "Short?"


def test_surface_matrix_columns():
    cov, names = surface_matrix([Q0, Q1], SURFACE_FEATURES)
    assert names == ["char_len", "word_count", "cap_words", "num_tokens", "punct"]
    assert cov.shape == (2, 5)
    # "Who was Ada Lovelace born in 1815, and why?" by hand:
    assert cov[0].tolist() == [float(len(Q0)),  # char_len
                               9.0,             # word_count
                               2.0,             # cap_words: Ada, Lovelace (not initial "Who")
                               1.0,             # num_tokens: "1815,"
                               2.0]             # punct: one ',' + one '?'
    assert cov[1].tolist() == [float(len(Q1)), 1.0, 0.0, 0.0, 1.0]


def test_surface_matrix_token_count_column():
    class Tok:
        def encode(self, q):
            return list(range(len(q.split()) + 1))  # one "BOS" + one id per word

    cov, names = surface_matrix([Q0, Q1], SURFACE_FEATURES, tokenizer=Tok())
    assert names[-1] == "token_count"
    assert cov[:, -1].tolist() == [10.0, 2.0]


def test_try_load_tokenizer_without_model_name_returns_none(capsys):
    assert try_load_tokenizer({}) is None
    assert "skipped" in capsys.readouterr().out


def test_tier2b_run_end_to_end(planted_xy, make_run, tmp_path):
    X, y = planted_xy([("easy", 0.0), ("hard", 6.0)], per_class=12, n_layers=2)
    # Constant-length questions: every surface covariate is identical across
    # rows, so the controls have nothing to remove and must NOT eat the
    # planted activation signal. No model_name in run_config -> the
    # token-count covariate is skipped (never attempts a tokenizer download).
    questions = [f"synthetic question number {i:03d}?" for i in range(X.shape[0])]
    run_dir = make_run("synth2b", X, y, questions=questions,
                       run_config={"pooling": "last_token", "dtype": "float32"})
    figs = tmp_path / "figs"
    tier2b_run(run_dir, plots_dir=str(figs), folds=4, seed=0, perms=5, n_boot=50)

    for name in ("tier2b_length_control_by_layer.png",
                 "tier2b_multicovariate_control_by_layer.png",
                 "tier2b_permutation_null_by_layer.png"):
        assert (figs / name).is_file()

    with open(figs / "tier2b_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["perms"] == 5
    assert summary["covariates"] == SURFACE_FEATURES
    assert len(summary["raw_auc"]) == 2 and len(summary["null_mean"]) == 2
    # Planted signal: the real probe beats the shuffled-label null...
    assert summary["raw_auc_mean"] > summary["null_auc_mean"]
    assert summary["raw_auc_mean"] > 0.9
    # ... a constant covariate alone predicts nothing (all scores tie) ...
    assert summary["length_only_auc"] == pytest.approx(0.5)
    # ... and the signal survives residualizing the constant surface features.
    assert summary["surface_residualized_auc_mean"] > 0.9

    # Bootstrap CIs: the peak matches the curve, and the CI brackets it.
    boot = summary["bootstrap"]
    assert boot["n_boot"] == 50
    for key, curve in (("raw", "raw_auc"), ("surface_residualized", "surface_residualized_auc")):
        b = boot[key]
        assert b["auc"] == pytest.approx(max(summary[curve]))
        assert b["ci95"][0] <= b["auc"] <= b["ci95"][1]
        assert b["half_width"] == pytest.approx((b["ci95"][1] - b["ci95"][0]) / 2)


def test_tier2b_bootstrap_can_be_disabled(planted_xy, make_run, tmp_path):
    X, y = planted_xy([("easy", 0.0), ("hard", 3.0)], per_class=6, n_layers=1)
    run_dir = make_run("noboot", X, y)
    figs = tmp_path / "figs"
    tier2b_run(run_dir, plots_dir=str(figs), folds=3, seed=0, perms=2, n_boot=0)
    with open(figs / "tier2b_summary.json", encoding="utf-8") as fh:
        assert "bootstrap" not in json.load(fh)


def test_tier2b_run_skips_single_label(planted_xy, make_run, tmp_path, capsys):
    X, y = planted_xy([("easy", 0.0)], per_class=6, n_layers=1)
    run_dir = make_run("synth2b_one", X, y)
    tier2b_run(run_dir, plots_dir=str(tmp_path / "figs"))
    assert "SKIP: need >=2 labels" in capsys.readouterr().out


def test_tier2b_length_confound_collapses(make_run, tmp_path):
    # Activations encode ONLY question length, and length determines the label:
    # the length-residualized probe must collapse to ~the null band.
    rng = np.random.default_rng(3)
    n, dim = 40, 16
    lengths = np.concatenate([rng.integers(20, 40, size=20),
                              rng.integers(120, 180, size=20)]).astype(float)
    u = rng.normal(size=dim)
    u /= np.linalg.norm(u)
    X = (rng.normal(size=(n, 1, dim)) * 0.05
         + np.outer(lengths / 100.0, u)[:, None, :]).astype(np.float32)
    y = np.asarray(["easy"] * 20 + ["hard"] * 20, dtype=object)
    questions = ["x" * int(c - 1) + "?" for c in lengths]  # char_len == planted driver
    run_dir = make_run("confound", X, y, questions=questions,
                       run_config={"pooling": "last_token", "dtype": "float32"})

    figs = tmp_path / "figs"
    tier2b_run(run_dir, plots_dir=str(figs), folds=4, seed=0, perms=5, n_boot=20)
    with open(figs / "tier2b_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["raw_auc_mean"] > 0.95          # looks decodable...
    assert summary["length_only_auc"] > 0.95       # ...but length alone suffices
    assert summary["residualized_auc_mean"] < 0.75  # and the control catches it
