"""Tests for src/probe_controls.py (Tier 2b): the surface-covariate matrix,
the tokenizer-free degradation path, and an end-to-end controls run (small
permutation count) over a fabricated run."""

import json
import os

import numpy as np
import pytest

from src.probe_controls import _surface_covariates, _try_load_tokenizer, run as tier2b_run


def test_surface_covariates_columns(planted_xy, make_run):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=1, n_layers=1)
    q0 = "Who was Ada Lovelace born in 1815, and why?"
    q1 = "Short?"
    run_dir = make_run("cov", X, y, questions=[q0, q1])
    ids = ["cov_q0", "cov_q1"]

    cov, names = _surface_covariates(os.path.join(run_dir, "metadata.jsonl"), ids,
                                     tokenizer=None)
    assert names == ["char_len", "word_count", "cap_words", "num_tokens", "punct"]
    assert cov.shape == (2, 5)
    # "Who was Ada Lovelace born in 1815, and why?" by hand:
    assert cov[0].tolist() == [float(len(q0)),  # char_len
                               9.0,             # word_count
                               2.0,             # cap_words: Ada, Lovelace (not initial "Who")
                               1.0,             # num_tokens: "1815,"
                               2.0]             # punct: one ',' + one '?'
    assert cov[1].tolist() == [float(len(q1)), 1.0, 0.0, 0.0, 1.0]


def test_try_load_tokenizer_without_model_name_returns_none(capsys):
    assert _try_load_tokenizer({}) is None
    assert "skipped" in capsys.readouterr().out


def test_tier2b_run_end_to_end(planted_xy, make_run, tmp_path):
    X, y = planted_xy([("easy", 0.0), ("hard", 6.0)], per_class=12, n_layers=2)
    # Constant-length questions: every surface covariate is identical across
    # rows, so the controls have nothing to remove and must NOT eat the
    # planted activation signal. No model_name in run_config -> the
    # token-count covariate is skipped everywhere (never attempts a tokenizer
    # download on a cluster node).
    questions = [f"synthetic question number {i:03d}?" for i in range(X.shape[0])]
    run_dir = make_run("synth2b", X, y, questions=questions,
                       run_config={"pooling": "last_token", "dtype": "float32"})
    figs = tmp_path / "figs"
    tier2b_run(run_dir, plots_dir=str(figs), folds=4, seed=0, perms=5)

    for name in ("tier2b_length_control_by_layer.png",
                 "tier2b_multicovariate_control_by_layer.png",
                 "tier2b_permutation_null_by_layer.png"):
        assert (figs / name).is_file()

    with open(figs / "tier2b_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["perms"] == 5
    assert summary["covariates"] == ["char_len", "word_count", "cap_words",
                                     "num_tokens", "punct"]
    assert len(summary["raw_auc"]) == 2 and len(summary["null_mean"]) == 2
    # Planted signal: the real probe beats the shuffled-label null...
    assert summary["raw_auc_mean"] > summary["null_auc_mean"]
    assert summary["raw_auc_mean"] > 0.9
    # ... a constant covariate alone predicts nothing (all scores tie) ...
    assert summary["length_only_auc"] == pytest.approx(0.5)
    # ... and the signal survives residualizing the constant surface features.
    assert summary["surface_residualized_auc_mean"] > 0.9


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
    tier2b_run(run_dir, plots_dir=str(figs), folds=4, seed=0, perms=5)
    with open(figs / "tier2b_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["raw_auc_mean"] > 0.95          # looks decodable...
    assert summary["length_only_auc"] > 0.95       # ...but length alone suffices
    assert summary["residualized_auc_mean"] < 0.75  # and the control catches it
