"""Tests for src/transfer.py: the 2-train/1-test fused transfer behind Table II
(job enumeration, directory naming, compatibility guards, and an end-to-end
sweep over three fabricated runs sharing one planted difficulty direction)."""

import json
import os

import numpy as np
import pytest

from src.run_data import RunData
from src.surface import SURFACE_FEATURES
from src.transfer import (
    build_fused_jobs,
    check_compatible,
    fused_plots_dir,
    run as transfer_run,
    tag_of,
)


def mk_rundata(name, n_layers=2, dim=8, layer_indices=None, run_config=None):
    """Hand-build a RunData (what load_run would return) for the guards."""
    X = np.zeros((4, n_layers, dim), dtype=np.float32)
    y = np.asarray(["easy", "easy", "hard", "hard"], dtype=object)
    return RunData(
        name=name, run_dir=name, metadata_path="", activations_dir="",
        X=X, y=y, ids=[f"{name}_q{i}" for i in range(4)], questions=["q?"] * 4,
        labels=["easy", "hard"], counts={"easy": 2, "hard": 2},
        layer_indices=(list(range(n_layers)) if layer_indices is None else list(layer_indices)),
        missing=[], run_config=run_config or {},
    )


# --------------------------------------------------------------------------- #
# Jobs and naming                                                             #
# --------------------------------------------------------------------------- #
def test_build_fused_jobs_three_runs():
    jobs = build_fused_jobs(["a/run_2wiki", "a/run_hotpot", "a/run_musique"])
    assert [(j["train_outputs"], j["test_output"]) for j in jobs] == [
        (["a/run_hotpot", "a/run_musique"], "a/run_2wiki"),
        (["a/run_2wiki", "a/run_musique"], "a/run_hotpot"),
        (["a/run_2wiki", "a/run_hotpot"], "a/run_musique"),
    ]


def test_build_fused_jobs_rejects_duplicates_and_too_few():
    with pytest.raises(ValueError, match="duplicate"):
        build_fused_jobs(["a/run_x", "a/run_x/", "a/run_y"])
    with pytest.raises(ValueError, match="at least 3"):
        build_fused_jobs(["a/run_x", "a/run_y"])


def test_fused_plots_dir_matches_published_names():
    assert tag_of("outputs/qwen32b/run_musique/") == "musique"
    root = os.path.join("figures", "m")
    assert fused_plots_dir(root, ["o/run_hotpot", "o/run_2wiki"], "o/run_musique") == \
        os.path.join(root, "transfer_sweep", "transfer_2wiki_hotpot--musique")
    assert fused_plots_dir(root, ["o/run_hotpot", "o/run_musique"], "o/run_2wiki") == \
        os.path.join(root, "transfer_sweep", "transfer_musique_hotpot--2wiki")
    assert fused_plots_dir(root, ["o/run_musique", "o/run_2wiki"], "o/run_hotpot") == \
        os.path.join(root, "transfer_sweep", "transfer_2wiki_musique--hotpot")


# --------------------------------------------------------------------------- #
# Compatibility guard                                                         #
# --------------------------------------------------------------------------- #
def test_check_compatible_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        check_compatible(mk_rundata("a", dim=8), mk_rundata("b", dim=9))


def test_check_compatible_layer_indices_mismatch():
    with pytest.raises(ValueError, match="layer selections"):
        check_compatible(mk_rundata("a", layer_indices=[0, 1]),
                         mk_rundata("b", layer_indices=[0, 5]))


@pytest.mark.parametrize("key", ["model_name", "pooling"])
def test_check_compatible_provenance_mismatch(key):
    with pytest.raises(ValueError, match=key):
        check_compatible(mk_rundata("a", run_config={key: "x"}),
                         mk_rundata("b", run_config={key: "y"}))


def test_check_compatible_missing_provenance_is_tolerated():
    check_compatible(mk_rundata("a", run_config={"model_name": "m"}), mk_rundata("b"))


# --------------------------------------------------------------------------- #
# End-to-end sweep                                                            #
# --------------------------------------------------------------------------- #
def test_fused_transfer_end_to_end(make_run, tmp_path):
    rng = np.random.default_rng(0)
    n_layers, dim, per_class = 3, 40, 25
    u = rng.normal(size=dim)
    u /= np.linalg.norm(u)

    def fabricate(name, labels_strengths):
        offset = rng.normal(size=dim) * 0.5  # dataset-specific shift
        rows, ys, qs = [], [], []
        for label, strength in labels_strengths:
            for i in range(per_class):
                rows.append((rng.normal(size=(n_layers, dim)) + offset
                             + strength * u).astype(np.float32))
                ys.append(label)
                qs.append(f"What is item {i} of {label} in Some Place?")
        return make_run(name, np.stack(rows), np.asarray(ys, dtype=object), questions=qs)

    runs = [
        fabricate("run_2wiki", [("easy", 0.0), ("hard", 6.0)]),
        fabricate("run_hotpot", [("easy", 0.0), ("medium", 3.0), ("hard", 6.0)]),
        fabricate("run_musique", [("2hop", 0.0), ("3hop", 3.0), ("4hop", 6.0)]),
    ]
    root = tmp_path / "figs"
    summary = transfer_run(runs, str(root), folds=4, seed=0)

    sweep = root / "transfer_sweep"
    assert (sweep / "fused_transfer_sweep_summary.json").is_file()
    assert [j["status"] for j in summary["jobs"]] == ["ok"] * 3

    combo = sweep / "transfer_2wiki_hotpot--musique"
    for name in ("fused_transfer_report.png", "fused_transfer_report.json",
                 "fused_transfer_summary.json", "fused_transfer_residual_summary.json"):
        assert (combo / name).is_file()

    with open(combo / "fused_transfer_summary.json", encoding="utf-8") as fh:
        raw = json.load(fh)
    with open(combo / "fused_transfer_residual_summary.json", encoding="utf-8") as fh:
        resid = json.load(fh)

    d = raw["direction"]
    assert d["test"] == "run_musique" and d["test_middles"] == ["3hop"]
    assert d["mean_transfer_auc_informative"] > 0.9   # the shared axis transfers
    assert d["retention"] > 0.8                       # ... close to the ceiling
    assert 0.0 < d["mean_position_informative"]["3hop"] < 1.0  # middle interpolates
    assert raw["informative_auc_threshold"] == 0.7
    # Same features as Tier 2b ("synth-model" has no tokenizer -> the five
    # tokenizer-free ones).
    assert resid["residualization"]["covariates"] == SURFACE_FEATURES
    assert resid["direction"]["residualized"] is True
