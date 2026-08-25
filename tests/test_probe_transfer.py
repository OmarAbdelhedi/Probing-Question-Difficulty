"""Tests for src/probe_transfer.py (Tier 2t): run-compatibility guards, the
skip path, and an end-to-end transfer between two fabricated runs that share
ONE planted depth direction (a compact, asserting version of the module's
__main__ smoke test)."""

import json

import numpy as np
import pytest

from src.run_data import RunData
from src.probe_transfer import (
    _check_compatible,
    run as tier2t_run,
    transfer_direction,
)


def mk_rundata(name, X, y, labels, layer_indices=None, run_config=None):
    """Hand-build a RunData (what load_run would return) for unit tests."""
    y = np.asarray(y, dtype=object)
    return RunData(
        name=name, run_dir=name, metadata_path="", activations_dir="",
        X=X, y=y, ids=[f"{name}_q{i}" for i in range(X.shape[0])],
        labels=list(labels),
        counts={str(lab): int(np.sum(y == lab)) for lab in labels},
        layer_indices=(list(range(X.shape[1])) if layer_indices is None
                       else list(layer_indices)),
        missing=[], run_config=run_config or {},
    )


def binary_rundata(name, n_layers=2, dim=8, **kw):
    X = np.zeros((4, n_layers, dim), dtype=np.float32)
    return mk_rundata(name, X, ["easy", "easy", "hard", "hard"], ["easy", "hard"], **kw)


# --------------------------------------------------------------------------- #
# Compatibility guard                                                         #
# --------------------------------------------------------------------------- #
def test_check_compatible_shape_mismatch():
    with pytest.raises(ValueError, match="incompatible runs"):
        _check_compatible(binary_rundata("a", dim=8), binary_rundata("b", dim=16))


def test_check_compatible_layer_indices_mismatch():
    a = binary_rundata("a", layer_indices=[0, 5])
    b = binary_rundata("b", layer_indices=[0, 9])
    with pytest.raises(ValueError, match="incompatible layer selections"):
        _check_compatible(a, b)


@pytest.mark.parametrize("key", ["model_name", "pooling"])
def test_check_compatible_provenance_mismatch(key):
    a = binary_rundata("a", run_config={key: "x"})
    b = binary_rundata("b", run_config={key: "y"})
    with pytest.raises(ValueError, match=key):
        _check_compatible(a, b)


def test_check_compatible_dtype_mismatch_only_warns(capsys):
    a = binary_rundata("a", run_config={"dtype": "float16"})
    b = binary_rundata("b", run_config={"dtype": "bfloat16"})
    _check_compatible(a, b)  # no raise
    assert "dtype differs" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# transfer_direction skip path                                                #
# --------------------------------------------------------------------------- #
def test_transfer_direction_skips_tiny_train_extremes():
    X = np.zeros((2, 1, 4), dtype=np.float32)
    train = mk_rundata("a", X, ["easy", "hard"], ["easy", "hard"])
    test = binary_rundata("b", n_layers=1, dim=4)
    d = transfer_direction(train, test)
    assert d["skipped"] and "train extremes too small" in d["reason"]


# --------------------------------------------------------------------------- #
# End-to-end: graded run A <-> binary run B sharing one planted axis          #
# --------------------------------------------------------------------------- #
def test_tier2t_run_end_to_end(make_run, tmp_path):
    rng = np.random.default_rng(0)
    n_layers, dim, per_class = 3, 40, 25
    u = rng.normal(size=dim)
    u /= np.linalg.norm(u)

    def fabricate(labels_strengths):
        offset = rng.normal(size=dim) * 0.5  # dataset-specific wholesale shift
        rows, ys = [], []
        for label, strength in labels_strengths:
            for _ in range(per_class):
                rows.append((rng.normal(size=(n_layers, dim)) + offset
                             + strength * u).astype(np.float32))
                ys.append(label)
        return np.stack(rows), np.asarray(ys, dtype=object)

    Xa, ya = fabricate([("2hop", 0.0), ("3hop", 3.0), ("4hop", 6.0)])
    Xb, yb = fabricate([("easy", 0.0), ("hard", 6.0)])
    run_a = make_run("runA", Xa, ya)
    run_b = make_run("runB", Xb, yb)

    figs = tmp_path / "figs"
    tier2t_run(run_a, run_b, plots_dir=str(figs), folds=4, seed=0)

    assert (figs / "tier2t_transfer_by_layer.png").is_file()
    with open(figs / "tier2t_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)

    a_to_b, b_to_a = summary["directions"]
    for d in (a_to_b, b_to_a):
        assert not d["skipped"]
        assert d["best_transfer_auc"] > 0.85   # the shared axis transfers
        assert d["retention"] > 0.6            # ... and retains most of the ceiling

    # Label vocabularies need not match: graded A maps onto binary B.
    assert a_to_b["train_low"] == "2hop" and a_to_b["test_low"] == "easy"
    assert a_to_b["test_middles"] == []
    # B's binary axis places A's held-out 3hop between A's extremes.
    assert b_to_a["test_middles"] == ["3hop"]
    assert 0.0 < b_to_a["mean_position_informative"]["3hop"] < 1.0


def test_tier2t_run_same_run_warns(planted_xy, make_run, tmp_path, capsys):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.5)], per_class=8, n_layers=1)
    run_dir = make_run("twice", X, y)
    tier2t_run(run_dir, run_dir, plots_dir=str(tmp_path / "figs"))
    out = capsys.readouterr().out
    assert "SAME run" in out  # circular-transfer warning fired
