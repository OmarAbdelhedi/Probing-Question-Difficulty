"""Tests for src/run_data.py -- the shared data layer every tier loads through:
metadata records, label ordering, root-activation loading, RunData, and the
plots-dir default. Also enforces the layering guarantee: this layer (and the
math core) must import without matplotlib."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from src.run_data import (
    load_metadata,
    load_root_activations,
    load_run,
    order_labels,
    resolve_plots_dir,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Layering                                                                    #
# --------------------------------------------------------------------------- #
def test_data_and_math_layers_import_without_matplotlib():
    """The whole point of the split: tiers' IO must not drag in plotting."""
    code = ("import sys; import src.run_data, src.probe_core; "
            "assert 'matplotlib' not in sys.modules, "
            "'the data/math layer pulled in matplotlib'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


# --------------------------------------------------------------------------- #
# order_labels                                                                #
# --------------------------------------------------------------------------- #
def test_order_labels_hardness_words_then_alphabetical():
    got = order_labels(["zebra", "hard", "easy", "medium", "apple"])
    assert got == ["easy", "medium", "hard", "apple", "zebra"]


def test_order_labels_case_insensitive_but_preserving():
    assert order_labels(["HARD", "Easy"]) == ["Easy", "HARD"]


def test_order_labels_dedupes():
    assert order_labels(["2hop", "4hop", "2hop", "3hop"]) == ["2hop", "3hop", "4hop"]


# --------------------------------------------------------------------------- #
# load_metadata                                                               #
# --------------------------------------------------------------------------- #
def write_metadata(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write((json.dumps(line) if isinstance(line, dict) else line) + "\n")


def test_load_metadata_happy_path_tolerates_blank_lines(tmp_path):
    path = tmp_path / "metadata.jsonl"
    write_metadata(path, [{"id": "q1", "label": "easy", "nodes": []}, "",
                          {"id": "q2", "label": "hard", "nodes": []}])
    records = load_metadata(str(path))
    assert [r["id"] for r in records] == ["q1", "q2"]


def test_load_metadata_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(str(tmp_path / "nope.jsonl"))


@pytest.mark.parametrize(
    "lines, match",
    [
        (["{bad"], "Invalid JSON"),
        (["[1]"], "not a JSON object"),
        ([{"id": "q1", "label": "easy"}], "Missing key 'nodes'"),
        ([{"id": "q1", "label": "easy", "nodes": "oops"}], "'nodes' must be a list"),
    ],
)
def test_load_metadata_malformed_lines(tmp_path, lines, match):
    path = tmp_path / "metadata.jsonl"
    write_metadata(path, lines)
    with pytest.raises(ValueError, match=match):
        load_metadata(str(path))


def test_load_metadata_empty_file_raises(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="No question records"):
        load_metadata(str(path))


# --------------------------------------------------------------------------- #
# load_root_activations                                                       #
# --------------------------------------------------------------------------- #
def test_load_root_activations_round_trip(planted_xy, make_run):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=4)
    run_dir = make_run("r", X, y)
    loaded = load_root_activations(os.path.join(run_dir, "metadata.jsonl"),
                                   os.path.join(run_dir, "activations"))
    assert loaded.X.dtype == np.float32 and loaded.X.shape == X.shape
    np.testing.assert_array_equal(loaded.X, X)  # metadata order preserved
    assert loaded.y == [str(lab) for lab in y]
    assert loaded.layer_indices == [0, 1, 2]  # "all" -> positional
    assert loaded.missing == []


def test_load_root_activations_skips_unusable_npz(planted_xy, make_run):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=3)
    run_dir = make_run("r", X, y, skip_npz={"r_q1"}, drop_root_key={"r_q4"})
    loaded = load_root_activations(os.path.join(run_dir, "metadata.jsonl"),
                                   os.path.join(run_dir, "activations"))
    assert sorted(loaded.missing) == ["r_q1", "r_q4"]
    assert loaded.X.shape[0] == 4


def test_load_root_activations_layer_provenance(planted_xy, make_run):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=3, n_layers=3)
    run_dir = make_run("r", X, y, layers=[0, 7, 21])  # explicit selection
    loaded = load_root_activations(os.path.join(run_dir, "metadata.jsonl"),
                                   os.path.join(run_dir, "activations"))
    assert loaded.layer_indices == [0, 7, 21]

    bad = make_run("r_bad", X, y, layers=[0, 7])  # provenance/shape mismatch
    loaded = load_root_activations(os.path.join(bad, "metadata.jsonl"),
                                   os.path.join(bad, "activations"))
    assert loaded.layer_indices == [0, 1, 2]  # falls back to positional


def test_load_root_activations_inconsistent_shapes_raise(planted_xy, make_run):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=3)
    run_dir = make_run("r", X, y)
    np.savez(os.path.join(run_dir, "activations", "r_q0.npz"),
             root=np.zeros((5, 9), dtype=np.float32))  # different layers/hidden
    with pytest.raises(ValueError, match="Inconsistent activation shapes"):
        load_root_activations(os.path.join(run_dir, "metadata.jsonl"),
                              os.path.join(run_dir, "activations"))


def test_load_root_activations_nothing_loadable_raises(planted_xy, make_run):
    X, y = planted_xy([("easy", 0.0)], per_class=2)
    run_dir = make_run("r", X, y, skip_npz={"r_q0", "r_q1"})
    with pytest.raises(FileNotFoundError, match="No root activations"):
        load_root_activations(os.path.join(run_dir, "metadata.jsonl"),
                              os.path.join(run_dir, "activations"))


# --------------------------------------------------------------------------- #
# load_run / resolve_plots_dir                                                #
# --------------------------------------------------------------------------- #
def test_load_run_defaults_labels_and_provenance(planted_xy, make_run):
    X, y = planted_xy([("hard", 2.0), ("easy", 0.0)], per_class=3)
    run_dir = make_run("myrun", X, y)
    data = load_run(run_dir)
    assert data.name == "myrun"
    assert data.labels == ["easy", "hard"]  # ordered easy-ish -> hard-ish
    assert data.counts == {"easy": 3, "hard": 3}
    assert data.run_config["model_name"] == "synth-model"


def test_load_run_missing_activations_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no activations dir"):
        load_run(str(tmp_path))


def test_load_run_unreadable_run_config_degrades_to_empty(planted_xy, make_run):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=2)
    run_dir = make_run("r", X, y)
    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        fh.write("{broken")
    assert load_run(run_dir).run_config == {}


def test_resolve_plots_dir_creates_requested_dir(tmp_path):
    target = tmp_path / "figs" / "nested"
    assert resolve_plots_dir(str(target)) == str(target)
    assert target.is_dir()
