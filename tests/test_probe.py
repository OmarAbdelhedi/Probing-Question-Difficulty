"""Tests for src/probe.py (Tier 2): the centroid-separation descriptive
statistic and an end-to-end ``run()`` over a fabricated run directory.
(Run loading is covered in test_run_data.py; the probe math and the CV driver
in test_probe_core.py.)"""

import json

from src.probe import centroid_separation_by_layer, run as tier2_run


def test_centroid_separation_peaks_at_signal_layer(planted_xy):
    X, y = planted_xy([("easy", 0.0), ("hard", 6.0)], per_class=20,
                      n_layers=3, signal_layers=[1])
    sep = centroid_separation_by_layer(X, y, ["easy", "hard"])
    assert len(sep) == 3
    assert sep[1] > sep[0] and sep[1] > sep[2]


# --------------------------------------------------------------------------- #
# End-to-end Tier 2                                                           #
# --------------------------------------------------------------------------- #
def test_tier2_run_end_to_end(planted_xy, make_run, tmp_path):
    X, y = planted_xy([("easy", 0.0), ("hard", 6.0)], per_class=15,
                      n_layers=3, signal_layers=[1])
    run_dir = make_run("synth", X, y)
    figs = tmp_path / "figs"
    tier2_run(run_dir, plots_dir=str(figs), folds=4, seed=0)

    expected = {"tier2_probe_auc_by_layer.png", "tier2_centroid_separation_by_layer.png",
                "tier2_pca_scatter_by_layer.png", "tier2_diff_projection_by_layer.png",
                "tier2_activation_norm_by_layer.png", "tier2_summary.json"}
    assert expected <= {p.name for p in figs.iterdir()}

    with open(figs / "tier2_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_questions"] == 30
    assert summary["probe"]["best_layer_index"] == 1  # the planted layer
    assert summary["probe"]["best_auc"] > 0.95
    assert sorted(summary["plots"]) == sorted(expected - {"tier2_summary.json"})


def test_tier2_run_skips_gracefully_without_activations(tmp_path, capsys):
    tier2_run(str(tmp_path / "missing_run"), plots_dir=str(tmp_path / "figs"))
    assert "SKIP Tier 2" in capsys.readouterr().out
