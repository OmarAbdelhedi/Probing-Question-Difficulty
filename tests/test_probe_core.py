"""Unit tests for src/probe_core.py -- the shared cross-tier probe math.

Each metric is checked against an independent, brute-force reference
implementation written here (not against golden constants), so the tests pin
the DEFINITION of the math without being sensitive to the numpy version.
"""

import numpy as np
import pytest

from src.probe_core import (
    EPS,
    REPO_ROOT,
    argmax_finite,
    balanced_acc,
    fit_diag_lda,
    middle_positions,
    oof_diag_lda,
    probe_auc_by_layer,
    project_diag_lda,
    rankdata,
    residualize,
    roc_auc,
    stratified_test_folds,
)


# --------------------------------------------------------------------------- #
# Brute-force references                                                      #
# --------------------------------------------------------------------------- #
def ref_avg_ranks(a):
    """O(n^2) tie-aware average ranks, 1-based, straight from the definition."""
    a = np.asarray(a, dtype=float)
    return np.array([np.sum(a < v) + (np.sum(a == v) + 1) / 2.0 for v in a])


def ref_auc(scores, pos_mask):
    """Pairwise P(pos > neg) + 0.5 * P(pos == neg), over finite scores only."""
    finite = np.isfinite(scores)
    pos = scores[finite & pos_mask]
    neg = scores[finite & ~pos_mask]
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------- #
# Scoring metrics                                                             #
# --------------------------------------------------------------------------- #
def test_rankdata_matches_reference_with_ties():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 8, size=60).astype(float)  # many ties
    np.testing.assert_allclose(rankdata(a), ref_avg_ranks(a))


def test_rankdata_simple_cases():
    np.testing.assert_allclose(rankdata(np.array([3.0, 1.0, 2.0])), [3, 1, 2])
    np.testing.assert_allclose(rankdata(np.array([1.0, 1.0])), [1.5, 1.5])


def test_rankdata_matches_scipy_if_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, size=80).astype(float)
    np.testing.assert_allclose(rankdata(a), scipy_stats.rankdata(a))


def test_roc_auc_matches_pairwise_reference():
    rng = np.random.default_rng(2)
    scores = rng.integers(0, 10, size=50).astype(float)  # ties included
    pos = rng.random(50) > 0.6
    assert roc_auc(scores, pos) == pytest.approx(ref_auc(scores, pos))


def test_roc_auc_perfect_and_reversed_and_empty():
    pos = np.array([False, False, True, True])
    assert roc_auc(np.array([0.0, 1.0, 2.0, 3.0]), pos) == 1.0
    assert roc_auc(np.array([3.0, 2.0, 1.0, 0.0]), pos) == 0.0
    assert np.isnan(roc_auc(np.array([1.0, 2.0]), np.array([True, True])))


def test_roc_auc_ignores_non_finite_scores():
    scores = np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.inf])
    pos = np.array([False, False, True, True, True, False])
    # NaN/Inf rows are dropped entirely, leaving the perfect 2v2 split.
    assert roc_auc(scores, pos) == 1.0


def test_balanced_acc_known_confusions():
    pos = np.array([True, True, True, False, False])
    pred = np.array([True, True, False, False, True])  # TPR=2/3, TNR=1/2
    assert balanced_acc(pos, pred) == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_balanced_acc_degenerate_single_class():
    pos = np.array([True, True])
    assert balanced_acc(pos, np.array([True, True])) == 1.0  # TNR is NaN -> mean(TPR)


def test_argmax_finite():
    assert argmax_finite([np.nan, 1.0, 3.0, np.nan, 2.0]) == 2
    assert argmax_finite([np.nan, np.nan]) == 0
    assert argmax_finite([]) == 0


def test_repo_root_points_at_the_repo():
    import os
    assert os.path.isfile(os.path.join(REPO_ROOT, "config.yaml"))
    assert os.path.isdir(os.path.join(REPO_ROOT, "src"))


# --------------------------------------------------------------------------- #
# CV folds & residualization                                                  #
# --------------------------------------------------------------------------- #
def test_stratified_test_folds_partition_and_stratification():
    rng = np.random.default_rng(3)
    pos = np.zeros(34, dtype=bool)
    pos[:13] = True
    k = 4
    folds = stratified_test_folds(pos, k, rng)
    assert len(folds) == k
    all_idx = np.sort(np.concatenate(folds))
    np.testing.assert_array_equal(all_idx, np.arange(34))  # disjoint + exhaustive
    for fold in folds:
        n_pos = int(pos[fold].sum())
        assert n_pos in (3, 4)  # 13 positives over 4 folds
        train = np.setdiff1d(np.arange(34), fold)
        assert pos[train].any() and (~pos[train]).any()  # both classes in train


def test_stratified_test_folds_deterministic_per_seed():
    pos = np.arange(20) < 8
    folds_a = stratified_test_folds(pos, 3, np.random.default_rng(5))
    folds_b = stratified_test_folds(pos, 3, np.random.default_rng(5))
    for fa, fb in zip(folds_a, folds_b):
        np.testing.assert_array_equal(fa, fb)


def test_residualize_removes_linear_dependence_on_train():
    rng = np.random.default_rng(4)
    c_tr = rng.normal(size=40)
    c_te = rng.normal(size=10)
    w = rng.normal(size=6)
    Xtr = np.outer(c_tr, w) + 3.0  # exactly linear in the covariate
    Xte = np.outer(c_te, w) + 3.0
    Rtr, Rte = residualize(Xtr, c_tr, Xte, c_te)
    np.testing.assert_allclose(Rtr, 0.0, atol=1e-10)
    np.testing.assert_allclose(Rte, 0.0, atol=1e-10)


def test_residualize_fits_on_train_only():
    rng = np.random.default_rng(5)
    c_tr = rng.normal(size=50)
    c_te = rng.normal(size=20)
    w = rng.normal(size=4)
    Xtr = np.outer(c_tr, w)          # slope w on train
    Xte = np.outer(c_te, 3.0 * w)    # different slope on test
    Rtr, Rte = residualize(Xtr, c_tr, Xte, c_te)
    np.testing.assert_allclose(Rtr, 0.0, atol=1e-10)
    # Train coefficients remove only the train slope -> 2x the signal remains.
    np.testing.assert_allclose(Rte, np.outer(c_te, 2.0 * w), atol=1e-8)


def test_residualize_2d_and_constant_covariates():
    rng = np.random.default_rng(6)
    C_tr = rng.normal(size=(30, 2))
    C_te = rng.normal(size=(8, 2))
    C_tr[:, 1] = 7.0  # ~constant column must be neutralized, not divide-by-zero
    C_te[:, 1] = 7.0
    X = rng.normal(size=(30, 5))
    Rtr, Rte = residualize(X, C_tr, rng.normal(size=(8, 5)), C_te)
    assert np.isfinite(Rtr).all() and np.isfinite(Rte).all()


# --------------------------------------------------------------------------- #
# The diagonal-LDA axis                                                       #
# --------------------------------------------------------------------------- #
def test_fit_diag_lda_matches_inline_definition():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 9))
    pos = np.arange(24) < 10
    mu, sd, w, thr = fit_diag_lda(X, pos)

    mu_ref = X.mean(axis=0)
    sd_ref = X.std(axis=0)
    sd_ref[sd_ref < EPS] = 1.0
    Xs = (X - mu_ref) / sd_ref
    w_ref = Xs[pos].mean(axis=0) - Xs[~pos].mean(axis=0)
    s = Xs @ w_ref
    thr_ref = 0.5 * (s[pos].mean() + s[~pos].mean())

    np.testing.assert_array_equal(mu, mu_ref)
    np.testing.assert_array_equal(sd, sd_ref)
    np.testing.assert_array_equal(w, w_ref)
    assert thr == thr_ref


def test_fit_diag_lda_orientation_and_threshold(planted_xy):
    X, y = planted_xy([("easy", 0.0), ("hard", 3.0)], per_class=30, n_layers=1)
    Xl = X[:, 0, :].astype(np.float64)
    pos = y == "hard"
    mu, sd, w, thr = fit_diag_lda(Xl, pos)
    s = project_diag_lda(Xl, mu, sd, w)
    assert s[pos].mean() > thr > s[~pos].mean()  # oriented negative -> positive


def test_fit_diag_lda_constant_feature_is_neutralized():
    X = np.ones((10, 3))
    X[:, 0] = np.arange(10, dtype=float)  # only feature 0 varies
    pos = np.arange(10) >= 5
    mu, sd, w, thr = fit_diag_lda(X, pos)
    assert np.isfinite(w).all()
    assert w[1] == 0.0 and w[2] == 0.0  # constant features contribute nothing


def test_project_diag_lda_uses_train_standardization():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(20, 4))
    mu, sd, w, _ = fit_diag_lda(X, np.arange(20) < 10)
    delta = rng.normal(size=4)
    shifted = project_diag_lda(X + delta, mu, sd, w)
    np.testing.assert_allclose(shifted, project_diag_lda(X, mu, sd, w) + (delta / sd) @ w)


def test_oof_diag_lda_scores_every_row_once_and_separates(planted_xy):
    X, y = planted_xy([("easy", 0.0), ("hard", 6.0)], per_class=25, n_layers=1)
    pos = y == "hard"
    rng = np.random.default_rng(9)
    folds = stratified_test_folds(pos, 5, rng)
    scores, preds = oof_diag_lda(X[:, 0, :], pos, folds)
    assert np.isfinite(scores).all()  # every row scored exactly once
    assert preds.dtype == bool
    assert roc_auc(scores, pos) > 0.95
    assert balanced_acc(pos, preds) > 0.85


def test_oof_diag_lda_shuffled_labels_near_chance(planted_xy):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.5)], per_class=30, n_layers=1)
    rng = np.random.default_rng(10)
    pos_shuffled = rng.permutation(y == "hard")
    folds = stratified_test_folds(pos_shuffled, 5, rng)
    scores, _ = oof_diag_lda(X[:, 0, :], pos_shuffled, folds)
    assert 0.3 < roc_auc(scores, pos_shuffled) < 0.7


def test_oof_diag_lda_covariate_removes_confounded_signal():
    # The class signal IS the covariate direction: residualizing it out within
    # each train fold must collapse the probe to chance (Tier 2b's machinery).
    rng = np.random.default_rng(11)
    n, dim = 60, 16
    c = rng.normal(size=n)
    u = rng.normal(size=dim)
    u /= np.linalg.norm(u)
    X = rng.normal(size=(n, dim)) * 0.05 + np.outer(c, u)
    pos = c > np.median(c)
    folds = stratified_test_folds(pos, 5, rng)
    raw_scores, _ = oof_diag_lda(X, pos, folds)
    resid_scores, _ = oof_diag_lda(X, pos, folds, covariate=c)
    assert roc_auc(raw_scores, pos) > 0.95
    assert 0.25 < roc_auc(resid_scores, pos) < 0.75


# --------------------------------------------------------------------------- #
# Middle-class interpolation                                                  #
# --------------------------------------------------------------------------- #
def test_middle_positions_planted_means():
    scores = np.array([0.0, 0.0, 1.0, 1.0, 0.25, 0.75])
    pos = middle_positions(scores, low_idx=np.array([0, 1]), high_idx=np.array([2, 3]),
                           mid_idx={"m1": np.array([4]), "m2": np.array([5])})
    assert pos == {"m1": pytest.approx(0.25), "m2": pytest.approx(0.75)}


def test_middle_positions_outside_extremes_and_order():
    scores = np.array([0.0, 1.0, -0.5, 1.5])
    pos = middle_positions(scores, np.array([0]), np.array([1]),
                           {"below": np.array([2]), "above": np.array([3])})
    assert list(pos) == ["below", "above"]  # mid_idx key order preserved
    assert pos["below"] == pytest.approx(-0.5)
    assert pos["above"] == pytest.approx(1.5)


def test_middle_positions_degenerate_extremes_give_nan():
    scores = np.full(6, 2.0)  # extremes do not separate -> |denom| <= EPS
    pos = middle_positions(scores, np.array([0, 1]), np.array([2, 3]),
                           {"m": np.array([4, 5])})
    assert np.isnan(pos["m"])


def test_middle_positions_ignores_nan_scores():
    scores = np.array([0.0, np.nan, 1.0, np.nan, 0.5])
    pos = middle_positions(scores, np.array([0, 1]), np.array([2, 3]),
                           {"m": np.array([4])})
    assert pos["m"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# The per-layer CV driver                                                     #
# --------------------------------------------------------------------------- #
def test_probe_auc_by_layer_finds_the_signal_layer(planted_xy):
    X, y = planted_xy([("easy", 0.0), ("hard", 6.0)], per_class=25,
                      n_layers=3, signal_layers=[1])
    res = probe_auc_by_layer(X, y, ["easy", "hard"], 5, np.random.default_rng(0))
    assert res["mode"] == "binary" and res["positive_label"] == "hard"
    auc = res["auc"]
    assert auc[1] > 0.95
    assert 0.2 < auc[0] < 0.8 and 0.2 < auc[2] < 0.8  # noise layers near chance


def test_probe_auc_by_layer_clamps_folds_and_modes(planted_xy):
    X, y = planted_xy([("easy", 0.0), ("hard", 2.5)], per_class=8, n_layers=1)
    res = probe_auc_by_layer(X, y, ["easy", "hard"], 20, np.random.default_rng(0))
    assert res["k"] == 8  # clamped to the smallest class

    X3, y3 = planted_xy([("easy", 0.0), ("medium", 1.5), ("hard", 3.0)],
                        per_class=10, n_layers=1)
    res3 = probe_auc_by_layer(X3, y3, ["easy", "medium", "hard"], 5,
                              np.random.default_rng(0))
    assert res3["mode"] == "ovr-macro" and res3["positive_label"] is None


def test_probe_auc_by_layer_degenerate_inputs(planted_xy):
    X, y = planted_xy([("easy", 0.0)], per_class=6, n_layers=1)
    assert probe_auc_by_layer(X, y, ["easy"], 5, np.random.default_rng(0)) is None

    X2, y2 = planted_xy([("easy", 0.0), ("hard", 2.0)], per_class=3, n_layers=1)
    y2[y2 == "hard"] = "easy"
    y2[0] = "hard"  # one lone positive -> CV impossible
    res = probe_auc_by_layer(X2, y2, ["easy", "hard"], 5, np.random.default_rng(0))
    assert res["skipped"] and "<2 samples" in res["reason"]


def test_probe_auc_by_layer_covariate_control_collapses_confound():
    rng = np.random.default_rng(12)
    n, dim = 60, 16
    c = rng.normal(size=n)
    u = rng.normal(size=dim)
    u /= np.linalg.norm(u)
    X = (rng.normal(size=(n, 1, dim)) * 0.05 + np.outer(c, u)[:, None, :]).astype(np.float32)
    y = np.where(c > np.median(c), "hard", "easy").astype(object)
    raw = probe_auc_by_layer(X, y, ["easy", "hard"], 5, np.random.default_rng(0))
    resid = probe_auc_by_layer(X, y, ["easy", "hard"], 5, np.random.default_rng(0),
                               covariate=c)
    assert raw["auc"][0] > 0.95
    assert 0.25 < resid["auc"][0] < 0.75
