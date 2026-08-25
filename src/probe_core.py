"""Pure-numpy probe-math core shared by every FATHOM activation-probe tier.

Tier 2 (``src/probe.py``), Tier 2b (``src/probe_controls.py``), Tier 2g
(``src/probe_generalization.py``) and Tier 2t (``src/probe_transfer.py``) all
measure the SAME thing -- a diagonal-LDA axis over root activations -- so the
math lives HERE, once, under public names: the axis fit / projection
(``fit_diag_lda`` / ``project_diag_lda`` / ``oof_diag_lda``), its scoring
metrics (``roc_auc``, ``balanced_acc``) and the per-layer cross-validated probe
driver that combines them (``probe_auc_by_layer``, Tier 2's headline + the
covariate control), the middle-class interpolation
fraction (``middle_positions``, Tier 2g/2t's ordinality metric) and the
informative-band mean that summarizes it (``mean_over_informative``),
stratified CV fold construction (``stratified_test_folds`` plus the extreme-pool
``extreme_pool_folds`` shared by Tier 2g and Tier 2's held-out axis), covariate
residualization (Tier 2b's controls), and the shared constants (``EPS``
variance floor, ``INFORMATIVE_AUC`` band threshold, ``REPO_ROOT``).

numpy + stdlib ONLY -- no matplotlib / torch / sklearn -- so the math layer
imports (and unit-tests) without a plotting stack. Tier modules import these
by public name; no module reaches into another module's underscore-privates.
"""

import os

import numpy as np

# This file is <repo_root>/src/probe_core.py, so the repo root is two levels
# up. Every analysis tier defaults its figures dir to <repo_root>/figures,
# regardless of the cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Variance floor: features whose train-fold std is below this are treated as
# constant (std set to 1) so they contribute ~nothing instead of blowing up.
EPS = 1e-6

# A per-layer AUC "counts" toward a fold-averaged summary only where it clears
# this -- the informative band Tier 2g (middle position) and Tier 2t (retention
# / position) average over. One definition so the band cannot drift.
INFORMATIVE_AUC = 0.70


# --------------------------------------------------------------------------- #
# Scoring metrics                                                             #
# --------------------------------------------------------------------------- #
def rankdata(a: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks (1-based), like scipy.stats.rankdata, in numpy."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0  # mean of the tied 1-based ranks
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, pos_mask: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U rank statistic (tie-aware). NaN if a class is empty."""
    pos_mask = np.asarray(pos_mask, dtype=bool)
    finite = np.isfinite(scores)
    scores = scores[finite]
    pos_mask = pos_mask[finite]
    n_pos = int(pos_mask.sum())
    n_neg = int(pos_mask.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balanced_acc(pos_mask: np.ndarray, pred_pos: np.ndarray) -> float:
    """Balanced accuracy = mean(TPR, TNR)."""
    pos_mask = np.asarray(pos_mask, dtype=bool)
    pred_pos = np.asarray(pred_pos, dtype=bool)
    tp = np.sum(pred_pos & pos_mask)
    fn = np.sum(~pred_pos & pos_mask)
    tn = np.sum(~pred_pos & ~pos_mask)
    fp = np.sum(pred_pos & ~pos_mask)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(np.nanmean([tpr, tnr]))


def middle_positions(scores: np.ndarray, low_idx, high_idx, mid_idx: dict) -> dict:
    """Interpolation fraction of each middle class on a low -> high score axis.

    Anchored on the extreme classes' mean scores:

        pos = (mean(scores[mid]) - mean(scores[low]))
              / (mean(scores[high]) - mean(scores[low]))

    0 -> sits on the low class, 1 -> on the high class, strictly in between ->
    interpolates (the ordinal-depth signal). THE single definition of the
    Tier 2g / Tier 2t headline ordinality metric: 2g feeds out-of-fold
    projections, 2t foreign-axis projections, and the position math is this,
    once. NaN scores are ignored (``nanmean``); if the extremes fail to
    separate (|denominator| <= ``EPS``) every position is NaN.

    Args:
        scores: scalar score per row ``[n]``.
        low_idx / high_idx: row indices of the low / high extreme class.
        mid_idx: middle label -> row indices (key order is preserved).

    Returns:
        dict middle label -> float position.
    """
    mean_low = float(np.nanmean(scores[low_idx]))
    mean_high = float(np.nanmean(scores[high_idx]))
    denom = mean_high - mean_low
    return {
        m: ((float(np.nanmean(scores[idx])) - mean_low) / denom
            if abs(denom) > EPS else float("nan"))
        for m, idx in mid_idx.items()
    }


def argmax_finite(values) -> int:
    """argmax ignoring NaNs; 0 if all NaN (keeps plotting/annotation safe)."""
    arr = np.asarray(values, dtype=float)
    return int(np.nanargmax(arr)) if np.any(np.isfinite(arr)) else 0


def mean_over_informative(values, informative_mask, fallback_mask=None) -> float:
    """``nanmean`` of ``values`` over the informative layers, with a fallback.

    ``informative_mask`` selects the layers that count (typically where the
    relevant AUC clears :data:`INFORMATIVE_AUC`). When NO layer qualifies, fall
    back to ``fallback_mask`` if given, else to EVERY layer. THE single
    definition of "average the metric only where the axis actually separates",
    shared by Tier 2g (each middle's interpolation position) and Tier 2t
    (retention and foreign-axis positions), so the band reduction -- and its
    empty-band fallback -- cannot drift between them.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(informative_mask, dtype=bool)
    if not np.any(mask):
        mask = (np.ones(values.shape, dtype=bool) if fallback_mask is None
                else np.asarray(fallback_mask, dtype=bool))
    return float(np.nanmean(values[mask]))


# --------------------------------------------------------------------------- #
# CV folds & covariate residualization                                        #
# --------------------------------------------------------------------------- #
def stratified_test_folds(pos_mask: np.ndarray, k: int, rng) -> list:
    """Stratified k-fold TEST-index split; each fold keeps the class ratio.

    Requires ``k <= min(#pos, #neg)`` (the caller guarantees this), so every
    train split still contains both classes.
    """
    pos = np.where(pos_mask)[0].copy()
    neg = np.where(~pos_mask)[0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    pf = np.array_split(pos, k)
    nf = np.array_split(neg, k)
    return [np.concatenate([pf[i], nf[i]]) for i in range(k)]


def extreme_pool_folds(y, low, high, middles, k: int, rng):
    """Stratified k-fold TEST split over the EXTREME pool (low + high classes).

    The shared INPUT side of the "train on the extremes, project the held-out
    middle" protocol (whose math side is :func:`extreme_axis_scores`): row
    indices of the low / high extreme classes and of each middle class, then
    ``k`` stratified test folds drawn from the extreme pool ONLY (each fold
    keeps the low:high ratio), plus the full extreme-pool index. Middle rows are
    NEVER placed in a fold -- they are projected as always-out-of-sample
    downstream.

    Shared by Tier 2g (``interpolation_by_layer``) and Tier 2's diff-projection
    scatter (``diff_projection_by_layer``) so the fold construction -- and the
    RNG it consumes (``shuffle(low)`` then ``shuffle(high)``) -- cannot drift
    between them. The caller computes ``k`` and guards the <2-per-extreme case,
    so each tier keeps its own skip message.

    Returns:
        ``(low_idx, high_idx, mid_idx, test_folds, extreme_all)``.
    """
    low_idx = np.where(y == low)[0]
    high_idx = np.where(y == high)[0]
    mid_idx = {m: np.where(y == m)[0] for m in middles}
    lo, hi = low_idx.copy(), high_idx.copy()
    rng.shuffle(lo)
    rng.shuffle(hi)
    lo_f, hi_f = np.array_split(lo, k), np.array_split(hi, k)
    test_folds = [np.concatenate([lo_f[i], hi_f[i]]) for i in range(k)]
    extreme_all = np.concatenate([low_idx, high_idx])
    return low_idx, high_idx, mid_idx, test_folds, extreme_all


def residualize(Xtr, Ctr, Xte, Cte):
    """Linear-regress X on one or more covariates (fit on TRAIN), return residuals.

    ``Ctr`` / ``Cte`` may be 1-D ``[n]`` (a single covariate, e.g. question length)
    or 2-D ``[n, k]`` (a surface-feature matrix -- Tier 2b's multi-covariate
    control). Covariates are z-scored on the train split (numerical conditioning;
    residuals are invariant to it) and an intercept is added; the OLS fit is
    computed on TRAIN ONLY (no leak) and subtracted from both splits. This asks
    whether class signal survives after removing everything the covariates
    linearly explain. ``lstsq`` tolerates collinear covariates (min-norm solution).
    """
    Ctr = np.asarray(Ctr, dtype=np.float64)
    Cte = np.asarray(Cte, dtype=np.float64)
    if Ctr.ndim == 1:
        Ctr = Ctr[:, None]
        Cte = Cte[:, None]

    cmean = Ctr.mean(axis=0)
    cstd = Ctr.std(axis=0)
    cstd[cstd < EPS] = 1.0  # neutralize ~constant covariates
    Dtr = np.column_stack([np.ones(len(Ctr)), (Ctr - cmean) / cstd])  # [n_tr, k+1]
    Dte = np.column_stack([np.ones(len(Cte)), (Cte - cmean) / cstd])  # [n_te, k+1]

    coef, *_ = np.linalg.lstsq(Dtr, Xtr.astype(np.float64), rcond=None)  # [k+1, d]
    return Xtr - Dtr @ coef, Xte - Dte @ coef


# --------------------------------------------------------------------------- #
# The diagonal-LDA axis                                                       #
# --------------------------------------------------------------------------- #
def fit_diag_lda(Xtr: np.ndarray, pos_mask: np.ndarray):
    """Fit the diagonal-LDA axis on one train split. THE single source of truth.

    Standardize each feature on the train split (stds below ``EPS`` are set to
    1 so ~constant features contribute ~nothing instead of blowing up), then
    take the difference of class means as the direction ``w`` -- oriented
    negative -> positive class. The decision threshold ``thr`` is the midpoint
    of the two train-class mean projections.

    Every axis in FATHOM is fit here -- the Tier-2 CV probe (via
    ``oof_diag_lda``), Tier 2g's extremes-only axis and Tier 2t's
    cross-dataset transfer axis -- so the probe definition cannot drift
    between tiers.

    Args:
        Xtr: train rows ``[n_train, n_features]``.
        pos_mask: bool ``[n_train]``; True = positive (high / hard) class.

    Returns:
        ``(mu, sd, w, thr)``: per-feature train mean and (floored) std, the
        axis ``[n_features]``, and the scalar midpoint threshold.
    """
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < EPS] = 1.0  # neutralize ~constant features instead of dividing by ~0
    Xtr_s = (Xtr - mu) / sd

    w = Xtr_s[pos_mask].mean(axis=0) - Xtr_s[~pos_mask].mean(axis=0)
    s_tr = Xtr_s @ w
    thr = 0.5 * (s_tr[pos_mask].mean() + s_tr[~pos_mask].mean())
    return mu, sd, w, thr


def project_diag_lda(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Score rows on a fitted axis: standardize with the TRAIN ``mu``/``sd``
    (never the scored split's own), then project onto ``w`` -> scores ``[n]``."""
    return ((X - mu) / sd) @ w


def oof_diag_lda(Xl: np.ndarray, pos_mask: np.ndarray, folds: list, covariate=None):
    """Out-of-fold diagonal-LDA scores + hard predictions for ONE layer.

    For each fold: (optionally regress out a scalar ``covariate`` fit on the train
    split — Tier 2b's length control), fit the axis on the train split via
    :func:`fit_diag_lda`, then score the held-out rows. ``covariate=None``
    (the default, used by Tier 2) reproduces the plain probe exactly.
    """
    n = Xl.shape[0]
    scores = np.full(n, np.nan)
    preds = np.zeros(n, dtype=bool)
    for test_idx in folds:
        train = np.ones(n, dtype=bool)
        train[test_idx] = False
        Xtr = Xl[train]
        Xte = Xl[test_idx]
        if covariate is not None:
            Xtr, Xte = residualize(Xtr, covariate[train], Xte, covariate[test_idx])

        mu, sd, w, thr = fit_diag_lda(Xtr, pos_mask[train])
        s_te = project_diag_lda(Xte, mu, sd, w)
        scores[test_idx] = s_te
        preds[test_idx] = s_te >= thr
    return scores, preds


def extreme_axis_scores(Xl, y, high, mid_idx: dict, test_folds: list, extreme_all):
    """Per-row scores on the low -> high axis trained on the EXTREME pool only.

    The "train on the extremes, generalize to the held-out middle" protocol, for
    ONE layer. For each extreme test fold: fit the oriented (low -> high)
    diagonal-LDA axis on the extreme-train split (:func:`fit_diag_lda`), then
    project the held-out extreme rows (their single out-of-fold score) and EVERY
    middle row. Middle rows -- never in any training split -- get their
    projection averaged over all folds.

    Returns a full-length ``scores`` vector ``[n]``: extreme rows carry their
    out-of-fold score, middle rows the fold-averaged projection, any row in
    neither pool stays NaN. THE single definition of the extreme-trained axis,
    shared by Tier 2g (its interpolation fraction reads the middle vs the
    extremes) and Tier 2's diff-projection scatter (it plots this as the x-axis),
    so the held-out axis cannot drift between them.

    Args:
        Xl: rows at one layer ``[n, n_features]``.
        y: object label array ``[n]`` (only ``== high`` is taken).
        high: the high/positive extreme label (orients the axis low -> high).
        mid_idx: middle label -> row indices (held-out every fold).
        test_folds: extreme-pool test-index folds (low + high rows only).
        extreme_all: row indices of the extreme pool (low + high).
    """
    n = Xl.shape[0]
    scores = np.full(n, np.nan)
    mid_acc = {m: np.zeros(len(idx)) for m, idx in mid_idx.items()}  # summed over folds
    for test_idx in test_folds:
        train_idx = np.setdiff1d(extreme_all, test_idx)
        mu, sd, w, _thr = fit_diag_lda(Xl[train_idx], y[train_idx] == high)
        scores[test_idx] = project_diag_lda(Xl[test_idx], mu, sd, w)
        for m, idx in mid_idx.items():
            mid_acc[m] += project_diag_lda(Xl[idx], mu, sd, w)
    for m, idx in mid_idx.items():
        scores[idx] = mid_acc[m] / len(test_folds)  # per-row mean over folds
    return scores


# --------------------------------------------------------------------------- #
# Per-layer cross-validated probe driver                                      #
# --------------------------------------------------------------------------- #
def probe_auc_by_layer(X: np.ndarray, y, labels_ordered, k_req: int, rng, covariate=None):
    """Cross-validated linear-probe AUC (+ balanced acc) for every layer.

    Binary -> one task (positive = the LAST / hardest ordered label). >2 labels
    -> one-vs-rest per class, macro-averaged. Returns ``None`` if <2 labels, or a
    ``{"skipped": True, "reason": ...}`` dict if a class has <2 samples (CV
    impossible); otherwise a dict with per-layer ``auc`` / ``balanced_acc`` lists.

    ``covariate`` (optional, shape ``[n]``) is regressed out of every feature
    within each train fold before probing — Tier 2b's length control. ``None``
    (default) is the plain probe.

    THE single per-layer probe driver: Tier 2 (``src/probe.py``) plots it, Tier 2b
    (``src/probe_controls.py``) reuses it for the residualized controls, and the
    transfer tiers restrict it to a class pair -- so the CV protocol cannot drift
    between them.
    """
    n, n_layers, _ = X.shape
    y = np.asarray(y, dtype=object)
    classes = list(labels_ordered)
    if len(classes) < 2:
        return None

    counts = {c: int(np.sum(y == c)) for c in classes}
    min_count = min(counts.values())
    if min_count < 2:
        return {"skipped": True, "reason": f"a class has <2 samples ({counts}); CV impossible"}
    k = max(2, min(k_req, min_count))

    if len(classes) == 2:
        tasks = [(classes[-1], y == classes[-1])]  # positive = hardest label
        mode = "binary"
    else:
        tasks = [(c, y == c) for c in classes]
        mode = "ovr-macro"

    per_task_auc, per_task_bacc = [], []
    for _label, pmask in tasks:
        folds = stratified_test_folds(pmask, k, rng)
        aucs, baccs = [], []
        for L in range(n_layers):
            scores, preds = oof_diag_lda(X[:, L, :], pmask, folds, covariate)
            aucs.append(roc_auc(scores, pmask))
            baccs.append(balanced_acc(pmask, preds))
        per_task_auc.append(aucs)
        per_task_bacc.append(baccs)

    auc = np.nanmean(np.asarray(per_task_auc), axis=0)
    bacc = np.nanmean(np.asarray(per_task_bacc), axis=0)
    return {
        "skipped": False,
        "mode": mode,
        "k": k,
        "positive_label": (tasks[0][0] if mode == "binary" else None),
        "classes": classes,
        "counts": counts,
        "auc": auc.tolist(),
        "balanced_acc": bacc.tolist(),
    }
