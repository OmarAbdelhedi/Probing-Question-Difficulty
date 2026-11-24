"""Tier 2: is difficulty linearly decodable from the question, and at which layer?

Paper Sec. III-C / IV-B (Fig. 1, Table I "Layer" and "Probe AUC") and Fig. 5.
Writes to the figures directory:

  tier2_probe_auc_by_layer.png        cross-validated probe AUC and balanced
                                      accuracy per layer (Fig. 1)
  tier2_diff_projection_by_layer.png  questions projected on the held-out
                                      low -> high axis at six layers (Fig. 5)
  tier2_centroid_separation_by_layer.png, tier2_pca_scatter_by_layer.png,
  tier2_activation_norm_by_layer.png  supplementary diagnostics (not in the paper)
  tier2_summary.json                  every per-layer number behind the plots

The probe is the diagonal-LDA probe of ``src/probe_core.py``; AUC is computed
on out-of-fold scores pooled across the folds. numpy + matplotlib only.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless backend; MUST precede the pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.probe_core import (  # noqa: E402
    EPS,
    argmax_finite,
    extreme_axis_scores,
    extreme_pool_folds,
    oof_diag_lda,
    probe_auc_by_layer,
    stratified_test_folds,
)
from src.run_data import load_run, resolve_plots_dir  # noqa: E402


# --------------------------------------------------------------------------- #
# Per-layer probe drivers (the math itself lives in src/probe_core.py)        #
# --------------------------------------------------------------------------- #
def centroid_separation_by_layer(X: np.ndarray, y, labels_ordered) -> list:
    """RMS per-feature standardized centroid distance per layer (descriptive).

    For 2 classes this is the diagonal-Mahalanobis distance between centroids
    (scaled to a per-feature RMS so it is comparable across layers); for >2
    classes it is the mean over all class pairs. Standardization uses ALL data
    (this is a descriptive statistic, not a generalization estimate).
    """
    n, n_layers, _ = X.shape
    y = np.asarray(y, dtype=object)
    classes = list(labels_ordered)
    seps = []
    for L in range(n_layers):
        Xl = X[:, L, :]
        sd = Xl.std(axis=0)
        sd[sd < EPS] = 1.0
        means = {c: Xl[y == c].mean(axis=0) for c in classes}
        dists = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                d = (means[classes[i]] - means[classes[j]]) / sd
                dists.append(float(np.sqrt(np.mean(d * d))))
        seps.append(float(np.mean(dists)) if dists else float("nan"))
    return seps


def diff_projection_by_layer(X, y, labels_ordered, panel_layers, k_req, rng, max_points: int = 1500):
    """Held-out supervised "diff" projection coordinates, per panel layer.

    The companion to the unsupervised PCA scatter. That one views the data along
    its top *total-variance* axes (now of the layer-to-layer diff) -- which in
    hidden states are still dominated by label-irrelevant structure (norm, rogue
    dims, topic), so classes can look unseparated at the very layer the probe
    decodes them. This views them along the *class-difference* axis instead:

      * x = HELD-OUT diff-of-means score (== the rank-1 PCA of the across-class
        difference vectors). Binary: out-of-fold diagonal-LDA score, positive =
        hardest label. >=3 ordered classes: the Tier-2g protocol via
        :func:`extreme_axis_scores` -- the low -> high axis is trained on the
        EXTREME pool only and the middle class(es) are projected as
        always-out-of-sample, so the middle landing between the extremes along x
        is the geometric form of Tier 2g's interpolation fraction.
      * y = leading PC of the data after the diff axis is projected out -- shown
        purely for vertical spread. It is not optimized for class separation and
        carries no held-out guarantee; the figure's honesty rests entirely on x
        never being fit on the point it scores (which matters here, where a
        diff-of-means on shuffled labels already looks separated at p >> n).

    Returns ``None`` (<2 labels), a ``{"skipped": True, "reason": ...}`` dict (a
    class too small to CV), else a dict with ``mode`` / ``k`` / ``counts`` /
    ``classes`` / ``sel`` (rows kept after subsampling) / ``xlabel`` and
    ``coords`` (panel layer -> ``{"x": [m], "y": [m]}`` over ``sel``).
    """
    n, _, _ = X.shape
    y = np.asarray(y, dtype=object)
    classes = list(labels_ordered)
    if len(classes) < 2:
        return None
    counts = {c: int(np.sum(y == c)) for c in classes}
    low, high = classes[0], classes[-1]

    # Subsample rows once to bound the residual-PC SVD (mirrors plot_pca_diff_scatter).
    sel = (np.sort(rng.choice(n, size=max_points, replace=False))
           if n > max_points else np.arange(n))

    # ---- primary (x) axis: the held-out diff-of-means score, full length. ----
    if len(classes) == 2:
        if min(counts.values()) < 2:
            return {"skipped": True, "reason": f"a class has <2 samples ({counts}); CV impossible"}
        k = max(2, min(k_req, min(counts.values())))
        pos_mask = (y == high)
        folds = stratified_test_folds(pos_mask, k, rng)
        x_by_layer = {L: oof_diag_lda(X[:, L, :], pos_mask, folds)[0] for L in panel_layers}
        mode = "binary"
        xlabel = f"held-out diff-of-means score   ({low}  →  {high})"
    else:
        if min(counts[low], counts[high]) < 2:
            return {"skipped": True, "reason": f"extreme classes too small ({counts}); CV impossible"}
        k = max(2, min(k_req, counts[low], counts[high]))
        _, _, mid_idx, test_folds, extreme_all = extreme_pool_folds(
            y, low, high, classes[1:-1], k, rng)
        x_by_layer = {L: extreme_axis_scores(X[:, L, :], y, high, mid_idx, test_folds, extreme_all)
                      for L in panel_layers}
        mode = "ordinal-extremes"
        xlabel = f"held-out {low} → {high} axis   (middle projected out-of-sample)"

    # ---- secondary (y) axis: leading residual PC, for vertical spread only. ----
    pos_sel, neg_sel = (y[sel] == high), (y[sel] == low)
    coords = {}
    for L in panel_layers:
        Z = X[sel, L, :].astype(np.float64)
        mu, sd = Z.mean(0), Z.std(0)
        sd[sd < EPS] = 1.0
        Z = (Z - mu) / sd
        if pos_sel.any() and neg_sel.any():  # project out the class-diff axis so y is complementary
            wdir = Z[pos_sel].mean(0) - Z[neg_sel].mean(0)
            nrm = np.linalg.norm(wdir)
            if nrm > EPS:
                wdir = wdir / nrm
                Z = Z - np.outer(Z @ wdir, wdir)
        Z = Z - Z.mean(0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            yc = Z @ Vt[0]
        except np.linalg.LinAlgError:
            yc = np.zeros(len(sel))
        coords[L] = {"x": np.asarray(x_by_layer[L], dtype=float)[sel], "y": yc}

    return {"skipped": False, "mode": mode, "k": k, "counts": counts, "classes": classes,
            "sel": sel, "coords": coords, "xlabel": xlabel}


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _panel_layers(n_layers: int, n_panels: int = 6) -> list:
    """Evenly-spaced indices to show as scatter panels (first + last always
    included). Used by the diff-projection scatter (over ``n_layers``) and the
    PCA scatter (over its ``n_layers-1`` layer transitions)."""
    if n_layers <= n_panels:
        return list(range(n_layers))
    return sorted(set(int(round(t)) for t in np.linspace(0, n_layers - 1, n_panels)))


def plot_probe_auc(layer_indices, result: dict, n: int, out_path: str) -> None:
    """Probe AUC (+ balanced accuracy) vs layer (paper Fig. 1)."""
    x = np.asarray(layer_indices)
    auc = np.asarray(result["auc"], dtype=float)
    bacc = np.asarray(result["balanced_acc"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    ax.plot(x, auc, marker="o", ms=4, lw=1.9, color="#C44E52", label="probe AUC")
    ax.plot(x, bacc, marker="s", ms=3, lw=1.2, color="#4C72B0", alpha=0.85, label="balanced acc.")

    best = argmax_finite(auc)
    ax.scatter([x[best]], [auc[best]], s=95, facecolor="none", edgecolor="black", zorder=5)
    ax.annotate(
        f"best: layer {x[best]}, AUC={auc[best]:.2f}",
        (x[best], auc[best]), textcoords="offset points", xytext=(6, -26), fontsize=8,
    )

    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Out-of-fold score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_centroid_separation(layer_indices, sep, classes, out_path: str) -> None:
    """CV-free companion: standardized centroid separation vs layer."""
    x = np.asarray(layer_indices)
    sep = np.asarray(sep, dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(x, sep, marker="o", ms=4, lw=1.9, color="#55A868")
    best = argmax_finite(sep)
    ax.scatter([x[best]], [sep[best]], s=95, facecolor="none", edgecolor="black", zorder=5)
    ax.annotate(f"peak: layer {x[best]}", (x[best], sep[best]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Centroid separation (std units)")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_diff_scatter(X, y, labels_ordered, label_colors, layer_indices, out_path, rng,
                          max_points: int = 1500, n_panels: int = 6) -> None:
    """Small-multiples 2D *unsupervised* PCA scatter of the LAYER-TO-LAYER diff.

    Instead of PCA on the raw root vectors at each layer -- whose top PCs track
    total variance (norm / topic / rogue dims), mostly label-irrelevant, so the
    classes can look unseparated at the very layer the probe decodes them -- this
    PCAs the residual-stream UPDATE each layer applies:

        D[:, k, :] = X[:, k+1, :] - X[:, k, :]    (transition layer_k -> layer_{k+1})

    centered and SVD'd per transition, colored by label. Subtracting the previous
    layer removes the large component carried unchanged across layers, so each
    panel's PCs reflect what that layer *adds* -- a cleaner, still label-blind
    look at whether the classes pull apart as the representation is built up.

    Needs >=3 questions and >=2 layers (one transition); raises otherwise so the
    caller logs a SKIP rather than emitting an empty figure.
    """
    n, n_layers, _ = X.shape
    if n < 3:
        raise ValueError(f"PCA scatter needs >=3 questions, have {n}")
    if n_layers < 2:
        raise ValueError(f"diff PCA scatter needs >=2 layers, have {n_layers}")

    if n > max_points:  # bound SVD cost on big runs; reproducible subsample
        sel = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        sel = np.arange(n)
    ys = np.asarray(y, dtype=object)[sel]

    D = np.diff(X[sel], axis=1)  # [m, n_layers-1, hidden]: D[:,k]=act[k+1]-act[k]
    n_trans = D.shape[1]
    panel_trans = _panel_layers(n_trans, n_panels)  # indices into the transition axis

    ncols = min(3, len(panel_trans))
    nrows = int(np.ceil(len(panel_trans) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, k in enumerate(panel_trans):
        ax = axes.flat[idx]
        ax.set_visible(True)
        Dc = D[:, k, :].astype(np.float64)
        Dc = Dc - Dc.mean(axis=0, keepdims=True)  # center (standard PCA on the diff)
        _, S, Vt = np.linalg.svd(Dc, full_matrices=False)
        pcs = Dc @ Vt[:2].T  # [m, 2]
        var = S ** 2
        ev = (var[:2] / var.sum()) if var.sum() > 0 else np.zeros(2)
        for lab in labels_ordered:
            m = ys == lab
            if np.any(m):
                ax.scatter(pcs[m, 0], pcs[m, 1], s=18, alpha=0.75, color=label_colors[lab],
                           edgecolor="white", linewidth=0.3)
        ax.set_title(f"layer {layer_indices[k]}→{layer_indices[k + 1]}  "
                     f"(PC1 {ev[0] * 100:.0f}% / PC2 {ev[1] * 100:.0f}%)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [Line2D([0], [0], marker="o", ls="", color=label_colors[lab], label=str(lab))
               for lab in labels_ordered]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(labels_ordered), 6),
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_diff_projection(result, y, label_colors, layer_indices, out_path) -> None:
    """Small-multiples diff-projection scatter: held-out diff axis (x) vs a
    residual PC (y), at the same layers as the PCA scatter, colored by label.

    Faint vertical guides mark each class's mean x; for an ordinal scheme the
    middle class's guide sitting between the extremes is Tier 2g's interpolation,
    drawn. ``result`` is :func:`diff_projection_by_layer`'s output."""
    classes = result["classes"]
    sel = result["sel"]
    ys = np.asarray(y, dtype=object)[sel]
    coords = result["coords"]
    panel_layers = list(coords.keys())

    ncols = min(3, len(panel_layers))
    nrows = int(np.ceil(len(panel_layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, L in enumerate(panel_layers):
        ax = axes.flat[idx]
        ax.set_visible(True)
        xs, yc = coords[L]["x"], coords[L]["y"]
        for lab in classes:  # class-mean x guides (the anchors the middle sits between)
            m = ys == lab
            if np.any(m):
                ax.axvline(float(np.mean(xs[m])), color=label_colors[lab], lw=0.9, alpha=0.35, zorder=1)
        for lab in classes:
            m = ys == lab
            if np.any(m):
                ax.scatter(xs[m], yc[m], s=18, alpha=0.75, color=label_colors[lab],
                           edgecolor="white", linewidth=0.3, zorder=2)
        ax.set_title(f"layer {layer_indices[L]}", fontsize=9)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)

    handles = [Line2D([0], [0], marker="o", ls="", color=label_colors[lab], label=str(lab))
               for lab in classes]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=min(len(classes), 6), fontsize=9, frameon=False)
    fig.text(0.5, 0.055, result["xlabel"] + "       (y: residual PC — vertical spread only)",
             ha="center", fontsize=9, color="0.3")  # x-axis caption sits ABOVE the legend (no overlap)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_activation_norm(X, y, labels_ordered, label_colors, layer_indices, out_path) -> None:
    """Mean L2 norm of the root vector per layer, split by label (±SEM)."""
    norms = np.linalg.norm(X, axis=2)  # [n, n_layers]
    x = np.asarray(layer_indices)
    ys = np.asarray(y, dtype=object)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for lab in labels_ordered:
        m = ys == lab
        if not np.any(m):
            continue
        sub = norms[m]
        mean = sub.mean(axis=0)
        sem = sub.std(axis=0) / np.sqrt(sub.shape[0]) if sub.shape[0] > 1 else np.zeros(sub.shape[1])
        ax.plot(x, mean, marker="o", ms=3, lw=1.8, color=label_colors[lab], label=f"{lab} (n={int(m.sum())})")
        ax.fill_between(x, mean - sem, mean + sem, color=label_colors[lab], alpha=0.18)

    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Mean L2 norm of root activation")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run(output_dir: str, plots_dir: str | None = None, folds: int = 5, seed: int = 0) -> None:
    """Build the Tier-2 artifacts for a finished run.

    Args:
        output_dir: run directory passed to ``main.py --output``. If it has no
            usable activations, Tier 2 is skipped with a message.
        plots_dir: output dir (default ``<output_dir>/figures``, created if missing).
        folds: requested CV folds (clamped to the smallest class count).
        seed: RNG seed for fold shuffling and PCA subsampling.
    """
    try:
        data = load_run(output_dir, log_tag="probe")
    except FileNotFoundError as exc:
        print(f"[probe] SKIP Tier 2: {exc}")
        return
    plots_dir = resolve_plots_dir(plots_dir, output_dir)

    X, y, labels_ordered, layer_indices = data.X, data.y, data.labels, data.layer_indices
    n, n_layers, hidden = X.shape
    cmap = plt.get_cmap("tab10")
    label_colors = {lab: cmap(i % 10) for i, lab in enumerate(labels_ordered)}

    rng = np.random.default_rng(seed)
    summary = {
        "metadata_path": data.metadata_path,
        "activations_dir": data.activations_dir,
        "n_questions": n,
        "n_layers": n_layers,
        "hidden_dim": hidden,
        "layer_indices": layer_indices,
        "labels": labels_ordered,
        "n_skipped_missing": len(data.missing),
        "plots_dir": plots_dir,
        "plots": [],
    }

    def _emit(name: str, fn) -> None:
        """Render one figure, logging (not crashing) if a single plot fails."""
        path = os.path.join(plots_dir, name)
        try:
            fn(path)
            summary["plots"].append(os.path.basename(path))
            print(f"[probe]   wrote {path}")
        except Exception as exc:  # one bad plot must not lose the others / the json
            print(f"[probe]   WARNING: skipped {name}: {type(exc).__name__}: {exc}")

    # 1) Probe AUC by layer (Fig. 1).
    probe = probe_auc_by_layer(X, y, labels_ordered, folds, rng)
    if probe is None:
        print("[probe]   SKIP probe AUC: need >=2 labels.")
    elif probe.get("skipped"):
        print(f"[probe]   SKIP probe AUC: {probe['reason']}")
    else:
        best = argmax_finite(probe["auc"])
        summary["probe"] = {
            **{k: probe[k] for k in
               ("mode", "k", "positive_label", "classes", "counts", "auc", "balanced_acc")},
            "best_layer_index": layer_indices[best],
            "best_auc": probe["auc"][best],
        }
        _emit("tier2_probe_auc_by_layer.png", lambda p: plot_probe_auc(layer_indices, probe, n, p))

    # 2) Centroid separation by layer (needs >=2 labels).
    if len(labels_ordered) >= 2:
        sep = centroid_separation_by_layer(X, y, labels_ordered)
        summary["centroid_separation"] = sep
        summary["centroid_separation_best_layer_index"] = layer_indices[argmax_finite(sep)]
        _emit("tier2_centroid_separation_by_layer.png",
              lambda p: plot_centroid_separation(layer_indices, sep, labels_ordered, p))
    else:
        print("[probe]   SKIP centroid separation: need >=2 labels.")

    # 3) PCA scatter of the layer-to-layer diff (needs >=3 questions, >=2 layers).
    if n >= 3 and n_layers >= 2:
        _emit("tier2_pca_scatter_by_layer.png",
              lambda p: plot_pca_diff_scatter(X, y, labels_ordered, label_colors, layer_indices, p, rng))
    else:
        print(f"[probe]   SKIP PCA scatter: need >=3 questions and >=2 layers "
              f"(n={n}, n_layers={n_layers}).")

    # 3b) Diff-projection scatter — the SUPERVISED companion to (3): view the
    #     data along the held-out class-difference axis instead of the top
    #     unsupervised PCs, so separation appears where the probe says it should.
    if n >= 3 and len(labels_ordered) >= 2:
        panels = _panel_layers(n_layers)
        diffproj = diff_projection_by_layer(X, y, labels_ordered, panels, folds, rng)
        if diffproj is None:
            print("[probe]   SKIP diff projection: need >=2 labels.")
        elif diffproj.get("skipped"):
            print(f"[probe]   SKIP diff projection: {diffproj['reason']}")
        else:
            summary["diff_projection"] = {
                "mode": diffproj["mode"], "k": diffproj["k"],
                "panel_layer_indices": [layer_indices[L] for L in panels],
                "n_shown": int(len(diffproj["sel"])),
            }
            _emit("tier2_diff_projection_by_layer.png",
                  lambda p: plot_diff_projection(diffproj, y, label_colors, layer_indices, p))
    else:
        print(f"[probe]   SKIP diff projection: need >=3 questions and >=2 labels (n={n}).")

    # 4) Activation norm by layer (always works, even single-label).
    _emit("tier2_activation_norm_by_layer.png",
          lambda p: plot_activation_norm(X, y, labels_ordered, label_colors, layer_indices, p))

    summary_path = os.path.join(plots_dir, "tier2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[probe]   wrote {summary_path}")

    if "probe" in summary:
        print(f"[probe] best layer for difficulty decodability: layer "
              f"{summary['probe']['best_layer_index']} (AUC={summary['probe']['best_auc']:.3f})")
    print("[probe] DONE.")
