"""Tier-2g GENERALIZATION-ACROSS-DEPTH test for FATHOM.

Tier 2b showed the hop-count signal survives surface controls. The remaining
worry: the probe might be memorizing class-specific templates/topics rather than
a genuine *ordinal depth* axis. This test settles it.

Train the diagonal-LDA probe on ONLY the EXTREME classes (lowest vs highest hop
count, e.g. 2hop vs 4hop), then project the HELD-OUT middle class(es) (3hop --
never seen in training) onto that learned axis. If the middle lands *between* the
extremes, the model places "3 hops" between "2" and "4" without being told 3hop
exists -- evidence of a real ordinal depth representation, not class memorization.

Per layer, k-fold over the extreme pool: train the oriented (high - low) direction
on the extreme-train split, project the extreme-test split (-> honest held-out
low-vs-high AUC) and ALL middle rows. The headline metric is the INTERPOLATION
FRACTION of each middle class:

    pos = (mean_proj[middle] - mean_proj[low]) / (mean_proj[high] - mean_proj[low])

  pos = 0 -> looks like the low class; pos = 1 -> looks like the high class;
  0 < pos < 1 -> interpolates (ordinal); pos ~ 0.5 -> right in the middle.

numpy + matplotlib only; reuses the shared probe core (``src.probe_core``).
Requires >=3 ordered classes (a graded label scheme, e.g. MuSiQue hops_multiclass);
skips gracefully otherwise.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless backend; MUST precede the pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.probe_core import (  # noqa: E402
    INFORMATIVE_AUC,
    argmax_finite,
    extreme_axis_scores,
    extreme_pool_folds,
    mean_over_informative,
    middle_positions,
    roc_auc,
)
from src.run_data import load_run, resolve_plots_dir  # noqa: E402


def interpolation_by_layer(X, y, labels_ordered, k_req, rng):
    """Train on extreme classes, test whether middle class(es) interpolate.

    Returns ``None`` if <3 ordered classes, a ``{"skipped": True, ...}`` dict if
    an extreme class has <2 samples, else a dict with per-layer ``auc_2v4`` and,
    per middle class, the per-layer ``interp`` (interpolation fraction).
    """
    n_layers = X.shape[1]
    y = np.asarray(y, dtype=object)
    classes = list(labels_ordered)
    if len(classes) < 3:
        return None

    low, high = classes[0], classes[-1]
    middles = classes[1:-1]
    counts = {c: int(np.sum(y == c)) for c in classes}
    if min(counts[low], counts[high]) < 2:
        return {"skipped": True, "reason": f"extreme classes too small ({counts})"}
    k = max(2, min(k_req, counts[low], counts[high]))

    # Stratified k-fold over the EXTREME pool only (low + high); the middle
    # rows are held out of every fold (shared with Tier 2's diff-projection
    # scatter so the fold construction cannot drift).
    low_idx, high_idx, mid_idx, test_folds, extreme_all = extreme_pool_folds(
        y, low, high, middles, k, rng)

    auc_2v4 = []
    interp = {m: [] for m in middles}
    for L in range(n_layers):
        # Held-out low -> high axis: trained on the extreme pool only, middle
        # rows projected as always-out-of-sample (fold-averaged). Defined once
        # in src.probe_core, shared with Tier 2's diff-projection scatter.
        scores = extreme_axis_scores(X[:, L, :], y, high, mid_idx, test_folds, extreme_all)
        # AUC over the extreme rows only (the low-vs-high task excludes middles).
        auc_2v4.append(roc_auc(scores[extreme_all], y[extreme_all] == high))
        for m, p in middle_positions(scores, low_idx, high_idx, mid_idx).items():
            interp[m].append(p)

    return {
        "skipped": False,
        "low": low, "high": high, "middles": middles, "counts": counts, "k": k,
        "auc_2v4": [float(a) for a in auc_2v4],
        "interp": {m: [float(v) for v in interp[m]] for m in middles},
    }


def plot_generalization(layer_indices, result, out_path) -> None:
    """1x2: held-out low-vs-high AUC, and the middle class's position by layer."""
    x = np.asarray(layer_indices)
    auc = np.asarray(result["auc_2v4"], dtype=float)
    low, high = result["low"], result["high"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax0 = axes[0]
    ax0.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    ax0.plot(x, auc, marker="o", ms=4, lw=1.9, color="#C44E52",
             label=f"{low} vs {high} AUC (held-out)")
    ax0.set_ylim(0.0, 1.02)
    ax0.set_xlabel("Layer (0 = embeddings)")
    ax0.set_ylabel("Held-out AUC")
    ax0.grid(alpha=0.25)
    ax0.set_axisbelow(True)
    ax0.legend(loc="lower right", fontsize=8)

    ax1 = axes[1]
    ax1.axhspan(0.0, 1.0, color="#55A868", alpha=0.10, label="interpolation zone [0, 1]")
    ax1.axhline(0.5, color="0.5", ls="--", lw=1, label="ideal midpoint (0.5)")
    ax1.axhline(0.0, color="0.7", lw=0.8)
    ax1.axhline(1.0, color="0.7", lw=0.8)
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(result["middles"]):
        ax1.plot(x, np.asarray(result["interp"][m], dtype=float), marker="^", ms=4, lw=1.9,
                 color=cmap(i % 10), label=f"{m} position")
    ax1.set_ylim(-0.4, 1.4)
    ax1.set_xlabel("Layer (0 = embeddings)")
    ax1.set_ylabel(f"Position on {low} → {high} axis")
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(output_dir: str, metadata_path: str | None = None, activations_dir: str | None = None,
        plots_dir: str | None = None, folds: int = 5, seed: int = 0) -> None:
    """Build the Tier-2g generalization-across-depth artifact.

    Args mirror :func:`src.probe.run`. Writes ``tier2g_generalization_by_layer.png``
    and ``tier2g_summary.json`` to ``plots_dir`` (default ``<repo_root>/figures``).
    Needs >=3 ordered classes (a graded label scheme); skips gracefully otherwise.
    """
    plots_dir = resolve_plots_dir(plots_dir)
    try:
        data = load_run(output_dir, metadata_path, activations_dir, log_tag="probe2g")
    except FileNotFoundError as exc:
        print(f"[probe2g] SKIP Tier 2g: {exc}")
        return

    X, y, labels_ordered, layer_indices = data.X, data.y, data.labels, data.layer_indices
    if len(labels_ordered) < 3:
        print(f"[probe2g] SKIP: need >=3 ordered classes for a held-out middle "
              f"(have {labels_ordered}); use a graded label scheme (e.g. hops_multiclass).")
        return

    rng = np.random.default_rng(seed)
    result = interpolation_by_layer(X, y, labels_ordered, folds, rng)
    if result is None:
        print("[probe2g] SKIP: <3 classes.")
        return
    if result.get("skipped"):
        print(f"[probe2g] SKIP: {result['reason']}")
        return

    path = os.path.join(plots_dir, "tier2g_generalization_by_layer.png")
    plot_generalization(layer_indices, result, path)
    print(f"[probe2g]   wrote {path}")

    auc = np.asarray(result["auc_2v4"], dtype=float)
    informative = auc > INFORMATIVE_AUC  # layers where the extreme axis actually separates
    best = argmax_finite(auc)

    positions = {
        m: mean_over_informative(np.asarray(result["interp"][m], dtype=float), informative)
        for m in result["middles"]
    }

    summary = {
        "low": result["low"], "high": result["high"], "middles": result["middles"],
        "counts": result["counts"], "k": result["k"],
        "layer_indices": layer_indices,
        "auc_2v4": result["auc_2v4"],
        "interp": result["interp"],
        "best_layer_index": layer_indices[best],
        "best_auc_2v4": float(auc[best]),
        "mean_position_informative": positions,
        "plots_dir": os.path.abspath(plots_dir),
        "plots": [os.path.basename(path)],
    }
    summary_path = os.path.join(plots_dir, "tier2g_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[probe2g]   wrote {summary_path}")

    print("[probe2g] ---- verdict ----")
    print(f"[probe2g]   trained on {result['low']} vs {result['high']}; "
          f"best held-out AUC = {auc[best]:.3f} @ layer {layer_indices[best]}")
    for m in result["middles"]:
        pos = positions[m]
        print(f"[probe2g]   {m}: mean position on {result['low']}->{result['high']} axis "
              f"= {pos:.2f}  (0={result['low']}, 1={result['high']}; over informative layers)")
        if 0.0 < pos < 1.0:
            print(f"[probe2g]   => {m} INTERPOLATES between the extremes: genuine ORDINAL depth "
                  "representation (not class-specific memorization).")
        else:
            print(f"[probe2g]   => {m} does NOT interpolate (pos {pos:.2f}): the probe likely keys "
                  "on class-specific features, not a clean depth axis.")
    print("[probe2g] DONE.")
