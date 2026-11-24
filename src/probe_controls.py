"""Tier 2b: is the decoded signal real, or surface form / probe capacity?

Paper Sec. III-D / IV-C (Figs. 2-3, Table I "Surf. only", "Surf. resid.",
"Null95" and the ± CIs). On the same activations as Tier 2:

  1. Surface control: regress the surface features of ``src/surface.py``
     (six, or five without a tokenizer) out of every activation feature within
     each training fold, then re-probe; plus a probe on the surface features
     alone. -> ``tier2b_multicovariate_control_by_layer.png`` (Fig. 3)
  2. Label-permutation null: refit the probe on shuffled labels ``perms``
     times. -> ``tier2b_permutation_null_by_layer.png`` (Fig. 2)
  3. Length-only control (character length alone; supplementary).
     -> ``tier2b_length_control_by_layer.png``
  4. Bootstrap CIs over questions for the raw and surface-residualised
     peak-layer AUCs (Table I ±), from the out-of-fold scores of 1. and Tier 2.

Everything lands in ``tier2b_summary.json``. The probes share one RNG in a
fixed call order (raw, length-residualised, length-only, surface-residualised,
surface-only, permutations), so the folds, and hence every number, match the
published runs; the bootstrap uses its own RNG (``seed + 1``).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede the pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.probe_core import bootstrap_peak_ci, probe_auc_by_layer  # noqa: E402
from src.run_data import load_run, order_labels, resolve_plots_dir  # noqa: E402
from src.surface import SURFACE_FEATURES, surface_matrix, try_load_tokenizer  # noqa: E402


def _plot_control(x, raw_auc, resid_auc, ref_auc, ref_label, resid_label, out_path) -> None:
    """Raw vs residualised probe AUC by layer, with a horizontal reference."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    ax.axhline(ref_auc, color="#8172B3", ls=":", lw=1.7, label=ref_label)
    ax.plot(x, raw_auc, marker="o", ms=4, lw=1.9, color="#C44E52", label="raw probe AUC")
    ax.plot(x, resid_auc, marker="^", ms=4, lw=1.9, color="#55A868", label=resid_label)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Cross-validated AUC")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_null(x, raw_auc, null_mean, null_p05, null_p95, out_path) -> None:
    """Real probe AUC against the shuffled-label null band."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    ax.fill_between(x, null_p05, null_p95, color="0.7", alpha=0.35,
                    label="shuffled-label null (5–95%)")
    ax.plot(x, null_mean, color="0.45", lw=1.3, label="null mean")
    ax.plot(x, raw_auc, marker="o", ms=4, lw=1.9, color="#C44E52", label="real probe AUC")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Cross-validated AUC")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(output_dir: str, plots_dir: str | None = None, folds: int = 5, seed: int = 0,
        perms: int = 200, n_boot: int = 2000) -> None:
    """Build the Tier-2b artifacts for a finished run.

    Args mirror :func:`src.probe.run`; ``perms`` is the number of label
    shuffles for the null (paper: 200) and ``n_boot`` the number of bootstrap
    resamples for the CIs (paper: 2000; 0 skips them).
    """
    try:
        data = load_run(output_dir, log_tag="probe2b")
    except FileNotFoundError as exc:
        print(f"[probe2b] SKIP Tier 2b: {exc}")
        return
    plots_dir = resolve_plots_dir(plots_dir, output_dir)

    X, y, labels_ordered, layer_indices = data.X, data.y, data.labels, data.layer_indices
    n, n_layers, _ = X.shape

    counts = data.counts
    if len(labels_ordered) < 2 or min(counts.values()) < 2:
        print(f"[probe2b] SKIP: need >=2 labels with >=2 samples each (have {counts}).")
        return

    tokenizer = try_load_tokenizer(data.run_config, log_tag="probe2b")
    cov, cov_names = surface_matrix(data.questions, SURFACE_FEATURES, tokenizer)
    char_len = cov[:, 0]
    print(f"[probe2b] surface covariates ({len(cov_names)}): {cov_names}")

    rng = np.random.default_rng(seed)

    # The call order below fixes the CV folds; keep it.
    raw = probe_auc_by_layer(X, y, labels_ordered, folds, rng, return_scores=True)
    resid_len = probe_auc_by_layer(X, y, labels_ordered, folds, rng, covariate=char_len)
    length_only = probe_auc_by_layer(char_len.reshape(n, 1, 1), y, labels_ordered, folds, rng)
    resid_multi = probe_auc_by_layer(X, y, labels_ordered, folds, rng, covariate=cov,
                                     return_scores=True)
    surface_only = probe_auc_by_layer(cov.reshape(n, 1, cov.shape[1]), y, labels_ordered,
                                      folds, rng)

    raw_auc = np.asarray(raw["auc"], dtype=float)
    resid_len_auc = np.asarray(resid_len["auc"], dtype=float)
    length_only_auc = float(length_only["auc"][0])
    resid_multi_auc = np.asarray(resid_multi["auc"], dtype=float)
    surface_only_auc = float(surface_only["auc"][0])

    print(f"[probe2b] running {perms} label-permutation shuffles ...")
    null = np.full((perms, n_layers), np.nan)
    for p in range(perms):
        y_perm = y[rng.permutation(n)]
        res = probe_auc_by_layer(X, y_perm, order_labels(y_perm), folds, rng)
        if res and not res.get("skipped"):
            null[p] = res["auc"]
    null_mean = np.nanmean(null, axis=0)
    null_p05 = np.nanpercentile(null, 5, axis=0)
    null_p95 = np.nanpercentile(null, 95, axis=0)

    bootstrap = None
    if n_boot > 0:
        print(f"[probe2b] bootstrapping peak-layer AUCs ({n_boot} resamples) ...")
        boot_rng = np.random.default_rng(seed + 1)
        bootstrap = {"n_boot": n_boot}
        for key, res in (("raw", raw), ("surface_residualized", resid_multi)):
            ci = bootstrap_peak_ci(res["oof_scores"], res["pos_masks"], n_boot, boot_rng)
            ci["layer_index"] = layer_indices[ci["layer"]]
            bootstrap[key] = ci

    x = np.asarray(layer_indices)
    path_len = os.path.join(plots_dir, "tier2b_length_control_by_layer.png")
    _plot_control(x, raw_auc, resid_len_auc, length_only_auc,
                  f"length-only AUC = {length_only_auc:.2f}", "length-residualized AUC", path_len)
    path_multi = os.path.join(plots_dir, "tier2b_multicovariate_control_by_layer.png")
    _plot_control(x, raw_auc, resid_multi_auc, surface_only_auc,
                  f"surface-only AUC = {surface_only_auc:.2f}  ({len(cov_names)} feats)",
                  "surface-residualized AUC", path_multi)
    path_null = os.path.join(plots_dir, "tier2b_permutation_null_by_layer.png")
    _plot_null(x, raw_auc, null_mean, null_p05, null_p95, path_null)
    for path in (path_len, path_multi, path_null):
        print(f"[probe2b]   wrote {path}")

    raw_mean = float(np.nanmean(raw_auc))
    resid_len_mean = float(np.nanmean(resid_len_auc))
    resid_multi_mean = float(np.nanmean(resid_multi_auc))
    null_overall = float(np.nanmean(null_mean))
    null_p95_overall = float(np.nanmean(null_p95))

    summary = {
        "n_questions": n,
        "n_layers": n_layers,
        "perms": perms,
        "labels": labels_ordered,
        "counts": counts,
        "covariates": cov_names,
        "length_chars": {"min": float(char_len.min()), "max": float(char_len.max()),
                         "mean": float(char_len.mean())},
        "length_only_auc": length_only_auc,
        "surface_only_auc": surface_only_auc,
        "raw_auc": raw_auc.tolist(),
        "residualized_auc": resid_len_auc.tolist(),            # length-residualised
        "surface_residualized_auc": resid_multi_auc.tolist(),  # all surface features
        "null_mean": null_mean.tolist(),
        "null_p05": null_p05.tolist(),
        "null_p95": null_p95.tolist(),
        "raw_auc_mean": raw_mean,
        "residualized_auc_mean": resid_len_mean,
        "surface_residualized_auc_mean": resid_multi_mean,
        "null_auc_mean": null_overall,
        "plots_dir": plots_dir,
        "plots": [os.path.basename(p) for p in (path_len, path_multi, path_null)],
    }
    if bootstrap is not None:
        summary["bootstrap"] = bootstrap
    summary_path = os.path.join(plots_dir, "tier2b_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[probe2b]   wrote {summary_path}")

    print("[probe2b] ---- summary ----")
    print(f"[probe2b]   raw probe AUC (mean over layers)        = {raw_mean:.3f}")
    print(f"[probe2b]   length-only AUC                         = {length_only_auc:.3f}")
    print(f"[probe2b]   length-residualized AUC (mean)          = {resid_len_mean:.3f}")
    print(f"[probe2b]   surface-only AUC ({len(cov_names)} features)         = {surface_only_auc:.3f}")
    print(f"[probe2b]   surface-residualized AUC (mean)         = {resid_multi_mean:.3f}")
    print(f"[probe2b]   shuffled-label null AUC (mean, p95)     = {null_overall:.3f}, "
          f"{null_p95_overall:.3f}")
    print(f"[probe2b]   Null95 (max over layers of the null p95) = {float(np.nanmax(null_p95)):.2f}")
    if bootstrap is not None:
        for key in ("raw", "surface_residualized"):
            b = bootstrap[key]
            print(f"[probe2b]   peak {key:<20s} layer {b['layer_index']:>3}: "
                  f"AUC {b['auc']:.3f} ± {b['half_width']:.3f}")
    if resid_multi_mean <= null_p95_overall + 0.05:
        print("[probe2b]   NOTE: the surface-residualised signal is not clearly above the null.")
    print("[probe2b] DONE.")
