"""Tier-2t CROSS-DATASET TRANSFER probe for FATHOM.

Tier 2g showed the hop-count signal behaves like an ordinal depth axis WITHIN
one dataset. The remaining worry: that axis could still be DATASET-specific --
keyed to MuSiQue's composed templates / topic mix -- rather than a general
"reasoning depth" direction in the model. This test settles it with the
strongest cheap evidence available: train the axis on run A and evaluate it on
run B (a DIFFERENT dataset extracted by the SAME model / pooling / layers),
and vice versa. Both directions are computed in one invocation.

Protocol, per direction (A -> B) and per layer:

  * Fit the diagonal-LDA low-vs-high axis on ALL of A's extreme-class rows
    (standardize on A, class-mean difference, oriented low -> high). No CV is
    needed here -- run B is entirely held out.
  * Project ALL of B onto that axis and score the low-vs-high AUC over B's
    extreme classes (the "transfer AUC"). As the ceiling, compare against B's
    OWN within-dataset cross-validated extreme AUC (same diagonal-LDA
    protocol, k-fold). Headline metric:

        retention = (transfer_auc - 0.5) / (within_auc - 0.5)

  * Project B's MIDDLE classes (if any -- e.g. 3hop) onto the A-trained axis
    and compute the interpolation fraction anchored on B's OWN extreme
    projections -- Tier 2g's ordinality test, but across datasets.

Label vocabularies do NOT need to match: each run's labels are ordered by
``order_labels`` (hardness words first, anything else alphabetical) and the
axis maps low <-> low / high <-> high, so graded ``2hop/3hop/4hop`` transfers
directly onto binary ``easy/hard``.

Honesty notes. Standardization (mu/sd), the axis and the decision threshold
all come from the TRAIN run only -- nothing is fit on the test run. A
wholesale shift of B along the axis therefore cannot fake a high transfer AUC
(AUC is rank-based within B) nor the interpolation fraction (anchored on B's
own extremes); only the thresholded balanced accuracy is shift-sensitive,
which is why it is reported as a secondary, calibration-style metric.

Validity guard: both runs must come from the SAME model, pooling and layer
selection. ``run_config.json`` provenance is checked (model/pooling mismatch
is a hard error); a shape or layer-index mismatch is always a hard error.

numpy + matplotlib only; reuses the shared probe core (``src.probe_core``).
Needs >=2 labels in each run; skips gracefully otherwise.

Synthetic no-GPU smoke test (fabricates two tiny compatible runs, then runs
the full analysis end-to-end):  cd <repo_root> && python -m src.probe_transfer
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
    balanced_acc,
    fit_diag_lda,
    mean_over_informative,
    middle_positions,
    probe_auc_by_layer,
    project_diag_lda,
    roc_auc,
)
from src.run_data import RunData, load_run, resolve_plots_dir  # noqa: E402

# Verdict thresholds on retention = (transfer - 0.5) / (within - 0.5).
_RET_STRONG = 0.70
_RET_PARTIAL = 0.30


# --------------------------------------------------------------------------- #
# Compatibility                                                               #
# --------------------------------------------------------------------------- #
def _check_compatible(a: RunData, b: RunData) -> None:
    """Hard-error unless the two runs share model / pooling / layers / shapes."""
    if a.X.shape[1:] != b.X.shape[1:]:
        raise ValueError(
            f"incompatible runs: {a.name} stores per-question arrays of shape "
            f"{a.X.shape[1:]} (layers, hidden) but {b.name} stores "
            f"{b.X.shape[1:]}. Transfer is only valid within ONE model/layers/pooling."
        )
    if list(a.layer_indices) != list(b.layer_indices):
        raise ValueError(
            f"incompatible layer selections: {a.name} has layer indices "
            f"{a.layer_indices} vs {b.name} {b.layer_indices}."
        )
    for key in ("model_name", "pooling"):
        va, vb = a.run_config.get(key), b.run_config.get(key)
        if va is not None and vb is not None and va != vb:
            raise ValueError(
                f"incompatible runs: {key} differs ({a.name}={va!r} vs "
                f"{b.name}={vb!r}). Cross-dataset transfer is only valid within "
                "ONE model/pooling."
            )
    va, vb = a.run_config.get("dtype"), b.run_config.get("dtype")
    if va is not None and vb is not None and va != vb:
        print(f"[probe2t]   WARNING: dtype differs across runs ({va!r} vs {vb!r}); "
              "numerics may shift slightly but the comparison stays valid.")


# --------------------------------------------------------------------------- #
# Core math                                                                   #
# --------------------------------------------------------------------------- #
def within_extremes_reference(run: RunData, folds: int, seed: int):
    """The ceiling: k-fold CV low-vs-high AUC per layer WITHIN one run.

    Uses the exact Tier-2 probe (``probe_auc_by_layer``) restricted to the
    run's extreme classes, so it is apples-to-apples with the transfer AUC.
    Returns a per-layer list, or ``None`` (with a warning) if uncomputable.
    """
    X, y, labels = run.X, run.y, run.labels
    low, high = labels[0], labels[-1]
    mask = (y == low) | (y == high)
    res = probe_auc_by_layer(X[mask], y[mask], [low, high], folds,
                             np.random.default_rng(seed))
    if res is None or res.get("skipped"):
        reason = (res or {}).get("reason", "<2 labels")
        print(f"[probe2t]   WARNING: no within-{run.name} ceiling ({reason}).")
        return None
    return [float(v) for v in res["auc"]]


def transfer_direction(train: RunData, test: RunData) -> dict:
    """Train the low-vs-high axis on ALL of ``train``'s extremes; score ``test``.

    Per layer: standardize on the train extremes, take the oriented
    (high - low) class-mean direction and the midpoint threshold, then project
    every test row. Returns per-layer ``transfer_auc`` (test extremes),
    ``transfer_balanced_acc`` (train threshold -- shift-sensitive, secondary)
    and ``interp`` (each test middle class's position anchored on the test
    set's own extreme projections), or ``{"skipped": True, ...}``.
    """
    Xa, ya, labels_a = train.X, train.y, train.labels
    Xb, yb, labels_b = test.X, test.y, test.labels
    low_a, high_a = labels_a[0], labels_a[-1]
    low_b, high_b = labels_b[0], labels_b[-1]
    middles_b = list(labels_b[1:-1])

    tr_mask = (ya == low_a) | (ya == high_a)
    pos_tr = ya[tr_mask] == high_a
    n_lo_a, n_hi_a = int((~pos_tr).sum()), int(pos_tr.sum())

    low_b_idx = np.where(yb == low_b)[0]
    high_b_idx = np.where(yb == high_b)[0]
    ext_b = np.concatenate([low_b_idx, high_b_idx])
    pos_b = yb[ext_b] == high_b
    mid_idx_b = {m: np.where(yb == m)[0] for m in middles_b}

    out = {
        "train": train.name, "test": test.name,
        "train_low": str(low_a), "train_high": str(high_a),
        "test_low": str(low_b), "test_high": str(high_b),
        "test_middles": [str(m) for m in middles_b],
        "train_extreme_counts": {str(low_a): n_lo_a, str(high_a): n_hi_a},
        "test_extreme_counts": {str(low_b): int(low_b_idx.size),
                                str(high_b): int(high_b_idx.size)},
    }
    if min(n_lo_a, n_hi_a) < 2:
        out.update(skipped=True,
                   reason=f"train extremes too small ({out['train_extreme_counts']})")
        return out

    n_layers = Xa.shape[1]
    t_auc, t_bacc = [], []
    interp = {m: [] for m in middles_b}
    for L in range(n_layers):
        # Axis + threshold fit on ALL of the train run's extremes (no CV needed
        # -- the test run is entirely held out), oriented low -> high.
        mu, sd, w, thr = fit_diag_lda(Xa[tr_mask, L, :], pos_tr)
        s_b = project_diag_lda(Xb[:, L, :], mu, sd, w)  # EVERY test row on the foreign axis
        t_auc.append(roc_auc(s_b[ext_b], pos_b))
        t_bacc.append(balanced_acc(pos_b, s_b[ext_b] >= thr))
        for m, p in middle_positions(s_b, low_b_idx, high_b_idx, mid_idx_b).items():
            interp[m].append(p)

    out.update(
        skipped=False,
        transfer_auc=[float(v) for v in t_auc],
        transfer_balanced_acc=[float(v) for v in t_bacc],
        interp={str(m): [float(v) for v in vs] for m, vs in interp.items()},
    )
    return out


def _finalize_direction(d: dict, within_cv: dict, layer_indices: list) -> None:
    """Attach best-layer, retention and interpolation summaries to a direction.

    * ``retention`` averages over layers where the WITHIN-test ceiling clears
      ``INFORMATIVE_AUC`` (denominator-defined selection -- using the transfer
      curve itself would bias retention upward).
    * ``mean_position_informative`` averages each middle's position over layers
      where the TRANSFERRED axis itself separates the test extremes (positions
      are only meaningful where the axis works), mirroring Tier 2g.
    """
    if d.get("skipped"):
        return
    t = np.asarray(d["transfer_auc"], dtype=float)
    best = argmax_finite(t)
    d["best_layer_index"] = layer_indices[best]
    d["best_transfer_auc"] = float(t[best])

    within = within_cv.get(d["test"])
    if within is not None:
        wv = np.asarray(within, dtype=float)
        # Band selected by the WITHIN ceiling (fallback: all finite), applied to
        # both curves so transfer and ceiling average over the SAME layers.
        inf_w = wv > INFORMATIVE_AUC
        fallback_w = np.isfinite(wv)
        tm = mean_over_informative(t, inf_w, fallback_w)
        wm = mean_over_informative(wv, inf_w, fallback_w)
        d["mean_transfer_auc_informative"] = tm
        d["mean_within_auc_informative"] = wm
        d["retention"] = (float((tm - 0.5) / (wm - 0.5))
                          if abs(wm - 0.5) > 1e-9 else None)
    else:
        d["mean_transfer_auc_informative"] = None
        d["mean_within_auc_informative"] = None
        d["retention"] = None

    # Each middle's position over layers where the TRANSFERRED axis separates
    # the test extremes (fallback: all finite), mirroring Tier 2g.
    inf_t = t > INFORMATIVE_AUC
    fallback_t = np.isfinite(t)
    d["mean_position_informative"] = {
        m: mean_over_informative(np.asarray(d["interp"][m], dtype=float), inf_t, fallback_t)
        for m in d["test_middles"]
    }


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def plot_transfer(layer_indices, directions, within_cv, out_path: str) -> None:
    """1x2 (or 1x1 without middles): transfer AUC vs ceilings, + interpolation."""
    x = np.asarray(layer_indices)
    live = [d for d in directions if not d.get("skipped")]
    has_middles = any(d["test_middles"] for d in live)
    dir_colors = ("#C44E52", "#4C72B0")

    ncols = 2 if has_middles else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11.5 if has_middles else 8.2, 4.6),
                             squeeze=False)

    ax0 = axes[0][0]
    ax0.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    for d, color in zip(directions, dir_colors):
        if d.get("skipped"):
            continue
        auc = np.asarray(d["transfer_auc"], dtype=float)
        ax0.plot(x, auc, marker="o", ms=3.5, lw=1.9, color=color,
                 label=f"{d['train']} → {d['test']}  (transfer)")
        within = within_cv.get(d["test"])
        if within is not None:
            ax0.plot(x, np.asarray(within, dtype=float), ls="--", lw=1.3, color=color,
                     alpha=0.45, label=f"within {d['test']} CV (ceiling)")
        best = argmax_finite(auc)
        ax0.scatter([x[best]], [auc[best]], s=85, facecolor="none", edgecolor="black",
                    zorder=5)
    ax0.set_ylim(0.0, 1.02)
    ax0.set_xlabel("Layer (0 = embeddings)")
    ax0.set_ylabel("Low-vs-high AUC")
    ax0.grid(alpha=0.25)
    ax0.set_axisbelow(True)
    ax0.legend(loc="lower right", fontsize=8)

    if has_middles:
        ax1 = axes[0][1]
        ax1.axhspan(0.0, 1.0, color="#55A868", alpha=0.10, label="interpolation zone [0, 1]")
        ax1.axhline(0.5, color="0.5", ls="--", lw=1, label="ideal midpoint (0.5)")
        ax1.axhline(0.0, color="0.7", lw=0.8)
        ax1.axhline(1.0, color="0.7", lw=0.8)
        cmap = plt.get_cmap("tab10")
        ci = 0
        for d in live:
            for m in d["test_middles"]:
                ax1.plot(x, np.asarray(d["interp"][m], dtype=float), marker="^", ms=3.5,
                         lw=1.8, color=cmap(ci % 10),
                         label=f"{d['test']} {m} (axis from {d['train']})")
                ci += 1
        ax1.set_ylim(-0.4, 1.4)
        ax1.set_xlabel("Layer (0 = embeddings)")
        ax1.set_ylabel("Position between test extremes")
        ax1.grid(alpha=0.25)
        ax1.set_axisbelow(True)
        ax1.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _print_verdict(d: dict) -> None:
    """One direction's console verdict (mirrors the Tier-2b/2g style)."""
    if d.get("skipped"):
        print(f"[probe2t]   {d['train']} -> {d['test']}: SKIPPED ({d['reason']})")
        return
    print(f"[probe2t]   {d['train']} -> {d['test']}  "
          f"(axis {d['train_low']}->{d['train_high']}, "
          f"tested on {d['test_low']} vs {d['test_high']}):")
    print(f"[probe2t]     best transfer AUC = {d['best_transfer_auc']:.3f} "
          f"@ layer {d['best_layer_index']}")
    ret = d.get("retention")
    if ret is not None:
        print(f"[probe2t]     mean transfer AUC = {d['mean_transfer_auc_informative']:.3f} "
              f"vs within-{d['test']} ceiling {d['mean_within_auc_informative']:.3f} "
              f"(informative layers) -> retention {ret * 100:.0f}%")
        if d["mean_transfer_auc_informative"] < 0.5:
            print("[probe2t]     => axis ORIENTATION FLIPPED on the test set (AUC < 0.5): "
                  "check the label ordering of both runs.")
        elif ret >= _RET_STRONG:
            print("[probe2t]     => axis TRANSFERS across datasets: the depth direction is "
                  "largely dataset-GENERAL, not template-specific.")
        elif ret >= _RET_PARTIAL:
            print("[probe2t]     => PARTIAL transfer: a shared depth component plus "
                  "dataset-specific structure.")
        else:
            print("[probe2t]     => little/no transfer: the within-dataset signal is mostly "
                  "dataset-SPECIFIC (template/topic-tied).")
    else:
        print("[probe2t]     (no within-test ceiling available; read the raw transfer AUC "
              "against chance 0.5 directly)")
    for m, p in d.get("mean_position_informative", {}).items():
        tag = ("INTERPOLATES on the foreign axis (ordinal across datasets)"
               if 0.0 < p < 1.0 else "does NOT interpolate on the foreign axis")
        print(f"[probe2t]     {m}: mean position = {p:.2f} "
              f"(0={d['test_low']}, 1={d['test_high']}) -> {tag}")


def run(output_dir: str, test_output_dir: str,
        metadata_path: str | None = None, activations_dir: str | None = None,
        test_metadata_path: str | None = None, test_activations_dir: str | None = None,
        plots_dir: str | None = None, folds: int = 5, seed: int = 0) -> None:
    """Build the Tier-2t cross-dataset transfer artifacts from TWO finished runs.

    Args:
        output_dir: run A's directory (as passed to ``main.py --output``).
        test_output_dir: run B's directory. Both directions (A->B and B->A) are
            computed; the order only affects naming.
        metadata_path / activations_dir: run A overrides (defaults inside A).
        test_metadata_path / test_activations_dir: run B overrides.
        plots_dir: output dir (default ``<repo_root>/figures``, created if missing).
        folds: CV folds for the within-dataset ceilings (clamped per class count).
        seed: RNG seed for the ceiling CV folds (the transfer itself is
            deterministic -- no folds, no sampling).

    Writes ``tier2t_transfer_by_layer.png`` and ``tier2t_summary.json``.
    """
    plots_dir = resolve_plots_dir(plots_dir)
    try:
        run_a = load_run(output_dir, metadata_path, activations_dir, log_tag="probe2t")
        run_b = load_run(test_output_dir, test_metadata_path, test_activations_dir,
                         log_tag="probe2t")
    except FileNotFoundError as exc:
        print(f"[probe2t] SKIP Tier 2t: {exc}")
        return

    for r in (run_a, run_b):
        if not r.run_config:
            print(f"[probe2t]   WARNING: no usable run_config.json in {r.run_dir!r}; cannot "
                  "verify model/pooling provenance -- make sure both runs used the same model.")

    if os.path.abspath(output_dir) == os.path.abspath(test_output_dir):
        print("[probe2t]   WARNING: --output and --test-output are the SAME run; "
              "'transfer' is then circular (train set == test set) and the AUC is "
              "inflated. Point them at two different runs.")
    if run_a.name == run_b.name:  # same basename (or same dir twice) -> disambiguate
        run_a.name += " (A)"
        run_b.name += " (B)"

    _check_compatible(run_a, run_b)

    for r in (run_a, run_b):
        if len(r.labels) < 2:
            print(f"[probe2t] SKIP: run {r.name} has <2 labels ({r.labels}); "
                  "no low-vs-high axis to train or test.")
            return
        mid = r.labels[1:-1]
        if mid:
            print(f"[probe2t]   note: {r.name} middles {mid} are EXCLUDED from axis "
                  "training and used only as held-out interpolation probes.")

    layer_indices = run_a.layer_indices
    print(f"[probe2t] within-dataset CV ceilings (extremes only, {folds}-fold) ...")
    within_cv = {
        run_a.name: within_extremes_reference(run_a, folds, seed),
        run_b.name: within_extremes_reference(run_b, folds, seed),
    }

    directions = [transfer_direction(run_a, run_b), transfer_direction(run_b, run_a)]
    for d in directions:
        _finalize_direction(d, within_cv, layer_indices)

    plots = []
    if any(not d.get("skipped") for d in directions):
        path = os.path.join(plots_dir, "tier2t_transfer_by_layer.png")
        plot_transfer(layer_indices, directions, within_cv, path)
        plots.append(os.path.basename(path))
        print(f"[probe2t]   wrote {path}")
    else:
        print("[probe2t]   no computable direction; skipping the plot.")

    def _run_info(r: RunData) -> dict:
        return {
            "name": r.name, "output_dir": os.path.abspath(r.run_dir),
            "n_questions": int(r.X.shape[0]), "labels": [str(lab) for lab in r.labels],
            "counts": r.counts,
            "model_name": r.run_config.get("model_name"),
            "pooling": r.run_config.get("pooling"),
        }

    summary = {
        "run_a": _run_info(run_a),
        "run_b": _run_info(run_b),
        "layer_indices": layer_indices,
        "folds": folds,
        "seed": seed,
        "informative_auc_threshold": INFORMATIVE_AUC,
        "retention_thresholds": {"strong": _RET_STRONG, "partial": _RET_PARTIAL},
        "within_cv_auc": within_cv,
        "directions": directions,
        "plots_dir": os.path.abspath(plots_dir),
        "plots": plots,
    }
    summary_path = os.path.join(plots_dir, "tier2t_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[probe2t]   wrote {summary_path}")

    print("[probe2t] ---- verdict ----")
    for d in directions:
        _print_verdict(d)
    print("[probe2t] DONE.")


if __name__ == "__main__":  # pragma: no cover -- synthetic no-GPU smoke test
    # Fabricate two tiny compatible runs sharing ONE planted depth direction
    # (plus dataset-specific offsets and noise), then run the full Tier-2t
    # analysis end-to-end. Expect: high transfer AUC both ways, retention near
    # 100%, and the 3hop middle interpolating (~0.5) on the foreign axis.
    #
    #   cd <repo_root> && python -m src.probe_transfer
    import tempfile

    rng = np.random.default_rng(0)
    n_layers, dim, per_class = 6, 48, 40
    shared = rng.normal(size=dim)
    shared /= np.linalg.norm(shared)

    base = tempfile.mkdtemp(prefix="fathom_tier2t_smoke_")
    specs = {
        "smokeA": (("2hop", "3hop", "4hop"), (0.0, 1.5, 3.0)),
        "smokeB": (("easy", "hard"), (0.0, 3.0)),
    }
    for name, (labs, strengths) in specs.items():
        out = os.path.join(base, name)
        os.makedirs(os.path.join(out, "activations"))
        offset = rng.normal(size=dim) * 0.5  # dataset-specific wholesale shift
        with open(os.path.join(out, "metadata.jsonl"), "w", encoding="utf-8") as fh:
            q = 0
            for lab, s in zip(labs, strengths):
                for _ in range(per_class):
                    qid = f"{name}_q{q}"
                    q += 1
                    vec = rng.normal(size=(n_layers, dim)) + offset + s * shared
                    np.savez(os.path.join(out, "activations", f"{qid}.npz"),
                             root=vec.astype(np.float32))
                    fh.write(json.dumps({"id": qid, "label": lab, "nodes": []}) + "\n")
        with open(os.path.join(out, "run_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_name": "smoke-model", "pooling": "last_token",
                       "dtype": "float32"}, fh)

    run(os.path.join(base, "smokeA"), os.path.join(base, "smokeB"),
        plots_dir=os.path.join(base, "figures"))
    print(f"[probe2t] smoke artifacts in {base}")
