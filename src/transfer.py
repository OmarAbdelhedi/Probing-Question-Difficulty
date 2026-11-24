"""Cross-dataset transfer of the difficulty axis (paper Sec. III-F / IV-E).

For every way to hold one run out as the test set, the low -> high axis is
fit on the pooled extreme classes of the other two runs and scored on the
held-out run. Labels are matched by rank, not by name: each run's lowest
label is "low" and its highest is "high", so 2Wiki's easy/hard, HotpotQA's
easy/medium/hard and MuSiQue's 2hop/3hop/4hop all line up.

Per layer, for a raw and a surface-residualised axis:

  * transfer AUC on the test run's extreme classes;
  * the ceiling: the test run's own cross-validated extreme-class AUC;
  * retention R = (AUC_transfer - 0.5) / (AUC_within - 0.5), both averaged
    over the informative layers (within-dataset AUC > ``INFORMATIVE_AUC``)
    (paper Eq. 5, Table II);
  * the interpolation fraction of the test run's middle class(es) on the
    foreign axis (Fig. 6, right).

The residualised axis removes the surface features of ``src/surface.py``
(the same six as Tier 2b), with the regression fit on the pooled training
extremes and applied to the test run.

Outputs, per combination, in ``<plots_root>/transfer_sweep/transfer_<a>_<b>--<test>/``:
``fused_transfer_report.png`` (Fig. 6) / ``.json``, ``fused_transfer_summary.json``
(raw axis) and ``fused_transfer_residual_summary.json`` (residualised axis);
plus ``<plots_root>/transfer_sweep/fused_transfer_sweep_summary.json``.
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede the pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.probe_core import (  # noqa: E402
    EPS,
    INFORMATIVE_AUC,
    argmax_finite,
    balanced_acc,
    fit_diag_lda,
    middle_positions,
    probe_auc_by_layer,
    project_diag_lda,
    residualize,
    roc_auc,
)
from src.run_data import RunData, load_run  # noqa: E402
from src.surface import SURFACE_FEATURES, surface_matrix, try_load_tokenizer  # noqa: E402

RAW_COLOR = "#C44E52"
RESID_COLOR = "#4C72B0"
GREEN = "#55A868"
LOG = "transfer"


# --------------------------------------------------------------------------- #
# Loading and compatibility                                                   #
# --------------------------------------------------------------------------- #
class RunWithControls(NamedTuple):
    run: RunData
    covariates: np.ndarray
    covariate_names: list[str]


def load_run_with_controls(run_dir: str, tokenizers: dict | None = None) -> RunWithControls:
    """Load a run with its surface-feature matrix.

    ``tokenizers`` caches one tokenizer per model name across runs; a model
    whose tokenizer cannot be loaded gets the five tokenizer-free features.
    """
    run = load_run(run_dir, log_tag=LOG)
    tokenizers = {} if tokenizers is None else tokenizers
    model_name = run.run_config.get("model_name")
    if model_name not in tokenizers:
        tokenizers[model_name] = try_load_tokenizer(run.run_config, log_tag=LOG)
    covariates, names = surface_matrix(run.questions, SURFACE_FEATURES, tokenizers[model_name])
    return RunWithControls(run=run, covariates=covariates, covariate_names=names)


def check_compatible(reference: RunData, other: RunData) -> None:
    """Runs must share array shape, layer selection, model and pooling."""
    if reference.X.shape[1:] != other.X.shape[1:]:
        raise ValueError(
            f"incompatible runs: {reference.name} stores arrays of shape "
            f"{reference.X.shape[1:]} but {other.name} stores {other.X.shape[1:]}"
        )
    if list(reference.layer_indices) != list(other.layer_indices):
        raise ValueError(
            f"incompatible layer selections: {reference.name} has {reference.layer_indices} "
            f"vs {other.name} {other.layer_indices}"
        )
    for key in ("model_name", "pooling"):
        va, vb = reference.run_config.get(key), other.run_config.get(key)
        if va is not None and vb is not None and va != vb:
            raise ValueError(
                f"incompatible runs: {key} differs ({reference.name}={va!r} vs "
                f"{other.name}={vb!r})"
            )


def check_bundle_set(bundles: list[RunWithControls]) -> None:
    reference = bundles[0].run
    for bundle in bundles[1:]:
        check_compatible(reference, bundle.run)
        if bundle.covariate_names != bundles[0].covariate_names:
            raise ValueError(
                f"covariate names differ: {bundle.covariate_names} vs {bundles[0].covariate_names}"
            )


def run_info(bundle_or_run) -> dict:
    run = bundle_or_run.run if isinstance(bundle_or_run, RunWithControls) else bundle_or_run
    out = {
        "name": run.name,
        "output_dir": run.run_dir,
        "n_questions": int(run.X.shape[0]),
        "labels": [str(label) for label in run.labels],
        "counts": run.counts,
        "model_name": run.run_config.get("model_name"),
        "pooling": run.run_config.get("pooling"),
    }
    if isinstance(bundle_or_run, RunWithControls):
        out["covariates"] = bundle_or_run.covariate_names
    return out


def tag_of(path: str) -> str:
    """``outputs/qwen32b/run_musique`` -> ``musique``."""
    base = os.path.basename(os.path.normpath(path))
    return base[4:] if base.startswith("run_") else base


def build_fused_jobs(run_paths: list[str]) -> list[dict]:
    """Every (two training runs, one held-out test run) combination."""
    norm = [os.path.abspath(path) for path in run_paths]
    if len(set(norm)) != len(norm):
        raise ValueError("duplicate run directories detected")
    if len(run_paths) < 3:
        raise ValueError(f"need at least 3 runs for 2-train/1-test fusion; got {len(run_paths)}")

    jobs = []
    for i, test_output in enumerate(run_paths):
        candidates = [path for j, path in enumerate(run_paths) if j != i]
        for combo in combinations(candidates, 2):
            jobs.append({"train_outputs": list(combo), "test_output": test_output})
    return jobs


def fused_plots_dir(plots_root: str, train_outputs: list[str], test_output: str) -> str:
    order = {"2wiki": 0, "musique": 1, "hotpot": 2}
    train_tags = sorted(
        [tag_of(path) for path in train_outputs],
        key=lambda value: (order.get(value, 999), value),
    )
    return os.path.join(plots_root, "transfer_sweep",
                        f"transfer_{'_'.join(train_tags)}--{tag_of(test_output)}")


# --------------------------------------------------------------------------- #
# Math                                                                        #
# --------------------------------------------------------------------------- #
def extreme_indices(run: RunData):
    low, high = run.labels[0], run.labels[-1]
    low_idx = np.where(run.y == low)[0]
    high_idx = np.where(run.y == high)[0]
    row_idx = np.concatenate([low_idx, high_idx])
    pos = run.y[row_idx] == high
    return low, high, low_idx, high_idx, row_idx, pos


def within_extremes_reference(run: RunData, folds: int, seed: int):
    """The raw ceiling: k-fold CV low-vs-high AUC per layer within one run."""
    low, high = run.labels[0], run.labels[-1]
    mask = (run.y == low) | (run.y == high)
    res = probe_auc_by_layer(run.X[mask], run.y[mask], [low, high], folds,
                             np.random.default_rng(seed))
    if res is None or res.get("skipped"):
        reason = (res or {}).get("reason", "<2 labels")
        print(f"[{LOG}]   WARNING: no within-{run.name} ceiling ({reason}).")
        return None
    return [float(v) for v in res["auc"]]


def within_residual_reference(bundle: RunWithControls, folds: int, seed: int):
    """The residualised ceiling: as above, with the surface features regressed out."""
    run = bundle.run
    low, high, _low_idx, _high_idx, row_idx, _pos = extreme_indices(run)
    result = probe_auc_by_layer(
        run.X[row_idx],
        run.y[row_idx],
        [low, high],
        folds,
        np.random.default_rng(seed),
        covariate=bundle.covariates[row_idx],
    )
    if result is None or result.get("skipped"):
        return None
    return [float(value) for value in result["auc"]]


def _train_extreme_pool(train_bundles: list[RunWithControls]):
    """Stack the extreme-class rows of every training run (low=0, high=1)."""
    train_blocks, cov_blocks, pos_blocks, components = [], [], [], []
    for bundle in train_bundles:
        run = bundle.run
        low, high, _low_idx, _high_idx, row_idx, pos = extreme_indices(run)
        n_low, n_high = int((~pos).sum()), int(pos.sum())
        if min(n_low, n_high) < 2:
            raise ValueError(
                f"{run.name} has too few extreme rows for transfer ({low}: {n_low}, {high}: {n_high})"
            )
        train_blocks.append(run.X[row_idx])
        cov_blocks.append(bundle.covariates[row_idx])
        pos_blocks.append(pos)
        components.append({
            "run": run.name,
            "low_label": str(low),
            "high_label": str(high),
            "counts": {str(low): n_low, str(high): n_high},
        })
    return (
        np.concatenate(train_blocks, axis=0),
        np.concatenate(cov_blocks, axis=0),
        np.concatenate(pos_blocks, axis=0),
        components,
    )


def transfer_direction(train_bundles: list[RunWithControls], test_bundle: RunWithControls,
                       *, residualized: bool) -> dict:
    """Fit the axis on the pooled training extremes and score the test run, per layer.

    Also records, per layer, how much each feature dimension helps or hurts
    the separation of the test extremes (for the diagnostic bar plots).
    """
    X_train, cov_train, pos_train, train_components = _train_extreme_pool(train_bundles)
    test_run = test_bundle.run
    low_test, high_test, low_idx, high_idx, ext_idx, pos_test = extreme_indices(test_run)
    middles = list(test_run.labels[1:-1])
    mid_idx = {middle: np.where(test_run.y == middle)[0] for middle in middles}

    direction = {
        "train": " + ".join(bundle.run.name for bundle in train_bundles),
        "train_runs": train_components,
        "train_semantics": {
            "negative": "each train run's lowest ordered label",
            "positive": "each train run's highest ordered label",
        },
        "train_extreme_counts_total": {
            "negative": int((~pos_train).sum()),
            "positive": int(pos_train.sum()),
        },
        "test": test_run.name,
        "test_low": str(low_test),
        "test_high": str(high_test),
        "test_middles": [str(middle) for middle in middles],
        "test_extreme_counts": {str(low_test): int(low_idx.size), str(high_test): int(high_idx.size)},
        "residualized": bool(residualized),
    }
    if residualized:
        direction["residualizer_fit_rows"] = "fused_train_extremes"

    transfer_auc, transfer_bacc = [], []
    interp = {middle: [] for middle in middles}
    per_layer = {
        "helping_mass": [],
        "hurting_mass": [],
        "net_gap": [],
        "axis_norm": [],
        "test_gap_norm": [],
        "axis_test_cosine": [],
    }
    cache = {"layer_indices": [], "weight": [], "test_gap_std": [], "contribution": []}

    for layer_pos, layer_index in enumerate(test_run.layer_indices):
        Xtr_layer = X_train[:, layer_pos, :]
        Xtest_layer = test_run.X[:, layer_pos, :]
        if residualized:
            Xtr_fit, Xtest_eval = residualize(Xtr_layer, cov_train, Xtest_layer,
                                              test_bundle.covariates)
        else:
            Xtr_fit, Xtest_eval = Xtr_layer, Xtest_layer

        mu, sd, weight, threshold = fit_diag_lda(Xtr_fit, pos_train)
        scores_test = project_diag_lda(Xtest_eval, mu, sd, weight)
        transfer_auc.append(roc_auc(scores_test[ext_idx], pos_test))
        transfer_bacc.append(balanced_acc(pos_test, scores_test[ext_idx] >= threshold))
        for middle, position in middle_positions(scores_test, low_idx, high_idx, mid_idx).items():
            interp[middle].append(position)

        low_std = (Xtest_eval[low_idx] - mu) / sd
        high_std = (Xtest_eval[high_idx] - mu) / sd
        test_gap_std = high_std.mean(axis=0) - low_std.mean(axis=0)
        contrib = weight * test_gap_std
        weight_norm = float(np.linalg.norm(weight))
        gap_norm = float(np.linalg.norm(test_gap_std))
        cosine = (
            float(np.dot(weight, test_gap_std) / (weight_norm * gap_norm))
            if weight_norm > EPS and gap_norm > EPS
            else float("nan")
        )

        per_layer["helping_mass"].append(float(np.maximum(contrib, 0.0).sum()))
        per_layer["hurting_mass"].append(float(-np.minimum(contrib, 0.0).sum()))
        per_layer["net_gap"].append(float(contrib.sum()))
        per_layer["axis_norm"].append(weight_norm)
        per_layer["test_gap_norm"].append(gap_norm)
        per_layer["axis_test_cosine"].append(cosine)
        cache["layer_indices"].append(int(layer_index))
        cache["weight"].append(weight)
        cache["test_gap_std"].append(test_gap_std)
        cache["contribution"].append(contrib)

    direction.update(
        skipped=False,
        transfer_auc=[float(value) for value in transfer_auc],
        transfer_balanced_acc=[float(value) for value in transfer_bacc],
        interp={str(middle): [float(value) for value in values] for middle, values in interp.items()},
        per_layer=per_layer,
        _cache=cache,
    )
    return direction


def finalize_direction(direction: dict, within_auc: list[float] | None,
                       layer_indices: list[int]) -> None:
    """Attach best layers, informative-band means, retention and middle positions."""
    if direction.get("skipped"):
        return
    transfer = np.asarray(direction["transfer_auc"], dtype=float)
    bacc = np.asarray(direction["transfer_balanced_acc"], dtype=float)
    best_auc = argmax_finite(transfer)
    best_bacc = argmax_finite(bacc)
    direction["best_layer_index"] = int(layer_indices[best_auc])
    direction["best_transfer_auc"] = float(transfer[best_auc])
    direction["best_balanced_acc_layer_index"] = int(layer_indices[best_bacc])
    direction["best_balanced_acc"] = float(bacc[best_bacc])

    if within_auc is not None:
        within = np.asarray(within_auc, dtype=float)
        informative = within > INFORMATIVE_AUC
        if not np.any(informative):
            informative = np.isfinite(within)
        mean_transfer = float(np.nanmean(transfer[informative]))
        mean_within = float(np.nanmean(within[informative]))
        direction["mean_transfer_auc_informative"] = mean_transfer
        direction["mean_within_auc_informative"] = mean_within
        direction["retention"] = (
            float((mean_transfer - 0.5) / (mean_within - 0.5))
            if abs(mean_within - 0.5) > 1e-9
            else None
        )
    else:
        direction["mean_transfer_auc_informative"] = None
        direction["mean_within_auc_informative"] = None
        direction["retention"] = None

    informative_transfer = transfer > INFORMATIVE_AUC
    band = informative_transfer if np.any(informative_transfer) else np.isfinite(transfer)
    direction["mean_position_informative"] = {
        middle: float(np.nanmean(np.asarray(direction["interp"][middle], dtype=float)[band]))
        for middle in direction["test_middles"]
    }


def top_dims_for_layer(direction: dict, layer_index: int, top_k: int) -> tuple[list[dict], list[dict]]:
    """The ``top_k`` feature dimensions that most help / hurt at one layer."""
    layer_indices = direction["_cache"]["layer_indices"]
    try:
        layer_pos = layer_indices.index(int(layer_index))
    except ValueError:
        layer_pos = argmax_finite(direction["per_layer"]["net_gap"])

    contrib = np.asarray(direction["_cache"]["contribution"][layer_pos], dtype=float)
    weight = np.asarray(direction["_cache"]["weight"][layer_pos], dtype=float)
    test_gap_std = np.asarray(direction["_cache"]["test_gap_std"][layer_pos], dtype=float)
    pos_dims = np.argsort(contrib)[-top_k:][::-1]
    neg_dims = np.argsort(contrib)[:top_k]

    def pack(indices) -> list[dict]:
        return [
            {
                "dim": int(dim),
                "contribution": float(contrib[dim]),
                "weight": float(weight[dim]),
                "test_gap_std": float(test_gap_std[dim]),
            }
            for dim in indices
        ]

    return pack(pos_dims), pack(neg_dims)


def jsonable_direction(direction: dict, top_k: int) -> dict:
    out = {key: value for key, value in direction.items() if key != "_cache"}
    if not direction.get("skipped"):
        helping, hurting = top_dims_for_layer(direction, direction["best_layer_index"], top_k)
        out["top_helping_dims_at_best_transfer_layer"] = helping
        out["top_hurting_dims_at_best_transfer_layer"] = hurting
    return out


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _plot_auc_compare(ax, layer_indices, raw_direction, residual_direction,
                      raw_within=None, residual_within=None) -> None:
    x = np.asarray(layer_indices)
    ax.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    raw_auc = np.asarray(raw_direction["transfer_auc"], dtype=float)
    resid_auc = np.asarray(residual_direction["transfer_auc"], dtype=float)
    ax.plot(x, raw_auc, color=RAW_COLOR, ls=":", marker="o", ms=3.0, lw=1.8, label="raw transfer")
    ax.plot(x, resid_auc, color=RESID_COLOR, marker="o", ms=3.0, lw=1.8, label="residual transfer")
    if raw_within is not None:
        ax.plot(x, np.asarray(raw_within, dtype=float), color=RAW_COLOR, ls="--", lw=1.1,
                alpha=0.35, label="raw within-test CV")
    if residual_within is not None:
        ax.plot(x, np.asarray(residual_within, dtype=float), color=RESID_COLOR, ls="--", lw=1.1,
                alpha=0.35, label="residual within-test CV")
    for arr, color in [(raw_auc, RAW_COLOR), (resid_auc, RESID_COLOR)]:
        best = argmax_finite(arr)
        ax.scatter([x[best]], [arr[best]], s=64, facecolor="none", edgecolor=color, zorder=5)
    ax.legend(loc="lower right", fontsize=7.5)


def _plot_middle_compare(ax, layer_indices, raw_direction, residual_direction) -> None:
    x = np.asarray(layer_indices)
    middles = raw_direction.get("test_middles") or residual_direction.get("test_middles")
    if not middles:
        ax.axis("off")
        ax.set_title("No middle class in this held-out test run", fontsize=10)
        return

    ax.axhspan(0.0, 1.0, color=GREEN, alpha=0.10, label="interpolation zone [0, 1]")
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="midpoint (0.5)")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axhline(1.0, color="0.7", lw=0.8)
    cmap = plt.get_cmap("tab10")
    for i, middle in enumerate(middles):
        color = cmap(i % 10)
        if middle in raw_direction.get("interp", {}):
            ax.plot(x, np.asarray(raw_direction["interp"][middle], dtype=float), color=color,
                    ls=":", marker="^", ms=3.0, lw=1.5, label=f"raw {middle}")
        if middle in residual_direction.get("interp", {}):
            ax.plot(x, np.asarray(residual_direction["interp"][middle], dtype=float), color=color,
                    marker="^", ms=3.0, lw=1.7, label=f"residual {middle}")
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Position between test extremes")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)


def _plot_transfer_feature_bars(ax, direction: dict, top_k: int, *, title_prefix: str,
                                residualized: bool) -> None:
    helping, hurting = top_dims_for_layer(direction, direction["best_layer_index"], top_k)
    dims = helping[::-1] + hurting[::-1]
    values = [entry["contribution"] for entry in dims]
    ypos = np.arange(len(dims))
    ax.barh(ypos, values, color=[GREEN if v >= 0 else RAW_COLOR for v in values], alpha=0.92)
    ax.axvline(0.0, color="0.5", lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"d{entry['dim']}" for entry in dims], fontsize=8)
    max_abs = max((abs(float(v)) for v in values), default=1.0)
    if max_abs <= 0:
        max_abs = 1.0
    ax.set_xlim(-1.08 * max_abs, 1.08 * max_abs)
    ax.set_xlabel("Signed contribution on residualized test extremes" if residualized
                  else "Signed contribution on test extremes")
    ax.set_title(
        f"{title_prefix}: {direction['train']} -> {direction['test']}\n"
        f"layer {direction['best_layer_index']}  AUC={direction['best_transfer_auc']:.3f}",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


def _fused_report_plot(layer_indices, raw_direction, residual_direction, raw_within,
                       residual_within, top_k: int, report_level: str, out_path: str) -> None:
    diagnostics = report_level == "diagnostics"
    nrows = 2 if diagnostics else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 10.5 if diagnostics else 5.4), squeeze=False)
    _plot_auc_compare(axes[0, 0], layer_indices, raw_direction, residual_direction,
                      raw_within, residual_within)
    _plot_middle_compare(axes[0, 1], layer_indices, raw_direction, residual_direction)
    if diagnostics:
        _plot_transfer_feature_bars(axes[1, 0], raw_direction, top_k, title_prefix="Raw",
                                    residualized=False)
        _plot_transfer_feature_bars(axes[1, 1], residual_direction, top_k,
                                    title_prefix="Residualized", residualized=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def build_fused_one(train_outputs: list[str], test_output: str, plots_root: str, folds: int,
                    seed: int, top_k: int, report_level: str, bundle_cache: dict,
                    tokenizers: dict) -> dict:
    """One combination: train on ``train_outputs``, test on ``test_output``."""
    plots_dir = fused_plots_dir(plots_root, train_outputs, test_output)
    os.makedirs(plots_dir, exist_ok=True)

    def get_bundle(path: str):
        key = os.path.abspath(path)
        if key not in bundle_cache:
            bundle_cache[key] = load_run_with_controls(path, tokenizers)
        return bundle_cache[key]

    train_bundles = [get_bundle(path) for path in train_outputs]
    test_bundle = get_bundle(test_output)
    check_bundle_set(train_bundles + [test_bundle])
    layer_indices = [int(value) for value in test_bundle.run.layer_indices]
    covariate_names = list(test_bundle.covariate_names)

    raw_within = within_extremes_reference(test_bundle.run, folds, seed)
    residual_within = within_residual_reference(test_bundle, folds, seed)
    raw_direction = transfer_direction(train_bundles, test_bundle, residualized=False)
    residual_direction = transfer_direction(train_bundles, test_bundle, residualized=True)
    finalize_direction(raw_direction, raw_within, layer_indices)
    finalize_direction(residual_direction, residual_within, layer_indices)

    common = {
        "train_runs": [run_info(bundle) for bundle in train_bundles],
        "test_run": run_info(test_bundle),
        "layer_indices": layer_indices,
        "folds": int(folds),
        "seed": int(seed),
        "informative_auc_threshold": INFORMATIVE_AUC,
    }
    _write_json(os.path.join(plots_dir, "fused_transfer_summary.json"), {
        **common,
        "within_test_cv_auc": raw_within,
        "direction": jsonable_direction(raw_direction, top_k),
        "plots_dir": plots_dir,
        "plots": ["fused_transfer_report.png"],
    })
    _write_json(os.path.join(plots_dir, "fused_transfer_residual_summary.json"), {
        **common,
        "within_residual_test_cv_auc": residual_within,
        "direction": jsonable_direction(residual_direction, top_k),
        "residualization": {
            "covariates": covariate_names,
            "fit": "per layer on the fused train pool's extreme rows only",
            "applied_to": "all rows in the held-out test run before scoring",
        },
        "plots_dir": plots_dir,
        "plots": ["fused_transfer_report.png"],
    })

    out_png = os.path.join(plots_dir, "fused_transfer_report.png")
    _fused_report_plot(layer_indices, raw_direction, residual_direction, raw_within,
                       residual_within, top_k, report_level, out_png)
    payload = {
        "kind": "fused_transfer_report",
        "plots_dir": plots_dir,
        "train_runs": common["train_runs"],
        "test_run": common["test_run"],
        "layer_indices": layer_indices,
        "top_k": int(top_k),
        "folds": int(folds),
        "seed": int(seed),
        "report_level": report_level,
        "residualization": {"covariates": covariate_names, "fit": "fused train extremes only"},
        "within_test_cv_auc": raw_within,
        "within_residual_test_cv_auc": residual_within,
        "raw_direction": jsonable_direction(raw_direction, top_k),
        "residual_direction": jsonable_direction(residual_direction, top_k),
        "cache_summaries": ["fused_transfer_summary.json", "fused_transfer_residual_summary.json"],
        "plots": ["fused_transfer_report.png"],
    }
    _write_json(os.path.join(plots_dir, "fused_transfer_report.json"), payload)
    print(f"[{LOG}] wrote {out_png}")

    for name, d in (("raw", raw_direction), ("residualized", residual_direction)):
        ret = d.get("retention")
        print(f"[{LOG}]   {name:<12s} -> {d['test']}: mean AUC (informative) "
              f"{d['mean_transfer_auc_informative']:.3f}, retention "
              f"{'n/a' if ret is None else f'{ret:.2f}'}")
    return payload


def run(runs: list[str], plots_root: str, folds: int = 5, seed: int = 0, top_k: int = 10,
        report_level: str = "diagnostics") -> dict:
    """Run every 2-train/1-test combination over ``runs`` (at least three)."""
    jobs = build_fused_jobs(runs)
    summary = {
        "kind": "fused_transfer_sweep",
        "runs": list(runs),
        "train_run_count": 2,
        "folds": int(folds),
        "seed": int(seed),
        "top_k": int(top_k),
        "report_level": report_level,
        "plots_root": os.path.join(plots_root, "transfer_sweep"),
        "jobs": [],
    }
    bundle_cache: dict = {}
    tokenizers: dict = {}
    for idx, job in enumerate(jobs, start=1):
        label = (f"{' + '.join(tag_of(path) for path in job['train_outputs'])} -> "
                 f"{tag_of(job['test_output'])}")
        record = {
            "index": idx,
            "label": label,
            "train_outputs": list(job["train_outputs"]),
            "test_output": job["test_output"],
            "plots_dir": fused_plots_dir(plots_root, job["train_outputs"], job["test_output"]),
        }
        summary["jobs"].append(record)
        print(f"[{LOG}] [{idx}/{len(jobs)}] {label}")
        payload = build_fused_one(job["train_outputs"], job["test_output"], plots_root, folds,
                                  seed, top_k, report_level, bundle_cache, tokenizers)
        record["status"] = "ok"
        record["report_json"] = os.path.join(payload["plots_dir"], "fused_transfer_report.json")

    sweep_root = os.path.join(plots_root, "transfer_sweep")
    os.makedirs(sweep_root, exist_ok=True)
    summary_path = os.path.join(sweep_root, "fused_transfer_sweep_summary.json")
    _write_json(summary_path, summary)
    print(f"[{LOG}] wrote {summary_path}")
    return summary
