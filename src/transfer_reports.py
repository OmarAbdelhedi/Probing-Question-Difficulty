"""Consolidated dataset, pairwise, and fused transfer reports."""

from __future__ import annotations

import glob
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib

matplotlib.use("Agg")  # headless backend; MUST precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.probe_core import argmax_finite, fit_diag_lda, probe_auc_by_layer, residualize  # noqa: E402
from src.probe_transfer import within_extremes_reference  # noqa: E402
from src.run_data import resolve_plots_dir  # noqa: E402
from src.transfer_common import (  # noqa: E402
    INFORMATIVE_AUC,
    SURFACE_COVARIATES,
    build_fused_jobs,
    check_bundle_set,
    dataset_top_features_from_weights,
    default_dataset_runs,
    default_fused_runs,
    discover_runs,
    extreme_indices,
    finalize_direction,
    fused_plots_dir,
    jsonable_direction,
    load_run_with_controls,
    resolve_run_dir,
    run_info,
    surface_covariates,
    tag_of,
    top_dims_for_layer,
    transfer_direction,
    within_residual_reference,
)


RAW_COLOR = "#C44E52"
RESID_COLOR = "#4C72B0"
GREEN = "#55A868"


def _load_bundles(run_dirs: list[str], log_tag: str) -> list:
    bundles = [load_run_with_controls(path, log_tag=log_tag) for path in run_dirs]
    check_bundle_set(bundles)
    return bundles


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _best_x(values, layer_indices):
    arr = np.asarray(values, dtype=float)
    pos = argmax_finite(arr)
    return int(layer_indices[pos]), float(arr[pos])


# --------------------------------------------------------------------------- #
# Shared plotting helpers                                                     #
# --------------------------------------------------------------------------- #
def _setup_auc_axis(ax, ylabel: str = "AUC") -> None:
    ax.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def _plot_auc_compare(
    ax,
    layer_indices: list[int],
    raw_direction: dict,
    residual_direction: dict,
    raw_within=None,
    residual_within=None,
) -> None:
    x = np.asarray(layer_indices)
    _setup_auc_axis(ax)
    raw_auc = np.asarray(raw_direction["transfer_auc"], dtype=float)
    resid_auc = np.asarray(residual_direction["transfer_auc"], dtype=float)
    ax.plot(x, raw_auc, color=RAW_COLOR, ls=":", marker="o", ms=3.0, lw=1.8, label="raw transfer")
    ax.plot(x, resid_auc, color=RESID_COLOR, marker="o", ms=3.0, lw=1.8, label="residual transfer")
    if raw_within is not None:
        ax.plot(x, np.asarray(raw_within, dtype=float), color=RAW_COLOR, ls="--", lw=1.1, alpha=0.35, label="raw within-test CV")
    if residual_within is not None:
        ax.plot(
            x,
            np.asarray(residual_within, dtype=float),
            color=RESID_COLOR,
            ls="--",
            lw=1.1,
            alpha=0.35,
            label="residual within-test CV",
        )
    for arr, color in [(raw_auc, RAW_COLOR), (resid_auc, RESID_COLOR)]:
        best = argmax_finite(arr)
        ax.scatter([x[best]], [arr[best]], s=64, facecolor="none", edgecolor=color, zorder=5)
    ax.legend(loc="lower right", fontsize=7.5)


def _plot_middle_compare(ax, layer_indices: list[int], raw_direction: dict, residual_direction: dict) -> None:
    x = np.asarray(layer_indices)
    if not (raw_direction.get("test_middles") or residual_direction.get("test_middles")):
        ax.axis("off")
        ax.set_title("No middle class in this held-out test run", fontsize=10)
        return

    ax.axhspan(0.0, 1.0, color=GREEN, alpha=0.10, label="interpolation zone [0, 1]")
    ax.axhline(0.5, color="0.5", ls="--", lw=1, label="midpoint (0.5)")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axhline(1.0, color="0.7", lw=0.8)
    cmap = plt.get_cmap("tab10")
    color_i = 0
    middles = raw_direction.get("test_middles") or residual_direction.get("test_middles")
    for middle in middles:
        color = cmap(color_i % 10)
        if middle in raw_direction.get("interp", {}):
            ax.plot(
                x,
                np.asarray(raw_direction["interp"][middle], dtype=float),
                color=color,
                ls=":",
                marker="^",
                ms=3.0,
                lw=1.5,
                label=f"raw {middle}",
            )
        if middle in residual_direction.get("interp", {}):
            ax.plot(
                x,
                np.asarray(residual_direction["interp"][middle], dtype=float),
                color=color,
                marker="^",
                ms=3.0,
                lw=1.7,
                label=f"residual {middle}",
            )
        color_i += 1
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Position between test extremes")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)


def _feature_xlim(direction: dict, top_k: int) -> tuple[float, float]:
    helping, hurting = top_dims_for_layer(direction, direction["best_layer_index"], top_k)
    values = [item["contribution"] for item in helping + hurting]
    max_abs = max((abs(float(value)) for value in values), default=1.0)
    if max_abs <= 0:
        max_abs = 1.0
    return -1.08 * max_abs, 1.08 * max_abs


def _plot_transfer_feature_bars(ax, direction: dict, top_k: int, *, title_prefix: str, residualized: bool) -> None:
    helping, hurting = top_dims_for_layer(direction, direction["best_layer_index"], top_k)
    dims = helping[::-1] + hurting[::-1]
    values = [entry["contribution"] for entry in dims]
    labels = [f"d{entry['dim']}" for entry in dims]
    colors = [GREEN if value >= 0 else RAW_COLOR for value in values]
    ypos = np.arange(len(dims))
    ax.barh(ypos, values, color=colors, alpha=0.92)
    ax.axvline(0.0, color="0.5", lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(*_feature_xlim(direction, top_k))
    xlabel = "Signed contribution on residualized test extremes" if residualized else "Signed contribution on test extremes"
    ax.set_xlabel(xlabel)
    ax.set_title(
        f"{title_prefix}: {direction['train']} -> {direction['test']}\n"
        f"layer {direction['best_layer_index']}  AUC={direction['best_transfer_auc']:.3f}",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Dataset report                                                              #
# --------------------------------------------------------------------------- #
def _compute_dataset(bundle, folds: int, seed: int, top_k: int) -> dict:
    run = bundle.run
    low, high, _low_idx, _high_idx, row_idx, pos = extreme_indices(run)
    X_ext = run.X[row_idx]
    y_ext = run.y[row_idx]
    cov_ext = bundle.covariates[row_idx]

    raw = probe_auc_by_layer(X_ext, y_ext, [low, high], folds, np.random.default_rng(seed))
    resid = probe_auc_by_layer(
        X_ext,
        y_ext,
        [low, high],
        folds,
        np.random.default_rng(seed),
        covariate=cov_ext,
    )
    if raw is None or raw.get("skipped"):
        raise ValueError(f"raw probe failed for {run.name}")
    if resid is None or resid.get("skipped"):
        raise ValueError(f"residualized probe failed for {run.name}")

    raw_auc = np.asarray(raw["auc"], dtype=float)
    resid_auc = np.asarray(resid["auc"], dtype=float)
    raw_best_pos = argmax_finite(raw_auc)
    resid_best_pos = argmax_finite(resid_auc)

    _mu, _sd, raw_w, _thr = fit_diag_lda(X_ext[:, raw_best_pos, :], pos)
    X_resid, _ = residualize(
        X_ext[:, resid_best_pos, :],
        cov_ext,
        X_ext[:, resid_best_pos, :],
        cov_ext,
    )
    _mu_r, _sd_r, resid_w, _thr_r = fit_diag_lda(X_resid, pos)

    return {
        "name": run.name,
        "run_dir": os.path.abspath(run.run_dir),
        "low_label": str(low),
        "high_label": str(high),
        "layer_indices": [int(value) for value in run.layer_indices],
        "covariates": bundle.covariate_names,
        "raw_auc": [float(value) for value in raw_auc],
        "surface_residualized_auc": [float(value) for value in resid_auc],
        "raw_best_layer_index": int(run.layer_indices[raw_best_pos]),
        "raw_best_auc": float(raw_auc[raw_best_pos]),
        "surface_residualized_best_layer_index": int(run.layer_indices[resid_best_pos]),
        "surface_residualized_best_auc": float(resid_auc[resid_best_pos]),
        "raw_top_features": dataset_top_features_from_weights(raw_w, top_k),
        "surface_residualized_top_features": dataset_top_features_from_weights(resid_w, top_k),
    }


def _plot_dataset_auc(ax, result: dict) -> None:
    x = np.asarray(result["layer_indices"])
    raw_auc = np.asarray(result["raw_auc"], dtype=float)
    resid_auc = np.asarray(result["surface_residualized_auc"], dtype=float)
    ax.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    ax.plot(x, raw_auc, color=RAW_COLOR, lw=1.8, marker="o", ms=3.0, label="raw")
    ax.plot(x, resid_auc, color=RESID_COLOR, lw=1.8, marker="^", ms=3.0, label="residual")
    for key, values, color in [
        ("raw_best_layer_index", raw_auc, RAW_COLOR),
        ("surface_residualized_best_layer_index", resid_auc, RESID_COLOR),
    ]:
        best_layer = result[key]
        best_pos = int(np.where(x == best_layer)[0][0]) if np.any(x == best_layer) else argmax_finite(values)
        ax.scatter([x[best_pos]], [values[best_pos]], s=64, facecolor="none", edgecolor=color, zorder=5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("AUC")
    ax.set_title(f"{result['name']}\n{result['low_label']} -> {result['high_label']}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)


def _plot_dataset_features(ax, features: list[dict], title: str, xmax: float) -> None:
    feats = list(reversed(features))
    values = [item["importance"] for item in feats]
    labels = [f"d{item['dim']}" for item in feats]
    colors = [RESID_COLOR if item["sign"] == "high>low" else RAW_COLOR for item in feats]
    ypos = np.arange(len(feats))
    ax.barh(ypos, values, color=colors, alpha=0.92)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.0, xmax if xmax > 0 else 1.0)
    ax.set_xlabel("Feature importance")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)


def _plot_dataset_report(results: list[dict], out_path: str, top_k: int) -> None:
    fig, axes = plt.subplots(3, len(results), figsize=(18, 14), squeeze=False)
    raw_max = max(item["importance"] for result in results for item in result["raw_top_features"])
    resid_max = max(
        item["importance"] for result in results for item in result["surface_residualized_top_features"]
    )
    for col, result in enumerate(results):
        _plot_dataset_auc(axes[0, col], result)
        _plot_dataset_features(
            axes[1, col],
            result["raw_top_features"],
            f"Raw top {top_k}\nlayer {result['raw_best_layer_index']}  AUC={result['raw_best_auc']:.3f}",
            raw_max * 1.08,
        )
        _plot_dataset_features(
            axes[2, col],
            result["surface_residualized_top_features"],
            f"Residual top {top_k}\nlayer {result['surface_residualized_best_layer_index']}  "
            f"AUC={result['surface_residualized_best_auc']:.3f}",
            resid_max * 1.08,
        )
    axes[1, 0].set_ylabel("raw top features")
    axes[2, 0].set_ylabel("residualized top features")
    fig.legend(
        handles=[
            Patch(facecolor=RESID_COLOR, label="feature increases toward high class"),
            Patch(facecolor=RAW_COLOR, label="feature decreases toward high class"),
        ],
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_dataset_report(
    run_outputs: list[str] | None,
    outputs_root: str,
    plots_root: str,
    folds: int,
    seed: int,
    top_k: int,
    report_level: str,
) -> dict:
    del report_level
    plots_root = resolve_plots_dir(plots_root)
    run_dirs = run_outputs if run_outputs else default_dataset_runs(outputs_root)
    if len(run_dirs) != 3:
        raise ValueError("dataset report expects exactly three run directories")
    bundles = _load_bundles(run_dirs, log_tag="transfer-suite-dataset")
    results = [_compute_dataset(bundle, folds, seed, top_k) for bundle in bundles]
    order = {"run_hotpot": 0, "run_2wiki": 1, "run_musique": 2}
    results.sort(key=lambda item: order.get(item["name"], 999))

    png_path = os.path.join(plots_root, "dataset_report.png")
    json_path = os.path.join(plots_root, "dataset_report.json")
    _plot_dataset_report(results, png_path, top_k)
    payload = {
        "kind": "dataset_report",
        "top_k": int(top_k),
        "folds": int(folds),
        "seed": int(seed),
        "residualization": {"covariates": list(SURFACE_COVARIATES)},
        "runs": results,
        "plots": ["dataset_report.png"],
    }
    _write_json(json_path, payload)
    print(f"[transfer-suite] wrote {png_path}")
    print(f"[transfer-suite] wrote {json_path}")
    return payload


# --------------------------------------------------------------------------- #
# Pairwise report                                                             #
# --------------------------------------------------------------------------- #
def discover_pairwise_transfer_dirs(plots_root: str) -> list[str]:
    candidates = sorted(glob.glob(os.path.join(plots_root, "transfer_*")))
    return [
        path
        for path in candidates
        if os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "tier2t_summary.json"))
        and "transfer_sweep" not in os.path.abspath(path).split(os.sep)
    ]


def _pairwise_report_plot(
    transfer_dir: str,
    layer_indices: list[int],
    raw_directions: list[dict],
    residual_directions: list[dict],
    raw_within: dict,
    residual_within: dict,
    top_k: int,
    report_level: str,
    out_path: str,
) -> None:
    diagnostics = report_level == "diagnostics"
    nrows = 4 if diagnostics else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(17, 18 if diagnostics else 9.5), squeeze=False)

    for col, (raw_dir, resid_dir) in enumerate(zip(raw_directions, residual_directions)):
        _plot_auc_compare(
            axes[0, col],
            layer_indices,
            raw_dir,
            resid_dir,
            raw_within.get(raw_dir["test"]),
            residual_within.get(resid_dir["test"]),
        )
        _plot_middle_compare(axes[1, col], layer_indices, raw_dir, resid_dir)
        if diagnostics:
            _plot_transfer_feature_bars(axes[2, col], raw_dir, top_k, title_prefix="Raw", residualized=False)
            _plot_transfer_feature_bars(
                axes[3, col],
                resid_dir,
                top_k,
                title_prefix="Residualized",
                residualized=True,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_pairwise_one(
    transfer_dir: str,
    outputs_root: str,
    folds: int,
    seed: int,
    top_k: int,
    report_level: str,
) -> dict:
    transfer_dir = resolve_plots_dir(transfer_dir)
    summary_path = os.path.join(transfer_dir, "tier2t_summary.json")
    summary = _load_json(summary_path)
    run_a_dir = resolve_run_dir(summary["run_a"], outputs_root)
    run_b_dir = resolve_run_dir(summary["run_b"], outputs_root)
    run_a, run_b = _load_bundles([run_a_dir, run_b_dir], log_tag="transfer-suite-pairwise")
    layer_indices = [int(value) for value in run_a.run.layer_indices]

    raw_within = {
        run_a.run.name: within_extremes_reference(run_a.run, folds, seed),
        run_b.run.name: within_extremes_reference(run_b.run, folds, seed),
    }
    residual_within = {
        run_a.run.name: within_residual_reference(run_a, folds, seed),
        run_b.run.name: within_residual_reference(run_b, folds, seed),
    }

    raw_directions = [
        transfer_direction([run_a], run_b, residualized=False),
        transfer_direction([run_b], run_a, residualized=False),
    ]
    residual_directions = [
        transfer_direction([run_a], run_b, residualized=True),
        transfer_direction([run_b], run_a, residualized=True),
    ]
    for direction in raw_directions:
        finalize_direction(direction, raw_within.get(direction["test"]), layer_indices)
    for direction in residual_directions:
        finalize_direction(direction, residual_within.get(direction["test"]), layer_indices)

    out_png = os.path.join(transfer_dir, "tier2t_transfer_report.png")
    _pairwise_report_plot(
        transfer_dir,
        layer_indices,
        raw_directions,
        residual_directions,
        raw_within,
        residual_within,
        top_k,
        report_level,
        out_png,
    )
    payload = {
        "kind": "pairwise_transfer_report",
        "transfer_dir": os.path.abspath(transfer_dir),
        "source_summary": os.path.abspath(summary_path),
        "top_k": int(top_k),
        "folds": int(folds),
        "seed": int(seed),
        "report_level": report_level,
        "residualization": {"covariates": list(SURFACE_COVARIATES), "fit": "train extremes only"},
        "run_a": run_info(run_a),
        "run_b": run_info(run_b),
        "layer_indices": layer_indices,
        "within_cv_auc": raw_within,
        "within_residual_cv_auc": residual_within,
        "raw_directions": [jsonable_direction(direction, top_k) for direction in raw_directions],
        "residual_directions": [jsonable_direction(direction, top_k) for direction in residual_directions],
        "plots": ["tier2t_transfer_report.png"],
    }
    out_json = os.path.join(transfer_dir, "tier2t_transfer_report.json")
    _write_json(out_json, payload)
    print(f"[transfer-suite] wrote {out_png}")
    print(f"[transfer-suite] wrote {out_json}")
    return payload


def build_pairwise_reports(
    transfer_dirs: list[str] | None,
    outputs_root: str,
    plots_root: str,
    folds: int,
    seed: int,
    top_k: int,
    report_level: str,
) -> list[dict]:
    dirs = transfer_dirs if transfer_dirs else discover_pairwise_transfer_dirs(plots_root)
    if not dirs:
        raise FileNotFoundError("no pairwise transfer folders with tier2t_summary.json found")
    return [
        build_pairwise_one(path, outputs_root, folds, seed, top_k, report_level)
        for path in dirs
    ]


# --------------------------------------------------------------------------- #
# Fused report                                                                #
# --------------------------------------------------------------------------- #
def _fused_report_plot(
    plots_dir: str,
    layer_indices: list[int],
    raw_direction: dict,
    residual_direction: dict,
    raw_within,
    residual_within,
    top_k: int,
    report_level: str,
    out_path: str,
) -> None:
    diagnostics = report_level == "diagnostics"
    nrows = 2 if diagnostics else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 10.5 if diagnostics else 5.4), squeeze=False)
    _plot_auc_compare(axes[0, 0], layer_indices, raw_direction, residual_direction, raw_within, residual_within)
    _plot_middle_compare(axes[0, 1], layer_indices, raw_direction, residual_direction)
    if diagnostics:
        _plot_transfer_feature_bars(axes[1, 0], raw_direction, top_k, title_prefix="Raw", residualized=False)
        _plot_transfer_feature_bars(
            axes[1, 1],
            residual_direction,
            top_k,
            title_prefix="Residualized",
            residualized=True,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_fused_cache(
    plots_dir: str,
    train_bundles: list,
    test_bundle,
    layer_indices: list[int],
    folds: int,
    seed: int,
    raw_within,
    residual_within,
    raw_direction: dict,
    residual_direction: dict,
) -> None:
    raw_summary = {
        "train_runs": [run_info(bundle) for bundle in train_bundles],
        "test_run": run_info(test_bundle),
        "layer_indices": layer_indices,
        "folds": int(folds),
        "seed": int(seed),
        "informative_auc_threshold": INFORMATIVE_AUC,
        "within_test_cv_auc": raw_within,
        "direction": jsonable_direction(raw_direction, top_k=10),
        "plots_dir": os.path.abspath(plots_dir),
        "plots": ["fused_transfer_report.png"],
    }
    residual_summary = {
        "train_runs": [run_info(bundle) for bundle in train_bundles],
        "test_run": run_info(test_bundle),
        "layer_indices": layer_indices,
        "folds": int(folds),
        "seed": int(seed),
        "informative_auc_threshold": INFORMATIVE_AUC,
        "within_residual_test_cv_auc": residual_within,
        "direction": jsonable_direction(residual_direction, top_k=10),
        "residualization": {
            "covariates": list(SURFACE_COVARIATES),
            "fit": "per layer on the fused train pool's extreme rows only",
            "applied_to": "all rows in the held-out test run before scoring",
        },
        "plots_dir": os.path.abspath(plots_dir),
        "plots": ["fused_transfer_report.png"],
    }
    _write_json(os.path.join(plots_dir, "fused_transfer_summary.json"), raw_summary)
    _write_json(os.path.join(plots_dir, "fused_transfer_residual_summary.json"), residual_summary)


def build_fused_one(
    train_outputs: list[str],
    test_output: str,
    plots_root: str,
    folds: int,
    seed: int,
    top_k: int,
    report_level: str,
    bundle_cache: dict[str, object],
) -> dict:
    plots_dir = resolve_plots_dir(fused_plots_dir(plots_root, train_outputs, test_output))

    def get_bundle(path: str):
        key = os.path.abspath(path)
        if key not in bundle_cache:
            bundle_cache[key] = load_run_with_controls(key, log_tag="transfer-suite-fused")
        return bundle_cache[key]

    train_bundles = [get_bundle(path) for path in train_outputs]
    test_bundle = get_bundle(test_output)
    check_bundle_set(train_bundles + [test_bundle])
    layer_indices = [int(value) for value in test_bundle.run.layer_indices]

    raw_within = within_extremes_reference(test_bundle.run, folds, seed)
    residual_within = within_residual_reference(test_bundle, folds, seed)
    raw_direction = transfer_direction(train_bundles, test_bundle, residualized=False)
    residual_direction = transfer_direction(train_bundles, test_bundle, residualized=True)
    finalize_direction(raw_direction, raw_within, layer_indices)
    finalize_direction(residual_direction, residual_within, layer_indices)

    _save_fused_cache(
        plots_dir,
        train_bundles,
        test_bundle,
        layer_indices,
        folds,
        seed,
        raw_within,
        residual_within,
        raw_direction,
        residual_direction,
    )
    out_png = os.path.join(plots_dir, "fused_transfer_report.png")
    _fused_report_plot(
        plots_dir,
        layer_indices,
        raw_direction,
        residual_direction,
        raw_within,
        residual_within,
        top_k,
        report_level,
        out_png,
    )
    payload = {
        "kind": "fused_transfer_report",
        "plots_dir": os.path.abspath(plots_dir),
        "train_runs": [run_info(bundle) for bundle in train_bundles],
        "test_run": run_info(test_bundle),
        "layer_indices": layer_indices,
        "top_k": int(top_k),
        "folds": int(folds),
        "seed": int(seed),
        "report_level": report_level,
        "residualization": {"covariates": list(SURFACE_COVARIATES), "fit": "fused train extremes only"},
        "within_test_cv_auc": raw_within,
        "within_residual_test_cv_auc": residual_within,
        "raw_direction": jsonable_direction(raw_direction, top_k),
        "residual_direction": jsonable_direction(residual_direction, top_k),
        "cache_summaries": ["fused_transfer_summary.json", "fused_transfer_residual_summary.json"],
        "plots": ["fused_transfer_report.png"],
    }
    out_json = os.path.join(plots_dir, "fused_transfer_report.json")
    _write_json(out_json, payload)
    print(f"[transfer-suite] wrote {out_png}")
    print(f"[transfer-suite] wrote {out_json}")
    return payload


def build_fused_reports(
    runs: list[str] | None,
    outputs_root: str,
    plots_root: str,
    folds: int,
    seed: int,
    top_k: int,
    report_level: str,
    dry_run: bool = False,
) -> dict:
    plots_root = resolve_plots_dir(plots_root)
    run_paths = runs if runs else default_fused_runs(outputs_root)
    if not all(os.path.isdir(path) for path in run_paths):
        run_paths = discover_runs(outputs_root)
    jobs = build_fused_jobs(run_paths)
    summary = {
        "kind": "fused_transfer_sweep",
        "runs": [os.path.abspath(path) for path in run_paths],
        "train_run_count": 2,
        "folds": int(folds),
        "seed": int(seed),
        "top_k": int(top_k),
        "report_level": report_level,
        "plots_root": os.path.abspath(os.path.join(plots_root, "transfer_sweep")),
        "dry_run": bool(dry_run),
        "jobs": [],
    }
    bundle_cache = {}
    for idx, job in enumerate(jobs, start=1):
        label = f"{' + '.join(tag_of(path) for path in job['train_outputs'])} -> {tag_of(job['test_output'])}"
        record = {
            "index": idx,
            "label": label,
            "train_outputs": [os.path.abspath(path) for path in job["train_outputs"]],
            "test_output": os.path.abspath(job["test_output"]),
            "plots_dir": os.path.abspath(fused_plots_dir(plots_root, job["train_outputs"], job["test_output"])),
        }
        summary["jobs"].append(record)
        print(f"[transfer-suite] fused [{idx}/{len(jobs)}] {label}")
        if dry_run:
            record["status"] = "planned"
            continue
        try:
            payload = build_fused_one(
                job["train_outputs"],
                job["test_output"],
                plots_root,
                folds,
                seed,
                top_k,
                report_level,
                bundle_cache,
            )
            record["status"] = "ok"
            record["report_json"] = os.path.join(payload["plots_dir"], "fused_transfer_report.json")
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            raise

    sweep_root = resolve_plots_dir(os.path.join(plots_root, "transfer_sweep"))
    summary_path = os.path.join(sweep_root, "fused_transfer_sweep_summary.json")
    _write_json(summary_path, summary)
    print(f"[transfer-suite] wrote {summary_path}")
    return summary
