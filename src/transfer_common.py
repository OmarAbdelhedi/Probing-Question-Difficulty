"""Shared transfer/report helpers for FATHOM.

This module centralizes the mechanics that used to be repeated across the
dataset, pairwise-transfer, and fused-transfer scripts:

* loading finished runs with question-surface controls,
* checking that runs are compatible,
* fitting raw or surface-residualized transfer axes,
* computing middle-class interpolation, and
* extracting best-layer helping/hurting feature contributions.

The residual control set is intentionally fixed here so every report removes
the same covariates: char_len, word_count, cap_words, and punct.
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import NamedTuple

import numpy as np

from src.analyze import load_metadata
from src.probe_core import (
    EPS,
    argmax_finite,
    balanced_acc,
    fit_diag_lda,
    middle_positions,
    probe_auc_by_layer,
    project_diag_lda,
    residualize,
    roc_auc,
)
from src.run_data import RunData, load_run
from src.probe_transfer import within_extremes_reference


SURFACE_COVARIATES = ["char_len", "word_count", "cap_words", "punct"]
INFORMATIVE_AUC = 0.70


class RunWithControls(NamedTuple):
    run: RunData
    covariates: np.ndarray
    covariate_names: list[str]


def tag_of(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    return base[4:] if base.startswith("run_") else base


def default_dataset_runs(outputs_root: str) -> list[str]:
    return [
        os.path.join(outputs_root, "run_hotpot"),
        os.path.join(outputs_root, "run_2wiki"),
        os.path.join(outputs_root, "run_musique"),
    ]


def default_fused_runs(outputs_root: str) -> list[str]:
    return [
        os.path.join(outputs_root, "run_2wiki"),
        os.path.join(outputs_root, "run_hotpot"),
        os.path.join(outputs_root, "run_musique"),
    ]


def surface_covariates(metadata_path: str, ids) -> tuple[np.ndarray, list[str]]:
    by_id = {record["id"]: record for record in load_metadata(metadata_path)}
    rows = []
    for qid in ids:
        question = str(by_id.get(qid, {}).get("original_question", ""))
        toks = question.split()
        rows.append(
            [
                float(len(question)),
                float(len(toks)),
                float(sum(1 for i, word in enumerate(toks) if i > 0 and word[:1].isupper())),
                float(question.count(",") + question.count("?")),
            ]
        )
    return np.asarray(rows, dtype=np.float64), list(SURFACE_COVARIATES)


def load_run_with_controls(run_dir: str, log_tag: str = "transfer-suite") -> RunWithControls:
    run = load_run(run_dir, log_tag=log_tag)
    covariates, names = surface_covariates(run.metadata_path, run.ids)
    return RunWithControls(run=run, covariates=covariates, covariate_names=names)


def check_compatible(reference: RunData, other: RunData) -> None:
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
        "output_dir": os.path.abspath(run.run_dir),
        "n_questions": int(run.X.shape[0]),
        "labels": [str(label) for label in run.labels],
        "counts": run.counts,
        "model_name": run.run_config.get("model_name"),
        "pooling": run.run_config.get("pooling"),
    }
    if isinstance(bundle_or_run, RunWithControls):
        out["covariates"] = bundle_or_run.covariate_names
    return out


def resolve_run_dir(run_info_or_path, outputs_root: str) -> str:
    if isinstance(run_info_or_path, str):
        candidates = [run_info_or_path]
    else:
        name = str(run_info_or_path.get("name", ""))
        output_dir = run_info_or_path.get("output_dir")
        base = os.path.basename(str(output_dir).rstrip("/")) if output_dir else ""
        candidates = []
        if name:
            candidates.append(os.path.join(outputs_root, name))
        if base:
            candidates.append(os.path.join(outputs_root, base))
        if output_dir:
            candidates.append(output_dir)

    seen = set()
    tried = []
    for path in candidates:
        if not path:
            continue
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        tried.append(norm)
        if os.path.isdir(norm):
            return norm
    raise FileNotFoundError("could not resolve run directory; tried: " + ", ".join(tried))


def ready_run(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "activations")) and os.path.isfile(
        os.path.join(path, "metadata.jsonl")
    )


def discover_runs(outputs_root: str) -> list[str]:
    if not os.path.isdir(outputs_root):
        raise FileNotFoundError(f"outputs root {outputs_root!r} does not exist")
    runs = []
    for name in sorted(os.listdir(outputs_root)):
        path = os.path.join(outputs_root, name)
        if os.path.isdir(path) and ready_run(path):
            runs.append(path)
    return runs


def build_fused_jobs(run_paths: list[str]) -> list[dict]:
    norm = [os.path.abspath(path) for path in run_paths]
    if len(set(norm)) != len(norm):
        raise ValueError("duplicate run directories detected")
    if len(norm) < 3:
        raise ValueError(f"need at least 3 runs for 2-train/1-test fusion; got {len(norm)}")

    jobs = []
    for test_output in norm:
        candidates = [path for path in norm if path != test_output]
        for combo in combinations(candidates, 2):
            jobs.append({"train_outputs": list(combo), "test_output": test_output})
    return jobs


def fused_plots_dir(plots_root: str, train_outputs: list[str], test_output: str) -> str:
    order = {"2wiki": 0, "musique": 1, "hotpot": 2}
    train_tags = sorted(
        [tag_of(path) for path in train_outputs],
        key=lambda value: (order.get(value, 999), value),
    )
    train_tag = "_".join(train_tags)
    test_tag = tag_of(test_output)
    return os.path.join(plots_root, "transfer_sweep", f"transfer_{train_tag}--{test_tag}")


def extreme_indices(run: RunData):
    low, high = run.labels[0], run.labels[-1]
    low_idx = np.where(run.y == low)[0]
    high_idx = np.where(run.y == high)[0]
    row_idx = np.concatenate([low_idx, high_idx])
    pos = run.y[row_idx] == high
    return low, high, low_idx, high_idx, row_idx, pos


def within_residual_reference(bundle: RunWithControls, folds: int, seed: int):
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


def _fused_name(bundles: list[RunWithControls]) -> str:
    return " + ".join(bundle.run.name for bundle in bundles)


def _train_extreme_pool(train_bundles: list[RunWithControls]):
    train_blocks = []
    cov_blocks = []
    pos_blocks = []
    components = []

    for bundle in train_bundles:
        run = bundle.run
        low, high, _low_idx, _high_idx, row_idx, pos = extreme_indices(run)
        n_low = int((~pos).sum())
        n_high = int(pos.sum())
        if min(n_low, n_high) < 2:
            raise ValueError(
                f"{run.name} has too few extreme rows for transfer ({low}: {n_low}, {high}: {n_high})"
            )
        train_blocks.append(run.X[row_idx])
        cov_blocks.append(bundle.covariates[row_idx])
        pos_blocks.append(pos)
        components.append(
            {
                "run": run.name,
                "low_label": str(low),
                "high_label": str(high),
                "counts": {str(low): n_low, str(high): n_high},
            }
        )

    return (
        np.concatenate(train_blocks, axis=0),
        np.concatenate(cov_blocks, axis=0),
        np.concatenate(pos_blocks, axis=0),
        components,
    )


def transfer_direction(
    train_bundles: list[RunWithControls],
    test_bundle: RunWithControls,
    *,
    residualized: bool,
) -> dict:
    X_train, cov_train, pos_train, train_components = _train_extreme_pool(train_bundles)
    test_run = test_bundle.run
    low_test, high_test, low_idx, high_idx, ext_idx, pos_test = extreme_indices(test_run)
    middles = list(test_run.labels[1:-1])
    mid_idx = {middle: np.where(test_run.y == middle)[0] for middle in middles}

    direction = {
        "train": _fused_name(train_bundles),
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
    if len(train_bundles) == 1:
        train_run = train_bundles[0].run
        direction["train_low"] = str(train_run.labels[0])
        direction["train_high"] = str(train_run.labels[-1])
        direction["train_extreme_counts"] = train_components[0]["counts"]
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
            Xtr_fit, Xtest_eval = residualize(
                Xtr_layer,
                cov_train,
                Xtest_layer,
                test_bundle.covariates,
            )
        else:
            Xtr_fit = Xtr_layer
            Xtest_eval = Xtest_layer

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
        helping = np.maximum(contrib, 0.0)
        hurting = np.minimum(contrib, 0.0)
        weight_norm = float(np.linalg.norm(weight))
        gap_norm = float(np.linalg.norm(test_gap_std))
        cosine = (
            float(np.dot(weight, test_gap_std) / (weight_norm * gap_norm))
            if weight_norm > EPS and gap_norm > EPS
            else float("nan")
        )

        per_layer["helping_mass"].append(float(helping.sum()))
        per_layer["hurting_mass"].append(float(-hurting.sum()))
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


def finalize_direction(direction: dict, within_auc: list[float] | None, layer_indices: list[int]) -> None:
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


def dataset_top_features_from_weights(weights: np.ndarray, top_k: int) -> list[dict]:
    importance = np.asarray(weights, dtype=float) ** 2
    top_dims = np.argsort(importance)[-top_k:][::-1]
    return [
        {
            "dim": int(dim),
            "importance": float(importance[dim]),
            "weight": float(weights[dim]),
            "sign": "high>low" if weights[dim] >= 0 else "high<low",
        }
        for dim in top_dims
    ]
