"""Tier-2b activation-probe CONTROLS for FATHOM.

Tier 2 showed hardness is linearly decodable from the root activation. Before
believing that means the model *represents difficulty*, Tier 2b runs controls on
the SAME root activations:

  1. LENGTH CONTROL -- is the decodability just question length?
       * length-only probe: CV AUC of question CHARACTER-LENGTH alone, and
       * length-RESIDUALIZED probe: regress char length out of every feature
         within each train fold, then re-probe.
  2. MULTI-COVARIATE (SURFACE) CONTROL -- the stronger test: residualize out a
     whole basket of SURFACE features at once (char length, word count, an
     entity proxy = capitalized-word count, numeric-token count, clause
     punctuation, and -- if a tokenizer is available -- exact token count). If
     the probe survives removing ALL of these, the signal is robustly beyond
     surface form, not merely beyond raw length.
  3. PERMUTATION NULL -- shuffle the labels and re-probe many times; the real
     probe must sit FAR above this null band (validates the MEASUREMENT).

A signal that stays well above the null after the multi-covariate residualization
is real, length-/surface-independent reasoning-depth structure. One that collapses
to the null was surface form.

Reuses the shared probe core (``src.probe_core``) and data layer
(``src.run_data``); numpy + matplotlib only, plus an OPTIONAL transformers
tokenizer (token-count covariate, loaded weight-free and skipped gracefully
if unavailable).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless backend; MUST precede the pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.probe_core import probe_auc_by_layer  # noqa: E402
from src.run_data import load_metadata, load_run, order_labels, resolve_plots_dir  # noqa: E402


def _try_load_tokenizer(run_config: dict):
    """Best-effort tokenizer load (for an exact token-count covariate); None on failure.

    Takes ``model_name`` from the run's already-loaded ``run_config.json`` and
    loads only the tokenizer (no model weights, no GPU). Any failure
    (transformers missing, offline, unknown model) degrades gracefully to the
    tokenizer-free covariates.
    """
    model_name = run_config.get("model_name")
    if not model_name:
        print("[probe2b]   (token-count covariate skipped: no model_name in run_config.json)")
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        print(f"[probe2b] token-count covariate enabled via tokenizer for {model_name!r}")
        return tok
    except Exception as exc:  # missing dep / offline / unknown id -> skip the column
        print(f"[probe2b]   (token-count covariate skipped: {type(exc).__name__}: {exc})")
        return None


def _surface_covariates(metadata_path: str, ids, tokenizer=None):
    """Surface-feature matrix ``[n, k]`` aligned to ``ids``, plus the column names.

    Columns (all tokenizer-free except the optional last one):
      char_len, word_count, cap_words (capitalized non-initial words = named-entity
      proxy), num_tokens (tokens containing a digit), punct (',' + '?' = clause /
      structure), and -- if ``tokenizer`` is given -- token_count (exact).
    """
    by_id = {r["id"]: r for r in load_metadata(metadata_path)}
    names = ["char_len", "word_count", "cap_words", "num_tokens", "punct"]
    if tokenizer is not None:
        names.append("token_count")

    rows = []
    for qid in ids:
        q = str(by_id.get(qid, {}).get("original_question", ""))
        toks = q.split()
        row = [
            float(len(q)),                                                       # char_len
            float(len(toks)),                                                    # word_count
            float(sum(1 for i, w in enumerate(toks) if i > 0 and w[:1].isupper())),  # cap_words
            float(sum(1 for w in toks if any(ch.isdigit() for ch in w))),        # num_tokens
            float(q.count(",") + q.count("?")),                                  # punct
        ]
        if tokenizer is not None:
            row.append(float(len(tokenizer.encode(q))))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64), names


def _plot_control(x, raw_auc, resid_auc, ref_auc, ref_label, resid_label, out_path) -> None:
    """Shared control plot: raw vs residualized probe AUC by layer, with refs."""
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


def run(output_dir: str, metadata_path: str | None = None, activations_dir: str | None = None,
        plots_dir: str | None = None, folds: int = 5, seed: int = 0, perms: int = 200) -> None:
    """Build the Tier-2b control artifacts (length + multi-covariate + permutation null).

    Args mirror :func:`src.probe.run`; ``perms`` is the number of label shuffles
    for the null (default 200). Writes ``tier2b_length_control_by_layer.png``,
    ``tier2b_multicovariate_control_by_layer.png``,
    ``tier2b_permutation_null_by_layer.png`` and ``tier2b_summary.json``.
    """
    plots_dir = resolve_plots_dir(plots_dir)
    try:
        data = load_run(output_dir, metadata_path, activations_dir, log_tag="probe2b")
    except FileNotFoundError as exc:
        print(f"[probe2b] SKIP Tier 2b: {exc}")
        return

    X, y, labels_ordered, layer_indices = data.X, data.y, data.labels, data.layer_indices
    n, n_layers, _ = X.shape

    # Controls need a valid 2+-class probe (same guard as Tier 2).
    counts = data.counts
    if len(labels_ordered) < 2 or min(counts.values()) < 2:
        print(f"[probe2b] SKIP: need >=2 labels with >=2 samples each (have {counts}).")
        return

    tokenizer = _try_load_tokenizer(data.run_config)
    cov, cov_names = _surface_covariates(data.metadata_path, data.ids, tokenizer)
    char_len = cov[:, 0]  # char_len is column 0 (the single-covariate length control)
    print(f"[probe2b] surface covariates ({len(cov_names)}): {cov_names}")

    rng = np.random.default_rng(seed)

    # Raw probe.
    raw = probe_auc_by_layer(X, y, labels_ordered, folds, rng)
    raw_auc = np.asarray(raw["auc"], dtype=float)

    # (a) Single-covariate LENGTH control.
    resid_len = probe_auc_by_layer(X, y, labels_ordered, folds, rng, covariate=char_len)
    length_only = probe_auc_by_layer(char_len.reshape(n, 1, 1), y, labels_ordered, folds, rng)
    resid_len_auc = np.asarray(resid_len["auc"], dtype=float)
    length_only_auc = float(length_only["auc"][0])

    # (b) Multi-covariate SURFACE control (the stronger test).
    resid_multi = probe_auc_by_layer(X, y, labels_ordered, folds, rng, covariate=cov)
    surface_only = probe_auc_by_layer(cov.reshape(n, 1, cov.shape[1]), y, labels_ordered, folds, rng)
    resid_multi_auc = np.asarray(resid_multi["auc"], dtype=float)
    surface_only_auc = float(surface_only["auc"][0])

    # (c) Permutation null on the raw probe.
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

    x = np.asarray(layer_indices)

    # ---- Plot A: single-covariate length control ----
    path_len = os.path.join(plots_dir, "tier2b_length_control_by_layer.png")
    _plot_control(
        x, raw_auc, resid_len_auc, length_only_auc,
        f"length-only AUC = {length_only_auc:.2f}", "length-residualized AUC",
        path_len,
    )
    print(f"[probe2b]   wrote {path_len}")

    # ---- Plot B: multi-covariate surface control ----
    path_multi = os.path.join(plots_dir, "tier2b_multicovariate_control_by_layer.png")
    _plot_control(
        x, raw_auc, resid_multi_auc, surface_only_auc,
        f"surface-only AUC = {surface_only_auc:.2f}  ({len(cov_names)} feats)",
        "surface-residualized AUC",
        path_multi,
    )
    print(f"[probe2b]   wrote {path_multi}")

    # ---- Plot C: permutation null ----
    figC, axC = plt.subplots(figsize=(8.2, 4.8))
    axC.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (0.5)")
    axC.fill_between(x, null_p05, null_p95, color="0.7", alpha=0.35,
                     label="shuffled-label null (5–95%)")
    axC.plot(x, null_mean, color="0.45", lw=1.3, label="null mean")
    axC.plot(x, raw_auc, marker="o", ms=4, lw=1.9, color="#C44E52", label="real probe AUC")
    axC.set_ylim(0.0, 1.02)
    axC.set_xlabel("Layer (0 = embeddings)")
    axC.set_ylabel("Cross-validated AUC")
    axC.grid(alpha=0.25)
    axC.set_axisbelow(True)
    axC.legend(loc="lower right", fontsize=8, framealpha=0.95)
    figC.tight_layout()
    path_null = os.path.join(plots_dir, "tier2b_permutation_null_by_layer.png")
    figC.savefig(path_null, dpi=150)
    plt.close(figC)
    print(f"[probe2b]   wrote {path_null}")

    # ---- Summary + plain-language verdict ----
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
        "residualized_auc": resid_len_auc.tolist(),            # length-residualized (column 0)
        "surface_residualized_auc": resid_multi_auc.tolist(),  # all-surface-residualized
        "null_mean": null_mean.tolist(),
        "null_p05": null_p05.tolist(),
        "null_p95": null_p95.tolist(),
        "raw_auc_mean": raw_mean,
        "residualized_auc_mean": resid_len_mean,
        "surface_residualized_auc_mean": resid_multi_mean,
        "null_auc_mean": null_overall,
        "plots_dir": os.path.abspath(plots_dir),
        "plots": [os.path.basename(path_len), os.path.basename(path_multi), os.path.basename(path_null)],
    }
    summary_path = os.path.join(plots_dir, "tier2b_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[probe2b]   wrote {summary_path}")

    drop_len = raw_mean - resid_len_mean
    drop_multi = raw_mean - resid_multi_mean
    null_floor = null_p95_overall + 0.05
    print("[probe2b] ---- verdict ----")
    print(f"[probe2b]   raw probe AUC (mean)            = {raw_mean:.3f}")
    print(f"[probe2b]   length-only AUC                 = {length_only_auc:.3f}")
    print(f"[probe2b]   length-residualized AUC (mean)  = {resid_len_mean:.3f}  (drop {drop_len:+.3f})")
    print(f"[probe2b]   surface-only AUC ({len(cov_names)} feats)        = {surface_only_auc:.3f}")
    print(f"[probe2b]   surface-residualized AUC (mean) = {resid_multi_mean:.3f}  (drop {drop_multi:+.3f})")
    print(f"[probe2b]   shuffled-null AUC (mean)        = {null_overall:.3f}  (p95 {null_p95_overall:.3f})")

    # The headline test: does signal survive removing ALL surface covariates?
    if resid_multi_mean > null_floor:
        print(f"[probe2b]   => SIGNAL SURVIVES the full surface control "
              f"(mean {resid_multi_mean:.2f} vs null p95 {null_p95_overall:.2f}): robust, "
              f"surface-independent reasoning-depth structure (raw drops {drop_multi:.2f}).")
    else:
        print(f"[probe2b]   => signal COLLAPSES after the full surface control "
              f"(mean {resid_multi_mean:.2f} ~ null p95 {null_p95_overall:.2f}): largely surface form.")
    if raw_mean > null_floor:
        print("[probe2b]   => probe machinery OK: real AUC sits well above the shuffled-label null.")
    else:
        print("[probe2b]   => WARNING: real AUC not clearly above null -- inspect the probe.")
    print("[probe2b] DONE.")
