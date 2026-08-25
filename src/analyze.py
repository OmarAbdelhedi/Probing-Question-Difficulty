"""Tier-1 validation plots for FATHOM (metadata-only, NO activations).

This module answers ONE question before any activation analysis is trusted:
*does the model's decomposition behavior actually track the human hardness
label?* If hard questions do not grow deeper / larger trees than easy ones,
there is no point looking at the hidden states yet.

It reads a run's ``metadata.jsonl`` (written by ``src/pipeline.py``) and renders
two cheap sanity figures:

  1. **Tree depth & node-count distributions, split by label** -- a 1x2 figure of
     violins (with the raw points overlaid, so small / discrete samples never
     mislead) of per-question max depth and node count, grouped by label.
     Expect easy -> shallow & small, hard -> deep & large.
  2. **Stop-reason composition by label** -- stacked bars of the
     ``atomic`` / ``max_depth`` / ``budget`` fractions over the LEAF nodes of
     each label. Easy questions should be dominated by ``atomic`` (the root was
     answerable directly); hard ones should branch and pick up ``max_depth`` /
     ``budget`` leaves.

Design notes:
  * Imports are deliberately LIGHT -- matplotlib + numpy + stdlib only, NO torch
    / transformers -- so this runs straight after the pipeline on a login node
    or a laptop with no GPU.
  * The ``Agg`` backend is selected BEFORE importing pyplot so the script works
    headless (slurm / no DISPLAY).
  * Labels are free-form strings (``load_questions`` does not constrain them);
    nothing here hardcodes "easy"/"hard". Known hardness words are merely
    ordered intuitively (``order_labels`` in ``src/run_data.py``), everything
    else falls back to alphabetical.
"""

import json
import os
from collections import Counter

import matplotlib

# MUST precede the pyplot import: pick a non-interactive backend so saving
# figures works with no display (slurm batch job / login node / CI).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after backend selection, by design)
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from src.run_data import load_metadata, order_labels, resolve_plots_dir  # noqa: E402


# Canonical leaf stop reasons, in the order they are stacked / reported. These
# mirror the three terminal stop conditions in ``src/decompose.build_tree``.
_CANONICAL_STOP_REASONS = ("atomic", "max_depth", "budget")

# Fixed colors so the stop-reason plot is comparable across runs.
_STOP_REASON_COLORS = {
    "atomic": "#4C72B0",     # blue   -- model judged the node answerable directly
    "max_depth": "#DD8452",  # orange -- hit the recursion hard cap
    "budget": "#C44E52",     # red    -- hit the per-question node budget
}


def summarize_question(record: dict) -> dict:
    """Reduce one question's node list to the metrics the Tier-1 plots need.

    Returns a dict with:
      * ``id`` / ``label``                : straight from the record.
      * ``n_nodes``                       : total nodes in the tree.
      * ``tree_depth``                    : max node depth (root is depth 0, so an
                                            atomic root => 0).
      * ``leaf_stop_reasons`` (Counter)   : counts of non-null ``stop_reason``
                                            (internal nodes carry ``None`` and are
                                            excluded -- only LEAVES have a reason).
      * ``parse_failures``                : how many nodes had ``parse_failed``.
    """
    nodes = record["nodes"]
    depths = [int(n.get("depth", 0)) for n in nodes]
    leaf_stop_reasons = Counter(
        n["stop_reason"] for n in nodes if n.get("stop_reason") is not None
    )
    parse_failures = sum(1 for n in nodes if n.get("parse_failed"))
    return {
        "id": record.get("id"),
        "label": record["label"],
        "n_nodes": len(nodes),
        "tree_depth": max(depths) if depths else 0,
        "leaf_stop_reasons": leaf_stop_reasons,
        "parse_failures": parse_failures,
    }


def _draw_distribution(ax, values, pos: float, color, rng) -> None:
    """Draw one label's distribution at x=``pos``: violin (if well-defined),
    the raw jittered points (always), and a black mean marker.

    The violin is skipped when it would be degenerate -- fewer than 2 points or
    zero spread (all-identical data makes the KDE singular, which is the common
    case for easy questions that are all depth 0 / 1 node). The overlaid points
    are the source of truth; the violin is only a density hint.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return

    if n >= 2 and np.ptp(values) > 0:
        try:
            parts = ax.violinplot([values], positions=[pos], widths=0.6, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.25)
        except Exception:
            pass  # Any KDE hiccup -> fall back to points only.

    # Raw points, jittered horizontally so ties (very common for integer
    # depths / counts) don't stack into a single invisible dot.
    jitter = rng.uniform(-0.09, 0.09, size=n)
    ax.scatter(
        pos + jitter, values, s=28, color=color, alpha=0.75,
        edgecolor="white", linewidth=0.5, zorder=3,
    )

    # Mean marker: a short solid black line across the group.
    mean = float(values.mean())
    ax.plot([pos - 0.22, pos + 0.22], [mean, mean], color="black", linewidth=1.8, zorder=4)


def plot_depth_and_node_count(summaries, labels, label_colors, out_path: str) -> None:
    """Render the depth & node-count distributions (1x2 figure) to ``out_path``."""
    rng = np.random.default_rng(0)  # fixed seed -> reproducible jitter.
    by_label = {lab: [s for s in summaries if s["label"] == lab] for lab in labels}
    positions = list(range(1, len(labels) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    metrics = (
        ("tree_depth", "Tree depth (max node depth)"),
        ("n_nodes", "Node count (per question)"),
    )
    for ax, (key, ylabel) in zip(axes, metrics):
        for pos, lab in zip(positions, labels):
            vals = [s[key] for s in by_label[lab]]
            _draw_distribution(ax, vals, pos, label_colors[lab], rng)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{lab}\n(n={len(by_label[lab])})" for lab in labels])
        ax.set_xlim(0.5, len(labels) + 0.5)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # depths/counts are integers
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.margins(y=0.12)

    axes[0].legend(
        handles=[Line2D([0], [0], color="black", lw=1.8, label="mean")],
        loc="upper left", fontsize=8, framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stop_reason_composition(summaries, labels, out_path: str) -> None:
    """Render the per-label stop-reason composition (stacked bars) to ``out_path``.

    Bars are normalized to FRACTIONS of leaf nodes per label (so labels with
    different question counts are comparable); the raw leaf count behind each
    bar is annotated above it as ``n=...``.
    """
    by_label = {lab: [s for s in summaries if s["label"] == lab] for lab in labels}

    # Aggregate leaf stop reasons per label once.
    label_counts = {}
    for lab in labels:
        c = Counter()
        for s in by_label[lab]:
            c.update(s["leaf_stop_reasons"])
        label_counts[lab] = c
    totals = [sum(label_counts[lab].values()) for lab in labels]

    # Which reasons actually appear -- canonical ones first, then any extras.
    seen = set().union(*(c.keys() for c in label_counts.values())) if label_counts else set()
    reasons = [r for r in _CANONICAL_STOP_REASONS if r in seen]
    reasons += sorted(r for r in seen if r not in _CANONICAL_STOP_REASONS)
    if not reasons:  # no leaves at all (shouldn't happen) -> still draw sane axes.
        reasons = list(_CANONICAL_STOP_REASONS)

    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3.5, 5))
    fallback = plt.get_cmap("tab10")
    bottoms = [0.0] * len(labels)

    for ri, reason in enumerate(reasons):
        color = _STOP_REASON_COLORS.get(reason, fallback(ri % 10))
        heights = [
            (label_counts[lab].get(reason, 0) / totals[li]) if totals[li] else 0.0
            for li, lab in enumerate(labels)
        ]
        ax.bar(positions, heights, bottom=bottoms, width=0.6, color=color,
               edgecolor="white", label=reason)
        # Label any segment tall enough to fit text.
        for li, (h, b) in enumerate(zip(heights, bottoms)):
            if h >= 0.08:
                ax.text(positions[li], b + h / 2, f"{h * 100:.0f}%", ha="center",
                        va="center", color="white", fontsize=9, fontweight="bold")
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # Sample size behind each (fraction) bar.
    for li in positions:
        ax.text(li, 1.02, f"n={totals[li]}", ha="center", va="bottom", fontsize=8, color="0.3")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lab}\n({len(by_label[lab])} q)" for lab in labels])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Fraction of leaf nodes")
    ax.legend(title="stop reason", loc="lower right", framealpha=0.95, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _stats(arr: np.ndarray) -> dict:
    """Summary stats for a 1D array (None-filled if empty)."""
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_by_label(summaries, labels) -> dict:
    """Aggregate per-question summaries into per-label stats for JSON / console."""
    out = {}
    for lab in labels:
        subset = [s for s in summaries if s["label"] == lab]
        depths = np.array([s["tree_depth"] for s in subset], dtype=float)
        n_nodes = np.array([s["n_nodes"] for s in subset], dtype=float)
        reasons = Counter()
        for s in subset:
            reasons.update(s["leaf_stop_reasons"])
        n_leaves = sum(reasons.values())
        out[lab] = {
            "n_questions": len(subset),
            "tree_depth": _stats(depths),
            "n_nodes": _stats(n_nodes),
            "n_leaves": n_leaves,
            "leaf_stop_reasons": dict(reasons),
            "leaf_stop_reason_fractions": (
                {k: round(v / n_leaves, 4) for k, v in reasons.items()} if n_leaves else {}
            ),
            "parse_failures": sum(s["parse_failures"] for s in subset),
        }
    return out


def _print_summary(agg: dict, labels) -> None:
    """Print a compact per-label table to stdout (mirrors the pipeline's style)."""
    print("[analyze] per-label summary:")
    for lab in labels:
        a = agg[lab]
        d, n = a["tree_depth"], a["n_nodes"]
        line = (
            f"[analyze]   {lab}: {a['n_questions']} q | "
            f"depth mean={d['mean']} max={d['max']} | "
            f"nodes mean={n['mean']} max={n['max']} | "
            f"leaves={a['n_leaves']} {dict(a['leaf_stop_reasons'])}"
        )
        if a["parse_failures"]:
            line += f" | parse_failures={a['parse_failures']}"
        print(line)


def run(output_dir: str, metadata_path: str | None = None, plots_dir: str | None = None) -> None:
    """Build the Tier-1 validation artifacts for a finished run.

    Args:
        output_dir: the run directory passed to ``main.py --output``.
        metadata_path: path to ``metadata.jsonl`` (default ``<output>/metadata.jsonl``).
        plots_dir: where to write plots + summary (default ``<repo_root>/figures``,
            created if it does not exist).

    Writes ``tier1_depth_nodecount_by_label.png``,
    ``tier1_stop_reason_composition_by_label.png`` and ``tier1_summary.json``.
    """
    if metadata_path is None:
        metadata_path = os.path.join(output_dir, "metadata.jsonl")
    plots_dir = resolve_plots_dir(plots_dir)

    print(f"[analyze] reading {metadata_path}")
    records = load_metadata(metadata_path)
    summaries = [summarize_question(r) for r in records]
    labels = order_labels(s["label"] for s in summaries)
    cmap = plt.get_cmap("tab10")
    label_colors = {lab: cmap(i % 10) for i, lab in enumerate(labels)}
    print(f"[analyze] {len(summaries)} question(s); labels: {labels}")

    depth_path = os.path.join(plots_dir, "tier1_depth_nodecount_by_label.png")
    stop_path = os.path.join(plots_dir, "tier1_stop_reason_composition_by_label.png")
    plot_depth_and_node_count(summaries, labels, label_colors, depth_path)
    plot_stop_reason_composition(summaries, labels, stop_path)
    print(f"[analyze]   wrote {depth_path}")
    print(f"[analyze]   wrote {stop_path}")

    agg = aggregate_by_label(summaries, labels)
    summary = {
        "metadata_path": metadata_path,
        "n_questions": len(summaries),
        "labels": labels,
        "per_label": agg,
        "plots_dir": os.path.abspath(plots_dir),
        "plots": [os.path.basename(depth_path), os.path.basename(stop_path)],
    }
    summary_path = os.path.join(plots_dir, "tier1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[analyze]   wrote {summary_path}")

    _print_summary(agg, labels)
    print("[analyze] DONE.")
