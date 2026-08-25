"""The shared DATA layer for FATHOM's analysis tiers -- no plotting, no torch.

Every tier starts from the same artifacts: a run directory holding
``metadata.jsonl`` (one record per question, written by ``src/pipeline.py``)
and ``activations/<id>.npz`` (per-node pooled hidden states). The loaders for
those artifacts live HERE, once: Tier 1 (``src/analyze.py``) reads the
metadata records, and the probe tiers (2 / 2b / 2g / 2t) load the root
activations through :func:`load_run` -- so no tier imports another tier's
plotting module just to get at the data.

Contents: ``load_metadata`` (validated ``metadata.jsonl`` records),
``order_labels`` (hardness-aware label ordering, with ``_HARDNESS_ORDER``),
``load_root_activations`` / :class:`RootActivations`, :func:`load_run` /
:class:`RunData` (THE shared per-tier ``run()`` preamble), and
``resolve_plots_dir``.

numpy + stdlib ONLY -- like ``src/probe_core``, this layer imports (and
unit-tests) without matplotlib / torch / transformers.
"""

import json
import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.probe_core import REPO_ROOT

# Intuitive ordering for common hardness vocabularies; unknown labels sort
# alphabetically AFTER these. Matching is case-insensitive.
_HARDNESS_ORDER = (
    "trivial", "very_easy", "very easy", "easy", "simple",
    "medium", "moderate", "intermediate",
    "hard", "difficult", "very_hard", "very hard", "expert",
)


# --------------------------------------------------------------------------- #
# metadata.jsonl                                                              #
# --------------------------------------------------------------------------- #
def load_metadata(metadata_path: str) -> list[dict]:
    """Load a run's ``metadata.jsonl`` into a list of per-question records.

    Each line is one JSON object as written by ``src/pipeline.py`` (keys
    ``id``, ``label``, ``original_question``, ``nodes``, ...). Only ``id``,
    ``label`` and ``nodes`` are required by the analyses.

    Raises
    ------
    FileNotFoundError
        If ``metadata_path`` does not exist (most likely: ``main.py`` has not
        been run for this ``--output`` yet).
    ValueError
        If a line is not valid JSON, is missing a required key, has a non-list
        ``nodes`` field, or the file contains no records. The message names the
        offending 1-based line number.
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"metadata.jsonl not found at {metadata_path!r}. Run main.py first "
            "(point this analysis' --output at the SAME directory you passed to "
            "main.py --output)."
        )

    records: list[dict] = []
    with open(metadata_path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                # Tolerate stray blank lines (e.g. a trailing newline).
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {metadata_path} (line {line_no}): {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line is not a JSON object in {metadata_path} (line {line_no}): "
                    f"got {type(record).__name__}"
                )
            for key in ("id", "label", "nodes"):
                if key not in record:
                    raise ValueError(
                        f"Missing key {key!r} in {metadata_path} (line {line_no})"
                    )
            if not isinstance(record["nodes"], list):
                raise ValueError(
                    f"'nodes' must be a list in {metadata_path} (line {line_no}); "
                    f"got {type(record['nodes']).__name__}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"No question records found in {metadata_path!r} (empty file?).")
    return records


def order_labels(labels) -> list[str]:
    """Return the unique labels ordered intuitively (easy-ish before hard-ish).

    Known hardness words (see ``_HARDNESS_ORDER``) come first in that order;
    any other label sorts alphabetically after them. Matching is
    case-insensitive but the original label string is preserved.
    """
    rank = {name: i for i, name in enumerate(_HARDNESS_ORDER)}
    return sorted(
        set(labels),
        key=lambda lab: (rank.get(str(lab).lower(), len(_HARDNESS_ORDER)), str(lab)),
    )


# --------------------------------------------------------------------------- #
# Root activations & whole runs                                               #
# --------------------------------------------------------------------------- #
class RootActivations(NamedTuple):
    """What :func:`load_root_activations` returns (unpacks like a plain tuple)."""

    X: np.ndarray          # float32 [n_questions, n_layers, hidden_dim]
    y: list                # label per row of X
    ids: list              # question id per row of X
    layer_indices: list    # real model layer index for each array row
    missing: list          # ids skipped: .npz absent or had no "root" key


def load_root_activations(metadata_path: str, activations_dir: str,
                          log_tag: str | None = None) -> RootActivations:
    """Load every question's ROOT-node activation, aligned to its label.

    Returns a :class:`RootActivations` with:
      * ``X``            : float32 ndarray ``[n_questions, n_layers, hidden_dim]``.
      * ``y``            : list[str] label per row of ``X``.
      * ``ids``          : list[str] question id per row.
      * ``layer_indices``: list[int] the real model layer index for each array row
        (from the run's ``layers`` provenance: ``"all"`` -> ``range(n_layers)``;
        an explicit list -> that list). Used purely for the x-axis.
      * ``missing``      : list[str] ids whose ``.npz`` was absent or had no
        ``"root"`` key (skipped, not fatal). If ``log_tag`` is given (the
        caller's console prefix, e.g. ``"probe2b"``), a WARNING naming the
        first few skipped ids is printed HERE, once, instead of by every tier.

    Raises:
        FileNotFoundError: if NO root activations could be loaded.
        ValueError: if loaded arrays disagree in shape (mixed runs).
    """
    records = load_metadata(metadata_path)

    ids: list[str] = []
    labels: list[str] = []
    vecs: list[np.ndarray] = []
    missing: list[str] = []
    shape0 = None
    layers_provenance = None

    for rec in records:
        qid = rec["id"]
        npz_path = os.path.join(activations_dir, f"{qid}.npz")
        if not os.path.isfile(npz_path):
            missing.append(qid)
            continue
        with np.load(npz_path) as npz:
            if "root" not in npz:
                missing.append(qid)
                continue
            arr = np.asarray(npz["root"], dtype=np.float32)  # [n_layers, hidden_dim]

        if arr.ndim != 2:
            raise ValueError(
                f"{npz_path}: 'root' must be 2D [n_layers, hidden_dim], got {arr.shape}"
            )
        if shape0 is None:
            shape0 = arr.shape
        elif arr.shape != shape0:
            raise ValueError(
                f"Inconsistent activation shapes: {qid} is {arr.shape}, expected "
                f"{shape0}. All questions must come from ONE run (same layers/pooling)."
            )

        ids.append(qid)
        labels.append(rec["label"])
        vecs.append(arr)
        if layers_provenance is None:
            layers_provenance = rec.get("layers", "all")

    if not vecs:
        raise FileNotFoundError(
            f"No root activations loaded from {activations_dir!r}. Run main.py first "
            "(Tier 2 needs the activations/ dir, not just metadata.jsonl)."
        )

    X = np.stack(vecs, axis=0)  # [n, n_layers, hidden_dim]
    n_layers = X.shape[1]
    if layers_provenance in (None, "all"):
        layer_indices = list(range(n_layers))
    else:
        layer_indices = list(layers_provenance)
        if len(layer_indices) != n_layers:  # provenance/shape mismatch -> positional
            layer_indices = list(range(n_layers))

    if missing and log_tag is not None:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        print(f"[{log_tag}]   WARNING: {len(missing)} question(s) had no usable "
              f".npz/root, skipped: {shown}")
    return RootActivations(X, labels, ids, layer_indices, missing)


@dataclass
class RunData:
    """One finished run's probe inputs + provenance, as loaded by :func:`load_run`."""

    name: str               # run dir basename (Tier 2t may suffix " (A)"/" (B)")
    run_dir: str
    metadata_path: str      # resolved (defaults applied inside run_dir)
    activations_dir: str    # resolved (defaults applied inside run_dir)
    X: np.ndarray           # float32 [n_questions, n_layers, hidden_dim]
    y: np.ndarray           # object array [n_questions]: label per row of X
    ids: list               # question id per row of X
    labels: list            # unique labels, ordered easy-ish -> hard-ish
    counts: dict            # label -> number of rows
    layer_indices: list     # real model layer index per array row
    missing: list           # ids skipped by load_root_activations
    run_config: dict        # run_config.json contents ({} if absent/unreadable)


def load_run(run_dir: str, metadata_path: str | None = None,
             activations_dir: str | None = None, log_tag: str = "probe") -> RunData:
    """Load one finished run for probing -- THE shared per-tier ``run()`` preamble.

    Every tier starts the same way: default the ``metadata.jsonl`` /
    ``activations/`` paths inside ``run_dir``, load + sanity-log the root
    activations, order the labels, and read ``run_config.json`` provenance
    (absent/unreadable -> ``{}``, not fatal). That lives here, ONCE;
    ``log_tag`` is the calling tier's console prefix (e.g. ``"probe2t"``).

    Raises:
        FileNotFoundError: if the activations dir is missing, or it yields no
            usable activation. Tiers catch this and print their SKIP line.
    """
    if metadata_path is None:
        metadata_path = os.path.join(run_dir, "metadata.jsonl")
    if activations_dir is None:
        activations_dir = os.path.join(run_dir, "activations")
    if not os.path.isdir(activations_dir):
        raise FileNotFoundError(
            f"no activations dir at {activations_dir!r} (run main.py to produce activations/)"
        )

    print(f"[{log_tag}] loading root activations from {activations_dir}")
    X, y, ids, layer_indices, missing = load_root_activations(
        metadata_path, activations_dir, log_tag=log_tag
    )

    run_config: dict = {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                run_config = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[{log_tag}]   WARNING: could not read {cfg_path!r}: {exc}")

    name = os.path.basename(os.path.normpath(run_dir)) or run_dir
    y = np.asarray(y, dtype=object)
    labels = order_labels(y)
    counts = {str(lab): int(np.sum(y == lab)) for lab in labels}
    print(f"[{log_tag}]   {name}: n={X.shape[0]} · {X.shape[1]} layers · "
          f"hidden_dim={X.shape[2]} · labels={labels} {counts}")
    return RunData(name=name, run_dir=run_dir, metadata_path=metadata_path,
                   activations_dir=activations_dir, X=X, y=y, ids=ids, labels=labels,
                   counts=counts, layer_indices=layer_indices, missing=missing,
                   run_config=run_config)


def resolve_plots_dir(plots_dir: str | None) -> str:
    """Default ``plots_dir`` to ``<repo_root>/figures`` and make sure it exists."""
    if plots_dir is None:
        plots_dir = os.path.join(REPO_ROOT, "figures")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir
