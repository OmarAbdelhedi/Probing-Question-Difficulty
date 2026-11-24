"""Loading a finished extraction run (the output of ``main.py``).

A run directory holds ``metadata.jsonl`` (one record per question) and
``activations/<id>.npz`` (one ``[n_layers, hidden_dim]`` array under the key
``"root"``). :func:`load_run` returns everything an analysis needs, with
labels ordered from easiest to hardest.

numpy + stdlib only, so this layer imports without matplotlib or torch.
"""

import json
import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# Known difficulty words, easiest first; any other label sorts alphabetically
# after them (so "2hop" < "3hop" < "4hop"). Matching is case-insensitive.
_HARDNESS_ORDER = (
    "trivial", "very_easy", "very easy", "easy", "simple",
    "medium", "moderate", "intermediate",
    "hard", "difficult", "very_hard", "very hard", "expert",
)


def load_metadata(metadata_path: str) -> list[dict]:
    """Read a run's ``metadata.jsonl``; every record needs ``id`` and ``label``.

    Blank lines are skipped. Raises ``FileNotFoundError`` if the file is
    missing and ``ValueError`` (naming the line) on malformed or empty input.
    Extra keys are kept as they are.
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"metadata.jsonl not found at {metadata_path!r}. Run main.py first "
            "and point the analysis at the same --output directory."
        )

    records: list[dict] = []
    with open(metadata_path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
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
            for key in ("id", "label"):
                if key not in record:
                    raise ValueError(
                        f"Missing key {key!r} in {metadata_path} (line {line_no})"
                    )
            records.append(record)

    if not records:
        raise ValueError(f"No question records found in {metadata_path!r} (empty file?).")
    return records


def order_labels(labels) -> list[str]:
    """Unique labels, easiest first (see ``_HARDNESS_ORDER``), others alphabetical."""
    rank = {name: i for i, name in enumerate(_HARDNESS_ORDER)}
    return sorted(
        set(labels),
        key=lambda lab: (rank.get(str(lab).lower(), len(_HARDNESS_ORDER)), str(lab)),
    )


class RootActivations(NamedTuple):
    X: np.ndarray          # float32 [n_questions, n_layers, hidden_dim]
    y: list                # label per row of X
    ids: list              # question id per row of X
    questions: list        # question text per row of X
    layer_indices: list    # model layer index of each array row
    missing: list          # ids skipped: .npz absent or without a "root" key


def load_root_activations(metadata_path: str, activations_dir: str,
                          log_tag: str | None = None) -> RootActivations:
    """Load every question's activation array, aligned with its label and text.

    Questions whose ``.npz`` is missing or has no ``"root"`` key are skipped
    and listed in ``missing`` (with a warning when ``log_tag`` is given).
    ``layer_indices`` comes from the run's ``layers`` provenance (``"all"`` ->
    ``0..n_layers-1``; an explicit list is used as is when its length matches).

    Raises ``FileNotFoundError`` if nothing loads and ``ValueError`` if arrays
    disagree in shape (questions from different runs).
    """
    records = load_metadata(metadata_path)

    ids, labels, questions, vecs, missing = [], [], [], [], []
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
            arr = np.asarray(npz["root"], dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(
                f"{npz_path}: 'root' must be 2D [n_layers, hidden_dim], got {arr.shape}"
            )
        if shape0 is None:
            shape0 = arr.shape
        elif arr.shape != shape0:
            raise ValueError(
                f"Inconsistent activation shapes: {qid} is {arr.shape}, expected "
                f"{shape0}. All questions must come from one run (same layers/pooling)."
            )

        ids.append(qid)
        labels.append(rec["label"])
        questions.append(str(rec.get("original_question", "")))
        vecs.append(arr)
        if layers_provenance is None:
            layers_provenance = rec.get("layers", "all")

    if not vecs:
        raise FileNotFoundError(
            f"No activations loaded from {activations_dir!r}. Run main.py first."
        )

    X = np.stack(vecs, axis=0)
    n_layers = X.shape[1]
    if layers_provenance in (None, "all"):
        layer_indices = list(range(n_layers))
    else:
        layer_indices = list(layers_provenance)
        if len(layer_indices) != n_layers:
            layer_indices = list(range(n_layers))

    if missing and log_tag is not None:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        print(f"[{log_tag}]   WARNING: {len(missing)} question(s) had no usable "
              f".npz, skipped: {shown}")
    return RootActivations(X, labels, ids, questions, layer_indices, missing)


@dataclass
class RunData:
    """One finished run, as returned by :func:`load_run`."""

    name: str               # run directory basename, e.g. "run_musique"
    run_dir: str
    metadata_path: str
    activations_dir: str
    X: np.ndarray           # float32 [n_questions, n_layers, hidden_dim]
    y: np.ndarray           # object array [n_questions]: label per row of X
    ids: list               # question id per row of X
    questions: list         # question text per row of X
    labels: list            # unique labels, easiest first
    counts: dict            # label -> number of rows
    layer_indices: list     # model layer index per array row
    missing: list           # ids skipped by load_root_activations
    run_config: dict        # run_config.json ({} if absent or unreadable)


def load_run(run_dir: str, log_tag: str = "probe") -> RunData:
    """Load a run directory written by ``main.py``.

    Raises ``FileNotFoundError`` if it has no ``activations/`` or nothing loads;
    the analysis entry points catch this and print a SKIP line.
    """
    metadata_path = os.path.join(run_dir, "metadata.jsonl")
    activations_dir = os.path.join(run_dir, "activations")
    if not os.path.isdir(activations_dir):
        raise FileNotFoundError(
            f"no activations dir at {activations_dir!r} (run main.py to produce it)"
        )

    print(f"[{log_tag}] loading activations from {activations_dir}")
    X, y, ids, questions, layer_indices, missing = load_root_activations(
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
                   activations_dir=activations_dir, X=X, y=y, ids=ids,
                   questions=questions, labels=labels, counts=counts,
                   layer_indices=layer_indices, missing=missing, run_config=run_config)


def resolve_plots_dir(plots_dir: str | None, run_dir: str) -> str:
    """Return (and create) the figures directory; defaults to ``<run_dir>/figures``."""
    if plots_dir is None:
        plots_dir = os.path.join(run_dir, "figures")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir
