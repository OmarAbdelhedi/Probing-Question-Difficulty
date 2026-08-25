"""Run-metadata loading + hardness-label ordering for FATHOM.

Two dependency-light helpers that EVERY analysis tier needs at startup, so they
must NOT live inside any one tier:

  * ``load_metadata`` -- read + validate a run's ``metadata.jsonl`` (one JSON
    record per question, as written by ``src/pipeline.py``).
  * ``order_labels``  -- order a label set easy-ish -> hard-ish (known hardness
    words first via ``_HARDNESS_ORDER``, everything else alphabetical).

Pure stdlib -- no numpy / matplotlib / torch -- so Tier 1 (``src/analyze.py``)
and the activation-probe tiers (via ``src/probe_io.py``) share ONE definition
instead of one tier importing them from another.
"""

import json
import os

# Intuitive ordering for common hardness vocabularies; unknown labels sort
# alphabetically AFTER these. Matching is case-insensitive.
_HARDNESS_ORDER = (
    "trivial", "very_easy", "very easy", "easy", "simple",
    "medium", "moderate", "intermediate",
    "hard", "difficult", "very_hard", "very hard", "expert",
)


def load_metadata(metadata_path: str) -> list[dict]:
    """Load a run's ``metadata.jsonl`` into a list of per-question records.

    Each line is one JSON object as written by ``src/pipeline.py`` (keys
    ``id``, ``label``, ``original_question``, ``nodes``, ...). Only ``id``,
    ``label`` and ``nodes`` are required by this analysis.

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
