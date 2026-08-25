"""Load labeled QA questions for the FATHOM pipeline.

INPUT SCHEMA
------------
The input is a directory containing one or more ``*.jsonl`` files. Each file has
ONE JSON object per line, with EXACTLY these three string keys::

    {"id": "q001", "question": "What is the capital of France?", "label": "easy"}

Field meaning:
  - ``id``       : unique question identifier (used to key output artifacts,
                   e.g. ``activations/<id>.npz``). Must be non-empty and UNIQUE
                   across ALL ``*.jsonl`` files in the directory.
  - ``question`` : the natural-language question text (the ROOT of the
                   decomposition tree built later in the pipeline).
  - ``label``    : the supervision label for the question (free-form string;
                   this module does not constrain its values).

Questions are returned in deterministic (file, line) order: files are visited
in sorted filename order, and lines within a file in their natural order.

SWITCHING INPUT FORMAT
----------------------
To switch the input format (e.g. to CSV, a single JSON file, or a
directory-of-files where each question is its own file), change ONLY this
module. Keep the public contract identical: ``load_questions(input_dir)`` must
still return a ``list[dict]`` where every dict has the keys ``id``,
``question``, and ``label``. The rest of the pipeline depends on nothing else
here.
"""

import glob
import json
import os

# Required keys for every question record, per the input schema above.
_REQUIRED_KEYS = ("id", "question", "label")


def load_questions(input_dir: str) -> list[dict]:
    """Read every ``*.jsonl`` file in ``input_dir`` and return the questions.

    Each line of each file must be a JSON object with the keys ``id``,
    ``question``, and ``label`` (see the module docstring for the schema).

    Returns
    -------
    list[dict]
        Question records in (file, line) order. Files are processed in sorted
        filename order; lines within a file keep their natural order.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` contains no ``*.jsonl`` files.
    ValueError
        If any line is empty, is not valid JSON, is missing one of the required
        keys, has an empty ``id``, or repeats an ``id`` already seen in any
        file. The error message names the offending file and 1-based line
        number.
    """
    # Discover input files. Sort by full path for deterministic, reproducible
    # ordering regardless of filesystem enumeration order.
    pattern = os.path.join(input_dir, "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No *.jsonl files found in input directory: {input_dir!r}"
        )

    questions: list[dict] = []
    seen_ids: dict[str, str] = {}  # id -> "file:line" where it was first seen

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                # Skip nothing implicitly: blank lines are treated as errors so
                # malformed inputs surface loudly rather than being dropped.
                line = raw_line.strip()
                if not line:
                    raise ValueError(
                        f"Empty line in {file_path} (line {line_no})"
                    )

                # Parse the JSON object for this line.
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {file_path} (line {line_no}): {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line is not a JSON object in {file_path} "
                        f"(line {line_no}): got {type(record).__name__}"
                    )

                # Validate required keys are all present.
                missing = [k for k in _REQUIRED_KEYS if k not in record]
                if missing:
                    raise ValueError(
                        f"Missing key(s) {missing} in {file_path} "
                        f"(line {line_no})"
                    )

                qid = record["id"]

                # id must be a non-empty string.
                if not isinstance(qid, str) or not qid.strip():
                    raise ValueError(
                        f"Empty or non-string 'id' in {file_path} "
                        f"(line {line_no})"
                    )

                # ids must be unique across ALL files.
                if qid in seen_ids:
                    raise ValueError(
                        f"Duplicate id {qid!r} in {file_path} (line {line_no}); "
                        f"first seen at {seen_ids[qid]}"
                    )
                seen_ids[qid] = f"{file_path}:{line_no}"

                # Keep only the contract keys, in a fresh dict, so downstream
                # consumers see a clean, predictable record shape.
                questions.append(
                    {
                        "id": qid,
                        "question": record["question"],
                        "label": record["label"],
                    }
                )

    return questions
