"""Shared core for the dataset builders (build_musique / build_2wiki / build_hotpot).

Each ``src/datasets/build_*.py`` differs only in how a raw example yields a
label (hop count, native level, type, ...) and in its argparse surface. The
tolerant JSON/JSONL loader and the whole bucket -> balance-to-smallest ->
shuffle -> length-per-label report -> uniquify-ids -> write-jsonl pipeline are
identical across the three and live here.

REPRODUCIBILITY -- why ``order_labels`` exists: the per-label shuffles all
consume ONE shared RNG, so the order the label buckets are visited in changes
which questions get sampled (and the final cross-label shuffle). The three
builders historically visit buckets in different orders, so each pins its own:

    musique: ``sorted``         (alphabetical labels)
    2wiki:   ``list``           (insertion order = first occurrence in the raw file)
    hotpot:  ``_display_order`` (easy < medium < hard, else alphabetical)

Changing a builder's ``order_labels`` silently resamples its question set --
only do that if you intend to regenerate (and re-run) that dataset.
"""

import json
import os
import random


def load_flexible_json(path: str) -> list:
    """Load a dataset file: JSON array, {'data': [...]}, single object, or JSONL."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        obj = json.loads(text)  # whole-file: JSON array or object
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return obj["data"]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass  # not a single JSON value -> treat as JSONL
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_and_write(
    data: list,
    *,
    tag: str,
    label_fn,
    id_keys: tuple,
    order_labels,
    per_class: int,
    seed: int,
    output: str,
    too_few_hint: str,
    length_note: str,
) -> None:
    """Bucket -> balance -> sample -> length report -> uniquify -> write jsonl.

    tag          -- log prefix, e.g. 'musique' (lines print as '[musique] ...').
    label_fn     -- ex -> label string, or None to DROP the example.
    id_keys      -- raw-id key precedence, e.g. ('_id', 'id'); falsy values skip.
    order_labels -- labels-dict -> ordered label list; pinned per builder (see
                    module docstring: changing it resamples the question set).
    per_class    -- cap per class, 0 = no cap; classes balance to the smallest.
    too_few_hint -- appended to the error when <2 labels survive bucketing.
    length_note  -- dataset-specific line closing the length-confound report.
    """
    # Bucket by derived label.
    buckets: dict[str, list] = {}
    for i, ex in enumerate(data):
        question = str(ex.get("question", "")).strip()
        if not question:
            continue
        label = label_fn(ex)
        if label is None:
            continue
        rid = f"q{i}"
        for key in id_keys:
            if ex.get(key):
                rid = str(ex[key])
                break
        buckets.setdefault(label, []).append({"id": rid, "question": question, "label": label})

    if len(buckets) < 2:
        raise SystemExit(f"[{tag}] ERROR: need >=2 labels, got {list(buckets)}. {too_few_hint}")

    # Balance: equal count per class = min(per-class, smallest available class).
    avail = {lab: len(recs) for lab, recs in buckets.items()}
    take_n = min(avail.values())
    if per_class > 0:
        take_n = min(take_n, per_class)

    rng = random.Random(seed)
    selected: list[dict] = []
    for lab in order_labels(buckets):
        recs = buckets[lab]
        rng.shuffle(recs)
        selected.extend(recs[:take_n])
        print(f"[{tag}] label {lab!r}: {avail[lab]} available -> {take_n} kept")
    rng.shuffle(selected)

    # Confound preview: average question length per label (chars + words).
    print(f"[{tag}] --- length-per-label (confound preview) ---")
    by_lab: dict[str, list] = {}
    for r in selected:
        by_lab.setdefault(r["label"], []).append(r["question"])
    for lab in order_labels(by_lab):
        qs = by_lab[lab]
        chars = [len(q) for q in qs]
        words = [len(q.split()) for q in qs]
        print(f"[{tag}]   {lab}: n={len(qs)} | chars mean={sum(chars) / len(chars):.0f} "
              f"| words mean={sum(words) / len(words):.1f}")
    print(f"[{tag}] {length_note}")

    # Guarantee globally-unique ids (load_questions enforces this).
    seen: set = set()
    unique: list[dict] = []
    for r in selected:
        rid = r["id"]
        if rid in seen:
            rid = f"{rid}_{len(unique)}"
            r = {**r, "id": rid}
        seen.add(rid)
        unique.append(r)

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        for r in unique:
            fh.write(json.dumps(r) + "\n")
    print(f"[{tag}] wrote {len(unique)} questions ({take_n}/class) -> {output}")
