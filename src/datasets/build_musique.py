"""Convert MuSiQue into FATHOM's labeled-question jsonl.

MuSiQue (Trivedi et al. 2022) composes multi-hop questions bottom-up from
single-hop steps, so the HOP COUNT is the construction parameter -- a genuine,
graded (2 / 3 / 4) reasoning-depth axis, decorrelated from any single question
template and explicitly built to resist shortcut reasoning. That makes hop count
a far cleaner "difficulty" signal than 2Wiki's type-tied 2-vs-4 split.

Hop count = ``len(question_decomposition)`` (the gold single-hop sub-questions),
with a fallback to the ``id`` prefix (``2hop__``, ``3hop1__``, ``4hop2__`` ...).

This prints a length-per-label report BEFORE any GPU time -- ESPECIALLY important
for MuSiQue, whose composed questions get longer with more hops, so length is the
confound to watch (Tier 2b residualizes it; you can also length-bin).

Bonus: ``question_decomposition`` is a gold decomposition tree, so a later
full-decomposition run can validate FATHOM's own decomposition against it.

Input: a MuSiQue file -- JSONL (``musique_ans_v1.0_dev.jsonl``) or a JSON array.
Output: jsonl of ``{"id", "question", "label"}`` for ``main.py --input``.
Pure stdlib -- no torch / datasets needed.

Example:
    python -m src.datasets.build_musique --input data/raw/musique/musique_ans_v1.0_dev.jsonl \\
        --output data/questions_musique/musique.jsonl --label-scheme hops_multiclass --per-class 300
"""

import argparse
import re
from collections import Counter

from src.datasets.common import build_and_write, load_flexible_json

_HOP_ID_RE = re.compile(r"(\d+)hop")


def n_hops(ex: dict) -> int:
    """Reasoning-hop count: number of gold single-hop steps (fallback: id prefix)."""
    qd = ex.get("question_decomposition")
    if isinstance(qd, list) and qd:
        return len(qd)
    match = _HOP_ID_RE.match(str(ex.get("id", "")))
    return int(match.group(1)) if match else 0


def make_label(ex: dict, scheme: str, easy_hops: int, hard_hops: int):
    """Return the label string for one example, or None to DROP it."""
    h = n_hops(ex)
    if scheme == "hops_multiclass":
        return f"{h}hop" if h > 0 else None
    # hops_binary: clean 2-vs-4 contrast, drop the in-between band.
    if 0 < h <= easy_hops:
        return "easy"
    if h >= hard_hops:
        return "hard"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a FATHOM jsonl from MuSiQue.")
    ap.add_argument("--input", required=True, help="MuSiQue file (jsonl or json), e.g. musique_ans_v1.0_dev.jsonl.")
    ap.add_argument("--output", default="data/questions_musique/musique.jsonl",
                    help="Output jsonl path (default: data/questions_musique/musique.jsonl).")
    ap.add_argument("--label-scheme", choices=["hops_multiclass", "hops_binary"],
                    default="hops_multiclass",
                    help="Graded '2hop'/'3hop'/'4hop' (default) or binary easy(<=k)/hard(>=K).")
    ap.add_argument("--easy-hops", type=int, default=2, help="hops_binary: <= this -> 'easy'.")
    ap.add_argument("--hard-hops", type=int, default=4, help="hops_binary: >= this -> 'hard'.")
    ap.add_argument("--per-class", type=int, default=300,
                    help="Cap per class; classes are balanced to the smallest. 0 = all.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = ap.parse_args()

    data = load_flexible_json(args.input)
    print(f"[musique] loaded {len(data)} examples from {args.input}")
    print(f"[musique] hop-count histogram: {dict(sorted(Counter(n_hops(e) for e in data).items()))}")

    build_and_write(
        data,
        tag="musique",
        label_fn=lambda ex: make_label(ex, args.label_scheme, args.easy_hops, args.hard_hops),
        id_keys=("id", "_id"),
        order_labels=sorted,  # pinned sampling order (alphabetical) -- see common.py
        per_class=args.per_class,
        seed=args.seed,
        output=args.output,
        too_few_hint="Try a different --label-scheme or hop thresholds.",
        length_note="MuSiQue length GROWS with hops -- if the means differ a lot, the probe will "
                    "partly read length; Tier 2b residualizes it (and you can length-bin to be safe).",
    )


if __name__ == "__main__":
    main()
