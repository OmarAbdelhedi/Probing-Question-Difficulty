"""Build the MuSiQue question set: the structural hop count, 2hop / 3hop / 4hop.

MuSiQue composes each question from single-hop steps, so the hop count is
``len(question_decomposition)`` (fallback: the ``2hop__`` / ``3hop1__`` /
``4hop2__`` id prefix).

    python -m src.datasets.build_musique --input data/raw/musique/musique_ans_v1.0_dev.jsonl \\
        --output data/questions_musique/musique.jsonl --per-class 300 --seed 0

Input: ``musique_ans_v1.0_dev.jsonl`` from https://github.com/StonyBrookNLP/musique.
"""

import argparse
import re
from collections import Counter

from src.datasets.common import build_and_write, load_flexible_json

_HOP_ID_RE = re.compile(r"(\d+)hop")


def n_hops(ex: dict) -> int:
    """Number of gold single-hop steps (fallback: the id prefix)."""
    qd = ex.get("question_decomposition")
    if isinstance(qd, list) and qd:
        return len(qd)
    match = _HOP_ID_RE.match(str(ex.get("id", "")))
    return int(match.group(1)) if match else 0


def make_label(ex: dict):
    """``<n>hop``, or None to drop an example whose hop count is unknown."""
    h = n_hops(ex)
    return f"{h}hop" if h > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the labeled MuSiQue question set.")
    ap.add_argument("--input", required=True,
                    help="MuSiQue file (jsonl or json), e.g. musique_ans_v1.0_dev.jsonl.")
    ap.add_argument("--output", default="data/questions_musique/musique.jsonl",
                    help="Output jsonl path (default: data/questions_musique/musique.jsonl).")
    ap.add_argument("--per-class", type=int, default=300,
                    help="Questions per class; classes are balanced to the smallest. 0 = all.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = ap.parse_args()

    data = load_flexible_json(args.input)
    print(f"[musique] loaded {len(data)} examples from {args.input}")
    print(f"[musique] hop-count histogram: {dict(sorted(Counter(n_hops(e) for e in data).items()))}")

    build_and_write(
        data,
        tag="musique",
        label_fn=make_label,
        id_keys=("id", "_id"),
        order_labels=sorted,  # sampling order alphabetical (see common.py)
        per_class=args.per_class,
        seed=args.seed,
        output=args.output,
        too_few_hint="Is this the MuSiQue answerable dev file?",
    )


if __name__ == "__main__":
    main()
