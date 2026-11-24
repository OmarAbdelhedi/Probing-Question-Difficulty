"""Build the 2WikiMultiHopQA question set: binary easy/hard by evidence-chain length.

2Wiki has no difficulty label, so one is derived from the number of gold
evidence items (``evidences``; fallbacks ``evidence``, ``supporting_facts``,
``supporting_facts_titles``): at most two -> ``easy``, at least four ->
``hard``. Questions with exactly three are dropped to sharpen the contrast.

    python -m src.datasets.build_2wiki --input data/raw/2wiki/dev.json \\
        --output data/questions_2wiki/2wiki.jsonl --per-class 300 --seed 0

Input: ``dev.json`` from https://github.com/Alab-NII/2wikimultihop.
"""

import argparse
from collections import Counter

from src.datasets.common import build_and_write, load_flexible_json

EASY_MAX_HOPS = 2  # <= this many evidence items -> "easy"
HARD_MIN_HOPS = 4  # >= this many evidence items -> "hard"


def n_hops(ex: dict) -> int:
    """Length of the gold evidence chain (fallbacks: supporting facts)."""
    for key in ("evidences", "evidence", "supporting_facts", "supporting_facts_titles"):
        val = ex.get(key)
        if val:
            return len(val)
    return 0


def make_label(ex: dict):
    """``easy`` / ``hard``, or None to drop the example."""
    h = n_hops(ex)
    if h <= EASY_MAX_HOPS:
        return "easy"
    if h >= HARD_MIN_HOPS:
        return "hard"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the labeled 2WikiMultiHopQA question set.")
    ap.add_argument("--input", required=True, help="2Wiki json file (e.g. dev.json).")
    ap.add_argument("--output", default="data/questions_2wiki/2wiki.jsonl",
                    help="Output jsonl path (default: data/questions_2wiki/2wiki.jsonl).")
    ap.add_argument("--per-class", type=int, default=300,
                    help="Questions per class; classes are balanced to the smallest. 0 = all.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = ap.parse_args()

    data = load_flexible_json(args.input)
    print(f"[2wiki] loaded {len(data)} examples from {args.input}")
    print(f"[2wiki] evidence-chain length histogram: "
          f"{dict(sorted(Counter(n_hops(e) for e in data).items()))}")

    build_and_write(
        data,
        tag="2wiki",
        label_fn=make_label,
        id_keys=("_id", "id"),
        order_labels=list,  # sampling order = first occurrence (see common.py)
        per_class=args.per_class,
        seed=args.seed,
        output=args.output,
        too_few_hint="Is this the 2Wiki dev.json?",
    )


if __name__ == "__main__":
    main()
