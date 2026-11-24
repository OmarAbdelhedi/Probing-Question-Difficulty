"""Build the HotpotQA question set: the native human-annotated ``level``.

HotpotQA labels each question easy / medium / hard. Only the TRAIN split
varies this label (dev/distractor is all ``hard``; test has none), so the
question set is sampled from ``hotpot_train_v1.1.json``.

    python -m src.datasets.build_hotpot --input data/raw/hotpot/hotpot_train_v1.1.json \\
        --output data/questions_hotpot/hotpot.jsonl --per-class 300 --seed 0

Input: ``hotpot_train_v1.1.json`` from https://hotpotqa.github.io.
"""

import argparse
from collections import Counter

from src.datasets.common import build_and_write, load_flexible_json

_LEVEL_RANK = {"easy": 0, "medium": 1, "hard": 2}


def level_of(ex: dict) -> str:
    """``level`` normalised: 'easy' / 'medium' / 'hard' / ''."""
    return str(ex.get("level", "")).strip().lower()


def make_label(ex: dict):
    """The example's level, or None to drop it when it has none."""
    return level_of(ex) or None


def _display_order(labels) -> list:
    """Sampling order: easy < medium < hard, anything else alphabetical after."""
    return sorted(set(labels), key=lambda lab: (_LEVEL_RANK.get(str(lab), 99), str(lab)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the labeled HotpotQA question set.")
    ap.add_argument("--input", required=True,
                    help="HotpotQA json file; must be the train split (hotpot_train_v1.1.json).")
    ap.add_argument("--output", default="data/questions_hotpot/hotpot.jsonl",
                    help="Output jsonl path (default: data/questions_hotpot/hotpot.jsonl).")
    ap.add_argument("--per-class", type=int, default=300,
                    help="Questions per class; classes are balanced to the smallest. 0 = all.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = ap.parse_args()

    data = load_flexible_json(args.input)
    print(f"[hotpot] loaded {len(data)} examples from {args.input}")
    print(f"[hotpot] level histogram: {dict(Counter(level_of(e) or '?' for e in data))}")

    build_and_write(
        data,
        tag="hotpot",
        label_fn=make_label,
        id_keys=("_id", "id"),
        order_labels=_display_order,  # sampling order easy < medium < hard (see common.py)
        per_class=args.per_class,
        seed=args.seed,
        output=args.output,
        too_few_hint="HotpotQA dev/test are all level='hard'; use the TRAIN split.",
    )


if __name__ == "__main__":
    main()
