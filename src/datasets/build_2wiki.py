"""Convert 2WikiMultihopQA into FATHOM's labeled-question jsonl.

2Wiki has NO native difficulty label, so this derives one from intrinsic
reasoning structure -- the HOP COUNT (length of the gold evidence chain) -- so
the probe tests a sharp, FATHOM-aligned question: *does the root activation
encode reasoning depth?* rather than vague "difficulty".

It ALSO prints a length-per-label report BEFORE any GPU time, so the
length<->label confound that sank the toy set is visible up front: if the easy
and hard classes have very different average length, expect Tier 2b to flag a
length confound again -- and you can re-pick the label scheme cheaply.

Input: the raw 2Wiki distribution (a JSON array of objects with keys like
``question``, ``evidences``, ``supporting_facts``, ``type``), e.g. ``dev.json``
from https://github.com/Alab-NII/2wikimultihop . Output: jsonl of
``{"id", "question", "label"}`` consumable by ``main.py --input``.

Pure stdlib -- no torch / datasets needed.

Example:
    python -m src.datasets.build_2wiki --input data/raw/2wiki/dev.json \\
        --output data/questions_2wiki/2wiki.jsonl \\
        --label-scheme hops_binary --easy-hops 2 --hard-hops 4 --per-class 300
"""

import argparse
from collections import Counter

from src.datasets.common import build_and_write, load_flexible_json


def n_hops(ex: dict) -> int:
    """Reasoning-hop count: length of the gold evidence chain (fallbacks: supporting facts)."""
    for key in ("evidences", "evidence", "supporting_facts", "supporting_facts_titles"):
        val = ex.get(key)
        if val:
            return len(val)
    return 0


def make_label(ex: dict, scheme: str, easy_hops: int, hard_hops: int):
    """Return the label string for one example, or None to DROP it."""
    if scheme == "type":
        return str(ex.get("type", "unknown"))
    h = n_hops(ex)
    if scheme == "hops_multiclass":
        return f"{h}hop"
    # hops_binary: clean contrast, drop the in-between band.
    if h <= easy_hops:
        return "easy"
    if h >= hard_hops:
        return "hard"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a FATHOM jsonl from 2WikiMultihopQA.")
    ap.add_argument("--input", required=True, help="Path to a 2Wiki json file (e.g. dev.json).")
    ap.add_argument("--output", default="data/questions_2wiki/2wiki.jsonl",
                    help="Output jsonl path (default: data/questions_2wiki/2wiki.jsonl).")
    ap.add_argument("--label-scheme", choices=["hops_binary", "hops_multiclass", "type"],
                    default="hops_binary",
                    help="How to derive the label: hops_binary (easy<=k / hard>=K), "
                         "hops_multiclass ('2hop','4hop',...), or question type.")
    ap.add_argument("--easy-hops", type=int, default=2, help="hops_binary: <= this -> 'easy'.")
    ap.add_argument("--hard-hops", type=int, default=4, help="hops_binary: >= this -> 'hard'.")
    ap.add_argument("--per-class", type=int, default=300,
                    help="Cap per class; classes are balanced to the smallest. 0 = all.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = ap.parse_args()

    data = load_flexible_json(args.input)
    print(f"[2wiki] loaded {len(data)} examples from {args.input}")
    print(f"[2wiki] hop-count histogram : {dict(sorted(Counter(n_hops(e) for e in data).items()))}")
    print(f"[2wiki] type histogram      : {dict(Counter(str(e.get('type', '?')) for e in data))}")

    build_and_write(
        data,
        tag="2wiki",
        label_fn=lambda ex: make_label(ex, args.label_scheme, args.easy_hops, args.hard_hops),
        id_keys=("_id", "id"),
        order_labels=list,  # pinned sampling order (label insertion) -- see common.py
        per_class=args.per_class,
        seed=args.seed,
        output=args.output,
        too_few_hint="Try a different --label-scheme or hop thresholds.",
        length_note="If char/word means are CLOSE across labels, length is controlled; if they "
                    "differ a lot, expect Tier 2b to flag a length confound (re-pick the scheme).",
    )


if __name__ == "__main__":
    main()
