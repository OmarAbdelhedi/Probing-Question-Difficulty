"""Convert HotpotQA into FATHOM's labeled-question jsonl.

Unlike 2Wiki and MuSiQue -- where difficulty is *derived* from the gold evidence
chain (hop count) -- HotpotQA ships a NATIVE, human-annotated difficulty label:
``level`` in {easy, medium, hard}. That makes HotpotQA the one dataset here whose
hardness axis is a HUMAN judgement rather than a structural hop count, so it
COMPLEMENTS the others: evidence that the root activation encodes difficulty along
BOTH a structural axis (MuSiQue/2Wiki hops) AND a human-perceived axis (HotpotQA
level) is far stronger than either axis alone. And because ``level`` is an ORDERED
3-class vocabulary that FATHOM's ``order_labels`` already knows (easy < medium <
hard; see src/run_data.py ``_HARDNESS_ORDER``), it drops straight into Tier 2g
(interpolation across depth) and Tier 2t (cross-dataset transfer: easy<->low,
hard<->high) with no special-casing.

HotpotQA also exposes ``type`` in {bridge, comparison} -- a reasoning-TYPE axis,
not a difficulty axis -- offered here as an alternative ``--label-scheme``.

Like the sibling builders, this prints a length-per-label report BEFORE any GPU
time, so the length<->label confound is visible up front. HotpotQA's 'hard'
questions are only mildly longer than 'easy' ones (length is a WEAKER confound
here than in MuSiQue, whose composed questions grow with hops), but Tier 2b still
adjudicates it.

Input: the raw HotpotQA distribution -- a JSON ARRAY of objects with keys
``question``, ``_id``, ``level``, ``type``, ``supporting_facts``, ``context`` --
e.g. ``hotpot_train_v1.1.json`` from https://hotpotqa.github.io . NOTE: for the
``level`` scheme you MUST use the TRAIN split -- HotpotQA's easy/medium/hard
annotation is constant (all ``hard``) on dev/distractor and absent on test, so
only train carries the contrast (the downloader defaults to train). Output: jsonl
of ``{"id", "question", "label"}`` consumable by ``main.py --input``.

Pure stdlib -- no torch / datasets needed.

Example:
    python -m src.datasets.build_hotpot --input data/raw/hotpot/hotpot_train_v1.1.json \\
        --output data/questions_hotpot/hotpot.jsonl --label-scheme level --per-class 300
"""

import argparse
from collections import Counter

from src.datasets.common import build_and_write, load_flexible_json

# Label ordering: easy -> medium -> hard, for the log previews AND the bucket
# sampling loop. NOTE: because the per-label shuffles share one RNG (see
# common.py), this order helps pin WHICH questions get sampled -- it is
# not just cosmetic. (The analysis re-orders labels via src.run_data.order_labels.)
_LEVEL_RANK = {"easy": 0, "medium": 1, "hard": 2}


def level_of(ex: dict) -> str:
    """Native human difficulty label, normalized: 'easy' / 'medium' / 'hard' / ''."""
    return str(ex.get("level", "")).strip().lower()


def type_of(ex: dict) -> str:
    """Native reasoning type, normalized: 'bridge' / 'comparison' / ''."""
    return str(ex.get("type", "")).strip().lower()


def n_supporting(ex: dict) -> int:
    """Number of gold supporting facts (sentence-level; ~2-4 -- HotpotQA is 2-hop)."""
    sf = ex.get("supporting_facts")
    return len(sf) if isinstance(sf, list) else 0


def make_label(ex: dict, scheme: str):
    """Return the label string for one example, or None to DROP it."""
    if scheme == "level":
        # Native graded difficulty (easy < medium < hard) -- the headline scheme.
        return level_of(ex) or None
    if scheme == "level_binary":
        # Clean easy-vs-hard contrast; drop 'medium' (and anything unlabeled).
        lv = level_of(ex)
        if lv == "easy":
            return "easy"
        if lv == "hard":
            return "hard"
        return None
    if scheme == "type":
        # Reasoning TYPE (bridge vs comparison), not difficulty -- an alt axis.
        return type_of(ex) or None
    raise ValueError(f"unknown --label-scheme: {scheme!r}")


def _display_order(labels) -> list:
    """Label order for sampling and the log previews: levels by difficulty, else alphabetical."""
    return sorted(set(labels), key=lambda lab: (_LEVEL_RANK.get(str(lab), 99), str(lab)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a FATHOM jsonl from HotpotQA.")
    ap.add_argument("--input", required=True,
                    help="HotpotQA json file (e.g. hotpot_train_v1.1.json; 'level' needs the train split).")
    ap.add_argument("--output", default="data/questions_hotpot/hotpot.jsonl",
                    help="Output jsonl path (default: data/questions_hotpot/hotpot.jsonl).")
    ap.add_argument("--label-scheme", choices=["level", "level_binary", "type"],
                    default="level",
                    help="Native graded difficulty 'level' (easy/medium/hard, default), "
                         "'level_binary' (easy vs hard, drop medium), or reasoning 'type' "
                         "(bridge vs comparison).")
    ap.add_argument("--per-class", type=int, default=300,
                    help="Cap per class; classes are balanced to the smallest. 0 = all.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = ap.parse_args()

    data = load_flexible_json(args.input)
    print(f"[hotpot] loaded {len(data)} examples from {args.input}")
    print(f"[hotpot] level histogram     : {dict(Counter(level_of(e) or '?' for e in data))}")
    print(f"[hotpot] type histogram      : {dict(Counter(type_of(e) or '?' for e in data))}")
    print(f"[hotpot] #supporting-facts   : {dict(sorted(Counter(n_supporting(e) for e in data).items()))}")

    hint = ("HotpotQA dev/test are all level='hard' (test has none) -- use the TRAIN split "
            "for level schemes, or LABEL_SCHEME=type on dev"
            if args.label_scheme in ("level", "level_binary") else "try --label-scheme level")
    build_and_write(
        data,
        tag="hotpot",
        label_fn=lambda ex: make_label(ex, args.label_scheme),
        id_keys=("_id", "id"),
        order_labels=_display_order,  # pinned sampling order (easy<medium<hard) -- see common.py
        per_class=args.per_class,
        seed=args.seed,
        output=args.output,
        too_few_hint=f"({hint})",
        length_note="HotpotQA difficulty is HUMAN-annotated (not hop count), so length tracks it "
                    "only weakly -- if the means are close, the probe reads difficulty, not length; "
                    "Tier 2b residualizes any residual gap.",
    )


if __name__ == "__main__":
    main()
