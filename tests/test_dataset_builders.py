"""Tests for the dataset builders in src/datasets/: the shared bucket ->
balance -> sample -> uniquify -> write core (src.datasets.common) and each
builder's label-derivation functions."""

import json

import pytest

from src.datasets import build_2wiki as wiki
from src.datasets import build_hotpot as hotpot
from src.datasets import build_musique as musique
from src.datasets.common import build_and_write, load_flexible_json
from src.load_questions import load_questions


# --------------------------------------------------------------------------- #
# load_flexible_json                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ('[{"a": 1}, {"a": 2}]', [{"a": 1}, {"a": 2}]),       # JSON array
        ('{"data": [{"a": 1}]}', [{"a": 1}]),                 # {'data': [...]}
        ('{"a": 1}', [{"a": 1}]),                             # single object
        ('{"a": 1}\n\n{"a": 2}\n', [{"a": 1}, {"a": 2}]),     # JSONL
    ],
)
def test_load_flexible_json_formats(tmp_path, text, expected):
    path = tmp_path / "data.json"
    path.write_text(text, encoding="utf-8")
    assert load_flexible_json(str(path)) == expected


# --------------------------------------------------------------------------- #
# build_and_write                                                             #
# --------------------------------------------------------------------------- #
def raw_examples():
    # 5 'easy', 3 'hard', one unlabeled (dropped), one blank question (dropped).
    data = [{"id": f"e{i}", "question": f"easy question {i}?", "label": "easy"}
            for i in range(5)]
    data += [{"id": f"h{i}", "question": f"hard question {i}?", "label": "hard"}
             for i in range(3)]
    data += [{"id": "x", "question": "no label?"},
             {"id": "y", "question": "   ", "label": "easy"}]
    return data


def run_build(tmp_path, data, per_class=0, seed=0, name="out.jsonl"):
    output = str(tmp_path / "q" / name)
    build_and_write(
        data, tag="test", label_fn=lambda ex: ex.get("label"), id_keys=("id",),
        order_labels=sorted, per_class=per_class, seed=seed, output=output,
        too_few_hint="hint", length_note="note",
    )
    with open(output, encoding="utf-8") as fh:
        return output, [json.loads(line) for line in fh]


def test_build_and_write_balances_to_smallest_class(tmp_path):
    _, recs = run_build(tmp_path, raw_examples())
    by_label = {}
    for r in recs:
        by_label.setdefault(r["label"], []).append(r)
    assert {lab: len(v) for lab, v in by_label.items()} == {"easy": 3, "hard": 3}


def test_build_and_write_per_class_cap(tmp_path):
    _, recs = run_build(tmp_path, raw_examples(), per_class=2)
    assert len(recs) == 4


def test_build_and_write_output_satisfies_load_questions(tmp_path):
    output, recs = run_build(tmp_path, raw_examples())
    import os
    loaded = load_questions(os.path.dirname(output))  # the consumer contract
    assert [q["id"] for q in loaded] == [r["id"] for r in recs]


def test_build_and_write_deterministic_per_seed(tmp_path):
    out1, _ = run_build(tmp_path, raw_examples(), seed=3, name="a.jsonl")
    out2, _ = run_build(tmp_path, raw_examples(), seed=3, name="b.jsonl")
    assert open(out1).read() == open(out2).read()


def test_build_and_write_uniquifies_duplicate_raw_ids(tmp_path):
    data = [{"id": "dup", "question": f"q {i}?", "label": lab}
            for i, lab in enumerate(["easy", "easy", "hard", "hard"])]
    _, recs = run_build(tmp_path, data)
    ids = [r["id"] for r in recs]
    assert len(set(ids)) == len(ids) == 4


def test_build_and_write_needs_two_labels(tmp_path):
    data = [{"id": "a", "question": "q?", "label": "easy"}]
    with pytest.raises(SystemExit, match="need >=2 labels"):
        run_build(tmp_path, data)


# --------------------------------------------------------------------------- #
# MuSiQue labels                                                              #
# --------------------------------------------------------------------------- #
def test_musique_n_hops_decomposition_then_id_fallback():
    assert musique.n_hops({"question_decomposition": [1, 2, 3]}) == 3
    assert musique.n_hops({"id": "4hop2__abc"}) == 4
    assert musique.n_hops({}) == 0


def test_musique_make_label_schemes():
    assert musique.make_label({"question_decomposition": [1, 2]},
                              "hops_multiclass", 2, 4) == "2hop"
    assert musique.make_label({}, "hops_multiclass", 2, 4) is None  # 0 hops
    mk = lambda h: musique.make_label({"question_decomposition": list(range(h))},  # noqa: E731
                                      "hops_binary", 2, 4)
    assert (mk(2), mk(3), mk(4)) == ("easy", None, "hard")  # drop the in-between band


# --------------------------------------------------------------------------- #
# 2Wiki labels                                                                #
# --------------------------------------------------------------------------- #
def test_2wiki_n_hops_key_precedence():
    assert wiki.n_hops({"evidences": [1, 2], "supporting_facts": [1, 2, 3, 4]}) == 2
    assert wiki.n_hops({"supporting_facts": [1, 2, 3]}) == 3
    assert wiki.n_hops({}) == 0


def test_2wiki_make_label_schemes():
    assert wiki.make_label({"type": "comparison"}, "type", 2, 4) == "comparison"
    assert wiki.make_label({"evidences": [1, 2]}, "hops_multiclass", 2, 4) == "2hop"
    assert wiki.make_label({"evidences": [1, 2]}, "hops_binary", 2, 4) == "easy"
    assert wiki.make_label({"evidences": [1, 2, 3]}, "hops_binary", 2, 4) is None


# --------------------------------------------------------------------------- #
# HotpotQA labels                                                             #
# --------------------------------------------------------------------------- #
def test_hotpot_make_label_schemes():
    assert hotpot.make_label({"level": " Hard "}, "level") == "hard"  # normalized
    assert hotpot.make_label({}, "level") is None
    assert hotpot.make_label({"level": "medium"}, "level_binary") is None  # dropped
    assert hotpot.make_label({"level": "easy"}, "level_binary") == "easy"
    assert hotpot.make_label({"type": "bridge"}, "type") == "bridge"
    with pytest.raises(ValueError, match="unknown --label-scheme"):
        hotpot.make_label({}, "nope")


def test_hotpot_display_order():
    assert hotpot._display_order(["hard", "bridge", "easy", "medium"]) == \
        ["easy", "medium", "hard", "bridge"]
