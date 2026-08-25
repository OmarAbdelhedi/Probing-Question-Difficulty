"""Unit tests for src/load_questions.py -- the input contract."""

import json

import pytest

from src.load_questions import load_questions


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")


def test_files_sorted_lines_in_order_extra_keys_stripped(tmp_path):
    write_jsonl(tmp_path / "b.jsonl",
                [{"id": "b1", "question": "qb", "label": "hard", "extra": 1}])
    write_jsonl(tmp_path / "a.jsonl",
                [{"id": "a1", "question": "qa1", "label": "easy"},
                 {"id": "a2", "question": "qa2", "label": "easy"}])
    qs = load_questions(str(tmp_path))
    assert [q["id"] for q in qs] == ["a1", "a2", "b1"]  # sorted files, natural lines
    assert all(set(q) == {"id", "question", "label"} for q in qs)


def test_no_jsonl_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_questions(str(tmp_path))


@pytest.mark.parametrize(
    "lines, match",
    [
        ([""], "Empty line"),
        (["{not json"], "Invalid JSON"),
        (['["a", "b"]'], "not a JSON object"),
        (['{"id": "x", "question": "q"}'], "Missing key"),
        (['{"id": "  ", "question": "q", "label": "easy"}'], "Empty or non-string"),
        (['{"id": 3, "question": "q", "label": "easy"}'], "Empty or non-string"),
    ],
)
def test_malformed_lines_raise_with_line_number(tmp_path, lines, match):
    write_jsonl(tmp_path / "q.jsonl", lines)
    with pytest.raises(ValueError, match=match) as exc:
        load_questions(str(tmp_path))
    assert "line 1" in str(exc.value)


def test_duplicate_ids_across_files_raise_naming_first_location(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", [{"id": "dup", "question": "q1", "label": "easy"}])
    write_jsonl(tmp_path / "b.jsonl", [{"id": "dup", "question": "q2", "label": "hard"}])
    with pytest.raises(ValueError, match="Duplicate id 'dup'") as exc:
        load_questions(str(tmp_path))
    assert "a.jsonl:1" in str(exc.value)  # where it was first seen
