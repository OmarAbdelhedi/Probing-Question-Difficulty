"""Tests for src/pipeline.py's pure helpers (flatten_tree, _validate_io).

The module imports torch + transformers at the top level, so this file skips
wherever transformers is absent (e.g. a laptop with only the analysis stack);
the full pipeline is exercised by src/sanity_checks.py on the cluster.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.pipeline import _validate_io, flatten_tree  # noqa: E402


def node(nid, depth, children=(), stop_reason=None, parse_failed=False):
    return {"id": nid, "question": f"q {nid}", "depth": depth,
            "children": list(children), "stop_reason": stop_reason,
            "parse_failed": parse_failed}


def test_flatten_tree_preorder_and_parent_ids():
    tree = node("root", 0, children=[
        node("root.0", 1, stop_reason="atomic"),
        node("root.1", 1, children=[node("root.1.0", 2, stop_reason="max_depth")]),
    ])
    flat = flatten_tree(tree)
    assert [r["id"] for r in flat] == ["root", "root.0", "root.1", "root.1.0"]  # pre-order
    assert [r["parent_id"] for r in flat] == [None, "root", "root", "root.1"]
    assert set(flat[0]) == {"id", "parent_id", "depth", "question",
                            "stop_reason", "parse_failed"}


def test_validate_io_missing_input_dir(tmp_path):
    with pytest.raises(NotADirectoryError):
        _validate_io(str(tmp_path / "nope"), str(tmp_path / "out"))


def test_validate_io_creates_output_dir(tmp_path):
    out = tmp_path / "a" / "b"
    _validate_io(str(tmp_path), str(out))
    assert out.is_dir()
