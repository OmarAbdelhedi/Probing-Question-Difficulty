"""Unit tests for src/decompose.py: JSON-array parsing and the recursive tree
builder, driven through stub decomposers (the seam -- no model needed).

The module imports torch at the top level (for the HF-backed decomposer
factory), so the whole file skips on a torch-free machine.
"""

import pytest

pytest.importorskip("torch")

from src.decompose import _normalize, build_tree, parse_subquestions  # noqa: E402


# --------------------------------------------------------------------------- #
# parse_subquestions                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ('["a", "b"]', (["a", "b"], False)),
        ("[]", ([], False)),                                  # legitimate atomic signal
        ('Sure! Here you go: ["a"] hope that helps', (["a"], False)),
        ('```json\n["a", "b"]\n```', (["a", "b"], False)),    # fenced output
        ('["keep [brackets] inside", "b"]', (["keep [brackets] inside", "b"], False)),
        ('"[" is a bracket, anyway: ["x"]', (["x"], False)),  # bracket inside a string literal
        ("[1, 2]", (["1", "2"], False)),                      # non-strings coerced
        ('["  ", ""]', ([], False)),                          # blanks dropped -> atomic
        ('["a", ', ([], True)),                               # unbalanced
        ('{"a": 1}', ([], True)),                             # no array at all
        ("no array here", ([], True)),
        ('["a" "b"]', ([], True)),                            # balanced but invalid JSON
    ],
)
def test_parse_subquestions(text, expected):
    assert parse_subquestions(text) == expected


def test_parse_subquestions_non_string_input():
    assert parse_subquestions(None) == ([], True)
    assert parse_subquestions(42) == ([], True)


# --------------------------------------------------------------------------- #
# _normalize (the parent-restatement guard's equality)                        #
# --------------------------------------------------------------------------- #
def test_normalize_equivalences():
    assert _normalize("  The CAPITAL of   France??") == _normalize("the capital of france")
    assert _normalize("What is X?") == _normalize("what is x")
    assert _normalize("What is X?") != _normalize("What is Y?")


# --------------------------------------------------------------------------- #
# build_tree                                                                  #
# --------------------------------------------------------------------------- #
def recording_decomposer(mapping=None, default=(), parse_failed=False):
    """Stub decompose_fn answering from ``mapping`` (else ``default``), and
    recording every ``(question, parent_question)`` call."""
    mapping = mapping or {}
    calls = []

    def fn(question, parent_question=None):
        calls.append((question, parent_question))
        if callable(default):
            subs = mapping.get(question, default(question))
        else:
            subs = mapping.get(question, list(default))
        return list(subs), parse_failed

    fn.calls = calls
    return fn


def collect(node):
    out = [node]
    for child in node["children"]:
        out.extend(collect(child))
    return out


def test_atomic_root():
    fn = recording_decomposer()
    tree = build_tree("Q?", 0, 3, decompose_fn=fn, node_budget=10)
    assert tree["id"] == "root" and tree["depth"] == 0
    assert tree["stop_reason"] == "atomic" and tree["parse_failed"] is False
    assert tree["children"] == []


def test_two_level_tree_structure():
    fn = recording_decomposer({"Q?": ["A?", "B?"], "B?": ["C?"]})
    tree = build_tree("Q?", 0, 5, decompose_fn=fn, node_budget=50)
    by_id = {n["id"]: n for n in collect(tree)}
    assert set(by_id) == {"root", "root.0", "root.1", "root.1.0"}
    assert by_id["root"]["stop_reason"] is None        # internal
    assert by_id["root.0"]["stop_reason"] == "atomic"  # A
    assert by_id["root.1"]["stop_reason"] is None      # B, internal
    assert by_id["root.1.0"]["stop_reason"] == "atomic"
    assert by_id["root.1.0"]["depth"] == 2


def test_max_depth_hard_cap():
    fanout = lambda q: [q + " l", q + " r"]  # noqa: E731 -- grows forever
    fn = recording_decomposer(default=fanout)
    tree = build_tree("Q?", 0, 2, decompose_fn=fn, node_budget=100)
    nodes = collect(tree)
    assert len(nodes) == 7  # 1 + 2 + 4
    assert max(n["depth"] for n in nodes) == 2
    leaves = [n for n in nodes if not n["children"]]
    assert all(n["stop_reason"] == "max_depth" for n in leaves)


def test_node_budget_circuit_breaker():
    fanout = lambda q: [q + " l", q + " r"]  # noqa: E731
    fn = recording_decomposer(default=fanout)
    tree = build_tree("Q?", 0, 10, decompose_fn=fn, node_budget=9)
    nodes = collect(tree)
    assert len(nodes) == 9  # never exceeds the budget
    reasons = {n["stop_reason"] for n in nodes if not n["children"]}
    assert "budget" in reasons
    assert None not in reasons  # every leaf carries a real stop reason


def test_budget_of_one_stops_at_the_root():
    fn = recording_decomposer(default=lambda q: [q + " sub"])
    tree = build_tree("Q?", 0, 10, decompose_fn=fn, node_budget=1)
    assert tree["children"] == [] and tree["stop_reason"] == "budget"
    assert fn.calls == []  # never even asked the model


def test_restatement_guard_drops_echoes_and_keeps_index():
    fn = recording_decomposer({"Q?": ["q", "Q ?", "Real sub?"]})
    tree = build_tree("Q?", 0, 3, decompose_fn=fn, node_budget=10)
    assert len(tree["children"]) == 1
    child = tree["children"][0]
    assert child["question"] == "Real sub?"
    assert child["id"] == "root.2"  # child path keeps its slot in the model's list


def test_all_restatements_is_atomic():
    fn = recording_decomposer({"Q?": ["q?", "  Q "]})
    tree = build_tree("Q?", 0, 3, decompose_fn=fn, node_budget=10)
    assert tree["children"] == [] and tree["stop_reason"] == "atomic"


def test_parse_failed_is_recorded_and_treated_atomic():
    fn = recording_decomposer(parse_failed=True)
    tree = build_tree("Q?", 0, 3, decompose_fn=fn, node_budget=10)
    assert tree["stop_reason"] == "atomic" and tree["parse_failed"] is True


@pytest.mark.parametrize("context, expect_parent", [("isolated", None), ("with_parent", "Q?")])
def test_decompose_context_controls_parent_visibility(context, expect_parent):
    fn = recording_decomposer({"Q?": ["A?"]})
    build_tree("Q?", 0, 3, decompose_fn=fn, node_budget=10, decompose_context=context)
    assert fn.calls[0] == ("Q?", None)  # the root never has a parent
    assert fn.calls[1] == ("A?", expect_parent)
