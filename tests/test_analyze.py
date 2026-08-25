"""Unit tests for src/analyze.py's Tier-1 summarization helpers
(summarize_question, aggregate_by_label). The metadata loading and label
ordering they build on live in src/run_data.py and are covered by
test_run_data.py."""

from collections import Counter

from src.analyze import aggregate_by_label, summarize_question


def sample_record():
    return {
        "id": "q1",
        "label": "hard",
        "nodes": [
            {"id": "root", "depth": 0, "stop_reason": None, "parse_failed": False},
            {"id": "root.0", "depth": 1, "stop_reason": "atomic", "parse_failed": False},
            {"id": "root.1", "depth": 1, "stop_reason": None, "parse_failed": True},
            {"id": "root.1.0", "depth": 2, "stop_reason": "max_depth", "parse_failed": False},
        ],
    }


def test_summarize_question_metrics():
    s = summarize_question(sample_record())
    assert s["n_nodes"] == 4
    assert s["tree_depth"] == 2
    # Internal nodes (stop_reason=None) are excluded; only leaves counted.
    assert s["leaf_stop_reasons"] == Counter({"atomic": 1, "max_depth": 1})
    assert s["parse_failures"] == 1


def test_aggregate_by_label():
    summaries = [summarize_question(sample_record()),
                 summarize_question({"id": "q2", "label": "easy",
                                     "nodes": [{"id": "root", "depth": 0,
                                                "stop_reason": "atomic",
                                                "parse_failed": False}]})]
    agg = aggregate_by_label(summaries, ["easy", "hard"])
    assert agg["easy"]["n_questions"] == 1
    assert agg["easy"]["tree_depth"]["max"] == 0.0
    assert agg["hard"]["n_leaves"] == 2
    assert agg["hard"]["leaf_stop_reason_fractions"] == {"atomic": 0.5, "max_depth": 0.5}
