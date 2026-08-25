"""End-to-end sanity checks for the FATHOM activation-extraction pipeline.

This module is RUNNABLE: it drives ``src.pipeline.run`` to produce artifacts
(decomposition trees + stored activations) on a small slice of questions, then
re-loads those artifacts and verifies the structural + numerical invariants the
rest of the pipeline depends on.

What it checks (per the SPEC/CONTRACT):
  - prints every question tree indented by depth (node question + stop_reason)
    and the shape of every stored activation array;
  - no node has depth > max_depth;
  - every LEAF has stop_reason in {atomic, max_depth, budget};
  - no child question text equals its parent (the restatement guard worked);
  - the flat node list reconstructs into a VALID tree
    (every parent_id resolves; exactly one root per question);
  - cosine(vector, itself) ~= 1.0 and two DIFFERENT nodes give cosine < 1.0;
  - pooled hidden width == model hidden_size and n_layers == expected;
  - no NaN / Inf anywhere in the stored arrays.

Run directly as a script:  ``python src/sanity_checks.py``
(uses ./config.yaml, input data/questions, output outputs/sanity).
"""

import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Small numeric helper
# ---------------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (numpy only)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Tree reconstruction from the FLAT metadata node list
# ---------------------------------------------------------------------------
def _reconstruct_tree(nodes):
    """Rebuild a parent->children tree from FLAT records.

    Each flat record (CONTRACT):
      {"id", "parent_id", "depth", "question", "stop_reason", "parse_failed"}

    Returns (root_record, children_map) where children_map[id] is the list of
    child records (sorted by id for stable, depth-ordered printing). Asserts the
    flat list is a VALID single-rooted tree: every parent_id resolves and there
    is exactly one root.
    """
    by_id = {}
    for rec in nodes:
        nid = rec["id"]
        assert nid not in by_id, f"duplicate node id in flat list: {nid}"
        by_id[nid] = rec

    roots = []
    children_map = {nid: [] for nid in by_id}
    for rec in nodes:
        pid = rec["parent_id"]
        if pid is None:
            roots.append(rec)
        else:
            assert pid in by_id, f"node {rec['id']} has unresolved parent_id {pid}"
            children_map[pid].append(rec)

    assert len(roots) == 1, f"expected exactly one root, found {len(roots)}: {[r['id'] for r in roots]}"
    # Deterministic child order (root.0 before root.1 ...). ids share a numeric
    # path scheme, so a plain sort over the dotted-int path is sufficient.
    for nid in children_map:
        children_map[nid].sort(key=lambda r: [int(p) for p in r["id"].split(".")[1:]])
    return roots[0], children_map


def _print_tree(root, children_map, shapes):
    """Print the reconstructed tree indented by depth, with array shapes."""
    stack = [root]
    while stack:
        rec = stack.pop()
        indent = "  " * rec["depth"]
        shape = shapes.get(rec["id"])
        shape_str = f"  shape={tuple(shape)}" if shape is not None else "  shape=<missing>"
        print(f"{indent}- [{rec['id']}] ({rec['stop_reason']}) {rec['question']}{shape_str}")
        # push children in reverse so they pop in ascending id order
        for child in reversed(children_map[rec["id"]]):
            stack.append(child)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_sanity_checks(input_dir, output_dir, config, limit=2):
    """Produce artifacts via the pipeline, then load + verify them.

    Parameters
    ----------
    input_dir : str   directory of labeled *.jsonl questions
    output_dir : str  where artifacts are written (created by the pipeline)
    config : dict     experiment params (model_name, pooling, layers, ...)
    limit : int       only run the first ``limit`` questions (keep it tiny)
    """
    # Lazy imports: the __main__ block puts the project ROOT on sys.path FIRST,
    # then calls us, so `from src...` resolves under `python src/sanity_checks.py`
    # (at module-import time sys.path[0] is src/, not the project root).
    import shutil
    from transformers import AutoConfig

    from src.activations import resolve_layers
    from src.decompose import normalize_question
    from src.pipeline import run

    # --- 1) Build the artifacts (the only step that touches the model) -------
    # Start from a CLEAN output dir so we verify ONLY this invocation's artifacts
    # (run() appends to metadata.jsonl and skips questions whose npz exists; a
    # stale dir could otherwise leave duplicate/old records behind).
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    print(f"[sanity] running pipeline on first {limit} question(s) ...")
    run(input_dir, output_dir, config, limit=limit)

    # Ground truth for the numerical-shape checks: the model's REAL hidden_size
    # and the REAL expected number of selected layers. Cheap -- loads the model
    # config only (no weights), and they were just cached by run().
    model_cfg = AutoConfig.from_pretrained(config["model_name"])
    gt_hidden = model_cfg.hidden_size
    gt_n_layers = len(resolve_layers(config["layers"], model_cfg.num_hidden_layers + 1))

    # --- 2) Load metadata.jsonl ---------------------------------------------
    meta_path = os.path.join(output_dir, "metadata.jsonl")
    assert os.path.isfile(meta_path), f"metadata.jsonl not found at {meta_path}"

    records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    assert records, "metadata.jsonl is empty -- the pipeline produced no questions"

    max_depth = config["max_depth"]
    pooling = config["pooling"]
    layers_cfg = config["layers"]
    valid_leaf_reasons = {"atomic", "max_depth", "budget"}

    # Track global invariants across all questions.
    total_nodes = 0
    expected_hidden = gt_hidden       # model hidden_size (ground truth)
    expected_n_layers = gt_n_layers   # n_selected_layers (ground truth)

    # Collect a few distinct vectors (across the whole run) for the
    # "two different nodes -> cosine < 1" check.
    distinct_vectors = []  # list of (key, flat_vector)

    for rec in records:
        qid = rec["id"]
        nodes = rec["nodes"]
        assert nodes, f"question {qid} has no nodes"

        # --- load the per-question npz (node-keyed arrays) ------------------
        npz_path = os.path.join(output_dir, "activations", f"{qid}.npz")
        assert os.path.isfile(npz_path), f"activations npz missing for {qid}: {npz_path}"
        # Materialize the arrays then close the handle immediately (an open
        # NpzFile keeps the file locked, which can block reuse on Windows).
        with np.load(npz_path) as _npz:
            arrays = {k: _npz[k] for k in _npz.files}
        shapes = {k: v.shape for k, v in arrays.items()}

        # --- reconstruct + validate the tree shape -------------------------
        root, children_map = _reconstruct_tree(nodes)

        # the root flat record's id is the (full) question id-space root; the
        # node ids inside a tree use the path scheme "root", "root.0", ...
        assert root["id"] == "root", f"question {qid}: root node id is {root['id']!r}, expected 'root'"
        assert root["parent_id"] is None, f"question {qid}: root parent_id must be None"
        assert root["depth"] == 0, f"question {qid}: root depth must be 0, got {root['depth']}"

        # --- print the tree (indented by depth) + array shapes -------------
        print(f"\n=== question {qid}  (label={rec.get('label')!r}) ===")
        print(f"original: {rec.get('original_question')}")
        _print_tree(root, children_map, shapes)

        # --- per-node assertions -------------------------------------------
        node_by_id = {n["id"]: n for n in nodes}
        for n in nodes:
            nid = n["id"]
            total_nodes += 1

            # depth never exceeds the hard cap
            assert n["depth"] <= max_depth, (
                f"question {qid} node {nid}: depth {n['depth']} > max_depth {max_depth}"
            )

            # every stored array must exist and be 2-D [n_layers, hidden]
            assert nid in shapes, f"question {qid}: no stored array for node {nid}"
            arr = arrays[nid]
            assert arr.ndim == 2, (
                f"question {qid} node {nid}: array ndim {arr.ndim} != 2 "
                f"(expected [n_layers, hidden_dim])"
            )
            assert np.isfinite(arr).all(), (
                f"question {qid} node {nid}: array contains NaN/Inf"
            )

            # pooled width == model hidden_size and n_layers == expected,
            # validated against the REAL model config (not just self-consistency).
            n_layers, hidden = arr.shape
            assert hidden == expected_hidden, (
                f"question {qid} node {nid}: hidden_dim {hidden} != model "
                f"hidden_size {expected_hidden}"
            )
            assert n_layers == expected_n_layers, (
                f"question {qid} node {nid}: n_layers {n_layers} != expected "
                f"{expected_n_layers}"
            )

            # LEAF (no children) must have a real stop_reason
            is_leaf = len(children_map[nid]) == 0
            if is_leaf:
                assert n["stop_reason"] in valid_leaf_reasons, (
                    f"question {qid} leaf {nid}: stop_reason "
                    f"{n['stop_reason']!r} not in {valid_leaf_reasons}"
                )

            # guard check: no child question equals its parent (normalized)
            pid = n["parent_id"]
            if pid is not None:
                parent_q = node_by_id[pid]["question"]
                assert normalize_question(n["question"]) != normalize_question(parent_q), (
                    f"question {qid} node {nid}: child text equals parent text "
                    f"(restatement guard failed)"
                )

            # collect a distinct vector (use mean over layers as a stable rep)
            vec = arr.reshape(-1).astype(np.float64)
            distinct_vectors.append((f"{qid}:{nid}", vec))

        # --- self-cosine ~= 1 for at least one node in this question -------
        first_key = nodes[0]["id"]
        v = arrays[first_key].reshape(-1)
        self_cos = _cosine(v, v)
        assert abs(self_cos - 1.0) < 1e-4, (
            f"question {qid} node {first_key}: cosine(v, v) = {self_cos}, expected ~1.0"
        )

        # --- metadata provenance sanity ------------------------------------
        assert rec.get("max_depth") == max_depth, (
            f"question {qid}: metadata max_depth {rec.get('max_depth')} != config {max_depth}"
        )
        assert rec.get("pooling") == pooling, (
            f"question {qid}: metadata pooling {rec.get('pooling')!r} != config {pooling!r}"
        )
        assert rec.get("layers") == layers_cfg, (
            f"question {qid}: metadata layers {rec.get('layers')!r} != config {layers_cfg!r}"
        )

    # --- 3) cross-node cosine < 1 for two DIFFERENT nodes -------------------
    # Find a genuinely different pair (vectors not numerically identical).
    found_pair = False
    for i in range(len(distinct_vectors)):
        ki, vi = distinct_vectors[i]
        for j in range(i + 1, len(distinct_vectors)):
            kj, vj = distinct_vectors[j]
            if vi.shape != vj.shape:
                continue
            if np.array_equal(vi, vj):
                continue  # identical text -> identical activations, skip
            c = _cosine(vi, vj)
            assert c < 1.0 - 1e-4, (
                f"two different nodes {ki} and {kj} have cosine {c} ~= 1.0 "
                f"(expected < 1.0)"
            )
            found_pair = True
            print(f"\n[sanity] distinct-node cosine check: {ki} vs {kj} -> {c:.4f}")
            break
        if found_pair:
            break
    if not found_pair:
        print("\n[sanity] WARNING: could not find two distinct node vectors to compare "
              "(too few/identical nodes); skipping cross-node cosine check.")

    # --- 4) summary --------------------------------------------------------
    print("\n[sanity] ALL CHECKS PASSED")
    print(f"[sanity] questions verified : {len(records)}")
    print(f"[sanity] total nodes        : {total_nodes}")
    print(f"[sanity] hidden_dim         : {expected_hidden}")
    print(f"[sanity] n_layers stored    : {expected_n_layers}")


# ---------------------------------------------------------------------------
# Runnable as a script: `python src/sanity_checks.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Put the project ROOT on sys.path so `from src...` resolves when this file
    # is executed directly (src/ is one level below the project root).
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    import yaml

    with open(os.path.join(_ROOT, "config.yaml"), "r", encoding="utf-8") as _f:
        _config = yaml.safe_load(_f)

    _input_dir = os.path.join(_ROOT, "data", "questions")
    _output_dir = os.path.join(_ROOT, "outputs", "sanity")
    run_sanity_checks(_input_dir, _output_dir, _config, limit=2)
