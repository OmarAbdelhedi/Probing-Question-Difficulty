"""FATHOM pipeline orchestration.

This module ties the experiment together. It is deliberately split into a
TWO-PASS, RESUMABLE design:

  Pass 1 -- BUILD TREES (the ONLY place that calls ``model.generate``):
           recursively decompose every question into a ragged tree of
           sub-questions via :func:`src.decompose.build_tree`.
  Pass 2 -- EXTRACT ACTIVATIONS (forward passes only): flatten every tree into
           its full node list, batch ALL node texts through the SAME target
           model once, and store per-node pooled hidden states.

Validity-critical invariant: a SINGLE HF causal LM is loaded ONCE and used for
BOTH decomposition (generation) and activation extraction. The decomposer is
injected as a callable (the "seam"), so an API decomposer can be swapped in
later while activations always stay on the target model.

Outputs (under ``output_dir``):
  - ``activations/<id>.npz`` : one array PER NODE keyed by node path
      (``root``, ``root.0``, ``root.0.1`` ...), each shape ``[n_layers, hidden_dim]``.
  - ``metadata.jsonl``       : one JSON record per QUESTION.
  - ``run_config.json``      : full reproducibility snapshot.
"""

import json
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import extract_activations
from src.decompose import build_tree, make_target_decomposer
from src.load_questions import load_questions

# String -> torch dtype mapping for the config "dtype" key.
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype: str, device: str, *, log_tag: str = "load_model") -> str:
    """Resolve the compute dtype actually usable on ``device``.

    fp16 math on CPU is numerically unreliable, so float16 on CPU falls back to
    float32 (warning tagged ``log_tag``); every other (dtype, device) passes
    through unchanged. Centralized so ``run`` (which resolves early to log and
    record the EFFECTIVE dtype) and ``load_model`` (which guards its own entry
    point) apply the SAME rule and cannot drift.
    """
    if device == "cpu" and dtype == "float16":
        print(f"[{log_tag}] WARNING: float16 on CPU is unreliable; falling back to float32.")
        return "float32"
    return dtype


def load_model(model_name: str, dtype: str, device: str) -> tuple:
    """Load the target causal LM + tokenizer ONCE for both roles.

    Maps the ``dtype`` string to a torch dtype. fp16 on CPU is unreliable, so
    if ``device == "cpu"`` and ``dtype`` is float16 we fall back to float32 and
    warn. Ensures the tokenizer has a pad token (decoder-only models often lack
    one) by aliasing it to the eos token. Returns ``(model, tokenizer)`` with
    the model placed on ``device`` and in eval mode.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_MAP)}."
        )

    # fp16 math on CPU is numerically unreliable -> fall back to fp32.
    dtype = resolve_dtype(dtype, device)

    torch_dtype = _DTYPE_MAP[dtype]

    # Large models (e.g. 32B) must NOT be fully materialized in CPU RAM and then
    # copied to the GPU -- that needs ~2 bytes/param of host RAM and easily OOMs a
    # modest --mem allocation. On GPU we let accelerate stream the weights straight
    # onto the device(s) via device_map="auto" (which also shards across multiple
    # GPUs when one is not enough); low_cpu_mem_usage keeps the host-RAM peak to
    # roughly one shard. Both options require the `accelerate` package.
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        # device_map has already placed every weight; calling model.to(device)
        # here would try to move a (possibly sharded) dispatched model and fail,
        # so we deliberately skip it.

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Decoder-only tokenizers frequently have no pad token; alias to eos so that
    # batched padding (Pass 2) and generation (Pass 1) work.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def flatten_tree(node) -> list:
    """Flatten a decomposition tree into FLAT RECORDs via PRE-ORDER traversal.

    Each record:
        {"id", "parent_id", "depth", "question", "stop_reason", "parse_failed"}
    ``parent_id`` is the node id with the trailing ``.<n>`` stripped; the root
    ("root") has ``parent_id = None``.
    """
    records = []

    def _parent_id(node_id: str):
        # "root" -> None; "root.0.1" -> "root.0".
        if "." not in node_id:
            return None
        return node_id.rsplit(".", 1)[0]

    def _visit(n):
        records.append(
            {
                "id": n["id"],
                "parent_id": _parent_id(n["id"]),
                "depth": n["depth"],
                "question": n["question"],
                "stop_reason": n["stop_reason"],
                "parse_failed": n["parse_failed"],
            }
        )
        for child in n["children"]:
            _visit(child)

    _visit(node)
    return records


def _seed_everything(seed: int) -> None:
    """Seed python / numpy / torch RNGs for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_io(input_dir: str, output_dir: str) -> None:
    """Fast path validation BEFORE the (expensive) model load.

    Verifies the input directory exists and the output directory is creatable
    and writable (by creating it and round-tripping a temp file). Raises with a
    clear message on failure so callers fail fast.
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(
            f"--input must be an existing directory: {input_dir!r}"
        )

    # Create the output dir (and parents) up front and prove we can write to it.
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create --output directory {output_dir!r}: {exc}")

    probe = os.path.join(output_dir, ".fathom_write_test")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise OSError(f"--output directory {output_dir!r} is not writable: {exc}")


def run(input_dir, output_dir, config, *, limit=None, max_depth=None, device=None,
        model_name=None) -> None:
    """Run the full two-pass FATHOM pipeline.

    See the module docstring for the high-level design. ``config`` holds
    experiment params only (no paths). ``limit`` / ``max_depth`` / ``device`` /
    ``model_name`` are optional CLI overrides (each falls back to ``config`` when
    ``None``); ``model_name`` lets a script point several runs at different HF
    models without editing ``config.yaml``.
    """
    # ------------------------------------------------------------------ #
    # 1) VALIDATE FAST -- before any model load.                          #
    # ------------------------------------------------------------------ #
    _validate_io(input_dir, output_dir)

    # ------------------------------------------------------------------ #
    # 2) Resolve effective settings + seed.                              #
    # ------------------------------------------------------------------ #
    eff_max_depth = max_depth if max_depth is not None else config["max_depth"]
    eff_device = device if device is not None else config["device"]

    # Requested CUDA but none available -> fall back to CPU with a warning.
    if eff_device == "cuda" and not torch.cuda.is_available():
        print("[run] WARNING: cuda requested but unavailable; falling back to cpu.")
        eff_device = "cpu"

    seed = config["seed"]
    _seed_everything(seed)

    dtype = config["dtype"]
    # Resolve the fp16->fp32-on-CPU fallback HERE too (load_model applies the
    # same guard) so the log line below and run_config.json record the dtype
    # actually used, not the requested one.
    dtype = resolve_dtype(dtype, eff_device, log_tag="run")
    pooling = config["pooling"]
    layers_cfg = config["layers"]  # provenance value (e.g. "all"); see step 8/9.
    node_budget = config["node_budget"]
    decompose_context = config["decompose_context"]
    batch_size = config["batch_size"]
    # CLI --model overrides config.yaml's model_name; falls back to config when
    # unset. Recorded into run_config.json below, so downstream analysis/transfer
    # see the model ACTUALLY used.
    model_name = model_name if model_name is not None else config["model_name"]

    # ------------------------------------------------------------------ #
    # 3) Output subdir for activations.                                  #
    # ------------------------------------------------------------------ #
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 4) Load questions (+ optional limit).                             #
    # ------------------------------------------------------------------ #
    questions = load_questions(input_dir)
    if limit is not None:
        questions = questions[:limit]

    # ------------------------------------------------------------------ #
    # 5) RESUMABILITY: skip questions whose npz already exists.          #
    # ------------------------------------------------------------------ #
    pending = []
    for q in questions:
        npz_path = os.path.join(activations_dir, f"{q['id']}.npz")
        if os.path.exists(npz_path):
            print(f"[run] skip (exists): {q['id']}")
            continue
        pending.append(q)

    if not pending:
        print("[run] nothing to do -- all requested questions already processed.")
        return

    # ------------------------------------------------------------------ #
    # 6) Load the model ONCE and build the decomposer seam.             #
    # ------------------------------------------------------------------ #
    print(f"[run] loading model {model_name!r} (dtype={dtype}, device={eff_device}) ...")
    model, tokenizer = load_model(model_name, dtype, eff_device)
    decompose_fn = make_target_decomposer(model, tokenizer, seed=seed)

    # ------------------------------------------------------------------ #
    # 7) PASS 1 -- BUILD TREES (generation only).                       #
    # ------------------------------------------------------------------ #
    built = []  # list of (question_dict, tree_root)
    print(f"[run] PASS 1: building {len(pending)} decomposition tree(s) ...")
    for i, q in enumerate(pending, start=1):
        tree = build_tree(
            q["question"],
            0,
            eff_max_depth,
            "root",
            decompose_fn=decompose_fn,
            node_budget=node_budget,
            decompose_context=decompose_context,
        )
        built.append((q, tree))
        n_nodes = len(flatten_tree(tree))
        print(f"[run]   tree {i}/{len(pending)} id={q['id']} nodes={n_nodes}")

    # ------------------------------------------------------------------ #
    # 8) PASS 2 -- EXTRACT ACTIVATIONS (forward passes only).           #
    # ------------------------------------------------------------------ #
    # Build ONE global text list across ALL trees so a single batched extraction
    # covers everything, with an index map back to (qid, node_id).
    all_texts = []
    index_map = []  # parallel to all_texts: (qid, node_id)
    flats_by_qid = {}  # qid -> list[FLAT RECORD]
    for q, tree in built:
        flat = flatten_tree(tree)
        flats_by_qid[q["id"]] = flat
        for rec in flat:
            all_texts.append(rec["question"])
            index_map.append((q["id"], rec["id"]))

    print(
        f"[run] PASS 2: extracting activations for {len(all_texts)} node(s) "
        f"across {len(built)} question(s) ..."
    )
    acts = extract_activations(
        all_texts,
        model,
        tokenizer,
        pooling=pooling,
        layers=layers_cfg,
        batch_size=batch_size,
        device=eff_device,
    )

    # Regroup the flat activation rows back by question -> {node_id: vector}.
    per_q_arrays = {q["id"]: {} for q, _ in built}
    for row, (qid, node_id) in enumerate(index_map):
        per_q_arrays[qid][node_id] = acts[row]  # [n_layers, hidden_dim]

    # Persist per-question: npz FIRST (resumability), THEN the metadata line.
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    stop_reason_counts = Counter()
    total_nodes = 0
    questions_processed = 0

    with open(metadata_path, "a", encoding="utf-8") as meta_fh:
        for q, _tree in built:
            qid = q["id"]
            flat = flats_by_qid[qid]
            node_arrays = per_q_arrays[qid]

            # node-keyed npz: one array per node path.
            npz_path = os.path.join(activations_dir, f"{qid}.npz")
            np.savez(npz_path, **node_arrays)

            # one metadata record per QUESTION.
            record = {
                "id": qid,
                "label": q["label"],
                "original_question": q["question"],
                "pooling": pooling,
                "layers": layers_cfg,  # provenance value (e.g. "all").
                "max_depth": eff_max_depth,
                "decompose_context": decompose_context,
                "nodes": flat,
            }
            meta_fh.write(json.dumps(record) + "\n")
            meta_fh.flush()

            # Tally counts for the final summary.
            for rec in flat:
                total_nodes += 1
                stop_reason_counts[rec["stop_reason"]] += 1
            questions_processed += 1
            print(f"[run]   wrote {qid}: {len(flat)} node(s) -> {npz_path}")

    # ------------------------------------------------------------------ #
    # 9) run_config.json -- full reproducibility snapshot.              #
    # ------------------------------------------------------------------ #
    run_config = {
        "model_name": model_name,
        "dtype": dtype,
        "layers": layers_cfg,  # config value for provenance (e.g. "all").
        "pooling": pooling,
        "max_depth": eff_max_depth,
        "node_budget": node_budget,
        "decompose_context": decompose_context,
        "seed": seed,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "transformers_version": transformers.__version__,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    # ------------------------------------------------------------------ #
    # 10) Final summary.                                                #
    # ------------------------------------------------------------------ #
    print("[run] DONE.")
    print(f"[run]   questions processed: {questions_processed}")
    print(f"[run]   total nodes:         {total_nodes}")
    print("[run]   nodes per stop_reason:")
    # Internal (non-leaf) nodes have stop_reason=None; report them too.
    for reason in ("atomic", "max_depth", "budget", None):
        if reason in stop_reason_counts:
            label = reason if reason is not None else "internal (None)"
            print(f"[run]     {label}: {stop_reason_counts[reason]}")
