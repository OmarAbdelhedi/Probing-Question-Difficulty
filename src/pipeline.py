"""Activation extraction: labeled questions in, per-layer hidden states out.

Each question is tokenised as raw text (no chat template) and passed once
through a frozen causal LM; every hidden state is pooled to one vector
(``src/activations.py``). Nothing is generated.

Outputs, under ``output_dir``:

  activations/<id>.npz  one float32 array under the key ``"root"``, shape
                        ``[n_layers, hidden_dim]``, layer 0 = embeddings
  metadata.jsonl        one record per question: id, label,
                        original_question, pooling, layers
  run_config.json       model, dtype, pooling, layers, seed, library versions

The ``"root"`` key and the metadata field names are those of the runs behind
the paper, so those activations load unchanged.
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import extract_activations
from src.load_questions import load_questions

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype: str, device: str, *, log_tag: str = "load_model") -> str:
    """Return the dtype to use on ``device``: float16 on CPU falls back to float32."""
    if device == "cpu" and dtype == "float16":
        print(f"[{log_tag}] WARNING: float16 on CPU is unreliable; falling back to float32.")
        return "float32"
    return dtype


def load_model(model_name: str, dtype: str, device: str) -> tuple:
    """Load the causal LM and its tokenizer in eval mode.

    On GPU the weights are streamed onto the device(s) with
    ``device_map="auto"`` (sharding across GPUs if needed) so a 24-32B model
    never has to fit in host RAM. Decoder-only tokenizers often lack a pad
    token; it is aliased to EOS so batches can be padded.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_MAP)}.")
    torch_dtype = _DTYPE_MAP[resolve_dtype(dtype, device)]

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, low_cpu_mem_usage=True, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_io(input_dir: str, output_dir: str) -> None:
    """Fail fast, before the model load, on a missing input or unwritable output dir."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"--input must be an existing directory: {input_dir!r}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create --output directory {output_dir!r}: {exc}")
    probe = os.path.join(output_dir, ".write_test")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise OSError(f"--output directory {output_dir!r} is not writable: {exc}")


def run(input_dir, output_dir, config, *, limit=None, device=None, model_name=None) -> None:
    """Extract and store activations for every question in ``input_dir``.

    ``config`` holds the extraction settings (``config.yaml``); ``device`` and
    ``model_name`` override it when given. Questions whose ``.npz`` already
    exists are skipped, so an interrupted run resumes with the same command.
    """
    _validate_io(input_dir, output_dir)

    device = device if device is not None else config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[run] WARNING: cuda requested but unavailable; falling back to cpu.")
        device = "cpu"
    seed = config["seed"]
    _seed_everything(seed)
    dtype = resolve_dtype(config["dtype"], device, log_tag="run")
    pooling = config["pooling"]
    layers = config["layers"]
    batch_size = config["batch_size"]
    model_name = model_name if model_name is not None else config["model_name"]

    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    questions = load_questions(input_dir)
    if limit is not None:
        questions = questions[:limit]
    pending = [q for q in questions
               if not os.path.exists(os.path.join(activations_dir, f"{q['id']}.npz"))]
    if len(pending) < len(questions):
        print(f"[run] {len(questions) - len(pending)} question(s) already extracted, skipping them.")
    if not pending:
        print("[run] nothing to do.")
        return

    print(f"[run] loading {model_name!r} (dtype={dtype}, device={device}) ...")
    model, tokenizer = load_model(model_name, dtype, device)

    print(f"[run] extracting activations for {len(pending)} question(s) ...")
    acts = extract_activations(
        [q["question"] for q in pending],
        model,
        tokenizer,
        pooling=pooling,
        layers=layers,
        batch_size=batch_size,
        device=device,
    )

    # Write each .npz before its metadata line, so a crash never leaves a
    # metadata record without activations.
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "a", encoding="utf-8") as meta_fh:
        for q, arr in zip(pending, acts):
            np.savez(os.path.join(activations_dir, f"{q['id']}.npz"), root=arr)
            meta_fh.write(json.dumps({
                "id": q["id"],
                "label": q["label"],
                "original_question": q["question"],
                "pooling": pooling,
                "layers": layers,
            }) + "\n")
            meta_fh.flush()

    run_config = {
        "model_name": model_name,
        "dtype": dtype,
        "layers": layers,
        "pooling": pooling,
        "seed": seed,
        "batch_size": batch_size,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    print(f"[run] done: {len(pending)} question(s), shape per question {tuple(acts.shape[1:])}"
          f" -> {output_dir}")
