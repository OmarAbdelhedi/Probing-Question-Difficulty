"""Shared pytest setup.

Path setup: the repo root goes onto ``sys.path`` so ``from src...`` resolves
(mirroring what ``main.py`` / ``analyze.py`` rely on).

Fixtures: factories for planted-signal activations and fabricated run
directories, shared by every probe-tier test. The suite needs only
numpy + matplotlib (same as the analysis); tests that touch torch or
transformers ``importorskip`` them individually, so the suite runs anywhere.
"""

import json
import os
import sys

import numpy as np
import pytest

# The fabricated runs name a model ("synth-model") that does not exist; keep
# the tokenizer lookup for the token-count feature offline so it fails fast.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture
def planted_xy():
    """Factory for synthetic activations with a planted class-strength axis.

    ``_make(strengths, ...)`` returns ``(X, y)`` where ``X`` is float32
    ``[n, n_layers, dim]`` standard-normal noise plus, at every layer in
    ``signal_layers`` (default: all), ``strength * u`` for a fixed unit
    direction ``u`` shared by all classes -- i.e. classes differ only in how
    far along ONE axis they sit, exactly the structure the probes assume.
    """
    def _make(strengths, per_class=20, n_layers=3, dim=32, signal_layers=None,
              noise=1.0, seed=7):
        rng = np.random.default_rng(seed)
        u = rng.normal(size=dim)
        u /= np.linalg.norm(u)
        layers = range(n_layers) if signal_layers is None else signal_layers
        rows, ys = [], []
        for label, strength in strengths:
            for _ in range(per_class):
                v = rng.normal(size=(n_layers, dim)) * noise
                for L in layers:
                    v[L] += strength * u
                rows.append(v.astype(np.float32))
                ys.append(label)
        return np.stack(rows), np.asarray(ys, dtype=object)

    return _make


@pytest.fixture
def make_run(tmp_path):
    """Factory fabricating a finished run directory under ``tmp_path``.

    Writes ``metadata.jsonl`` (one record per row of ``X``, with the keys
    ``main.py`` writes: id / label / original_question / pooling / layers),
    ``activations/<id>.npz`` (key ``"root"``), and ``run_config.json``.
    Returns the run dir as a str. ``skip_npz`` / ``drop_root_key`` name ids
    whose npz is absent / lacks the ``"root"`` key (for missing-file paths);
    ``questions`` overrides the per-row question text (Tier 2b covariates).
    """
    def _make(name, X, y, *, layers="all", questions=None, skip_npz=(),
              drop_root_key=(), run_config={"model_name": "synth-model",
                                            "pooling": "last_token",
                                            "dtype": "float32"}):
        out = tmp_path / name
        (out / "activations").mkdir(parents=True)
        with open(out / "metadata.jsonl", "w", encoding="utf-8") as fh:
            for i in range(X.shape[0]):
                qid = f"{name}_q{i}"
                question = (questions[i] if questions is not None
                            else f"synthetic question number {i}?")
                fh.write(json.dumps({"id": qid, "label": str(y[i]),
                                     "original_question": question,
                                     "pooling": "last_token", "layers": layers}) + "\n")
                if qid in skip_npz:
                    continue
                arrays = ({"not_root": X[i]} if qid in drop_root_key
                          else {"root": X[i]})
                np.savez(out / "activations" / f"{qid}.npz", **arrays)
        if run_config is not None:
            with open(out / "run_config.json", "w", encoding="utf-8") as fh:
                json.dump(run_config, fh)
        return str(out)

    return _make
