"""Tests for src/pipeline.py (needs torch + transformers; skipped otherwise).

The end-to-end test builds a tiny randomly initialised Llama and a word-level
tokenizer locally, so it needs no download.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from src.pipeline import _validate_io, run  # noqa: E402


def test_validate_io_missing_input_dir(tmp_path):
    with pytest.raises(NotADirectoryError):
        _validate_io(str(tmp_path / "nope"), str(tmp_path / "out"))


def test_validate_io_creates_output_dir(tmp_path):
    out = tmp_path / "a" / "b"
    _validate_io(str(tmp_path), str(out))
    assert out.is_dir()


@pytest.fixture
def tiny_model_dir(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = "what is the capital of france who wrote it when was born ?".split()
    vocab = {"[UNK]": 0, "[EOS]": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                                eos_token="[EOS]")
    cfg = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=2, num_attention_heads=2,
                                   num_key_value_heads=2, max_position_embeddings=64,
                                   eos_token_id=1)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg)
    path = tmp_path / "tiny"
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return str(path)


def test_run_end_to_end_and_resume(tiny_model_dir, tmp_path, capsys):
    qdir = tmp_path / "questions"
    qdir.mkdir()
    questions = [("q0", "what is the capital of france ?", "easy"),
                 ("q1", "who wrote it ?", "easy"),
                 ("q2", "when was the capital of france born ?", "hard")]
    with open(qdir / "set.jsonl", "w", encoding="utf-8") as fh:
        for qid, q, lab in questions:
            fh.write(json.dumps({"id": qid, "question": q, "label": lab}) + "\n")

    out = tmp_path / "run"
    config = {"model_name": tiny_model_dir, "dtype": "float32", "device": "cpu",
              "pooling": "last_token", "layers": "all", "seed": 0, "batch_size": 2}
    run(str(qdir), str(out), config, limit=2)

    for qid in ("q0", "q1"):
        with np.load(out / "activations" / f"{qid}.npz") as npz:
            assert list(npz.keys()) == ["root"]
            assert npz["root"].shape == (3, 16)  # embeddings + 2 blocks, hidden 16
            assert npz["root"].dtype == np.float32
    assert not (out / "activations" / "q2.npz").exists()
    records = [json.loads(line) for line in open(out / "metadata.jsonl", encoding="utf-8")]
    assert [r["id"] for r in records] == ["q0", "q1"]
    assert records[0]["original_question"] == questions[0][1]
    run_config = json.loads((out / "run_config.json").read_text())
    assert run_config["model_name"] == tiny_model_dir and run_config["pooling"] == "last_token"

    # Resume: only the missing question is extracted; existing files untouched.
    before = np.load(out / "activations" / "q0.npz")["root"]
    run(str(qdir), str(out), config)
    assert "2 question(s) already extracted" in capsys.readouterr().out
    records = [json.loads(line) for line in open(out / "metadata.jsonl", encoding="utf-8")]
    assert [r["id"] for r in records] == ["q0", "q1", "q2"]
    np.testing.assert_array_equal(np.load(out / "activations" / "q0.npz")["root"], before)

    # The analysis layer reads what the pipeline wrote.
    from src.run_data import load_run
    data = load_run(str(out))
    assert data.X.shape == (3, 3, 16) and data.labels == ["easy", "hard"]
