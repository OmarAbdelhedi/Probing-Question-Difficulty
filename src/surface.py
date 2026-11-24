"""Surface statistics of a question, used to control for surface form.

Each feature is a cheap function of the raw question text:

  char_len     number of characters
  word_count   number of whitespace-separated words
  cap_words    capitalised words after the first (a named-entity proxy)
  num_tokens   words containing a digit
  punct        number of ',' and '?'
  token_count  number of model tokens (needs the run's tokenizer)

The within-dataset controls (Tier 2b) and the cross-dataset transfer both
residualise all six, or the first five when the tokenizer cannot be loaded
(the summaries record which).
"""

import numpy as np

SURFACE_FEATURES = ["char_len", "word_count", "cap_words", "num_tokens", "punct"]


def _feature(name: str, question: str, words: list[str]) -> float:
    if name == "char_len":
        return float(len(question))
    if name == "word_count":
        return float(len(words))
    if name == "cap_words":
        return float(sum(1 for i, w in enumerate(words) if i > 0 and w[:1].isupper()))
    if name == "num_tokens":
        return float(sum(1 for w in words if any(ch.isdigit() for ch in w)))
    if name == "punct":
        return float(question.count(",") + question.count("?"))
    raise ValueError(f"unknown surface feature {name!r}")


def surface_matrix(questions, names, tokenizer=None) -> tuple[np.ndarray, list[str]]:
    """Feature matrix ``[n_questions, k]`` for ``names``, plus the column names.

    With a ``tokenizer``, a final ``token_count`` column is appended.
    """
    names = list(names)
    rows = []
    for q in questions:
        q = str(q)
        words = q.split()
        row = [_feature(name, q, words) for name in names]
        if tokenizer is not None:
            row.append(float(len(tokenizer.encode(q))))
        rows.append(row)
    if tokenizer is not None:
        names.append("token_count")
    return np.asarray(rows, dtype=np.float64), names


def try_load_tokenizer(run_config: dict, log_tag: str = "probe2b"):
    """Load the run's tokenizer (weights-free) for ``token_count``; None on failure.

    Any failure (transformers missing, offline, unknown model id) drops the
    ``token_count`` column with a printed note instead of failing the analysis.
    """
    model_name = run_config.get("model_name")
    if not model_name:
        print(f"[{log_tag}]   (token-count covariate skipped: no model_name in run_config.json)")
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        print(f"[{log_tag}] token-count covariate enabled via tokenizer for {model_name!r}")
        return tok
    except Exception as exc:
        print(f"[{log_tag}]   (token-count covariate skipped: {type(exc).__name__}: {exc})")
        return None
