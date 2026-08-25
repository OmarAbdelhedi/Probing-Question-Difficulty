"""Recursive question decomposition for FATHOM.

This module owns the *generation* side of the pipeline. It exposes:

  * ``DECOMPOSE_PROMPT`` -- the verbatim system prompt applied IDENTICALLY at
    every node (root and sub-questions alike).
  * ``parse_subquestions`` -- robust extraction of the first JSON array of
    sub-questions from raw model output.
  * ``make_target_decomposer`` -- THE SEAM. A factory that wraps the target HF
    model into a callable decomposer. Swap this for an API-backed factory later
    (see its docstring) while activations always stay on the target model.
  * ``build_tree`` -- the recursive tree builder with TWO independent stop
    conditions (hard depth cap + adaptive atomic signal), a parent-restatement
    guard, and a per-question node-budget circuit breaker.

Decomposition is the ONLY step in the pipeline that calls ``model.generate``.
"""

import json

import torch


# Verbatim prompt from the SPEC. Applied identically at every node so that the
# decomposition rule is the same for the root question and every sub-question.
DECOMPOSE_PROMPT = """You break a question into simpler sub-questions.
First decide: can this question be answered directly with a single fact, lookup, or reasoning step, WITHOUT breaking it down? If yes, return exactly: []
Otherwise return a JSON array of sub-questions. Each sub-question must be:
  - strictly narrower than the original (a proper part of it, not a rephrasing),
  - independently answerable on its own,
  - necessary -- do not pad with sub-questions that aren't needed.
Return ONLY the JSON array, no prose, no explanation, no markdown fences."""


def parse_subquestions(text: str) -> tuple[list[str], bool]:
    """Extract the FIRST balanced JSON array from ``text``.

    Scans for a top-level ``[ ... ]`` while respecting string literals and
    backslash escapes (so brackets inside quoted strings are ignored). The
    matched substring is then parsed with ``json.loads``.

    Returns ``(subquestions, parse_failed)``:
      * On a successfully-parsed list -> ``([str(x).strip() for x in arr
        if non-empty], False)``. A parsed EMPTY array returns ``([], False)``,
        which is the *legitimate* atomic signal.
      * On ANY failure / no array found -> ``([], True)``. ``parse_failed=True``
        is still treated as atomic by ``build_tree``, but the flag is recorded.
    """
    if not isinstance(text, str):
        return [], True

    start = None          # index of the opening '[' of the first array
    depth = 0             # bracket nesting depth
    in_string = False     # are we inside a JSON string literal?
    escaped = False       # was the previous char a backslash inside a string?

    for i, ch in enumerate(text):
        if in_string:
            # Inside a string literal: only track escapes and the closing quote.
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        # Not inside a string literal.
        if ch == '"':
            in_string = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    # Found a balanced top-level array: text[start:i+1].
                    candidate = text[start:i + 1]
                    try:
                        arr = json.loads(candidate)
                    except (ValueError, TypeError):
                        return [], True
                    if not isinstance(arr, list):
                        return [], True
                    subqs = [str(x).strip() for x in arr if str(x).strip()]
                    return subqs, False

    # No balanced top-level array found.
    return [], True


def make_target_decomposer(model, tokenizer, *, max_new_tokens: int = 512, seed: int = 0):
    """THE SEAM: build a decomposer callable backed by the target HF model.

    Returns ``decompose_fn(question, parent_question=None) -> (subquestions,
    parse_failed)``. The same single model that extracts activations also does
    the decomposition, keeping the experiment valid (one model, both roles).

    Generation is greedy/deterministic (``do_sample=False``) so trees are
    reproducible. Only the newly generated tokens are decoded (the prompt is
    sliced off) before parsing.

    To use an API decomposer later, write ANOTHER factory that returns a
    callable with the SAME ``(question, parent_question) -> (list[str], bool)``
    signature and pass it to ``build_tree`` as ``decompose_fn``. Activations
    always stay on the target model regardless of which decomposer is used.
    """
    # Greedy decoding is deterministic, but we seed anyway so any future change
    # to sampling stays reproducible and the seam's contract is honoured.
    torch.manual_seed(seed)

    def decompose_fn(question: str, parent_question: str | None = None) -> tuple[list[str], bool]:
        # Build the user content: optionally prefix the parent for context
        # (only when caller passed a parent_question, i.e. "with_parent" mode).
        if parent_question is not None:
            user_content = (
                "Parent question (for context only): "
                f"{parent_question}\n\nQuestion to decompose: {question}"
            )
        else:
            user_content = question

        messages = [
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode ONLY the newly generated tokens (slice off the prompt length).
        new_tokens = out[0][input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_subquestions(decoded)

    return decompose_fn


def normalize_question(s: str) -> str:
    """Normalize a question for the parent-restatement guard.

    Lowercase, collapse all whitespace runs to single spaces, then strip
    surrounding spaces and trailing/leading sentence punctuation (``? . !``).
    Two questions that differ only in case / spacing / terminal punctuation are
    treated as identical. Public so the sanity-check suite verifies the guard
    with THIS exact rule instead of a drifting copy.
    """
    return " ".join(s.lower().split()).strip(" ?.!")


def build_tree(
    question: str,
    depth: int,
    max_depth: int,
    path: str = "root",
    *,
    decompose_fn,
    node_budget: int,
    decompose_context: str = "isolated",
    parent_question: str | None = None,
    state: dict | None = None,
) -> dict:
    """Recursively decompose ``question`` into a ragged tree of node dicts.

    Two INDEPENDENT stop conditions (never collapsed into one):

      1. HARD CAP: ``depth >= max_depth`` -> ``stop_reason="max_depth"``.
      2. ADAPTIVE: ``decompose_fn`` returns ``[]`` -> the model judged the
         question atomic -> ``stop_reason="atomic"``.

    Plus a per-question NODE BUDGET circuit breaker
    (``stop_reason="budget"``) and a guard that drops any child whose
    normalized text equals its parent (a restatement, not a proper part).

    ``state`` is a SHARED mutable counter ``{"count": int}`` for the WHOLE tree
    so the node budget spans the entire recursion. The root call leaves it
    ``None`` and it is initialized here.

    Returns a node dict with keys: ``id``, ``question``, ``depth``,
    ``children``, ``stop_reason``, ``parse_failed``. Every leaf ends with a
    non-null ``stop_reason``; internal nodes keep ``stop_reason=None``.
    """
    if state is None:
        state = {"count": 0}

    # 1. Fresh node; count it against the shared budget.
    node = {
        "id": path,
        "question": question,
        "depth": depth,
        "children": [],
        "stop_reason": None,
        "parse_failed": False,
    }
    state["count"] += 1

    # 2. HARD CAP: we are as deep as allowed -> no children.
    if depth >= max_depth:
        node["stop_reason"] = "max_depth"
        return node

    # 3. CIRCUIT BREAKER: already at budget -> cannot afford any children.
    if state["count"] >= node_budget:
        node["stop_reason"] = "budget"
        return node

    # 4. Ask the model to decompose. In "isolated" mode we do NOT show the
    #    parent (prevents drift back to the original question); in
    #    "with_parent" mode we pass the parent through for context.
    subqs, parse_failed = decompose_fn(
        question, parent_question if decompose_context == "with_parent" else None
    )
    node["parse_failed"] = parse_failed

    # 5. ADAPTIVE: empty (or parse-failed) -> model judged it atomic.
    if not subqs:
        node["stop_reason"] = "atomic"
        return node

    # 6. Expand children, honouring the guard and the budget mid-expansion.
    for i, sq in enumerate(subqs):
        # GUARD: drop a sub-question that merely restates its parent.
        if normalize_question(sq) == normalize_question(question):
            continue
        # Budget hit while expanding children -> stop adding more.
        if state["count"] >= node_budget:
            break
        child = build_tree(
            sq,
            depth + 1,
            max_depth,
            path + "." + str(i),
            decompose_fn=decompose_fn,
            node_budget=node_budget,
            decompose_context=decompose_context,
            parent_question=question,
            state=state,
        )
        node["children"].append(child)

    # 7. If decompose returned sub-questions but every one was a restatement,
    #    the node is effectively atomic.
    if not node["children"]:
        node["stop_reason"] = "atomic"
    return node
