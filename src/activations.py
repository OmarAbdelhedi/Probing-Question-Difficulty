"""Activation extraction for FATHOM.

Forward-pass-only helpers that turn node question texts into per-layer pooled
hidden-state vectors using a single HuggingFace causal LM. NO generation here —
decomposition (the only generation step) lives in ``src/decompose.py``.

Design rules (validity-critical):
  * Same model / layers / pooling used everywhere (set by the caller).
  * Pooling NEVER touches padding — every sequence is pooled by its OWN
    attention mask, so right- or left-padding both work.
  * eval() + torch.no_grad(), float32 math and storage, deterministic output.
"""

import numpy as np
import torch


def resolve_layers(layers_cfg, n_hidden_states: int) -> list[int]:
    """Resolve a config layer spec into a concrete, validated list of indices.

    ``n_hidden_states`` is ``len(out.hidden_states)`` for a forward pass, which
    equals ``num_transformer_layers + 1`` (index 0 = the embedding layer, the
    final index = the last transformer block's output).

    Args:
        layers_cfg: "all" -> every hidden state (embeddings .. last layer), OR a
            list/tuple of integer indices to keep.
        n_hidden_states: number of hidden-state tensors emitted by the model.

    Returns:
        A list of int layer indices in [0, n_hidden_states).

    Raises:
        ValueError: on an unrecognized spec or an out-of-range index.
    """
    if isinstance(layers_cfg, str):
        if layers_cfg == "all":
            return list(range(n_hidden_states))
        raise ValueError(
            f"layers must be 'all' or a list of ints, got string {layers_cfg!r}"
        )

    if isinstance(layers_cfg, (list, tuple)):
        resolved: list[int] = []
        for layer in layers_cfg:
            # Reject bools (a subclass of int) and non-integers up front.
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise ValueError(
                    f"layer index must be an int, got {layer!r} ({type(layer).__name__})"
                )
            if layer < 0 or layer >= n_hidden_states:
                raise ValueError(
                    f"layer index {layer} out of range [0, {n_hidden_states})"
                )
            resolved.append(layer)
        return resolved

    raise ValueError(
        f"layers must be 'all' or a list of ints, got {type(layers_cfg).__name__}"
    )


def pool_hidden_states(hidden, attention_mask, pooling: str):
    """Pool a layer's per-token hidden states into one vector per sequence.

    Pooling is done strictly over REAL tokens using ``attention_mask`` — padding
    is never included. The math is performed in float32 for numerical stability.

    Args:
        hidden: float tensor of shape [batch, seq, hidden_dim].
        attention_mask: tensor of shape [batch, seq] (1 = real token, 0 = pad).
        pooling: "last_token" or "mean".

    Returns:
        A float32 tensor of shape [batch, hidden_dim].

    Raises:
        ValueError: on an unknown pooling mode.
    """
    hidden = hidden.float()
    # Mask in float32 on the hidden tensor's device for broadcasting/indexing.
    mask = attention_mask.to(device=hidden.device, dtype=torch.float32)
    batch, seq, _ = hidden.shape

    if pooling == "last_token":
        # Padding-side agnostic: position index of the LAST real token per row.
        # arange * mask zeroes out pad positions; argmax then picks the largest
        # surviving position, i.e. the final real token regardless of pad side.
        pos = torch.arange(seq, device=hidden.device, dtype=torch.float32)
        idx = (mask * pos.unsqueeze(0)).argmax(dim=1)  # [batch]
        rows = torch.arange(batch, device=hidden.device)
        return hidden[rows, idx]  # [batch, hidden_dim]

    if pooling == "mean":
        # Mask-weighted mean over real tokens only; clamp denom to avoid /0 on an
        # (unexpected) all-pad row.
        mask3 = mask.unsqueeze(-1)  # [batch, seq, 1]
        summed = (hidden * mask3).sum(dim=1)  # [batch, hidden_dim]
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [batch, 1]
        return summed / counts

    raise ValueError(f"unknown pooling mode {pooling!r}; expected 'last_token' or 'mean'")


def extract_activations(
    texts: list[str],
    model,
    tokenizer,
    *,
    pooling: str,
    layers,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Extract pooled per-layer activations for a list of texts.

    Runs forward passes only (no generation). The caller must have set
    ``tokenizer.pad_token`` and fixed all seeds for determinism.

    Args:
        texts: node question strings to embed.
        model: a loaded HF causal LM (already on ``device``).
        tokenizer: the matching tokenizer (pad_token set).
        pooling: "last_token" or "mean" (see ``pool_hidden_states``).
        layers: "all" or a list of int layer indices (see ``resolve_layers``).
        batch_size: number of texts per forward pass.
        device: torch device string the inputs are moved to.

    Returns:
        float32 ndarray of shape [len(texts), n_selected_layers, hidden_dim].
    """
    model.eval()
    batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple of length n_hidden_states
            sel = resolve_layers(layers, len(hs))

            # Pool each selected layer, then stack along a new layer axis ->
            # [batch, n_selected_layers, hidden_dim]. Each pooled vector is moved to
            # CPU/float32 BEFORE stacking: under device_map="auto" different layers
            # can live on different GPUs, so they cannot be stacked in place.
            pooled = [
                pool_hidden_states(hs[l], enc["attention_mask"], pooling).to(
                    "cpu", dtype=torch.float32
                )
                for l in sel
            ]
            stacked = torch.stack(pooled, dim=1)  # [batch, n_sel, hidden_dim] on CPU

            batches.append(stacked.numpy())

    acts = np.concatenate(batches, axis=0)  # [len(texts), n_sel, hidden_dim]

    # Validity guard: pooled activations must be finite.
    assert np.isfinite(acts).all(), "extracted activations contain NaN or Inf"

    return acts
