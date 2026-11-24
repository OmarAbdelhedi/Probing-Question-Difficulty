"""Tests for src/activations.py: layer-spec resolution and mask-aware pooling.

Needs torch (CPU only -- tiny tensors, no model); skips without it.
``extract_activations`` is exercised end to end in test_pipeline.py.
"""

import pytest

torch = pytest.importorskip("torch")

from src.activations import pool_hidden_states, resolve_layers  # noqa: E402


# --------------------------------------------------------------------------- #
# resolve_layers                                                              #
# --------------------------------------------------------------------------- #
def test_resolve_layers_all():
    assert resolve_layers("all", 5) == [0, 1, 2, 3, 4]


def test_resolve_layers_explicit_list_and_tuple():
    assert resolve_layers([0, 3], 5) == [0, 3]
    assert resolve_layers((4,), 5) == [4]


@pytest.mark.parametrize("bad", ["some", [True], [3.5], [-1], [5], 7])
def test_resolve_layers_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        resolve_layers(bad, 5)


# --------------------------------------------------------------------------- #
# pool_hidden_states                                                          #
# --------------------------------------------------------------------------- #
def test_last_token_is_padding_side_agnostic():
    real = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)  # 3 real tokens
    pad = torch.full((1, 1, 3), 99.0)
    right = torch.cat([real, pad], dim=1)
    left = torch.cat([pad, real], dim=1)
    mask_right = torch.tensor([[1, 1, 1, 0]])
    mask_left = torch.tensor([[0, 1, 1, 1]])

    out_right = pool_hidden_states(right, mask_right, "last_token")
    out_left = pool_hidden_states(left, mask_left, "last_token")
    expected = real[:, -1, :]  # the final REAL token, never the pad
    assert torch.equal(out_right, expected)
    assert torch.equal(out_left, expected)


def test_mean_pooling_ignores_padding():
    hidden = torch.tensor([[[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]]])  # last is pad
    mask = torch.tensor([[1, 1, 0]])
    out = pool_hidden_states(hidden, mask, "mean")
    assert torch.equal(out, torch.tensor([[4.0, 6.0]]))  # mean of real tokens only


def test_pooling_returns_float32():
    hidden = torch.zeros((2, 3, 4), dtype=torch.float64)
    mask = torch.ones((2, 3), dtype=torch.long)
    for mode in ("last_token", "mean"):
        assert pool_hidden_states(hidden, mask, mode).dtype == torch.float32


def test_unknown_pooling_mode_raises():
    with pytest.raises(ValueError, match="unknown pooling mode"):
        pool_hidden_states(torch.zeros((1, 2, 3)), torch.ones((1, 2)), "cls")
