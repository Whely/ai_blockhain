from __future__ import annotations

import torch

from core.hashing import hash_weights
from core.model import get_weights, make_model


def test_same_weights_same_hash() -> None:
    weights = get_weights(make_model())
    assert hash_weights(weights) == hash_weights(weights)


def test_hash_is_independent_of_key_order() -> None:
    weights = get_weights(make_model())
    reordered = dict(reversed(list(weights.items())))
    assert hash_weights(weights) == hash_weights(reordered)


def test_changed_weights_change_hash() -> None:
    weights = get_weights(make_model())
    changed = {k: v.clone() for k, v in weights.items()}
    changed[next(iter(changed))] += 1e-4

    assert hash_weights(changed) != hash_weights(weights)


def test_hash_has_fixed_length() -> None:
    assert len(hash_weights(get_weights(make_model()))) == 64


def test_hash_accepts_non_float32_inputs() -> None:
    weights = {"w": torch.zeros(2, dtype=torch.float64)}
    as_float64 = hash_weights(weights)
    as_float32 = hash_weights({"w": torch.zeros(2, dtype=torch.float32)})
    assert as_float64 == as_float32
