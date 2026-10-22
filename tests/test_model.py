from __future__ import annotations

import torch

from core.hashing import hash_weights
from core.model import evaluate, initial_weights, train

from .helpers import synthetic_dataset


def test_training_improves_accuracy() -> None:
    data = synthetic_dataset()
    weights = initial_weights(0)

    before = evaluate(weights, data)
    trained = train(weights, data, epochs=2, seed=0, lr=0.1)
    after = evaluate(trained, data)

    assert before < 0.5, f"untrained model should be near chance, got {before:.3f}"
    assert after > 0.8, f"training should learn the blobs, got {after:.3f}"


def test_training_is_deterministic() -> None:
    data = synthetic_dataset()
    weights = initial_weights(0)

    first = train(weights, data, epochs=1, seed=7)
    second = train(weights, data, epochs=1, seed=7)

    assert hash_weights(first) == hash_weights(second)


def test_different_seeds_diverge() -> None:
    data = synthetic_dataset()
    weights = initial_weights(0)

    first = train(weights, data, epochs=1, seed=1)
    second = train(weights, data, epochs=1, seed=2)

    assert hash_weights(first) != hash_weights(second)


def test_training_does_not_mutate_input_weights() -> None:
    data = synthetic_dataset()
    weights = initial_weights(0)
    before = hash_weights(weights)

    train(weights, data, epochs=1, seed=0)

    assert hash_weights(weights) == before


def test_weights_are_cpu_tensors() -> None:
    data = synthetic_dataset(n=256)
    trained = train(initial_weights(0), data, epochs=1, seed=0)

    assert all(tensor.device.type == "cpu" for tensor in trained.values())
    assert all(tensor.dtype == torch.float32 for tensor in trained.values())
