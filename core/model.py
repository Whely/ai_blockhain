"""Tiny MLP used as the test fixture for the protocol.

The model is deliberately small and boring: a round must take seconds on a CPU,
because the interesting part of this project is the incentive mechanism, not
accuracy. Miners may run the same code on a GPU -- `device` is a parameter, not
an assumption baked into the logic.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGE_SHAPE = (1, 28, 28)
N_CLASSES = 10


def make_model() -> nn.Module:
    """784 -> 128 -> 10 MLP. Same graph everywhere, so weights are comparable."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, N_CLASSES),
    )


def get_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of the state dict, safe to serialize and hash."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def initial_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    """Seeded starting point, so two runs begin from identical numbers."""
    torch.manual_seed(seed)
    return get_weights(make_model())



def set_weights(model: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v for k, v in weights.items()})


def train(
    weights: dict[str, torch.Tensor],
    dataset: Dataset,
    *,
    epochs: int = 1,
    seed: int = 0,
    device: str = "cpu",
    lr: float = 0.1,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Fine-tune a copy of `weights` on `dataset` and return the new weights.

    The input dict is never mutated: callers keep the original model intact,
    which matters once several miners train from the same round snapshot.
    """
    torch.manual_seed(seed)

    model = make_model().to(device)
    set_weights(model, weights)
    model.train()

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()

    return get_weights(model)


@torch.no_grad()
def evaluate(
    weights: dict[str, torch.Tensor],
    dataset: Dataset,
    *,
    device: str = "cpu",
    batch_size: int = 512,
) -> float:
    """Accuracy in [0, 1] of `weights` on `dataset`."""
    model = make_model().to(device)
    set_weights(model, weights)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == targets).sum())
        total += int(targets.numel())
    return correct / max(total, 1)
