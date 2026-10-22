from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def synthetic_dataset(n: int = 2000, seed: int = 0) -> TensorDataset:
    """Linearly separable blobs shaped like MNIST, so tests stay offline and fast."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 10, (n,), generator=generator)
    images = 0.1 * torch.randn(n, 1, 28, 28, generator=generator)
    flat = images.view(n, -1)
    flat[torch.arange(n), labels] += 3.0
    return TensorDataset(flat.view(n, 1, 28, 28), labels)
