"""Deterministic data splits.

Four pools, not the usual train/test pair, because the protocol needs them:

    miner       -- sliced between miners for local training
    provider    -- data that participants may submit for a reward
    public_val  -- open validation, anyone may measure on it
    hidden_test -- only the coordinator sees it, used to score a round

The split is seeded, so every participant derives the same pools from the same
dataset without shipping the data around.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

# MNIST train split is 60_000, test split is 10_000.
MINER_SIZE = 40_000
PROVIDER_SIZE = 10_000
PUBLIC_VAL_SIZE = 10_000


@dataclass(frozen=True)
class DataPools:
    miner: TensorDataset
    provider: TensorDataset
    public_val: TensorDataset
    hidden_test: TensorDataset


def _as_tensors(dataset: datasets.MNIST) -> tuple[torch.Tensor, torch.Tensor]:
    images = dataset.data.to(torch.float32).unsqueeze(1) / 255.0
    labels = dataset.targets.to(torch.int64)
    return images, labels


def _subset(
    images: torch.Tensor, labels: torch.Tensor, index: torch.Tensor
) -> TensorDataset:
    return TensorDataset(images[index].contiguous(), labels[index].contiguous())


def load_pools(root: str = "data", *, seed: int = 0, download: bool = True) -> DataPools:
    """Download (once) and split MNIST into the four protocol pools."""
    to_tensor = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=download, transform=to_tensor)
    test = datasets.MNIST(root, train=False, download=download, transform=to_tensor)

    images, labels = _as_tensors(train)
    order = torch.randperm(images.shape[0], generator=torch.Generator().manual_seed(seed))

    miner = order[:MINER_SIZE]
    provider = order[MINER_SIZE : MINER_SIZE + PROVIDER_SIZE]
    public_val = order[MINER_SIZE + PROVIDER_SIZE :]

    test_images, test_labels = _as_tensors(test)
    hidden = torch.arange(test_images.shape[0])

    return DataPools(
        miner=_subset(images, labels, miner),
        provider=_subset(images, labels, provider),
        public_val=_subset(images, labels, public_val),
        hidden_test=_subset(test_images, test_labels, hidden),
    )


def shard(
    dataset: TensorDataset, parts: int, *, seed: int = 0
) -> list[TensorDataset]:
    """Split a dataset into `parts` disjoint shards, one per miner."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    images, labels = dataset.tensors
    order = torch.randperm(images.shape[0], generator=torch.Generator().manual_seed(seed))
    chunks = order.chunk(parts)
    return [_subset(images, labels, chunk) for chunk in chunks]
