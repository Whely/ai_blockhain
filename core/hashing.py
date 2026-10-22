"""Weight hashing.

The hash is the identity of a model state: it travels in round messages and
lets the coordinator tell "same weights" from "new weights" without shipping
tensors twice. It is *not* proof that someone trained -- see README.
"""

from __future__ import annotations

import hashlib

import torch


def hash_weights(weights: dict[str, torch.Tensor]) -> str:
    """Stable sha256 over parameter names, shapes and raw float32 bytes."""
    digest = hashlib.sha256()
    for name in sorted(weights):
        tensor = weights[name].detach().cpu().contiguous()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
