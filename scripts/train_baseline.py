"""Train the reference MLP and report accuracy before/after.

Usage (from the repository root):

    python -m scripts.train_baseline
    python -m scripts.train_baseline --epochs 3 --device cpu --limit 5000
"""

from __future__ import annotations

import argparse

from torch.utils.data import Subset

from core.data import load_pools
from core.hashing import hash_weights
from core.model import evaluate, initial_weights, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu or cuda (miners may use GPUs)")
    parser.add_argument("--limit", type=int, default=0, help="use only N training samples")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    pools = load_pools(args.data_root, seed=args.seed)
    dataset = pools.miner
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    weights = initial_weights(args.seed)
    before = evaluate(weights, pools.public_val, device=args.device)

    trained = train(
        weights,
        dataset,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    after_val = evaluate(trained, pools.public_val, device=args.device)
    after_test = evaluate(trained, pools.hidden_test, device=args.device)

    print(f"samples   : {len(dataset)}")
    print(f"accuracy  : {before:.4f} -> {after_val:.4f} (public val), {after_test:.4f} (hidden test)")
    print(f"weight hash: {hash_weights(trained)}")


if __name__ == "__main__":
    main()
