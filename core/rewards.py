"""Reward formulas for the protocol.

Accuracy is a fraction in [0, 1], not a percentage.
"""



def mining_reward(acc_old, acc_new, k=100.0, threshold=0.002):
    """Reward for one training round.

    Args:
        acc_old: accuracy before the round, a fraction in [0, 1].
        acc_new: accuracy after the round, a fraction in [0, 1].
        k: tokens per unit of accuracy gain.
        threshold: minimum gain that counts, filters measurement noise.

    Returns:
        Non-negative reward; zero when the gain is at or below the threshold.

    Raises:
        ValueError: if accuracy is outside [0, 1], or k/threshold are invalid.
    """
    if not 0.0 <= acc_old <= 1.0:
        raise ValueError(f"acc_old must be in [0, 1], got {acc_old}")
    if not 0.0 <= acc_new <= 1.0:
        raise ValueError(f"acc_new must be in [0, 1], got {acc_new}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    gain = acc_new - acc_old
    return k * gain if gain > threshold else 0.0