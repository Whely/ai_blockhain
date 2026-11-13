import pytest

from core.rewards import mining_reward


def test_normal_gain_pays_proportionally():
    assert mining_reward(0.90, 0.95) == pytest.approx(5.0)


def test_gain_below_threshold_pays_nothing():
    assert mining_reward(0.90, 0.901) == 0.0


def test_worse_model_pays_nothing():
    assert mining_reward(0.95, 0.90) == 0.0


def test_invalid_accuracy_raises():
    with pytest.raises(ValueError):
        mining_reward(1.2, 0.9)


def test_gain_at_threshold_pays_nothing():
    assert mining_reward(0.5, 0.75, threshold=0.25) == 0.0


def test_acc_new_above_one_raises():
    with pytest.raises(ValueError, match="acc_new"):
        mining_reward(0.9, 1.5)


def test_zero_k_raises():
    with pytest.raises(ValueError, match="k must be > 0"):
        mining_reward(0.9, 0.95, k=0)


def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        mining_reward(0.9, 0.95, threshold=-0.1)


def test_reward_scales_linearly_with_k():
    base = mining_reward(0.90, 0.95, k=100)
    doubled = mining_reward(0.90, 0.95, k=200)
    assert doubled == pytest.approx(2 * base)
