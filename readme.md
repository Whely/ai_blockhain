# ai-blockhain

**Research prototype — test tokens have no value.**

A simulator and reference implementation of incentive mechanisms for decentralized
machine learning. Participants train a shared model, the network measures how much the
model improved on a hidden test set, and issues test tokens in proportion to that gain.
A second role — data providers — is rewarded for valid, unique samples.

The project is deliberately *not* presented as a decentralized AI network. It is a
small, auditable playground for the mechanism design behind such networks: reward
formulas, staking, slashing, and the attacks that try to break them.

## Roles

| Role | What it does | Reward | Protection |
|---|---|---|---|
| Miner | Takes the current weights, fine-tunes on a local data shard, submits the result | Tokens proportional to accuracy gain on a hidden set | Gain threshold, rotating hidden slices, duplicate-weight detection |
| Data provider | Uploads a batch of pre-labelled data | Flat reward per valid unique sample | Hash deduplication, format checks, stake and slashing, spot checks |
| Validator (planned) | Independently scores submitted weights | Share of the reward for honest scoring | Stake-weighted median, outlier penalties, dispute window |

## Status

Early development: repository skeleton only, no working code yet. Stages are built one
at a time, each ending in a runnable result and a commit.

Rough order: data and model → training rounds with several miners (FedAvg) →
adversarial miners and reward hardening → data providers → toy blockchain →
dashboard. Smart contracts on a public testnet are a later, optional step.

## Repository layout

```
core/        model, training, evaluation, reward formulas
chain/       toy blockchain (contracts later)
sim/         honest and adversarial participant scenarios
dashboard/   Streamlit dashboard
tests/       pytest
docs/        diagrams, threat model, notes
```

## How verification works

Rewards are paid for outcomes, not for compute. The network scores submitted weights on a
hidden test set and pays for the measured accuracy gain, so the cheap check is a score rather
than a proof of work: evaluating weights costs far less than producing them. It is the same
asymmetry that makes `hash < target` cheap to verify in Bitcoin, with "did the metric
improve" as the predicate.

That check covers lazy and noisy submissions — no gain, no reward — but it says nothing about
where a good result came from, which is the free-riding problem below. Where a stronger
guarantee is needed, the plan is deterministic re-execution: a validator replays the round
from the submitted seed, data and hyperparameters, recomputes the weights and compares hashes,
with an optimistic acceptance window and a dispute period around it. Re-execution only works
if training is bit-exact reproducible, which is why models stay small, CPU-only and seeded.

## Open problems

Unsolved here on purpose — they are the interesting part:

- **Attribution and free-riding.** Gain is non-additive under averaging: several miners
  together improve the model more than any of them alone, and no formula splits that gain
  uniquely. Copies of submitted weights are also hard to distinguish from independent work,
  and a submitter can withhold a good model to farm a larger "gain" later.
- **Proof of training.** Only becomes relevant if the reward starts paying for compute instead
  of results. The credible options (Proof-of-Learning, zkML) are far more expensive than
  training itself; deterministic re-execution is the practical middle ground at toy scale.
- **Overfitting to the hidden set.** A model can be tuned to the very data used to score it,
  making "gain" meaningless; hidden slices therefore have to rotate, and the slice commitment
  has to be fixed before the round rather than after it.
- **Non-determinism.** Floating point, thread scheduling and library versions make identical
  training runs diverge, while consensus needs bit-exact results; CPU-only runs, fixed seeds
  and quantised weights narrow the gap without closing it.
- **Sybil, poisoning and collusion.** Many identities for one operator, mislabelled data,
  duplicated submissions and coordinated groups all distort rewards; staking, slashing,
  deduplication and per-address caps raise the cost of an attack but never remove it.
- **Economics at the plateau.** The better the model, the more a further gain costs, so the
  incentive fades exactly when quality is high; emission schedules and subsidy mechanisms
  move that point around rather than eliminating it.

## Related work

Bittensor, Gensyn, Numerai, Prime Intellect; FedAvg (McMahan et al., 2017);
Proof-of-Learning (Jia et al., 2021).

## License

GPL-3.0 — see [LICENSE](LICENSE).
