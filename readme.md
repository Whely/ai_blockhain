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

## Open problems

These are unsolved here on purpose — they are the interesting part:

- proving that a miner actually trained rather than submitting noise;
- overfitting to the hidden validation set, hence rotation of hidden slices;
- non-determinism of training vs. reproducible consensus;
- Sybil, weight-copying, data-poisoning and collusion attacks;
- economics on the accuracy plateau, where gains become too expensive.

## Related work

Bittensor, Gensyn, Numerai, Prime Intellect; FedAvg (McMahan et al., 2017);
Proof-of-Learning (Jia et al., 2021).

## License

GPL-3.0 — see [LICENSE](LICENSE).
