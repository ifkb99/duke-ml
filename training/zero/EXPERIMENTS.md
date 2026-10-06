# Experiment log

Results and conclusions, newest last. Raw numbers live in
`runs/<run>/metrics.jsonl` (training + evals) and `runs/<run>/diagnostics/`
(one-off experiments); the scripts are in `diagnostics/`.

## az1: first run (2026-10-05 21:41 → 2026-10-06)

Settings: 96×6 ResNet, Gumbel search with 32 simulations and 16 candidates,
replay of 1M positions, sample reuse 4, lr 3e-4, warm-start on 3,000 minimax
games (4,000 SGD steps ≈ 22 passes over them).

**Outcome: no learning.** 2,330 iterations and 1.36M self-play games in 10.5 h.

Evaluations, averaged per iteration range (≈20 evals × 64 games each):

| iterations | vs minimax 2 | vs minimax 4 |
|---|---|---|
| 0–100 | 0.52 | 0.07 |
| 100–500 | 0.40 | 0.07 |
| 500–1000 | 0.39 | 0.06 |
| 1000–1500 | 0.42 | 0.06 |
| 1500–2000 | 0.39 | 0.06 |
| 2000–2400 | 0.37 | 0.07 |

### Diagnosis (2026-10-06 morning)

| check | result | meaning |
|---|---|---|
| 32 vs 200 simulations vs minimax 4 (iteration 100, 128 games) | 0.09 vs 0.13 | more search doesn't help, so the network's judgment is the bottleneck |
| iteration 2200 vs iteration 200 (256 games) | 0.49 | **no self-improvement at all** |
| search target vs policy (iteration 2200) | policy entropy 2.00 (uniform over 9 moves ≈ 2.2), target 1.28, argmax agreement 22% | policy is near-uniform and not fitting the targets |
| value consistency, v(s) vs its children (iteration 2200) | corr 0.37; side to move rated −0.17 on average | value head contradicts itself one move later |
| value targets vs true winner (39k positions) | 0 mismatches | targets are correct |
| input planes decoded back to the board | exact | inputs are correct |
| held-out value loss on fresh games | iteration 2200: MSE 0.73 vs 0.56 for predicting 0, corr 0.06 (training loss 0.29); warm-start: 1.61 vs 0.95, corr −0.01 | **value head is memorizing, not generalizing** |
| fresh net on 7.5k self-play games, split by game | test corr peaks at 0.13 after half an epoch, then falls (0.04 at 6 epochs) as train MSE drops 1.07 → 0.50 | the signal in self-play outcomes is weak, and memorization sets in after ~1 pass |
| fresh net on the 3k minimax games, split by game | test corr ≤ 0.04 (training MSE → 0.16) | memorizes; 2,400 training games is too few independent labels |
| same, no extra features / no board | board only: 0.08; features only: ≈0 | the board carries the signal; the features don't cause the memorization |
| tile-difference formula, minimax games | corr 0.13 overall; by turn: 0.22 (30–60), 0.32 (60–100), 0.72 (100+, leader with 3+ tiles wins 95%) | outcomes are learnable from the board in competent play |
| tile-difference formula, self-play games | corr 0.10–0.17 at every stage | self-play games are close to random |
| net trained to predict tile difference itself | test corr 0.82 after one pass (rising) | network and training code can learn board functions fine |

**Conclusion.** A degenerate loop, not a code bug:

1. The warm-start ran ~22 passes over 3,000 games. Its value head memorized
   them and came out confidently wrong on new positions.
2. Search with a wrong value function gives noisy Q-values. Gumbel rescales
   Q to 0–1 per position, so the noise becomes strong, inconsistent targets.
3. The policy can't fit inconsistent targets and drifts toward uniform, so
   self-play becomes near-random.
4. Outcomes of near-random games are barely predictable (corr ~0.1), and with
   each sample reused 4× the value head memorizes the replay buffer instead.
   That closes the loop.

Game outcomes as value targets are high-variance: one ±1 label per ~80
positions, decided by blunders far in the future. That's tolerable with
AlphaZero's data volume (each position seen ~once, millions of games), not
with 12k games in the buffer seen 4× each.

## az2: fixes for the az1 failure (started 2026-10-06)

Changes from az1 (everything else the same):

| setting | az1 | az2 | why |
|---|---|---|---|
| value target | final result | 10-step bootstrapped return (`value_nstep=10`) | far less noise per target |
| sample reuse | 4 | 1 | az1 memorized its buffer |
| replay buffer | 1M | 2M | more different games per window |
| warm-start | 4,000 steps (~22 passes) | early-stopped on 10% held-out games | az1's warm-start value head was memorized |
| replay at self-play start | kept the minimax samples | empty | minimax samples only have result targets |
| games per GPU | 960 / 704 | 872 / 792 | less SGD on GPU 0 now |

Warm-start data: reused `runs/az1/imitation_data.npz` (same 3,000 minimax games).

New logging (`metrics.jsonl`): `fresh_*` value and policy stats on every
iteration's new games before training on them, `ladder` matches against the
network from 100 iterations earlier, `imitation` held-out curves, and `start`
records with config and git commit.

**What success looks like:** `fresh_value_corr_mc` rising well above az1's
~0.05, `fresh_policy_kl` falling, ladder scores above 0.5, and then the minimax
scores moving.
