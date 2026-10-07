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
| warm-start | 4,000 steps (~22 passes), policy + value | policy only, early-stopped on 10% held-out games | az1's warm-start value head was memorized |
| value head at start | random init | output layer zero-initialized (value 0 everywhere) | honest "don't know" instead of noise |
| replay at self-play start | kept the minimax samples | empty | minimax samples only have result targets |
| games per GPU | 960 / 704 | 872 / 792 | less SGD on GPU 0 now |

Warm-start data: reused `runs/az1/imitation_data.npz` (same 3,000 minimax games).

First attempt (09:30, aborted after the warm-start): trained policy + value with
early stopping on their sum. Held-out value loss got worse from the first check
on (0.975 → 1.125 by step 896, already worse than predicting 0, ≈0.95) while
held-out policy loss kept improving (1.20 → 1.12), so the sum hid the value
head memorizing. Restarted with the policy-only warm-start.

New logging (`metrics.jsonl`): `fresh_*` value and policy stats on every
iteration's new games before training on them, `ladder` matches against the
network from 100 iterations earlier, `imitation` held-out curves, and `start`
records with config and git commit.

**What success looks like:** `fresh_value_corr_mc` rising well above az1's
~0.05, `fresh_policy_kl` falling, ladder scores above 0.5, and then the minimax
scores moving.

### az2 first 100 iterations (2026-10-06, 09:34 → 09:55)

Warm-start (policy only): early-stopped at step 1280 of 4,000 (held-out
policy loss 1.041, about 8 passes over the data).

| point | random | minimax 2 | minimax 4 | fresh value corr | policy/search agree |
|---|---|---|---|---|---|
| after warm-start | 1.00 | 0.44 | 0.11 | — | — |
| iteration 10 | | | | 0.07 | 0.67 |
| iteration 25 | 1.00 | 0.80 | 0.33 | | |
| iteration 50 | 1.00 | 0.78 | 0.27 | 0.26 | 0.66 |
| iteration 75 | 1.00 | 0.78 | 0.41 | | |
| iteration 100 | 1.00 | **0.91** | **0.46** | 0.26 | 0.75 |

**Ladder at iteration 100 vs the warm-start network: 0.94.** It is learning;
az1 scored 0.49 between iterations 2200 and 200. In 20 minutes az2 passed
everything az1 reached in 10.5 hours (best az1 scores: minimax 2 0.56,
minimax 4 0.14).

### az2 result (stopped 2026-10-06 13:37 at iteration 1100, ~3.6 h of self-play, both GPUs)

Evaluations averaged per 100 iterations (4 evals × 64 games each; minimax 6 is
one 64-game eval every 100 iterations from iteration 700):

| iterations | minimax 2 | minimax 4 | minimax 6 |
|---|---|---|---|
| 1–100 | 0.82 | 0.37 | |
| 201–300 | 0.87 | 0.60 | |
| 401–500 | 0.88 | 0.77 | |
| 601–700 | 0.91 | 0.76 | 0.59 (32 games at 600 offline: 0.59) |
| 801–900 | 0.94 | 0.85 | 0.65 |
| 901–1000 | 0.93 | 0.87 | 0.77 |
| 1001–1100 | 0.93 | 0.90 | 0.73 |

Ladder (64 games, vs 100 iterations earlier): 0.94, 0.74, 0.61, 0.73, 0.58,
0.57, 0.52 (700 vs 625), 0.57, 0.54, 0.32 (1000 vs 900).
Bigger checks, 256 games: 1000 vs 800 **0.52**, 1000 vs 600 **0.73**.
So the 0.32 was noise, but self-improvement had slowed to ~nothing per 200
iterations by iteration 1000, while the minimax scores still crept up.

Health at the end: fresh value corr 0.41 (from 0.21 at iteration 100, flat
since ~500), policy/search agreement 0.81 (flat since 300), games 88 plies,
89% decisive (from 56 plies, 96%).

## az3: bigger network vs more simulations (started 2026-10-06 13:38)

Two runs side by side, **one GPU each**, both from az2's code and settings
(policy-only warm-start on the same minimax games, n-step 10, reuse 1), each
changing one thing:

| run | GPU | change | games in parallel |
|---|---|---|---|
| az3-net | 0 | network 128 channels × 10 blocks (az2: 96 × 6) | 872 |
| az3-sims | 1 | 64 simulations per move (az2: 32), max depth 24 | 768 (search trees take 2× memory) |

Each has half of az2's hardware, so compare against az2 by **self-play games**
or **GPU-hours** (az2 used 2 GPUs), not wall-clock. Planned comparisons: their
minimax curves and ladders, and head-to-head matches between them and against
az2 checkpoints (`diagnostics/ladder2.py`).

### az3 result: bigger network wins (2026-10-06 17:38)

At ~3.4 GPU-hours each (az3-net iteration 725, az3-sims iteration 400),
head-to-head with **the same search for both (32 simulations)**, 256 games:

| match | score |
|---|---|
| az3-sims vs az3-net | 0.37 |
| az3-sims vs az2 iteration 400 (~2.8 GPU-hours) | 0.40 |
| az3-net vs az2 iteration 400 | 0.53 |

**Caveat found on the way:** a run's own training evals use its own search
settings, so az3-sims played minimax with 64 simulations. Its better minimax
curve (e.g. minimax 4 0.75 vs 0.60 per 100k games) partly measured extra
thinking at play time, not a better network. The evaluator (evaluator.py) now
rates every checkpoint at the same 32 simulations.

Last training evals before stopping: az3-net iteration 850: minimax 2 0.91,
minimax 4 0.84; az3-sims iteration 425 (at 64 simulations): minimax 2 0.95,
minimax 4 0.91.

**Decision:** stop az3-sims; continue az3-net (128 × 10) on both GPUs from
iteration 850 (games 872 / 792, memory fraction 0.7 to leave room for the
evaluator). Doubling simulations halves the games per hour, and at this stage
more games beat better targets per game.

## Phone network: spatial policy head (2026-10-06, branch web/phone-ai)

The goal is to play in the browser on a phone (tools/latency: on the phone's
CPU, a 128 × 10 network takes ~25 ms per position; on WebGPU, ~2–4 ms in
batches). The download was the problem: the dense policy layer
(1,280 → 3,605) is 4.6M of az3-net's 7.7M parameters, 15 MB as fp16.

**Spatial head** (`--policy-head spatial`, network.py): each action's logit
is read off the square it starts from, in rules.py's numbering. A 1 × 1 conv
gives every square 36 move targets + 64 command (src, dst) logits, moves add a
bilinear q(from)·k(to) term, and draw/placements come from the pooled board.
It has ~21K parameters: a 128 × 10 network drops to 3.1M parameters, 6.2 MB as
fp16, at the same speed.

**Does it lose anything?** Distillation (`distill.py`): az3-net iteration 1075
played 402,931 self-play positions (32 simulations, ~1,300 positions/s on half
a GPU). Two students learned them with az3-net's trunk and value head copied
in and only the policy head fresh, early-stopped on 340 held-out games:

| held-out games | policy loss | top-1 vs search | value loss |
|---|---|---|---|
| teacher (az3-net, dense head, millions of samples) | 0.617 | 0.815 | 0.082 |
| student, spatial head | **0.714** | 0.769 | 0.085 |
| student, dense head (control) | 0.719 | 0.769 | 0.086 |

Same data, same budget: the spatial head matches the dense one (and overfits
a little less: train policy loss 0.545 vs 0.504 at step 2,048). Both students
stop well short of the teacher because 360k positions are seen ~6 times
before held-out loss turns up; the gap is data, not the head.

**Next:** RL-train the spatial network from the distilled student
(`runs/distill-spatial/imitation.pkl`, 128 × 10, az3-net's trunk), which
continues where az3-net is instead of starting over.

## Network review before more training (2026-10-06 19:30)

`diagnostics/review.py` on the depth-8 benchmark (2,100 positions) and on each
network's own self-play (256 games). Training stopped at az3-net iteration 1150.

**The benchmark's accuracy/blunder numbers are biased toward drawing.**
Minimax's evaluation counts each board tile at full material value but the bag
only as 2 × its *average* tile value, so a draw gains a tile of material for
free. Where drawing is legal it beats the best board move by a median 3.8
points; in 54% of those positions not drawing counts as a "blunder", and 62% of
the "puzzles" are "draw". Minimax's best move is a draw in 46% of positions;
the networks draw ~25% of the time (and beat minimax 4 in 84% of games). So
acc/blunder/puzzle mostly measure agreement with that heuristic. The forced
outcomes don't depend on it:

| | az2 1000 | az3-net 1000 | az3-net 1150 | spatial student | dense student |
|---|---|---|---|---|---|
| missed forced win: policy | 0.62 | 0.66 | 0.63 | 0.68 | 0.68 |
| missed forced win: search 32 | 0.46 | 0.49 | **0.45** | 0.48 | 0.46 |
| missed forced win: search 128 | 0.39 | 0.39 | **0.35** | 0.39 | 0.43 |
| walked into a forced loss: search 32 | 0.8% | 0.7% | 0.7% | 0.7% | 1.0% |
| acc when minimax's best is a board move: search 128 | 0.70 | 0.68 | 0.69 | 0.68 | 0.68 |
| value: mean in forced-win positions | 0.55 | 0.56 | 0.57 | 0.59 | 0.60 |
| value vs outcome, own self-play (corr) | 0.41 | 0.35 | 0.43 | 0.38 | 0.41 |
| ... opening / middle / late / endgame | .21/.32/.58/.80 | .18/.30/.50/.70 | .20/.39/.57/.84 | .16/.34/.53/.75 | .20/.35/.56/.74 |
| calibration error | 0.040 | 0.017 | 0.057 | 0.015 | 0.019 |

(65 positions have a forced win, 1,522 a decisive minimax score.)

Findings:
- **Tactics are the clear weakness.** The raw policy misses ~2/3 of forced
  wins; search finds them (32 → 128 simulations: 0.45 → 0.35 missed). The
  value head doesn't recognise won positions (mean 0.57 where a forced win
  exists), so wins are found only when the search reaches the end.
- **More simulations at play time help directly**, which suits the phone
  (WebGPU fits ~800–1,400 per 3 s).
- **Search at 32 simulations barely beats the raw policy** on move choice
  (az3-net 1150: 0.581 vs 0.582 acc): with 16 root candidates most get 1–2
  visits, so the result is mostly prior + one-step values.
- The value head predicts outcomes poorly early (corr 0.2 in the opening),
  which is partly inherent (long, chancy games).
- az3-net 1000 → 1150 improved on every robust metric (and the ladder: 0.62).
- The distilled students sit at az3-net ~1000–1100 level: a fine start for
  RL with the spatial head.
- Policy probability on illegal moves (before masking) is ~99% for every
  network: the loss masks them, so they're never trained. Harmless (search
  masks too), and not a measure of rule knowledge.
- Commands are almost never best (0.3%) or played (0.1%).

## Tactics benchmark and the search's value scale (2026-10-06 20:45)

**Tactics benchmark** (`tactics.py`): 120,000 positions from fresh az3-net
iteration-1150 self-play, every move scored by depth-6 minimax. Kept: 2,839
positions with a forced win (does the move keep it?) and 6,157 where ≥30% of
the moves walk into a forced loss (does the move avoid them?). Forced results
don't depend on minimax's heuristic.

| network | policy: win kept | search 32 | search 128 | policy: loss avoided | search 32 | search 128 |
|---|---|---|---|---|---|---|
| az2 1000 | 0.29 | 0.53 | 0.65 | 0.78 | 0.85 | 0.88 |
| az3-net 600 | 0.21 | 0.49 | 0.60 | 0.71 | 0.79 | 0.81 |
| az3-net 1000 | 0.27 | 0.53 | 0.67 | 0.76 | 0.84 | 0.87 |
| az3-net 1150 | 0.27 | 0.54 | 0.67 | 0.79 | 0.86 | 0.89 |
| spatial student | 0.28 | 0.52 | 0.64 | 0.79 | 0.85 | 0.88 |

Avoiding losses keeps improving with training; finding wins stalled after
iteration ~1000, with the raw policy keeping barely a quarter of forced wins.

**Why:** the policy gives mating moves tiny prior probability (often <1%
against 70%+ for its favourite). A root action's Gumbel score is
logit + (50 + max visits) × value_scale × Q (Q rescaled to [0, 1]); at mctx's
default value_scale 0.1 and few visits, search can move an action by only
~5 nats, so it sees the mate and still plays the favourite. The training
target, softmax(logits + the same term), barely moves toward the mate either:
a self-reinforcing blind spot.

**value_scale sweep** (`diagnostics/value_scale.py`; az3-net 1150, 32
simulations, 256 games against itself at 0.1):

| value_scale | win kept | loss avoided | score vs 0.1 |
|---|---|---|---|
| 0.1 | 0.54 | 0.86 | – |
| 0.2 | 0.66 | 0.86 | 0.56 |
| 0.3 | 0.70 | 0.86 | 0.60 |
| 0.5 | 0.72 | 0.86 | 0.54 |
| 1.0 | 0.72 | 0.86 | 0.59 |

(±0.06 on match scores.) Every scale above 0.1 is stronger; at 0.3, 32
simulations keep more forced wins than 128 did at 0.1. **Use 0.3** for play
and for the next run's targets.
