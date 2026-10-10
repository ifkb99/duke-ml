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

## az4 (AlphaZero) and mz1 (MuZero), side by side (from 2026-10-06 22:16)

Both start from the distilled spatial-head student (az3-net's trunk), 128 × 10,
and share every setting: value_scale π/10 (spec.json), playout cap
randomization (25% of self-play calls at 128 simulations give the policy
targets, the rest 32 simulations and value targets only), cosine learning-rate
decay 3e-4 → 3e-5 over 1,500 iterations, tactics benchmark every 50
iterations. One GPU each (az4 ran on two GPUs for its first 125 iterations).

- **az4**: AlphaZero, searching the real rules.
- **mz1**: MuZero (`--algo muzero`, muzero.py): searches a learned model.
  h and f start from the same student; the dynamics (3 residual blocks each
  for afterstates and draws) start fresh. Unroll 5 moves. The policy is
  trained over all actions (no legal mask), since only the root knows the
  legal moves.

az4 so far (two GPUs): minimax 4: 0.75 / 0.80 / 0.82 / 0.87 at iterations
50 / 75 / 100 / 125 (az3-net's best: 0.84); minimax 6 0.70 at iteration 100;
tactics (search 32) win kept 0.72, loss avoided 0.86. Deep searches make
self-play ~45% as fast as az3-net's (a 128-simulation call takes ~6× a
32-simulation one).

Note: with value_scale π/10 the policy targets are much sharper (entropy
~0.17 vs 0.36) and agree with the network's top move ~63% of the time (az3-net:
~80%): search values now outweigh the prior, as intended.

### mz1 → mz2: warm up with the real rules (2026-10-06 22:30)

mz1's untrained model gave noise Q values, and at value_scale π/10 those
outweigh the prior: within 10 iterations its policy matched its own search
targets 8% of the time. Stopped (runs/mz1-aborted). **mz2** = mz1 +
`--muzero-warmup 150`: the first 150 iterations self-play with the real-rules
search (h + f are an AlphaZero network) while the dynamics learn from those
games; then the model search takes over.

### mz2 after the switch: the model can't see checkmate (2026-10-07 00:45)

At iteration 150, Q and value losses started rising together (0.139 → 0.172 and
0.118 → 0.152 by 230). Value targets are bootstrapped from search values, which
now came from the model search, so this was partly noisier targets. Strength
after the switch stayed far below az4 (minimax 4: 0.13–0.36 vs ~0.85).

Tactics on 1,536 benchmark positions with the iteration-225 checkpoint:

| 32-simulation search | wins kept | losses avoided |
|---|---|---|
| raw policy (no search) | 0.26 | 0.70 |
| mz2's network searching the **real rules** | **0.70** | **0.82** |
| learned model, value_scale 0.1 / π/10 / 1.0 | 0.25 | 0.63 / 0.61 / 0.62 |

The network is nearly as good as az4's (0.72 / 0.86); the model makes search
*worse* than no search, whatever the value scale. With 32 simulations over 16
candidates, a root move's first value mostly decides it: the Q head's guess in
the model, the value of the real next position in the rules search. Picking by
Q alone: 0.26 / 0.62; by the real next position: 0.63 / 0.77. Games end by
checkmate (the opponent has no legal move; a move that leaves your own Duke
capturable is illegal), and on checkmating moves the Q head's median is 0.42
(target 1); it prefers the mate to the other moves 25% of the time. The model
only trains on moves that were played, so it never sees what the others lead
to, and once its own search missed mates, self-play showed it fewer. By
iteration 250 the raw policy got worse too (loss avoided 0.77 → 0.64): the
model search's targets were teaching it bad moves.

### mz3: counterfactual moves labelled by the rules (from 2026-10-07 01:10)

mz2's iteration-250 checkpoint + `--counterfactual-moves 8` (muzero.py): every
sample also draws 8 more legal moves the way the search draws root candidates
(policy + Gumbel noise, without replacement; ~12.5 legal moves on average, so
this covers most of them) and trains Q, reward and chance on what the rules say:
the result if the move ends the game, else the network's value of the real next
position (a test checks these equal the rules search's first-visit values).
Not pure MuZero any more: the search still runs only on the model, but training
uses the rules. 10 iterations of real-rules self-play (`--muzero-warmup 260`)
while the empty replay refills. New metrics: cf_q_loss, cf_reward_loss,
cf_chance_loss, cf_q_on_wins (Q on moves that win at once; want 1).

### az4 and mz3 at ~1050 iterations (2026-10-07 08:20)

Every 100 iterations: minimax evals (64 games), the tactics benchmark, and a
ladder game against the checkpoint 100 iterations earlier (64 games).

| | az4 | mz3 |
|---|---|---|
| vs minimax6, last checks | 0.80, 0.72, 0.72, 0.80 (800–1100) | 0.54, 0.33, 0.45 (800–1000) |
| vs minimax4 | ~0.90 | ~0.60 |
| tactics, policy: win kept / loss avoided | 0.40 / 0.81 | 0.32 / 0.77 |
| tactics, search32: win kept / loss avoided | 0.75 / 0.88 | 0.35 / 0.78 |
| ladder vs 100 earlier | 0.56 0.59 0.64 0.58 0.56 0.54 0.59 | 0.57 0.45 0.62 0.48 0.60 0.51 0.47 |

az4 keeps improving: it beat its 100-earlier self in all seven ladders (very
roughly +390 Elo from 400 to 1100 chained), and beats minimax6 more often than
az3-net, the web app's model, did (0.50–0.71 at the same iterations). Search
has flattened at ~0.75 wins kept while the raw policy slowly absorbs it
(0.33 → 0.40).

mz3 recovered mz2's policy (loss avoided 0.64 → 0.77) but searching its model
still adds almost nothing: 0.35 vs 0.32 wins kept, 0.78 vs 0.77 losses avoided.
cf_q_on_wins crept 0.35 → 0.47 over 750 iterations; at that pace Q won't
recognise a mate before the learning rate bottoms out at iteration 1500.

Why so slowly: not because the picks miss mates. Real positions have few legal
moves (tactics positions: mean 10.6, p99 28, max 37), so 8 Gumbel picks already
cover most of them; on random-play positions a 32-move pool finds a mate in 4.2%
of positions against 2.6% for the 8 picks. Mates are rare because positions
with a mate in one are rare: 0.4% of picks win, so they are ~0.4% of the Q loss.

### mz4: weight winning counterfactual moves (from 2026-10-07 ~08:35)

mz3's iteration-1050 checkpoint + `--counterfactual-win-weight 20`: each pick
that wins at once counts 20 times in the cf Q and reward losses (~8% of the Q
loss instead of 0.4%). New metric cf_q_on_others (mean Q on the non-winning
picks) to catch the Q head calling too many moves wins. Hard stop: if search32
still doesn't clearly beat the raw policy on tactics ~100 iterations in, retire
MuZero and give GPU1 to AlphaZero.

mz4 result at the hard stop (iteration 1150): the weight helped but not enough.

| tactics: win kept / loss avoided | policy | search32 | search gain |
|---|---|---|---|
| mz3, 1050 | 0.340 / 0.779 | 0.354 / 0.798 | +0.01 / +0.02 |
| mz4, 1100 | 0.414 / 0.768 | 0.454 / 0.802 | +0.04 / +0.03 |
| mz4, 1150 | 0.413 / 0.775 | 0.462 / 0.801 | +0.05 / +0.03 |
| az4, 1150 | 0.408 / 0.807 | 0.753 / 0.888 | +0.35 / +0.08 |

cf_q_on_wins jumped 0.47 → 0.69 within 10 iterations, then settled at ~0.63–0.65
while cf_q_on_others stayed at −0.06: even with mates counted 20 times, Q
hedges rather than separating mates from other moves. That looks like a limit
on what the afterstate network can tell from its latent, not on training
signal. minimax4 dipped after the restart (0.71 → 0.44) and came back to 0.62
by 1150 (empty replay after a restart); ladder 1100 vs 1050: 0.49.

### az4 on two GPUs, through the learning-rate floor (2026-10-07 13:00)

From iteration 1250 az4 runs alone on both GPUs (learner GPU0, actor GPU1):
~2.9× the self-play rate of one GPU, ~1.5× the iterations. The learning rate
reached its floor (3e-5) at 1500.

| iter | minimax6 | tactics policy (win kept / loss avoided) | search32 | ladder |
|---|---|---|---|---|
| 1100 | 0.80 | 0.40 / 0.81 | 0.75 / 0.88 | 0.59 vs 1000 |
| 1300 | 0.88 | 0.40 / 0.82 | 0.75 / 0.89 | 0.48 vs 1250 |
| 1400 | 0.97 | 0.41 / 0.82 | 0.75 / 0.89 | 0.49 vs 1300 |
| 1500 | 0.94 | 0.42 / 0.81 | 0.76 / 0.89 | 0.55 vs 1400 |

Head-to-head, 128 games, 32 simulations each: az4:1575 vs az3-net:1150 (the
web app's model) 0.703 (W87 D6 L35), about +150 Elo. Ladders are near 0.5
since the move to two GPUs: az4 is close to what 128×10 with this search gives.

### az4 with more simulations vs minimax8 (2026-10-07 15:20)

az4:1575 against TypeScript minimax depth 8, 32 games per level (colours
alternate), Gumbel scale 0, all four levels at once on the actor GPU while
training ran (so slow: 9 min at 32 simulations, 2 h at 1,024).

| simulations | score | W-D-L | mean plies |
|---|---|---|---|
| 32 (training's budget) | 0.688 | 22-0-10 | 58 |
| 128 | 0.812 | 26-0-6 | 51 |
| 512 | 0.719 | 23-0-9 | 61 |
| 1,024 | 0.906 | 29-0-3 | 52 |

At 32 games one score has a standard error of ~0.08, so 512's dip is noise.
The trend is clear: going from 32 to 1,024 simulations takes az4 from ~0.69 to
~0.91 against minimax8. Phone budgets (tools/latency): 128 × 10 does ~120
simulations in 3 s on the user's phone's CPU, ~840–1,400 with WebGPU; older
phones without WebGPU get fewer. az4 as fp16 is 6.2 MB (spatial head) against
az3-net's 15.4 MB in the web app now.

## az5: a phone-sized network (from 2026-10-07 evening)

Goal: play well on older, CPU-only phones too, and download less. The web app
now thinks for 3 s per decision (TimeBudget), so a faster network gets more
simulations, and az4 vs minimax8 showed simulations are worth a lot
(32 → 128: 0.69 → 0.81). Candidate sizes, spatial head (export_onnx --random;
phone CPU time from the user's phone, ms ≈ MFLOP / 9):

| network | parameters | fp16 file | MFLOP | phone CPU | simulations in 3 s |
|---|---|---|---|---|---|
| 96×6 | 1.1M | 2.3 MB | 77 | ~8.6 ms | ~2.8× az4 |
| 96×8 | 1.5M | 2.9 MB | 101 | ~11 ms | ~2.2× |
| **96×10** | 1.8M | 3.6 MB | 125 | ~14 ms | ~1.75× |
| 128×10 (az4) | 3.1M | 6.2 MB | 219 | ~24 ms | 1× |

The user chose 96×10: az4's depth, 40% narrower. On WebGPU, time per call
depends more on layer count than width, so 96×10 helps CPU phones most.

Three changes from az4:
1. **Mirror augmentation** (`--mirror 0.5`, symmetry.py). Every tile pattern
   and the setup are left-right symmetric, so each sample is flipped with
   probability ½: board columns and policy-target actions. Tests check that a
   mirrored position has exactly the mirrored legal moves and that every move
   leads to the mirrored result (and fail if the command directions aren't
   mirrored).
2. **Start by imitating az4's deeper search.** distill.py now takes
   `--simulations`: az4 (iteration 1800) plays itself at 128 simulations
   instead of 32, so the student's targets are stronger than az4's own
   training data. 1M positions, half per GPU (~186 positions/s per GPU at
   128 simulations, 74/s at 256). The earlier distillation was limited by data
   (360k positions overfit after ~6 epochs); mirroring doubles it again.
3. **Then self-play RL** from the student, with az4's settings plus mirroring.

Judge: az5 vs az4 at equal phone time (az5 at ~1.75× az4's simulations), plus
the usual minimax, tactics and ladder.

### az5 imitation, first try: 1M positions isn't enough (2026-10-07 18:30)

Teacher data took 25 min (two GPUs, ~355 positions/s each at 128
simulations). The fresh 96×10 student stopped itself at step 9,728 (~11
epochs, with mirroring) as held-out loss levelled off while training loss kept
falling, i.e. it ran out of data:

| held-out (107k positions, 1,056 games) | policy | top-1 | value |
|---|---|---|---|
| az4 (teacher's raw network) | 0.842 | 0.679 | 0.110 |
| student 96×10 | 1.033 | 0.619 | 0.124 |

Head-to-head at 32 simulations it is far behind: az5-distill1 vs az4:1800
0.133 (W17 D0 L111), roughly −330 Elo. More teacher data is the cheaper fix
while the student is this far behind: a teacher game costs about what a
self-play game does, but carries az4's 128-simulation judgement. Generating
3M more positions (runs/az5-data2, -data3) before distilling again on ~4M.
(This first student is kept in runs/az5-distill1.)

### az5 imitation, second try: 4M positions (2026-10-07 20:00)

3M more teacher positions (runs/az5-data2, -data3: 70 min on two GPUs), 4.05M
in all (~40k games). The student (fresh 96×10, `--mirror 0.5`, checks every
2,048 steps, patience 4) kept step 43,008 (~12 epochs):

| held-out (414k positions, 4,029 games) | policy | top-1 | value |
|---|---|---|---|
| az4 (teacher's raw network) | 0.844 | 0.679 | 0.110 |
| student, 1M positions (az5-distill1) | 1.033 | 0.619 | 0.124 |
| **student, 4M positions** | **0.873** | **0.672** | **0.111** |

Policy loss vs value loss, for scale (20k teacher positions): a uniform guess
over the ~10 legal moves scores 2.01 and the targets' own entropy (the floor)
is 0.17; for value, always guessing 0 scores 0.257. So policy 0.87 and value
0.11 are each ~60% of the way from guessing to the floor.

Head-to-head with az4:1800:

| az5 student | az4 | games | score |
|---|---|---|---|
| 32 simulations | 32 | 256 | 0.396 (W96 D11 L149), ~−75 Elo |
| 56 (equal phone time) | 32 | 128 | **0.574** (W71 D5 L52) |
| 210 (equal phone time, ~3 s on the user's phone CPU) | 120 | 128 | **0.578** (W72 D4 L52) |

Before any RL, the 3.6 MB student already beats az4 at equal phone time,
~+50 Elo. RL started at 20:10 on both GPUs:
`run.sh az5 --channels 96 --blocks 10 --policy-head spatial --workers 15
--full-simulations 128 --full-fraction 0.25 --full-chunks 4 --lr-decay-start 0
--lr-decay-iterations 1500 --lr-min 3e-5 --tactics-every 50 --mirror 0.5`.

## big1: a 256×20 teacher (from 2026-10-07 21:00)

The az5 student matched az4 on held-out games, so the student's ceiling is
its teacher. The user chose to try a much bigger teacher, 256×20, to distil
into the phone network later. Both run at once: az5 RL moved to GPU1 alone
(from iteration 50), big1 on GPU0.

Speed on one GPU, random weights (scratchpad big_speed.py, 872 games):

| | 128×10 | 256×20 | ratio |
|---|---|---|---|
| parameters | 3.1M | 23.9M | 7.7× |
| self-play, 32 simulations | 2,375 positions/s | 712 | 3.3× slower |
| self-play, 128 simulations | 351 | 138 | 2.5× slower |
| SGD step, batch 1,024 | 25 ms | 125 ms | 5× slower |

Self-play slows less than the arithmetic (8×) because the rules and the tree
bookkeeping cost the same at any size. With training's mix (25% at 128
simulations) big1 makes ~350 positions/s on one GPU vs ~970 for 128×10; az4's
whole run was ~28M positions, ~1 day at that rate.

Start: imitation of the same 4.05M az4@128 positions (fresh init, `--mirror
0.5`), then self-play RL on GPU0 with az4's settings and `--mirror 0.5`.

Imitation stopped early at step 24,576 (of 100k; checks every 2,048, patience
4), 70 minutes. On held-out games: policy 0.865 (top-1 0.675), value 0.110,
vs az5's student 0.873 / 0.672 / 0.111 and the teacher 0.844 / 0.679 / 0.110.
So the extra capacity barely shows on held-out loss: 4M positions limit both
students. In play it shows more: big1's student scored **0.523** vs
az4:1800 at 32 simulations each (128 games, W64 D6 L58), where az5's scored
0.396. RL started 22:14 at 352 moves/s, as predicted.

### Check-in 2026-10-08 10:25 (az5 at 1565, big1 at 450)

Against az4:1800, 128 games each:

| | az5:1400 | big1:400 | at the start of RL |
|---|---|---|---|
| equal simulations (32 vs 32) | **0.578** (W72 D4 L52) | 0.523 (W65 D4 L59) | az5 0.396, big1 0.523 |
| equal phone time (210 vs 120) | **0.668** (W85 D1 L42) | | az5 0.578 |

az5, the 3.6 MB phone network, now beats az4 (6.2 MB) even at equal
simulations, ~+55 Elo, and ~+120 Elo at equal phone time. Its learning rate
reached the floor (3e-5) at 1500.

big1 has not moved in 400 iterations (~12 h on one GPU): the same 0.523 as
its imitation start, and its own ladder agrees (300 vs 200 0.45, 400 vs 300
0.48). It isn't overfitting: its loss on fresh self-play matches its training
loss (policy ~0.87, value ~0.108). Part of az5's early gain was catching up to
its teacher, which big1 never had to do; az4 itself gained only ~+33 Elo from
1400 to 1800.

az5 stopped at 1610 (11:00, 2026-10-08); its learning rate had reached the
floor at 1500. az5:1600 vs az5:1400: 0.521 over 256 games, so 1600 is the
one exported: **runs/web-models/az5-1600**, now the web app's model
(fp16 3.6 MB, 125 MFLOP per evaluation, down from az4's 6.2 MB and 219).
big1 resumed from iteration 475 on both GPUs; check it against az4:1800
again around iteration 800, and retire it if it's still flat.

Check 2026-10-08 13:30, big1:575 at 32 simulations, 128 games each:
vs az4:1800 **0.590** (W73 D5 L50; it was 0.523 at 400), vs az5:1600 0.402
(W49 D5 L74). Moving again, but not yet past az5, so not yet a useful
teacher for it. Its own ladder read 0.30 for 500 vs 475 right after the
restart: the replay buffer isn't saved, so for its first iterations after a
restart it trains on very few games. On both GPUs it does ~44 iterations an
hour.

Check 2026-10-08 17:20, big1:700 at 32 simulations: vs az5:1600 **0.465**
(W113 D12 L131, 256 games; 0.402 at 575), vs az4:1800 0.559 (W69 D5 L54,
128 games). Its raw policy is now the best we have on the tactics set: wins
kept 0.573 (az5:1500 0.528, its own start 0.42), losses avoided 0.847. With
search the tactics numbers are level with az5. Ladder 600 vs 500 0.66, 700 vs
600 0.49. Pace ~36 iterations an hour; the learning rate is 1.8e-4, reaching
the 3e-5 floor at 1500.

Check 2026-10-08 22:00, big1:900 at 32 simulations: vs az5:1600 **0.508**
(W127 D6 L123, 256 games; ±0.03, so level), vs az4:1800 **0.652** (W82 D3
L43, 128 games; about +110 Elo, where az5:1400 scored 0.578), vs big1:800
0.516 (128 games). The trainer's own checks: minimax6 0.92 (0.84 at 700),
ladders 800 vs 700 0.54 and 900 vs 800 0.45 (64 games each), raw-policy
tactics 0.554 / 0.589 / 0.516 at 800 / 850 / 900 (noisy), search32 flat at
~0.75. So big1 has caught az5 but is not clearly past it; still gaining
~+20 Elo per 100 iterations against az5, with the learning rate at 1.2e-4
and 600 iterations of decay left.

Check 2026-10-09, big1:1525 (learning rate at the 3e-5 floor since 1500),
32 simulations each unless noted:

| big1:1525 vs | games | score | W D L |
|---|---|---|---|
| az5:1600 | 256 | **0.666** | 166 9 81 |
| az5:1600, 128 simulations each | 128 | **0.691** | 88 1 39 |
| az4:1800 | 128 | 0.707 | 88 5 35 |
| big1:900 | 256 | 0.703 | 175 10 71 |
| big1:1200 | 256 | 0.482 | 115 17 124 |

About +120 Elo over az5 (+140 at 128 simulations), so it is now a real
teacher for it. All of the gain since 900 came by 1200: 1525 vs 1200 is
level, and the trainer's 64-game ladders read 0.43–0.49 at every step
after 1100. Raw-policy tactics still crept up (wins kept 0.629 at 1500,
the best of any network; 0.52–0.59 around 900), and policy loss fell
0.84 → 0.76, but in play it has plateaued.

## Smaller web download: block-quantized weights (2026-10-08)

The user asked about LittleBit (arXiv 2506.13771, 0.1 bits per weight for
LLMs via binarized low-rank factors). It doesn't fit here: it relies on huge
redundant matrices (our largest layer has 83k weights), and its speedup comes
from memory-bound LLM decoding with custom binary kernels, while our phone
inference is compute-bound and onnxruntime-web would expand the weights
anyway. The simpler idea from the same family does fit: store the weights at
fewer bits and expand them to fp16 when the page loads. The network then runs
exactly as the fp16 export does, at the same speed; only the download shrinks.

quantize.py: GGML-style blocks of 32 weights sharing an fp16 scale. az5:1600 on
the 2,100 benchmark positions, float32, against the unquantized network
(scratchpad q_variants.py; the kept numbers are in runs/quant/results.jsonl):

| weights | size | top move agrees | policy KL | value error, mean / max |
|---|---|---|---|---|
| 8 bits | 1.89 MB | 0.996 | 0.00004 | 0.004 / 0.02 |
| 6 bits | 1.45 | 0.983 | 0.0012 | 0.025 / 0.08 |
| 5 bits | 1.22 | 0.967 | 0.0048 | 0.029 / 0.13 |
| 4 bits | 1.00 | 0.932 | 0.0168 | 0.100 / 0.42 |
| 4 bits, MSE-fitted scales | 1.00 | 0.927 | 0.0166 | 0.105 / 0.41 |
| 4 bits, blocks of 16 | 1.11 | 0.955 | 0.0091 | 0.068 / 0.29 |
| **4 bits in the residual blocks, 8 elsewhere** | **1.06** | 0.959 | 0.0061 | **0.015** / 0.10 |

Nearly all of 4-bit's value error comes from the stem and heads, ~7% of the
weights; keeping those at 8 bits costs 6% of the size. Head to head at 32
simulations against the unquantized az5:1600: **4-bit mixed 0.498** (W235
D40 L237, 512 games), **8-bit 0.486** (W119 D11 L126, 256 games): no
measurable loss.

The web app now loads `duke-zero.q4.onnx` (44 KB: graph, biases, norms) and
`duke-zero.q4.bin` (1.06 MB, ~0.9 MB gzipped; the Docker build now
precompresses .onnx and .bin), expanded in packages/ai/src/zero/quant.ts into
onnxruntime-web's externalData. That's **1.1 MB, down from 3.6 MB** (fp16
az5) and 6.2 MB (az4). TypeScript rebuilds the fp16 weights bit for bit (test
`test_quantized_export_in_typescript`). In headless Chromium (WebAssembly, 8
threads) the built app fetched and unpacked the network in 24 ms and searched
as before. WebGPU wasn't tested here (no GPU in headless Chromium); the
external weights are loaded before the backend is chosen, so it should behave
the same.

## az6: big1 distilled into the phone network, 96×10 vs 96×12 (from 2026-10-09)

big1 stopped at 1590 (13:35); **big1:1525** (runs/big1/ckpt_001525.pkl) is
the teacher. Data: big1 playing itself at 128 simulations, 2M positions per
GPU into runs/az6-data{0,1} (seeds 10, 11). It runs at ~170-190 positions/s
per GPU (az4 as teacher: ~350; big1 costs ~8× az4 per evaluation but the
search overhead is shared), so ~3 h for 4M.

Two students on the same data, one per GPU, both starting from az5:1600
(`distill.py --init runs/az5/ckpt_001600.pkl`, new): **az6** is 96×10, and
**az6-12** is 96×12, whose two extra blocks start as the identity (so it
begins exactly as az5). On a phone 96×12 costs ~19% more per evaluation
(~150 vs 125 MFLOP) and ~190 KB more download at 4 bits (1.25 vs 1.06 MB).
At ~75 Elo per doubling of simulations (az5 vs az4 at equal simulations vs
equal phone time), it must be ~20 Elo stronger per evaluation to break even
on a phone. Then a short RL phase on whichever is ahead, and head-to-heads
against az5:1600 at equal simulations and at equal phone time.

Data: 4.0M positions in 3 h 40 min (~150 positions/s per GPU once warm;
GPU0 needs MEM_FRACTION 0.85 as the desktop holds 1.75 GB of it). Both
students, lr 1e-4 (az5 was already trained, so fine-tuning rather than
3e-4), mirror 0.5, early stopping; each took ~11 min:

| held-out (400k positions) | policy | top-1 | value | best step |
|---|---|---|---|---|
| big1:1525 (teacher) | 0.750 | 0.716 | 0.107 | |
| az6, 96×10 from az5:1600 | 0.817 | 0.693 | 0.109 | 24,576 |
| az6-12, 96×12 from az5:1600 | 0.817 | 0.693 | 0.110 | 22,528 |

Both reached big1's training loss (0.75) but stopped improving on held-out
games after ~6 epochs: more data, not more depth, is the limit here.

Head-to-heads (32 simulations unless noted):

| match | games | score | W D L |
|---|---|---|---|
| az6 vs az5:1600 | 256 | **0.578** | 142 12 102 |
| az6-12 vs az5:1600 | 256 | 0.562 | 137 14 105 |
| az6-12 vs az6 | 256 | 0.463 | 113 11 132 |
| az6-12@175 vs az6@210 (equal phone time) | 128 | 0.445 | 56 2 70 |

**96×12 doesn't pay:** level with 96×10 per evaluation, so behind at equal
phone time. az6 (96×10) gets about half of big1's +120 Elo over az5 from
imitation alone (+55). Next: a short RL phase on az6 (both GPUs, lr 1e-4
decaying to 3e-5 over 500 iterations), checking it against az6's imitation
start and az5:1600.

### az6 RL hurts; the imitation network is the one (2026-10-09 evening)

RL from az6's imitation start (both GPUs, lr 1e-4 → 3e-5 over 500
iterations, ~107 iterations/h) made it worse, 256 games each at 32
simulations: az6:200 vs its start **0.410**, az6:275 vs its start **0.426**,
az6:275 vs az5:1600 0.492. Stopped at 285 (checkpoints in runs/az6-rl). Its
own targets (mostly 32-simulation searches by a 96×10 network) are weaker
than big1's 128-simulation ones, so self-play pulls it back to the level a
96×10 network's self-play sustains, which is where az5 ended. For a small
network the teacher's ceiling is higher than self-play's.

The first az6 vs az5 match (0.578, 256 games) was on the lucky side. Bigger
matches, az6 = runs/az6/imitation_start.pkl:

| match | games | score | W D L |
|---|---|---|---|
| az6 vs az5:1600, 32 simulations (seed 5) | 1024 | **0.558** | 551 41 432 |
| az6 vs az5:1600, 128 simulations each | 512 | **0.546** | 274 11 227 |
| az6-q4 vs az5-q4 (what phones run), 32 simulations | 1024 | **0.545** | 540 36 448 |
| az6-q4 vs az5-q4, first try | 512 | 0.509 | 250 21 241 |
| az6-q4 vs az6 | 512 | 0.488 | 242 16 254 |

So az6 is about +30–40 Elo over az5, at the same size and speed; 4 bits costs
it about as little as it cost az5 (top-1 agrees 0.947 vs az5's 0.959, value
MAE 0.020 vs 0.015). Exported to runs/web-models/az6-0 (q4 1.06 MB).

## az7: more teacher data (from 2026-10-09 21:45)

The az6 students stopped improving on held-out games after ~6 epochs of 4M
positions, so: 4M more big1:1525 positions at 128 simulations
(runs/az6-data{2,3}, seeds 12, 13, ~3.7 h), then two students on all 8M, both
96×10 from az5:1600 at lr 1e-4, one per GPU:

- **az7**: early stopping as az6 (the data effect alone);
- **az7-drop**: on a plateau, back to the best step with the lr ×0.3, twice,
  before stopping (`--lr-drops 2`, new; Keras' ReduceLROnPlateau plus the
  restore).

Each against az6 and az5:1600 over 1024 games.

### Results (2026-10-10 01:20–02:10)

Both trained on 8M positions (~20 min for az7, ~33 min for az7-drop on one
GPU each). Held-out games here are 5% of the 8M, so not comparable to az6's.

| student | stops at | held-out loss | policy (top-1) | value |
|---|---|---|---|---|
| az7 | step 49152 | 0.902 | 0.793 (0.701) | 0.109 |
| az7-drop | step 88064 (rewinds to 47104 at 3e-5, 71680 at 9e-6) | **0.892** | 0.784 (0.705) | 0.108 |
| big1:1525 (the teacher) | | | 0.745 (0.718) | 0.108 |

Matches, 1024 games, 32 simulations (±0.015):

| match | score | W D L |
|---|---|---|
| az7 vs az6 | 0.527 | 515 50 459 |
| **az7-drop vs az6** | **0.554** | 544 46 434 |
| az7-drop vs az7 | 0.517 | 503 53 468 |
| az7-drop vs az5:1600 | 0.627 | 625 34 365 |
| az7-drop-q4 vs az6-q4 (what phones run) | 0.582 | 574 43 407 |
| az7-drop-q4 vs az5-q4 | 0.582 | 569 53 402 |
| az7-drop-q4 vs az7-drop | 0.487 | 473 52 499 |
| az6-q4 vs az6 | 0.474 | 460 51 513 |

- **The extra data alone is worth ~+20 Elo** (az7 vs az6, ~2 SE), the lr drops
  another ~+10 (az7-drop vs az7, ~1 SE); together ~+38 over az6 (~3.5 SE) and
  ~+90 over az5, consistent with az6 being ~+38 over az5.
- The drops kept improving the held-out loss after plain early stopping would
  have quit (0.902 → 0.892): the plateau was the step size, not only the data.
- 4-bit drifts a little more than az6's (benchmark value MAE 0.037 vs 0.020,
  top-1 agreement 0.938 vs 0.947) but costs about the same in play (~−10 to −18
  Elo each), and az7-drop-q4 beats az6-q4 0.582.
- Exported to runs/web-models/az7-drop (q4 1.06 MB). The ONNX check must run
  JAX on the CPU (`JAX_PLATFORMS=cpu`): on the GPU, TF32 matmuls put the
  reference 0.02 off and the 1e-3 assert fails.

## Phone: az7-drop in the web app, threads and WebGPU (2026-10-10)

az7-drop is the web app's model (q4, 1.06 MB). On the user's phone (Android
10, Chrome 155) "vs Neural Net" hung on creating the ORT session, on WebGPU
and wasm alike: ORT's threads are workers inside the AI worker, and Android
Chrome never starts them. One thread fixes it and costs nothing: the search
sends a few positions per call, too few for threads. Phone reports are in
tools/latency/results/2026-10-10T*-neural-*.json (`?report` now also sends
the worker's log; `?backend=wasm|webgpu`, `?threads=N` override).

Batch sizes of the search (40 positions, flat network): at 400 simulations
42% of evaluations are in batches of 1–2, 22% in 3–4, 36% in 5–16. So the
worker now opens both backends, times them (ms per call on the phone):

| positions | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| wasm, 1 thread | 13 | 23 | 44 | 85 | 169 |
| WebGPU | 32 | 29 | 30 | 26 | 43 |

and sends batches of 3+ to WebGPU (padded to a power of two): ~5.4 ms per
simulation in steady play vs ~6.0 on the CPU alone, at a one-off ~2 s
calibration on the first move.

## Next steps (2026-10-10)

Training:
- **16M big1 positions** (another 8M, ~7 h on both GPUs), then the az7-drop
  recipe. 4M → 8M gave ~+20 Elo, the lr drops ~+10 more; expect a bit less
  per doubling.
- **Deeper teacher search** for the targets (256 simulations instead of 128):
  slower data, stronger targets; test on a 4M slice against az7-drop.
- **Depthwise-separable (MobileNet-style) blocks** for older phones, distilled
  from big1: fewer FLOPs per evaluation, judged at equal phone time.
- **A stronger teacher**: big1 plateaued from iteration ~1200 at its lr floor;
  a learning-rate restart or more simulations in its self-play may move it.
- training/zero/evaluator.py has never been run.

Web app:
- **Draw moves take two full budgets** (~6 s on the phone): deciding to draw
  and placing the tile are separate searches, each with 3 s. Split the budget.
- **TimeBudget undershoots**: steady moves use 2.0–2.9 s of 3 s; the speed
  estimate lags as positions get cheaper.
- **Cache the calibration** per device (localStorage), skipping the ~2 s on
  later visits.
- **Bigger batches** would let WebGPU do more of the work, but the search
  batches only where the result can't change (parity with the JAX search);
  anything else needs its own strength test.
- **Production**: COOP/COEP headers are no longer needed (one thread); still
  serve the 28 MB jsep wasm gzipped (6.7 MB); difficulty levels; test on iOS
  Safari and an older Android phone.
