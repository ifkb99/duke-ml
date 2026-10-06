# Notes and next experiments

Ideas and context collected while setting up the first run (`az1`, started
2026-10-05). How to operate training is in [RUNNING.md](RUNNING.md).

## Status of run `az1` (as of 2026-10-05, 22:30)

Started 21:41. Restarted twice from checkpoints for speed fixes; stopped at
~iteration 80 for the BIOS update. Resume with `training/zero/run.sh az1`.

| point | random | minimax 2 | minimax 4 |
|---|---|---|---|
| after imitation warm-start | 0.92 | 0.47 | 0.06 |
| iteration 25 | 0.95 | 0.56 | 0.12 |
| iteration 50 | 0.97 | 0.48 | 0.03 |
| iteration 75 | 0.92 | 0.53 | 0.06 |
| iteration 100 (after BIOS restart) | 0.95 | 0.52 | 0.06 |
| iteration 125 | 0.97 | 0.36 | 0.09 |
| iteration 150 | 0.97 | 0.41 | 0.08 |

**More search doesn't help (iteration 100, 128 games vs minimax 4):** 32
simulations 0.09, 200 simulations 0.13, the same within noise. So the network's
judgment is the bottleneck, not the search budget. Policy loss has been flat at
~1.79 since the start of self-play while value loss falls (0.47 → 0.37).
Next checks if it stays flat: (1) is the value head blind to 3-4-move tactics
(compare it with minimax on positions where minimax wins material)? (2) are
the Gumbel targets at 32 simulations / 16 candidates too close to the prior?
(3) learning rate and replay staleness.

Flat so far, but within noise (see "Evaluation noise" below), and only about
45k self-play games in. **What to watch:** if minimax 4 hasn't moved by
roughly iteration 600 (about 3 hours of self-play), dig into why: check that the
search targets actually beat the raw policy, the learning rate, and replay staleness.

Speed: ~3,900 searched moves/s (≈14M moves, ~300k games per hour) after
running each GPU's self-play in its own thread. GPU 1 takes ~13.6 s per
iteration, GPU 0 ~11-12 s (self-play + SGD); there's room to move ~10% more
games onto GPU 0.

## Draws and "the jackpot ticking down"

About 6-10% of self-play games hit the 200-turn limit (a rule we added; the
real game has none). Diagnosis at iteration 75 (256 games, 15 timeouts): most
timeouts are **near-empty boards** (0-2 tiles per side besides the Dukes) with a
small material gap (median 2 tiles); only 27% had a lead of 3+ tiles. So they're
mostly genuinely drawish endgames, with a little dithering-while-winning.

- **A flat per-turn penalty doesn't do much here.** The setup is zero-sum
  (one value, sign flipped every turn), so a cost both players pay mostly
  cancels out.
- **A discount does:** a win n moves away is worth γⁿ (e.g. γ = 0.997: in 10
  moves 0.97, in 100 moves 0.74). The value head learns that slope, so moves
  that bring a win closer visibly raise the value and the search follows it.
  The winner hurries, the loser stalls, which is symmetric and fine. Keep γ close
  to 1, or it starts preferring risky quick attacks over sure slow wins.
  Implementation: the turn-passing discount −1 becomes −γ in `search.py`
  (`chance_out` discount), `selfplay.step_record` and the arena records.
- The network already **sees the clock**: the turn number / 200 is one of its
  inputs, so it can learn that a won position at turn 195 is really a draw.
- Verdict: worth an A/B run (γ = 0.997 vs `az1`), but expect only a small
  effect while timeouts are this rare.

## Evaluation noise

Each evaluation is 64 games per opponent, so a score near 0.5 has a 95%
uncertainty of about ±0.12. Changes smaller than that are noise. Better options:
more games (`--eval-games 256`, slower), and/or rating checkpoints against each
other (an Elo ladder of `ckpt_*.pkl`), which also measures progress after
minimax becomes too easy.

## Tuning settings: plan for "meta-learning"

The fitting technique is **Population-Based Training (PBT)**: several runs in
parallel with different settings; periodically the weaker ones copy the
stronger ones' weights and mutate their settings, so the tuning happens during
training. Not yet, because:

1. The baseline must clearly learn first, or PBT just selects whatever hides
   the problem best.
2. PBT needs a reliable score, and ours is ±0.12.
3. Two GPUs split across a population of 4 gives each member ~1/4 of the speed.

Order: (1) let `az1` show a trend, (2) make evaluation trustworthy, (3) a few
hand-picked A/B runs (discount γ, simulations 32 vs 64, no imitation
warm-start, network size via the sweep below), (4) PBT over the settings that
turned out to matter (most likely candidates: learning rate, simulations,
discount, sample reuse).

## BIOS update checklist

The i9-14900K's core 20 (logical CPUs 10 and 11) causes random crashes. Every
segfault logged since boot was on it, and none happened with it excluded.
Microcode before the update: **0x11d** (older than Intel's 0x129+ fixes for the
13th/14th-gen voltage/degradation problem).

**Result (2026-10-05, 23:07):** BIOS 3202 (dated 2026-05-07), microcode
**0x133**, power limits PL1 = PL2 = 253 W (Intel's "Performance" profile). The
full test suite on all cores: 3 of 3 runs with no segfaults (before the update:
3 of 3 crashed). So the instability was most likely voltage-related rather
than permanent damage. Training stays pinned off core 20 anyway, since it's
GPU-bound and two more CPU threads gain nothing. If segfaults show up on CPU
10/11 again, the core is degrading after all: consider a warranty replacement.

**Before** updating:

```bash
kill -- -$(cat runs/az1/pid)     # stop training (loses at most ~7 minutes)
```

**After** updating and booting:

1. Check the new microcode (should be 0x12F or newer):
   ```bash
   grep -m1 microcode /proc/cpuinfo
   ```
2. In the BIOS, prefer Intel's default power profile ("Intel Default Settings"
   / "Intel Baseline") over the motherboard's "unlimited" defaults.
3. Restart training (still pinned off core 20 by default):
   ```bash
   training/zero/run.sh az1
   ```
4. Test whether core 20 is still bad by running the test suite **on all
   cores**, a few times, then checking the kernel log:
   ```bash
   for i in 1 2 3; do uv run pytest training/zero/tests -q -p no:cacheprovider | tail -1; done
   journalctl -k --since "-1 hour" | grep -E "segfault|traps:"
   ```
   Before the update this crashed 3 out of 3 times, all on CPU 10/11.
   - **No crashes:** the update may have helped. You can let training use every
     core with `DUKE_CPUS=0-31 training/zero/run.sh az1`, but keep an eye on
     `journalctl` for a while.
   - **Still crashing on CPU 10/11:** the core is already degraded; microcode
     prevents further damage but can't undo it. Keep it excluded (the default)
     and consider Intel's warranty replacement (13th/14th-gen desktop CPUs got
     a 2-year warranty extension for this issue).

## Simulations: search effort per move, not training

A simulation is one step of the tree search: it adds one more position to the
look-ahead before a move is chosen. `--simulations 32` means 32 per move.

- **In self-play** it sets the quality of the training targets (the network
  learns to imitate the search). More simulations means better targets but
  fewer games per hour.
- **When playing for real** you can use far more (hundreds) with **no
  retraining**. The same network gets stronger just by thinking longer. The
  evaluations use 32, so they understate playing strength.

## Network size: how to choose

It's an empirical question. Signs to look for:

- **Too small:** losses plateau even though training keeps getting fresh games,
  and the loss on held-out recent games equals the training loss (no
  overfitting, just no more capacity).
- **Too big:** learns fine, but self-play slows enough that fewer games per
  hour costs more than the extra capacity gains. Small networks win early in a
  run; big ones win late. The best size grows with the compute budget (see
  e.g. Andy Jones, *Scaling Scaling Laws with Board Games*, 2021, on Hex, which
  also shows search at play time trading off against network size).

**Cheap sweep (planned):** once a run has a few million positions, save them as
a fixed dataset, train several sizes on that same data (e.g. 64×4, 96×6,
128×8, 192×10), and compare held-out loss against search speed. Pick the knee.
That takes hours instead of days (no new self-play). The definitive test is
full training runs at equal compute, played head-to-head.

Facts about the current network (96 channels × 6 blocks, 5.6M parameters):

- **Growing the trunk is almost free here.** Earlier benchmarks showed network
  size barely changes self-play speed (64×4 → 128×6 cost ~30%, less after the
  switch to bfloat16), because the game engine and search dominate.
- **Depth isn't a limit:** six blocks of 3×3 convolutions already see the whole
  6×6 board.
- **The policy head is the inefficient part:** 4.5M of the 5.6M parameters are
  one dense layer mapping to the 3,605 actions. Ideas, in order of effort:
  1. **Structured policy head:** score each (from-square, to-square) pair from
     those two squares' features (a bilinear or small-MLP over pairs), so
     "move A→B" shares what it learns across positions. Smaller and likely
     learns faster.
  2. **Global pooling inside blocks** (KataGo style), to help long-range
     effects such as slides, jump-slides and long strikes.
  3. **Small transformer over the 36 squares as tokens:** a natural fit for
     long-range moves, but a bigger change. Try after 1.

## The bitter lesson, applied to this project

Rich Sutton's essay *The Bitter Lesson* (2019) is often summarized as "bigger
is better", but its actual claim is: **general methods that scale with
computation, namely search and learning, eventually beat methods built on human
knowledge of the problem.** In this project:

- **Minimax vs. this AI is the lesson in miniature.** `evaluate.ts` is
  hand-written knowledge (tile values from counting move icons, a bonus for the
  centre). The new system replaces it with a learned value function plus
  search, and gets better with more compute instead of more human tuning.
- **The minimax warm-start is the one "human knowledge" shortcut.** It's a
  speed-up, like AlphaGo's start from human games, and AlphaZero showed that
  dropping it can end up stronger. An experiment worth running: a pure
  self-play run (`--imitation-games 0`) against `az1` at equal compute.
- **"Bigger is better" holds as compute grows, not at a fixed budget.** With two
  GPUs and a day, the best network balances size against games played and
  search per move (the scaling-laws point above). The lesson says to put
  effort into things that will keep paying off as compute grows, which in
  practice means more self-play, more search, and a network that grows with
  them, rather than hand-crafted features or evaluation rules.
- **What counts as "human knowledge" here:** the game rules are fine (we're
  told those). A feature plane like "my Duke is in danger", or a reward for
  material, would be hand-crafted strategy, which the lesson warns against. The
  network input is deliberately just the raw board, bags and turn.
- **This is a small game.** 6×6 with 15 tile types has bounded complexity, so
  the return on a much bigger network flattens sooner than in Go or chess.
  Measure it with the sweep above rather than assuming.

## Other ideas

- **Evaluate with more simulations** (e.g. 200) occasionally, to see the
  network's real strength rather than its 32-simulation self-play strength.
- **Add minimax depth 6** to `--eval-depths` once depth 4 is easy.
- **Train in setup choices:** setup is random now. Letting the network choose
  its Duke/Footman placement would add a learnable decision.
- **Export for the web app** (ONNX, or a small JAX→TS bridge) so the browser
  can play the trained network.
