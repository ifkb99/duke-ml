# Notes and next experiments

Ideas and context collected while setting up the first run (`az1`, started
2026-10-05). How to operate training is in [RUNNING.md](RUNNING.md).

## BIOS update checklist

The i9-14900K's core 20 (logical CPUs 10 and 11) causes random crashes. Every
segfault logged since boot was on it, and none happened with it excluded.
Microcode before the update: **0x11d** (older than Intel's 0x129+ fixes for the
13th/14th-gen voltage/degradation problem).

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
