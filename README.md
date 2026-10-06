# Duke ML
One of my favorite games to play with my brothers is "The Duke". In short: it's chess on a smaller board where you can summon random pieces from a bag

## Purpose
I made this repo for two reasons:
- I want a way to play online with my brothers (and others) from afar
- I am too lazy to calculate the optimal moves myself

This is a fantastic case for artificial intelligence!

## AI
I started with a random move AI, and then implemented minimax to familiarize myself with thinking about the game from a "computer perspective". The next step is alpha-beta pruning, and then I will train a ML algorithm to hopefully have the greatest "The Duke" bot in the world, and learn machine learning along the way

## How I Built This
I work as a fullstack dev for my day job, so for the webapp portion I used cursor/claude code to make things easier for myself and practice prompting. I would estimate ~95% of the webapp code was directly from Claude (thank you) with some small hand tweaks.

Almost all of the AI code is handmade since that is what I want to focus on learning for this project.

Have fun with the repo!

## Self-play training (AlphaZero-style, JAX)

`training/zero/` is a complete AlphaZero-style pipeline that runs on the GPU:

- **`env.py`**: the game ported to JAX (pure functions, thousands of games in
  parallel via `vmap`). It's checked move-for-move against the TypeScript engine
  by differential tests.
- **`search.py`**: tree search (Gumbel MuZero selection + Stochastic MuZero
  chance nodes, so tile draws are treated as luck, not choices).
- **`train.py`**: imitation warm-start from minimax, then self-play. One GPU
  plays, the other trains (and also plays in its spare time). It evaluates
  against random and minimax every few minutes.

```bash
uv sync && npm install && npm run build --workspace=packages/engine
uv run pytest training/zero/tests -q          # engine parity + unit tests

training/zero/run.sh az1                      # start (detached, auto-resumes after crashes)
tail -f runs/az1/train.log                    # watch
kill -- -$(cat runs/az1/pid)                  # stop
```

Runs live in `runs/<name>/`: `train.log`, `metrics.jsonl` (one JSON line per
log/eval), `latest.pkl` (resume point), and periodic `ckpt_*.pkl`.

**Learning how it works:** `lessons/` walks through the code in six short
lessons with exercises and tests. See `lessons/README.md`.

## Training a Neural Network (supervised baseline, older)

### Prerequisites

- Python 3.14+ with [uv](https://docs.astral.sh/uv/)
- Node.js 22+ with npm
- CUDA-capable GPU (for JAX)

Install the Python project in editable mode:

```bash
uv pip install -e .
```

### 1. Generate training data

Self-play games using minimax vs minimax. Each game produces ~30-80 training samples.

```bash
# 1000 games at minimax depth 4, output to a binary file
npx tsx packages/ai/src/generate.ts 1000 4 training/selfplay.bin
```

More data = better generalization. Aim for 5,000+ games (~500K samples) for decent results.

### 2. Train

```bash
cd training
uv run supervised/train.py selfplay.bin --epochs 50 --batch-size 256 --lr 1e-3
```

The training script prints per-epoch metrics and saves checkpoints to `training/checkpoints/`. It also tracks the best validation loss and saves an `epoch_best` checkpoint automatically.

Key things to watch:
- **Policy loss** should decrease steadily (initial value ~8.4 = random)
- **Value loss** should decrease from ~2.0 toward <0.5
- **Val vs train gap** — if val loss rises while train loss drops, the model is overfitting (need more data)

### 3. Evaluate against minimax or random

Play test games between the trained neural net and minimax/random:

```bash
# Net vs random (sanity check — net should win most games)
npx tsx packages/ai/src/neural-eval.ts --opponent random --games 20

# Net vs minimax depth 2
npx tsx packages/ai/src/neural-eval.ts --opponent minimax --depth 2 --games 20

# Use best checkpoint instead of epoch 50
npx tsx packages/ai/src/neural-eval.ts --opponent minimax --epoch best --games 10

# Net plays only as P1
npx tsx packages/ai/src/neural-eval.ts --net-player P1 --games 10

# 1-ply value search: the net evaluates each candidate decision with its
# value head and picks the one that gives the opponent the worst score.
# Board moves are deterministic; placements are treated as a chance node
# (expectimax over bag draws), matching how minimax handles draws — this
# prevents the player from "peeking" at the bag and hand-picking a killer
# tile. All candidate states are evaluated in a single batched call.
npx tsx packages/ai/src/neural-eval.ts --mode 1ply --opponent minimax --depth 2 --games 10
```

All commands should be run from the repository root.
