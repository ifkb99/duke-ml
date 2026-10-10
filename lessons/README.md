# Lessons: how the Duke AI works, one piece at a time

These lessons walk through the real training code in `training/zero/`, one
idea at a time. Each one has:

- a **README** explaining the concept, the array shapes involved, and where it
  lives in the real code;
- an **exercise** file (`exNN_*.py`) with a few functions for you to fill in;
- **tests** that tell you when you've got it right (they compare against the
  real implementation and against hand-worked examples).

Nothing in the training pipeline depends on these. Do them in any order, at
any pace, or not at all. Training keeps running either way.

```bash
uv run pytest lessons/01_board_as_tensors -q        # check one lesson
uv run pytest lessons -q                            # check everything
LESSONS_USE_SOLUTIONS=1 uv run pytest lessons -q    # run against the reference solutions
```

Stuck? The reference answers live in `lessons/solutions/`. Reading them after
an honest attempt is how most people learn this stuff, so don't feel bad about it.

## The map

```
            ┌────────────────────────── self-play (GPU 1) ───────────────────────────┐
            │                                                                        │
  env.init ─┼─► State ──observe──► planes [B,6,6,60] ──► network ──► policy logits [B,3605]
            │    ▲   (lesson 1)    features [B,47]      (lesson 5)   value [B]         │
            │    │                                                     │               │
            │    │                       tree search (lesson 6) ◄──────┘               │
            │    │                       uses env.step_decision / step_chance          │
            │    │                       (move generation: lesson 3)                   │
            │    └──── env.step(action) ◄── improved policy → action                   │
            │                                                                        │
            │  all of it for 1024 games at once with vmap / scan (lesson 2)          │
            └──────────────────────────────┬─────────────────────────────────────────┘
                                           │ finished games
                                           ▼
                        value targets z_t (lesson 4) ──► replay buffer
                                           │
                                           ▼
                    learner (GPU 0): loss = policy CE + value MSE (lesson 5)
                                     gradient step ──► new weights ──► back to self-play
```

| Lesson | Idea | Real code |
|---|---|---|
| 1 | The board as tensors: flat squares, the side-to-move frame, one-hot planes | `env.py` (`observe_arrays`, `_flip_perspective`), `rules.py` (`FLIP`) |
| 2 | `vmap` and `scan`: write it for one game, run it for thousands | `selfplay.py`, everywhere `jax.vmap` appears |
| 3 | Move generation as lookup tables plus boolean algebra | `rules.py` (tables), `env.py` (`_pseudo_legal`) |
| 4 | Value targets: whose win is it, and the -1 for passing the turn | `replay.py` (`_game_samples`) |
| 5 | The loss: masked softmax, cross-entropy against the search, MSE | `train.py` (`make_train_step`) |
| 6 | Search: selection, backup, chance nodes | `search.py` |

## Shape notation

Shapes are written `[B, 6, 6, 60]`: the batch of games first, then board
rows and columns, then channels. Squares are often flattened: `sq = row * 6 + col`,
so a board-sized array is `[36]`. Annotate shapes in comments as you go.
It's the single best habit for keeping "matrix alignment" from biting you.
