# Lesson 4: Value targets, and whose win is it?

The value head answers one question: **from the point of view of the player to
move, how will this game end?** +1 means I win, −1 means I lose, 0 means a draw
(the 200-turn limit). To train it we need the right answer for every position
of every finished game. That answer is the **value target** `z`.

## The bookkeeping

Self-play records, for each step `t` of a game:

- `reward[t]`: what the player who just acted got from this step. It's +1 if their
  move captured the Duke or left the opponent with no legal move, and 0 otherwise.
- `discount[t]`: how to translate the *next* position's value into this
  player's point of view:
  - **−1** when the turn passed to the opponent. Their +1 is my −1.
  - **+1** when the same player moves again. That only happens after choosing to draw:
    I draw, then *I* place the tile.
  - **0** when the game ended. Nothing comes after.

Then, walking **backwards** from the end:

```
z[t] = reward[t] + discount[t] * z[t+1]          (with z[T] = 0 past the end)
```

This is the same recursion as "return" in RL textbooks. The usual discount
factor γ (like 0.99) is replaced by ±1, and the −1 is the two-player-zero-sum twist.

### Worked example

P1 moves (t=0), P2 draws (t=1), P2 places (t=2), P1 captures the Duke (t=3).

| t | mover | reward | discount | z |
|---|---|---|---|---|
| 3 | P1 | +1 | 0 | +1 + 0·(…) = **+1** |
| 2 | P2 | 0 | −1 | 0 − 1·(+1) = **−1** |
| 1 | P2 | 0 | +1 | 0 + 1·(−1) = **−1** |
| 0 | P1 | 0 | −1 | 0 − 1·(−1) = **+1** |

P1's positions get +1 and P2's get −1. Correct, since P1 won.

In the real code this lives in `replay._game_samples` (a NumPy loop), and the
same backup happens inside the tree search (`search._backward`) on every simulation.

## Doing it on the GPU: `scan` in reverse

`jax.lax.scan(body, init, xs, reverse=True)` walks `xs` from the last element to
the first. That's exactly the backward recursion:

```python
def body(z_next, rd):
    reward, discount = rd
    z = reward + discount * z_next
    return z, z            # carry it to the previous step, and also output it
```

## Exercise (`ex04_values.py`)

1. `value_targets_numpy(reward, discount)`: a plain Python/NumPy loop.
2. `value_targets_jax(reward, discount)`: the same thing with `lax.scan(reverse=True)`.

```bash
uv run pytest lessons/04_value_targets -q
```

## Things to think about

- Why does a game that hits the 200-turn limit give every position z = 0?
- The search uses the same recursion, but its leaf value is the network's
  *guess* instead of a real game result. That's the whole "bootstrapping"
  idea: search makes the guesses better, and training makes future guesses
  match the improved search.
