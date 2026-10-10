# Lesson 2: `vmap` and `scan`, or one game written, thousands played

The biggest "aligning matrices" pain in ML code is the **batch dimension**:
every array suddenly has an extra axis in front, and every index has to account
for it. JAX lets you mostly not think about it.

## `jax.vmap`: write it for one, run it for many

Every function in `env.py` is written for **one game**: `tile` is `[36]`, the
legal mask is `[3605]`, and so on. To run 1024 games at once we never rewrite
anything; we wrap it:

```python
step_one   = E.step                  # (State, action, key) -> State     shapes [36], [], key
step_batch = jax.vmap(E.step)        # same, but every input has a leading [B] axis
```

`vmap` ("vectorizing map") traces your function once and produces a version
where every input and output has an extra leading axis. It isn't a Python
loop: the result is one big array program the GPU runs in parallel.

`in_axes` controls which inputs are batched. `jax.vmap(f, in_axes=(0, None))(xs, y)`
maps over the first argument but shares `y` across the batch (e.g. the same
network weights for every game).

Inside a vmapped function you **cannot** use Python `if` on array values
(`if state.done:`), because there's no single value: it's a different boolean per
game. That's why `env.py` uses `jnp.where(cond, a, b)` everywhere: compute both
answers, then pick per element. It feels wasteful, but on a GPU it's usually
cheaper than branching.

## `jax.lax.scan`: loops that compile

A Python `for` loop inside `jax.jit` gets **unrolled**: 200 steps means 200
copies of the step function in the compiled program, and slow compiles. `scan`
compiles the body once and loops on the device:

```python
def body(carry, x):
    new_carry = ...
    y = ...                  # something to collect per step
    return new_carry, y

final_carry, ys = jax.lax.scan(body, init_carry, xs)   # ys stacks every y: [T, ...]
```

`selfplay.make_selfplay` is exactly this. The carry is the batch of game states,
`xs` is one random key per step, and `ys` is the per-step records, which come out
shaped `[T, B, ...]`.

## Random numbers

JAX randomness is explicit: a `key` goes in, and you never reuse one. Make new keys
with `jax.random.split(key, n)` (→ `n` fresh keys) or `jax.random.fold_in(key, i)`.
Reusing a key gives the *same* "random" numbers twice, a classic silent bug.

## Exercise (`ex02_vmap_scan.py`)

1. `material(tile, owner)`: for one game, my tile count minus the opponent's.
   Then `material_batch` must work on `[B, 36]` inputs. Use `jax.vmap`, not
   a rewrite.
2. `random_playout(key, state, n_steps)`: play `n_steps` uniformly random legal
   moves in ONE game with `lax.scan`, and return the final state and how many
   steps were actual moves (stepping a finished game is a no-op in `env.py`).

```bash
uv run pytest lessons/02_vmap_and_scan -q
```

## Things to try afterwards

- Time `jax.jit(jax.vmap(E.step))` for B = 1, 64, 1024, 8192. Where does the
  time per game stop dropping?
- `jax.make_jaxpr(material)(tile, owner)` prints the traced program. Compare
  it with the vmapped version.
