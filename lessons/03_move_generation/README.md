# Lesson 3: Move generation as lookup tables plus boolean algebra

The TypeScript engine generates moves the obvious way: loop over your tiles,
loop over their patterns, walk slides square by square. That's great for a
CPU, but on a GPU, under `jit` and `vmap`, data-dependent loops are slow and
awkward. So `env.py` turns move generation into **fixed-shape array math**.

## Step 1: precompute everything that's pure geometry

`rules.py` builds these tables once, with NumPy, from `tiles.json` (which is exported
from the TS engine, so there's one source of truth for the rules):

| table | shape | meaning |
|---|---|---|
| `MOVE_OK[o, t, s, f, g]` | `[2, 15, 2, 36, 36]` | a tile of type `t` on side `s`, owned by `o` (0 me / 1 opponent), standing on `f`, can move or capture onto `g` if the path allows |
| `NEEDS_CLEAR[o, t, s, f, g]` | same | ...and that move needs every square between `f` and `g` to be empty (steps and slides; jumps don't) |
| `BETWEEN[f, g, x]` | `[36, 36, 36]` | square `x` lies strictly between `f` and `g` on a straight line |
| `STRIKE_OK[o, t, s, f, g]` | `[2, 15, 2, 36, 36]` | that tile can strike `g` from `f` |

Notice that `BETWEEN` doesn't mention the tile at all. *Which* squares block a
move only depends on where it starts and ends. That observation is what made
the env fast (see the comment above `BETWEEN` in `rules.py`).

## Step 2: gather the rows you need

For "me", each square `f` holds a tile type `t[f]` and side `side[f]`. One
advanced-indexing expression pulls the right `[36]` row for every square at
once:

```python
t = jnp.maximum(tile, 0)                    # [36]   (empty -> 0; masked out later)
SQ = jnp.arange(36)                         # [36]
move_ok = MOVE_OK[0, t, side, SQ]           # [36 f, 36 g]
```

How to read the shapes: the index arrays `t`, `side`, `SQ` are all `[36]`, so
they're used **together** (element `f` of each). That picks one `[36]` row per
`f`, and since a row is the last axis, the result is `[36, 36]`. Getting a feel for this
"indices broadcast together, leftover axes stay at the end" rule is most of the
battle with tensor code.

## Step 3: boolean algebra with broadcasting

```python
occ  = tile >= 0                            # [36]  occupied
mine = occ & (owner == 0)                   # [36]
opp  = occ & (owner == 1)                   # [36]

# Is any square between f and g occupied?  -> [36, 36]
occupied_between = jnp.any(BETWEEN & occ[None, None, :], axis=-1)
```

`occ[None, None, :]` has shape `[1, 1, 36]` and broadcasts against `[36, 36, 36]`:
"for every (f, g), look at all 36 squares x". `any(axis=-1)` then collapses x.

Then a legal move from `f` to `g` is a pile of ANDs, each broadcast to `[36, 36]`:

```
mine[f]                         -> mine[:, None]      (a column: depends on f only)
geometrically possible          -> move_ok            (depends on both)
path clear if it needs to be    -> ~(needs_clear & occupied_between)
not landing on my own tile      -> ~mine[None, :]     (a row: depends on g only)
```

`x[:, None]` turns `[36]` into `[36, 1]` (varies down the rows), and `x[None, :]`
into `[1, 36]` (varies across columns). Mixing them is how you say "f-thing AND
g-thing" without writing a loop.

## Exercise (`ex03_moves.py`)

`pseudo_moves(tile, owner, side) -> bool[36, 36]`: every move **or strike** the
player to move could make (from `f` to `g`), ignoring Duke safety and commands.
That's the first half of `env._pseudo_legal`. The test compares with it on
random positions.

```bash
uv run pytest lessons/03_move_generation -q
```

## Things to try afterwards

- How many `True`s does `MOVE_OK[0, t, 0].sum(axis=(1, 2))` give per tile
  type `t` (side A)? That's a rough "mobility on an empty board" score. Compare
  it with your sense of which tiles are strong.
- Commands are the same trick with an extra axis: `[36 commander, 8 from-dir, 8 to-dir]`.
  Read `_pseudo_legal` and work out why `NOT_EYE8` is there.
