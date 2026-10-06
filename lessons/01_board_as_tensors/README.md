# Lesson 1: The board as tensors

A neural network only understands arrays of numbers. This lesson covers how a
Duke position becomes those numbers, and the three tricks that make it easy
for the network to learn.

## 1. Flat squares

A 6×6 board has 36 squares. Instead of `[row, col]` pairs everywhere, we number
them `sq = row * 6 + col`:

```
 0  1  2  3  4  5      row 0
 6  7  8  9 10 11      row 1
...
30 31 32 33 34 35      row 5
```

So the board is three arrays of shape `[36]` (see `env.State`):

| array | values |
|---|---|
| `tile`  | tile type 0–14 (`rules.TILE_NAMES`), or -1 for empty |
| `owner` | 0 = the player to move ("me"), 1 = the opponent |
| `side`  | 0 = side A, 1 = side B |

Going back and forth: `row, col = divmod(sq, 6)`, and `array.reshape(6, 6)` turns
a `[36]` array into a board you can print.

## 2. The side-to-move frame (canonical frame)

The board is always stored **from the point of view of the player to move**:
"me" is owner 0, my home row is row 5, and my "forward" is toward row 0.

Why? A Footman that steps forward is the same idea whether it belongs to P1 or
P2. If the board were stored in absolute terms, the network would have to learn
every pattern twice, once for each direction. In the side-to-move frame it only
ever sees "my pieces moving up".

When the turn passes, we mirror the board top to bottom and swap owners. That
is `rules.FLIP`, a permutation of the 36 squares: `FLIP[sq]` is the square at
row `5 - row`, same column. (Columns are not mirrored: the engine only negates
the row direction when switching players.)

```
flipped_tile = tile[FLIP]       # gather: new square s takes the tile from FLIP[s]
```

This line is worth staring at: **indexing an array with an array** ("gather")
is the workhorse of this whole codebase. `x[idx]` has the shape of `idx`, and
element `i` is `x[idx[i]]`.

## 3. One-hot planes

The network gets `planes` of shape `[6, 6, 60]`, i.e. 60 "channels" per square.
Channel `owner*30 + type*2 + side` is 1.0 if that square holds that kind of tile,
and every other channel is 0.0.

Why not just feed the tile number (0–14) directly? Because numbers imply order
and distance: tile 14 is not "more" than tile 7. One-hot channels let the
network learn a separate set of weights for each kind of piece.

Why `[6, 6, 60]` and not `[60, 6, 6]`? Flax convolutions expect
"channels last" (`[batch, height, width, channels]`, called NHWC).

`jax.nn.one_hot(idx, 60)` turns an integer array `[36]` into `[36, 60]`.
Empty squares must become all zeros, and `reshape(6, 6, 60)` restores the
board layout.

## Exercise (`ex01_board.py`)

1. `flip_perspective(tile, owner, side, bags)`: hand the turn to the other player.
2. `board_planes(tile, owner, side)`: build the `[6, 6, 60]` one-hot planes.

```bash
uv run pytest lessons/01_board_as_tensors -q
```

## Things to try afterwards

- Print `planes.sum(axis=(0, 1))`: how many of each tile kind are on the board?
- In a Python shell, take a state from `training.zero.env.init(jax.random.key(0))`
  and print `state.tile.reshape(6, 6)` before and after flipping. Can you see
  the mirror?
