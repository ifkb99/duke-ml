# Lesson 6: Tree search, the part that does the thinking

The network alone plays by instinct: one forward pass, one move. **Monte-Carlo
Tree Search (MCTS)** lets it *think*: try moves, look at the replies, and
concentrate effort on the lines that look best. Then (lesson 5) the network
is trained to have the instincts the search arrived at.

## The loop

Each **simulation** does four things:

1. **Select**: start at the root and repeatedly pick a child, until you reach
   a move that hasn't been tried yet. The pick balances
   - *exploitation*: children with a high average value **Q**, and
   - *exploration*: children the policy likes (**prior P**) that have few visits **N**.

   The classic AlphaZero rule (**PUCT**):
   ```
   score(a) = Q(a) + c · P(a) · sqrt(N_parent) / (1 + N(a))
   ```
2. **Expand**: apply that move to get a new position (the real game rules,
   `env.step_decision` in our code).
3. **Evaluate**: ask the network for the new position's value (and priors for
   its moves). Classic MCTS used a random *rollout* to the end of the game
   instead. That's what you'll do in the exercise.
4. **Backup**: walk back up to the root, adding the value to each node's
   running average. **Flip the sign at every turn change**: a position good
   for my opponent is bad for me. (It's the same recursion as lesson 4.)

After all simulations, the **visit counts** at the root say how much the search
believed in each move. That distribution is the improved policy.

## What our search does differently (`search.py`)

- **Gumbel MuZero selection at the root** (`gumbel_muzero_root_action_selection`):
  instead of PUCT, sample a handful of candidate moves (Gumbel noise + policy),
  then run a small tournament between them ("sequential halving"). With few
  simulations (we use 32) this improves the policy far more reliably than PUCT,
  which needs hundreds of simulations to beat its own prior.
- **Chance nodes for draws.** After "draw", the next node isn't a decision:
  the bag decides. Its value is the *probability-weighted average* over tiles,
  and the search visits outcomes in proportion to their probability. That
  stops the AI from "planning" as if it could choose which tile it gets.
- **Batched**: 1024 trees grow at once, one simulation step for all of them per
  iteration. The env, the network and the tree updates are all `vmap`ped
  array programs.

| Step | In `search.py` |
|---|---|
| Select | `_make_selection` (Gumbel at the root, deterministic "most under-visited relative to the improved policy" below, probability-matching at chance nodes) |
| Expand + evaluate | `make_recurrent_fn` (steps the real env, runs the network once) |
| Backup | `_backward` (sign flips come from `discount = −1` on turn changes) |
| Improved policy | the end of `make_search`: `softmax(prior_logits + completed_q)` |

## Exercise (`ex06_mcts.py`)

**Nim**: a pile of stones, each turn take 1, 2 or 3, and whoever takes the last
stone wins. Perfect play: leave your opponent a multiple of 4.

1. `puct_scores(prior, q, visits, c)`: the formula above, vectorized over actions.
2. `mcts(stones, n_simulations, rng)`: a small MCTS with uniform priors and
   random-rollout evaluation. Return root visit counts for taking 1, 2, 3 stones.
   The tests check that with enough simulations it finds the winning move.

A `Node` class and the game helpers are provided. The interesting parts are
selection and the backup's sign flip.

## Things to try afterwards

- Lower `c` toward 0 (pure exploitation). Does the search get stuck on its
  first idea? Raise it to 10. Does it waste simulations?
- In the real code, `max_considered` (16) is how many root moves Gumbel puts in
  its tournament. What happens to strength if it's 4? (Try it in an
  evaluation match with `arena.py`.)
