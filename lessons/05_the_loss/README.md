# Lesson 5: The loss, or what "learning from the search" means

The network outputs two things per position:

- `logits` `[B, 3605]`: one score per action. Softmax turns them into
  probabilities, the network's **policy** (its guess at good moves).
- `value` `[B]`: a number in [−1, 1] (thanks to `tanh`), its guess at the outcome.

Training nudges the weights so that:

1. **the policy matches what the tree search found** (the search's *improved
   policy*, stored in the replay buffer). The network learns to make, at a
   glance, the moves the search had to think about.
2. **the value matches how the game actually ended** (the `z` from lesson 4).

That's AlphaZero in one sentence: *search makes better decisions than the raw
network; train the network to imitate the search; a better network makes the
search better still.*

## Policy loss: cross-entropy with a soft target

```
loss_policy = − Σ_a  target[a] · log softmax(logits)[a]
```

If `target` is one-hot (imitation of minimax), this is ordinary
classification loss. With the search's soft target (e.g. 70% move A, 25% move B,
5% move C), it teaches the whole distribution, not just the top move.

Three practical details:

**1. Mask illegal moves before the softmax.** Only legal actions should get
probability, so set illegal logits to a huge negative number first:
`jnp.where(legal, logits, -1e9)`. Why −1e9 and not −∞? Because `0 · (−∞) = NaN`,
and the target is 0 on exactly those actions. A NaN poisons the whole batch.

**2. Use `log_softmax`, not `log(softmax(x))`.** Softmax of large logits
overflows, and log of a tiny probability underflows to −∞. `jax.nn.log_softmax`
computes it stably (it subtracts the max first, the "log-sum-exp trick").

**3. Sparse targets.** The replay buffer stores only the top 32 entries of
each target (`pol_idx` `[B, 32]`, `pol_p` `[B, 32]`) to save memory. Turn them back
into a dense `[B, 3605]` array with a scatter:

```python
dense = jnp.zeros((B, 3605)).at[jnp.arange(B)[:, None], pol_idx].add(pol_p)
```

`jnp.arange(B)[:, None]` is `[B, 1]` and `pol_idx` is `[B, 32]`: they broadcast
together, so entry `(i, k)` writes to row `i`, column `pol_idx[i, k]`. It's the
same broadcasting rule as lesson 3, just used for writing instead of reading.

## Value loss: mean squared error

```
loss_value = mean( (value − z)² )
```

## Which samples count?

Some samples have no policy target: draw *afterstates* ("I chose to draw, the
tile isn't revealed yet") only teach the value, as do random moves in the
imitation data. `has_policy` `[B]` says which ones do; the policy loss is averaged
over those only.

## Exercise (`ex05_loss.py`)

1. `dense_target(pol_idx, pol_p, n_actions)`: the scatter above.
2. `policy_loss(logits, legal, target, has_policy)`: masked cross-entropy, averaged
   over samples with `has_policy`.
3. `value_loss(value, z)`

The real versions are inside `loss_fn` in `train.py` (`make_train_step`).

## Things to try afterwards

- Watch `policy_loss` and `policy_top1` in `runs/<run>/train.log`. What does a
  policy loss of ~1.0 mean in terms of "how many moves is the network unsure
  between"? (Hint: e^1 ≈ 2.7.)
- Why can't a perfect network get the policy loss to 0 against soft targets?
  (Look up *entropy*.)
