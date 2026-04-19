# Reinforcement learning for The Duke
_Exported on 3/25/2026 at 21:26:59 EDT from Cursor (2.6.21)_

---

**User**

Hey Claude, I'm working on making a rl net for the board game "The Duke". Eventually I'd like to work up to MuZero, but i'd like to start basic. Please take a look at the repo, and lead me towards developing a basic net for the game

---

**Cursor**

Let me explore your repo to understand what you have so far.

Great project — you've built a really solid foundation here. You have a complete game engine, a working minimax AI, a self-play data pipeline, and a data loader for JAX. The gap is the neural network and training loop. Let me lay out a clear path forward.

---

## Where you are now

- **Engine** (TS): Full rules, move gen, apply/undo — all working and tested
- **Classical AI** (TS): Minimax with alpha-beta, Zobrist TT, heuristic eval — solid baseline
- **Encoding** (TS): 33×6×6 board channels + 30 bag features = 1218-dim state; 4428 discrete actions
- **Self-play** (TS): Minimax vs minimax → binary `.bin` files with `(state, action, value)` tuples
- **Data loading** (Python): `load_data.py` → JAX arrays: `boards (N, 33, 6, 6)`, `bags (N, 30)`, `actions (N,)`, `values (N,)`
- **Dependencies**: JAX (CUDA), Flax, Optax — all set

**What's missing**: a model, a training loop, and eventually MCTS + RL.

---

## Recommended progression

### Stage 1: Supervised dual-head net (behavioral cloning from minimax) — **start here**

Train a network to imitate your minimax player. This is the fastest way to get a working neural net and validate your entire pipeline end-to-end.

**Why start here instead of jumping to RL:**
- Validates your encoding is correct (bugs in encoding will show up as terrible loss curves)
- Validates data loading, batching, training loop
- Gives you a baseline network you can evaluate against minimax
- The architecture you build here is *exactly* what you'll reuse for AlphaZero/MuZero

### Stage 2: Self-play RL (AlphaZero-style)
Replace minimax self-play with MCTS guided by the network. Train on MCTS policy targets + game outcomes.

### Stage 3: MuZero
Add a learned dynamics model so the network can plan without needing the game rules.

---

## Stage 1: The Model

For a 6×6 board, you don't need anything huge. A small ResNet with a dual head (policy + value) is the standard architecture. Here's what I'd recommend:

### Architecture

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class ResBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x + residual)
        return x

class DukeNet(nn.Module):
    num_filters: int = 64
    num_blocks: int = 4
    num_actions: int = 4428

    @nn.compact
    def __call__(self, boards, bags, train: bool = True):
        # boards: (B, 33, 6, 6) → need NHWC for Flax conv
        x = jnp.transpose(boards, (0, 2, 3, 1))  # → (B, 6, 6, 33)

        # Initial conv to project channels
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Residual tower
        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters)(x, train=train)

        # Flatten spatial and concat bag features
        flat = x.reshape(x.shape[0], -1)           # (B, 6*6*64)
        flat = jnp.concatenate([flat, bags], axis=-1)  # (B, 6*6*64 + 30)

        # Policy head
        p = nn.Dense(256)(flat)
        p = nn.relu(p)
        p = nn.Dense(self.num_actions)(p)  # raw logits

        # Value head
        v = nn.Dense(128)(flat)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = nn.tanh(v).squeeze(-1)  # (B,) in [-1, 1]

        return p, v
```

Key design decisions:
- **4 res blocks, 64 filters** — plenty for a 6×6 board. AlphaGo Zero used 19 or 39 blocks on 19×19; you need far less.
- **Bags concatenated after flattening** — bag counts aren't spatial, so they don't belong in the conv stack. Concat them into the FC layers.
- **Policy outputs raw logits** — you'll mask illegal actions and softmax during loss computation.
- **Value is tanh** — bounded to [-1, +1] matching your value targets.

### Training loop

```python
import optax
from flax.training import train_state

def create_train_state(rng, learning_rate=1e-3):
    model = DukeNet()
    dummy_boards = jnp.zeros((1, 33, 6, 6))
    dummy_bags = jnp.zeros((1, 30))
    variables = model.init(rng, dummy_boards, dummy_bags, train=False)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'],
        tx=tx, # you'll need to track batch_stats separately
    )

def loss_fn(params, batch_stats, boards, bags, actions, values, train=True):
    variables = {'params': params, 'batch_stats': batch_stats}
    (logits, pred_values), updates = model.apply(
        variables, boards, bags, train=train,
        mutable=['batch_stats'],
    )

    # Policy loss: cross-entropy on the action taken
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, actions
    ).mean()

    # Value loss: MSE
    value_loss = jnp.mean((pred_values - values) ** 2)

    total_loss = policy_loss + value_loss
    return total_loss, (policy_loss, value_loss, updates)
```

For the training step, you'd use `jax.grad` on `loss_fn` and apply updates. I'd recommend tracking these metrics:
- **Policy accuracy**: does argmax of logits match the minimax action?
- **Policy loss**: cross-entropy
- **Value loss**: MSE
- **Value accuracy**: sign(prediction) matches sign(target)?

### Data generation

You need a decent amount of data. Your self-play pipeline writes minimax-vs-minimax games at depth 4. Each game produces roughly 30-80 samples. I'd suggest:

1. Generate ~1,000–10,000 games initially (`npx tsx packages/ai/src/generate.ts 1000 4 training/selfplay.bin`)
2. That gives you roughly 30K–500K training samples
3. Split 90/10 for train/val
4. Train for ~50–100 epochs with batch size 256–512

### Evaluation

Once trained, you can evaluate by:
1. **Policy accuracy on held-out data** — "does the net pick the same move as minimax?" Even 30-40% top-1 accuracy is meaningful given the branching factor.
2. **Play the net against minimax** — hardest part is wiring inference back to TS (ONNX export or running in Python). For now, you can do evaluation entirely in Python by reimplementing a simple game loop, or just use offline metrics.
3. **Play the net against random** — should crush random very quickly if training works.

---

## Concrete next steps (what to build)

1. **`training/model.py`** — the `DukeNet` Flax module (architecture above)
2. **`training/train.py`** — data loading, batching, training loop, checkpointing, logging
3. **Generate data** — run your self-play generator for a few thousand games
4. **Train and iterate** — watch loss curves, tune hyperparams
5. **`training/evaluate.py`** — offline evaluation metrics on held-out data

Once Stage 1 is working end-to-end and you can see the network learning (decreasing loss, reasonable policy accuracy), you'll have all the infrastructure to pivot to Stage 2 (MCTS self-play) which is the real path toward AlphaZero and eventually MuZero.

---

## A few things I noticed in the repo

- **`load_data.py` lines 71-74**: The nested single quotes inside single-quoted f-strings work on Python 3.12+ (PEP 701), so since you're targeting 3.14, you're fine.
- **`selfplay.worker.ts`**: The imports reference `SelfPlayOptions` and `result.samples`, but your AI package exports `SelfPlayConfig` and `SelfPlayResult` with `data`/`numSamples`/`outcome`. This worker won't compile against the current API — worth fixing when you get to browser-side self-play.
- **`pyproject.toml`**: You have `jax[cuda13]` — just make sure your CUDA version matches. If you're also developing on CPU sometimes, you might want a separate CPU install path.

Want me to go deeper on any of these pieces — the model architecture, the training loop, or how to wire things up for MCTS in Stage 2?

---

**User**

Thanks Claude. I'll use this as reference. I'd like to figure out the code myself so I can learn though. I'm a decent programmer but have very little ml knowledge. please make a folder "supervised", stub out the project, and give me some info to get the ball rolling

---

**Cursor**

Here's what I created:

```
training/supervised/
├── __init__.py
├── model.py       ← DukeNet architecture (ResBlock + dual-head net)
├── train.py       ← training loop with data loading, batching, checkpointing
└── evaluate.py    ← offline metrics (policy accuracy, value MSE, etc.)
```

Each file has the function signatures, docstrings, and detailed guidance in comments explaining *what* to do and *why*, but the implementations are yours to write. Here's the recommended order to tackle them and some concepts to ground you:

---

### 1. Start with `model.py` — the architecture

This is the most self-contained piece. You're building two things:

- **`ResBlock`**: Two convolutions with a skip connection. The key insight is `output = relu(x + conv(conv(x)))`. The `+ x` (skip) is what makes it "residual" — the network only has to learn the *delta* from identity, which makes training much more stable.

- **`DukeNet`**: Wire up an initial conv, a stack of ResBlocks, then split into two heads. The policy head outputs 4428 raw numbers (one per possible action). The value head outputs a single number squeezed through `tanh` to stay in [-1, 1].

**Sanity check**: once written, you should be able to do:

```python
model = DukeNet()
rng = jax.random.PRNGKey(0)
variables = model.init(rng, jnp.zeros((1, 33, 6, 6)), jnp.zeros((1, 30)), train=False)
print(jax.tree.map(lambda x: x.shape, variables['params']))
```

If that prints a tree of parameter shapes without crashing, your architecture is wired up.

### 2. Then `train.py` — the training loop

The mental model: you have data (state, action, value) from minimax games. You're asking the network "given this board, predict which move minimax would play and who wins." The **loss function** measures how wrong it is, and **gradients** tell you which direction to nudge each parameter to be less wrong. Repeat thousands of times.

The key JAX pattern:

```python
grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
(loss, aux), grads = grad_fn(state.params, ...)
state = state.apply_gradients(grads=grads)
```

That's the entire core — everything else is data plumbing and logging.

**Sanity checks along the way**:
- Loss should decrease each epoch. If it goes up or stays flat, something is wrong.
- On the very first batch (random weights), policy loss should be roughly `ln(4428) ≈ 8.4` (that's cross-entropy with a uniform random prediction over 4428 classes). If you see something wildly different, check your loss computation.
- Value loss should start near ~0.5-1.0 (random predictions against +1/-1 targets) and decrease.

### 3. Finally `evaluate.py` — validation

Simpler than training since there's no gradient computation. Load the checkpoint, run forward passes, compute metrics. The key metric is **policy top-1 accuracy** — if your network picks the same move as minimax even 25-35% of the time, it's genuinely learned something (random would be ~3-5%).

---

### Generating data

Before any of this, you'll need training data. Run something like:

```bash
npx tsx packages/ai/src/generate.ts 1000 4 training/selfplay.bin
```

That plays 1000 minimax-vs-minimax games at depth 4 and writes the samples. Start with a smaller number (like 100) to test your pipeline quickly, then scale up.

---

### Learning resources for the JAX/Flax side

- [Flax Getting Started](https://flax.readthedocs.io/en/latest/quick_start.html) — the MNIST example is very close to what you're building, just with a simpler model
- [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html) — especially the sections on `jit`, `grad`, and `vmap`
- [Optax basics](https://optax.readthedocs.io/en/latest/getting_started.html) — optimizers and loss functions

The BatchNorm + TrainState subclass part is the trickiest Flax-specific bit. The Flax docs have an example under "Managing State" that covers exactly this pattern. If that feels like too much friction initially, you can skip BatchNorm entirely and use `nn.LayerNorm` instead (which doesn't need mutable state) — it'll train just fine for this use case and simplifies everything.

