"""
MuZero for The Duke: search a learned model of the game instead of the rules.

AlphaZero (search.py) steps the real environment inside the search tree; the
network only judges positions. MuZero (Schrittwieser et al. 2020) also learns
the transitions, so the search runs entirely on hidden states:

  representation  h: board -> latent s                (DukeZeroNet.represent)
  prediction      f: s -> policy logits, value        (DukeZeroNet.predict)
  afterstate      φ: (s, action) -> afterstate latent  (new)
  afterstate pred ψ: afterstate -> value Q, reward, chance distribution (new)
  chance dynamics g: (afterstate, outcome) -> next s   (new)

The tree is Stochastic MuZero's (Antonoglou et al. 2022), the same as our
AlphaZero search: decision nodes alternate with chance nodes. Stochastic
MuZero learns discrete "chance codes" with an encoder because it can't see
the randomness; here the chance outcome is observable (which tile came out
of the bag), so the true outcome is the code: 15 tile types, and outcome 0
after any move that isn't a draw. Turn order is also known from the action:
a draw keeps the mover, anything else passes the turn (discount -1).

Only the root knows the real legal moves. Deeper in the tree the policy's own
priors are used, so the policy is trained over all 3,605 actions (no mask) and
learns to put its mass on legal ones.

Training (as MuZero): from a stored position, unroll K moves through the model
with the actions and draws that were really played, and match at every step
  policy, value          to the search policy and n-step value target
  afterstate Q, reward   to reward + discount * next value target, and the reward
  chance distribution    to the outcome that happened
Past the end of a game the targets are those of an absorbing state (value 0,
reward 0, outcome 0, no policy) and the actions random.

Counterfactual moves (make_loss_fn's `counterfactual`, not in MuZero): trained
only on moves that were played, the model never learns what other moves lead
to. mz2's Q head scored checkmating moves at a median 0.42 (target 1), and
searching the model did worse than the raw policy (EXPERIMENTS.md). So each
sample also takes a few more legal moves (drawn as the search draws root
candidates: policy + Gumbel noise) and trains Q, reward and chance on what the
rules say they lead to: the result if the game ends, else the network's value
of the real next position, which is what the rules search sees on a first visit.

Warm start: h and f have exactly DukeZeroNet's shape, so a MuZero run can start
from an AlphaZero checkpoint (train.py --init-from) with only φ, ψ, g fresh.
"""

import jax
import jax.numpy as jnp
import mctx
import numpy as np
from flax import nnx
from mctx._src import base as mctx_base

from training.zero import env as E
from training.zero import replay as RP
from training.zero import rules as R
from training.zero import search as S
from training.zero.network import DukeZeroNet, ResBlock, observe_batch

A, C, SQ = R.N_ACTIONS, R.N_TYPES, R.N_SQ
NEG = S.NEG


# ---------------------------------------------------------------------------
# Action encoding for the dynamics network
# ---------------------------------------------------------------------------

def _action_tables():
    """Per action: 3 board planes (source, destination, commander square) and
    5 features (draw, place direction one-hot)."""
    planes = np.zeros((A, SQ, 3), np.float32)
    feats = np.zeros((A, 5), np.float32)
    for a in range(R.N_MOVE_ACTIONS):
        planes[a, a // SQ, 0] = 1
        planes[a, a % SQ, 1] = 1
    for a in range(R.CMD_BASE, R.DRAW_ACTION):
        c = a - R.CMD_BASE
        cmd, sd, dd = c // 64, (c // 8) % 8, c % 8
        src, dst = R.NBR8[cmd, sd], R.NBR8[cmd, dd]
        if src >= 0:
            planes[a, src, 0] = 1
        if dst >= 0:
            planes[a, dst, 1] = 1
        planes[a, cmd, 2] = 1
    feats[R.DRAW_ACTION, 0] = 1
    for d in range(4):
        feats[R.PLACE_BASE + d, 1 + d] = 1
    return planes.reshape(A, R.BOARD, R.BOARD, 3), feats


ACTION_PLANES, ACTION_FEATURES = (jnp.asarray(x) for x in _action_tables())


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class MuZeroNet(nnx.Module):
    def __init__(self, channels: int = 96, blocks: int = 6, policy_head: str = 'spatial',
                 dynamics_blocks: int = 3, *, dtype=jnp.bfloat16, rngs: nnx.Rngs):
        c = channels
        self.az = DukeZeroNet(channels, blocks, policy_head, dtype=dtype, rngs=rngs)   # h and f
        # φ: afterstate dynamics
        self.after_in_norm = nnx.LayerNorm(c, rngs=rngs)
        self.action_conv = nnx.Conv(c + 3, c, (3, 3), dtype=dtype, rngs=rngs)
        self.action_film = nnx.Linear(5, c, rngs=rngs)
        self.after_blocks = nnx.List([ResBlock(c, dtype=dtype, rngs=rngs) for _ in range(dynamics_blocks)])
        # ψ: afterstate prediction
        self.after_norm = nnx.LayerNorm(c, rngs=rngs)
        self.after_conv = nnx.Conv(c, 8, (1, 1), dtype=dtype, rngs=rngs)
        self.after_hidden = nnx.Linear(SQ * 8 + c, 128, rngs=rngs)
        self.after_out = nnx.Linear(128, 2 + C, rngs=rngs,
                                    kernel_init=nnx.initializers.zeros)  # Q 0, reward 0, uniform chance
        # g: chance dynamics
        self.chance_in_norm = nnx.LayerNorm(c, rngs=rngs)
        self.chance_film = nnx.Linear(C, c, rngs=rngs)
        self.chance_blocks = nnx.List([ResBlock(c, dtype=dtype, rngs=rngs) for _ in range(dynamics_blocks)])

    def __call__(self, planes, features):
        """Initial inference: board -> (logits, value), exactly as DukeZeroNet."""
        return self.az(planes, features)

    def represent(self, planes, features):
        return self.az.represent(planes, features)

    def predict(self, latent):
        return self.az.predict(latent)

    def afterstate(self, latent, action):
        """φ: (latent, action [B]) -> afterstate latent. Inputs are normalized, so
        latents from h and from g (different scales) look alike to it."""
        x = self.after_in_norm(latent)
        y = self.action_conv(jnp.concatenate([x, ACTION_PLANES[action]], -1)).astype(jnp.float32)
        x = x + y + self.action_film(ACTION_FEATURES[action])[:, None, None, :]
        for block in self.after_blocks:
            x = block(x)
        return x

    def afterstate_predict(self, after):
        """ψ: afterstate -> (Q for the mover, reward for the mover, chance logits [B, 15])."""
        b = after.shape[0]
        x = nnx.relu(self.after_norm(after))
        pooled = x.mean(axis=(1, 2))
        v = nnx.relu(self.after_conv(x)).reshape(b, -1).astype(jnp.float32)
        h = nnx.relu(self.after_hidden(jnp.concatenate([v, pooled], -1)))
        out = self.after_out(h)
        return jnp.tanh(out[:, 0]), jnp.tanh(out[:, 1]), out[:, 2:]

    def chance(self, after, outcome):
        """g: (afterstate latent, outcome [B]) -> next latent, from the next mover's view."""
        x = self.chance_in_norm(after) + self.chance_film(jax.nn.one_hot(outcome, C))[:, None, None, :]
        for block in self.chance_blocks:
            x = block(x)
        return x


def copy_alphazero(params, az_params):
    """MuZero params with h and f taken from an AlphaZero network's params."""
    mz, az = nnx.to_pure_dict(params), nnx.to_pure_dict(az_params)
    assert jax.tree.structure(mz['az']) == jax.tree.structure(az), 'AlphaZero network shape differs'
    mz['az'] = az
    nnx.replace_by_pure_dict(params, mz)
    return params


# ---------------------------------------------------------------------------
# Search over the learned model
# ---------------------------------------------------------------------------

def make_search(graphdef, num_simulations, max_considered=S.SEARCH['maxConsidered'], max_depth=None,
                value_scale=S.SEARCH['valueScale']):
    """search(params, rng, states, gumbel_scale) -> S.SearchOutput, like S.make_search.

    Node embeddings: decision nodes hold a latent; chance nodes hold the
    afterstate latent with its predicted reward and the turn's discount."""

    def root_fn(params, states):
        model = nnx.merge(graphdef, params)
        latent = model.represent(*observe_batch(states))
        logits, value = model.predict(latent)
        b = latent.shape[0]
        after = {'latent': jnp.zeros_like(latent), 'reward': jnp.zeros(b), 'discount': jnp.zeros(b)}
        return logits, value, mctx_base.StochasticRecurrentState(
            state_embedding=latent, afterstate_embedding=after, is_decision_node=jnp.ones(b, bool))

    def recurrent_fn(params, rng, action_or_chance, emb):
        del rng
        model = nnx.merge(graphdef, params)
        is_decision = emb.is_decision_node
        b = is_decision.shape[0]
        # Both branches are computed and the right one kept per game (as in search.py).
        action = jnp.clip(action_or_chance, 0, A - 1)
        after = model.afterstate(emb.state_embedding, action)
        q, reward, chance_logits = model.afterstate_predict(after)
        outcome = jnp.clip(action_or_chance - A, 0, C - 1)
        nxt = model.chance(emb.afterstate_embedding['latent'], outcome)
        logits, value = model.predict(nxt)

        decision_out = mctx.RecurrentFnOutput(
            reward=jnp.zeros(b), discount=jnp.ones(b),
            prior_logits=jnp.concatenate([jnp.full((b, A), NEG), chance_logits - chance_logits.max(-1, keepdims=True)], -1),
            value=q)
        chance_out = mctx.RecurrentFnOutput(
            reward=emb.afterstate_embedding['reward'], discount=emb.afterstate_embedding['discount'],
            prior_logits=jnp.concatenate([logits - logits.max(-1, keepdims=True), jnp.full((b, C), NEG)], -1),
            value=value)
        out = S._select(is_decision, decision_out, chance_out)
        new_emb = mctx_base.StochasticRecurrentState(
            state_embedding=nxt,
            afterstate_embedding={'latent': after, 'reward': reward,
                                  'discount': jnp.where(action == R.DRAW_ACTION, 1.0, -1.0)},
            is_decision_node=~is_decision)
        return out, new_emb

    return S.make_search_from(root_fn, recurrent_fn, num_simulations, max_considered, max_depth, value_scale)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

TRAJ_SPEC = {
    **RP.SAMPLE_SPEC,
    'action': ((), np.int16),       # what was played
    'outcome': ((), np.int8),       # chance outcome that followed (drawn tile, or 0)
    'reward': ((), np.float32),     # for the mover
    'discount': ((), np.float32),   # -1 turn passes, +1 same mover (draw), 0 game over
    'remaining': ((), np.int16),    # steps left in the game, this one included
}


def game_steps(game: dict, nstep: int) -> dict:
    """One finished game -> its steps in order, with value targets (no extra
    afterstate samples: MuZero learns afterstates through the model)."""
    r, d = game['reward'], game['discount']
    z_mc = np.zeros(len(r), np.float32)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + d[t] * acc
        z_mc[t] = acc
    out = {k: game[k] for k in TRAJ_SPEC if k in game and k not in ('z', 'z_mc', 'remaining')}
    out['z_mc'] = z_mc
    out['z'] = RP.nstep_targets(r, d, game['search_value'], nstep) if nstep > 0 else z_mc.copy()
    out['remaining'] = np.arange(len(r), 0, -1).astype(np.int16)
    return out


class TrajectoryAssembler(RP.Assembler):
    def game_samples(self, game):
        return game_steps(game, self.nstep)


class TrajectoryReplay:
    """Steps stored in game order in a ring buffer; samples unroll windows."""

    def __init__(self, capacity: int, unroll: int):
        self.capacity, self.unroll = capacity, unroll
        self.data = {k: np.zeros((capacity,) + shape, dtype) for k, (shape, dtype) in TRAJ_SPEC.items()}
        self.pos = self.size = 0

    def add(self, steps: dict):
        n = len(steps['z'])
        idx = (self.pos + np.arange(n)) % self.capacity
        for k, arr in self.data.items():
            arr[idx] = steps[k]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, rng) -> dict:
        """n windows: the state at t, and for k = 0..K the targets and what was
        played at t+k ('u_*' arrays [n, K+1]); u_valid marks steps inside the game."""
        K = self.unroll
        start = rng.integers(0, self.size, n)
        if self.size == self.capacity:          # oldest data starts at pos
            start = (self.pos + start) % self.capacity
        else:
            start = start % self.capacity
        k = np.arange(K + 1)
        idx = (start[:, None] + k) % self.capacity
        valid = k[None, :] < self.data['remaining'][start][:, None]
        out = {key: self.data[key][start] for key in RP.STATE_KEYS}
        for key in ('pol_idx', 'pol_p', 'has_policy', 'z', 'outcome', 'reward', 'discount', 'action'):
            out[f'u_{key}'] = self.data[key][idx]
        out['u_valid'] = valid
        # Past the end: an absorbing state reached by random actions.
        out['u_action'] = np.where(valid, out['u_action'], rng.integers(0, A, valid.shape)).astype(np.int16)
        for key in ('u_z', 'u_reward', 'u_discount'):
            out[key] = np.where(valid, out[key], 0).astype(np.float32)
        out['u_outcome'] = np.where(valid, out['u_outcome'], 0).astype(np.int8)
        out['u_has_policy'] = out['u_has_policy'] & valid
        out['cf_seed'] = rng.integers(0, 2 ** 31 - 1, n).astype(np.int32)    # counterfactual move draws
        # Fresh-sample diagnostics (train.make_diagnostics) read these at t.
        for key in ('pol_idx', 'pol_p', 'has_policy', 'z'):
            out[key] = out[f'u_{key}'][:, 0]
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def scale_gradient(x, scale):
    return x * scale + jax.lax.stop_gradient(x) * (1 - scale)


def counterfactual_moves(legal, logits, key, n: int):
    """n distinct legal moves per position [B, n], drawn as the search draws
    root candidates (Gumbel top-k on the policy). ok marks real picks: a
    position with fewer than n legal moves fills the rest with illegal ones."""
    g = jax.random.gumbel(key, legal.shape)
    score = jnp.where(legal, jax.lax.stop_gradient(logits).astype(jnp.float32) + g, -jnp.inf)
    _, moves = jax.lax.top_k(score, n)
    return moves, jnp.take_along_axis(legal, moves, -1)


def real_targets(graphdef, params, states: E.State, moves):
    """What the rules say moves [B, n] lead to, as afterstate targets:
    (Q for the mover, reward, chance distribution [B, n, C]). Q is the rules
    search's first-visit value (search.make_recurrent_fn): the result if the
    game ends, else the network's value of the real next position."""
    model = nnx.merge(graphdef, jax.lax.stop_gradient(params))
    b, n = moves.shape
    after = jax.vmap(jax.vmap(E.step_decision, (None, 0)))(states, moves)
    flat = jax.tree.map(lambda x: x.reshape((b * n,) + x.shape[2:]), after)
    _, value = model.predict(model.represent(*observe_batch(flat)))
    value = value.astype(jnp.float32).reshape(b, n)
    q = jnp.where(after.done, after.reward, jnp.where(after.switched, -value, value))
    return q, after.reward, jax.vmap(jax.vmap(E.chance_probs))(after)


def make_loss_fn(graphdef, counterfactual: int = 0):
    """loss_fn(params, batch) -> (loss, metrics) over K-step unrolls (TrajectoryReplay.sample).
    counterfactual: extra legal moves per sample labelled by the rules (module docstring)."""
    observe = jax.vmap(E.observe_arrays)
    legal_actions = jax.vmap(E.legal_actions)

    def counterfactual_loss(model, params, latent, logits, b, legal, place_ok, overflow):
        empty = E._empty_state()
        cols = [b[k].astype(jnp.int32) for k in ('tile', 'owner', 'side', 'bags', 'pending', 'ply')]
        states = jax.vmap(lambda tile, owner, side, bags, pending, ply, legal, place_ok, overflow: empty._replace(
            tile=tile, owner=owner, side=side, bags=bags, pending=pending, ply=ply,
            legal=legal, place_ok=place_ok, overflow=overflow))(*cols, legal, place_ok, overflow)
        moves, ok = counterfactual_moves(legal, logits, jax.random.key(b['cf_seed'][0]), counterfactual)
        q_target, reward_target, chance_target = real_targets(graphdef, params, states, moves)
        bsz = latent.shape[0]
        after = model.afterstate(jnp.repeat(latent, counterfactual, axis=0), moves.reshape(-1))
        q, reward, chance_logits = (x.astype(jnp.float32).reshape((bsz, counterfactual) + x.shape[1:])
                                    for x in model.afterstate_predict(after))
        w = ok.astype(jnp.float32)
        mean = lambda x: jnp.sum(x * w) / jnp.maximum(w.sum(), 1)
        wins = w * (reward_target > 0)
        return {
            'cf_q_loss': mean((q - q_target) ** 2),
            'cf_reward_loss': mean((reward - reward_target) ** 2),
            'cf_chance_loss': mean(-jnp.sum(chance_target * jax.nn.log_softmax(chance_logits), -1)),
            'cf_q_on_wins': jnp.sum(q * wins) / jnp.maximum(wins.sum(), 1),     # want 1
            'cf_win_rate': wins.sum() / jnp.maximum(w.sum(), 1),                 # how often a pick wins
        }

    def dense_policy(idx, p):
        n = idx.shape[0]
        return jnp.zeros((n, A)).at[jnp.arange(n)[:, None], idx.astype(jnp.int32)].add(p.astype(jnp.float32))

    def loss_fn(params, b):
        model = nnx.merge(graphdef, params)
        tile, owner, side = (b[k].astype(jnp.int32) for k in ('tile', 'owner', 'side'))
        bags, pending = b['bags'].astype(jnp.int32), b['pending'].astype(jnp.int32)
        latent = model.represent(*observe(tile, owner, side, bags, pending, b['ply']))
        legal, place_ok, overflow = legal_actions(tile, owner, side, bags, pending)
        K = b['u_z'].shape[1] - 1
        m, total = {}, 0.0
        for k in range(K + 1):
            w = 1.0 if k == 0 else 1.0 / K
            logits, value = model.predict(latent)
            logp = jax.nn.log_softmax(logits.astype(jnp.float32))      # all actions: learn legality
            target = dense_policy(b['u_pol_idx'][:, k], b['u_pol_p'][:, k])
            hp = b['u_has_policy'][:, k].astype(jnp.float32)
            policy = jnp.sum(-jnp.sum(target * logp, -1) * hp) / jnp.maximum(hp.sum(), 1)
            value_loss = jnp.mean((value - b['u_z'][:, k]) ** 2)
            total = total + w * (policy + value_loss)
            m[f'policy_{k}'], m[f'value_{k}'] = policy, value_loss
            if k == 0:
                m['policy_top1'] = jnp.sum((jnp.argmax(jnp.where(legal, logits, -jnp.inf), -1)
                                            == jnp.argmax(target, -1)) * hp) / jnp.maximum(hp.sum(), 1)
                m['legal_mass'] = jnp.mean(jnp.sum(jnp.where(legal, jnp.exp(logp), 0.0), -1))
                if counterfactual:
                    cf = counterfactual_loss(model, params, latent, logits, b, legal, place_ok, overflow)
                    total = total + cf['cf_q_loss'] + cf['cf_reward_loss'] + cf['cf_chance_loss']
            if k == K:
                break
            after = model.afterstate(latent, b['u_action'][:, k].astype(jnp.int32))
            q, reward, chance_logits = model.afterstate_predict(after)
            q_target = b['u_reward'][:, k] + b['u_discount'][:, k] * b['u_z'][:, k + 1]
            outcome = b['u_outcome'][:, k].astype(jnp.int32)
            chance = -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(chance_logits), outcome[:, None], -1))
            q_loss = jnp.mean((q - q_target) ** 2)
            reward_loss = jnp.mean((reward - b['u_reward'][:, k]) ** 2)
            total = total + w * (q_loss + reward_loss + chance)
            m[f'q_{k}'], m[f'reward_{k}'], m[f'chance_{k}'] = q_loss, reward_loss, chance
            latent = scale_gradient(model.chance(after, outcome), 0.5)
        metrics = {
            'policy_loss': m['policy_0'], 'value_loss': m['value_0'], 'policy_top1': m['policy_top1'],
            'legal_mass': m['legal_mass'],
            'policy_loss_unrolled': sum(m[f'policy_{k}'] for k in range(1, K + 1)) / K,
            'value_loss_unrolled': sum(m[f'value_{k}'] for k in range(1, K + 1)) / K,
            'q_loss': sum(m[f'q_{k}'] for k in range(K)) / K,
            'reward_loss': sum(m[f'reward_{k}'] for k in range(K)) / K,
            'chance_loss': sum(m[f'chance_{k}'] for k in range(K)) / K,
            **(cf if counterfactual else {}),
        }
        return total, metrics

    return loss_fn
