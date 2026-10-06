"""
Matches between agents: the network (with search), TypeScript minimax, random.

Games run in the JAX environment as one batch. Each step, every agent picks
moves for the games where it is to move: the network with one batched search,
minimax via the process pool (in parallel across games), random on the host.
Agent A plays P1 in even-numbered games and P2 in odd ones, so first-move
advantage cancels out.

Minimax answers a draw with a resolved `place` (it draws uniformly from the
bag itself). We replay that as draw -> chance outcome forced to that tile ->
place on the following step, so the env and the TS engine stay in sync.

Optionally records every step, which is how imitation data is generated.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from training.zero import env as E
from training.zero import rules as R
from training.zero import selfplay as SP
from training.zero import ts_bridge as B


@dataclass
class Agent:
    kind: str                      # 'net' | 'minimax' | 'random'
    name: str = ''
    params: object = None          # net
    search: object = None          # net: jitted search(params, key, states, gumbel_scale)
    depth: int | list[int] = 4     # minimax: fixed depth, or per-game list
    epsilon: float = 0.0           # chance to play a random legal move instead (diversity)

    def __post_init__(self):
        self.name = self.name or self.kind


@jax.jit
def _advance(s, actions, forced, use_forced, key):
    after = jax.vmap(E.step_decision)(s, actions)
    sampled = jax.random.categorical(key, jnp.log(jax.vmap(E.chance_probs)(after)))
    nxt = jax.vmap(E.step_chance)(after, jnp.where(use_forced, forced, sampled))
    return nxt, after


_init = jax.jit(jax.vmap(E.init))
_compact = jax.jit(SP.compact_state)
_one_hot = jax.jit(SP.one_hot_policy)


def _unbatch(s, i):
    return jax.tree.map(lambda x: np.asarray(x[i]), s)


def play(agent_a: Agent, agent_b: Agent, n_games: int, key, pool=None, record=False):
    """Play n_games; returns (results dict, records or None).

    results: wins/losses/draws from agent A's point of view.
    records: list of per-step dicts ([B]-shaped arrays) for replay.Assembler,
             with one-hot policy targets for minimax/net moves (random and
             epsilon moves get has_policy=False).
    """
    rng = np.random.default_rng(np.asarray(jax.random.key_data(key)).sum().item())
    key, k0 = jax.random.split(key)
    s = _init(jax.random.split(k0, n_games))
    a_player = np.arange(n_games) % 2
    queued = np.full(n_games, -1)
    steps = []

    while True:
        player, pending, done, legal = jax.device_get((s.player, s.pending, s.done, s.legal))
        if done.all():
            break
        actions = np.zeros(n_games, np.int32)
        forced = np.zeros(n_games, np.int32)
        use_forced = np.zeros(n_games, bool)
        has_policy = np.zeros(n_games, bool)

        for agent, turn in ((agent_a, player == a_player), (agent_b, player != a_player)):
            mask = turn & ~done
            if not mask.any():
                continue
            if agent.kind == 'net':
                key, k = jax.random.split(key)
                out = agent.search(agent.params, k, s, 0.0)
                actions[mask] = np.asarray(out.action)[mask]
                has_policy[mask] = True
            elif agent.kind == 'random':
                for i in np.flatnonzero(mask):
                    actions[i] = rng.choice(np.flatnonzero(legal[i]))
            elif agent.kind == 'minimax':
                queue = mask & (queued >= 0)          # second half of a draw
                actions[queue] = queued[queue]
                queued[queue] = -1
                has_policy[queue] = True
                # After an epsilon-random draw, minimax can't see the mid-draw
                # state (TS has none), so place that tile randomly.
                for i in np.flatnonzero(mask & ~queue & (pending >= 0)):
                    actions[i] = rng.choice(np.flatnonzero(legal[i]))
                ask = np.flatnonzero(mask & ~queue & (pending == E.NONE))
                depths = agent.depth if isinstance(agent.depth, int) else [agent.depth[i] for i in ask]
                moves = pool.best_moves([B.state_to_ts(_unbatch(s, i)) for i in ask], depths)
                for i, move in zip(ask, moves):
                    st = _unbatch(s, i)
                    seq = B.move_to_actions(move, int(st.player), B.duke_square(st.tile, st.owner))
                    actions[i] = seq[0][1]
                    has_policy[i] = True
                    if len(seq) == 3:
                        forced[i], use_forced[i], queued[i] = seq[1][1], True, seq[2][1]
            else:
                raise ValueError(agent.kind)

            # Epsilon-random moves (only at fresh decision points, never mid-draw).
            if agent.epsilon > 0:
                for i in np.flatnonzero(mask & (pending == E.NONE)):
                    if rng.random() < agent.epsilon:
                        actions[i] = rng.choice(np.flatnonzero(legal[i]))
                        has_policy[i] = False
                        use_forced[i], queued[i] = False, -1

        key, k = jax.random.split(key)
        nxt, after = _advance(s, jnp.asarray(actions), jnp.asarray(forced), jnp.asarray(use_forced), k)
        if record:
            rec = {**_compact(s), **_one_hot(jnp.asarray(actions)),
                   **SP.step_record(s, after, jnp.asarray(actions))}
            rec = jax.device_get(rec)
            rec['has_policy'] = has_policy & ~done
            steps.append(rec)
        s = nxt

    winner = np.asarray(s.winner)
    results = {
        'a': agent_a.name, 'b': agent_b.name, 'games': n_games,
        'wins': int((winner == a_player).sum()),
        'losses': int(((winner >= 0) & (winner != a_player)).sum()),
        'draws': int((winner < 0).sum()),
        'mean_plies': float(np.asarray(s.ply).mean()),
    }
    results['score'] = (results['wins'] + 0.5 * results['draws']) / n_games
    return results, steps if record else None


def stack_steps(steps: list[dict]) -> dict:
    """List of per-step [B] dicts -> one [T, B] chunk for replay.Assembler."""
    return {k: np.stack([s[k] for s in steps]) for k in steps[0]}
