"""
AlphaZero-style training for The Duke: imitation warm-start, then self-play.

    uv run python -m training.zero.train --run my-run

Loop (one "iteration"):
  1. ACTOR (last GPU): play `unroll` moves in each of `games` parallel games,
     every move chosen by tree search with the current network.
  2. LEARNER (first GPU), at the same time: `train_steps` minibatches of SGD
     on the replay buffer:
       policy loss = cross-entropy(network policy, search's improved policy)
       value  loss = (network value - final game result)^2
  3. Finished games go into the replay buffer; the actor gets the new weights.
Every `eval_every` iterations: matches vs random and TypeScript minimax,
checkpoint, metrics line. Everything is logged to runs/<run>/metrics.jsonl.

Before self-play starts, the network is warm-started by imitating minimax:
we play minimax-vs-minimax games and train the policy head to predict
minimax's move and the value head to predict the result. It gives self-play
a sensible starting point instead of random moves.

Resuming: rerunning with the same --run continues from runs/<run>/latest.pkl
(the replay buffer is not saved; it refills from fresh self-play).
"""

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from training.zero import arena as AR
from training.zero import env as E
from training.zero import replay as RP
from training.zero import rules as R
from training.zero import search as S
from training.zero import selfplay as SP
from training.zero.network import DukeZeroNet, make_forward

RUNS = Path(__file__).resolve().parents[2] / 'runs'


@dataclass
class Config:
    run: str = 'default'
    seed: int = 0
    # network
    channels: int = 96
    blocks: int = 6
    # self-play
    games: int = 1024              # parallel self-play games
    unroll: int = 32               # moves per game per iteration
    simulations: int = 32
    max_considered: int = 16
    # learning
    batch: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4
    sample_reuse: float = 4.0      # each new sample is trained on ~this many times
    replay: int = 1_000_000
    min_replay: int = 50_000
    # imitation warm-start
    imitation_games: int = 3000
    imitation_depths: list = field(default_factory=lambda: [2, 3, 4])
    imitation_epsilon: float = 0.1
    imitation_steps: int = 4000
    # evaluation / bookkeeping
    iterations: int = 100_000
    eval_every: int = 25
    eval_games: int = 64
    eval_depths: list = field(default_factory=lambda: [2, 4])
    workers: int = 0               # minimax processes (0 = one per CPU thread)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def make_train_step(forward, tx):
    legal_actions = jax.vmap(E.legal_actions)
    observe = jax.vmap(E.observe_arrays)

    def loss_fn(params, b):
        tile, owner, side = (b[k].astype(jnp.int32) for k in ('tile', 'owner', 'side'))
        bags, pending = b['bags'].astype(jnp.int32), b['pending'].astype(jnp.int32)
        legal, _, _ = legal_actions(tile, owner, side, bags, pending)
        logits, value = forward(params, *observe(tile, owner, side, bags, pending, b['ply']))

        logp = jax.nn.log_softmax(jnp.where(legal, logits, -1e9))
        n = tile.shape[0]
        target = jnp.zeros((n, R.N_ACTIONS)).at[jnp.arange(n)[:, None], b['pol_idx'].astype(jnp.int32)] \
            .add(b['pol_p'].astype(jnp.float32))
        has_policy = b['has_policy'].astype(jnp.float32)
        per_sample = -jnp.sum(target * logp, axis=-1)
        policy_loss = jnp.sum(per_sample * has_policy) / jnp.maximum(has_policy.sum(), 1)
        value_loss = jnp.mean((value - b['z']) ** 2)

        # Diagnostics: does the network's favourite move match the target's?
        top1 = jnp.argmax(logp, -1) == jnp.argmax(target, -1)
        metrics = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_top1': jnp.sum(top1 * has_policy) / jnp.maximum(has_policy.sum(), 1),
        }
        return policy_loss + value_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, opt_state, batch):
        (_, metrics), grads = grad_fn(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        metrics['grad_norm'] = optax.global_norm(grads)
        return optax.apply_updates(params, updates), opt_state, metrics

    return train_step


class MetricAverager:
    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, m):
        for k, v in m.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1

    def pop(self):
        out = {k: v / max(self.n, 1) for k, v in self.sums.items()}
        self.sums, self.n = {}, 0
        return out


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, params, opt_state, iteration, cfg):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump({
            'params': jax.device_get(nnx.to_pure_dict(params)),
            'opt_state': jax.device_get(opt_state),
            'iteration': iteration,
            'config': asdict(cfg),
        }, f)
    tmp.replace(path)


def load_params(path: Path):
    """(params State, config dict) from a checkpoint, for evaluation/export."""
    with path.open('rb') as f:
        ckpt = pickle.load(f)
    c = ckpt['config']
    model = DukeZeroNet(c['channels'], c['blocks'], rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    nnx.replace_by_pure_dict(params, ckpt['params'])
    return graphdef, params, ckpt


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def imitation_data(cfg, pool, key, log):
    """Minimax-vs-minimax games -> replay samples with one-hot policy targets."""
    replay = RP.Replay(cfg.replay)
    rng = np.random.default_rng(cfg.seed)
    per_round = 256
    played = 0
    while played < cfg.imitation_games:
        n = min(per_round, cfg.imitation_games - played)
        depths_a = rng.choice(cfg.imitation_depths, n).tolist()
        depths_b = rng.choice(cfg.imitation_depths, n).tolist()
        a = AR.Agent('minimax', depth=depths_a, epsilon=cfg.imitation_epsilon)
        b = AR.Agent('minimax', depth=depths_b, epsilon=cfg.imitation_epsilon)
        key, k = jax.random.split(key)
        t = time.time()
        _, steps = AR.play(a, b, n, k, pool=pool, record=True)
        samples = RP.Assembler(n).add(AR.stack_steps(steps))
        replay.add(samples)
        played += n
        log(f'imitation: {played}/{cfg.imitation_games} games, {replay.size} samples '
            f'({time.time() - t:.0f}s)')
    return replay


def evaluate(cfg, search, params, pool, key):
    net = AR.Agent('net', params=params, search=search)
    results = []
    opponents = [AR.Agent('random')] + [AR.Agent('minimax', name=f'minimax{d}', depth=d)
                                        for d in cfg.eval_depths]
    for opp in opponents:
        key, k = jax.random.split(key)
        res, _ = AR.play(net, opp, cfg.eval_games, k, pool=pool)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config):
    run_dir = RUNS / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (run_dir / 'metrics.jsonl').open('a')

    def log(msg, **record):
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
        if record:
            metrics_file.write(json.dumps({'time': time.time(), **record}) + '\n')
            metrics_file.flush()

    devices = jax.devices()
    learner, actor = devices[0], devices[-1]
    log(f'devices: learner={learner} actor={actor}')

    model = DukeZeroNet(cfg.channels, cfg.blocks, rngs=nnx.Rngs(cfg.seed))
    graphdef, params = nnx.split(model)
    forward = make_forward(graphdef)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(cfg.lr, weight_decay=cfg.weight_decay))
    opt_state = tx.init(params)
    start_iteration = 0

    latest = run_dir / 'latest.pkl'
    if latest.exists():
        _, params, ckpt = load_params(latest)
        opt_state = ckpt['opt_state']
        start_iteration = ckpt['iteration']
        log(f'resumed from {latest} at iteration {start_iteration}')
    else:
        (run_dir / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    params = jax.device_put(params, learner)
    opt_state = jax.device_put(opt_state, learner)

    train_step = make_train_step(forward, tx)
    search = jax.jit(S.make_search(forward, cfg.simulations, cfg.max_considered))
    selfplay = SP.make_selfplay(S.make_search(forward, cfg.simulations, cfg.max_considered), cfg.unroll)
    np_rng = np.random.default_rng(cfg.seed + start_iteration)
    key = jax.random.key(cfg.seed + start_iteration)

    from training.zero.minimax_pool import MinimaxPool
    pool = MinimaxPool(cfg.workers or None)
    avg = MetricAverager()
    try:
        # --- imitation warm-start (fresh runs only) ---
        if start_iteration == 0 and cfg.imitation_games > 0:
            key, k = jax.random.split(key)
            replay = imitation_data(cfg, pool, k, log)
            for step in range(1, cfg.imitation_steps + 1):
                batch = jax.device_put(replay.sample(cfg.batch, np_rng), learner)
                params, opt_state, m = train_step(params, opt_state, batch)
                avg.add(m)
                if step % 500 == 0:
                    log(f'imitation step {step}: ' + ' '.join(f'{k}={v:.3f}' for k, v in avg.pop().items()))
            key, k = jax.random.split(key)
            results = evaluate(cfg, search, params, pool, k)
            log('after imitation: ' + ', '.join(f"{r['b']} {r['score']:.2f}" for r in results),
                kind='eval', iteration=0, results=results)
            save_checkpoint(run_dir / 'imitation.pkl', params, opt_state, 0, cfg)
        else:
            replay = RP.Replay(cfg.replay)

        # --- self-play ---
        key, k = jax.random.split(key)
        states = jax.device_put(jax.jit(jax.vmap(E.init))(jax.random.split(k, cfg.games)), actor)
        actor_params = jax.device_put(params, actor)
        assembler = RP.Assembler(cfg.games)
        new_per_iter = cfg.games * cfg.unroll
        train_steps = max(1, round(cfg.sample_reuse * new_per_iter / cfg.batch))
        games_done, t_last = 0, time.time()

        for it in range(start_iteration + 1, cfg.iterations + 1):
            key, k = jax.random.split(key)
            states, records = selfplay(actor_params, states, k)        # async on actor GPU

            if replay.size >= cfg.min_replay:                          # overlaps with self-play
                for _ in range(train_steps):
                    batch = jax.device_put(replay.sample(cfg.batch, np_rng), learner)
                    params, opt_state, m = train_step(params, opt_state, batch)
                    avg.add(m)

            records = jax.device_get(records)                          # waits for self-play
            search_value = records.pop('search_value')
            samples = assembler.add(records)
            if samples is not None:
                replay.add(samples)
            actor_params = jax.device_put(params, actor)

            # Self-play statistics for this iteration.
            done = records['done'] & records['valid']
            games_done += int(done.sum())
            rew = records['reward'][done]
            avg.add({'sp_decisive': float((rew != 0).mean()) if rew.size else 0.0,
                     'sp_value_abs': float(np.abs(search_value).mean())})

            if it % 5 == 0:
                dt = time.time() - t_last
                t_last = time.time()
                m = avg.pop()
                log(f'iter {it}: {games_done} games, replay {replay.size}, '
                    f'{5 * new_per_iter / dt:,.0f} moves/s | ' + ' '.join(f'{k}={v:.3f}' for k, v in m.items()),
                    kind='train', iteration=it, games=games_done, replay=replay.size,
                    moves_per_s=5 * new_per_iter / dt, **m)

            if it % cfg.eval_every == 0:
                save_checkpoint(latest, params, opt_state, it, cfg)
                key, k = jax.random.split(key)
                results = evaluate(cfg, search, params, pool, k)
                log(f'eval {it}: ' + ', '.join(f"{r['b']} {r['score']:.2f}" for r in results),
                    kind='eval', iteration=it, results=results)
                if it % (cfg.eval_every * 8) == 0:
                    save_checkpoint(run_dir / f'ckpt_{it:06d}.pkl', params, opt_state, it, cfg)
                t_last = time.time()
    finally:
        pool.close()
        metrics_file.close()


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in asdict(cfg).items():
        if isinstance(value, list):
            parser.add_argument(f'--{name.replace("_", "-")}', type=int, nargs='+', default=value)
        else:
            parser.add_argument(f'--{name.replace("_", "-")}', type=type(value), default=value)
    return Config(**vars(parser.parse_args()))


if __name__ == '__main__':
    main(parse_args())
