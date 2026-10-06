"""
AlphaZero-style training for The Duke: imitation warm-start, then self-play.

    uv run python -m training.zero.train --run my-run

Loop (one "iteration"):
  1. ACTOR (last GPU): play `unroll` moves in each of `games` parallel games,
     every move chosen by tree search with the current network. The learner
     GPU also plays `learner_games` games, since SGD alone leaves it mostly idle.
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
    games: int = 1024              # parallel self-play games on the actor GPU
    learner_games: int = 512       # more self-play games on the learner GPU, which
                                   # would otherwise idle between SGD steps (0 = off)
    unroll: int = 32               # moves per game per iteration
    simulations: int = 32
    max_considered: int = 16
    max_depth: int = 16            # tree edges per simulation (2 per move: decision + chance)
    # learning
    batch: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4
    sample_reuse: float = 4.0      # each new sample is trained on ~this many times
    steps_per_call: int = 32       # SGD steps per jitted call (amortizes launch overhead)
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


def make_train_steps(forward, tx):
    """Several SGD steps in ONE jitted call: train_steps(params, opt, batches).

    `batches` has a leading [n_steps] axis. Launching a jitted function costs
    ~12 ms of Python here (walking the big params/optimizer pytrees), which made
    one-call-per-step training launch-bound; scanning over steps on the device
    pays that once per call. Returns metrics averaged over the steps.
    """
    step = make_train_step(forward, tx)

    @jax.jit
    def train_steps(params, opt_state, batches):
        def body(carry, batch):
            params, opt_state, metrics = step(*carry, batch)
            return (params, opt_state), metrics
        (params, opt_state), metrics = jax.lax.scan(body, (params, opt_state), batches)
        return params, opt_state, jax.tree.map(jnp.mean, metrics)

    return train_steps


def sample_stacked(replay, n_steps, batch, rng):
    """n_steps minibatches from the replay buffer, stacked as [n_steps, batch, ...]."""
    flat = replay.sample(n_steps * batch, rng)
    return {k: v.reshape((n_steps, batch) + v.shape[1:]) for k, v in flat.items()}


class MetricAverager:
    """Running means, counted per key (metrics arrive at different rates:
    training metrics per SGD step, self-play metrics per iteration)."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, m):
        # No float() here: values may be device arrays, and converting forces
        # a GPU sync after every train step. Summing stays on the device.
        for k, v in m.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
            self.counts[k] = self.counts.get(k, 0) + 1

    def pop(self):
        out = {k: float(v) / self.counts[k] for k, v in self.sums.items()}
        self.sums, self.counts = {}, {}
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

def imitation_data(cfg, pool, key, log, cache: Path):
    """Minimax-vs-minimax games -> replay samples with one-hot policy targets.

    Saved to `cache` so a restart doesn't replay all the minimax games."""
    replay = RP.Replay(cfg.replay)
    if cache.exists():
        with np.load(cache) as f:
            replay.add({k: f[k] for k in f.files})
        log(f'imitation: loaded {replay.size} samples from {cache}')
        return replay
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
    np.savez(cache.with_suffix('.tmp.npz'), **{k: v[:replay.size] for k, v in replay.data.items()})
    cache.with_suffix('.tmp.npz').replace(cache)
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
    # Reuse compiled XLA programs across runs and restarts: compiling is slow,
    # and (with jax 0.9.1's CUDA 13 plugin) occasionally crashes, so the less
    # of it a restart does the better.
    jax.config.update('jax_compilation_cache_dir', str(RUNS / '.jax_cache'))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)

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
    imitation_ckpt = run_dir / 'imitation.pkl'
    resume_from = latest if latest.exists() else imitation_ckpt if imitation_ckpt.exists() else None
    if resume_from:
        # The checkpoint defines the network shape, whatever the flags say.
        graphdef, params, ckpt = load_params(resume_from)
        forward = make_forward(graphdef)
        opt_state = ckpt['opt_state']
        start_iteration = ckpt['iteration']
        shape = (ckpt['config']['channels'], ckpt['config']['blocks'])
        if shape != (cfg.channels, cfg.blocks):
            log(f'note: using the checkpoint network {shape}, not --channels/--blocks')
        log(f'resumed from {resume_from} at iteration {start_iteration}')
    else:
        (run_dir / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    params = jax.device_put(params, learner)
    opt_state = jax.device_put(opt_state, learner)

    train_steps_fn = make_train_steps(forward, tx)
    chunk = cfg.steps_per_call
    make = lambda: S.make_search(forward, cfg.simulations, cfg.max_considered, cfg.max_depth)
    search = jax.jit(make())
    selfplay = SP.make_selfplay(make(), cfg.unroll)
    np_rng = np.random.default_rng(cfg.seed + start_iteration)
    key = jax.random.key(cfg.seed + start_iteration)

    from training.zero.minimax_pool import MinimaxPool
    pool = MinimaxPool(cfg.workers or None)
    avg = MetricAverager()
    try:
        # --- imitation warm-start (fresh runs only) ---
        if resume_from is None and cfg.imitation_games > 0:
            key, k = jax.random.split(key)
            replay = imitation_data(cfg, pool, k, log, run_dir / 'imitation_data.npz')
            for step in range(chunk, cfg.imitation_steps + 1, chunk):
                batches = jax.device_put(sample_stacked(replay, chunk, cfg.batch, np_rng), learner)
                params, opt_state, m = train_steps_fn(params, opt_state, batches)
                avg.add(m)
                if step % (chunk * 16) == 0:
                    log(f'imitation step {step}: ' + ' '.join(f'{k}={v:.3f}' for k, v in avg.pop().items()))
            key, k = jax.random.split(key)
            results = evaluate(cfg, search, params, pool, k)
            log('after imitation: ' + ', '.join(f"{r['b']} {r['score']:.2f}" for r in results),
                kind='eval', iteration=0, results=results)
            save_checkpoint(run_dir / 'imitation.pkl', params, opt_state, 0, cfg)
        else:
            replay = RP.Replay(cfg.replay)

        # --- self-play ---
        # One self-play stream per GPU. Each call is dispatched asynchronously,
        # so the streams (and the learner's SGD, queued behind its own stream
        # on the learner GPU) all run at the same time.
        actors = [(actor, cfg.games)]
        if actor != learner and cfg.learner_games > 0:
            actors.append((learner, cfg.learner_games))
        sp_states, sp_params = [], []
        for dev, n in actors:
            key, k = jax.random.split(key)
            sp_states.append(jax.device_put(jax.jit(jax.vmap(E.init))(jax.random.split(k, n)), dev))
            sp_params.append(jax.device_put(params, dev))
        total_games = sum(n for _, n in actors)
        assembler = RP.Assembler(total_games)
        new_per_iter = total_games * cfg.unroll
        calls = max(1, round(cfg.sample_reuse * new_per_iter / cfg.batch / chunk))
        games_done, t_last = 0, time.time()

        def ingest(records):
            """Host bookkeeping for one iteration's self-play records."""
            nonlocal games_done
            search_value = records.pop('search_value')
            samples = assembler.add(records)
            if samples is not None:
                replay.add(samples)
            done = records['done'] & records['valid']
            games_done += int(done.sum())
            rew = records['reward'][done]
            if rew.size:
                avg.add({'sp_decisive': float((rew != 0).mean()),
                         'sp_game_plies': float(records['ply'][done].mean()) + 1})
            avg.add({'sp_value_abs': float(np.abs(search_value).mean())})

        previous = None
        for it in range(start_iteration + 1, cfg.iterations + 1):
            t_iter = time.time()
            # Everything below up to device_get overlaps with the GPUs:
            # 1. launch self-play on every GPU (asynchronous),
            pending = []
            for i in range(len(actors)):
                key, k = jax.random.split(key)
                sp_states[i], rec = selfplay(sp_params[i], sp_states[i], k)
                pending.append(rec)
            # 2. queue SGD on the learner GPU, a chunk of steps per call,
            if replay.size >= cfg.min_replay:
                for _ in range(calls):
                    batches = jax.device_put(sample_stacked(replay, chunk, cfg.batch, np_rng), learner)
                    params, opt_state, m = train_steps_fn(params, opt_state, batches)
                    avg.add(m)
            # 3. and meanwhile file the previous iteration's games.
            if previous is not None:
                ingest(previous)

            # Wait for self-play, noting when each GPU finished (seconds since
            # this iteration's launch): shows which GPU is the bottleneck.
            t_wait = time.time()
            finished = [None] * len(pending)
            while any(f is None for f in finished):
                for i, rec in enumerate(pending):
                    if finished[i] is None and all(x.is_ready() for x in jax.tree.leaves(rec)):
                        finished[i] = time.time() - t_iter
                time.sleep(0.02)
            avg.add({'wait_s': time.time() - t_wait,
                     **{f'sp_gpu{dev.id}_s': f for (dev, _), f in zip(actors, finished)}})
            parts = jax.device_get(pending)
            previous = {k: np.concatenate([p[k] for p in parts], axis=1) for k in parts[0]}
            sp_params = [jax.device_put(params, dev) for dev, _ in actors]

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
