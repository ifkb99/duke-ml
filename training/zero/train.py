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
       value  loss = (network value - value target)^2, where the target is
                     the n-step bootstrapped return (`value_nstep`, see
                     replay.py) or, with value_nstep=0, the final result
  3. Finished games go into the replay buffer; the actor gets the new weights.
Every `eval_every` iterations: matches vs random and TypeScript minimax,
checkpoint, metrics line. Everything is logged to runs/<run>/metrics.jsonl.

Before self-play starts, the network is warm-started by imitating minimax:
we play minimax-vs-minimax games and train the policy head to predict
minimax's move. It gives self-play a sensible starting point instead of
random moves. 10% of the games are held out, and the warm-start stops when
the loss on them stops improving. The value head is NOT trained on these
games by default (`imitation_value_weight=0`): a few thousand game results
are easy to memorize, and a memorized value head is confidently wrong on
everything else (see EXPERIMENTS.md, az1). It starts at exactly 0 and learns
from self-play.

Diagnostics logged to metrics.jsonl (kind='train'), measured on each
iteration's new games *before* training on them, so they are honest
held-out numbers:
  fresh_value_corr_mc  correlation of the value head with real game results
  fresh_policy_kl      how far search targets are from the network's policy
  fresh_*_entropy      sharpness of the policy and of the search targets
Every `ladder_every` iterations the network also plays the network from
`ladder_every` iterations earlier (kind='ladder'): above 0.5 = it improved.

Resuming: rerunning with the same --run continues from runs/<run>/latest.pkl
(the replay buffer is not saved; it refills from fresh self-play).
"""

import argparse
import json
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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
    policy_head: str = 'dense'     # 'spatial': ~21K-parameter head instead of ~4.6M (network.py)
    # self-play
    games: int = 872               # parallel self-play games on the actor GPU
    learner_games: int = 792       # more self-play games on the learner GPU, which
                                   # would otherwise idle between SGD steps (0 = off);
                                   # balanced so both GPUs finish together (sp_gpu*_s)
    unroll: int = 32               # moves per game per iteration
    simulations: int = 32
    max_considered: int = 16
    max_depth: int = 16            # tree edges per simulation (2 per move: decision + chance)
    # learning
    batch: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4
    sample_reuse: float = 1.0      # each new sample is trained on ~this many times
                                   # (az1 used 4 and the value head memorized the buffer)
    steps_per_call: int = 32       # SGD steps per jitted call (amortizes launch overhead)
    replay: int = 2_000_000
    min_replay: int = 100_000
    value_nstep: int = 10          # value target: n-step bootstrapped return (0 = final result)
    # imitation warm-start
    imitation_games: int = 3000
    imitation_depths: list = field(default_factory=lambda: [2, 3, 4])
    imitation_epsilon: float = 0.1
    imitation_steps: int = 4000    # maximum; stops early on held-out loss
    imitation_holdout: float = 0.1 # fraction of minimax games held out
    imitation_check_every: int = 128
    imitation_patience: int = 3    # checks without improvement before stopping
    imitation_value_weight: float = 0.0  # 0 = warm-start the policy only: a value head
                                   # fit to 3k minimax games just memorizes them (az1/az2)
    # evaluation / bookkeeping
    iterations: int = 100_000
    eval_every: int = 25
    eval_games: int = 64
    eval_depths: list = field(default_factory=lambda: [2, 4])
    slow_eval_depths: list = field(default_factory=lambda: [6])  # deeper minimax, evaluated
    slow_eval_every: int = 100     # less often since it's slow (~2 min per 64 games at depth 6)
    workers: int = 0               # minimax processes (0 = one per CPU thread)
    ladder_every: int = 100        # play the network from this many iterations ago (0 = off)
    ladder_games: int = 64
    diag_samples: int = 4096       # new samples per iteration used for fresh_* metrics


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def make_loss_fn(forward, value_weight: float = 1.0):
    """loss_fn(params, batch) -> (policy loss + value_weight * value loss, metrics)."""
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
        return policy_loss + value_weight * value_loss, metrics

    return loss_fn


def make_train_step(forward, tx, value_weight: float = 1.0):
    grad_fn = jax.value_and_grad(make_loss_fn(forward, value_weight), has_aux=True)

    @jax.jit
    def train_step(params, opt_state, batch):
        (_, metrics), grads = grad_fn(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        metrics['grad_norm'] = optax.global_norm(grads)
        return optax.apply_updates(params, updates), opt_state, metrics

    return train_step


def make_diagnostics(forward):
    """diag(params, batch) -> dict of scalar metrics on samples not yet trained on."""
    legal_actions = jax.vmap(E.legal_actions)
    observe = jax.vmap(E.observe_arrays)

    def corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        return jnp.sum(x * y) / jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y) + 1e-12)

    @jax.jit
    def diag(params, b):
        tile, owner, side = (b[k].astype(jnp.int32) for k in ('tile', 'owner', 'side'))
        bags, pending = b['bags'].astype(jnp.int32), b['pending'].astype(jnp.int32)
        legal, _, _ = legal_actions(tile, owner, side, bags, pending)
        logits, value = forward(params, *observe(tile, owner, side, bags, pending, b['ply']))
        logp = jax.nn.log_softmax(jnp.where(legal, logits.astype(jnp.float32), -1e9))
        n = tile.shape[0]
        target = jnp.zeros((n, R.N_ACTIONS)).at[jnp.arange(n)[:, None], b['pol_idx'].astype(jnp.int32)] \
            .add(b['pol_p'].astype(jnp.float32))
        hp = b['has_policy'].astype(jnp.float32)
        mean_p = lambda x: jnp.sum(x * hp) / jnp.maximum(hp.sum(), 1)
        prior = jnp.exp(logp)
        xlogx = lambda p: jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-30)), 0.0)
        ent_t = -jnp.sum(xlogx(target), -1)
        ce = -jnp.sum(target * jnp.where(legal, logp, 0.0), -1)
        z, z_mc = b['z'], b['z_mc']
        return {
            'fresh_value_mse': jnp.mean((value - z) ** 2),
            'fresh_value_mse_mc': jnp.mean((value - z_mc) ** 2),
            'fresh_mc_baseline': jnp.mean(z_mc ** 2),          # MSE of always predicting 0
            'fresh_value_corr_mc': corr(value, z_mc),
            'fresh_target_corr_mc': corr(z, z_mc),             # n-step target vs real result
            'fresh_policy_ce': mean_p(ce),
            'fresh_policy_kl': mean_p(ce - ent_t),
            'fresh_target_entropy': mean_p(ent_t),
            'fresh_prior_entropy': mean_p(-jnp.sum(jnp.where(legal, prior * logp, 0.0), -1)),
            'fresh_top1_agree': mean_p((jnp.argmax(logp, -1) == jnp.argmax(target, -1)).astype(jnp.float32)),
        }

    return diag


def make_train_steps(forward, tx, value_weight: float = 1.0):
    """Several SGD steps in ONE jitted call: train_steps(params, opt, batches).

    `batches` has a leading [n_steps] axis. Launching a jitted function costs
    ~12 ms of Python here (walking the big params/optimizer pytrees), which made
    one-call-per-step training launch-bound; scanning over steps on the device
    pays that once per call. Returns metrics averaged over the steps.
    """
    step = make_train_step(forward, tx, value_weight)

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
    model = DukeZeroNet(c['channels'], c['blocks'], c.get('policy_head', 'dense'), rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    nnx.replace_by_pure_dict(params, ckpt['params'])
    return graphdef, params, ckpt


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def imitation_data(cfg, pool, key, log, cache: Path):
    """Minimax-vs-minimax games -> samples with one-hot policy targets.

    Saved to `cache` so a restart doesn't replay all the minimax games."""
    if cache.exists():
        with np.load(cache) as f:
            samples = {k: f[k] for k in f.files}
        samples.setdefault('z_mc', samples['z'])      # caches from before z_mc existed
        log(f'imitation: loaded {len(samples["z"])} samples from {cache}')
        return samples
    replay = RP.Replay(cfg.imitation_games * 400)
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
    samples = {k: v[:replay.size] for k, v in replay.data.items()}
    np.savez(cache.with_suffix('.tmp.npz'), **samples)
    cache.with_suffix('.tmp.npz').replace(cache)
    return samples


def game_ids(samples) -> np.ndarray:
    """Game index of each sample. Samples are stored game by game, and every
    game starts with its ply-0 decision (draw-afterstate extras come last)."""
    start = (samples['ply'] == 0) & (samples['pending'] == E.NONE)
    return np.cumsum(start) - 1


def imitate(cfg, samples, params, opt_state, train_steps_fn, loss_eval, learner, np_rng, log):
    """Supervised warm-start with early stopping on held-out games."""
    gid = game_ids(samples)
    held = gid >= int(round((gid.max() + 1) * (1 - cfg.imitation_holdout)))
    train = RP.Replay(int((~held).sum()))
    train.add({k: v[~held] for k, v in samples.items()})
    test = {k: v[held] for k, v in samples.items()}
    log(f'imitation: {train.size} training samples, {int(held.sum())} held out '
        f'({int(gid.max() + 1 - gid[held].min())} games)')

    def held_out():
        # Chunked: the dense policy targets of all held-out samples at once
        # would take a lot of GPU memory. (Policy loss is averaged per chunk,
        # which weights chunks equally; close enough for early stopping.)
        ms = [jax.device_get(loss_eval(params, {k: v[i:i + 4096] for k, v in test.items()}))
              for i in range(0, len(test['z']), 4096)]
        return (float(np.mean([m['policy_loss'] for m in ms])),
                float(np.mean([m['value_loss'] for m in ms])))

    chunk = cfg.steps_per_call
    best = (float('inf'), params, opt_state, 0)
    bad = 0
    avg = MetricAverager()
    for step in range(chunk, cfg.imitation_steps + 1, chunk):
        batches = jax.device_put(sample_stacked(train, chunk, cfg.batch, np_rng), learner)
        params, opt_state, m = train_steps_fn(params, opt_state, batches)
        avg.add(m)
        if step % cfg.imitation_check_every == 0:
            tr = avg.pop()
            pl, vl = held_out()
            log(f'imitation step {step}: train policy {tr["policy_loss"]:.3f} value {tr["value_loss"]:.3f} | '
                f'held-out policy {pl:.3f} value {vl:.3f}',
                kind='imitation', step=step, epoch=step * cfg.batch / train.size,
                train_policy_loss=tr['policy_loss'], train_value_loss=tr['value_loss'],
                heldout_policy_loss=pl, heldout_value_loss=vl)
            score = pl + cfg.imitation_value_weight * vl
            if score < best[0] - 1e-3:
                best, bad = (score, params, opt_state, step), 0
            else:
                bad += 1
                if bad >= cfg.imitation_patience:
                    break
    log(f'imitation: keeping step {best[3]} (held-out loss {best[0]:.3f})')
    return best[1], best[2]


def git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=RUNS.parent,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ''


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

    model = DukeZeroNet(cfg.channels, cfg.blocks, cfg.policy_head, rngs=nnx.Rngs(cfg.seed))
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
        shape = (ckpt['config']['channels'], ckpt['config']['blocks'], ckpt['config'].get('policy_head', 'dense'))
        if shape != (cfg.channels, cfg.blocks, cfg.policy_head):
            log(f'note: using the checkpoint network {shape}, not --channels/--blocks/--policy-head')
        log(f'resumed from {resume_from} at iteration {start_iteration}')
    else:
        (run_dir / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    log(f'start (commit {git_commit()})', kind='start', commit=git_commit(),
        resumed_from=str(resume_from) if resume_from else None,
        iteration=start_iteration, gpus=len(devices), config=asdict(cfg))
    params = jax.device_put(params, learner)
    opt_state = jax.device_put(opt_state, learner)

    train_steps_fn = make_train_steps(forward, tx)
    loss_fn = make_loss_fn(forward)
    loss_eval = jax.jit(lambda p, b: loss_fn(p, b)[1])
    diagnostics = make_diagnostics(forward)
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
            samples = imitation_data(cfg, pool, k, log, run_dir / 'imitation_data.npz')
            params, opt_state = imitate(cfg, samples, params, opt_state,
                                        make_train_steps(forward, tx, cfg.imitation_value_weight),
                                        loss_eval, learner, np_rng, log)
            key, k = jax.random.split(key)
            results = evaluate(cfg, search, params, pool, k)
            log('after imitation: ' + ', '.join(f"{r['b']} {r['score']:.2f}" for r in results),
                kind='eval', iteration=0, results=results)
            save_checkpoint(run_dir / 'imitation.pkl', params, opt_state, 0, cfg)
        # Self-play starts from an empty buffer: the minimax games have done
        # their job, and their value targets are plain game results.
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
        assembler = RP.Assembler(total_games, nstep=cfg.value_nstep)
        ladder_params, ladder_iteration = jax.device_get(params), start_iteration
        new_per_iter = total_games * cfg.unroll
        calls = max(1, round(cfg.sample_reuse * new_per_iter / cfg.batch / chunk))
        games_done, t_last = 0, time.time()

        def ingest(records):
            """Host bookkeeping for one iteration's self-play records."""
            nonlocal games_done
            search_value = records['search_value']
            samples = assembler.add(records)
            if samples is not None:
                # Measured before these games are trained on: honest held-out numbers.
                n = min(cfg.diag_samples, len(samples['z']))
                idx = np_rng.choice(len(samples['z']), n, replace=False)
                avg.add(diagnostics(params, jax.device_put({k: v[idx] for k, v in samples.items()}, learner)))
                replay.add(samples)
            done = records['done'] & records['valid']
            games_done += int(done.sum())
            rew = records['reward'][done]
            if rew.size:
                avg.add({'sp_decisive': float((rew != 0).mean()),
                         'sp_game_plies': float(records['ply'][done].mean()) + 1})
            avg.add({'sp_value_abs': float(np.abs(search_value).mean())})

        # One thread per self-play GPU. The search's while_loops have
        # data-dependent exit conditions, which XLA:GPU evaluates on the host,
        # so launching self-play keeps the calling thread busy until it's done.
        # From a single thread the GPUs therefore took turns (23.7 s/iteration);
        # one thread each lets them run concurrently (13.6 s).
        sp_threads = ThreadPoolExecutor(len(actors), thread_name_prefix='selfplay')

        def run_selfplay(i, k, t0):
            new_states, rec = jax.block_until_ready(selfplay(sp_params[i], sp_states[i], k))
            return new_states, rec, time.time() - t0

        previous = None
        for it in range(start_iteration + 1, cfg.iterations + 1):
            t_iter = time.time()
            # Everything below up to collecting the futures overlaps with the GPUs:
            # 1. self-play on every GPU, each from its own thread,
            futures = []
            for i in range(len(actors)):
                key, k = jax.random.split(key)
                futures.append(sp_threads.submit(run_selfplay, i, k, t_iter))
            # 2. queue SGD on the learner GPU, a chunk of steps per call,
            if replay.size >= cfg.min_replay:
                for _ in range(calls):
                    batches = jax.device_put(sample_stacked(replay, chunk, cfg.batch, np_rng), learner)
                    params, opt_state, m = train_steps_fn(params, opt_state, batches)
                    avg.add(m)
            # 3. and meanwhile file the previous iteration's games.
            if previous is not None:
                ingest(previous)

            # Collect self-play; note when each GPU finished (seconds since this
            # iteration started) to show which one is the bottleneck.
            t_wait = time.time()
            results = [f.result() for f in futures]
            for i, (new_states, _, _) in enumerate(results):
                sp_states[i] = new_states
            avg.add({'wait_s': time.time() - t_wait,
                     **{f'sp_gpu{dev.id}_s': r[2] for (dev, _), r in zip(actors, results)}})
            parts = jax.device_get([r[1] for r in results])
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

            if cfg.slow_eval_every and it % cfg.slow_eval_every == 0 and cfg.slow_eval_depths:
                net = AR.Agent('net', params=params, search=search)
                results = []
                for d in cfg.slow_eval_depths:
                    key, k = jax.random.split(key)
                    res, _ = AR.play(net, AR.Agent('minimax', name=f'minimax{d}', depth=d),
                                     cfg.eval_games, k, pool=pool)
                    results.append(res)
                log(f'eval {it}: ' + ', '.join(f"{r['b']} {r['score']:.2f}" for r in results),
                    kind='eval', iteration=it, results=results)
                t_last = time.time()

            if cfg.ladder_every and it % cfg.ladder_every == 0:
                # Self-improvement check that doesn't depend on minimax.
                key, k = jax.random.split(key)
                old = AR.Agent('net', name=f'iter{ladder_iteration}',
                               params=jax.device_put(ladder_params, learner), search=search)
                res, _ = AR.play(AR.Agent('net', name=f'iter{it}', params=params, search=search),
                                 old, cfg.ladder_games, k)
                log(f'ladder {it} vs {ladder_iteration}: {res["score"]:.2f}',
                    kind='ladder', iteration=it, opponent=ladder_iteration, results=res)
                ladder_params, ladder_iteration = jax.device_get(params), it
                t_last = time.time()
    finally:
        if 'sp_threads' in locals():
            sp_threads.shutdown(wait=False, cancel_futures=True)
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
