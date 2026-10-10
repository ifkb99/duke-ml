"""
Distill a trained network (the teacher) into a new one (the student), e.g. to
swap in the small spatial policy head or shrink the network for the browser.

    python -m training.zero.distill runs/az3-net/latest.pkl --run az3-spatial \\
        --policy-head spatial --init teacher

1. The teacher plays itself, by default with the same search as training (32
   simulations); --simulations searches deeper, so the policy and value
   targets are stronger than the teacher's own training data. The samples
   (search policy + n-step value targets) are cached in
   runs/<run>/distill_samples.npz, so other students can reuse them.
2. The student learns them with early stopping on held-out games
   (train.imitate, policy and value).
3. The teacher's own losses on the same held-out games are the bar to reach.
4. The student is saved as runs/<run>/imitation.pkl: `run.sh <run> ...` then
   continues it with self-play RL, skipping the minimax warm-start.

Data on several GPUs: run one `--samples-only` job per GPU (different --run
and --seed), then train with `--data a.npz b.npz`, which concatenates them.

--init teacher copies every trunk and value-head weight from the teacher and
starts only the policy head fresh, so a dense student is the fair control for
a spatial one: same data, same steps, only the head differs.

--init <checkpoint> starts from another trained network of the same width,
e.g. the current phone network, to teach it a stronger teacher's play. With
more --blocks than it has, the extra blocks start as the identity:

    python -m training.zero.distill runs/big1/ckpt_001525.pkl --run az6-12 \
        --channels 96 --blocks 12 --init runs/az5/ckpt_001600.pkl --data ...
"""

import argparse
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from training.zero import env as E
from training.zero import replay as RP
from training.zero import search as S
from training.zero import selfplay as SP
from training.zero import train as T
from training.zero.network import DukeZeroNet, make_forward


def teacher_games(forward, params, cfg, n_samples, games, key, log, chunks=1):
    """Self-play samples from the teacher, in whole games. chunks: run each
    search over the games in this many pieces (deep trees take memory)."""
    search = S.make_search(forward, cfg.simulations, cfg.max_considered, cfg.max_depth)
    selfplay = SP.make_selfplay(S.chunked(search, chunks), cfg.unroll)
    key, k = jax.random.split(key)
    states = jax.jit(jax.vmap(E.init))(jax.random.split(k, games))
    asm, parts, n, t0 = RP.Assembler(games, nstep=cfg.value_nstep), [], 0, time.time()
    while n < n_samples:
        key, k = jax.random.split(key)
        states, rec = selfplay(params, states, k)
        out = asm.add(jax.device_get(rec))
        if out is not None:
            parts.append(out)
            n += len(out['z'])
        log(f'teacher self-play: {n} samples ({n / (time.time() - t0):.0f}/s)')
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def copy_from_teacher(student, teacher):
    """Student params with every non-policy weight taken from the teacher."""
    s, t = nnx.to_pure_dict(student), nnx.to_pure_dict(teacher)
    for name in s:
        if not name.startswith('policy_'):
            assert jax.tree.structure(s[name]) == jax.tree.structure(t[name]), name
            s[name] = t[name]
    return s


def copy_from_checkpoint(student, params):
    """Student params with every weight taken from a trained network of the same
    width (`params`, a checkpoint's). A deeper student keeps its extra residual
    blocks fresh, appended after the trained ones; a fresh block starts as the
    identity (its last conv is zero), so the student plays exactly as the
    checkpoint does until it learns to use them."""
    s = nnx.to_pure_dict(student)
    assert len(params['blocks']) <= len(s['blocks']), 'the student must be at least as deep'
    for name in s:
        if name != 'blocks':
            assert jax.tree.structure(s[name]) == jax.tree.structure(params[name]), name
            s[name] = params[name]
    for i in s['blocks']:
        if i in params['blocks']:
            s['blocks'][i] = params['blocks'][i]
        else:
            assert not np.any(s['blocks'][i]['conv2']['kernel']), 'a fresh block must start as the identity'
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('teacher', type=Path)
    p.add_argument('--run', required=True, help='student run directory under runs/')
    p.add_argument('--channels', type=int, help='default: the teacher\'s')
    p.add_argument('--blocks', type=int, help='default: the teacher\'s')
    p.add_argument('--policy-head', default='spatial')
    p.add_argument('--init', default='teacher',
                   help="'teacher', 'fresh', or a checkpoint of the same width to start from")
    p.add_argument('--samples', type=int, default=400_000, help='teacher self-play samples to generate')
    p.add_argument('--games', type=int, default=512, help='parallel self-play games')
    p.add_argument('--simulations', type=int, help='teacher search simulations (default: its training\'s)')
    p.add_argument('--chunks', type=int, default=1, help='split each teacher search over the games')
    p.add_argument('--mirror', type=float, default=0.0, help='chance each training sample is flipped')
    p.add_argument('--data', type=Path, nargs='+', help='reuse samples (distill_samples.npz files, concatenated)')
    p.add_argument('--samples-only', action='store_true', help='generate the teacher samples, then stop')
    p.add_argument('--steps', type=int, default=20_000, help='maximum SGD steps (stops early)')
    p.add_argument('--check-every', type=int, default=256, help='SGD steps between held-out checks')
    p.add_argument('--patience', type=int, default=4, help='held-out checks without improvement before stopping')
    p.add_argument('--lr-drops', type=int, default=0,
                   help='on a plateau, go back to the best step and cut the lr (x --lr-factor) this many times before stopping')
    p.add_argument('--lr-factor', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    run_dir = T.RUNS / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = (run_dir / 'metrics.jsonl').open('a')

    def log(msg, **record):
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
        if record:
            metrics.write(json.dumps({'time': time.time(), **record}) + '\n')
            metrics.flush()

    t_graphdef, t_params, t_ckpt = T.load_params(a.teacher)
    t_forward = make_forward(t_graphdef)
    t_cfg = T.Config(**{k: v for k, v in t_ckpt['config'].items() if k in T.Config.__dataclass_fields__})
    cfg = replace(t_cfg, run=a.run, seed=a.seed, lr=a.lr,
                  channels=a.channels or t_cfg.channels, blocks=a.blocks or t_cfg.blocks,
                  policy_head=a.policy_head, imitation_steps=a.steps, imitation_value_weight=1.0,
                  imitation_check_every=min(a.check_every, a.steps), imitation_patience=a.patience,
                  imitation_lr_drops=a.lr_drops, imitation_lr_factor=a.lr_factor,
                  mirror=a.mirror)
    log(f'start (commit {T.git_commit()})', kind='start', commit=T.git_commit(), distill=True,
        teacher=str(a.teacher), teacher_iteration=t_ckpt['iteration'], init=a.init,
        teacher_simulations=a.simulations or t_cfg.simulations,
        gpus=len(jax.devices()), config=asdict(cfg))
    (run_dir / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))

    key = jax.random.key(a.seed)
    data = a.data or [run_dir / 'distill_samples.npz']
    if all(d.exists() for d in data):
        parts = [dict(np.load(d)) for d in data]
        samples = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
        log(f'samples: {len(samples["z"])} from {", ".join(map(str, data))}')
    else:
        data = data[0]
        key, k = jax.random.split(key)
        t_play = replace(t_cfg, simulations=a.simulations or t_cfg.simulations)
        log(f'teacher self-play: {a.games} games at {t_play.simulations} simulations')
        samples = teacher_games(t_forward, t_params, t_play, a.samples, a.games, k, log, a.chunks)
        np.savez(data, **samples)
        log(f'samples: {len(samples["z"])} saved to {data}', kind='distill_data',
            samples=len(samples['z']), games=int(T.game_ids(samples).max() + 1))
    if a.samples_only:
        return

    student = DukeZeroNet(cfg.channels, cfg.blocks, cfg.policy_head, rngs=nnx.Rngs(a.seed))
    graphdef, params = nnx.split(student)
    if a.init == 'teacher':
        assert (cfg.channels, cfg.blocks) == (t_cfg.channels, t_cfg.blocks), '--init teacher needs the same trunk'
        nnx.replace_by_pure_dict(params, copy_from_teacher(params, t_params))
    elif a.init != 'fresh':
        with open(a.init, 'rb') as f:
            init_params = pickle.load(f)['params']
        nnx.replace_by_pure_dict(params, copy_from_checkpoint(params, init_params))
    forward = make_forward(graphdef)
    tx = T.make_optimizer(cfg)
    opt_state = tx.init(params)
    n_params = sum(np.size(x) for x in jax.tree.leaves(params))
    log(f'student {cfg.channels}x{cfg.blocks} {cfg.policy_head}: {n_params / 1e6:.2f}M parameters, init {a.init}')

    # The bar: the teacher on the held-out games (the split imitate uses).
    gid = T.game_ids(samples)
    held = gid >= int(round((gid.max() + 1) * (1 - cfg.imitation_holdout)))
    test = {k: v[held] for k, v in samples.items()}
    t_eval = jax.jit(lambda prm, b: T.make_loss_fn(t_forward)(prm, b)[1])
    t_diag = T.make_diagnostics(t_forward)

    def held_out(eval_fn, prm):
        ms = [jax.device_get(eval_fn(prm, {k: v[i:i + 4096] for k, v in test.items()}))
              for i in range(0, len(test['z']), 4096)]
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    tm = held_out(t_eval, t_params)
    log(f'teacher on held-out games: policy {tm["policy_loss"]:.3f} (top-1 {tm["policy_top1"]:.3f}) '
        f'value {tm["value_loss"]:.3f}', kind='distill_teacher', **{f'heldout_{k}': v for k, v in tm.items()})

    learner = jax.devices()[0]
    params, opt_state = T.imitate(cfg, samples, jax.device_put(params, learner), opt_state,
                                  T.make_train_steps(forward, tx), jax.jit(lambda prm, b: T.make_loss_fn(forward)(prm, b)[1]),
                                  learner, np.random.default_rng(a.seed), log)

    sm = held_out(jax.jit(lambda prm, b: T.make_loss_fn(forward)(prm, b)[1]), params)
    sd = held_out(T.make_diagnostics(forward), params)
    td = held_out(t_diag, t_params)
    log(f'student on held-out games: policy {sm["policy_loss"]:.3f} (top-1 {sm["policy_top1"]:.3f}) '
        f'value {sm["value_loss"]:.3f} | teacher {tm["policy_loss"]:.3f} ({tm["policy_top1"]:.3f}) '
        f'{tm["value_loss"]:.3f}', kind='distill_result',
        **{f'student_{k}': v for k, v in {**sm, **sd}.items()},
        **{f'teacher_{k}': v for k, v in {**tm, **td}.items()})
    T.save_checkpoint(run_dir / 'imitation.pkl', params, opt_state, 0, cfg)
    log(f'saved {run_dir / "imitation.pkl"}')


if __name__ == '__main__':
    main()
