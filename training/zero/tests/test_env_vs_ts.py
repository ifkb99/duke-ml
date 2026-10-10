"""
Differential tests: the JAX environment must agree exactly with the TypeScript
engine (the reference implementation) on legal moves and transitions.

Fixtures are random-play positions dumped by
packages/ai/src/zero/dump-positions.ts. If the data directory is empty they are
generated automatically (16 games per process, in parallel).

Run:  uv run pytest training/zero/tests -q
Set DUKE_TEST_ALL=1 to check every dumped position instead of a sample.
"""

import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.zero import env as E
from training.zero import rules as R
from training.zero import ts_bridge as B

REPO = Path(__file__).resolve().parents[3]
DATA = Path(__file__).parent / 'data'
SAMPLE = None if os.environ.get('DUKE_TEST_ALL') else 6000


def _ensure_data():
    DATA.mkdir(exist_ok=True)
    if any(DATA.glob('positions-*.jsonl')):
        return
    procs = [
        subprocess.Popen(['npx', 'tsx', 'packages/ai/src/zero/dump-positions.ts', '16',
                          str(DATA / f'positions-{i}.jsonl')], cwd=REPO)
        for i in range(16)
    ]
    assert all(p.wait() == 0 for p in procs)


@pytest.fixture(scope='module')
def records():
    _ensure_data()
    recs = []
    for path in sorted(DATA.glob('positions-*.jsonl')):
        with path.open() as f:
            recs.extend(json.loads(line) for line in f)
    if SAMPLE and len(recs) > SAMPLE:
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in rng.choice(len(recs), SAMPLE, replace=False)]
    return recs


def chunked(fn, chunk=4096):
    """Apply a batched function in chunks to bound GPU memory."""
    def run(*args):
        n = jax.tree.leaves(args)[0].shape[0]
        parts = [fn(*jax.tree.map(lambda x: x[i:i + chunk], args)) for i in range(0, n, chunk)]
        return jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)
    return run


with_legal = chunked(jax.jit(jax.vmap(E._with_legal)))


@pytest.fixture(scope='module')
def batch(records):
    return with_legal(B.stack_states([B.state_arrays_from_ts(r['state']) for r in records]))


def _expected_actions(rec, player, duke_sq):
    out = set()
    for m in rec['legal']:
        steps = B.move_to_actions(m, player, duke_sq)
        out.add(steps[0][1])
    return out


def test_no_candidate_overflow(batch):
    assert int(batch.overflow.max()) == 0


def test_legal_moves_match(records, batch):
    legal = np.asarray(batch.legal)
    tile, owner = np.asarray(batch.tile), np.asarray(batch.owner)
    mismatches = []
    for i, rec in enumerate(records):
        player = int(batch.player[i])
        expected = _expected_actions(rec, player, B.duke_square(tile[i], owner[i]))
        got = set(np.flatnonzero(legal[i]).tolist())
        if got != expected:
            mismatches.append((i, sorted(expected - got)[:5], sorted(got - expected)[:5]))
    assert not mismatches, f'{len(mismatches)} positions differ, e.g. {mismatches[:3]}'


def test_draw_targets_match(records, batch):
    """After drawing any tile, the legal place squares are the TS draw targets."""
    idx = [i for i, r in enumerate(records) if r['drawTargets']]
    sub = jax.tree.map(lambda x: x[np.array(idx)], batch)
    # Reveal the first tile type still in the bag; legality is tile-independent.
    outcome = jnp.argmax(sub.bags[:, 0] > 0, axis=-1)
    after = chunked(jax.jit(jax.vmap(E.step_decision)))(sub, jnp.full(len(idx), R.DRAW_ACTION))
    placing = chunked(jax.jit(jax.vmap(E.step_chance)))(after, outcome)
    legal = np.asarray(placing.legal)
    tile, owner = np.asarray(sub.tile), np.asarray(sub.owner)
    for j, i in enumerate(idx):
        player = int(sub.player[j])
        duke_sq = B.duke_square(tile[j], owner[j])
        expected = {B.move_to_actions({'type': 'draw', 'position': p}, player, duke_sq)[1][1]
                    for p in records[i]['drawTargets']}
        assert set(np.flatnonzero(legal[j]).tolist()) == expected, f'record {i}'


def _apply_one(s, a1, chance, a2, has_a2):
    s1 = E.step_chance(E.step_decision(s, a1), chance)
    s2 = E.step_decision(s1, a2)
    return jax.tree.map(lambda x, y: jnp.where(has_a2, y, x), s1, s2)


_apply = chunked(jax.jit(jax.vmap(_apply_one)))


def test_transitions_match(records, batch):
    tile, owner = np.asarray(batch.tile), np.asarray(batch.owner)
    a1, ch, a2, has2 = [], [], [], []
    for i, rec in enumerate(records):
        steps = B.move_to_actions(rec['move'], int(batch.player[i]), B.duke_square(tile[i], owner[i]))
        a1.append(steps[0][1])
        ch.append(steps[1][1] if len(steps) == 3 else 0)
        a2.append(steps[2][1] if len(steps) == 3 else 0)
        has2.append(len(steps) == 3)
    got = _apply(batch, *(jnp.array(x) for x in (a1, ch, a2, has2)))
    expected = B.stack_states([B.state_arrays_from_ts(r['next']) for r in records])

    for field in ('tile', 'owner', 'side', 'bags', 'player', 'ply', 'done', 'winner', 'pending'):
        g, e = np.asarray(getattr(got, field)), np.asarray(getattr(expected, field))
        bad = np.flatnonzero((g != e).reshape(len(records), -1).any(-1))
        assert bad.size == 0, f'{field} differs in {bad.size} transitions, e.g. record {bad[0]}: ' \
                              f'move={records[bad[0]]["move"]}'
    assert not np.asarray(got.legal).any(axis=-1)[np.asarray(got.done)].any()
