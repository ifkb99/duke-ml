"""
ts_bridge round trips, from both players' perspectives:
  - states: env -> TS JSON -> env, also through the TS engine's (de)serializer;
  - actions: env action -> TS move -> env actions, for every legal action of
    every kind (move, strike, command, draw + place);
  - env actions converted to TS moves are legal in TS and lead to the same state.
The existing differential tests cover the other direction (TS move -> env).
"""

import json
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

from training.zero import env as E
from training.zero import rules as R
from training.zero import ts_bridge as B
from training.zero.tests.conftest import REPO, random_game, ts_oracle

FIELDS = ('tile', 'owner', 'side', 'bags', 'player', 'pending', 'ply', 'done', 'winner')


def _key(move):
    return json.dumps(move, sort_keys=True)


@pytest.fixture(scope='module')
def positions():
    """(decision state, [(place action, drawn tile, placing state)]) from random games."""
    out = []
    for seed in range(100, 116):
        prev = None
        for s, a, after, nxt in random_game(seed, max_steps=300, draw_bias=0.4):
            if int(s.pending) == E.NONE:
                prev = (jax.device_get(s), [])
                out.append(prev)
            else:
                # s is a placing state: every legal place, with the tile drawn.
                for p in np.flatnonzero(np.asarray(s.legal)):
                    prev[1].append((int(p), int(s.pending), jax.device_get(s)))
    return out


def test_tiles_json_matches_engine(tmp_path):
    """rules.py reads tiles.json; it must be what the engine exports today."""
    out = tmp_path / 'tiles.json'
    subprocess.run(['npx', 'tsx', 'packages/ai/src/zero/export-tiles.ts', str(out)],
                   cwd=REPO, check=True, capture_output=True)
    assert json.loads(out.read_text()) == json.loads((Path(R.__file__).parent / 'tiles.json').read_text())


def test_state_round_trip(positions):
    states = [s for s, _ in positions[::5]]
    assert {int(s.player) for s in states} == {0, 1}
    replies = ts_oracle([{'state': B.state_to_ts(s)} for s in states])
    for s, rep in zip(states, replies):
        for got in (B.state_arrays_from_ts(B.state_to_ts(s)), B.state_arrays_from_ts(rep['roundTrip'])):
            for f in FIELDS:
                np.testing.assert_array_equal(got[f], np.asarray(getattr(s, f)), err_msg=f)


def test_state_to_ts_rejects_mid_draw(positions):
    _, places = next(p for p in positions if p[1])
    with pytest.raises(ValueError):
        B.state_to_ts(places[0][2])


def test_action_round_trip_and_ts_legality(positions):
    sample = positions[::3]
    replies = ts_oracle([{'state': B.state_to_ts(s)} for s, _ in sample])
    kinds = set()
    for (s, places), rep in zip(sample, replies):
        player = int(s.player)
        duke = B.duke_square(s.tile, s.owner)
        ts_legal = {_key(m) for m in rep['legal']}
        ours = set()
        for a in np.flatnonzero(s.legal):
            a = int(a)
            if a == R.DRAW_ACTION:
                continue
            m = B.action_to_move(a, s)
            kinds.add(m['type'])
            assert B.move_to_actions(m, player, duke) == [('action', a)], (a, m)
            ours.add(_key(m))
        ts_board = {k for k in ts_legal if json.loads(k)['type'] != 'draw'}
        assert ours == ts_board
        assert bool(s.legal[R.DRAW_ACTION]) == any(json.loads(k)['type'] == 'draw' for k in ts_legal)

        for p, tile, placing in places:
            m = B.action_to_move(p, placing, tile)
            assert m['type'] == 'place' and m['tileName'] == R.TILE_NAMES[tile]
            assert B.move_to_actions(m, player, duke) == [
                ('action', R.DRAW_ACTION), ('chance', tile), ('action', p)]
            draw = _key({'type': 'draw', 'position': m['position']})
            assert draw in ts_legal
            kinds.add('place')
    assert kinds == {'move', 'strike', 'command', 'place'}


def test_env_actions_applied_in_ts_match_env(positions):
    """env action -> TS move -> TS applyMove gives the env's next state."""
    rng = np.random.default_rng(0)
    reqs, expected = [], []
    step_decision = jax.jit(E.step_decision)
    step_chance = jax.jit(E.step_chance)
    for s, places in positions[::2]:
        board = [int(a) for a in np.flatnonzero(s.legal) if a < R.DRAW_ACTION]
        for a in rng.choice(board, min(3, len(board)), replace=False) if board else []:
            reqs.append({'state': B.state_to_ts(s), 'move': B.action_to_move(int(a), s)})
            expected.append(jax.device_get(step_decision(s, int(a))))
        for p, tile, placing in places[:2]:
            reqs.append({'state': B.state_to_ts(s), 'move': B.action_to_move(p, placing, tile)})
            expected.append(jax.device_get(step_decision(placing, p)))
    replies = ts_oracle(reqs)
    n_timeouts = 0
    for req, rep, exp in zip(reqs, replies, expected):
        got = B.state_arrays_from_ts(rep['next'])
        if bool(exp.done) and int(exp.winner) == -1:
            # TS has no turn limit; the env ends the game at MAX_PLY unflipped.
            assert int(exp.ply) == E.MAX_PLY and got['ply'] == E.MAX_PLY
            n_timeouts += 1
            continue
        for f in FIELDS:
            np.testing.assert_array_equal(got[f], np.asarray(getattr(exp, f)), err_msg=f'{f} {req["move"]}')
