import jax
import jax.numpy as jnp
import numpy as np

from conftest import load_exercise
from training.zero import replay as RP

ex = load_exercise('ex04_values')

# The worked example from the README: P1 moves, P2 draws, P2 places, P1 wins.
REWARD = np.array([0, 0, 0, 1], np.float32)
DISCOUNT = np.array([-1, 1, -1, 0], np.float32)
EXPECTED = np.array([1, -1, -1, 1], np.float32)


def test_worked_example_numpy():
    np.testing.assert_array_equal(ex.value_targets_numpy(REWARD, DISCOUNT), EXPECTED)


def test_worked_example_jax():
    got = jax.jit(ex.value_targets_jax)(jnp.asarray(REWARD), jnp.asarray(DISCOUNT))
    np.testing.assert_array_equal(np.asarray(got), EXPECTED)


def test_timeout_gives_zero():
    reward = np.zeros(7, np.float32)
    discount = np.array([-1] * 6 + [0], np.float32)
    np.testing.assert_array_equal(ex.value_targets_numpy(reward, discount), 0)


def _random_game(rng):
    n = rng.integers(2, 60)
    discount = rng.choice([-1.0, 1.0], n, p=[0.8, 0.2]).astype(np.float32)
    discount[-1] = 0
    reward = np.zeros(n, np.float32)
    reward[-1] = rng.choice([1.0, 0.0])
    return reward, discount


def test_matches_replay_assembler():
    """Your targets must agree with the ones the real replay buffer computes."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        reward, discount = _random_game(rng)
        n = len(reward)
        game = {k: np.zeros((n,) + shape, dtype) for k, (shape, dtype) in RP.SAMPLE_SPEC.items() if k != 'z'}
        game.update(reward=reward, discount=discount, is_draw=np.zeros(n, bool))
        want = RP._game_samples(game)['z']
        np.testing.assert_allclose(ex.value_targets_numpy(reward, discount), want)
        np.testing.assert_allclose(np.asarray(ex.value_targets_jax(jnp.asarray(reward), jnp.asarray(discount))), want)
