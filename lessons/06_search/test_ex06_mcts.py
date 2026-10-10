import numpy as np
import pytest

from conftest import load_exercise

ex = load_exercise('ex06_mcts')


def test_puct_prefers_unvisited_with_prior():
    prior = np.array([0.5, 0.5, 0.0])
    q = np.array([0.1, 0.0, 0.0])
    visits = np.array([10, 0, 0])
    s = ex.puct_scores(prior, q, visits, c=1.25)
    assert s.shape == (3,)
    assert np.isclose(s[0], 0.1 + 1.25 * 0.5 * np.sqrt(10) / 11)
    assert np.isclose(s[1], 1.25 * 0.5 * np.sqrt(10) / 1)
    assert s[1] > s[0]


def test_puct_first_visit_uses_prior():
    s = ex.puct_scores(np.array([0.2, 0.8, 0.0]), np.zeros(3), np.zeros(3), c=1.0)
    assert np.argmax(s) == 1


@pytest.mark.parametrize('stones, winning_take', [(5, 1), (6, 2), (7, 3), (9, 1), (10, 2), (11, 3)])
def test_finds_winning_move(stones, winning_take):
    """Leaving a multiple of 4 wins; the search should discover that."""
    visits = ex.mcts(stones, 3000, np.random.default_rng(0))
    assert visits.shape == (3,)
    assert visits.sum() == 3000
    assert np.argmax(visits) == winning_take - 1, f'visits={visits}'


def test_takes_the_last_stones():
    visits = ex.mcts(3, 500, np.random.default_rng(1))
    assert np.argmax(visits) == 2          # take all 3 and win immediately


def test_only_legal_moves_with_few_stones():
    visits = ex.mcts(2, 200, np.random.default_rng(2))
    assert visits[2] == 0                  # can't take 3 from 2
