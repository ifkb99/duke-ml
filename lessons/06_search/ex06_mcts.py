"""Lesson 6 exercise: a small MCTS for Nim (take 1-3 stones; taking the last one wins)."""

import math

import numpy as np

ACTIONS = (1, 2, 3)


def legal_actions(stones: int) -> list[int]:
    """Indices into ACTIONS that are legal with `stones` left."""
    return [i for i, take in enumerate(ACTIONS) if take <= stones]


def rollout(stones: int, rng: np.random.Generator) -> float:
    """Play random moves to the end. Returns the result for the player TO MOVE
    at `stones`: +1 if they end up taking the last stone, -1 otherwise.
    With 0 stones the player to move has already lost (the opponent took the last one)."""
    sign = 1.0
    while stones > 0:
        stones -= ACTIONS[rng.choice(legal_actions(stones))]
        if stones == 0:
            return sign         # the player who just moved won
        sign = -sign
    return -1.0


class Node:
    """One position (`stones` left, someone to move). Values are stored from
    the point of view of the player to move AT THIS NODE's PARENT, i.e. the
    player who chose the move leading here. That's what the parent needs when
    comparing its children."""

    def __init__(self, stones: int):
        self.stones = stones
        self.children: dict[int, 'Node'] = {}   # action index -> child
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def puct_scores(prior, q, visits, c=1.25):
    """PUCT score per action: q + c * prior * sqrt(sum(visits)) / (1 + visits).

    prior, q, visits: np.ndarray [A]. Returns np.ndarray [A].
    (For a node never visited before, sum(visits) is 0. Use sqrt(max(sum, 1))
    so the prior still breaks ties on the very first selection.)
    """
    raise NotImplementedError


def mcts(stones: int, n_simulations: int, rng: np.random.Generator, c: float = 1.25) -> np.ndarray:
    """Run MCTS from a position with `stones` left. Returns root visit counts [3].

    Each simulation:
      1. Select: from the root, while the node is non-terminal and every legal
         action already has a child, move to the child with the best
         puct_scores (uniform prior over LEGAL actions, -inf for illegal ones).
      2. Expand: if the node is non-terminal, create the child for one untried
         legal action (any choice works, e.g. the first untried one).
      3. Evaluate the new node (or the terminal node you stopped at) from the
         point of view of the player who MOVED INTO it:
           - stones == 0: that player took the last stone -> +1
           - otherwise: -rollout(child.stones)  (rollout is from the
             perspective of the player to move there, the opponent)
      4. Backup: walk back up the path. Add the value to each node, flipping
         the sign at every step up (the parent of a node is the other player).

    Hint: keep the path as a list of nodes from the root down.
    """
    raise NotImplementedError
