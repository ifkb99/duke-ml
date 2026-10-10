"""Reference solution for lesson 6."""

import math

import numpy as np

ACTIONS = (1, 2, 3)


def legal_actions(stones: int) -> list[int]:
    return [i for i, take in enumerate(ACTIONS) if take <= stones]


def rollout(stones: int, rng: np.random.Generator) -> float:
    sign = 1.0
    while stones > 0:
        stones -= ACTIONS[rng.choice(legal_actions(stones))]
        if stones == 0:
            return sign
        sign = -sign
    return -1.0


class Node:
    def __init__(self, stones: int):
        self.stones = stones
        self.children: dict[int, 'Node'] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def puct_scores(prior, q, visits, c=1.25):
    total = max(visits.sum(), 1)
    return q + c * prior * math.sqrt(total) / (1 + visits)


def mcts(stones: int, n_simulations: int, rng: np.random.Generator, c: float = 1.25) -> np.ndarray:
    root = Node(stones)
    for _ in range(n_simulations):
        node, path = root, [root]

        # 1. Select
        while node.stones > 0 and len(node.children) == len(legal_actions(node.stones)):
            legal = legal_actions(node.stones)
            prior = np.zeros(len(ACTIONS))
            prior[legal] = 1.0 / len(legal)
            q = np.array([node.children[a].q if a in node.children else 0.0 for a in range(len(ACTIONS))])
            n = np.array([node.children[a].visits if a in node.children else 0 for a in range(len(ACTIONS))])
            scores = np.where(prior > 0, puct_scores(prior, q, n, c), -np.inf)
            node = node.children[int(np.argmax(scores))]
            path.append(node)

        # 2. Expand
        if node.stones > 0:
            a = next(a for a in legal_actions(node.stones) if a not in node.children)
            child = Node(node.stones - ACTIONS[a])
            node.children[a] = child
            node = child
            path.append(node)

        # 3. Evaluate, from the point of view of the player who moved into `node`
        value = 1.0 if node.stones == 0 else -rollout(node.stones, rng)

        # 4. Backup, flipping perspective at each level
        for n in reversed(path):
            n.visits += 1
            n.value_sum += value
            value = -value

    return np.array([root.children[a].visits if a in root.children else 0 for a in range(len(ACTIONS))])
