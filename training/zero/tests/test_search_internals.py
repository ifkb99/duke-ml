"""
search.py replaces three mctx internals for speed: the backward pass
(`_backward`), the stochastic action selection (`_row_view` +
`_make_selection`), and the simulation loop (`_run_search`). These tests check
each one against the stock mctx code it replaces:

  - on randomly built trees (all path lengths, interior "leaves" as happen on
    max-depth revisits, decision and chance nodes, depth 0 and deeper);
  - on whole searches over the real environment, where the final trees must
    be identical.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mctx._src import action_selection, base as mctx_base, qtransforms, search as mctx_search
from mctx._src.policies import _make_stochastic_action_selection_fn
import mctx

from training.zero import env as E
from training.zero import rules as R
from training.zero import search as S
from training.zero.network import observe_batch
from training.zero.tests.conftest import stack

A, C = S.A, S.C
NEG = S.NEG
QT = qtransforms.qtransform_completed_by_mix_value


# ---------------------------------------------------------------------------
# Random trees
# ---------------------------------------------------------------------------

def random_tree(rng, num_simulations, n_used, chain=0.5, legal_per_node=12):
    """An unbatched, type-consistent stochastic tree with `n_used` nodes.

    Node 0 is a decision root; a decision node's children (actions < A) are
    chance nodes, a chance node's children (actions >= A) are decision nodes.
    Statistics are random but plausible: visits are subtree sizes, values in
    [-1, 1], discounts in {-1, 0, 1} below chance nodes.
    """
    n = num_simulations + 1
    width = A + C
    parents = np.full(n, -1, np.int32)
    afp = np.full(n, -1, np.int32)
    children = np.full((n, width), -1, np.int32)
    is_decision = np.zeros(n, bool)
    is_decision[0] = True

    prior = np.full((n, width), NEG, np.float32)
    node_legal = {}

    def setup(i):
        if is_decision[i]:
            acts = rng.choice(A, legal_per_node, replace=False)
            prior[i, acts] = rng.normal(size=legal_per_node)
        else:
            k = rng.integers(1, C + 1)
            acts = A + rng.choice(C, k, replace=False)
            p = rng.dirichlet(np.ones(k))
            prior[i, acts] = np.log(p)
        node_legal[i] = list(acts)

    setup(0)
    for i in range(1, n_used):
        candidates = [j for j in range(i) if node_legal[j]]
        p = i - 1 if (rng.random() < chain and node_legal[i - 1]) else int(rng.choice(candidates))
        a = node_legal[p].pop(rng.integers(len(node_legal[p])))
        parents[i], afp[i], children[p, a] = p, a, i
        is_decision[i] = not is_decision[p]
        setup(i)

    # Visits = subtree sizes (each node counts its own expansion).
    visits = np.zeros(n, np.int32)
    visits[:n_used] = 1
    for i in range(n_used - 1, 0, -1):
        visits[parents[i]] += visits[i]
    child_visits = np.zeros((n, width), np.int32)
    child_values = np.zeros((n, width), np.float32)
    rewards = np.zeros((n, width), np.float32)
    discounts = np.zeros((n, width), np.float32)
    values = np.zeros(n, np.float32)
    values[:n_used] = rng.uniform(-1, 1, n_used)
    for i in range(1, n_used):
        p, a = parents[i], afp[i]
        child_visits[p, a] = visits[i]
        child_values[p, a] = values[i]
        if is_decision[p]:
            discounts[p, a] = 1.0
        else:
            rewards[p, a] = rng.choice([-1.0, 0.0, 0.0, 1.0])
            discounts[p, a] = rng.choice([-1.0, -1.0, 1.0, 0.0])

    invalid = np.ones(width, bool)
    invalid[np.flatnonzero(prior[0, :A] > NEG)] = False
    emb = mctx_base.StochasticRecurrentState(
        state_embedding=jnp.zeros(n), afterstate_embedding=jnp.zeros(n),
        is_decision_node=jnp.asarray(is_decision))
    return mctx.Tree(
        node_visits=jnp.asarray(visits),
        raw_values=jnp.asarray(rng.uniform(-1, 1, n).astype(np.float32)),
        node_values=jnp.asarray(values),
        parents=jnp.asarray(parents),
        action_from_parent=jnp.asarray(afp),
        children_index=jnp.asarray(children),
        children_prior_logits=jnp.asarray(prior),
        children_visits=jnp.asarray(child_visits),
        children_rewards=jnp.asarray(rewards),
        children_discounts=jnp.asarray(discounts),
        children_values=jnp.asarray(child_values),
        embeddings=emb,
        root_invalid_actions=jnp.asarray(invalid),
        extra_data=action_selection.GumbelMuZeroExtraData(
            root_gumbel=jnp.asarray(rng.gumbel(size=A).astype(np.float32))),
    ), is_decision


def _assert_trees_equal(a, b, rtol=1e-6, atol=1e-6):
    for name in ('node_visits', 'raw_values', 'node_values', 'parents', 'action_from_parent',
                 'children_index', 'children_prior_logits', 'children_visits',
                 'children_rewards', 'children_discounts', 'children_values'):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if np.issubdtype(x.dtype, np.integer):
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=name)


# ---------------------------------------------------------------------------
# _backward vs mctx backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('num_simulations,chain', [(8, 1.0), (16, 0.5), (32, 0.9), (32, 0.1)])
def test_backward_matches_mctx_on_random_trees(num_simulations, chain):
    rng = np.random.default_rng(num_simulations * 7 + int(chain * 10))
    ours = jax.jit(jax.vmap(functools.partial(S._backward, max_len=num_simulations + 1)))
    theirs = jax.jit(mctx_search.backward)          # already vmapped in mctx
    trees, leaves = [], []
    depths = set()
    for _ in range(24):
        n_used = int(rng.integers(2, num_simulations + 2))
        tree, _ = random_tree(rng, num_simulations, n_used, chain=chain)
        # Any non-root node can be a backward start: fresh leaves, and interior
        # nodes re-expanded when a simulation stops at max_depth.
        leaf = int(rng.integers(1, n_used))
        d, i = 0, leaf
        while i != 0:
            i, d = int(tree.parents[i]), d + 1
        depths.add(d)
        trees.append(tree)
        leaves.append(leaf)
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *trees)
    leaves = jnp.asarray(leaves, jnp.int32)
    _assert_trees_equal(ours(batch, leaves), theirs(batch, leaves))
    assert len(depths) > 3


def test_backward_full_depth_chain():
    """A pure chain of num_simulations edges: the longest path _backward supports."""
    n_sims = 12
    rng = np.random.default_rng(1)
    tree, _ = random_tree(rng, n_sims, n_sims + 1, chain=1.0)
    batch = jax.tree.map(lambda x: x[None], tree)
    leaf = jnp.array([n_sims], jnp.int32)
    assert int(tree.parents[n_sims]) == n_sims - 1
    ours = jax.vmap(functools.partial(S._backward, max_len=n_sims + 1))(batch, leaf)
    _assert_trees_equal(ours, mctx_search.backward(batch, leaf))
    # Every node on the chain got one more visit.
    assert (np.asarray(ours.node_visits[0, :n_sims]) == np.asarray(tree.node_visits[:n_sims]) + 1).all()


# ---------------------------------------------------------------------------
# _make_selection (row views) vs mctx's stochastic selection
# ---------------------------------------------------------------------------

def _reference_selection(num_simulations, max_considered):
    root = functools.partial(action_selection.gumbel_muzero_root_action_selection,
                             num_simulations=num_simulations,
                             max_num_considered_actions=max_considered, qtransform=QT)
    interior = functools.partial(action_selection.gumbel_muzero_interior_action_selection,
                                 qtransform=QT)

    def decision(key, tree, node_index, depth):
        return jax.lax.cond(depth == 0, lambda: root(key, tree, node_index),
                            lambda: interior(key, tree, node_index, depth))

    return _make_stochastic_action_selection_fn(decision, A)


@pytest.mark.parametrize('num_simulations,max_considered', [(16, 16), (32, 4), (32, 16)])
def test_selection_matches_mctx_on_random_trees(num_simulations, max_considered):
    rng = np.random.default_rng(num_simulations + max_considered)
    ours = jax.jit(S._make_selection(num_simulations, max_considered, QT))
    ref = jax.jit(_reference_selection(num_simulations, max_considered))
    key = jax.random.key(0)
    seen = set()
    for _ in range(40):
        # Root selection needs sum(root visits) < num_simulations.
        n_used = int(rng.integers(1, num_simulations + 1))
        tree, is_decision = random_tree(rng, num_simulations, n_used, chain=rng.random(),
                                        legal_per_node=int(rng.integers(1, 30)))
        for node in range(n_used):
            depths = [0.0, 1.0, 5.0] if node == 0 else [1.0, 2.0, 7.0]
            for depth in depths:
                d = jnp.float32(depth)
                a, b = int(ours(key, tree, node, d)), int(ref(key, tree, node, d))
                assert a == b, (node, depth, a, b)
                seen.add((bool(is_decision[node]), depth == 0))
    assert seen == {(True, True), (True, False), (False, False)}


# ---------------------------------------------------------------------------
# Whole searches over the real environment
# ---------------------------------------------------------------------------

def make_random_forward(seed=0):
    """A fixed random linear 'network': deterministic, input-dependent priors/values."""
    k1, k2 = jax.random.split(jax.random.key(seed))
    n_in = R.N_SQ * E.N_PLANES + E.N_FEATURES
    w = jax.random.normal(k1, (n_in, A)) * 0.3
    v = jax.random.normal(k2, (n_in,)) * 0.3

    def forward(params, planes, features):
        x = jnp.concatenate([planes.reshape(planes.shape[0], -1), features], -1)
        return x @ w, jnp.tanh(x @ v)
    return forward


def _root(forward, states, gumbel):
    batch = states.tile.shape[0]
    logits, value = forward(None, *observe_batch(states))
    prior = S._masked(logits, states.legal)
    root = mctx.RootFnOutput(
        prior_logits=jnp.concatenate([prior, jnp.full((batch, C), NEG)], -1),
        value=value,
        embedding=mctx_base.StochasticRecurrentState(
            state_embedding=states, afterstate_embedding=states,
            is_decision_node=jnp.ones(batch, bool)))
    invalid = jnp.concatenate([~states.legal, jnp.ones((batch, C), bool)], -1)
    return root, invalid, action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)


@pytest.mark.parametrize('num_simulations, max_depth', [(8, 8), (24, 24), (8, 4), (24, 4)])
def test_run_search_matches_stock_mctx(random_states, num_simulations, max_depth):
    forward = make_random_forward()
    states = stack(random_states[::max(1, len(random_states) // 8)][:8])
    batch = states.tile.shape[0]
    gumbel = jax.random.gumbel(jax.random.key(3), (batch, A))
    root, invalid, extra = _root(forward, states, gumbel)
    recurrent_fn = S.make_recurrent_fn(forward)
    rng = jax.random.key(4)

    @jax.jit
    def ours(root, invalid, extra):
        return S._run_search(None, rng, root=root, recurrent_fn=recurrent_fn,
                             selection_fn=S._make_selection(num_simulations, 16, QT),
                             num_simulations=num_simulations, max_depth=max_depth,
                             invalid_actions=invalid, extra_data=extra)

    @jax.jit
    def theirs(root, invalid, extra):
        sel = _reference_selection(num_simulations, 16)
        return mctx_search.search(None, rng, root=root, recurrent_fn=recurrent_fn,
                                  root_action_selection_fn=functools.partial(sel, depth=0),
                                  interior_action_selection_fn=sel,
                                  num_simulations=num_simulations, max_depth=max_depth,
                                  invalid_actions=invalid, extra_data=extra)

    a, b = ours(root, invalid, extra), theirs(root, invalid, extra)
    _assert_trees_equal(a, b, rtol=1e-5, atol=1e-5)
    # The searches did go below the root's children (chance + decision levels).
    assert int(jnp.max(jnp.sum(a.parents > 0, -1))) > 0
