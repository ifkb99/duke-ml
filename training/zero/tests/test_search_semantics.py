"""
Search semantics on the real environment: the sign conventions of the fused
recurrent function, terminal and illegal-action handling, chance nodes, and
the policy target.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mctx._src import base as mctx_base

from training.zero import env as E
from training.zero import rules as R
from training.zero import search as S
from training.zero.network import observe_batch
from training.zero.tests.conftest import make_state, stack
from training.zero.tests.test_search_internals import make_random_forward

A, C = S.A, S.C


def uniform_forward(params, planes, features):
    b = planes.shape[0]
    return jnp.zeros((b, A)), jnp.zeros(b)


def _expand_decision(recurrent_fn, s, actions):
    emb = mctx_base.StochasticRecurrentState(
        state_embedding=s, afterstate_embedding=s, is_decision_node=jnp.ones(len(actions), bool))
    return recurrent_fn(None, None, jnp.asarray(actions, jnp.int32), emb)


def _expand_chance(recurrent_fn, chance_emb, outcomes):
    return recurrent_fn(None, None, A + jnp.asarray(outcomes, jnp.int32), chance_emb)


def test_recurrent_fn_signs(random_states):
    """Decision -> chance edge and chance -> decision edge for every action kind."""
    forward = make_random_forward(1)
    recurrent_fn = jax.jit(S.make_recurrent_fn(forward))
    rng = np.random.default_rng(0)
    states, actions, kinds = [], [], []
    for s in random_states:
        legal = np.flatnonzero(np.asarray(s.legal))
        board = legal[legal < R.DRAW_ACTION]
        if len(board):
            states.append(s); actions.append(int(rng.choice(board))); kinds.append('board')
        if legal[-1] == R.DRAW_ACTION:
            states.append(s); actions.append(R.DRAW_ACTION); kinds.append('draw')
        if legal[0] >= R.PLACE_BASE:
            states.append(s); actions.append(int(rng.choice(legal))); kinds.append('place')
        illegal = np.flatnonzero(~np.asarray(s.legal))
        states.append(s); actions.append(int(rng.choice(illegal))); kinds.append('illegal')
    assert {'board', 'draw', 'place', 'illegal'} <= set(kinds)
    s = stack(states)
    kinds = np.array(kinds)

    out, emb = _expand_decision(recurrent_fn, s, actions)
    after = jax.vmap(E.step_decision)(s, jnp.asarray(actions))
    _, v_after = forward(None, *observe_batch(after))
    after_np = jax.device_get(after)

    np.testing.assert_array_equal(out.reward, 0)
    np.testing.assert_array_equal(out.discount, 1)
    np.testing.assert_array_equal(out.prior_logits[:, :A], S.NEG)
    expected = np.where(after_np.done, after_np.reward,
                        np.where(after_np.switched, -np.asarray(v_after), np.asarray(v_after)))
    np.testing.assert_allclose(out.value, expected, rtol=1e-6)
    # Spot-check the cases by kind.
    ill = kinds == 'illegal'
    np.testing.assert_array_equal(np.asarray(out.value)[ill], -1.0)
    draw = kinds == 'draw'
    assert not after_np.switched[draw].any() and not after_np.done[draw].any()
    np.testing.assert_allclose(np.asarray(out.value)[draw], np.asarray(v_after)[draw], rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(emb.is_decision_node), False)

    # Chance outcome probabilities: the bag for draws, a single certain outcome otherwise.
    probs = np.exp(np.asarray(out.prior_logits[:, A:]))
    bag = np.asarray(s.bags[:, 0], np.float32)
    np.testing.assert_allclose(probs[draw], bag[draw] / bag[draw].sum(-1, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(probs[~draw], np.eye(C)[np.zeros((~draw).sum(), int)], atol=1e-30)

    # Expand each chance node with its most likely outcome.
    outcomes = probs.argmax(-1)
    out2, emb2 = _expand_chance(recurrent_fn, emb, outcomes)
    nxt = jax.device_get(emb2.state_embedding)
    _, v_next = forward(None, *observe_batch(emb2.state_embedding))
    np.testing.assert_array_equal(out2.reward, after_np.reward)
    np.testing.assert_array_equal(
        out2.discount, np.where(after_np.done, 0.0, np.where(after_np.switched, -1.0, 1.0)))
    np.testing.assert_allclose(out2.value, np.where(nxt.done, 0.0, np.asarray(v_next)), rtol=1e-6)
    # The new decision node's prior is the legal mask of the next state.
    prior_ok = np.asarray(out2.prior_logits[:, :A]) > S.NEG
    np.testing.assert_array_equal(prior_ok, nxt.legal)
    np.testing.assert_array_equal(np.asarray(out2.prior_logits[:, A:]), S.NEG)
    # After a draw the same player places the revealed tile.
    assert (nxt.pending[draw] == outcomes[draw]).all()
    assert (nxt.player[draw] == np.asarray(s.player)[draw]).all()
    np.testing.assert_array_equal(np.asarray(emb2.is_decision_node), True)


def _capture_position():
    """My Footman can take the enemy Duke; a handful of other legal moves."""
    return make_state([
        (5, 0, 'Duke', 0, 0),
        (1, 3, 'Footman', 0, 0),
        (0, 3, 'Duke', 1, 0),
        (0, 0, 'Footman', 1, 0),
    ])


def test_search_takes_the_duke_and_terminal_nodes_are_zero():
    s = _capture_position()
    capture = (1 * 6 + 3) * R.N_SQ + (0 * 6 + 3)
    legal = np.flatnonzero(np.asarray(s.legal))
    assert capture in legal and 2 < len(legal) <= 16
    search = jax.jit(S.make_search(uniform_forward, 32, 16))
    states = stack([s, s])
    out = search(None, jax.random.key(0), states, 0.0)
    assert (np.asarray(out.action) == capture).all()
    assert float(out.search_value[0]) > 0

    # Inspect the tree: same search, but keep it.
    tree = _search_tree(uniform_forward, states, 32)
    chance = int(tree.children_index[0, 0, capture])
    assert chance > 0
    assert float(tree.children_values[0, 0, capture]) == 1.0           # chance node: X won
    term = int(tree.children_index[0, chance, A])                       # its only outcome
    assert term > 0
    assert float(tree.children_rewards[0, chance, A]) == 1.0
    assert float(tree.children_discounts[0, chance, A]) == 0.0
    assert float(tree.node_values[0, term]) == 0.0                      # terminal: value 0
    assert bool(tree.embeddings.state_embedding.done[0, term])
    # Revisits of the terminal decision node keep it at 0 and the edge at +1.
    assert int(tree.node_visits[0, chance]) >= 2


def _search_tree(forward, states, num_simulations, gumbel_scale=0.0, seed=0):
    import mctx
    from mctx._src import action_selection, qtransforms
    batch = states.tile.shape[0]
    logits, value = forward(None, *observe_batch(states))
    prior = S._masked(logits, states.legal)
    rng, g = jax.random.split(jax.random.key(seed))
    root = mctx.RootFnOutput(
        prior_logits=jnp.concatenate([prior, jnp.full((batch, C), S.NEG)], -1), value=value,
        embedding=mctx_base.StochasticRecurrentState(
            state_embedding=states, afterstate_embedding=states,
            is_decision_node=jnp.ones(batch, bool)))
    return jax.jit(lambda root: S._run_search(
        None, rng, root=root, recurrent_fn=S.make_recurrent_fn(forward),
        selection_fn=S._make_selection(num_simulations, 16, qtransforms.qtransform_completed_by_mix_value),
        num_simulations=num_simulations, max_depth=num_simulations,
        invalid_actions=jnp.concatenate([~states.legal, jnp.ones((batch, C), bool)], -1),
        extra_data=action_selection.GumbelMuZeroExtraData(
            root_gumbel=gumbel_scale * jax.random.gumbel(g, (batch, A)))))(root)


def _draw_position():
    """Only a draw or a couple of Duke moves are legal, with a mixed bag."""
    bags = np.zeros((2, R.N_TYPES), np.int32)
    bags[0] = R.INITIAL_BAG
    bags[1] = R.INITIAL_BAG
    return make_state([
        (5, 2, 'Duke', 0, 0),
        (5, 1, 'Footman', 0, 0),
        (5, 3, 'Footman', 0, 0),
        (0, 2, 'Duke', 1, 0),
    ], bags=bags)


def draw_favouring_forward(params, planes, features):
    """Prefers drawing, values every position the same (so visits follow priors)."""
    b = planes.shape[0]
    return jnp.zeros((b, A)).at[:, R.DRAW_ACTION].set(5.0), jnp.zeros(b)


def test_chance_visits_track_bag_probabilities():
    s = _draw_position()
    assert bool(s.legal[R.DRAW_ACTION])
    n_sims = 200
    tree = _search_tree(draw_favouring_forward, stack([s]), n_sims)
    chance = int(tree.children_index[0, 0, R.DRAW_ACTION])
    visits = np.asarray(tree.children_visits[0, chance, A:])
    n = visits.sum()
    assert n >= 50, n
    p = R.INITIAL_BAG / R.INITIAL_BAG.sum()
    # Deterministic "least visited relative to probability" selection keeps
    # every outcome within one visit of its expected share.
    assert np.abs(visits - n * p).max() <= 1.0, (visits, n * p)
    # And outcomes that can't be drawn are never visited.
    assert (visits[p == 0] == 0).all()


def test_action_weights_zero_on_illegal(random_states):
    forward = make_random_forward(2)
    search = jax.jit(S.make_search(forward, 8, 16))
    states = stack(random_states[:64])
    out = search(None, jax.random.key(1), states, 1.0)
    w = np.asarray(out.action_weights)
    legal = np.asarray(states.legal)
    assert (w[~legal] == 0).all()
    np.testing.assert_allclose(w.sum(-1), 1.0, rtol=1e-5)
    assert legal[np.arange(len(w)), np.asarray(out.action)].all()
