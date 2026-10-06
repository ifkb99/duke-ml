"""
Monte-Carlo tree search with real game rules and real chance nodes.

We use DeepMind's `mctx` building blocks to assemble a search that combines:
  - Gumbel MuZero action selection ("Policy improvement by planning with
    Gumbel", Danihelka et al. 2022). It improves the policy reliably even with
    few simulations (tens instead of the hundreds/thousands AlphaZero used).
  - Stochastic MuZero's tree structure ("Planning in Stochastic Environments
    with a Learned Model", Antonoglou et al. 2022): decision nodes alternate
    with chance nodes, so a draw is averaged over the tiles you could pull
    instead of being "decided" by the searching player.

Unlike MuZero we don't learn a model: the "recurrent function" steps the real
JAX environment, and the network only supplies priors and value estimates.

Tree structure and sign conventions
-----------------------------------
  decision node (state, player X to move)
     | action a                  reward 0, discount +1
  chance node (afterstate)       value from X's point of view
     | outcome c                 reward r (for X), discount -1 if the turn
  decision node (next state)     passed, +1 if X still moves (after a draw),
                                 0 if the game ended. Value from the point of
                                 view of whoever moves there.
mctx backs values up as  Q = reward + discount * child_value, so every node's
value ends up in the perspective of the player choosing at its parent.

Non-draw actions also go through a chance node, with a single certain outcome
(outcome 0). It costs one extra cheap simulation step per move.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
from mctx._src import action_selection, base as mctx_base, qtransforms, search as mctx_search, seq_halving
from mctx._src.policies import _mask_tree

from training.zero import env as E
from training.zero import rules as R
from training.zero.network import observe_batch

A = R.N_ACTIONS     # decision actions
C = R.N_TYPES       # chance outcomes: which tile type was drawn
NEG = jnp.finfo(jnp.float32).min


def _masked(logits, legal):
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(legal, logits, NEG)


def _select(cond, a, b):
    """Per-batch-element select between two pytrees with leading batch axis."""
    def pick(x, y):
        c = cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim))
        return jnp.where(c, x, y)
    return jax.tree.map(pick, a, b)


def make_recurrent_fn(forward):
    """Builds the mctx recurrent function for decision AND chance nodes.

    mctx's stock stochastic recurrent wrapper runs two network evaluations per
    simulation (one for each node type, then discards one). Here both env
    branches are computed, the right state is selected per batch element, and
    the network runs once.
    """
    step_decision = jax.vmap(E.step_decision)
    step_chance = jax.vmap(E.step_chance)
    chance_probs = jax.vmap(E.chance_probs)

    def recurrent_fn(params, rng, action_or_chance, emb: mctx_base.StochasticRecurrentState):
        del rng
        is_decision = emb.is_decision_node
        after = step_decision(emb.state_embedding, jnp.clip(action_or_chance, 0, A - 1))
        nxt = step_chance(emb.afterstate_embedding, jnp.clip(action_or_chance - A, 0, C - 1))

        node = _select(is_decision, after, nxt)
        logits, value = forward(params, *observe_batch(node))

        # Expanding a decision node -> new chance node (afterstate) for player X.
        after_value = jnp.where(after.done, after.reward,
                                jnp.where(after.switched, -value, value))
        probs = chance_probs(after)
        chance_logits = jnp.where(probs > 0, jnp.log(jnp.maximum(probs, 1e-30)), NEG)
        decision_out = mctx.RecurrentFnOutput(
            reward=jnp.zeros_like(value),
            discount=jnp.ones_like(value),
            prior_logits=jnp.concatenate([jnp.full((value.shape[0], A), NEG), chance_logits], -1),
            value=after_value,
        )

        # Expanding a chance node -> new decision node (next state).
        afterstate = emb.afterstate_embedding
        chance_out = mctx.RecurrentFnOutput(
            reward=afterstate.reward,
            discount=jnp.where(nxt.done, 0.0, jnp.where(afterstate.switched, -1.0, 1.0)),
            prior_logits=jnp.concatenate([_masked(logits, nxt.legal),
                                          jnp.full((value.shape[0], C), NEG)], -1),
            value=jnp.where(nxt.done, 0.0, value),
        )

        out = _select(is_decision, decision_out, chance_out)
        new_emb = mctx_base.StochasticRecurrentState(
            state_embedding=nxt, afterstate_embedding=after,
            is_decision_node=~is_decision)
        return out, new_emb

    return recurrent_fn


def _row_view(tree, node_index, lo, hi):
    """A one-node view of `tree` holding only node_index's children in [lo, hi).

    mctx's stochastic helper slices the WHOLE tree (every node x 3620 children,
    several arrays) on every selection step to separate decision children from
    chance children. That copy dominated search time (seconds per move). The
    selection rules only ever read the current node's row, so a row view
    gives identical results at a tiny fraction of the memory traffic.
    """
    node = lambda x: x[node_index][None]
    row = lambda x: x[node_index, lo:hi][None]
    return tree.replace(
        node_visits=node(tree.node_visits),
        raw_values=node(tree.raw_values),
        node_values=node(tree.node_values),
        parents=node(tree.parents),
        action_from_parent=node(tree.action_from_parent),
        children_index=row(tree.children_index),
        children_prior_logits=row(tree.children_prior_logits),
        children_visits=row(tree.children_visits),
        children_rewards=row(tree.children_rewards),
        children_discounts=row(tree.children_discounts),
        children_values=row(tree.children_values),
        embeddings=jnp.zeros(()),
        root_invalid_actions=tree.root_invalid_actions[lo:hi],
    )


def _make_selection(num_simulations, max_considered, qtransform):
    """Action selection for both node types (unbatched tree, as mctx expects).

    Decision nodes: Gumbel MuZero (sequential halving at the root, deterministic
    "visit what's most under-visited relative to the improved policy" below).
    Chance nodes: visit the outcome whose visit share lags its probability
    most, so visits track the true draw distribution.
    """
    root_select = functools.partial(
        action_selection.gumbel_muzero_root_action_selection,
        num_simulations=num_simulations,
        max_num_considered_actions=max_considered,
        qtransform=qtransform)
    interior_select = functools.partial(
        action_selection.gumbel_muzero_interior_action_selection, qtransform=qtransform)

    def select(rng, tree, node_index, depth):
        is_decision = tree.embeddings.is_decision_node[node_index]
        view = _row_view(tree, node_index, 0, A)

        def decision():
            return jax.lax.cond(depth == 0,
                                lambda: root_select(rng, view, 0),
                                lambda: interior_select(rng, view, 0, depth))

        def chance():
            visits = tree.children_visits[node_index, A:]
            probs = jax.nn.softmax(tree.children_prior_logits[node_index, A:])
            score = probs / (visits + 1)
            # Exact ties are common (equal bag counts) and float rounding of
            # softmax(log p) would break them at random: take the lowest outcome.
            return (jnp.argmax(score >= jnp.max(score) - 1e-6) + A).astype(jnp.int32)

        return jax.lax.cond(is_decision, decision, chance)

    return select


def _backward(tree, leaf_index, max_len):
    """Back up the leaf's value to the root (unbatched; vmapped by the caller).

    Same math as mctx's `backward`, restructured for speed. mctx walks up with a
    while_loop that carries the whole tree; under vmap a batched while_loop
    re-selects its entire carry on every iteration, i.e. copies the full tree
    (nodes x 3620 actions x several arrays) per step. Here the loop carries
    only node indices, values are accumulated with a short scan, and the tree
    is written once with scatters.
    """
    Tree = mctx.Tree
    n_nodes = tree.node_values.shape[0]

    # 1. Collect the path leaf -> root: path[0] = leaf, path[k+1] = parent(path[k]).
    def cond(c):
        k, idx, _ = c
        return (idx != Tree.ROOT_INDEX) & (k < max_len - 1)

    def body(c):
        k, idx, path = c
        parent = tree.parents[idx]
        return k + 1, parent, path.at[k + 1].set(parent)

    path0 = jnp.full(max_len, Tree.ROOT_INDEX, jnp.int32).at[0].set(leaf_index)
    n_edges, _, path = jax.lax.while_loop(cond, body, (jnp.int32(0), leaf_index, path0))

    child, parent = path[:-1], path[1:]                    # edge k: parent[k] --a[k]--> child[k]
    valid = jnp.arange(max_len - 1) < n_edges
    action = tree.action_from_parent[child]
    reward = tree.children_rewards[parent, action]
    discount = tree.children_discounts[parent, action]

    # 2. Value seen from each parent: v[k] = reward[k] + discount[k] * v[k-1].
    def accumulate(v, rd):
        v = rd[0] + rd[1] * v
        return v, v
    leaf_value = tree.node_values[leaf_index]
    _, v = jax.lax.scan(accumulate, leaf_value, (reward, discount))

    # 3. Running means at each parent; each edge's child value is the child's
    #    (already updated) node value.
    count = tree.node_visits[parent]
    new_value = (tree.node_values[parent] * count + v) / (count + 1.0)
    child_value = jnp.concatenate([leaf_value[None], new_value[:-1]])

    # Path nodes are distinct, so the scatters don't collide. Invalid edges
    # are routed out of bounds and dropped.
    p = jnp.where(valid, parent, n_nodes)
    return tree.replace(
        node_values=tree.node_values.at[p].set(new_value, mode='drop'),
        node_visits=tree.node_visits.at[p].set(count + 1, mode='drop'),
        children_values=tree.children_values.at[p, action].set(child_value, mode='drop'),
        children_visits=tree.children_visits.at[p, action].set(
            tree.children_visits[parent, action] + 1, mode='drop'),
    )


def _run_search(params, rng, *, root, recurrent_fn, selection_fn, num_simulations,
                max_depth, invalid_actions, extra_data):
    """mctx.search's simulation loop, using the faster `_backward` above."""
    batch = root.value.shape[0]
    batch_range = jnp.arange(batch)
    action_selection_fn = action_selection.switching_action_selection_wrapper(
        root_action_selection_fn=functools.partial(selection_fn, depth=0),
        interior_action_selection_fn=selection_fn)
    backward = jax.vmap(functools.partial(_backward, max_len=max_depth + 1))

    def body(sim, carry):
        rng, tree = carry
        rng, simulate_key, expand_key = jax.random.split(rng, 3)
        parent, action = mctx_search.simulate(
            jax.random.split(simulate_key, batch), tree, action_selection_fn, max_depth)
        nxt = tree.children_index[batch_range, parent, action]
        nxt = jnp.where(nxt == mctx.Tree.UNVISITED, sim + 1, nxt)
        tree = mctx_search.expand(params, expand_key, tree, recurrent_fn, parent, action, nxt)
        return rng, backward(tree, nxt)

    tree = mctx_search.instantiate_tree_from_root(
        root, num_simulations, root_invalid_actions=invalid_actions, extra_data=extra_data)
    _, tree = jax.lax.fori_loop(0, num_simulations, body, (rng, tree))
    return tree


class SearchOutput(NamedTuple):
    action: jax.Array          # [B] chosen action
    action_weights: jax.Array  # [B, A] improved policy (training target)
    root_value: jax.Array      # [B] network value at the root
    search_value: jax.Array    # [B] value of the root after search


def make_search(forward, num_simulations: int, max_considered: int = 16, max_depth: int | None = None):
    """Returns search(params, rng, states, gumbel_scale) -> SearchOutput.

    `states` must be decision states (not afterstates). Use gumbel_scale=1.0
    for exploration in self-play and 0.0 for strongest play in evaluation.

    `max_depth` caps how many edges (decision + chance, so 2 per move) one
    simulation may descend. Batched tree walks run until the deepest walk in
    the whole batch finishes, so one deep forcing line would otherwise stall
    every game in the batch.
    """
    max_depth = max_depth or num_simulations
    recurrent_fn = make_recurrent_fn(forward)
    qtransform = qtransforms.qtransform_completed_by_mix_value
    selection = _make_selection(num_simulations, max_considered, qtransform)

    def search(params, rng, states: E.State, gumbel_scale=1.0) -> SearchOutput:
        batch = states.tile.shape[0]
        logits, value = forward(params, *observe_batch(states))
        invalid = ~states.legal
        prior = _masked(logits, states.legal)

        rng, gumbel_rng = jax.random.split(rng)
        gumbel = gumbel_scale * jax.random.gumbel(gumbel_rng, (batch, A))
        root = mctx.RootFnOutput(
            prior_logits=jnp.concatenate([prior, jnp.full((batch, C), NEG)], -1),
            value=value,
            embedding=mctx_base.StochasticRecurrentState(
                state_embedding=states, afterstate_embedding=states,
                is_decision_node=jnp.ones(batch, bool)),
        )
        tree = _run_search(
            params, rng, root=root, recurrent_fn=recurrent_fn, selection_fn=selection,
            num_simulations=num_simulations, max_depth=max_depth,
            invalid_actions=jnp.concatenate([invalid, jnp.ones((batch, C), bool)], -1),
            extra_data=action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel),
        )

        # Read the result off the decision part of the root, as Gumbel MuZero does.
        tree = _mask_tree(tree, A, 'decision')
        summary = tree.summary()
        completed_q = jax.vmap(qtransform, in_axes=[0, None])(tree, tree.ROOT_INDEX)
        considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
        score = seq_halving.score_considered(
            considered_visit, gumbel, prior, completed_q, summary.visit_counts)
        action = action_selection.masked_argmax(score, invalid)
        weights = jax.nn.softmax(_masked(prior + completed_q, states.legal))
        return SearchOutput(action=action, action_weights=weights,
                            root_value=value, search_value=summary.value)

    return search
