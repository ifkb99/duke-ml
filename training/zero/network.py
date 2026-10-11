"""
DukeZeroNet: a small residual network with a policy head and a value head.

Inputs (from env.observe, batched):
  planes   [B, 6, 6, 60]  one-hot tiles: owner (me/opponent) x type x side
  features [B, 47]        bag counts, tile in hand, drawing flag, turn number

Outputs:
  logits   [B, 3605]      unnormalized log-probabilities over actions
  value    [B]            expected outcome for the player to move, in [-1, 1]

Shapes through the trunk (C = channels):
  planes --conv3x3--> [B,6,6,C] + features projected to a per-channel bias
  -> residual blocks [B,6,6,C] -> heads

Two policy heads (`policy_head`):
  'dense'    a 1x1 conv to 32 channels, then one dense layer to all 3,605
             actions. ~4.6M parameters at C=128: most of the network.
  'spatial'  reads each action's logit off the square it starts from, following
             the action numbering in rules.py. A 1x1 conv gives every square
             36 + 64 channels: "move/strike from here to square t" and
             "command from here: src dir x dst dir". Moves also get a
             bilinear term q(from)·k(to), so the target square's own features
             count. Draw and the 4 placements come from the pooled board.
             ~21K parameters at C=128.
  'factored' the spatial head, deciding a move in steps ("semantic ifs"):
             draw or play a board move (one logit from the pooled board),
             which piece (a logit per square), then where (the spatial
             logits, normalized among that piece's legal actions only).
             log p(move) = log p(kind) + log p(piece | kind) + log p(move | piece).
             It outputs 3,605 + 37 numbers; policy_logp turns them into
             log-probabilities given the legal moves.

Semantic features (semantic.py: attack maps, moves per piece, check):
  semantic_inputs   computes them from the planes inside the network and adds
                    them to its input (3 planes, 2 features).
  semantic_targets  adds small auxiliary heads that predict them; only the
                    training loss reads those (with_aux), play never does.

Convolutions and dense layers compute in bfloat16 (tensor cores; ~2-3x faster
on Ampere) while weights stay float32; norms and the value output use float32.

LayerNorm is used instead of BatchNorm so the network behaves identically in
training and inside the search (no running statistics to manage), and the
module holds nothing but Params — `nnx.split` gives a clean (graphdef, params).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from training.zero import env as E
from training.zero import rules as R
from training.zero import semantic as M


class ResBlock(nnx.Module):
    """Pre-activation residual block: x + conv(relu(norm(conv(relu(norm(x))))))."""

    def __init__(self, ch: int, *, dtype, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(ch, rngs=rngs)
        self.conv1 = nnx.Conv(ch, ch, (3, 3), dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(ch, rngs=rngs)
        self.conv2 = nnx.Conv(ch, ch, (3, 3), dtype=dtype, rngs=rngs,
                              kernel_init=nnx.initializers.zeros)  # block starts as identity

    def __call__(self, x):
        y = self.conv1(nnx.relu(self.norm1(x)))
        y = self.conv2(nnx.relu(self.norm2(y)))
        return x + y.astype(x.dtype)


POLICY_HEADS = ('dense', 'spatial', 'factored')
POLICY_KEY = 32                              # spatial head: size of the from/to vectors
N_CMD = 64                                   # command logits per commander square
assert R.N_MOVE_ACTIONS == R.N_SQ * R.N_SQ and R.N_CMD_ACTIONS == R.N_SQ * N_CMD
assert R.N_ACTIONS - R.DRAW_ACTION == 5      # draw + 4 placements


class DukeZeroNet(nnx.Module):
    def __init__(self, channels: int = 96, blocks: int = 6, policy_head: str = 'dense', *,
                 semantic_inputs: bool = False, semantic_targets: bool = False,
                 dtype=jnp.bfloat16, rngs: nnx.Rngs):
        assert policy_head in POLICY_HEADS, policy_head
        c = channels
        self.policy_head = policy_head
        self.semantic_inputs = semantic_inputs
        extra_planes, extra_features = (M.N_PLANES, M.N_FEATURES) if semantic_inputs else (0, 0)
        self.stem = nnx.Conv(E.N_PLANES + extra_planes, c, (3, 3), dtype=dtype, rngs=rngs)
        self.film = nnx.Linear(E.N_FEATURES + extra_features, c, rngs=rngs)
        if semantic_targets:
            self.aux_planes = nnx.Conv(c, M.N_PLANES, (1, 1), dtype=dtype, rngs=rngs)
            self.aux_features = nnx.Linear(c, M.N_FEATURES, dtype=dtype, rngs=rngs)
        self.blocks = nnx.List([ResBlock(c, dtype=dtype, rngs=rngs) for _ in range(blocks)])
        self.norm = nnx.LayerNorm(c, rngs=rngs)

        sq = R.N_SQ
        zeros = nnx.initializers.zeros       # policy outputs start at 0: uniform prior at init
        if policy_head == 'dense':
            self.policy_conv = nnx.Conv(c, 32, (1, 1), dtype=dtype, rngs=rngs)
            self.policy_out = nnx.Linear(sq * 32 + c, R.N_ACTIONS, dtype=dtype, rngs=rngs, kernel_init=zeros)
        else:
            if policy_head == 'factored':
                self.policy_piece = nnx.Conv(c, 1, (1, 1), dtype=dtype, rngs=rngs, kernel_init=zeros)
                # draw or not: a linear map of the mean-pooled board was too weak
                # (az8-factored v1); the max-pooled board shows local threats.
                self.policy_kind_hidden = nnx.Linear(2 * c, 64, dtype=dtype, rngs=rngs)
                self.policy_kind = nnx.Linear(64, 1, dtype=dtype, rngs=rngs, kernel_init=zeros)
            self.policy_sq = nnx.Conv(c, sq + N_CMD, (1, 1), dtype=dtype, rngs=rngs, kernel_init=zeros)
            self.policy_q = nnx.Conv(c, POLICY_KEY, (1, 1), dtype=dtype, rngs=rngs)
            self.policy_k = nnx.Conv(c, POLICY_KEY, (1, 1), dtype=dtype, rngs=rngs, kernel_init=zeros)
            self.policy_misc = nnx.Linear(c, 5, dtype=dtype, rngs=rngs, kernel_init=zeros)
        self.value_conv = nnx.Conv(c, 8, (1, 1), dtype=dtype, rngs=rngs)
        self.value_hidden = nnx.Linear(sq * 8 + c, 128, rngs=rngs)
        self.value_out = nnx.Linear(128, 1, rngs=rngs,
                                    kernel_init=nnx.initializers.zeros)  # value 0 at init

    def __call__(self, planes, features):
        return self.predict(self.represent(planes, features))

    def with_aux(self, planes, features):
        """(logits, value, semantic plane logits [B,6,6,3], semantic feature logits [B,2])."""
        x = self.represent(planes, features)
        logits, value = self.predict(x)
        h = nnx.relu(self.norm(x))
        return (logits, value, self.aux_planes(h).astype(jnp.float32),
                self.aux_features(h.mean(axis=(1, 2))).astype(jnp.float32))

    def represent(self, planes, features):
        """Board -> latent [B,6,6,C]: the trunk's residual stream (MuZero's h)."""
        if self.semantic_inputs:
            sem_planes, sem_features = M.rule_features(planes)
            planes = jnp.concatenate([planes, sem_planes.astype(planes.dtype)], -1)
            features = jnp.concatenate([features, sem_features.astype(features.dtype)], -1)
        x = self.stem(planes).astype(jnp.float32) + self.film(features)[:, None, None, :]  # [B,6,6,C]
        for block in self.blocks:
            x = block(x)
        return x

    def predict(self, x):
        """Latent -> (logits, value): the heads (MuZero's f). Normalizes its input,
        so the latent's scale doesn't matter."""
        b = x.shape[0]
        x = nnx.relu(self.norm(x))
        pooled = x.mean(axis=(1, 2))                                      # [B,C] board summary

        logits = self.policy_logits(x, pooled)                            # [B, 3605]

        v = nnx.relu(self.value_conv(x)).reshape(b, -1).astype(jnp.float32)  # [B, 36*8]
        v = nnx.relu(self.value_hidden(jnp.concatenate([v, pooled], axis=-1)))
        value = jnp.tanh(self.value_out(v))[:, 0]                         # [B]
        return logits, value

    def policy_logits(self, x, pooled):
        b, sq = x.shape[0], R.N_SQ
        if self.policy_head == 'dense':
            p = nnx.relu(self.policy_conv(x)).reshape(b, -1)              # [B, 36*32]
            return self.policy_out(jnp.concatenate([p, pooled], axis=-1)).astype(jnp.float32)
        per_sq = self.policy_sq(x).astype(jnp.float32).reshape(b, sq, sq + N_CMD)   # [B, from, 100]
        q = self.policy_q(x).astype(jnp.float32).reshape(b, sq, POLICY_KEY)
        k = self.policy_k(x).astype(jnp.float32).reshape(b, sq, POLICY_KEY)
        moves = per_sq[..., :sq] + jnp.einsum('bfd,btd->bft', q, k) / POLICY_KEY ** 0.5  # [B, from, to]
        commands = per_sq[..., sq:]                                       # [B, commander, src*8+dst]
        rest = self.policy_misc(pooled).astype(jnp.float32)               # [B, 5]: draw, 4 placements
        # Row-major flattening matches rules.py: from*36+to, then
        # CMD_BASE + commander*64 + src*8 + dst, then DRAW_ACTION, PLACE_BASE+d.
        out = [moves.reshape(b, -1), commands.reshape(b, -1), rest]
        if self.policy_head == 'factored':
            out += [self.policy_piece(x).astype(jnp.float32).reshape(b, sq),   # which piece
                    self.policy_kind(nnx.relu(self.policy_kind_hidden(            # draw vs board move
                        jnp.concatenate([pooled, x.max(axis=(1, 2))], -1)))).astype(jnp.float32)]
        return jnp.concatenate(out, axis=-1)


def _log_softmax(x, mask, neg):
    return jax.nn.log_softmax(jnp.where(mask, x, neg), axis=-1)


def policy_logp(logits, legal, neg=-1e9):
    """A policy head's output -> log-probabilities over actions [..., 3605]
    (illegal actions get about `neg`).

    Flat heads: a softmax over the legal actions. The factored head: one
    softmax per decision step, each over its legal options only, added up.
    """
    if logits.shape[-1] == R.N_ACTIONS:
        return _log_softmax(logits, legal, neg)
    lead, sq = logits.shape[:-1], R.N_SQ
    m, piece, kind = logits[..., :R.N_ACTIONS], logits[..., R.N_ACTIONS:R.N_ACTIONS + sq], logits[..., -1]

    def by_piece(x):   # board actions [..., 3600] -> [..., acting square, 36 moves + 64 commands]
        return jnp.concatenate([x[..., :R.N_MOVE_ACTIONS].reshape(*lead, sq, sq),
                                x[..., R.CMD_BASE:R.DRAW_ACTION].reshape(*lead, sq, N_CMD)], -1)

    board_legal = by_piece(legal)
    piece_legal = board_legal.any(-1)
    kinds = jnp.stack([jnp.zeros_like(kind), kind], -1)                       # board move, draw
    kind_lp = _log_softmax(kinds, jnp.stack([piece_legal.any(-1), legal[..., R.DRAW_ACTION]], -1), neg)
    board = (kind_lp[..., 0, None, None] + _log_softmax(piece, piece_legal, neg)[..., None]
             + _log_softmax(by_piece(m), board_legal, neg))                   # [..., 36, 100]
    place = _log_softmax(m[..., R.PLACE_BASE:], legal[..., R.PLACE_BASE:], neg)  # a tile in hand: only placing
    out = jnp.concatenate([board[..., :sq].reshape(*lead, -1), board[..., sq:].reshape(*lead, -1),
                           kind_lp[..., 1:], place], -1)
    return jnp.where(legal, out, neg)


def make_forward(graphdef):
    """Pure function (params, planes, features) -> (logits, value).

    Search and training pass `params` around explicitly as a pytree, which is
    what jit/vmap/mctx expect; this rebuilds the module from it on each call
    (free under jit — it all traces away).
    """
    def forward(params, planes, features):
        return nnx.merge(graphdef, params)(planes, features)
    return forward


def make_forward_aux(graphdef):
    """As make_forward, returning the semantic auxiliary outputs too (with_aux)."""
    def forward(params, planes, features):
        return nnx.merge(graphdef, params).with_aux(planes, features)
    return forward


def observe_batch(states):
    return jax.vmap(E.observe)(states)
