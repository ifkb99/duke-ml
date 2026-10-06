"""
Export a DukeZeroNet checkpoint to ONNX, for running in a browser
(onnxruntime-web) or anywhere else.

    uv run python -m training.zero.export_onnx runs/az3-net/latest.pkl web/latency/models/az3net
    uv run python -m training.zero.export_onnx --random 64x4 web/latency/models/r64x4

Writes <out>.fp32.onnx, <out>.fp16.onnx (half the size; fast on WebGPU) and
<out>.int8.onnx (dense layers quantized to 8 bits; the policy layer is most of
the parameters), plus <out>.json with sizes and, for real checkpoints, a few
positions with the expected outputs so the browser can check its results.

The graph is written by hand from the weights (no PyTorch needed) and checked
against a float32 JAX forward pass. Same inputs and outputs as the network:
  planes   float32 [B, 6, 6, 60]   (env.observe layout, NHWC)
  features float32 [B, 47]
  -> logits float32 [B, 3605], value float32 [B]
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from flax import nnx
from onnx import TensorProto, helper, numpy_helper

from training.zero import env as E
from training.zero import rules as R
from training.zero.network import DukeZeroNet, make_forward, observe_batch

EPS = 1e-6       # nnx.LayerNorm default
OPSET = 18


class Graph:
    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def name(self, base):
        self.n += 1
        return f'{base}_{self.n}'

    def const(self, arr, base='c'):
        nm = self.name(base)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), nm))
        return nm

    def op(self, kind, inputs, base=None, **attrs):
        out = self.name(base or kind.lower())
        self.nodes.append(helper.make_node(kind, inputs, [out], **attrs))
        return out

    # --- layers (activations are NCHW inside the graph) ---
    def conv(self, x, p):
        w = np.transpose(np.asarray(p['kernel'], np.float32), (3, 2, 0, 1))      # HWIO -> OIHW
        k = w.shape[2]
        # Every attribute spelled out: ONNX has defaults, but onnxruntime-web's
        # WebGPU Conv rejects a node without dilations ("dilations should be 2D").
        return self.op('Conv', [x, self.const(w, 'w'), self.const(np.asarray(p['bias'], np.float32), 'b')],
                       kernel_shape=[k, k], strides=[1, 1], dilations=[1, 1], group=1,
                       pads=[k // 2] * 4)

    def layernorm_c(self, x, p):
        """LayerNorm over the channel axis of an NCHW tensor."""
        axes = self.const(np.array([1], np.int64), 'ax')
        mean = self.op('ReduceMean', [x, axes], keepdims=1)
        d = self.op('Sub', [x, mean])
        var = self.op('ReduceMean', [self.op('Mul', [d, d]), axes], keepdims=1)
        inv = self.op('Reciprocal', [self.op('Sqrt', [self.op('Add', [var, self.const(np.float32(EPS))])])])
        y = self.op('Mul', [d, inv])
        scale = np.asarray(p['scale'], np.float32).reshape(1, -1, 1, 1)
        bias = np.asarray(p['bias'], np.float32).reshape(1, -1, 1, 1)
        return self.op('Add', [self.op('Mul', [y, self.const(scale, 'g')]), self.const(bias, 'beta')])

    def dense(self, x, p):
        return self.op('Add', [self.op('MatMul', [x, self.const(np.asarray(p['kernel'], np.float32), 'w')]),
                               self.const(np.asarray(p['bias'], np.float32), 'b')])

    def relu(self, x):
        return self.op('Relu', [x])

    def flatten_nhwc(self, x):
        """[B,C,6,6] -> [B, 6*6*C] in the same order as JAX's NHWC reshape."""
        return self.op('Flatten', [self.op('Transpose', [x], perm=[0, 2, 3, 1])], axis=1)


def build(params: dict, blocks: int) -> onnx.ModelProto:
    g = Graph()
    x = g.op('Transpose', ['planes'], perm=[0, 3, 1, 2])                  # NHWC -> NCHW
    x = g.conv(x, params['stem'])
    film = g.dense('features', params['film'])                           # [B, C]
    film = g.op('Unsqueeze', [film, g.const(np.array([2, 3], np.int64))])
    x = g.op('Add', [x, film])
    for i in range(blocks):
        p = params['blocks'][i]
        y = g.conv(g.relu(g.layernorm_c(x, p['norm1'])), p['conv1'])
        y = g.conv(g.relu(g.layernorm_c(y, p['norm2'])), p['conv2'])
        x = g.op('Add', [x, y])
    x = g.relu(g.layernorm_c(x, params['norm']))
    pooled = g.op('ReduceMean', [x, g.const(np.array([2, 3], np.int64))], keepdims=0)    # [B, C]

    pol = g.flatten_nhwc(g.relu(g.conv(x, params['policy_conv'])))
    logits = g.dense(g.op('Concat', [pol, pooled], axis=1), params['policy_out'])
    val = g.flatten_nhwc(g.relu(g.conv(x, params['value_conv'])))
    val = g.relu(g.dense(g.op('Concat', [val, pooled], axis=1), params['value_hidden']))
    val = g.op('Tanh', [g.dense(val, params['value_out'])])
    g.nodes.append(helper.make_node('Identity', [logits], ['logits']))
    g.nodes.append(helper.make_node('Squeeze', [val, g.const(np.array([1], np.int64))], ['value']))

    graph = helper.make_graph(
        g.nodes, 'DukeZeroNet',
        [helper.make_tensor_value_info('planes', TensorProto.FLOAT, ['B', R.BOARD, R.BOARD, E.N_PLANES]),
         helper.make_tensor_value_info('features', TensorProto.FLOAT, ['B', E.N_FEATURES])],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['B', R.N_ACTIONS]),
         helper.make_tensor_value_info('value', TensorProto.FLOAT, ['B'])],
        g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)])
    model.ir_version = 9            # widely supported by onnxruntime-web
    onnx.checker.check_model(model)
    return model


def sample_inputs(n=8, seed=0):
    """Real positions: from the benchmark if built, else random play."""
    from training.zero import benchmark as BM
    if BM.BENCH.exists():
        b = BM.load_benchmark()
        idx = np.random.default_rng(seed).choice(len(b['ply']), n, replace=False)
        states = BM.states_from_arrays({k: b[k][idx] for k in BM.STATE_KEYS})
    else:
        states = jax.vmap(E.init)(jax.random.split(jax.random.key(seed), n))
    planes, feats = observe_batch(states)
    return np.asarray(planes, np.float32), np.asarray(feats, np.float32), np.asarray(states.legal)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', nargs='?', type=Path)
    p.add_argument('out', type=Path, help='output path prefix, e.g. web/latency/models/az3net')
    p.add_argument('--random', metavar='CxB', help='random weights of this size instead of a checkpoint')
    a = p.parse_args()

    if a.random:
        c, b = (int(v) for v in a.random.split('x'))
        model32 = DukeZeroNet(c, b, dtype=jnp.float32, rngs=nnx.Rngs(0))
        _, params = nnx.split(model32)
        # Random (not zero) heads, so outputs aren't trivially constant.
        params = jax.tree.map(lambda x: x + 0.02 * jax.random.normal(jax.random.key(1), x.shape), params)
        label = f'random {c}x{b}'
    else:
        from training.zero.train import load_params
        _, params, ckpt = load_params(a.checkpoint)
        c, b = ckpt['config']['channels'], ckpt['config']['blocks']
        label = f'{a.checkpoint.parent.name} iteration {ckpt["iteration"]} ({c}x{b})'
    pure = nnx.to_pure_dict(params)
    pure = jax.tree.map(np.asarray, pure)
    pure['blocks'] = {int(k): v for k, v in pure['blocks'].items()}

    model = build(pure, b)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    f32 = a.out.with_name(a.out.name + '.fp32.onnx')
    onnx.save(model, f32)

    # Check against JAX in float32 (training computes convolutions in bfloat16).
    import onnxruntime as ort
    ref = DukeZeroNet(c, b, dtype=jnp.float32, rngs=nnx.Rngs(0))
    gd, ref_params = nnx.split(ref)
    nnx.replace_by_pure_dict(ref_params, jax.tree.map(jnp.asarray, nnx.to_pure_dict(params)))
    planes, feats, legal = sample_inputs()
    want_logits, want_value = (np.asarray(x) for x in make_forward(gd)(ref_params, jnp.asarray(planes), jnp.asarray(feats)))
    sess = ort.InferenceSession(str(f32), providers=['CPUExecutionProvider'])
    got_logits, got_value = sess.run(None, {'planes': planes, 'features': feats})
    err = float(max(np.abs(got_logits - want_logits).max(), np.abs(got_value - want_value).max()))
    print(f'{label}: fp32 max |onnx - jax| = {err:.2e}')
    assert err < 1e-3, err

    # fp16: half the download, and the fast path on WebGPU.
    from onnxruntime.transformers.float16 import convert_float_to_float16
    f16 = a.out.with_name(a.out.name + '.fp16.onnx')
    onnx.save(convert_float_to_float16(model, keep_io_types=True), f16)
    # int8: dense layers (mostly the 3,605-way policy layer) to 8 bits.
    from onnxruntime.quantization import QuantType, quantize_dynamic
    i8 = a.out.with_name(a.out.name + '.int8.onnx')
    quantize_dynamic(str(f32), str(i8), weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])

    variants = {}
    for name, path in (('fp32', f32), ('fp16', f16), ('int8', i8)):
        s = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        lg, v = s.run(None, {'planes': planes, 'features': feats})
        masked = np.where(legal, lg, -np.inf)
        variants[name] = {
            'file': path.name, 'bytes': path.stat().st_size,
            'max_value_err': float(np.abs(v - want_value).max()),
            'top_move_agrees': float((masked.argmax(1) == np.where(legal, want_logits, -np.inf).argmax(1)).mean()),
        }
        print(f'  {name}: {path.stat().st_size / 1e6:.1f} MB, value err {variants[name]["max_value_err"]:.3f}, '
              f'top move agrees {variants[name]["top_move_agrees"]:.2f}')

    n_params = int(sum(np.size(x) for x in jax.tree.leaves(pure)))
    flops = 2 * 36 * 9 * (E.N_PLANES * c + 2 * b * c * c) + 2 * ((36 * 32 + c) * R.N_ACTIONS + (36 * 8 + c) * 128)
    meta = {'label': label, 'channels': c, 'blocks': b, 'params': n_params, 'mflops_per_eval': flops / 1e6,
            'variants': variants,
            'check': {'planes': planes.reshape(-1).tolist(), 'features': feats.reshape(-1).tolist(),
                      'n': int(len(planes)), 'value': want_value.tolist(),
                      'top_move': np.where(legal, want_logits, -np.inf).argmax(1).tolist(),
                      'legal': [np.flatnonzero(l).tolist() for l in legal]}}
    a.out.with_name(a.out.name + '.json').write_text(json.dumps(meta))
    print(f'  {n_params / 1e6:.1f}M parameters, {flops / 1e6:.0f} MFLOP per evaluation')


if __name__ == '__main__':
    main()
