/**
 * Network evaluation with an onnxruntime session (onnxruntime-web in the
 * browser, onnxruntime-node in tests), for a model from
 * training/zero/export_onnx.py: inputs planes [B,6,6,60] and features [B,47],
 * outputs logits [B,3605] and value [B].
 *
 * Takes the runtime's Tensor class and session as arguments so this package
 * doesn't depend on a particular onnxruntime build.
 */

import {N_FEATURES, observe, PLANES_SIZE, type State} from './env.js';
import {BOARD, N_ACTIONS} from './rules.js';
import type {Evaluate, NetOutput} from './search.js';

interface OrtTensor { data: unknown }
type TensorCtor = new (type: 'float32', data: Float32Array, dims: number[]) => OrtTensor;
interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

export function onnxEvaluate(session: OrtSession, Tensor: TensorCtor): Evaluate {
  return async (states: State[]): Promise<NetOutput[]> => {
    const b = states.length;
    const planes = new Float32Array(b * PLANES_SIZE), features = new Float32Array(b * N_FEATURES);
    states.forEach((s, i) => observe(s, planes, features, i * PLANES_SIZE, i * N_FEATURES));
    const out = await session.run({
      planes: new Tensor('float32', planes, [b, BOARD, BOARD, PLANES_SIZE / (BOARD * BOARD)]),
      features: new Tensor('float32', features, [b, N_FEATURES]),
    });
    const logits = out.logits.data as Float32Array, value = out.value.data as Float32Array;
    return states.map((_, i) => ({logits: logits.subarray(i * N_ACTIONS, (i + 1) * N_ACTIONS), value: value[i]}));
  };
}
