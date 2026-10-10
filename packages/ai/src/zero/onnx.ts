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

/**
 * Sends batches of at least `minBatch` states to `large` and smaller ones to
 * `small`. Meant for WebGPU (`large`), whose every call has a fixed cost of
 * ~25 ms on a phone but whose extra positions are nearly free, next to the
 * CPU (`small`), whose cost grows with each position. `large` gets its batches
 * padded to a power of two, so it sees few shapes: WebGPU compiles shaders per
 * shape.
 */
export function routeByBatchSize(small: Evaluate, large: Evaluate, minBatch: number): Evaluate {
  return async (states: State[]): Promise<NetOutput[]> => {
    if (states.length < minBatch) return small(states);
    const n = 2 ** Math.ceil(Math.log2(states.length));
    const out = await large(states.concat(Array(n - states.length).fill(states[0])));
    return out.slice(0, states.length);
  };
}

export interface Crossover {
  /** smallest batch for `large`; Infinity: `small` is always faster */
  minBatch: number;
  /** median ms per call, by batch size */
  small: Record<number, number>;
  large: Record<number, number>;
}

/**
 * Times both evaluators on batches of 1, 2, 4, ... maxBatch copies of
 * `state` (after a warm-up call at each size) and returns the smallest batch
 * that `routeByBatchSize` should send to `large`: the first size at which
 * `large`, padded to a power of two, beats `small`, whose cost between the
 * measured sizes is interpolated linearly.
 */
export async function findCrossover(small: Evaluate, large: Evaluate, state: State,
                                    maxBatch = 16, reps = 2,
                                    now: () => number = () => performance.now()): Promise<Crossover> {
  async function time(evaluate: Evaluate, n: number): Promise<number> {
    const states = Array<State>(n).fill(state), ms: number[] = [];
    await evaluate(states);
    for (let r = 0; r < reps; r++) {
      const t0 = now();
      await evaluate(states);
      ms.push(now() - t0);
    }
    return ms.sort((a, b) => a - b)[Math.floor((ms.length - 1) / 2)];
  }
  const result: Crossover = {minBatch: Infinity, small: {}, large: {}};
  for (let n = 1; n <= maxBatch; n *= 2) {
    result.small[n] = await time(small, n);
    result.large[n] = await time(large, n);
  }
  for (let n = 1; n <= maxBatch && result.minBatch === Infinity; n++) {
    const lo = 2 ** Math.floor(Math.log2(n)), hi = Math.min(2 * lo, maxBatch);
    const cpu = hi === lo ? result.small[lo] : result.small[lo] + (result.small[hi] - result.small[lo]) * (n - lo) / (hi - lo);
    if (result.large[2 ** Math.ceil(Math.log2(n))] < cpu) result.minBatch = n;
  }
  return result;
}
