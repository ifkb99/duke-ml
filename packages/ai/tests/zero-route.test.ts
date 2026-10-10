import { describe, it, expect } from 'vitest';
import { createInitialState } from '@the-duke/engine';
import { findCrossover, routeByBatchSize, toZeroState, type ZeroEvaluate } from '../src/index.js';
import { N_ACTIONS } from '../src/zero/rules.js';

const state = toZeroState(createInitialState());

/** An evaluator that records its batch sizes and labels each output by its position in the batch. */
function recording(sizes: number[]): ZeroEvaluate {
  return async states => {
    sizes.push(states.length);
    return states.map((_, i) => ({ logits: new Float32Array(N_ACTIONS), value: i }));
  };
}

/** A clock advanced by fake evaluators: cost(n) ms for a batch of n. */
function clocked(cost: (n: number) => number) {
  const clock = { t: 0 };
  const evaluate = (c: (n: number) => number): ZeroEvaluate => async states => {
    clock.t += c(states.length);
    return states.map(() => ({ logits: new Float32Array(N_ACTIONS), value: 0 }));
  };
  return { clock, small: evaluate(cost), evaluate };
}

describe('routeByBatchSize', () => {
  it('sends small batches to small, padded large ones to large', async () => {
    const small: number[] = [], large: number[] = [];
    const route = routeByBatchSize(recording(small), recording(large), 5);
    for (const n of [1, 4, 5, 8, 9]) {
      const out = await route(Array(n).fill(state));
      expect(out.map(o => o.value)).toEqual([...Array(n).keys()]);
    }
    expect(small).toEqual([1, 4]);
    expect(large).toEqual([8, 8, 16]);
  });
});

describe('findCrossover', () => {
  it("finds the batch where the GPU's fixed cost pays off", async () => {
    // a phone: CPU 7 ms per position, GPU 25 ms per call + 0.5 per position
    const { clock, small, evaluate } = clocked(n => 7 * n);
    const r = await findCrossover(small, evaluate(n => 25 + 0.5 * n), state, 16, 2, () => clock.t);
    expect(r.small).toEqual({ 1: 7, 2: 14, 4: 28, 8: 56, 16: 112 });
    // 4 positions: 28 ms on the CPU vs 27 on the GPU; 3 (padded to 4): 21 vs 27
    expect(r.minBatch).toBe(4);
  });

  it('keeps everything on the CPU when the GPU never wins', async () => {
    const { clock, small, evaluate } = clocked(n => 1 * n);
    const r = await findCrossover(small, evaluate(n => 25 + n), state, 16, 2, () => clock.t);
    expect(r.minBatch).toBe(Infinity);
  });

  it('uses the GPU for everything when it always wins', async () => {
    const { clock, small, evaluate } = clocked(n => 30 * n);
    const r = await findCrossover(small, evaluate(n => 25 + n), state, 16, 2, () => clock.t);
    expect(r.minBatch).toBe(1);
  });
});
