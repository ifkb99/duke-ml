import { describe, it, expect, beforeEach } from 'vitest';
import { applySetupPlacement, createInitialState, resetInstanceCounter, type GameState } from '@the-duke/engine';
import { TimeBudget, zeroChooseMove, type ZeroEvaluate } from '../src/index.js';
import { N_ACTIONS } from '../src/zero/rules.js';

function setupGame(): GameState {
  let s = createInitialState();
  for (const [row, col] of [[0, 2], [0, 1], [1, 2], [5, 3], [5, 4], [4, 3]]) {
    s = applySetupPlacement(s, { row, col });
  }
  return s;
}

/** A network with no opinions: uniform policy, value 0. */
const flat: ZeroEvaluate = async states =>
  states.map(() => ({ logits: new Float32Array(N_ACTIONS), value: 0 }));

describe('TimeBudget', () => {
  it('starts at its first guess and ignores the warm-up search', () => {
    const b = new TimeBudget(3000);
    expect(b.simulations()).toBe(64);
    b.record(64, 10_000);              // slow first search (shader compilation)
    expect(b.simulations()).toBe(64);
  });

  it('fits the budget from measured speed, within its limits', () => {
    const b = new TimeBudget(3000);
    b.record(64, 640);
    b.record(100, 1000);              // 10 ms per simulation
    expect(b.simulations()).toBe(300);
    const fast = new TimeBudget(3000);
    fast.record(64, 1);
    fast.record(1000, 10);
    expect(fast.simulations()).toBe(4096);
    const slow = new TimeBudget(3000);
    slow.record(64, 1);
    slow.record(10, 10_000);
    expect(slow.simulations()).toBe(16);
  });

  it('adapts as the speed changes', () => {
    const b = new TimeBudget(3000);
    b.record(64, 1);
    b.record(100, 1000);              // 10 ms
    b.record(100, 500);               // 5 ms: the average moves halfway
    expect(b.simulations()).toBe(400);
  });
});

describe('zeroChooseMove', () => {
  beforeEach(() => resetInstanceCounter());

  it('plays with fixed simulations or a budget', async () => {
    const state = setupGame();
    const fixed = await zeroChooseMove(state, flat, { simulations: 32 });
    expect(fixed.info.simulations).toBe(32);
    const budget = new TimeBudget(3000);
    const timed = await zeroChooseMove(state, flat, { budget });
    expect(timed.move).not.toBeNull();
    expect(timed.info.simulations % 64).toBe(0);    // one or two searches at the first guess
    const again = await zeroChooseMove(state, flat, { budget });
    expect(again.info.simulations).toBeGreaterThan(0);
  });

  it('needs simulations or a budget', async () => {
    await expect(zeroChooseMove(setupGame(), flat, {})).rejects.toThrow(/simulations or a budget/);
  });
});
