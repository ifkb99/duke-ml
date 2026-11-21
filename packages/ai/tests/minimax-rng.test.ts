import { describe, it, expect, beforeEach } from 'vitest';
import {
  type GameState,
  createInitialState, applySetupPlacement, resetInstanceCounter,
} from '@the-duke/engine';
import { findBestMove, scoreRootMoves, ttClear } from '../src/minimax.js';

/** mulberry32: small seeded PRNG, uniform in [0, 1). */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Wrap an RNG and count how often the search calls it. */
function spy(rng: () => number): { rng: () => number; calls: () => number } {
  let n = 0;
  return { rng: () => { n++; return rng(); }, calls: () => n };
}

/**
 * Position right after setup: P1 has 16 tiles in the bag and the search
 * prefers drawing, so the root resolves a draw with the RNG.
 */
function openingState(): GameState {
  let s = createInitialState();
  for (const [row, col] of [[0, 2], [0, 1], [1, 2], [5, 3], [5, 4], [4, 3]]) {
    s = applySetupPlacement(s, { row, col });
  }
  return s;
}

describe('findBestMove with an injected RNG', () => {
  beforeEach(() => resetInstanceCounter());

  it('is reproducible with the same seed (state cleared between searches)', () => {
    const state = openingState();

    ttClear();
    const a = spy(seeded(42));
    const first = findBestMove(state, 3, a.rng);

    ttClear();
    const b = spy(seeded(42));
    const second = findBestMove(state, 3, b.rng);

    // The RNG is actually exercised: the search chose to draw.
    expect(a.calls()).toBeGreaterThan(0);
    expect(first.move?.type).toBe('place');

    expect(second.move).toEqual(first.move);
    expect(second.nodesSearched).toBe(first.nodesSearched);
    expect(second.score).toBe(first.score);
    expect(b.calls()).toBe(a.calls());
  });

  it('uses the RNG to pick the drawn tile', () => {
    const state = openingState();
    const bag = state.bags.P1;

    ttClear();
    const low = findBestMove(state, 2, () => 0);
    ttClear();
    const high = findBestMove(state, 2, () => 0.999999);

    expect(low.move).toMatchObject({ type: 'place', tileName: bag[0] });
    expect(high.move).toMatchObject({ type: 'place', tileName: bag[bag.length - 1] });
  });

  it('keeps the positional maxDepth parameter working without an RNG', () => {
    ttClear();
    const result = findBestMove(openingState(), 2);
    expect(result.move).not.toBeNull();
    expect(result.nodesSearched).toBeGreaterThan(0);
  });
});

describe('scoreRootMoves with an injected RNG', () => {
  beforeEach(() => resetInstanceCounter());

  it('is reproducible with the same seed and passes the RNG to its warm-up search', () => {
    const state = openingState();

    ttClear();
    const a = spy(seeded(7));
    const first = scoreRootMoves(state, 3, a.rng);

    ttClear();
    const second = scoreRootMoves(state, 3, seeded(7));

    expect(a.calls()).toBeGreaterThan(0);
    expect(second).toEqual(first);
  });
});
