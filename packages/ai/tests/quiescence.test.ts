import { describe, it, expect, beforeEach } from 'vitest';
import {
  type GameState, type TileInstance, type Player, type Side,
  createEmptyBoard, createInitialState, applySetupPlacement, applyMove, resetInstanceCounter,
  generateAllMoves, getStandardSetupTargets, cloneState,
} from '@the-duke/engine';
import { findBestMove, scoreRootMoves, ttClear } from '../src/minimax.js';
import { tileValue } from '../src/evaluate.js';

function emptyState(): GameState {
  return {
    board: createEmptyBoard(),
    tiles: new Map(),
    bags: { P1: [], P2: [] }, // empty bags: no draws, so depth 1 is purely board moves
    currentPlayer: 'P1',
    turnNumber: 10,
    status: 'active',
  };
}

function put(state: GameState, id: string, defName: string, owner: Player, row: number, col: number, side: Side = 'A'): void {
  const tile: TileInstance = { id, defName, owner, side, position: { row, col } };
  state.tiles.set(id, tile);
  state.board[row][col] = id;
}

const PRIEST_TAKES = { type: 'move', from: { row: 1, col: 0 }, to: { row: 3, col: 2 } } as const;

/**
 * Row 0 is P1's back row; P1 moves "down" the board (toward row 5).
 *
 *      0   1   2   3   4   5
 *   0  .   .   .   .   .   D1        D1 = P1 Duke (side A: slides along row 0)
 *   1  P1  .   .   .   .   .         P1 = P1 Priest (side A: slides diagonally), value 10
 *   2  .   .   .   .   .   .
 *   3  .   .   f   .   .   .         f  = P2 Footman, value 4.45, hanging to the Priest via (2,1)
 *   4  .   .   F   .   .   .         F  = P2 Footman defending (3,2) with its forward step
 *   5  .   .   .   .   .   D2        D2 = P2 Duke
 *
 * Depth 1 without quiescence scores Priest x f at the leaf as "+1 Footman" and
 * plays it. With quiescence the leaf is not quiet (F can take back), so the
 * capture is scored as Priest-for-Footman (-5.55) and every quiet move beats it.
 */
function defendedFootman(): GameState {
  const s = emptyState();
  put(s, 'd1', 'Duke', 'P1', 0, 5);
  put(s, 'priest', 'Priest', 'P1', 1, 0);
  put(s, 'bait', 'Footman', 'P2', 3, 2);
  put(s, 'guard', 'Footman', 'P2', 4, 2);
  put(s, 'd2', 'Duke', 'P2', 5, 5);
  return s;
}

/**
 * Same bait, but the only recapture is a *command*: the P2 General (side B)
 * on (2,2) commands the squares (3,1), (3,2), (3,3), (2,1), (2,3) and can move
 * its Pikeman from (3,3) onto (3,2), capturing whatever stands there. Neither
 * the General nor the Pikeman can reach (3,2) on its own, so this checks that
 * a capturing command counts as a noisy move.
 *
 *      0   1   2   3   4   5
 *   0  .   .   .   .   .   D1
 *   1  P1  .   .   .   .   .
 *   2  .   .   G   .   .   .         G  = P2 General, side B (command: BL B BR L R)
 *   3  .   .   f   K   .   .         K  = P2 Pikeman (side A: diagonal forward steps only)
 *   4  .   .   .   .   .   .
 *   5  .   .   .   .   .   D2
 */
function commandDefendedFootman(): GameState {
  const s = emptyState();
  put(s, 'd1', 'Duke', 'P1', 0, 5);
  put(s, 'priest', 'Priest', 'P1', 1, 0);
  put(s, 'general', 'General', 'P2', 2, 2, 'B');
  put(s, 'bait', 'Footman', 'P2', 3, 2);
  put(s, 'pike', 'Pikeman', 'P2', 3, 3);
  put(s, 'd2', 'Duke', 'P2', 5, 5);
  return s;
}

function rootScore(state: GameState, quiescence: boolean, move: typeof PRIEST_TAKES): number {
  ttClear();
  const scored = scoreRootMoves(state, 1, () => 0, { quiescence });
  const entry = scored.moves.find(m => JSON.stringify(m.move) === JSON.stringify(move));
  expect(entry).toBeDefined();
  return entry!.score;
}

describe('quiescence search at the horizon', () => {
  beforeEach(() => ttClear());

  for (const [name, build, recapture] of [
    ['a recapture by a step', defendedFootman, 'step'],
    ['a recapture by a capturing command', commandDefendedFootman, 'command'],
  ] as const) {
    it(`depth 1 avoids a capture that loses the Priest to ${name}`, () => {
      // Fresh state per search: make/unmake restores the position but not the
      // iteration order of state.tiles, which move ordering depends on.
      const state = build();
      const plain = findBestMove(build(), 1, () => 0, { quiescence: false });
      expect(plain.move).toEqual(PRIEST_TAKES);
      expect(plain.quiescenceNodes).toBe(0);

      ttClear();
      const q = findBestMove(build(), 1, () => 0, { quiescence: true });
      expect(q.move).not.toEqual(PRIEST_TAKES);
      // The chosen move is quiet: nothing on its destination.
      expect(q.move?.type).toBe('move');
      if (q.move?.type === 'move') expect(state.board[q.move.to.row][q.move.to.col]).toBeNull();
      expect(q.quiescenceNodes).toBeGreaterThan(0);

      // The capture's own score: plain sees +Footman, quiescence sees the trade.
      // They differ by the Priest's value, give or take the centre bonus.
      const plainScore = rootScore(build(), false, PRIEST_TAKES);
      const qScore = rootScore(build(), true, PRIEST_TAKES);
      expect(Math.abs(plainScore - qScore - tileValue('Priest'))).toBeLessThan(1);
      expect(qScore).toBeLessThan(0);

      // Sanity: the recapture really is of the advertised kind.
      const after = applyMove(state, PRIEST_TAKES);
      ttClear();
      const reply = findBestMove(after, 1, () => 0, { quiescence: false });
      expect(reply.move?.type).toBe(recapture === 'step' ? 'move' : 'command');
    });
  }

  it('leaves quiescence off by default; quiescence: true opts in', () => {
    const byDefault = findBestMove(defendedFootman(), 3, () => 0);
    ttClear();
    const explicitOff = findBestMove(defendedFootman(), 3, () => 0, { quiescence: false });
    expect(byDefault).toEqual(explicitOff);
    expect(byDefault.quiescenceNodes).toBe(0);

    ttClear();
    const on = findBestMove(defendedFootman(), 3, () => 0, { quiescence: true });
    expect(on.quiescenceNodes).toBeGreaterThan(0);
  });

  it('counts quiescence nodes separately and respects the ply cap', () => {
    resetInstanceCounter();
    let state = createInitialState();
    for (const [row, col] of [[0, 2], [0, 1], [1, 2], [5, 3], [5, 4], [4, 3]]) {
      state = applySetupPlacement(state, { row, col });
    }
    const off = findBestMove(cloneState(state), 2, () => 0, { quiescence: false });
    ttClear();
    const noPlies = findBestMove(cloneState(state), 2, () => 0, { quiescence: true, quiescenceDepth: 0 });
    // A cap of 0 is the plain search: same tree, same answer.
    expect(noPlies).toEqual(off);

    ttClear();
    const on = findBestMove(state, 2, () => 0, { quiescence: true });
    expect(on.nodesSearched).toBeGreaterThanOrEqual(on.quiescenceNodes);
    expect(() => findBestMove(state, 1, () => 0, { quiescence: true, quiescenceDepth: -1 })).toThrow();
  });

  it('stays bounded and reproducible on random mid-game positions', () => {
    const rng = seeded(3);
    for (let game = 0; game < 4; game++) {
      resetInstanceCounter();
      let state = createInitialState();
      while (state.status === 'setup') {
        const targets = getStandardSetupTargets(state);
        state = applySetupPlacement(state, targets[Math.floor(rng() * targets.length)]);
      }
      // Random play leaves plenty of tiles en prise, i.e. noisy leaves.
      for (let ply = 0; ply < 16 && state.status === 'active'; ply++) {
        const legal = generateAllMoves(state).filter(m => m.type !== 'draw');
        if (legal.length === 0) break;
        state = applyMove(state, legal[Math.floor(rng() * legal.length)]);
      }
      if (state.status !== 'active') continue;

      ttClear();
      const a = findBestMove(cloneState(state), 3, () => 0.5, { quiescence: true });
      ttClear();
      const b = findBestMove(cloneState(state), 3, () => 0.5, { quiescence: true });
      expect(b).toEqual(a);
      expect(a.move).not.toBeNull();
    }
  });
});

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
