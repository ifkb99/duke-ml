import { describe, it, expect, beforeEach } from 'vitest';
import {
  type GameState, type TileInstance,
  createEmptyBoard, createInitialState,
  applySetupPlacement, generateAllMoves, applyMove,
  resetInstanceCounter,
} from '@the-duke/engine';
import { pickRandomMove, findBestMove, evaluate } from '../src/index.js';

function makeState(overrides: Partial<GameState> = {}): GameState {
  return {
    board: createEmptyBoard(),
    tiles: new Map(),
    bags: { P1: [], P2: [] },
    currentPlayer: 'P1',
    turnNumber: 1,
    status: 'active',
    ...overrides,
  };
}

function placeTile(state: GameState, tile: TileInstance): void {
  state.tiles.set(tile.id, tile);
  state.board[tile.position.row][tile.position.col] = tile.id;
}

/** Run through the 6-step setup with fixed positions. */
function setupGame(): GameState {
  let s = createInitialState();
  // P1 Duke (0,2), Footmen (0,1), (1,2)
  s = applySetupPlacement(s, { row: 0, col: 2 });
  s = applySetupPlacement(s, { row: 0, col: 1 });
  s = applySetupPlacement(s, { row: 1, col: 2 });
  // P2 Duke (5,3), Footmen (5,4), (4,3)
  s = applySetupPlacement(s, { row: 5, col: 3 });
  s = applySetupPlacement(s, { row: 5, col: 4 });
  s = applySetupPlacement(s, { row: 4, col: 3 });
  return s;
}

describe('pickRandomMove', () => {
  beforeEach(() => resetInstanceCounter());

  it('returns a legal move from an active game', () => {
    const state = setupGame();
    const move = pickRandomMove(state);
    expect(move).not.toBeNull();
    const legal = generateAllMoves(state);
    expect(legal).toContainEqual(move);
  });

  it('returns null when no moves are available', () => {
    const state = makeState({ status: 'P1_wins' });
    // Game is over, generateAllMoves returns []
    expect(pickRandomMove(state)).toBeNull();
  });
});

describe('evaluate', () => {
  beforeEach(() => resetInstanceCounter());

  it('returns large positive for P1 win', () => {
    const state = makeState({ status: 'P1_wins' });
    expect(evaluate(state)).toBeGreaterThan(10_000);
  });

  it('returns large negative for P2 win', () => {
    const state = makeState({ status: 'P2_wins' });
    expect(evaluate(state)).toBeLessThan(-10_000);
  });

  it('favors the side with more material', () => {
    const state = makeState({
      bags: { P1: ['Footman'], P2: ['Footman'] },
    });
    placeTile(state, {
      id: 'P1-Duke-1', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 2 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 1, col: 2 },
    });
    placeTile(state, {
      id: 'P2-Duke-1', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 5, col: 3 },
    });
    // P1 has Duke + Footman, P2 has only Duke → should favor P1
    expect(evaluate(state)).toBeGreaterThan(0);
  });
});

describe('findBestMove', () => {
  beforeEach(() => resetInstanceCounter());

  it('returns a legal move', () => {
    const state = setupGame();
    const result = findBestMove(state, 2);
    expect(result.move).not.toBeNull();
    const legal = generateAllMoves(state);
    expect(legal).toContainEqual(result.move);
  });

  it('tracks nodes searched', () => {
    const state = setupGame();
    const result = findBestMove(state, 2);
    expect(result.nodesSearched).toBeGreaterThan(0);
  });

  it('chooses a winning capture when available', () => {
    // P1 Footman can step onto P2 Duke to win
    const state = makeState({ currentPlayer: 'P1' });
    placeTile(state, {
      id: 'P1-Duke-1', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 4, col: 3 },
    });
    placeTile(state, {
      id: 'P2-Duke-1', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 5, col: 3 },
    });

    const result = findBestMove(state, 2);
    expect(result.move).not.toBeNull();
    // The winning move captures the Duke
    expect(result.move!.type).toBe('move');
    if (result.move!.type === 'move') {
      expect(result.move!.to).toEqual({ row: 5, col: 3 });
    }
    expect(result.score).toBeGreaterThan(10_000);
  });

  it('works when AI plays as P2', () => {
    const state = setupGame();
    // Advance to P2's turn by making one move
    const moves = generateAllMoves(state);
    const afterP1 = applyMove(state, moves[0]);
    expect(afterP1.currentPlayer).toBe('P2');

    const result = findBestMove(afterP1, 2);
    expect(result.move).not.toBeNull();
    const legal = generateAllMoves(afterP1);
    expect(legal).toContainEqual(result.move);
  });

  it('returns null move when no moves available', () => {
    const state = makeState();
    // No tiles on board, no bag → no moves
    const result = findBestMove(state, 2);
    expect(result.move).toBeNull();
  });
});
