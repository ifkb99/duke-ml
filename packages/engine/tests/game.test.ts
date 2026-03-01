import { describe, it, expect, beforeEach } from 'vitest';
import {
  type GameState, type TileInstance,
  createEmptyBoard, createInitialState,
  applyMove, generateAllMoves, resetInstanceCounter,
  serialize, deserialize,
} from '../src/index.js';

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

describe('applyMove', () => {
  beforeEach(() => resetInstanceCounter());

  it('flips tile after a move', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });

    const next = applyMove(state, { type: 'move', from: { row: 3, col: 3 }, to: { row: 2, col: 3 } });
    const footman = next.tiles.get('P1-Footman-1')!;
    expect(footman.side).toBe('B');
    expect(footman.position).toEqual({ row: 2, col: 3 });
  });

  it('does not flip tile on strike', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Longbowman-1', defName: 'Longbowman', owner: 'P1', side: 'A',
      position: { row: 4, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 2, col: 3 },
    });

    const next = applyMove(state, { type: 'strike', from: { row: 4, col: 3 }, target: { row: 2, col: 3 } });

    const lb = next.tiles.get('P1-Longbowman-1')!;
    expect(lb.side).toBe('A'); // no flip
    expect(lb.position).toEqual({ row: 4, col: 3 }); // no move
    expect(next.tiles.has('P2-Footman-1')).toBe(false); // captured
  });

  it('captures enemy tile on move', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 2, col: 3 },
    });

    const next = applyMove(state, { type: 'move', from: { row: 3, col: 3 }, to: { row: 2, col: 3 } });
    expect(next.tiles.has('P2-Footman-1')).toBe(false);
    expect(next.board[2][3]).toBe('P1-Footman-1');
  });

  it('detects win when Duke is captured', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 4, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'B',
      position: { row: 5, col: 3 },
    });

    const next = applyMove(state, { type: 'move', from: { row: 4, col: 3 }, to: { row: 5, col: 3 } });
    expect(next.status).toBe('P1_wins');
  });

  it('places tile from bag correctly', () => {
    const state = makeState({ bags: { P1: ['Pikeman'], P2: [] } });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 2 },
    });

    const next = applyMove(state, { type: 'place', tileName: 'Pikeman', position: { row: 1, col: 2 } });
    expect(next.bags.P1).toHaveLength(0);
    expect(next.board[1][2]).toBeTruthy();
    const placed = next.tiles.get(next.board[1][2]!)!;
    expect(placed.defName).toBe('Pikeman');
    expect(placed.side).toBe('A');
  });

  it('toggles current player after move', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });

    const next = applyMove(state, { type: 'move', from: { row: 3, col: 3 }, to: { row: 2, col: 3 } });
    expect(next.currentPlayer).toBe('P2');
    expect(next.turnNumber).toBe(2);
  });

  it('does not mutate original state', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });

    const original = JSON.stringify(serialize(state));
    applyMove(state, { type: 'move', from: { row: 3, col: 3 }, to: { row: 2, col: 3 } });
    expect(JSON.stringify(serialize(state))).toBe(original);
  });
});

describe('createInitialState', () => {
  it('creates valid initial state', () => {
    const state = createInitialState();
    expect(state.currentPlayer).toBe('P1');
    expect(state.status).toBe('active');
    expect(state.turnNumber).toBe(1);

    // Each player has Duke + 2 Footmen = 6 tiles total
    expect(state.tiles.size).toBe(6);

    // Bags have 18 tiles each (3 Footmen + 3 Pikemen + 12 others)
    expect(state.bags.P1).toHaveLength(18);
    expect(state.bags.P2).toHaveLength(18);
  });

  it('generates legal moves from initial state', () => {
    const state = createInitialState();
    const moves = generateAllMoves(state);
    expect(moves.length).toBeGreaterThan(0);
  });
});

describe('serialization', () => {
  it('round-trips correctly', () => {
    const state = createInitialState();
    const serialized = serialize(state);
    const restored = deserialize(serialized);

    expect(restored.currentPlayer).toBe(state.currentPlayer);
    expect(restored.turnNumber).toBe(state.turnNumber);
    expect(restored.status).toBe(state.status);
    expect(restored.tiles.size).toBe(state.tiles.size);
    expect(restored.board).toEqual(state.board);
  });
});
