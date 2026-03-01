import { describe, it, expect, beforeEach } from 'vitest';
import {
  type GameState, type TileInstance,
  createEmptyBoard, createInitialState,
  applyMove, generateAllMoves, resetInstanceCounter,
  serialize, deserialize,
  getSetupTargets, applySetupPlacement,
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

describe('Setup phase', () => {
  beforeEach(() => resetInstanceCounter());

  it('createInitialState starts in setup phase', () => {
    const state = createInitialState();
    expect(state.status).toBe('setup');
    expect(state.setupPhase).toBe('p1_duke');
    expect(state.currentPlayer).toBe('P1');
    expect(state.tiles.size).toBe(0);
    expect(state.bags.P1).toHaveLength(18);
    expect(state.bags.P2).toHaveLength(18);
  });

  it('p1_duke targets are all cells on row 0', () => {
    const state = createInitialState();
    const targets = getSetupTargets(state);
    expect(targets).toHaveLength(6);
    for (const t of targets) expect(t.row).toBe(0);
  });

  it('full setup flow places Duke and 2 Footmen for each player', () => {
    let state = createInitialState();

    // P1 Duke at (0, 2)
    state = applySetupPlacement(state, { row: 0, col: 2 });
    expect(state.setupPhase).toBe('p1_footman1');
    expect(state.currentPlayer).toBe('P1');
    expect(state.tiles.size).toBe(1);

    // P1 Footman 1 at (1, 2) — below Duke
    const ft1Targets = getSetupTargets(state);
    expect(ft1Targets.some(t => t.row === 1 && t.col === 2)).toBe(true);
    state = applySetupPlacement(state, { row: 1, col: 2 });
    expect(state.setupPhase).toBe('p1_footman2');
    expect(state.tiles.size).toBe(2);

    // P1 Footman 2 at (0, 3) — right of Duke
    state = applySetupPlacement(state, { row: 0, col: 3 });
    expect(state.setupPhase).toBe('p2_duke');
    expect(state.currentPlayer).toBe('P2');
    expect(state.tiles.size).toBe(3);

    // P2 Duke at (5, 3)
    const p2DukeTargets = getSetupTargets(state);
    expect(p2DukeTargets).toHaveLength(6);
    for (const t of p2DukeTargets) expect(t.row).toBe(5);
    state = applySetupPlacement(state, { row: 5, col: 3 });
    expect(state.setupPhase).toBe('p2_footman1');

    // P2 Footman 1 at (4, 3)
    state = applySetupPlacement(state, { row: 4, col: 3 });
    expect(state.setupPhase).toBe('p2_footman2');

    // P2 Footman 2 at (5, 2)
    state = applySetupPlacement(state, { row: 5, col: 2 });

    // Setup complete — game is now active
    expect(state.status).toBe('active');
    expect(state.setupPhase).toBeUndefined();
    expect(state.currentPlayer).toBe('P1');
    expect(state.turnNumber).toBe(1);
    expect(state.tiles.size).toBe(6);

    // Each bag lost 2 Footmen
    expect(state.bags.P1).toHaveLength(16);
    expect(state.bags.P2).toHaveLength(16);
  });

  it('footman targets are orthogonally adjacent to Duke', () => {
    let state = createInitialState();
    state = applySetupPlacement(state, { row: 0, col: 0 });

    const targets = getSetupTargets(state);
    expect(targets).toContainEqual({ row: 1, col: 0 });
    expect(targets).toContainEqual({ row: 0, col: 1 });
    // (0,-1) and (-1,0) are out of bounds
    expect(targets).toHaveLength(2);
  });

  it('rejects invalid setup placement', () => {
    const state = createInitialState();
    expect(() => applySetupPlacement(state, { row: 3, col: 3 })).toThrow();
  });

  it('second footman cannot overlap first', () => {
    let state = createInitialState();
    state = applySetupPlacement(state, { row: 0, col: 2 }); // Duke
    state = applySetupPlacement(state, { row: 1, col: 2 }); // Footman 1

    const targets = getSetupTargets(state);
    // (1,2) is occupied by Footman 1, should not be in targets
    expect(targets).not.toContainEqual({ row: 1, col: 2 });
    // Should still have (0,1), (0,3) available
    expect(targets).toContainEqual({ row: 0, col: 1 });
    expect(targets).toContainEqual({ row: 0, col: 3 });
  });

  it('game has legal moves after setup completes', () => {
    let state = createInitialState();
    state = applySetupPlacement(state, { row: 0, col: 2 });
    state = applySetupPlacement(state, { row: 1, col: 2 });
    state = applySetupPlacement(state, { row: 0, col: 3 });
    state = applySetupPlacement(state, { row: 5, col: 3 });
    state = applySetupPlacement(state, { row: 4, col: 3 });
    state = applySetupPlacement(state, { row: 5, col: 2 });

    const moves = generateAllMoves(state);
    expect(moves.length).toBeGreaterThan(0);
  });
});

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
    expect(lb.side).toBe('A');
    expect(lb.position).toEqual({ row: 4, col: 3 });
    expect(next.tiles.has('P2-Footman-1')).toBe(false);
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

describe('Checkmate detection', () => {
  beforeEach(() => resetInstanceCounter());

  it('detects checkmate when a placement delivers check with no escape', () => {
    const state = makeState({ currentPlayer: 'P2', bags: { P1: [], P2: ['Footman'] } });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 0, col: 1 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 2, col: 0 },
    });

    const next = applyMove(state, {
      type: 'place', tileName: 'Footman', position: { row: 1, col: 0 },
    });

    expect(next.status).toBe('P2_wins');
  });

  it('no checkmate when Duke can escape check', () => {
    const state2 = makeState({ currentPlayer: 'P2' });
    placeTile(state2, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state2, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'B',
      position: { row: 3, col: 4 },
    });
    placeTile(state2, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 5, col: 5 },
    });

    const next2 = applyMove(state2, {
      type: 'move', from: { row: 3, col: 4 }, to: { row: 4, col: 3 },
    });

    expect(next2.status).toBe('active');
    const p1moves = generateAllMoves(next2);
    expect(p1moves.length).toBeGreaterThan(0);
  });

  it('detects checkmate from no legal moves (stalemate-like)', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 0, col: 1 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 2, col: 0 },
    });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 1, col: 0 },
    });

    const moves = generateAllMoves(state);
    expect(moves).toHaveLength(0);
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
    expect(restored.setupPhase).toBe(state.setupPhase);
    expect(restored.tiles.size).toBe(state.tiles.size);
    expect(restored.board).toEqual(state.board);
  });
});
