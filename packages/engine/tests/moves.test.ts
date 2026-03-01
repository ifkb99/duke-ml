import { describe, it, expect, beforeEach } from 'vitest';
import {
  type GameState, type TileInstance,
  createEmptyBoard, BOARD_SIZE, generateAllMoves,
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

describe('Step move generation', () => {
  it('generates step moves for a Footman side A (P1)', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    // Also place a Duke so placement moves don't interfere
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });

    const moves = generateAllMoves(state);
    const footmanMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3,
    );

    // Footman A: step F, B, L, R → (2,3), (4,3), (3,2), (3,4)
    expect(footmanMoves).toHaveLength(4);
    const targets = footmanMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');
    expect(targets).toContain('2,3');
    expect(targets).toContain('4,3');
    expect(targets).toContain('3,2');
    expect(targets).toContain('3,4');
  });

  it('blocks step onto friendly tile', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 2, col: 3 },
    });

    const moves = generateAllMoves(state);
    const footmanMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3,
    );

    // Should not include (2,3) because Duke is there
    const targets = footmanMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');
    expect(targets).not.toContain('2,3');
    expect(footmanMoves).toHaveLength(3);
  });

  it('allows step capture of enemy tile', () => {
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

    const moves = generateAllMoves(state);
    const footmanMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3,
    );

    const targets = footmanMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');
    expect(targets).toContain('2,3'); // capture
    expect(footmanMoves).toHaveLength(4);
  });
});

describe('Slide move generation', () => {
  it('generates slide moves for Duke side A', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });

    const moves = generateAllMoves(state);
    const dukeMoves = moves.filter(m => m.type === 'move');

    // Duke A slides L, R from (3,3)
    // L: cols 2,1,0 (3 squares); R: cols 4,5 (2 squares)
    expect(dukeMoves).toHaveLength(5);
  });

  it('slide is blocked by friendly piece', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 1 },
    });

    const moves = generateAllMoves(state);
    const dukeSlideLeft = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3
        && m.to.row === 3 && m.to.col < 3,
    );

    // Blocked at col 1 by friendly, so only col 2
    expect(dukeSlideLeft).toHaveLength(1);
    expect(dukeSlideLeft[0].type === 'move' && dukeSlideLeft[0].to.col).toBe(2);
  });

  it('slide captures first enemy and stops', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 3, col: 1 },
    });

    const moves = generateAllMoves(state);
    const dukeSlideLeft = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3
        && m.to.row === 3 && m.to.col < 3,
    );

    // Can move to col 2 (empty) and col 1 (capture), not col 0
    expect(dukeSlideLeft).toHaveLength(2);
  });
});

describe('Jump move generation', () => {
  it('jumps ignore intervening pieces', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Champion-1', defName: 'Champion', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    // Place blocking piece between champion and jump target
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 2, col: 3 },
    });

    const moves = generateAllMoves(state);
    const champMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3,
    );

    // Champion A: jump F2, B2, L2, R2 → (1,3), (5,3), (3,1), (3,5)
    // F2 target (1,3) is reachable even though (2,3) has friendly
    const targets = champMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');
    expect(targets).toContain('1,3');
  });

  it('jump blocked by friendly at destination', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Champion-1', defName: 'Champion', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 1, col: 3 },
    });

    const moves = generateAllMoves(state);
    const champMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3,
    );

    // Can't jump to (1,3) because friendly Duke is there
    const targets = champMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');
    expect(targets).not.toContain('1,3');
  });
});

describe('Strike move generation', () => {
  it('generates strike only on enemy tiles', () => {
    const state = makeState();
    // Longbowman side B has strikes at F2, F3
    placeTile(state, {
      id: 'P1-Longbowman-1', defName: 'Longbowman', owner: 'P1', side: 'B',
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

    const moves = generateAllMoves(state);
    const strikes = moves.filter(m => m.type === 'strike');

    // Longbowman B: strike F2 ({row:2,col:3}) and F3 ({row:1,col:3})
    // Only F2 has an enemy, F3 is empty → 1 strike
    expect(strikes).toHaveLength(1);
    expect(strikes[0].type === 'strike' && strikes[0].target.row).toBe(2);
  });

  it('no strikes when no enemies in range', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Longbowman-1', defName: 'Longbowman', owner: 'P1', side: 'B',
      position: { row: 4, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });

    const moves = generateAllMoves(state);
    const strikes = moves.filter(m => m.type === 'strike');
    expect(strikes).toHaveLength(0);
  });
});

describe('Player orientation', () => {
  it('P2 moves are mirrored (row negated)', () => {
    const state = makeState({ currentPlayer: 'P2' });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 5, col: 5 },
    });

    const moves = generateAllMoves(state);
    const footmanMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 3,
    );

    // Footman A for P2: step F(+1), B(-1), L(-1), R(+1) → (4,3), (2,3), (3,2), (3,4)
    const targets = footmanMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');
    expect(targets).toContain('4,3'); // P2's "forward"
    expect(targets).toContain('2,3');
    expect(targets).toContain('3,2');
    expect(targets).toContain('3,4');
  });
});

describe('Placement moves', () => {
  it('generates placement adjacent to Duke on empty squares', () => {
    const state = makeState({
      bags: { P1: ['Pikeman', 'Knight'], P2: [] },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 2 },
    });

    const moves = generateAllMoves(state);
    const placements = moves.filter(m => m.type === 'place');

    // Duke at (0,2): adjacent = (1,2), (0,1), (0,3) — (row -1 out of bounds)
    // 3 empty squares × 2 unique tiles = 6 placements
    expect(placements).toHaveLength(6);
  });

  it('no placements when bag is empty', () => {
    const state = makeState({ bags: { P1: [], P2: [] } });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 2 },
    });

    const moves = generateAllMoves(state);
    const placements = moves.filter(m => m.type === 'place');
    expect(placements).toHaveLength(0);
  });

  it('deduplicates bag tile names', () => {
    const state = makeState({
      bags: { P1: ['Pikeman', 'Pikeman'], P2: [] },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 2 },
    });

    const moves = generateAllMoves(state);
    const placements = moves.filter(m => m.type === 'place');

    // 3 empty adjacent squares × 1 unique tile = 3
    expect(placements).toHaveLength(3);
  });
});
