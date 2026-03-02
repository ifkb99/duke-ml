import { describe, it, expect, beforeEach } from 'vitest';
import {
  type GameState, type TileInstance,
  createEmptyBoard, BOARD_SIZE, generateAllMoves, isSquareAttackedBy,
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

    // P2 Footman at (3,1) attacks (3,2) via its R step, so Duke sliding
    // to (3,2) would be moving into check — only the capture at (3,1) is legal.
    expect(dukeSlideLeft).toHaveLength(1);
    expect(dukeSlideLeft[0].type === 'move' && dukeSlideLeft[0].to.col).toBe(1);
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
    // Longbowman side B has strikes at F2, F3 (P1's forward = toward row 5)
    placeTile(state, {
      id: 'P1-Longbowman-1', defName: 'Longbowman', owner: 'P1', side: 'B',
      position: { row: 2, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 0, col: 0 },
    });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 4, col: 3 },
    });

    const moves = generateAllMoves(state);
    const strikes = moves.filter(m => m.type === 'strike');

    // Longbowman B: strike F2 ({row:4,col:3}) and F3 ({row:5,col:3})
    // Only F2 has an enemy, F3 is empty → 1 strike
    expect(strikes).toHaveLength(1);
    expect(strikes[0].type === 'strike' && strikes[0].target.row).toBe(4);
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

// --- Check and checkmate tests ---

describe('isSquareAttackedBy', () => {
  it('detects step attack', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 4, col: 3 },
    });

    // P2 Footman side A: B(P2) = {dRow:-1} → attacks (3,3)
    expect(isSquareAttackedBy(state, { row: 3, col: 3 }, 'P2')).toBe(true);
    expect(isSquareAttackedBy(state, { row: 2, col: 3 }, 'P2')).toBe(false);
  });

  it('detects slide attack', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 3, col: 5 },
    });

    // P2 Duke side A slides L,R. L from (3,5): attacks (3,4),(3,3),(3,2),(3,1),(3,0)
    expect(isSquareAttackedBy(state, { row: 3, col: 0 }, 'P2')).toBe(true);
    expect(isSquareAttackedBy(state, { row: 2, col: 5 }, 'P2')).toBe(false);
  });

  it('slide attack blocked by intervening piece', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 3, col: 5 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 2 },
    });

    // P2 Duke slides left, blocked by P1 Footman at (3,2). Can capture at (3,2), not reach (3,1).
    expect(isSquareAttackedBy(state, { row: 3, col: 2 }, 'P2')).toBe(true);
    expect(isSquareAttackedBy(state, { row: 3, col: 1 }, 'P2')).toBe(false);
  });

  it('detects strike attack', () => {
    const state = makeState();
    // Longbowman side B strikes F2, F3 (P2's forward = toward row 0)
    placeTile(state, {
      id: 'P2-Longbowman-1', defName: 'Longbowman', owner: 'P2', side: 'B',
      position: { row: 5, col: 3 },
    });
    // Put a P1 piece at the target so strike is generated
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });

    // P2 Longbowman side B: F for P2 = dRow-1. F2 from (5,3) = (3,3)
    expect(isSquareAttackedBy(state, { row: 3, col: 3 }, 'P2')).toBe(true);
  });

  it('returns false when no attackers', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });

    expect(isSquareAttackedBy(state, { row: 3, col: 3 }, 'P2')).toBe(false);
  });
});

describe('Check legality filtering', () => {
  it('Duke cannot move into a square attacked by an enemy', () => {
    const state = makeState();
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    // P2 Footman at (4,4): B(P2)={dRow:-1} attacks (3,4)
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 4, col: 4 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 5, col: 5 },
    });

    const moves = generateAllMoves(state);
    const dukeMoves = moves.filter(m => m.type === 'move');
    const targets = dukeMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');

    // Duke slides L,R from (3,3). (3,4) attacked by P2 Footman → filtered.
    expect(targets).not.toContain('3,4');
    // Other slides are legal: (3,2), (3,1), (3,0), (3,5)
    expect(targets).toContain('3,2');
    expect(targets).toContain('3,1');
    expect(targets).toContain('3,0');
    expect(targets).toContain('3,5');
    expect(dukeMoves).toHaveLength(4);
  });

  it('piece pinned against Duke can only move along the pin line', () => {
    const state = makeState();
    // P1 Duke at (3,0), P1 Footman at (3,2), P2 Duke at (3,5)
    // P2 Duke's slide pins the Footman: moving off row 3 exposes P1 Duke.
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 0 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 3, col: 2 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 3, col: 5 },
    });

    const moves = generateAllMoves(state);
    const footmanMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 3 && m.from.col === 2,
    );
    const targets = footmanMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');

    // Moving off the pin line (to row 2 or row 4) exposes Duke → illegal
    expect(targets).not.toContain('2,2');
    expect(targets).not.toContain('4,2');
    // Moving along pin line keeps Duke protected
    expect(targets).toContain('3,1');
    expect(targets).toContain('3,3');
    expect(footmanMoves).toHaveLength(2);
  });

  it('when in check, only moves that resolve check are legal', () => {
    const state = makeState();
    // P1 Duke at (3,3) in check from P2 Footman at (4,3).
    // P2 Footman side A: B(P2) = {dRow:-1} → attacks (3,3).
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    // P1 Footman at (2,2) — can it help?
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 2, col: 2 },
    });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 4, col: 3 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 5, col: 5 },
    });

    const moves = generateAllMoves(state);

    // All moves must resolve check on (3,3)
    for (const m of moves) {
      // Verify each move is legal (Duke is safe after)
      expect(m.type).not.toBe('place');
    }

    // Duke can slide away: (3,2),(3,1),(3,0),(3,4),(3,5)
    const dukeMoves = moves.filter(m => m.type === 'move' && m.from.row === 3 && m.from.col === 3);
    expect(dukeMoves.length).toBeGreaterThanOrEqual(1);

    // Footman at (2,2) can step to (3,2) — does that resolve check?
    // No: Duke still at (3,3) attacked by P2 Footman at (4,3). Moving Footman doesn't help.
    // But Footman can step B to (3,2)... Duke stays at (3,3), still attacked. Illegal.
    // Footman can capture attacker if adjacent: (2,2) → step B(3,2), R(2,3), F(1,2), L(2,1).
    // None reach (4,3). So no Footman moves help.
    const footmanMoves = moves.filter(m => m.type === 'move' && m.from.row === 2 && m.from.col === 2);
    expect(footmanMoves).toHaveLength(0);
  });

  it('capturing the checking piece resolves check', () => {
    const state = makeState();
    // P1 Duke at (3,3) in check from P2 Footman at (4,3)
    // P1 Footman at (4,2) can capture the checker at (4,3)
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 3 },
    });
    placeTile(state, {
      id: 'P1-Footman-1', defName: 'Footman', owner: 'P1', side: 'A',
      position: { row: 4, col: 2 },
    });
    placeTile(state, {
      id: 'P2-Footman-1', defName: 'Footman', owner: 'P2', side: 'A',
      position: { row: 4, col: 3 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 5, col: 5 },
    });

    const moves = generateAllMoves(state);

    // Footman at (4,2) can step R to (4,3) — captures the checker
    const footmanMoves = moves.filter(
      m => m.type === 'move' && m.from.row === 4 && m.from.col === 2,
    );
    const targets = footmanMoves.map(m => m.type === 'move' ? `${m.to.row},${m.to.col}` : '');
    expect(targets).toContain('4,3');
  });

  it('placement can block a slide-based check', () => {
    const state = makeState({
      bags: { P1: ['Footman'], P2: [] },
    });
    // P1 Duke at (3,0) in check from P2 Duke sliding from (3,5)
    placeTile(state, {
      id: 'P1-Duke', defName: 'Duke', owner: 'P1', side: 'A',
      position: { row: 3, col: 0 },
    });
    placeTile(state, {
      id: 'P2-Duke', defName: 'Duke', owner: 'P2', side: 'A',
      position: { row: 3, col: 5 },
    });

    const moves = generateAllMoves(state);
    const placements = moves.filter(m => m.type === 'place');

    // Placing adjacent to P1 Duke: (2,0), (4,0), (3,1)
    // (3,1) blocks the slide from P2 Duke! Legal.
    // (2,0) and (4,0) don't block — Duke at (3,0) still attacked. Illegal.
    expect(placements).toHaveLength(1);
    expect(placements[0].type === 'place' && placements[0].position).toEqual({ row: 3, col: 1 });
  });
});
