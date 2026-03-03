import type {Coord, GameMove, GameState, SetupPhase, TileInstance} from './types.js';
import {BOARD_SIZE, cloneState} from './state.js';
import {hasAnyLegalMove} from './moves.js';

let nextInstanceCounter = 0;

/** Reset counter — only needed for deterministic testing. */
export function resetInstanceCounter(val = 0): void {
  nextInstanceCounter = val;
}

// ---------------------------------------------------------------------------
// Setup phase
// ---------------------------------------------------------------------------

const ORTHOGONAL: readonly Coord[] = [
  {row: -1, col: 0}, {row: 1, col: 0},
  {row: 0, col: -1}, {row: 0, col: 1},
];

function findDukeCoord(state: GameState, owner: 'P1' | 'P2'): Coord | null {
  for (const t of state.tiles.values()) {
    if (t.defName === 'Duke' && t.owner === owner) return t.position;
  }
  return null;
}

function adjacentEmpty(state: GameState, origin: Coord): Coord[] {
  const targets: Coord[] = [];
  for (const d of ORTHOGONAL) {
    const r = origin.row + d.row;
    const c = origin.col + d.col;
    if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && !state.board[r][c]) {
      targets.push({row: r, col: c});
    }
  }
  return targets;
}

/** Return valid placement squares for the current setup sub-phase. */
export function getSetupTargets(state: GameState): Coord[] {
  if (state.status !== 'setup' || !state.setupPhase) return [];

  switch (state.setupPhase) {
    case 'p1_duke': {
      const targets: Coord[] = [];
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (!state.board[0][c]) targets.push({row: 0, col: c});
      }
      return targets;
    }
    case 'p2_duke': {
      const targets: Coord[] = [];
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (!state.board[BOARD_SIZE - 1][c]) targets.push({row: BOARD_SIZE - 1, col: c});
      }
      return targets;
    }
    case 'p1_footman1':
    case 'p1_footman2': {
      const duke = findDukeCoord(state, 'P1');
      return duke ? adjacentEmpty(state, duke) : [];
    }
    case 'p2_footman1':
    case 'p2_footman2': {
      const duke = findDukeCoord(state, 'P2');
      return duke ? adjacentEmpty(state, duke) : [];
    }
  }
}

const SETUP_NEXT: Record<SetupPhase, SetupPhase | 'done'> = {
  p1_duke: 'p1_footman1',
  p1_footman1: 'p1_footman2',
  p1_footman2: 'p2_duke',
  p2_duke: 'p2_footman1',
  p2_footman1: 'p2_footman2',
  p2_footman2: 'done',
};

const SETUP_PLAYER: Record<SetupPhase, 'P1' | 'P2'> = {
  p1_duke: 'P1',
  p1_footman1: 'P1',
  p1_footman2: 'P1',
  p2_duke: 'P2',
  p2_footman1: 'P2',
  p2_footman2: 'P2',
};

/**
 * Place a piece during setup. Returns a new state.
 * Validates that `coord` is a valid target for the current sub-phase.
 */
export function applySetupPlacement(state: GameState, coord: Coord): GameState {
  if (state.status !== 'setup' || !state.setupPhase) {
    throw new Error('Not in setup phase');
  }

  const targets = getSetupTargets(state);
  if (!targets.some(t => t.row === coord.row && t.col === coord.col)) {
    throw new Error(`Invalid setup placement at (${coord.row},${coord.col})`);
  }

  const s = cloneState(state);
  const phase = s.setupPhase!;
  const player = SETUP_PLAYER[phase];

  const isDukePhase = phase === 'p1_duke' || phase === 'p2_duke';
  const defName = isDukePhase ? 'Duke' : 'Footman';

  const id = `${player}-${defName}-${++nextInstanceCounter}`;
  const tile: TileInstance = {
    defName,
    owner: player,
    side: 'A',
    position: {...coord},
    id,
  };
  s.tiles.set(id, tile);
  s.board[coord.row][coord.col] = id;

  if (!isDukePhase) {
    const idx = s.bags[player].indexOf('Footman');
    if (idx !== -1) s.bags[player].splice(idx, 1);
  }

  const next = SETUP_NEXT[phase];
  if (next === 'done') {
    s.status = 'active';
    s.setupPhase = undefined;
    s.currentPlayer = 'P1';
    s.turnNumber = 1;
  } else {
    s.setupPhase = next;
    s.currentPlayer = SETUP_PLAYER[next];
  }

  return s;
}

// ---------------------------------------------------------------------------
// Normal gameplay
// ---------------------------------------------------------------------------

/**
 * Apply a move without checkmate detection.
 * Used internally by move-legality filtering to avoid circular calls.
 */
export function applyMoveRaw(state: GameState, move: GameMove): GameState {
  const s = cloneState(state);

  switch (move.type) {
    case 'place':
      applyPlace(s, move);
      break;
    case 'move':
      applyMoveTile(s, move);
      break;
    case 'strike':
      applyStrike(s, move);
      break;
    case 'command':
      applyCommand(s, move);
      break;
  }

  if (s.status === 'active') {
    s.currentPlayer = s.currentPlayer === 'P1' ? 'P2' : 'P1';
    s.turnNumber++;
  }

  return s;
}

/**
 * Apply a move and detect checkmate/stalemate.
 * If the resulting position leaves the new current player with no legal
 * moves, the game ends as a loss for that player.
 */
export function applyMove(state: GameState, move: GameMove): GameState {
  const newState = applyMoveRaw(state, move);

  if (newState.status !== 'active') return newState;

  if (!hasAnyLegalMove(newState)) {
    newState.status = newState.currentPlayer === 'P1' ? 'P2_wins' : 'P1_wins';
  }

  return newState;
}

function removeTile(s: GameState, tileId: string): void {
  const tile = s.tiles.get(tileId);
  if (!tile) return;
  s.board[tile.position.row][tile.position.col] = null;
  s.tiles.delete(tileId);

  if (tile.defName === 'Duke') {
    s.status = tile.owner === 'P1' ? 'P2_wins' : 'P1_wins';
  }
}

function applyPlace(
  s: GameState,
  move: {type: 'place'; tileName: string; position: {row: number; col: number}},
): void {
  const player = s.currentPlayer;
  const bag = s.bags[player];
  const idx = bag.indexOf(move.tileName);
  if (idx === -1) throw new Error(`Tile ${move.tileName} not in ${player}'s bag`);
  bag.splice(idx, 1);

  const id = `${player}-${move.tileName}-${++nextInstanceCounter}`;
  const tile: TileInstance = {
    defName: move.tileName,
    owner: player,
    side: 'A',
    position: {...move.position},
    id,
  };
  s.tiles.set(id, tile);
  s.board[move.position.row][move.position.col] = id;
}

function applyMoveTile(
  s: GameState,
  move: {type: 'move'; from: {row: number; col: number}; to: {row: number; col: number}},
): void {
  const fromId = s.board[move.from.row][move.from.col];
  if (!fromId) throw new Error(`No tile at (${move.from.row},${move.from.col})`);

  const toId = s.board[move.to.row][move.to.col];
  if (toId) {
    removeTile(s, toId);
  }

  const tile = s.tiles.get(fromId)!;
  s.board[move.from.row][move.from.col] = null;
  s.board[move.to.row][move.to.col] = fromId;
  tile.position = {...move.to};

  tile.side = tile.side === 'A' ? 'B' : 'A';
}

function applyStrike(
  s: GameState,
  move: {type: 'strike'; from: {row: number; col: number}; target: {row: number; col: number}},
): void {
  const fromId = s.board[move.from.row][move.from.col];
  if (!fromId) throw new Error(`No tile at (${move.from.row},${move.from.col})`);
  const tile = s.tiles.get(fromId)!;
  const targetId = s.board[move.target.row][move.target.col];
  if (!targetId) throw new Error(`No tile at strike target (${move.target.row},${move.target.col})`);
  removeTile(s, targetId);
  tile.side = tile.side === 'A' ? 'B' : 'A';
}

function applyCommand(
  s: GameState,
  move: {
    type: 'command';
    commander: {row: number; col: number};
    target: {row: number; col: number};
    targetTo: {row: number; col: number};
  },
): void {
  const targetId = s.board[move.target.row][move.target.col];
  if (!targetId) throw new Error(`No tile at command target`);

  const destId = s.board[move.targetTo.row][move.targetTo.col];
  if (destId) {
    removeTile(s, destId);
  }

  const tile = s.tiles.get(targetId)!;
  s.board[move.target.row][move.target.col] = null;
  s.board[move.targetTo.row][move.targetTo.col] = targetId;
  tile.position = {...move.targetTo};

  tile.side = tile.side === 'A' ? 'B' : 'A';
}
