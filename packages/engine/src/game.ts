import type {Coord, GameMove, GameState, SetupPhase, Side, TileInstance, UndoRecord} from './types.js';
import {BOARD_SIZE, cloneState} from './state.js';
import {findDukePosition, hasAnyLegalMove} from './moves.js';

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
      const duke = findDukePosition(state, 'P1');
      return duke ? adjacentEmpty(state, duke) : [];
    }
    case 'p2_footman1':
    case 'p2_footman2': {
      const duke = findDukePosition(state, 'P2');
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
 * Apply a move immutably — clones state, applies in-place via makeMove,
 * then checks for checkmate/stalemate.
 */
export function applyMove(state: GameState, move: GameMove): GameState {
  const newState = cloneState(state);
  makeMove(newState, move);

  if (newState.status !== 'active') return newState;

  if (!hasAnyLegalMove(newState)) {
    newState.status = newState.currentPlayer === 'P1' ? 'P2_wins' : 'P1_wins';
  }

  return newState;
}

// ---------------------------------------------------------------------------
// Make / Unmake (in-place mutation for search)
// ---------------------------------------------------------------------------

/**
 * Apply a move in-place to the given state, returning an UndoRecord
 * that can be passed to unmakeMove to restore the previous state.
 * Does NOT perform checkmate/stalemate detection (except Duke capture).
 */
export function makeMove(state: GameState, move: GameMove): UndoRecord {
  const undo: UndoRecord = {
    move,
    prevCurrentPlayer: state.currentPlayer,
    prevTurnNumber: state.turnNumber,
    prevStatus: state.status,
    movedTileId: null,
    prevSide: null,
    prevPosition: null,
    capturedTileId: null,
    capturedTile: null,
    placedTileId: null,
    removedFromBag: null,
    commandedTileId: null,
    commandedPrevSide: null,
    commandedPrevPosition: null,
    commandCapturedTileId: null,
    commandCapturedTile: null,
  };

  switch (move.type) {
    case 'place': {
      const player = state.currentPlayer;
      const bag = state.bags[player];
      const idx = bag.indexOf(move.tileName);
      if (idx === -1) throw new Error(`Tile ${move.tileName} not in ${player}'s bag`);
      undo.removedFromBag = {player, name: move.tileName, index: idx};
      bag.splice(idx, 1);

      const id = `${player}-${move.tileName}-${++nextInstanceCounter}`;
      const tile: TileInstance = {
        defName: move.tileName,
        owner: player,
        side: 'A' as Side,
        position: {row: move.position.row, col: move.position.col},
        id,
      };
      state.tiles.set(id, tile);
      state.board[move.position.row][move.position.col] = id;
      undo.placedTileId = id;
      break;
    }
    case 'move': {
      const fromId = state.board[move.from.row][move.from.col]!;
      const tile = state.tiles.get(fromId)!;
      undo.movedTileId = fromId;
      undo.prevSide = tile.side;
      undo.prevPosition = {row: tile.position.row, col: tile.position.col};

      // Capture
      const toId = state.board[move.to.row][move.to.col];
      if (toId) {
        const captured = state.tiles.get(toId)!;
        undo.capturedTileId = toId;
        undo.capturedTile = {...captured, position: {...captured.position}};
        state.board[captured.position.row][captured.position.col] = null;
        state.tiles.delete(toId);
        if (captured.defName === 'Duke') {
          state.status = captured.owner === 'P1' ? 'P2_wins' : 'P1_wins';
        }
      }

      state.board[move.from.row][move.from.col] = null;
      state.board[move.to.row][move.to.col] = fromId;
      tile.position.row = move.to.row;
      tile.position.col = move.to.col;
      tile.side = tile.side === 'A' ? 'B' : 'A';
      break;
    }
    case 'strike': {
      const fromId = state.board[move.from.row][move.from.col]!;
      const striker = state.tiles.get(fromId)!;
      undo.movedTileId = fromId;
      undo.prevSide = striker.side;
      undo.prevPosition = {row: striker.position.row, col: striker.position.col};

      const targetId = state.board[move.target.row][move.target.col]!;
      const captured = state.tiles.get(targetId)!;
      undo.capturedTileId = targetId;
      undo.capturedTile = {...captured, position: {...captured.position}};
      state.board[captured.position.row][captured.position.col] = null;
      state.tiles.delete(targetId);
      if (captured.defName === 'Duke') {
        state.status = captured.owner === 'P1' ? 'P2_wins' : 'P1_wins';
      }

      striker.side = striker.side === 'A' ? 'B' : 'A';
      break;
    }
    case 'command': {
      const commandedId = state.board[move.target.row][move.target.col]!;
      const commanded = state.tiles.get(commandedId)!;
      undo.commandedTileId = commandedId;
      undo.commandedPrevSide = commanded.side;
      undo.commandedPrevPosition = {row: commanded.position.row, col: commanded.position.col};

      // Capture at destination
      const destId = state.board[move.targetTo.row][move.targetTo.col];
      if (destId) {
        const captured = state.tiles.get(destId)!;
        undo.commandCapturedTileId = destId;
        undo.commandCapturedTile = {...captured, position: {...captured.position}};
        state.board[captured.position.row][captured.position.col] = null;
        state.tiles.delete(destId);
        if (captured.defName === 'Duke') {
          state.status = captured.owner === 'P1' ? 'P2_wins' : 'P1_wins';
        }
      }

      state.board[move.target.row][move.target.col] = null;
      state.board[move.targetTo.row][move.targetTo.col] = commandedId;
      commanded.position.row = move.targetTo.row;
      commanded.position.col = move.targetTo.col;
      commanded.side = commanded.side === 'A' ? 'B' : 'A';
      break;
    }
  }

  // Switch player + increment turn (if game still active)
  if (state.status === 'active') {
    state.currentPlayer = state.currentPlayer === 'P1' ? 'P2' : 'P1';
    state.turnNumber++;
  }

  return undo;
}

/**
 * Reverse a makeMove using the undo record. Restores the state exactly.
 */
export function unmakeMove(state: GameState, undo: UndoRecord): void {
  // Restore player/turn/status
  state.currentPlayer = undo.prevCurrentPlayer;
  state.turnNumber = undo.prevTurnNumber;
  state.status = undo.prevStatus;

  const move = undo.move;

  switch (move.type) {
    case 'place': {
      // Remove placed tile from board and tiles map
      const id = undo.placedTileId!;
      state.board[move.position.row][move.position.col] = null;
      state.tiles.delete(id);
      // Re-insert into bag at original index
      const bag = undo.removedFromBag!;
      state.bags[bag.player].splice(bag.index, 0, bag.name);
      break;
    }
    case 'move': {
      const id = undo.movedTileId!;
      const tile = state.tiles.get(id)!;
      // Use undo.prevPosition instead of move.from — move.from may be a shared
      // reference to tile.position which was mutated in-place by makeMove.
      const prevPos = undo.prevPosition!;

      // Move tile back
      state.board[move.to.row][move.to.col] = null;
      state.board[prevPos.row][prevPos.col] = id;
      tile.position.row = prevPos.row;
      tile.position.col = prevPos.col;
      tile.side = undo.prevSide!;

      // Restore captured tile
      if (undo.capturedTileId) {
        const cap = undo.capturedTile!;
        state.tiles.set(undo.capturedTileId, cap);
        state.board[cap.position.row][cap.position.col] = undo.capturedTileId;
      }
      break;
    }
    case 'strike': {
      const id = undo.movedTileId!;
      const striker = state.tiles.get(id)!;
      striker.side = undo.prevSide!;

      // Restore captured tile
      const cap = undo.capturedTile!;
      state.tiles.set(undo.capturedTileId!, cap);
      state.board[cap.position.row][cap.position.col] = undo.capturedTileId!;
      break;
    }
    case 'command': {
      const commandedId = undo.commandedTileId!;
      const commanded = state.tiles.get(commandedId)!;

      // Move commanded tile back
      state.board[move.targetTo.row][move.targetTo.col] = null;
      state.board[move.target.row][move.target.col] = commandedId;
      commanded.position.row = undo.commandedPrevPosition!.row;
      commanded.position.col = undo.commandedPrevPosition!.col;
      commanded.side = undo.commandedPrevSide!;

      // Restore captured tile at dest
      if (undo.commandCapturedTileId) {
        const cap = undo.commandCapturedTile!;
        state.tiles.set(undo.commandCapturedTileId, cap);
        state.board[cap.position.row][cap.position.col] = undo.commandCapturedTileId;
      }
      break;
    }
  }
}
