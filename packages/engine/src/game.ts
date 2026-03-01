import type { GameMove, GameState, TileInstance } from './types.js';
import { cloneState } from './state.js';

let nextInstanceCounter = 0;

/** Reset counter — only needed for deterministic testing. */
export function resetInstanceCounter(val = 0): void {
  nextInstanceCounter = val;
}

export function applyMove(state: GameState, move: GameMove): GameState {
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
  move: { type: 'place'; tileName: string; position: { row: number; col: number } },
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
    position: { ...move.position },
    id,
  };
  s.tiles.set(id, tile);
  s.board[move.position.row][move.position.col] = id;
}

function applyMoveTile(
  s: GameState,
  move: { type: 'move'; from: { row: number; col: number }; to: { row: number; col: number } },
): void {
  const fromId = s.board[move.from.row][move.from.col];
  if (!fromId) throw new Error(`No tile at (${move.from.row},${move.from.col})`);

  // Capture if enemy at destination
  const toId = s.board[move.to.row][move.to.col];
  if (toId) {
    removeTile(s, toId);
  }

  // Move tile
  const tile = s.tiles.get(fromId)!;
  s.board[move.from.row][move.from.col] = null;
  s.board[move.to.row][move.to.col] = fromId;
  tile.position = { ...move.to };

  // Flip
  tile.side = tile.side === 'A' ? 'B' : 'A';
}

function applyStrike(
  s: GameState,
  move: { type: 'strike'; from: { row: number; col: number }; target: { row: number; col: number } },
): void {
  const targetId = s.board[move.target.row][move.target.col];
  if (!targetId) throw new Error(`No tile at strike target (${move.target.row},${move.target.col})`);
  removeTile(s, targetId);
  // Striker does NOT flip or move
}

function applyCommand(
  s: GameState,
  move: {
    type: 'command';
    commander: { row: number; col: number };
    target: { row: number; col: number };
    targetTo: { row: number; col: number };
  },
): void {
  const targetId = s.board[move.target.row][move.target.col];
  if (!targetId) throw new Error(`No tile at command target`);

  // Capture at destination
  const destId = s.board[move.targetTo.row][move.targetTo.col];
  if (destId) {
    removeTile(s, destId);
  }

  // Move commanded tile
  const tile = s.tiles.get(targetId)!;
  s.board[move.target.row][move.target.col] = null;
  s.board[move.targetTo.row][move.targetTo.col] = targetId;
  tile.position = { ...move.targetTo };

  // Commanded tile flips. Commander does NOT flip.
  tile.side = tile.side === 'A' ? 'B' : 'A';
}
