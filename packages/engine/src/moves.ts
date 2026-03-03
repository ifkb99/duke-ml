import type {
  Coord, GameMove, GameState, MovePattern, Offset, Player, TileInstance,
} from './types.js';
import { TILE_REGISTRY } from './tiles.js';
import { BOARD_SIZE } from './state.js';
import { applyMoveRaw } from './game.js';

function inBounds(coord: Coord): boolean {
  return coord.row >= 0 && coord.row < BOARD_SIZE
      && coord.col >= 0 && coord.col < BOARD_SIZE;
}

// Tile offsets are defined with negative dRow = upward (toward row 0 = toward P1's side).
// P2 sits at row 5 facing upward toward P1 → uses offsets directly (F = {dRow:-1} = correct forward).
// P1 sits at row 0 facing downward toward P2 → dRow must be negated (F = {dRow:+1} = correct forward).
function adjustOffset(offset: Offset, player: Player): Offset {
  return player === 'P2' ? offset : { dRow: -offset.dRow, dCol: offset.dCol };
}

function addOffset(coord: Coord, offset: Offset): Coord {
  return { row: coord.row + offset.dRow, col: coord.col + offset.dCol };
}

function tileAt(state: GameState, coord: Coord): TileInstance | null {
  const id = state.board[coord.row][coord.col];
  return id ? state.tiles.get(id) ?? null : null;
}

function isEnemy(tile: TileInstance, player: Player): boolean {
  return tile.owner !== player;
}

function isFriendly(tile: TileInstance, player: Player): boolean {
  return tile.owner === player;
}

// --- Helpers ---

function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

/**
 * For multi-square step moves, check that all intermediate squares are empty.
 * Steps cannot move through pieces (unlike jumps).
 * Only applies when the offset is reducible (gcd > 1), e.g. F2 = 2×F.
 */
function isStepPathClear(state: GameState, from: Coord, adjusted: Offset): boolean {
  const g = gcd(Math.abs(adjusted.dRow), Math.abs(adjusted.dCol));
  if (g <= 1) return true;
  const unitDr = adjusted.dRow / g;
  const unitDc = adjusted.dCol / g;
  for (let i = 1; i < g; i++) {
    const intermediate: Coord = { row: from.row + unitDr * i, col: from.col + unitDc * i };
    if (!inBounds(intermediate)) return false;
    if (state.board[intermediate.row][intermediate.col] !== null) return false;
  }
  return true;
}

// --- Per-pattern generators ---

function generateStepMoves(
  state: GameState, from: Coord, pattern: MovePattern, player: Player,
): GameMove[] {
  const moves: GameMove[] = [];
  for (const offset of pattern.offsets) {
    const adjusted = adjustOffset(offset, player);
    const to = addOffset(from, adjusted);
    if (!inBounds(to)) continue;
    if (!isStepPathClear(state, from, adjusted)) continue;
    const occupant = tileAt(state, to);
    if (!occupant || isEnemy(occupant, player)) {
      moves.push({ type: 'move', from, to });
    }
  }
  return moves;
}

function generateSlideMoves(
  state: GameState, from: Coord, pattern: MovePattern, player: Player,
): GameMove[] {
  const moves: GameMove[] = [];
  for (const dir of pattern.offsets) {
    const adjusted = adjustOffset(dir, player);
    let current = from;
    while (true) {
      current = addOffset(current, adjusted);
      if (!inBounds(current)) break;
      const occupant = tileAt(state, current);
      if (occupant && isFriendly(occupant, player)) break;
      moves.push({ type: 'move', from, to: current });
      if (occupant && isEnemy(occupant, player)) break;
    }
  }
  return moves;
}

function generateJumpMoves(
  state: GameState, from: Coord, pattern: MovePattern, player: Player,
): GameMove[] {
  const moves: GameMove[] = [];
  for (const offset of pattern.offsets) {
    const to = addOffset(from, adjustOffset(offset, player));
    if (!inBounds(to)) continue;
    const occupant = tileAt(state, to);
    if (!occupant || isEnemy(occupant, player)) {
      moves.push({ type: 'move', from, to });
    }
  }
  return moves;
}

function generateStrikeMoves(
  state: GameState, from: Coord, pattern: MovePattern, player: Player,
): GameMove[] {
  const moves: GameMove[] = [];
  for (const offset of pattern.offsets) {
    const target = addOffset(from, adjustOffset(offset, player));
    if (!inBounds(target)) continue;
    const occupant = tileAt(state, target);
    if (occupant && isEnemy(occupant, player)) {
      moves.push({ type: 'strike', from, target });
    }
  }
  return moves;
}

const ORTHOGONAL_DIRS: Offset[] = [
  { dRow: -1, dCol: 0 },
  { dRow: 1, dCol: 0 },
  { dRow: 0, dCol: -1 },
  { dRow: 0, dCol: 1 },
];

function generateCommandMoves(
  state: GameState, commander: Coord, pattern: MovePattern, player: Player,
): GameMove[] {
  const moves: GameMove[] = [];
  for (const offset of pattern.offsets) {
    const target = addOffset(commander, adjustOffset(offset, player));
    if (!inBounds(target)) continue;
    const occupant = tileAt(state, target);
    if (!occupant || !isFriendly(occupant, player)) continue;
    // Cannot command self
    if (target.row === commander.row && target.col === commander.col) continue;

    for (const dir of ORTHOGONAL_DIRS) {
      const dest = addOffset(target, dir);
      if (!inBounds(dest)) continue;
      const destOccupant = tileAt(state, dest);
      // Can't move commanded tile onto the commander's own square
      if (dest.row === commander.row && dest.col === commander.col) continue;
      if (!destOccupant || isEnemy(destOccupant, player)) {
        moves.push({ type: 'command', commander, target, targetTo: dest });
      }
    }
  }
  return moves;
}

/**
 * Jump-slide: same as slide, but allowed to jump over any piece in the way.
 * The piece can land on the 1st, 2nd, 3rd, ... square until the edge of the board
 */
function generateJumpSlideMoves( // TODO: cannot land on closest square
  state: GameState, from: Coord, pattern: MovePattern, player: Player,
): GameMove[] {
  const moves: GameMove[] = [];
  for (const dir of pattern.offsets) {
    const adjusted = adjustOffset(dir, player);
    let current = from;
    while (true) {
      current = addOffset(current, adjusted);
      if (!inBounds(current)) break;
      const occupant = tileAt(state, current);
      if (occupant && isFriendly(occupant, player)) continue;
      moves.push({ type: 'move', from, to: current });
    }
  }
  return moves;
}

function generateTileMoves(state: GameState, tile: TileInstance): GameMove[] {
  const def = TILE_REGISTRY.get(tile.defName);
  if (!def) return [];

  const side = tile.side === 'A' ? def.sideA : def.sideB;
  const moves: GameMove[] = [];

  for (const pattern of side.patterns) {
    switch (pattern.type) {
      case 'step':
        moves.push(...generateStepMoves(state, tile.position, pattern, tile.owner));
        break;
      case 'slide':
        moves.push(...generateSlideMoves(state, tile.position, pattern, tile.owner));
        break;
      case 'jump':
        moves.push(...generateJumpMoves(state, tile.position, pattern, tile.owner));
        break;
      case 'jump_slide':
        moves.push(...generateJumpSlideMoves(state, tile.position, pattern, tile.owner));
        break;
      case 'strike':
        moves.push(...generateStrikeMoves(state, tile.position, pattern, tile.owner));
        break;
      case 'command':
        moves.push(...generateCommandMoves(state, tile.position, pattern, tile.owner));
        break;
    }
  }

  return moves;
}

function generatePlacementMoves(state: GameState): GameMove[] {
  const player = state.currentPlayer;
  const bag = state.bags[player];
  if (bag.length === 0) return [];

  // Find the Duke
  let dukePos: Coord | null = null;
  for (const tile of state.tiles.values()) {
    if (tile.owner === player && tile.defName === 'Duke') {
      dukePos = tile.position;
      break;
    }
  }
  if (!dukePos) return [];

  // Deduplicate bag tile names
  const uniqueNames = [...new Set(bag)];

  const moves: GameMove[] = [];
  for (const dir of ORTHOGONAL_DIRS) {
    const pos = addOffset(dukePos, dir);
    if (!inBounds(pos)) continue;
    if (state.board[pos.row][pos.col] !== null) continue;

    for (const tileName of uniqueNames) {
      moves.push({ type: 'place', tileName, position: pos });
    }
  }

  return moves;
}

// --- Check detection ---

function findDukePosition(state: GameState, player: Player): Coord | null {
  for (const tile of state.tiles.values()) {
    if (tile.owner === player && tile.defName === 'Duke') {
      return tile.position;
    }
  }
  return null;
}

/**
 * Returns true if any tile owned by `byPlayer` can attack `square`
 * via move, strike, or command in the given state.
 */
export function isSquareAttackedBy(
  state: GameState, square: Coord, byPlayer: Player,
): boolean {
  for (const tile of state.tiles.values()) {
    if (tile.owner !== byPlayer) continue;
    const moves = generateTileMoves(state, tile);
    for (const m of moves) {
      if (m.type === 'move' && m.to.row === square.row && m.to.col === square.col) return true;
      if (m.type === 'strike' && m.target.row === square.row && m.target.col === square.col) return true;
      if (m.type === 'command' && m.targetTo.row === square.row && m.targetTo.col === square.col) return true;
    }
  }
  return false;
}

/**
 * Returns true if a pseudo-legal move doesn't leave the mover's Duke in check.
 */
function isMoveLegal(state: GameState, move: GameMove): boolean {
  const newState = applyMoveRaw(state, move);

  // Captured enemy Duke — game is over, always legal
  if (newState.status !== 'active') return true;

  const mover = state.currentPlayer;
  const opponent = mover === 'P1' ? 'P2' : 'P1';
  const dukePos = findDukePosition(newState, mover);
  if (!dukePos) return false;

  return !isSquareAttackedBy(newState, dukePos, opponent);
}

// --- Pseudo-legal move generation (no check filtering) ---

function generatePseudoLegalMoves(state: GameState): GameMove[] {
  if (state.status !== 'active') return [];

  const moves: GameMove[] = [];
  const player = state.currentPlayer;

  for (const tile of state.tiles.values()) {
    if (tile.owner === player) {
      moves.push(...generateTileMoves(state, tile));
    }
  }

  moves.push(...generatePlacementMoves(state));

  return moves;
}

// --- Public API ---

/**
 * Generate all legal moves for the current player, filtering out
 * any move that would leave the player's own Duke in check.
 */
export function generateAllMoves(state: GameState): GameMove[] {
  const pseudo = generatePseudoLegalMoves(state);
  return pseudo.filter(m => isMoveLegal(state, m));
}

/**
 * Returns true if the current player has at least one legal move.
 * Short-circuits on the first legal move found (faster than generateAllMoves
 * for checkmate detection).
 */
export function hasAnyLegalMove(state: GameState): boolean {
  if (state.status !== 'active') return false;

  const player = state.currentPlayer;
  const opponent = player === 'P1' ? 'P2' : 'P1';

  // Check tile moves
  for (const tile of state.tiles.values()) {
    if (tile.owner !== player) continue;
    const tileMoves = generateTileMoves(state, tile);
    for (const move of tileMoves) {
      const newState = applyMoveRaw(state, move);
      if (newState.status !== 'active') return true;
      const dukePos = findDukePosition(newState, player);
      if (dukePos && !isSquareAttackedBy(newState, dukePos, opponent)) return true;
    }
  }

  // Check placement moves
  const placements = generatePlacementMoves(state);
  for (const move of placements) {
    const newState = applyMoveRaw(state, move);
    if (newState.status !== 'active') return true;
    const dukePos = findDukePosition(newState, player);
    if (dukePos && !isSquareAttackedBy(newState, dukePos, opponent)) return true;
  }

  return false;
}
