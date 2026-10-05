import type {
  Coord, GameMove, GameState, MovePattern, Offset, Player, TileInstance,
} from './types.js';
import {TILE_REGISTRY} from './tiles.js';
import {BOARD_SIZE} from './state.js';
import {makeMove, unmakeMove} from './game.js';

function inBounds(coord: Coord): boolean {
  return coord.row >= 0 && coord.row < BOARD_SIZE
    && coord.col >= 0 && coord.col < BOARD_SIZE;
}

// Tile offsets are defined with negative dRow = upward (toward row 0 = toward P1's side).
// P2 sits at row 5 facing upward toward P1 → uses offsets directly (F = {dRow:-1} = correct forward).
// P1 sits at row 0 facing downward toward P2 → dRow must be negated (F = {dRow:+1} = correct forward).
function adjustOffset(offset: Offset, player: Player): Offset {
  return player === 'P2' ? offset : {dRow: -offset.dRow, dCol: offset.dCol};
}

function addOffset(coord: Coord, offset: Offset): Coord {
  return {row: coord.row + offset.dRow, col: coord.col + offset.dCol};
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
  while (b) {[a, b] = [b, a % b];}
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
    const intermediate: Coord = {row: from.row + unitDr * i, col: from.col + unitDc * i};
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
      moves.push({type: 'move', from, to});
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
      moves.push({type: 'move', from, to: current});
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
      moves.push({type: 'move', from, to});
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
      moves.push({type: 'strike', from, target});
    }
  }
  return moves;
}

const ORTHOGONAL_DIRS: Offset[] = [
  {dRow: -1, dCol: 0},
  {dRow: 1, dCol: 0},
  {dRow: 0, dCol: -1},
  {dRow: 0, dCol: 1},
];

/**
 * Command (official rule): move a friendly tile standing on one of the
 * commander's command squares to any *other* command square. The destination
 * must be empty or hold an enemy (captured). The commander flips; the
 * commanded tile does not (see makeMove).
 */
function generateCommandMoves(
  state: GameState, commander: Coord, pattern: MovePattern, player: Player,
): GameMove[] {
  const squares: Coord[] = [];
  for (const offset of pattern.offsets) {
    const sq = addOffset(commander, adjustOffset(offset, player));
    if (inBounds(sq)) squares.push(sq);
  }

  const moves: GameMove[] = [];
  for (const target of squares) {
    const occupant = tileAt(state, target);
    if (!occupant || !isFriendly(occupant, player)) continue;
    for (const dest of squares) {
      if (dest.row === target.row && dest.col === target.col) continue;
      const destOccupant = tileAt(state, dest);
      if (!destOccupant || isEnemy(destOccupant, player)) {
        moves.push({type: 'command', commander, target, targetTo: dest});
      }
    }
  }
  return moves;
}

/**
 * Jump-slide: same as slide, but allowed to jump over any piece in the way.
 * The piece can land on the 1st, 2nd, 3rd, ... square until the edge of the board
 */
function generateJumpSlideMoves(
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
      moves.push({type: 'move', from, to: current});
    }
  }
  return moves;
}

export function generateTileMoves(state: GameState, tile: TileInstance): GameMove[] {
  const def = TILE_REGISTRY.get(tile.defName);
  if (!def) return [];

  const side = tile.side === 'A' ? def.sideA : def.sideB;
  const moves: GameMove[] = [];

  // Copy position so moves don't share a mutable reference with the tile.
  // This prevents corruption when makeMove/unmakeMove mutates tile.position.
  const pos: Coord = {row: tile.position.row, col: tile.position.col};

  for (const pattern of side.patterns) {
    switch (pattern.type) {
      case 'step':
        moves.push(...generateStepMoves(state, pos, pattern, tile.owner));
        break;
      case 'slide':
        moves.push(...generateSlideMoves(state, pos, pattern, tile.owner));
        break;
      case 'jump':
        moves.push(...generateJumpMoves(state, pos, pattern, tile.owner));
        break;
      case 'jump_slide':
        moves.push(...generateJumpSlideMoves(state, pos, pattern, tile.owner));
        break;
      case 'strike':
        moves.push(...generateStrikeMoves(state, pos, pattern, tile.owner));
        break;
      case 'command':
        moves.push(...generateCommandMoves(state, pos, pattern, tile.owner));
        break;
    }
  }

  return moves;
}

/**
 * Pseudo-legal draw moves: one per empty square orthogonally adjacent to the
 * current player's Duke (no check filtering — that happens in isMoveLegal).
 *
 * A "draw" hides which tile is pulled from the bag; the tile is only revealed
 * when the move is resolved/applied. This is the honest model of the bag's
 * hidden information — callers that need per-tile detail (search chance nodes,
 * UI reveal) use `enumerateDrawOutcomes` instead.
 */
function generateDrawMoves(state: GameState): GameMove[] {
  const player = state.currentPlayer;
  if (state.bags[player].length === 0) return [];

  const dukePos = findDukePosition(state, player);
  if (!dukePos) return [];

  const moves: GameMove[] = [];
  for (const dir of ORTHOGONAL_DIRS) {
    const pos = addOffset(dukePos, dir);
    if (!inBounds(pos)) continue;
    if (state.board[pos.row][pos.col] !== null) continue;
    moves.push({type: 'draw', position: pos});
  }
  return moves;
}

/**
 * Legality of a draw is tile-independent: placing *any* tile on `pos` adds the
 * same blocker, and a newly placed friendly tile can never expose the mover's
 * own Duke. So we test with a throwaway probe occupant rather than a real draw
 * (which would consume randomness).
 */
function isDrawSquareLegal(state: GameState, pos: Coord, player: Player): boolean {
  const opponent = player === 'P1' ? 'P2' : 'P1';
  const probeId = '__draw_probe__';
  state.board[pos.row][pos.col] = probeId;
  state.tiles.set(probeId, {
    defName: 'Footman', owner: player, side: 'A',
    position: {row: pos.row, col: pos.col}, id: probeId,
  });

  const dukePos = findDukePosition(state, player);
  const legal = dukePos ? !isSquareAttackedBy(state, dukePos, opponent) : false;

  state.tiles.delete(probeId);
  state.board[pos.row][pos.col] = null;
  return legal;
}

/**
 * Legal squares where the current player may place a drawn tile: empty and
 * orthogonally adjacent to their Duke, and not leaving their Duke in check.
 * Returns [] if the bag is empty. Tile-independent (see isDrawSquareLegal).
 */
export function getDrawTargets(state: GameState): Coord[] {
  const player = state.currentPlayer;
  if (state.bags[player].length === 0) return [];

  const dukePos = findDukePosition(state, player);
  if (!dukePos) return [];

  const targets: Coord[] = [];
  for (const dir of ORTHOGONAL_DIRS) {
    const pos = addOffset(dukePos, dir);
    if (!inBounds(pos)) continue;
    if (state.board[pos.row][pos.col] !== null) continue;
    if (isDrawSquareLegal(state, pos, player)) targets.push(pos);
  }
  return targets;
}

/** A drawable tile, its probability, and the resolved placements it enables. */
export interface DrawOutcome {
  tileName: string;
  probability: number;
  placements: { type: 'place'; tileName: string; position: Coord }[];
}

/**
 * Expand the current player's "draw" decision into its chance outcomes: for
 * each distinct tile in the bag, the probability of drawing it and the
 * resolved `place` moves it permits (one per legal target square).
 *
 * This is the single source of truth for expectimax over draws — search and
 * the neural evaluator both consume it so no caller has to reconstruct the
 * bag's probability structure (and accidentally let a player "peek").
 */
export function enumerateDrawOutcomes(state: GameState): DrawOutcome[] {
  const player = state.currentPlayer;
  const bag = state.bags[player];
  if (bag.length === 0) return [];

  const squares = getDrawTargets(state);
  if (squares.length === 0) return [];

  const counts = new Map<string, number>();
  for (const name of bag) counts.set(name, (counts.get(name) ?? 0) + 1);

  const outcomes: DrawOutcome[] = [];
  for (const [tileName, count] of counts) {
    outcomes.push({
      tileName,
      probability: count / bag.length,
      placements: squares.map(pos => ({
        type: 'place' as const, tileName, position: {row: pos.row, col: pos.col},
      })),
    });
  }
  return outcomes;
}

// --- Check detection ---

export function findDukePosition(state: GameState, player: Player): Coord | null {
  for (const tile of state.tiles.values()) {
    if (tile.owner === player && tile.defName === 'Duke') {
      return tile.position;
    }
  }
  return null;
}

/**
 * Check if a specific tile can reach (attack/move to) a target square.
 * Uses direct coordinate math — no object allocation.
 */
function canTileReachSquare(
  state: GameState, tile: TileInstance, target: Coord,
): boolean {
  const def = TILE_REGISTRY.get(tile.defName);
  if (!def) return false;

  const side = tile.side === 'A' ? def.sideA : def.sideB;
  const from = tile.position;
  const dr = target.row - from.row;
  const dc = target.col - from.col;
  if (dr === 0 && dc === 0) return false;

  const player = tile.owner;

  for (const pattern of side.patterns) {
    switch (pattern.type) {
      case 'step': {
        for (const offset of pattern.offsets) {
          const adjDr = player === 'P2' ? offset.dRow : -offset.dRow;
          const adjDc = offset.dCol;
          if (dr === adjDr && dc === adjDc) {
            // Check path clearance for multi-square steps
            const g = gcd(Math.abs(adjDr), Math.abs(adjDc));
            if (g <= 1) {
              // Target must be empty or enemy
              const occupant = state.board[target.row][target.col];
              if (!occupant || state.tiles.get(occupant)!.owner !== player) return true;
            } else {
              const unitDr = adjDr / g;
              const unitDc = adjDc / g;
              let clear = true;
              for (let i = 1; i < g; i++) {
                const ir = from.row + unitDr * i;
                const ic = from.col + unitDc * i;
                if (ir < 0 || ir >= BOARD_SIZE || ic < 0 || ic >= BOARD_SIZE || state.board[ir][ic] !== null) {
                  clear = false;
                  break;
                }
              }
              if (clear) {
                const occupant = state.board[target.row][target.col];
                if (!occupant || state.tiles.get(occupant)!.owner !== player) return true;
              }
            }
          }
        }
        break;
      }
      case 'jump': {
        for (const offset of pattern.offsets) {
          const adjDr = player === 'P2' ? offset.dRow : -offset.dRow;
          if (dr === adjDr && dc === offset.dCol) {
            const occupant = state.board[target.row][target.col];
            if (!occupant || state.tiles.get(occupant)!.owner !== player) return true;
          }
        }
        break;
      }
      case 'slide': {
        for (const dir of pattern.offsets) {
          const adjDr = player === 'P2' ? dir.dRow : -dir.dRow;
          const adjDc = dir.dCol;
          // Check collinearity: dr/adjDr == dc/adjDc and same sign
          if (adjDr === 0 && adjDc === 0) continue;
          if (adjDr === 0) {
            if (dr !== 0) continue;
            if ((dc > 0) !== (adjDc > 0)) continue;
          } else if (adjDc === 0) {
            if (dc !== 0) continue;
            if ((dr > 0) !== (adjDr > 0)) continue;
          } else {
            if (dr * adjDc !== dc * adjDr) continue; // not collinear
            if ((dr > 0) !== (adjDr > 0)) continue;  // opposite direction
          }
          // Walk path from 'from' toward 'target' checking for blockers
          let cr = from.row + adjDr;
          let cc = from.col + adjDc;
          let reached = false;
          while (cr >= 0 && cr < BOARD_SIZE && cc >= 0 && cc < BOARD_SIZE) {
            if (cr === target.row && cc === target.col) {
              const occupant = state.board[target.row][target.col];
              if (!occupant || state.tiles.get(occupant)!.owner !== player) reached = true;
              break;
            }
            const occ = state.board[cr][cc];
            if (occ !== null) break; // blocked
            cr += adjDr;
            cc += adjDc;
          }
          if (reached) return true;
        }
        break;
      }
      case 'jump_slide': {
        for (const dir of pattern.offsets) {
          const adjDr = player === 'P2' ? dir.dRow : -dir.dRow;
          const adjDc = dir.dCol;
          if (adjDr === 0 && adjDc === 0) continue;
          // Check collinearity
          if (adjDr === 0) {
            if (dr !== 0) continue;
            if ((dc > 0) !== (adjDc > 0)) continue;
          } else if (adjDc === 0) {
            if (dc !== 0) continue;
            if ((dr > 0) !== (adjDr > 0)) continue;
          } else {
            if (dr * adjDc !== dc * adjDr) continue;
            if ((dr > 0) !== (adjDr > 0)) continue;
          }
          // Jump_slide skips over pieces — just check target isn't friendly
          const occupant = state.board[target.row][target.col];
          if (!occupant || state.tiles.get(occupant)!.owner !== player) return true;
        }
        break;
      }
      case 'strike': {
        for (const offset of pattern.offsets) {
          const adjDr = player === 'P2' ? offset.dRow : -offset.dRow;
          if (dr === adjDr && dc === offset.dCol) {
            // Strike only hits enemies
            const occupant = state.board[target.row][target.col];
            if (occupant && state.tiles.get(occupant)!.owner !== player) return true;
          }
        }
        break;
      }
      case 'command': {
        // Target must itself be a command square (empty or enemy), and some
        // *other* command square must hold a friendly tile to move there.
        let targetIsCommandSquare = false;
        let hasFriendlySource = false;
        for (const offset of pattern.offsets) {
          const adjDr = player === 'P2' ? offset.dRow : -offset.dRow;
          const cmdR = from.row + adjDr;
          const cmdC = from.col + offset.dCol;
          if (cmdR < 0 || cmdR >= BOARD_SIZE || cmdC < 0 || cmdC >= BOARD_SIZE) continue;
          if (cmdR === target.row && cmdC === target.col) {
            targetIsCommandSquare = true;
            continue;
          }
          const cmdId = state.board[cmdR][cmdC];
          if (cmdId && state.tiles.get(cmdId)!.owner === player) hasFriendlySource = true;
        }
        if (targetIsCommandSquare && hasFriendlySource) {
          const occupant = state.board[target.row][target.col];
          if (!occupant || state.tiles.get(occupant)!.owner !== player) return true;
        }
        break;
      }
    }
  }
  return false;
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
    if (canTileReachSquare(state, tile, square)) return true;
  }
  return false;
}

/**
 * Returns true if the given player's Duke is currently in check.
 */
export function isPlayerInCheck(state: GameState, player: Player): boolean {
  const dukePos = findDukePosition(state, player);
  if (!dukePos) return false;
  const opponent = player === 'P1' ? 'P2' : 'P1';
  return isSquareAttackedBy(state, dukePos, opponent);
}

/**
 * Returns true if a pseudo-legal move doesn't leave the mover's Duke in check.
 */
function isMoveLegal(state: GameState, move: GameMove): boolean {
  const mover = state.currentPlayer;
  const opponent = mover === 'P1' ? 'P2' : 'P1';

  // A draw's legality is tile-independent, so avoid actually drawing (which
  // would consume randomness); probe with a throwaway blocker instead.
  if (move.type === 'draw') {
    return isDrawSquareLegal(state, move.position, mover);
  }

  const undo = makeMove(state, move);

  // Captured enemy Duke — game is over, always legal
  if (state.status !== 'active') {
    unmakeMove(state, undo);
    return true;
  }

  const dukePos = findDukePosition(state, mover);
  const legal = dukePos ? !isSquareAttackedBy(state, dukePos, opponent) : false;
  unmakeMove(state, undo);
  return legal;
}

// --- Pseudo-legal move generation (no check filtering) ---

export function generatePseudoLegalMoves(state: GameState): GameMove[] {
  if (state.status !== 'active') return [];

  const moves: GameMove[] = [];
  const player = state.currentPlayer;

  for (const tile of state.tiles.values()) {
    if (tile.owner === player) {
      moves.push(...generateTileMoves(state, tile));
    }
  }

  moves.push(...generateDrawMoves(state));

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
  const pseudo = generatePseudoLegalMoves(state);
  return pseudo.some(m => isMoveLegal(state, m));
}
