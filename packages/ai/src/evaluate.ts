import { ALL_TILES, type GameState, type MoveType, type TileDefinition } from '@the-duke/engine';
import { TILE_NAME_INDEX } from './utils/zobrist.js';

const MOVE_TYPE_VALUES: Record<MoveType, number> = {
  step: 1,
  slide: 2.5,
  jump: 1.5,
  jump_slide: 3.5,
  strike: 2,
  command: 1,
};

const scoreTile = (tile: TileDefinition) => {
  return tile.sideA.patterns.reduce((acc, p) => acc + MOVE_TYPE_VALUES[p.type] * p.offsets.length, 0) * 0.55 +
    tile.sideB.patterns.reduce((acc, p) => acc + MOVE_TYPE_VALUES[p.type] * p.offsets.length, 0) * 0.45;
};

// Flat array indexed by tile name index for O(1) lookups
const TILE_VALUES_BY_INDEX = new Float64Array(ALL_TILES.length);
for (const tile of ALL_TILES) {
  const idx = TILE_NAME_INDEX.get(tile.name)!;
  TILE_VALUES_BY_INDEX[idx] = scoreTile(tile);
}

function tileValue(defName: string): number {
  const idx = TILE_NAME_INDEX.get(defName);
  return idx !== undefined ? TILE_VALUES_BY_INDEX[idx] : 1;
}

/**
 * Evaluate the board from P1's perspective.
 * Positive = P1 is winning, negative = P2 is winning.
 */
export function evaluate(state: GameState): number {
  if (state.status === 'P1_wins') return 100_000;
  if (state.status === 'P2_wins') return -100_000;

  let score = 0;

  // Material + center control in a single pass over tiles
  for (const tile of state.tiles.values()) {
    const value = tileValue(tile.defName);
    const centerDist = Math.abs(tile.position.row - 2.5) + Math.abs(tile.position.col - 2.5);
    const bonus = value + (5 - centerDist) * 0.1;
    score += tile.owner === 'P1' ? bonus : -bonus;
  }

  // Bag material — averaged value of tiles in each bag
  const bagP1 = state.bags.P1;
  const bagP2 = state.bags.P2;
  if (bagP1.length > 0) {
    let sum = 0;
    for (let i = 0; i < bagP1.length; i++) sum += tileValue(bagP1[i]);
    score += (sum / bagP1.length) * 2;
  }
  if (bagP2.length > 0) {
    let sum = 0;
    for (let i = 0; i < bagP2.length; i++) sum += tileValue(bagP2[i]);
    score -= (sum / bagP2.length) * 2;
  }

  return score;
}

/** Fast static score for a single placement (used for placement pruning). */
export function evaluatePlacement(state: GameState, defName: string): number {
  return tileValue(defName);
}
