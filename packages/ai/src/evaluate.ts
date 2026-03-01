import type { GameState } from '@the-duke/engine';

const TILE_VALUES: Record<string, number> = {
  Duke: 0,
  Footman: 2,
  Pikeman: 2,
  Knight: 3,
  Longbowman: 4,
  Bowman: 3,
  Champion: 3,
  Wizard: 3,
  Marshall: 4,
  General: 4,
  Ranger: 4,
  Priest: 3,
  Dragoon: 4,
  Assassin: 3,
  Seer: 3,
};

/**
 * Evaluate the board from P1's perspective.
 * Positive = P1 is winning, negative = P2 is winning.
 */
export function evaluate(state: GameState): number {
  if (state.status === 'P1_wins') return 100_000;
  if (state.status === 'P2_wins') return -100_000;

  let score = 0;

  // Material on board
  for (const tile of state.tiles.values()) {
    const value = TILE_VALUES[tile.defName] ?? 1;
    const multiplier = tile.owner === 'P1' ? 1 : -1;
    score += value * multiplier;
  }

  // Bag material (tiles in bag are worth less than tiles on board)
  for (const name of state.bags.P1) {
    score += (TILE_VALUES[name] ?? 1) * 0.5;
  }
  for (const name of state.bags.P2) {
    score -= (TILE_VALUES[name] ?? 1) * 0.5;
  }

  // Center control bonus — tiles near center are slightly better
  for (const tile of state.tiles.values()) {
    const centerDist = Math.abs(tile.position.row - 2.5) + Math.abs(tile.position.col - 2.5);
    const bonus = (5 - centerDist) * 0.1;
    score += tile.owner === 'P1' ? bonus : -bonus;
  }

  return score;
}
