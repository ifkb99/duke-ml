import { ALL_TILES, type GameState, type MoveType, type TileDefinition } from '@the-duke/engine';

const MOVE_TYPE_VALUES: Record<MoveType, number> = {
  step: 1,
  slide: 2.5,
  jump: 1.5,
  jump_slide: 3.5,
  strike: 2,
  command: 1,
};

/**
 * Tile is worth the score of amount of moves it can make on side A and side B
 * starting side weighted slightly higher
 */
const scoreTile = (tile: TileDefinition) => {
  return tile.sideA.patterns.reduce((acc, pattern) => acc + MOVE_TYPE_VALUES[pattern.type] * pattern.offsets.length, 0) * 0.55 +
    tile.sideB.patterns.reduce((acc, pattern) => acc + MOVE_TYPE_VALUES[pattern.type] * pattern.offsets.length, 0) * 0.45;
};

const TILE_VALUES: ReadonlyMap<string, number> = new Map(ALL_TILES.map(tile => [tile.name, scoreTile(tile)]));

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
    const value = TILE_VALUES.get(tile.defName) ?? 1;
    const multiplier = tile.owner === 'P1' ? 1 : -1;
    score += value * multiplier;
  }

  // Bag material (tiles in bag are worth less than tiles on board)
  // TODO: bag score should be sum(bag_piece_scores) / n_pieces_in_bag so likelihood of drawing a good tile is factored in
  //    BUT we also need to consider the fact that the pieces in bag are not in play.
  // Tile on the board is worth two in the bush [bag]
  // does bag len consider pieces with multiple copies?
  // in calc if bag is drawn from does it calc w/ a piece it draws or a superposition of all the pieces in the bag? is the num of pieces in bag changed?
  const bagScoreP1 = state.bags.P1.reduce((acc, name) => acc + (TILE_VALUES.get(name) ?? 1), 0) / state.bags.P1.length;
  const bagScoreP2 = state.bags.P2.reduce((acc, name) => acc + (TILE_VALUES.get(name) ?? 1), 0) / state.bags.P2.length;
  score += bagScoreP1 * 2 - bagScoreP2 * 2;

  // Center control bonus — tiles near center are slightly better
  for (const tile of state.tiles.values()) {
    const centerDist = Math.abs(tile.position.row - 2.5) + Math.abs(tile.position.col - 2.5);
    const bonus = (5 - centerDist) * 0.1;
    score += tile.owner === 'P1' ? bonus : -bonus;
  }

  return score;
}
