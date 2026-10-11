// Shared UI vocabulary: names, abbreviations, move-type colours and the board
// orientation. Colours live in index.css as variables; these point at them.
import type {Coord, GameMove, GameState, MovePattern, MoveType, Player, TileSide} from '@the-duke/engine';
import {BOARD_SIZE} from '@the-duke/engine';

export type GameMode = 'hotseat' | 'vs-random' | 'vs-minimax' | 'vs-zero';

export const PLAYER_NAME: Record<Player, string> = {P1: 'Light', P2: 'Dark'};
export const other = (p: Player): Player => (p === 'P1' ? 'P2' : 'P1');

export const MODE_NAME: Record<GameMode, string> = {
  'vs-zero': 'Neural Net',
  'vs-minimax': 'Minimax',
  'vs-random': 'Random',
  hotseat: 'Friend',
};

export const TILE_SHORT: Record<string, string> = {
  Duke: 'Du', Footman: 'Ft', Pikeman: 'Pk', Knight: 'Kn', Longbowman: 'Lb', Bowman: 'Bw',
  Champion: 'Ch', Wizard: 'Wz', Marshall: 'Ma', General: 'Ge', Ranger: 'Ra', Priest: 'Pr',
  Dragoon: 'Dr', Assassin: 'As', Seer: 'Se',
};
export const shortName = (name: string) => TILE_SHORT[name] ?? name.slice(0, 2);

export const MOVE_COLOR: Record<MoveType, string> = {
  step: 'var(--mv-step)',
  slide: 'var(--mv-slide)',
  jump: 'var(--mv-jump)',
  jump_slide: 'var(--mv-jump-slide)',
  strike: 'var(--mv-strike)',
  command: 'var(--mv-command)',
};

export const MOVE_INFO: {type: MoveType; label: string; help: string}[] = [
  {type: 'step', label: 'Move', help: 'Move to this square if the path is clear. Captures an enemy there.'},
  {type: 'slide', label: 'Slide', help: 'Move any distance in this direction until blocked.'},
  {type: 'jump', label: 'Jump', help: 'Leap straight to this square, over anything in between.'},
  {type: 'jump_slide', label: 'Jump-slide', help: 'Leap to this square, then keep sliding in the same direction.'},
  {type: 'strike', label: 'Strike', help: 'Capture an enemy on this square without moving.'},
  {type: 'command', label: 'Command', help: 'Move one of your tiles from one command square to another.'},
];

/** Which player's side sits at the bottom of the screen. */
export type Orientation = Player;

/**
 * Board row/col -> screen row/col. The engine mirrors only the rows for Light
 * (it starts on row 0, at the top), so the view mirrors rows too: each
 * player's own tiles then look exactly like the printed tile.
 */
export function toScreen({row, col}: Coord, bottom: Orientation): Coord {
  return bottom === 'P1' ? {row: BOARD_SIZE - 1 - row, col} : {row, col};
}

/** Row direction of a tile's printed pattern on screen: 1 = as printed, -1 = upside down. */
export function patternRowSign(owner: Player, bottom: Orientation): number {
  return (owner === 'P1' ? -1 : 1) * (bottom === 'P1' ? -1 : 1);
}

export interface PatternCell {type: MoveType; distance: number}
export type PatternGrid = PatternCell[][][];   // [row][col] -> move types on that square

/**
 * A tile side's patterns laid out on a size x size grid around (center, center).
 * Slides and jump-slides are drawn to the grid edge, fading with distance.
 */
export function patternGrid(side: TileSide, rowSign: number, size: number,
                            center: Coord = {row: (size - 1) / 2, col: (size - 1) / 2}): PatternGrid {
  const grid: PatternGrid = Array.from({length: size}, () => Array.from({length: size}, () => []));
  const add = (r: number, c: number, type: MoveType, distance: number) => {
    const cell = grid[r]?.[c];
    if (cell && !cell.some(e => e.type === type)) cell.push({type, distance});
  };
  const reach = (p: MovePattern) => (p.type === 'slide' || p.type === 'jump_slide' ? size : 1);
  for (const p of side.patterns) {
    for (const {dRow, dCol} of p.offsets) {
      for (let i = 1; i <= reach(p); i++) {
        add(center.row + dRow * rowSign * i, center.col + dCol * i, p.type, i);
      }
    }
  }
  return grid;
}

/** Squares a move touches, for the last-move highlight. */
export function moveSquares(m: GameMove): {from?: Coord; to?: Coord; struck?: Coord; commander?: Coord} {
  switch (m.type) {
    case 'move': return {from: m.from, to: m.to};
    case 'strike': return {from: m.from, struck: m.target};
    case 'command': return {commander: m.commander, from: m.target, to: m.targetTo};
    case 'place':
    case 'draw': return {to: m.position};
  }
}

export const sameCoord = (a: Coord | null | undefined, b: Coord | null | undefined) =>
  !!a && !!b && a.row === b.row && a.col === b.col;

/** How a finished game ended: the loser's Duke was taken, or the loser had no legal move. */
export function endReason(state: GameState): 'captured' | 'checkmate' {
  const loser: Player = state.status === 'P1_wins' ? 'P2' : 'P1';
  const dukeAlive = [...state.tiles.values()].some(t => t.defName === 'Duke' && t.owner === loser);
  return dukeAlive ? 'checkmate' : 'captured';
}
