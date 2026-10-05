/**
 * State renderers — these are the independent variable in the
 * representation ablation. Each produces a text view of the same GameState.
 *
 *   renderAscii      — 2D grid, tests spatial reasoning from layout
 *   renderCoordList  — flat piece list, tests reasoning without layout
 *   describeTile     — movement patterns of a tile's CURRENT side, in
 *                      owner-relative terms (forward = toward the enemy)
 */
import type { GameState, MovePattern, Offset, Player, TileInstance } from '@the-duke/engine';
import { TILE_REGISTRY, BOARD_SIZE } from '@the-duke/engine';
import { coordToSquare } from './notation';

// ---------------------------------------------------------------------------
// Offset description (owner-relative)
// ---------------------------------------------------------------------------
// Raw offsets use negative dRow = toward row 0. After the engine's
// perspective adjustment, negative dRow is always "forward" (toward the
// enemy) from the owner's point of view — so we can describe raw offsets
// in relative terms and they are correct for both players.

function describeOffset(o: Offset): string {
  const parts: string[] = [];
  if (o.dRow < 0) parts.push(`forward ${-o.dRow}`);
  if (o.dRow > 0) parts.push(`back ${o.dRow}`);
  if (o.dCol < 0) parts.push(`left ${-o.dCol}`);
  if (o.dCol > 0) parts.push(`right ${o.dCol}`);
  return parts.join(', ') || 'in place';
}

const MOVE_TYPE_EXPLANATION: Record<string, string> = {
  step: 'steps to (blocked by intervening pieces on multi-square steps)',
  slide: 'slides any distance toward (blocked by pieces)',
  jump: 'jumps directly to (ignores intervening pieces)',
  jump_slide: 'slides any distance toward, jumping over pieces in the way',
  strike: 'captures without moving at',
  command: 'commands: a friendly piece on one of these squares may be moved to another of these squares',
};

function describePattern(p: MovePattern): string {
  const dirs = p.offsets.map(describeOffset).join(' | ');
  return `${MOVE_TYPE_EXPLANATION[p.type] ?? p.type}: ${dirs}`;
}

/** Describe the movement abilities of a tile's current side. */
export function describeTile(tile: TileInstance): string {
  const def = TILE_REGISTRY.get(tile.defName);
  if (!def) return `${tile.defName}: unknown`;
  const side = tile.side === 'A' ? def.sideA : def.sideB;
  const lines = side.patterns.map(p => `    - ${describePattern(p)}`);
  return `${tile.defName} (side ${tile.side}) at ${coordToSquare(tile.position)}:\n${lines.join('\n')}`;
}

// ---------------------------------------------------------------------------
// Board renderers
// ---------------------------------------------------------------------------

function shortLabel(tile: TileInstance): string {
  // Two-letter piece code + side, prefixed with owner. e.g. "1DUa" / "2FOb"
  const code = tile.defName.slice(0, 2).toUpperCase();
  return `${tile.owner === 'P1' ? '1' : '2'}${code}${tile.side.toLowerCase()}`;
}

/**
 * ASCII grid. Row 6 printed first so P2's side is at top, matching the
 * chess convention of rank numbers increasing away from P1.
 */
export function renderAscii(state: GameState): string {
  const legend: string[] = [];
  const rows: string[] = [];

  for (let r = BOARD_SIZE - 1; r >= 0; r--) {
    const cells: string[] = [];
    for (let c = 0; c < BOARD_SIZE; c++) {
      const id = state.board[r][c];
      if (!id) {
        cells.push(' .. ');
      } else {
        const tile = state.tiles.get(id)!;
        cells.push(shortLabel(tile));
        legend.push(`${shortLabel(tile)} = ${tile.owner} ${tile.defName} (side ${tile.side}) at ${coordToSquare(tile.position)}`);
      }
    }
    rows.push(`${r + 1} | ${cells.join(' ')}`);
  }
  rows.push(`    ${['a', 'b', 'c', 'd', 'e', 'f'].map(c => ` ${c}  `).join(' ')}`);

  return `${rows.join('\n')}\n\nPieces:\n${legend.map(l => `  ${l}`).join('\n')}`;
}

/** Flat coordinate list — no 2D layout. */
export function renderCoordList(state: GameState): string {
  const mine: string[] = [];
  const theirs: string[] = [];
  const me = state.currentPlayer;

  for (const tile of state.tiles.values()) {
    const line = `${tile.defName} (side ${tile.side}) at ${coordToSquare(tile.position)}`;
    (tile.owner === me ? mine : theirs).push(line);
  }
  return [
    `Your pieces (${me}):`,
    ...mine.map(l => `  ${l}`),
    `Opponent pieces:`,
    ...theirs.map(l => `  ${l}`),
  ].join('\n');
}

/** Full tile-ability appendix for the current player's pieces. */
export function renderAbilities(state: GameState, player: Player): string {
  const out: string[] = [];
  for (const tile of state.tiles.values()) {
    if (tile.owner === player) out.push(describeTile(tile));
  }
  return out.join('\n');
}

/** Bag contents for the current player (perfect-information variant). */
export function renderBag(state: GameState): string {
  const bag = state.bags[state.currentPlayer];
  if (bag.length === 0) return 'Your bag is empty.';
  const counts = new Map<string, number>();
  for (const name of bag) counts.set(name, (counts.get(name) ?? 0) + 1);
  return `Your bag: ${[...counts.entries()].map(([n, k]) => (k > 1 ? `${n} ×${k}` : n)).join(', ')}`;
}
