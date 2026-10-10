/**
 * Export the engine's tile definitions to JSON for the JAX environment
 * (training/zero), so the rules have exactly one source of truth.
 *
 * Usage:
 *   npx tsx packages/ai/src/zero/export-tiles.ts [outPath]
 */

import {writeFileSync} from 'node:fs';
import {ALL_TILES, BAG_TILE_NAMES, BOARD_SIZE} from '@the-duke/engine';

const outPath = process.argv[2] ?? 'training/zero/tiles.json';

const data = {
  boardSize: BOARD_SIZE,
  // Offsets use the engine convention: negative dRow = forward for the
  // player whose home row is the last row (P2). See types.ts.
  tiles: ALL_TILES.map(t => ({
    name: t.name,
    sides: [t.sideA, t.sideB].map(side => side.patterns.map(p => ({
      type: p.type,
      offsets: p.offsets.map(o => [o.dRow, o.dCol]),
    }))),
  })),
  // Full per-player bag before setup. Setup moves two Footmen to the board.
  bag: [...BAG_TILE_NAMES],
};

writeFileSync(outPath, JSON.stringify(data, null, 2) + '\n');
console.log(`Wrote ${outPath} (${data.tiles.length} tiles)`);
