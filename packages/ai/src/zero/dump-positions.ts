/**
 * Dump random-play positions for differential testing of the JAX engine.
 *
 * Each output line is one position:
 *   {state, legal, drawTargets, move, next}
 * where `legal` is generateAllMoves(state), `drawTargets` the legal placement
 * squares, `move` the move actually played, and `next` = applyMove(state, move).
 * Draws are resolved to a concrete `place` move so `next` is reproducible.
 *
 * Usage (run several in parallel for more coverage):
 *   npx tsx packages/ai/src/zero/dump-positions.ts <numGames> <outPath> [seed]
 */

import {createWriteStream} from 'node:fs';
import {
  applyMove, applySetupPlacement, createInitialState, generateAllMoves,
  getDrawTargets, getSetupTargets, serialize, type GameMove,
} from '@the-duke/engine';

const numGames = parseInt(process.argv[2] ?? '50', 10);
const outPath = process.argv[3] ?? 'positions.jsonl';
const maxTurns = 150;

const pick = <T,>(a: T[]): T => a[Math.floor(Math.random() * a.length)];
const out = createWriteStream(outPath);

for (let g = 0; g < numGames; g++) {
  let s = createInitialState();
  while (s.status === 'setup') s = applySetupPlacement(s, pick(getSetupTargets(s)));

  for (let turn = 0; turn < maxTurns && s.status === 'active'; turn++) {
    const legal = generateAllMoves(s);
    if (legal.length === 0) break;

    // Favor drawing early so boards fill up and exercise more tile types.
    const draws = legal.filter(m => m.type === 'draw');
    let move: GameMove = draws.length && Math.random() < 0.3 ? pick(draws) : pick(legal);
    if (move.type === 'draw') {
      const bag = s.bags[s.currentPlayer];
      move = {type: 'place', tileName: pick(bag), position: move.position};
    }

    const next = applyMove(s, move);
    out.write(JSON.stringify({
      state: serialize(s),
      legal,
      drawTargets: getDrawTargets(s),
      move,
      next: serialize(next),
    }) + '\n');
    s = next;
  }
}

out.end();
