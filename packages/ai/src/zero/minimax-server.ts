/**
 * Minimax as a line-oriented subprocess, so Python can use it as an opponent
 * (evaluation) or teacher (imitation warm-start).
 *
 * Protocol: one JSON object per line on stdin, one reply per line on stdout.
 *   in:  {"id": any, "state": SerializedGameState, "depth": number}
 *   out: {"id": any, "move": GameMove | null, "score": number}
 * Draws come back as a resolved {type: 'place', tileName, position}: minimax
 * draws uniformly from the bag itself and then picks the best square.
 *
 *   in:  {"id": any, "reset": true}   → clears the transposition table
 *
 * Prints "READY" once on startup.
 *
 * Usage:
 *   npx tsx packages/ai/src/zero/minimax-server.ts
 */

import {createInterface} from 'node:readline';
import {deserialize, type SerializedGameState} from '@the-duke/engine';
import {findBestMove, ttClear} from '../minimax.js';

interface Request {
  id: unknown;
  state?: SerializedGameState;
  depth?: number;
  reset?: boolean;
}

const rl = createInterface({input: process.stdin});
process.stdout.write('READY\n');

rl.on('line', line => {
  const req = JSON.parse(line) as Request;
  if (req.reset) {
    ttClear();
    process.stdout.write(JSON.stringify({id: req.id, ok: true}) + '\n');
    return;
  }
  const state = deserialize(req.state!);
  const result = findBestMove(state, req.depth ?? 4);
  process.stdout.write(JSON.stringify({id: req.id, move: result.move, score: result.score}) + '\n');
});
