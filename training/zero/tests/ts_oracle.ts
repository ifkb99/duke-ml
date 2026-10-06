/**
 * The TypeScript engine as a test oracle for the Python tests.
 *
 * Reads one JSON request per line on stdin and writes one reply per line:
 *   in:  {"state": SerializedGameState, "move"?: GameMove}
 *   out: {"legal": GameMove[], "drawTargets": Coord[], "roundTrip": SerializedGameState,
 *         "next"?: SerializedGameState}
 * `roundTrip` is serialize(deserialize(state)); `next` is applyMove(state, move).
 *
 * Usage: npx tsx training/zero/tests/ts_oracle.ts < requests.jsonl
 */

import {createInterface} from 'node:readline';
import {
  applyMove, deserialize, generateAllMoves, getDrawTargets, serialize,
  type GameMove, type SerializedGameState,
} from '@the-duke/engine';

const rl = createInterface({input: process.stdin});
rl.on('line', line => {
  const req = JSON.parse(line) as {state: SerializedGameState; move?: GameMove};
  const state = deserialize(req.state);
  const reply: Record<string, unknown> = {
    legal: generateAllMoves(state),
    drawTargets: getDrawTargets(state),
    roundTrip: serialize(state),
  };
  if (req.move) reply.next = serialize(applyMove(state, req.move));
  process.stdout.write(JSON.stringify(reply) + '\n');
});
