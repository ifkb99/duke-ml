/**
 * Worker: plays one self-play game per request and posts results back.
 *
 * Protocol:
 *   Receive: {type: 'play', depth: number, maxTurns?: number}
 *   Send:    {numSamples: number, outcome: number, buffer: ArrayBuffer}
 *
 * The buffer is transferred (zero-copy) to the main thread.
 */

import {parentPort} from 'node:worker_threads';
import {selfPlay} from './selfplay.js';

if (!parentPort) {
  throw new Error('selfplay-worker.ts must be run as a worker thread');
}

interface PlayMessage {
  type: 'play';
  depth: number;
  maxTurns?: number;
}

parentPort.on('message', (msg: PlayMessage) => {
  if (msg.type !== 'play') return;

  const result = selfPlay({depth: msg.depth, maxTurns: msg.maxTurns});

  // Slice into a fresh ArrayBuffer so we can transfer ownership (zero-copy).
  const ab = result.data.buffer.slice(
    result.data.byteOffset,
    result.data.byteOffset + result.data.byteLength,
  );

  parentPort!.postMessage(
    {numSamples: result.numSamples, outcome: result.outcome, buffer: ab},
    [ab],
  );
});
