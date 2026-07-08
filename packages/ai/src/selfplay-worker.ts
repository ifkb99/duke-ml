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

  // Copy the samples into a fresh, standalone ArrayBuffer so ownership can be
  // transferred (zero-copy) to the main thread. Typing it as ArrayBuffer (not
  // the ArrayBufferLike that Float32Array.buffer exposes) keeps it Transferable.
  const ab = new ArrayBuffer(result.data.byteLength);
  new Float32Array(ab).set(result.data);

  parentPort!.postMessage(
    {numSamples: result.numSamples, outcome: result.outcome, buffer: ab},
    [ab],
  );
});
