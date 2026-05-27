/**
 * CLI: generate self-play training data as a flat binary file.
 *
 * Uses worker_threads to run games in parallel across all CPU cores
 * and streams completed game data to disk as it arrives.
 *
 * Usage:
 *   npx tsx packages/ai/src/generate.ts [numGames] [depth] [outPath] [numWorkers]
 *
 * Load in Python / JAX:
 *   import numpy as np
 *   data = np.fromfile('selfplay.bin', dtype=np.float32).reshape(-1, 1220)
 *   states  = data[:, :1218]              # 33×6×6 board + 30 bag features
 *   actions = data[:, 1218].astype(int)   # action index (0–4427)
 *   values  = data[:, 1219]               # +1 win / −1 loss / 0 draw (current player POV)
 */

import {createWriteStream} from 'node:fs';
import {Worker} from 'node:worker_threads';
import {availableParallelism} from 'node:os';
import {SAMPLE_FLOATS} from './selfplay.js';
import {STATE_TENSOR_SIZE, BOARD_TENSOR_SIZE, NUM_CHANNELS, TOTAL_ACTIONS, TILE_NAMES} from './encoding.js';

const numGames     = parseInt(process.argv[2] ?? '100', 10);
const depth        = parseInt(process.argv[3] ?? '4', 10);
const outPath      = process.argv[4] ?? 'selfplay.bin';
const numWorkers   = Math.min(
  parseInt(process.argv[5] ?? String(availableParallelism()), 10),
  numGames,
);

const PROGRESS_EVERY = Math.max(1, Math.floor(numGames / 50));

interface WorkerMessage {
  numSamples: number;
  outcome: number;
  buffer: ArrayBuffer;
}

console.log(`Self-play: ${numGames} games, depth ${depth}, ${numWorkers} workers`);
const t0 = performance.now();

const stream = createWriteStream(outPath);
let totalSamples = 0;
let totalBytes = 0;
const outcomes: number[] = [];
let gamesAssigned = 0;
let gamesCompleted = 0;

// Workers need tsx's loader to handle TypeScript and `.js` import rewrites.
// We bootstrap each worker via an eval shim that registers tsx before
// importing the actual worker module.
const workerUrl = new URL('./selfplay-worker.ts', import.meta.url).href;
const bootstrap = `
import('tsx/esm/api').then(({ register }) => {
  register();
  return import(${JSON.stringify(workerUrl)});
}).catch(err => { console.error(err); process.exit(1); });
`;

await new Promise<void>((resolve, reject) => {
  const workers: Worker[] = [];

  const dispatch = (worker: Worker) => {
    if (gamesAssigned >= numGames) return;
    gamesAssigned++;
    worker.postMessage({type: 'play', depth, maxTurns: 200});
  };

  const handleMessage = (worker: Worker, msg: WorkerMessage) => {
    if (msg.buffer.byteLength > 0) {
      const buf = Buffer.from(msg.buffer);
      stream.write(buf);
      totalBytes += buf.byteLength;
    }
    totalSamples += msg.numSamples;
    outcomes.push(msg.outcome);
    gamesCompleted++;

    if (gamesCompleted % PROGRESS_EVERY === 0 || gamesCompleted === numGames) {
      const pct = (100 * gamesCompleted / numGames).toFixed(1);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      const rate = (gamesCompleted / Number(elapsed)).toFixed(2);
      console.log(`  ${gamesCompleted}/${numGames} games (${pct}%)  ${totalSamples} samples  ${elapsed}s  ${rate} games/s`);
    }

    if (gamesCompleted >= numGames) {
      for (const w of workers) w.terminate();
      resolve();
      return;
    }

    dispatch(worker);
  };

  for (let i = 0; i < numWorkers; i++) {
    const worker = new Worker(bootstrap, {eval: true});
    workers.push(worker);
    worker.on('message', (msg: WorkerMessage) => handleMessage(worker, msg));
    worker.on('error', reject);
    dispatch(worker);
  }
});

await new Promise<void>((resolve, reject) => {
  stream.end((err?: Error | null) => err ? reject(err) : resolve());
});

const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
const p1Wins = outcomes.filter(o => o === 1).length;
const p2Wins = outcomes.filter(o => o === -1).length;
const draws  = outcomes.filter(o => o === 0).length;

console.log(`\nDone: ${totalSamples} samples from ${numGames} games in ${elapsed}s`);
console.log(`  P1 wins: ${p1Wins}  P2 wins: ${p2Wins}  draws: ${draws}`);

const mb = (totalBytes / 1024 / 1024).toFixed(1);
console.log(`\nWrote ${outPath} (${mb} MB)`);
console.log(`\nTensor layout (per sample = ${SAMPLE_FLOATS} float32s):`);
console.log(`  [0:${BOARD_TENSOR_SIZE}]     board planes (${NUM_CHANNELS}ch × 6×6)`);
console.log(`  [${BOARD_TENSOR_SIZE}:${STATE_TENSOR_SIZE}]  bag features (${TILE_NAMES.length}×2)`);
console.log(`  [${STATE_TENSOR_SIZE}]        action index (0–${TOTAL_ACTIONS - 1})`);
console.log(`  [${STATE_TENSOR_SIZE + 1}]        value target`);
console.log(`\nPython:`);
console.log(`  data = np.fromfile('${outPath}', dtype=np.float32).reshape(-1, ${SAMPLE_FLOATS})`);
