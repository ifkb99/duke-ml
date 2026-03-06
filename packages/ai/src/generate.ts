/**
 * CLI: generate self-play training data as a flat binary file.
 *
 * Usage:
 *   npx tsx packages/ai/src/generate.ts [numGames] [depth] [outPath]
 *
 * Load in Python / JAX:
 *   import numpy as np
 *   data = np.fromfile('selfplay.bin', dtype=np.float32).reshape(-1, 1220)
 *   states  = data[:, :1218]              # 33×6×6 board + 30 bag features
 *   actions = data[:, 1218].astype(int)   # action index (0–4427)
 *   values  = data[:, 1219]               # +1 win / −1 loss / 0 draw (current player POV)
 */

import {writeFileSync} from 'node:fs';
import {selfPlayBatch, SAMPLE_FLOATS} from './selfplay.js';
import {STATE_TENSOR_SIZE, BOARD_TENSOR_SIZE, NUM_CHANNELS, TOTAL_ACTIONS, TILE_NAMES} from './encoding.js';

const numGames = parseInt(process.argv[2] ?? '100', 10);
const depth    = parseInt(process.argv[3] ?? '4', 10);
const outPath  = process.argv[4] ?? 'selfplay.bin';

console.log(`Self-play: ${numGames} games, depth ${depth}`);
const t0 = performance.now();

const {data, totalSamples, outcomes} = selfPlayBatch(numGames, {depth});

const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
const p1Wins = outcomes.filter(o => o === 1).length;
const p2Wins = outcomes.filter(o => o === -1).length;
const draws  = outcomes.filter(o => o === 0).length;

console.log(`  ${totalSamples} samples from ${numGames} games in ${elapsed}s`);
console.log(`  P1 wins: ${p1Wins}  P2 wins: ${p2Wins}  draws: ${draws}`);

writeFileSync(outPath, Buffer.from(data.buffer, data.byteOffset, data.byteLength));

const kb = (data.byteLength / 1024).toFixed(0);
console.log(`\nWrote ${outPath} (${kb} KB)`);
console.log(`\nTensor layout (per sample = ${SAMPLE_FLOATS} float32s):`);
console.log(`  [0:${BOARD_TENSOR_SIZE}]     board planes (${NUM_CHANNELS}ch × 6×6)`);
console.log(`  [${BOARD_TENSOR_SIZE}:${STATE_TENSOR_SIZE}]  bag features (${TILE_NAMES.length}×2)`);
console.log(`  [${STATE_TENSOR_SIZE}]        action index (0–${TOTAL_ACTIONS - 1})`);
console.log(`  [${STATE_TENSOR_SIZE + 1}]        value target`);
console.log(`\nPython:`);
console.log(`  data = np.fromfile('${outPath}', dtype=np.float32).reshape(-1, ${SAMPLE_FLOATS})`);
