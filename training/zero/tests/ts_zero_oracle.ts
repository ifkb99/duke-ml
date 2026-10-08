/**
 * The TypeScript port of the zero environment (packages/ai/src/zero) as a
 * test oracle: the Python tests send JAX states and compare the replies.
 *
 * One JSON request per line on stdin, one reply per line on stdout:
 *   {"op": "tables"}                      -> rule tables as lists of set indices
 *   {"op": "legal", "state": S}           -> {"legal": [...], "placeOk": [...]}
 *   {"op": "step", "state": S, "action": a}     -> {"state": S}  (afterstate)
 *   {"op": "chance", "state": S, "outcome": c}  -> {"state": S, "probs": [...]}
 *   {"op": "observe", "state": S}         -> {"planes": [...], "features": [...]}
 *   {"op": "net", "W": [[..]], "U": [[..]], "v": [..]}   -> {} sets the test network
 *   {"op": "search", "state": S, "simulations": n, "maxConsidered": m, "maxDepth": d,
 *    "gumbel"?: [...], "maxBatch"?: b}  -> {"action", "weights" (legal only), "searchValue", ...}
 *   {"op": "fromTs", "game": SerializedGameState}  -> {"state": S}
 *   {"op": "toMove", "state": S, "actions": [...]}  -> {"moves": [GameMove...]}
 *   {"op": "onnx", "path": file}  -> {} uses that ONNX model (onnxruntime-web on the CPU)
 *     instead of the test network for "search"
 *   {"op": "onnxQuant", "path": file, "packed": file, "quant": QuantInfo, "expect": file,
 *    "planes": [...], "features": [...], "n": b}  -> {"mismatches", "value", "logits"}
 *     loads a quantized export as the web app does (quant.ts), compares the
 *     rebuilt weights with the export's .weights file, then evaluates the planes
 * The test network (h = tanh(x W), logits = h U, value = tanh(h v), x = the
 * flattened planes and features) is simple enough to write identically in JAX.
 * S: {tile, owner, side, bags (flat [2*15]), player, pending, ply, done,
 *     winner, reward, switched, legal (indices), placeOk ([4])}.
 */

import {createInterface} from 'node:readline';
import * as R from '../../../packages/ai/src/zero/rules.js';
import * as E from '../../../packages/ai/src/zero/env.js';
import {search, type NetOutput} from '../../../packages/ai/src/zero/search.js';
import {actionToMove, fromGameState, moveToAction} from '../../../packages/ai/src/zero/bridge.js';
import {onnxEvaluate} from '../../../packages/ai/src/zero/onnx.js';
import {dequantize} from '../../../packages/ai/src/zero/quant.js';
import {readFileSync} from 'node:fs';
import type {Evaluate} from '../../../packages/ai/src/zero/search.js';

type Json = Record<string, any>;

function fromJson(j: Json): E.State {
  const legal = new Uint8Array(R.N_ACTIONS);
  for (const a of j.legal) legal[a] = 1;
  return {
    tile: Int8Array.from(j.tile), owner: Uint8Array.from(j.owner), side: Uint8Array.from(j.side),
    bags: Int32Array.from(j.bags), player: j.player, pending: j.pending, ply: j.ply, done: j.done,
    winner: j.winner, reward: j.reward, switched: j.switched, legal, placeOk: Uint8Array.from(j.placeOk),
  };
}

function toJson(s: E.State): Json {
  return {
    tile: [...s.tile], owner: [...s.owner], side: [...s.side], bags: [...s.bags], player: s.player,
    pending: s.pending, ply: s.ply, done: s.done, winner: s.winner, reward: s.reward, switched: s.switched,
    legal: ones(s.legal), placeOk: [...s.placeOk],
  };
}

const ones = (a: ArrayLike<number>) => {
  const out = [];
  for (let i = 0; i < a.length; i++) if (a[i]) out.push(i);
  return out;
};

let net: {W: number[][]; U: number[][]; v: number[]} | null = null;
let onnx: Evaluate | null = null;

async function testNet(states: E.State[]): Promise<NetOutput[]> {
  const {W, U, v} = net!;
  const H = v.length;
  const planes = new Float32Array(E.PLANES_SIZE), features = new Float32Array(E.N_FEATURES);
  return states.map(s => {
    E.observe(s, planes, features);
    const x = [...planes, ...features];
    const h = new Float64Array(H);
    for (let j = 0; j < H; j++) {
      let acc = 0;
      for (let i = 0; i < x.length; i++) if (x[i]) acc += x[i] * W[i][j];
      h[j] = Math.tanh(acc);
    }
    const logits = new Float32Array(R.N_ACTIONS);
    for (let a = 0; a < R.N_ACTIONS; a++) {
      let acc = 0;
      for (let j = 0; j < H; j++) acc += h[j] * U[j][a];
      logits[a] = acc;
    }
    let value = 0;
    for (let j = 0; j < H; j++) value += h[j] * v[j];
    return {logits, value: Math.tanh(value)};
  });
}

const queue: Json[] = [];
let busy = false;
const rl = createInterface({input: process.stdin});
rl.on('line', line => { queue.push(JSON.parse(line)); if (!busy) drain(); });

async function drain() {
  busy = true;
  while (queue.length) process.stdout.write(JSON.stringify(await handle(queue.shift()!)) + '\n');
  busy = false;
}

async function handle(req: Json): Promise<Json> {
  let reply: Json;
  switch (req.op) {
    case 'fromTs':
      reply = {state: toJson(fromGameState(req.game))};
      break;
    case 'toMove': {
      const st = fromJson(req.state);
      const moves = req.actions.map((a: number) => actionToMove(a, st));
      reply = {moves, back: moves.map((m: any) => moveToAction(m, st))};
      break;
    }
    case 'net':
      net = {W: req.W, U: req.U, v: req.v};
      onnx = null;
      reply = {};
      break;
    case 'onnx': {
      const ort = await import('onnxruntime-web');
      ort.env.wasm.numThreads = 1;
      const session = await ort.InferenceSession.create(req.path, {executionProviders: ['wasm']});
      onnx = onnxEvaluate(session, ort.Tensor);
      reply = {};
      break;
    }
    case 'onnxQuant': {
      const ort = await import('onnxruntime-web');
      ort.env.wasm.numThreads = 1;
      const weights = dequantize(new Uint8Array(readFileSync(req.packed)), req.quant);
      const expect = new Uint8Array(readFileSync(req.expect));
      let mismatches = Math.abs(expect.length - weights.length);
      for (let i = 0; i < Math.min(expect.length, weights.length); i++) mismatches += +(expect[i] !== weights[i]);
      const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(req.path)), {
        executionProviders: ['wasm'], externalData: [{path: req.quant.external, data: weights}]});
      const out = await session.run({
        planes: new ort.Tensor('float32', Float32Array.from(req.planes), [req.n, R.BOARD, R.BOARD, req.planes.length / (req.n * R.BOARD * R.BOARD)]),
        features: new ort.Tensor('float32', Float32Array.from(req.features), [req.n, req.features.length / req.n])});
      reply = {mismatches, value: Array.from(out.value.data as Float32Array), logits: Array.from(out.logits.data as Float32Array)};
      break;
    }
    case 'search': {
      const r = await search(fromJson(req.state), onnx ?? testNet, {
        simulations: req.simulations, maxConsidered: req.maxConsidered, maxDepth: req.maxDepth,
        gumbel: req.gumbel, maxBatch: req.maxBatch, valueScale: req.valueScale});
      const legal = ones(fromJson(req.state).legal);
      reply = {action: r.action, weights: legal.map(a => r.weights[a]), searchValue: r.searchValue,
        rootValue: r.rootValue, evaluations: r.evaluations, batches: r.batches,
        visits: [...r.visits.entries()], q: [...r.q.entries()]};
      break;
    }
    case 'tables':
      reply = {MOVE_OK: ones(R.MOVE_OK), NEEDS_CLEAR: ones(R.NEEDS_CLEAR), STRIKE_OK: ones(R.STRIKE_OK),
        CMD_DIR: ones(R.CMD_DIR), BETWEEN: ones(R.BETWEEN), INITIAL_BAG: [...R.INITIAL_BAG],
        TILE_NAMES: R.TILE_NAMES};
      break;
    case 'legal': {
      const s = fromJson(req.state);
      const {legal, placeOk} = E.legalActions(s.tile, s.owner, s.side, s.bags, s.pending);
      reply = {legal: ones(legal), placeOk: [...placeOk]};
      break;
    }
    case 'step':
      reply = {state: toJson(E.stepDecision(fromJson(req.state), req.action))};
      break;
    case 'chance': {
      const s = fromJson(req.state);
      reply = {state: toJson(E.stepChance(s, req.outcome)), probs: [...E.chanceProbs(s)]};
      break;
    }
    case 'observe': {
      const planes = new Float32Array(E.PLANES_SIZE), features = new Float32Array(E.N_FEATURES);
      E.observe(fromJson(req.state), planes, features);
      reply = {planes: [...planes], features: [...features]};
      break;
    }
    default:
      reply = {error: `unknown op ${req.op}`};
  }
  return reply;
}
