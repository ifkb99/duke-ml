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
 * The test network (h = tanh(x W), logits = h U, value = tanh(h v), x = the
 * flattened planes and features) is simple enough to write identically in JAX.
 * S: {tile, owner, side, bags (flat [2*15]), player, pending, ply, done,
 *     winner, reward, switched, legal (indices), placeOk ([4])}.
 */

import {createInterface} from 'node:readline';
import * as R from '../../../packages/ai/src/zero/rules.js';
import * as E from '../../../packages/ai/src/zero/env.js';
import {search, type NetOutput} from '../../../packages/ai/src/zero/search.js';

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

async function evaluate(states: E.State[]): Promise<NetOutput[]> {
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
    case 'net':
      net = {W: req.W, U: req.U, v: req.v};
      reply = {};
      break;
    case 'search': {
      const r = await search(fromJson(req.state), evaluate, {
        simulations: req.simulations, maxConsidered: req.maxConsidered, maxDepth: req.maxDepth,
        gumbel: req.gumbel, maxBatch: req.maxBatch});
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
