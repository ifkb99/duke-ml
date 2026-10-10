/**
 * Gumbel MuZero search with chance nodes, ported from training/zero/search.py
 * (which assembles it from DeepMind's mctx). Given the same network outputs
 * it reproduces the JAX search: same tree, same action, same improved policy
 * (tested from Python in training/zero/tests/test_ts_zero.py).
 *
 * Tree: decision nodes (a state, player X to move) alternate with chance
 * nodes (the afterstate of X's action). A chance node's children are the draw
 * outcomes (which tile came out of the bag); after any other action there is
 * one certain outcome. Values are backed up as Q = reward + discount * value,
 * discount -1 when the turn passes, so every Q is from the chooser's view.
 *
 * Batching. The network is the slow part, and WebGPU is only fast on batches.
 * Simulations are batched whenever that cannot change the result:
 *   - each one in a batch starts with a different root action, and below the
 *     root a simulation only reads and writes its own subtree;
 *   - the root actions of a batch are a set fixed in advance: either every
 *     action still in the race gets its next visit (a round of sequential
 *     halving), or the actions are all still unvisited, whose ranking ignores
 *     Q values.
 * Choosing which actions survive a halving depends on the latest values, so
 * those simulations run one at a time. Network outputs are also cached: after
 * a move (not a draw) the afterstate IS the next state, which mctx evaluates
 * twice.
 */

import {DRAWING, type State, chanceProbs, stepChance, stepDecision} from './env.js';
import {N_ACTIONS, N_TYPES, SPEC} from './rules.js';

export interface NetOutput {
  logits: Float32Array;   // [3605] policy logits
  value: number;          // for the player to move, in [-1, 1]
}

/** Evaluates a batch of states with the network. */
export type Evaluate = (states: State[]) => Promise<NetOutput[]>;

export interface SearchOptions {
  simulations: number;
  maxConsidered?: number;       // root actions in the first halving round (spec.json)
  maxDepth?: number;            // tree edges per simulation; 2 per move (spec.json)
  gumbel?: ArrayLike<number>;   // [3605] root noise; omit for the strongest play
  maxBatch?: number;            // cap on positions per network call (64)
  valueScale?: number;          // search values vs prior (spec.json)
}

export interface SearchResult {
  action: number;
  weights: Float64Array;        // [3605] improved policy (the training target)
  rootValue: number;            // network value of the root
  searchValue: number;          // root value after search
  visits: Map<number, number>;  // root action -> visits
  q: Map<number, number>;       // root action -> Q (visited actions only)
  evaluations: number;          // network evaluations (positions)
  batches: number;              // network calls
}

// qtransform_completed_by_mix_value settings (shared with search.py), and two
// constants inside mctx itself: its clamp on prior probabilities (float32
// tiny) and score_considered's floor.
const MAXVISIT_INIT = SPEC.search.maxvisitInit;
const RESCALE_EPS = SPEC.search.rescaleEpsilon;
const CHANCE_TIE = SPEC.search.chanceTieTolerance;
const TINY = 1.1754944e-38;
const LOW_LOGIT = -1e9;

class DecisionNode {
  readonly actions: number[];             // legal actions (ascending)
  readonly logits: Float64Array;          // their prior logits
  rawValue: number;
  value: number;
  visits = 1;
  readonly child: (ChanceNode | null)[];
  readonly childVisits: Int32Array;
  readonly childValue: Float64Array;      // Q = 0 + 1 * child value

  constructor(readonly state: State, net: NetOutput | null, value: number,
              readonly parent: ChanceNode | null, readonly parentIdx: number) {
    this.actions = [];
    for (let a = 0; a < N_ACTIONS; a++) if (state.legal[a]) this.actions.push(a);
    this.logits = Float64Array.from(this.actions, a => (net ? net.logits[a] : 0));
    this.rawValue = this.value = value;
    this.child = this.actions.map(() => null);
    this.childVisits = new Int32Array(this.actions.length);
    this.childValue = new Float64Array(this.actions.length);
  }
}

class ChanceNode {
  readonly probs: Float64Array;           // [15] outcome probabilities
  rawValue: number;
  value: number;
  visits = 1;
  readonly child: (DecisionNode | null)[] = new Array(N_TYPES).fill(null);
  readonly childVisits = new Int32Array(N_TYPES);
  readonly childValue = new Float64Array(N_TYPES);
  readonly childReward = new Float64Array(N_TYPES);
  readonly childDiscount = new Float64Array(N_TYPES);

  constructor(readonly state: State, readonly net: NetOutput | null, value: number,
              readonly parent: DecisionNode, readonly parentIdx: number) {
    this.probs = chanceProbs(state);
    this.rawValue = this.value = value;
  }
}

type Node = DecisionNode | ChanceNode;

function softmax(x: ArrayLike<number>): Float64Array {
  let m = -Infinity;
  for (let i = 0; i < x.length; i++) m = Math.max(m, x[i]);
  const out = new Float64Array(x.length);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (out[i] = Math.exp(x[i] - m));
  for (let i = 0; i < x.length; i++) out[i] /= s;
  return out;
}

/**
 * mctx's qtransform_completed_by_mix_value over a decision node's children.
 * Illegal actions are unvisited children with zero prior: they get the mixed
 * value, which also enters the min/max rescaling.
 */
function completedQ(n: DecisionNode, valueScale: number): Float64Array {
  const k = n.actions.length;
  const probs = softmax(n.logits);
  let sumVisits = 0, sumProbs = 0, weighted = 0, maxVisit = 0;
  for (let i = 0; i < k; i++) {
    if (n.childVisits[i] > 0) {
      sumVisits += n.childVisits[i];
      sumProbs += Math.max(TINY, probs[i]);
      maxVisit = Math.max(maxVisit, n.childVisits[i]);
    }
  }
  for (let i = 0; i < k; i++) {
    if (n.childVisits[i] > 0) weighted += Math.max(TINY, probs[i]) * n.childValue[i] / sumProbs;
  }
  const mixed = (n.rawValue + sumVisits * weighted) / (sumVisits + 1);
  const q = new Float64Array(k);
  let lo = k < N_ACTIONS ? mixed : Infinity, hi = k < N_ACTIONS ? mixed : -Infinity;
  for (let i = 0; i < k; i++) {
    q[i] = n.childVisits[i] > 0 ? n.childValue[i] : mixed;
    lo = Math.min(lo, q[i]); hi = Math.max(hi, q[i]);
  }
  const scale = (MAXVISIT_INIT + maxVisit) * valueScale / Math.max(hi - lo, RESCALE_EPS);
  for (let i = 0; i < k; i++) q[i] = (q[i] - lo) * scale;
  return q;
}

/** Below the root: the child most under-visited relative to the improved policy. */
function interiorSelect(n: DecisionNode, valueScale: number): number {
  const q = completedQ(n, valueScale);
  const probs = softmax(n.logits.map((l, i) => l + q[i]));
  let total = 0;
  for (let i = 0; i < n.actions.length; i++) total += n.childVisits[i];
  let best = 0, bestScore = -Infinity;
  for (let i = 0; i < n.actions.length; i++) {
    const s = probs[i] - n.childVisits[i] / (1 + total);
    if (s > bestScore) { bestScore = s; best = i; }
  }
  return best;
}

/** Chance nodes: the outcome whose visit share lags its probability most
 *  (near-ties to the lowest outcome, as search.py does). */
function chanceSelect(n: ChanceNode): number {
  let top = -Infinity;
  for (let c = 0; c < N_TYPES; c++) top = Math.max(top, n.probs[c] / (n.childVisits[c] + 1));
  for (let c = 0; c < N_TYPES; c++) if (n.probs[c] / (n.childVisits[c] + 1) >= top - CHANCE_TIE) return c;
  return 0;
}

/** The Gumbel root score of every legal root action (-Infinity if not a candidate). */
function rootScores(root: DecisionNode, gumbel: ArrayLike<number> | undefined, target: number,
                    visits: ArrayLike<number>, valueScale: number): Float64Array {
  const q = completedQ(root, valueScale);
  let maxLogit = -Infinity;
  for (const l of root.logits) maxLogit = Math.max(maxLogit, l);
  return Float64Array.from(root.actions, (a, i) => {
    if (visits[i] !== target) return -Infinity;
    const g = gumbel ? gumbel[a] : 0;
    return Math.max(LOW_LOGIT, g + root.logits[i] - maxLogit + q[i]);
  });
}

function argmax(x: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

/** mctx seq_halving.get_sequence_of_considered_visits. */
export function consideredVisits(maxConsidered: number, simulations: number): number[] {
  if (maxConsidered <= 1) return Array.from({length: simulations}, (_, i) => i);
  const log2max = Math.ceil(Math.log2(maxConsidered));
  const seq: number[] = [];
  const visits = new Array(maxConsidered).fill(0);
  let considered = maxConsidered;
  while (seq.length < simulations) {
    const extra = Math.max(1, Math.floor(simulations / (log2max * considered)));
    for (let e = 0; e < extra; e++) {
      seq.push(...visits.slice(0, considered));
      for (let i = 0; i < considered; i++) visits[i]++;
    }
    considered = Math.max(2, Math.floor(considered / 2));
  }
  return seq.slice(0, simulations);
}

/** Back the leaf's value up to the root (mctx backward). */
function backup(leaf: Node) {
  let v = leaf.value;
  let node: Node = leaf;
  while (node.parent) {
    const p: Node = node.parent;
    const i = node.parentIdx;
    v = p instanceof ChanceNode ? p.childReward[i] + p.childDiscount[i] * v : v;
    p.value = (p.value * p.visits + v) / (p.visits + 1);
    p.visits += 1;
    p.childValue[i] = node.value;
    p.childVisits[i] += 1;
    node = p;
  }
}

/** A simulation waiting for its leaf to be evaluated. */
interface Pending {
  terminal?: DecisionNode; // walked into a finished game: no new node
  parent: Node;
  idx: number;            // child index in the parent
  state: State;           // the leaf's state
  existing: Node | null;  // depth cutoff on an existing child (see expand)
  net: NetOutput | null;  // known without the network (cache or terminal)
  needsNet: boolean;
}

export async function search(rootState: State, evaluate: Evaluate, opts: SearchOptions): Promise<SearchResult> {
  const sims = opts.simulations;
  const maxConsidered = opts.maxConsidered ?? SPEC.search.maxConsidered;
  const maxDepth = opts.maxDepth ?? SPEC.search.maxDepth;
  const maxBatch = opts.maxBatch ?? 64;
  const valueScale = opts.valueScale ?? SPEC.search.valueScale;
  let evaluations = 1, batches = 1;

  const [rootNet] = await evaluate([rootState]);
  const root = new DecisionNode(rootState, rootNet, rootNet.value, null, -1);
  const k = root.actions.length;
  const table = consideredVisits(Math.min(maxConsidered, k), sims);

  /** Walk down from the root; returns where the new leaf goes. */
  function simulate(rootIdx: number): Pending {
    let node: Node = root, idx = rootIdx, depth = 0;
    for (;;) {
      if (node instanceof DecisionNode && node.state.done) {
        return {terminal: node, parent: node, idx: 0, state: node.state, existing: null, net: null, needsNet: false};
      }
      if (depth > 0) idx = node instanceof DecisionNode ? interiorSelect(node, valueScale) : chanceSelect(node);
      const child: Node | null = node.child[idx];
      depth += 1;
      if (child && depth < maxDepth) { node = child; continue; }
      return leafFor(node, idx, child);
    }
  }

  function leafFor(parent: Node, idx: number, existing: Node | null): Pending {
    if (parent instanceof DecisionNode) {
      const after = existing?.state ?? stepDecision(parent.state, parent.actions[idx]);
      return {parent, idx, state: after, existing, net: null, needsNet: !after.done};
    }
    const nxt = existing?.state ?? stepChance(parent.state, idx);
    // After a move the afterstate is the next state: reuse its evaluation.
    const cached = parent.state.pending !== DRAWING ? parent.net : null;
    return {parent, idx, state: nxt, existing, net: cached, needsNet: !nxt.done && !cached};
  }

  /** Create the leaf (or, at the depth cutoff, refresh an existing node, as mctx does). */
  function expand(p: Pending): Node {
    const {parent, idx, state} = p;
    let leaf: Node;
    if (parent instanceof DecisionNode) {
      const v = state.done ? state.reward : state.switched ? -p.net!.value : p.net!.value;
      leaf = p.existing ?? (parent.child[idx] = new ChanceNode(state, p.net, v, parent, idx));
      leaf.rawValue = v;
    } else {
      const v = state.done ? 0 : p.net!.value;
      parent.childReward[idx] = parent.state.reward;
      parent.childDiscount[idx] = state.done ? 0 : parent.state.switched ? -1 : 1;
      leaf = p.existing ?? (parent.child[idx] = new DecisionNode(state, p.net, v, parent, idx));
      leaf.rawValue = v;
    }
    if (p.existing) {
      // mctx re-expands a node reached at the depth cutoff: its value is
      // reset to the network's, its visit count goes up by one.
      leaf.value = leaf.rawValue;
      leaf.visits += 1;
    }
    return leaf;
  }

  async function run(group: Pending[]) {
    const need = group.filter(p => p.needsNet);
    if (need.length) {
      const outs = await evaluate(need.map(p => p.state));
      need.forEach((p, i) => { p.net = outs[i]; });
      evaluations += need.length;
      batches += 1;
    }
    for (const p of group) {
      if (p.terminal) {
        // mctx expands a dummy child worth 0 below a finished game: the
        // node's value stays 0 and its parent gets reward + 0 * 0.
        const t = p.terminal;
        t.value = t.value * t.visits / (t.visits + 1);
        t.visits += 1;
        backup(t);
      } else {
        backup(expand(p));
      }
    }
  }

  let i = 0;
  while (i < sims) {
    const target = table[i];
    let run_ = 0;
    while (i + run_ < sims && table[i + run_] === target) run_++;
    const candidates = root.actions.filter((_, j) => root.childVisits[j] === target).length;
    const batchable = target === 0 || candidates === run_;
    const size = batchable ? Math.min(run_, maxBatch) : 1;

    // Pick this batch's root actions. `claimed` counts visits handed out but
    // not yet backed up, so the next pick in the batch skips them.
    const visits = Int32Array.from(root.childVisits);
    const group: Pending[] = [];
    for (let b = 0; b < size; b++) {
      const rootIdx = argmax(rootScores(root, opts.gumbel, target, visits, valueScale));
      visits[rootIdx] += 1;
      group.push(simulate(rootIdx));
    }
    await run(group);
    i += size;
  }

  // The result, read off the root as Gumbel MuZero does.
  const q = completedQ(root, valueScale);
  let maxVisit = 0;
  for (const v of root.childVisits) maxVisit = Math.max(maxVisit, v);
  const actionIdx = argmax(rootScores(root, opts.gumbel, maxVisit, root.childVisits, valueScale));
  const improved = softmax(root.logits.map((l, j) => l + q[j]));
  const weights = new Float64Array(N_ACTIONS);
  const visitMap = new Map<number, number>(), qMap = new Map<number, number>();
  root.actions.forEach((a, j) => {
    weights[a] = improved[j];
    if (root.childVisits[j]) { visitMap.set(a, root.childVisits[j]); qMap.set(a, root.childValue[j]); }
  });
  return {action: root.actions[actionIdx], weights, rootValue: rootNet.value, searchValue: root.value,
    visits: visitMap, q: qMap, evaluations, batches};
}
