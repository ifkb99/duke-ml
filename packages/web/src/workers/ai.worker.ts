import type {SerializedGameState, GameMove} from '@the-duke/engine';
import {deserialize} from '@the-duke/engine';
import {findBestMove, pickRandomMove, onnxEvaluate, zeroChooseMove, TimeBudget, type ZeroEvaluate} from '@the-duke/ai';

export interface AIWorkerRequest {
  type: 'findBestMove' | 'pickRandomMove' | 'zeroMove';
  state: SerializedGameState;
  depth?: number;
  /** zeroMove: search simulations per decision (fixed) */
  simulations?: number;
  /** zeroMove: thinking time per decision in ms, instead of fixed simulations */
  budgetMs?: number;
  /** zeroMove: post each move's timing to the test server (tools/latency/serve.py) */
  report?: boolean;
}

// The neural network (training/zero, exported with export_onnx.py), loaded on
// first use: WebGPU if the browser has it, else WebAssembly on the CPU. Its
// .json (also from export_onnx.py) has the search settings it was trained with.
const MODEL_URL = `${import.meta.env.BASE_URL}models/duke-zero.fp16.onnx`;
const META_URL = `${import.meta.env.BASE_URL}models/duke-zero.json`;
interface SearchSettings { maxConsidered?: number; maxDepth?: number; valueScale?: number }
let zero: Promise<{evaluate: ZeroEvaluate; search: SearchSettings}> | null = null;

async function loadZero(): Promise<{evaluate: ZeroEvaluate; search: SearchSettings}> {
  const meta: {search?: SearchSettings} = await fetch(META_URL).then(r => (r.ok ? r.json() : {})).catch(() => ({}));
  const search: SearchSettings = meta.search ?? {};   // missing: spec.json defaults
  const ort = await import('onnxruntime-web/all');
  const wasm = await import('onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url');
  const mjs = await import('onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url');
  ort.env.wasm.wasmPaths = {wasm: wasm.default, mjs: mjs.default};
  // Threads need a cross-origin isolated page (COOP/COEP headers).
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 1, 8) : 1;
  const t0 = performance.now();
  const gpu = 'gpu' in navigator && await (navigator as any).gpu.requestAdapter().catch(() => null);
  for (const name of gpu ? ['webgpu', 'wasm'] : ['wasm']) {
    try {
      const session = await ort.InferenceSession.create(MODEL_URL, {executionProviders: [name]});
      backend = name;
      console.log(`[AI] network loaded on ${name} in ${(performance.now() - t0).toFixed(0)}ms`
        + (name === 'wasm' ? ` (${ort.env.wasm.numThreads} threads)` : ''));
      return {evaluate: onnxEvaluate(session, ort.Tensor), search};
    } catch (err) {
      console.warn(`[AI] ${name} failed`, err);
    }
  }
  throw new Error('could not load the neural network');
}

export interface AIWorkerResponse {
  move: GameMove | null;
  score: number;
  nodesSearched: number;
  error?: string;
}

let backend = '';
let budget: TimeBudget | null = null;     // kept across moves: it learns this device's speed
const reportSession = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19) + '-neural';
const reported: object[] = [];

async function zeroMove(serialized: SerializedGameState, simulations: number | undefined, budgetMs: number,
                        report: boolean) {
  const t0 = performance.now();
  const firstMove = zero === null;
  zero ??= loadZero();
  zero.catch(() => { zero = null; });   // let a later move retry
  const {evaluate, search} = await zero;
  const loadMs = firstMove ? performance.now() - t0 : 0;
  if (budget?.ms !== budgetMs) budget = new TimeBudget(budgetMs);
  const {move, info} = await zeroChooseMove(serialized, evaluate, {
    simulations, budget, maxConsidered: search.maxConsidered, maxDepth: search.maxDepth, valueScale: search.valueScale});
  console.log(`[AI] neural: ${info.simulations} sims, ${info.evaluations} evals in ${info.batches} batches, `
    + `${info.ms.toFixed(0)}ms, value ${info.value.toFixed(2)}${info.drewTile ? `, drew ${info.drewTile}` : ''}`);
  if (report) {
    reported.push({turn: serialized.turnNumber, budgetMs: simulations ? undefined : budgetMs, backend, loadMs,
                   ...info, move});
    fetch(`${import.meta.env.BASE_URL}report`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session: reportSession, env: {browser: navigator.userAgent,
        threads: navigator.hardwareConcurrency, isolated: self.crossOriginIsolated}, results: reported}),
    }).catch(() => {});
  }
  self.postMessage({move, score: info.value, nodesSearched: info.evaluations} satisfies AIWorkerResponse);
}

self.onmessage = (e: MessageEvent<AIWorkerRequest>) => {
  if (e.data.type === 'zeroMove') {
    zeroMove(e.data.state, e.data.simulations, e.data.budgetMs ?? 3000, !!e.data.report).catch(err => {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[AI Worker Error]', err);
      self.postMessage({move: null, score: 0, nodesSearched: 0, error: msg} satisfies AIWorkerResponse);
    });
    return;
  }
  try {
    const {type, state: serialized, depth} = e.data;
    const state = deserialize(serialized);

    if (type === 'pickRandomMove') {
      const move = pickRandomMove(state);
      self.postMessage({move, score: 0, nodesSearched: 0} satisfies AIWorkerResponse);
      return;
    }

    const d = depth ?? 4;
    const t0 = performance.now();
    const result = findBestMove(state, d);
    const elapsed = performance.now() - t0;
    console.log(`[AI] depth=${d} nodes=${result.nodesSearched} time=${elapsed.toFixed(0)}ms score=${result.score}`);
    self.postMessage({
      move: result.move,
      score: result.score,
      nodesSearched: result.nodesSearched,
    } satisfies AIWorkerResponse);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[AI Worker Error]', err);
    self.postMessage({move: null, score: 0, nodesSearched: 0, error: msg} satisfies AIWorkerResponse);
  }
};
