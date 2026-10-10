import type {SerializedGameState, GameMove} from '@the-duke/engine';
import {deserialize} from '@the-duke/engine';
import {findBestMove, pickRandomMove, onnxEvaluate, zeroChooseMove, TimeBudget, dequantize, type QuantInfo,
        type ZeroEvaluate} from '@the-duke/ai';

export interface AIWorkerRequest {
  type: 'findBestMove' | 'pickRandomMove' | 'zeroMove';
  state: SerializedGameState;
  depth?: number;
  /** zeroMove: search simulations per decision (fixed) */
  simulations?: number;
  /** zeroMove: thinking time per decision in ms, instead of fixed simulations */
  budgetMs?: number;
  /** zeroMove: post each move's timing, and the worker's log, to the test server (tools/latency/serve.py) */
  report?: boolean;
  /** zeroMove: load the network on this backend only ('webgpu' or 'wasm'), e.g. from ?backend=wasm */
  backend?: string;
  /** zeroMove: WebAssembly threads, e.g. from ?threads=4 (default 1) */
  threads?: number;
}

// The neural network (training/zero, exported with export_onnx.py), loaded on
// first use: WebGPU if the browser has it, else WebAssembly on the CPU. Its
// .json (also from export_onnx.py) has the search settings it was trained with
// and names the variant to load: `web`. A quantized variant (q8, q4) is a small
// graph plus a .bin of packed weights, expanded here to the fp16 weights the
// graph expects; fp16 (or a .json without `web`) is the plain fp16 model.
const MODELS = `${import.meta.env.BASE_URL}models/`;
interface SearchSettings { maxConsidered?: number; maxDepth?: number; valueScale?: number }
interface ModelMeta {
  search?: SearchSettings;
  web?: string;
  variants?: Record<string, {file: string; packed?: string; quant?: QuantInfo}>;
}
let zero: Promise<{evaluate: ZeroEvaluate; search: SearchSettings}> | null = null;

async function fetchBytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function loadModel(meta: ModelMeta): Promise<{model: Uint8Array; externalData?: {path: string; data: Uint8Array}[]}> {
  const v = meta.web ? meta.variants?.[meta.web] : undefined;
  if (!v?.packed || !v.quant) return {model: await fetchBytes(`${MODELS}duke-zero.fp16.onnx`)};
  const [model, packed] = await Promise.all([fetchBytes(MODELS + v.file), fetchBytes(MODELS + v.packed)]);
  return {model, externalData: [{path: v.quant.external, data: dequantize(packed, v.quant)}]};
}

async function loadZero(only?: string, threads?: number): Promise<{evaluate: ZeroEvaluate; search: SearchSettings}> {
  const meta: ModelMeta = await fetch(`${MODELS}duke-zero.json`).then(r => (r.ok ? r.json() : {})).catch(() => ({}));
  const search: SearchSettings = meta.search ?? {};   // missing: spec.json defaults
  const ort = await import('onnxruntime-web/all');
  const wasm = await import('onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url');
  const mjs = await import('onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url');
  ort.env.wasm.wasmPaths = {wasm: wasm.default, mjs: mjs.default};
  // One thread unless asked (?threads=N, which needs a cross-origin isolated
  // page): ORT's threads are workers inside this worker, and on Android Chrome
  // they never start, so creating the session hangs. The search evaluates a few
  // positions at a time, too few for threads to pay anyway (~13 ms per
  // evaluation on a phone with 1 or 7).
  ort.env.wasm.numThreads = threads && self.crossOriginIsolated ? threads : 1;
  const t0 = performance.now();
  const {model, externalData} = await loadModel(meta);
  console.log(`[AI] network ${meta.web ?? 'fp16'} fetched and unpacked in ${(performance.now() - t0).toFixed(0)}ms`);
  const gpu = 'gpu' in navigator && await (navigator as any).gpu.requestAdapter().catch(() => null);
  for (const name of only ? [only] : gpu ? ['webgpu', 'wasm'] : ['wasm']) {
    try {
      console.log(`[AI] creating a session on ${name}`);
      const session = await ort.InferenceSession.create(model, {executionProviders: [name], externalData});
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
const logged: string[] = [];
let reporting = false;

function sendReport() {
  if (!reporting) return;
  fetch(`${import.meta.env.BASE_URL}report`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session: reportSession, env: {browser: navigator.userAgent,
      threads: navigator.hardwareConcurrency, isolated: self.crossOriginIsolated}, results: reported, log: logged}),
  }).catch(() => {});
}

// With ?report, the worker's console goes to the test server as well: a phone
// has no console to read, and a load that fails or hangs never reaches a move.
for (const level of ['log', 'warn', 'error'] as const) {
  const original = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    original(...args);
    logged.push(`${(performance.now() / 1000).toFixed(1)}s ${level}: ${args.map(a => a instanceof Error ? `${a.name}: ${a.message}` : String(a)).join(' ')}`);
    sendReport();
  };
}

async function zeroMove(serialized: SerializedGameState, simulations: number | undefined, budgetMs: number,
                        report: boolean, only?: string, threads?: number) {
  const t0 = performance.now();
  const firstMove = zero === null;
  reporting = report;
  if (firstMove) console.log(`[AI] loading the network${only ? ` (${only} only)` : ''}${threads ? `, ${threads} threads` : ''}`);
  zero ??= loadZero(only, threads);
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
    sendReport();
  }
  self.postMessage({move, score: info.value, nodesSearched: info.evaluations} satisfies AIWorkerResponse);
}

self.onmessage = (e: MessageEvent<AIWorkerRequest>) => {
  if (e.data.type === 'zeroMove') {
    zeroMove(e.data.state, e.data.simulations, e.data.budgetMs ?? 3000, !!e.data.report, e.data.backend, e.data.threads).catch(err => {
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
