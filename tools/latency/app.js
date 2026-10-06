// Times the Duke network in this browser with onnxruntime-web.
// Models and their check data come from models/index.json (written by setup.sh).
import * as ort from './vendor/ort.all.min.mjs';

ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;

const PLANES = 6 * 6 * 60;
const FEATURES = 47;
const $ = id => document.getElementById(id);
const results = [];
const failures = [];
const session = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

function log(msg) {
  $('log').textContent += msg + '\n';
  $('log').scrollTop = $('log').scrollHeight;
}

// Send everything so far to the server (serve.py saves it under results/).
// Called after every row, so a sweep cut short still leaves its data behind.
// Reports are chained so an older snapshot never lands after a newer one.
let reporting = Promise.resolve();
function report(done = false) {
  const body = JSON.stringify({ session, done, env, results, failures, log: $('log').textContent });
  reporting = reporting.then(() => fetch('report', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
  })).catch(() => { /* served without serve.py: Copy results still works */ });
  return reporting;
}
addEventListener('error', e => { log(`error: ${e.message}`); report(); });
addEventListener('unhandledrejection', e => { log(`error: ${e.reason?.message ?? e.reason}`); report(); });

const hasWebGPU = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
const cores = navigator.hardwareConcurrency || 1;
const env = {
  'browser': navigator.userAgent,
  'CPU threads': cores,
  'cross-origin isolated (multi-thread CPU)': crossOriginIsolated ? 'yes' : 'no — CPU runs single-threaded',
  'WebGPU': hasWebGPU ? 'yes' : 'no',
};
for (const [k, v] of Object.entries(env)) $('env').insertAdjacentHTML('beforeend', `<dt>${k}</dt><dd>${v}</dd>`);
$('threads').value = crossOriginIsolated ? Math.min(cores, 8) : 1;

const models = await (await fetch('models/index.json')).json();
log(`models: ${models.map(m => m.name).join(', ')}`);

function inputsFor(meta, batch) {
  // The check positions, repeated to fill the batch.
  const { planes, features, n } = meta.check;
  const p = new Float32Array(batch * PLANES), f = new Float32Array(batch * FEATURES);
  for (let i = 0; i < batch; i++) {
    const j = i % n;
    p.set(planes.slice(j * PLANES, (j + 1) * PLANES), i * PLANES);
    f.set(features.slice(j * FEATURES, (j + 1) * FEATURES), i * FEATURES);
  }
  return {
    planes: new ort.Tensor('float32', p, [batch, 6, 6, 60]),
    features: new ort.Tensor('float32', f, [batch, FEATURES]),
  };
}

function check(meta, out) {
  // Compare with the outputs JAX produced for the check positions.
  const { n, value, top_move, legal } = meta.check;
  const logits = out.logits.data, v = out.value.data, A = logits.length / out.value.data.length;
  let agree = 0, maxErr = 0;
  for (let i = 0; i < n; i++) {
    let best = -1, bestScore = -Infinity;
    for (const a of legal[i]) if (logits[i * A + a] > bestScore) { bestScore = logits[i * A + a]; best = a; }
    if (best === top_move[i]) agree++;
    maxErr = Math.max(maxErr, Math.abs(v[i] - value[i]));
  }
  return { agree: agree / n, maxErr };
}

async function bench(meta, variant, backend, batches, runs) {
  const file = meta.variants[variant].file;
  const options = backend === 'gpu'
    ? { executionProviders: ['webgpu'] }
    : { executionProviders: ['wasm'] };
  // onnxruntime-web reads the thread count once, when its WebAssembly starts.
  ort.env.wasm.numThreads = Number($('threads').value) || 1;
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(`models/${file}`, options);
  log(`${meta.name} ${variant} ${backend}: loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

  const correct = check(meta, await session.run(inputsFor(meta, meta.check.n)));
  for (const batch of batches) {
    const feeds = inputsFor(meta, batch);
    for (let i = 0; i < 3; i++) await session.run(feeds);           // warm-up
    const times = [];
    for (let i = 0; i < runs; i++) {
      const t = performance.now();
      await session.run(feeds);
      times.push(performance.now() - t);
    }
    times.sort((a, b) => a - b);
    const ms = times[Math.floor(times.length / 2)];                   // median
    const row = {
      network: meta.name, params_m: meta.params / 1e6, mflops: meta.mflops_per_eval,
      format: variant, backend: backend === 'gpu' ? 'GPU (WebGPU)' : `CPU ×${ort.env.wasm.numThreads}`,
      mb: meta.variants[variant].bytes / 1e6, batch, ms_per_batch: ms, ms_per_eval: ms / batch,
      evals_per_s: 1000 * batch / ms, sims_in_3s: Math.floor(3 * 1000 * batch / ms),
      top_move_agrees: correct.agree, value_err: correct.maxErr,
    };
    results.push(row);
    addRow(row);
    $('copy').disabled = false;
    report();
    await new Promise(r => setTimeout(r, 0));
  }
  await session.release();
}

function addRow(r) {
  const ok = r.top_move_agrees >= 0.99 && r.value_err < 0.05;
  $('rows').insertAdjacentHTML('beforeend', `<tr>
    <td>${r.network} <span style="color:var(--muted)">(${r.mflops.toFixed(0)} MFLOP)</span></td>
    <td>${r.format}</td><td>${r.backend}</td><td>${r.mb.toFixed(1)}</td><td>${r.batch}</td>
    <td>${r.ms_per_batch.toFixed(1)}</td><td>${r.ms_per_eval.toFixed(2)}</td>
    <td>${r.evals_per_s.toFixed(0)}</td><td><b>${r.sims_in_3s}</b></td>
    <td class="${ok ? 'ok' : 'bad'}">${ok ? '✓' : `✗ ${(r.top_move_agrees * 100).toFixed(0)}%`}</td></tr>`);
}

async function sweep(full, gpuOnly = false) {
  for (const b of BUTTONS) $(b).disabled = true;
  const runs = Number($('runs').value) || 20;
  const plan = [];
  for (const m of models) {
    if (!gpuOnly) plan.push([m, 'int8', 'cpu', full ? [1, 8, 32] : [1, 8]]);
    if (full && !gpuOnly) plan.push([m, 'fp32', 'cpu', [1, 8, 32]], [m, 'fp16', 'cpu', [8]]);
    if (hasWebGPU) plan.push([m, 'fp16', 'gpu', full || gpuOnly ? [1, 8, 32, 128] : [8, 32]]);
    if (hasWebGPU && (full || gpuOnly)) plan.push([m, 'fp32', 'gpu', [8, 32]]);
  }
  for (const [m, variant, backend, batches] of plan) {
    try {
      await bench(m, variant, backend, batches, runs);
    } catch (e) {
      const error = String(e?.message ?? e);
      failures.push({ network: m.name, format: variant, backend, error });
      log(`${m.name} ${variant} ${backend}: FAILED ${error}`);
      report();
    }
  }
  log('done');
  await report(true);
  log('results sent to the computer');
  for (const b of BUTTONS) $(b).disabled = false;
  $('copy').disabled = false;
  document.body.dataset.done = JSON.stringify({ env, results, failures });
}

const BUTTONS = ['quick', 'full', 'gpu'];
$('quick').onclick = () => sweep(false);
$('full').onclick = () => sweep(true);
$('gpu').onclick = () => sweep(false, true);
$('copy').onclick = async () => {
  const text = JSON.stringify({ env, results, failures }, null, 1);
  try { await navigator.clipboard.writeText(text); log('results copied to the clipboard'); }
  catch { log(text); }
};

// ?auto=quick|full|gpu&runs=N starts a sweep on load (for automated checks).
const params = new URLSearchParams(location.search);
if (params.get('runs')) $('runs').value = params.get('runs');
if (params.get('auto')) sweep(params.get('auto') === 'full', params.get('auto') === 'gpu');
