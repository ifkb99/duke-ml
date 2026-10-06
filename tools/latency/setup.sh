#!/usr/bin/env bash
# Fetch onnxruntime-web and export the networks for the latency test page.
#   tools/latency/setup.sh            # then: uv run python tools/latency/serve.py
set -eu
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
ORT_VERSION=${ORT_VERSION:-1.30.0}

if [ ! -f vendor/ort.all.min.mjs ]; then
  tmp=$(mktemp -d)
  (cd "$tmp" && npm pack "onnxruntime-web@$ORT_VERSION" >/dev/null && tar xzf onnxruntime-web-*.tgz)
  mkdir -p vendor
  cp "$tmp"/package/dist/ort.all.min.mjs "$tmp"/package/dist/ort-wasm-simd-threaded*.{mjs,wasm} vendor/
  rm -rf "$tmp"
fi

export JAX_PLATFORMS=cpu      # export on the CPU; leaves the GPUs to training
cd "$ROOT"
PY=".venv/bin/python -m training.zero.export_onnx"
[ -f runs/az2/ckpt_001000.pkl ] && $PY runs/az2/ckpt_001000.pkl tools/latency/models/az2
[ -f runs/az3-net/latest.pkl ] && $PY runs/az3-net/latest.pkl tools/latency/models/az3net
$PY --random 64x4 tools/latency/models/r64x4
$PY --random 192x12 tools/latency/models/r192x12

.venv/bin/python - <<'PY'
import json
from pathlib import Path
d = Path('tools/latency/models')
names = {'r64x4': 'tiny 64×4 (random)', 'az2': 'az2 96×6', 'az3net': 'az3-net 128×10', 'r192x12': 'big 192×12 (random)'}
index = []
for key in names:
    p = d / f'{key}.json'
    if p.exists():
        index.append({'name': names[key], **json.loads(p.read_text())})
index.sort(key=lambda m: m['mflops_per_eval'])
(d / 'index.json').write_text(json.dumps(index))
print('models:', ', '.join(m['name'] for m in index))
PY
