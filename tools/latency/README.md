# Network latency test (phone / browser)

Measures how fast a device can run the Duke network in the browser
(onnxruntime-web, CPU via WebAssembly and GPU via WebGPU), to choose the
network size and search budget for an in-browser AI.

```bash
tools/latency/setup.sh                       # fetch onnxruntime-web, export networks to ONNX
uv run python tools/latency/serve.py         # HTTPS on port 8443; prints the URL to open
```

On the phone (same Wi-Fi): open the printed `https://<computer-ip>:8443/`,
accept the self-signed certificate warning once, and press **Quick sweep**.
The page sends its results, failures and log to the server as it goes; each
visit is saved as `results/<time>-<device>.json` (and the server prints a line
per report). **Copy results** is still there for a page served some other way.

Networks tested: az2 (96×6) and az3-net (128×10) from checkpoints, plus random
64×4 and 192×12 to show how latency scales with size. Each comes as fp32,
fp16 (half size) and int8 (dense layers 8-bit). The page checks every network's
outputs against JAX's on 8 real positions (`correct` column).

## Results

Raw JSON per device is in `results/`.

**2026-10-06, Android 10 phone, Chrome 154, 7 CPU threads.** CPU: full sweep
([json](results/2026-10-06-android10-chrome154.json)). Its WebGPU runs failed
with "dilations should be 2D" (fixed in the exporter), so the GPU numbers come
from a separate GPU-only sweep
([json](results/2026-10-06T22-14-06-android10-chrome154.json)).

| network | MFLOP/eval | ms/eval (CPU, any format) | sims in 3 s | fp16 MB |
|---|---|---|---|---|
| 64×4 (random) | 33 | 3.6–5 | ~700 | 9.6 |
| az2 96×6 | 84 | 9.4–10 | ~300 | 11.2 |
| az3-net 128×10 | 227 | 24–25 | ~120 | 15.4 |
| 192×12 (random) | 590 | 59–63 | ~49 | 26.0 |

- **Latency is proportional to FLOPs**, at about 9 GFLOP/s effective
  (ms ≈ MFLOP / 9). That is about 3.2× slower than one desktop core.
- **Batching doesn't help on the CPU.** ms/eval is the same at batch 1, 8 and 32,
  so the search can evaluate one position at a time.
- **int8 is no faster than fp32**: the convolutions dominate and only the
  MatMuls are quantized. Its only benefit is download size. fp16 is the best
  default (half the size of fp32, value error < 4e-4).
- **Download size is mostly the dense policy layer** (4.6M of az3-net's 7.7M
  parameters). A structured policy head would roughly halve the download.

**GPU (WebGPU, fp16), same phone:**

| network | batch 1 | batch 8 | batch 32 | batch 128 | sims in 3 s (batch 8 / 32) |
|---|---|---|---|---|---|
| 64×4 (random) | 22.5 ms | 2.9 ms/eval | 0.87 | 0.50 | ~1,000 / 3,400 |
| az2 96×6 | 21.4 ms | 3.0 ms/eval | 1.15 | 0.82 | ~1,000 / 2,600 |
| az3-net 128×10 | 27.0 ms | 3.6 ms/eval | 2.1 | 1.6 | ~840 / 1,400 |
| 192×12 (random) | 32.2 ms | 5.6 ms/eval | 4.1 | 3.4 | ~530 / 740 |

- **Every call to the GPU costs about 20–25 ms**, however small the batch.
  At batch 1 the GPU is no faster than the CPU, but batch 8 costs about the
  same as batch 1. Batched, the GPU is 7–15× faster than the CPU
  (az3-net: 24.5 ms/eval on the CPU vs 3.6 at batch 8 and 2.1 at batch 32).
- **So the in-browser search must evaluate positions in batches.** Gumbel search
  does this naturally: each round of sequential halving runs one simulation
  per candidate move, and those simulations are independent.
- fp16 beats fp32 on the GPU too (az3-net batch 32: 68 vs 84 ms).
- Models load in under a second (over the LAN).
