# Network latency test (phone / browser)

Measures how fast a device can run the Duke network in the browser
(onnxruntime-web, CPU via WebAssembly and GPU via WebGPU), to choose the
network size and search budget for an in-browser AI.

```bash
tools/latency/setup.sh                       # fetch onnxruntime-web, export networks to ONNX
uv run python tools/latency/serve.py         # HTTPS on port 8443; prints the URL to open
```

On the phone (same Wi-Fi): open the printed `https://<computer-ip>:8443/`,
accept the self-signed certificate warning once, press **Quick sweep**, then
**Copy results** and paste them back.

Networks tested: az2 (96×6) and az3-net (128×10) from checkpoints, plus random
64×4 and 192×12 to show how latency scales with size. Each comes as fp32,
fp16 (half size) and int8 (dense layers 8-bit). The page checks every network's
outputs against JAX's on 8 real positions (`correct` column).

## Results

Raw JSON per device is in `results/`.

**2026-10-06, Android 10 phone, Chrome 154, 7 CPU threads (full sweep)**
([json](results/2026-10-06-android10-chrome154.json)). The WebGPU rows all
failed, and the error wasn't recorded then. The page now saves failures in the
copied JSON, and the **GPU only** button reruns just that part.

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
