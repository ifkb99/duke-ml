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
