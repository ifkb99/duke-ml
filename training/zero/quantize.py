"""
Block-wise weight quantization, to shrink the web app's download.

Like GGML's Q8_0 / Q4_0: each run of BLOCK weights (in the ONNX tensor's flat
order) shares one fp16 scale, and each weight is stored as a signed integer
of `bits` bits, w ~= q * scale. The browser expands the weights back to fp16
when the page loads (packages/ai/src/zero/quant.ts), so the network runs
exactly as the fp16 export does, at the same speed; only the download shrinks:
8 bits + 16/32 for the scale = 8.5 bits per weight, or 4.5 at 4 bits, vs 16.

The browser must produce the same fp16 values bit for bit: the product q *
scale is exact in float32 on both sides, and both round it to fp16 to nearest
even.
"""

import numpy as np

BLOCK = 32


def quantize(w: np.ndarray, bits: int, block: int = BLOCK):
    """(q int8 [w.size], scales fp16 [ceil(w.size / block)], w as the browser
    will see it: float32 q * scale, rounded to fp16)."""
    qmax = 2 ** (bits - 1) - 1
    flat = np.asarray(w, np.float32).reshape(-1)
    nb = -(-flat.size // block)
    padded = np.zeros(nb * block, np.float32)
    padded[:flat.size] = flat
    rows = padded.reshape(nb, block)
    scales = (np.abs(rows).max(1) / qmax).astype(np.float16)
    s = scales.astype(np.float32)
    q = np.round(rows / np.where(s > 0, s, 1)[:, None])
    q = np.clip(q, -qmax, qmax).astype(np.int8).reshape(-1)[:flat.size]
    deq = (q.astype(np.float32) * np.repeat(s, block)[:flat.size]).astype(np.float16)
    return q, scales, deq.reshape(np.shape(w))


def pack(q: np.ndarray, bits: int) -> bytes:
    """8 bits: one signed byte each. 4 bits: q + 8 (1..15), two per byte, the
    even-indexed weight in the low nibble."""
    if bits == 8:
        return q.astype(np.int8).tobytes()
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    if len(u) % 2:
        u = np.append(u, np.uint8(8))
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()


def quantize_params(pure: dict, bits: int, other_bits: int = 8) -> dict:
    """A checkpoint's params (nnx pure dict) with every conv and dense kernel
    replaced by what the browser would compute from its quantized export, in
    float32: for measuring strength in JAX. The residual blocks' convolutions
    (90% of the weights) get `bits`; the stem and heads, which are few weights
    but sensitive (4 bits there moves the value ~7x more than in the blocks),
    get `other_bits`. Convolutions are quantized in the ONNX layout (OIHW), as
    the export does."""
    def walk(d, in_blocks):
        out = {}
        for k, v in d.items():
            b = bits if in_blocks else other_bits
            if isinstance(v, dict):
                out[k] = walk(v, in_blocks or k == 'blocks')
            elif k == 'kernel' and np.ndim(v) == 4:          # HWIO
                _, _, deq = quantize(np.transpose(np.asarray(v, np.float32), (3, 2, 0, 1)), b)
                out[k] = np.transpose(deq.astype(np.float32), (2, 3, 1, 0))
            elif k == 'kernel':
                out[k] = quantize(np.asarray(v, np.float32), b)[2].astype(np.float32)
            else:
                out[k] = v
        return out
    return walk(pure, False)
