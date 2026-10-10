/**
 * Expands block-quantized network weights (training/zero/quantize.py) back to
 * the fp16 tensors the ONNX graph expects.
 *
 * The export ships a small .onnx (graph, biases, norms) and a .bin holding,
 * for each conv and dense weight tensor, its fp16 block scales followed by its
 * weights as 8-bit signed integers or 4-bit nibbles (q + 8, the even-indexed
 * weight in the low half). Every run of `block` weights shares a scale, and
 * w = q * scale. The result is the graph's external-data file, the same bytes
 * as the export's .weights file: the product is exact in float32 here and in
 * numpy, and both round it to fp16 to nearest even.
 */

export interface QuantTensor {
  /** where its scales start in the .bin; its weights follow them */
  offset: number;
  /** where its fp16 values go in the external-data file, in bytes */
  dst: number;
  count: number;
  bits: 4 | 8;
}

export interface QuantInfo {
  block: number;
  /** the file name the graph's weights refer to (onnxruntime's externalData path) */
  external: string;
  externalBytes: number;
  tensors: QuantTensor[];
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** float32 -> IEEE half bits, rounding to nearest even (as numpy's astype(float16)). */
export function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e === 0xff - 127 + 15) return sign | 0x7c00 | (m ? 0x200 : 0);    // inf, nan
  if (e >= 0x1f) return sign | 0x7c00;                                  // overflow
  let h: number, shift: number;
  if (e <= 0) {                                                         // subnormal half
    if (e < -10) return sign;
    m |= 0x800000;
    shift = 14 - e;
    h = m >>> shift;
  } else {
    shift = 13;
    h = (e << 10) | (m >>> 13);
  }
  const rest = m & ((1 << shift) - 1), half = 1 << (shift - 1);
  if (rest > half || (rest === half && (h & 1))) h++;                   // a carry into the exponent is right
  return sign | h;
}

/** IEEE half bits -> number. */
export function fromHalf(h: number): number {
  const sign = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return sign * m * 2 ** -24;
  if (e === 0x1f) return m ? NaN : sign * Infinity;
  return sign * (1 + m / 1024) * 2 ** (e - 15);
}

/** The external-data file (little-endian fp16) from the .bin. */
export function dequantize(packed: Uint8Array, info: QuantInfo): Uint8Array {
  const out = new Uint8Array(info.externalBytes);
  const block = info.block;
  for (const t of info.tensors) {
    const nb = Math.ceil(t.count / block);
    const scales = new Float32Array(nb);
    for (let i = 0; i < nb; i++) scales[i] = fromHalf(packed[t.offset + 2 * i] | (packed[t.offset + 2 * i + 1] << 8));
    const data = t.offset + 2 * nb;
    for (let i = 0; i < t.count; i++) {
      let q: number;
      if (t.bits === 8) {
        q = (packed[data + i] << 24) >> 24;                             // signed byte
      } else {
        const byte = packed[data + (i >> 1)];
        q = ((i & 1 ? byte >> 4 : byte) & 0xf) - 8;
      }
      const h = toHalf(Math.fround(q * scales[(i / block) | 0]));
      out[t.dst + 2 * i] = h & 0xff;
      out[t.dst + 2 * i + 1] = h >> 8;
    }
  }
  return out;
}
