import { describe, it, expect } from 'vitest';
import { dequantize, fromHalf, toHalf } from '../src/zero/quant.js';

describe('fp16 conversion', () => {
  it('round-trips every finite half', () => {
    for (let h = 0; h < 0x10000; h++) {
      if ((h & 0x7c00) === 0x7c00) continue;          // inf, nan
      expect(toHalf(fromHalf(h))).toBe(h === 0x8000 ? 0x8000 : h);
    }
  });

  it('rounds halfway values to even, and just past them up', () => {
    for (const h of [0x0001, 0x0002, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bfe]) {
      const lo = fromHalf(h), hi = fromHalf(h + 1), mid = (lo + hi) / 2;
      expect(toHalf(mid)).toBe(h & 1 ? h + 1 : h);
      expect(toHalf(mid + (hi - lo) / 64)).toBe(h + 1);
      expect(toHalf(-mid)).toBe(0x8000 | (h & 1 ? h + 1 : h));
    }
  });

  it('overflows to infinity and underflows to zero', () => {
    expect(toHalf(65520)).toBe(0x7c00);
    expect(toHalf(65519)).toBe(0x7bff);
    expect(toHalf(2 ** -26)).toBe(0);
    expect(toHalf(Infinity)).toBe(0x7c00);
  });
});

describe('dequantize', () => {
  it('expands 8- and 4-bit blocks', () => {
    // two tensors, block 2: 8-bit [1, -2, 3] with scales 0.5, 2; 4-bit [-7, 7, 0] with scales 1, 0.25
    const packed = new Uint8Array([
      ...[0x00, 0x38, 0x00, 0x40], 1, 0xfe, 3,           // fp16 0.5, 2; then int8
      ...[0x00, 0x3c, 0x00, 0x34], (1 | (15 << 4)), 8,   // fp16 1, 0.25; then nibbles q + 8
    ]);
    const out = dequantize(packed, {block: 2, external: 'w', externalBytes: 12, tensors: [
      {offset: 0, dst: 0, count: 3, bits: 8}, {offset: 7, dst: 6, count: 3, bits: 4}]});
    const values = Array.from({length: 6}, (_, i) => fromHalf(out[2 * i] | (out[2 * i + 1] << 8)));
    expect(values).toEqual([0.5, -1, 6, -7, 7, 0]);
  });
});
