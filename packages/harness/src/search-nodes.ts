/**
 * Node counts and time for findBestMove with and without quiescence on a few
 * fixed positions.
 *
 *   npx tsx packages/harness/src/search-nodes.ts [--depths 1,2,3,4]
 *
 * Positions are reproducible: a seeded standard setup, then N plies of
 * depth-2 plain minimax self-play with Math.random seeded (draws resolve via
 * Math.random). Each search starts from a fresh clone and an empty TT.
 */
import { pathToFileURL } from 'node:url';
import type { GameState } from '@the-duke/engine';
import { applyMove, applySetupPlacement, cloneState, createInitialState, getStandardSetupTargets, resetInstanceCounter } from '@the-duke/engine';
import { findBestMove, ttClear } from '@the-duke/ai';
import { makeRng } from './runner';

export interface FixedPosition {
  label: string;
  state: GameState;
}

export function fixedPositions(seed: number, plies: number[]): FixedPosition[] {
  const savedRandom = Math.random;
  Math.random = makeRng(seed * 7 + 3);
  try {
    resetInstanceCounter(0);
    const rng = makeRng(seed);
    let state = createInitialState();
    while (state.status === 'setup') {
      const t = getStandardSetupTargets(state);
      state = applySetupPlacement(state, t[Math.floor(rng() * t.length)]);
    }
    const out: FixedPosition[] = [];
    for (let ply = 0; ply <= Math.max(...plies) && state.status === 'active'; ply++) {
      if (plies.includes(ply)) out.push({ label: `seed ${seed} ply ${ply}`, state: cloneState(state) });
      ttClear();
      const { move } = findBestMove(state, 2, Math.random, { quiescence: false });
      if (!move) break;
      state = applyMove(state, move);
    }
    return out;
  } finally {
    Math.random = savedRandom;
  }
}

function measure(state: GameState, depth: number, quiescence: boolean) {
  ttClear();
  const t0 = performance.now();
  const r = findBestMove(cloneState(state), depth, () => 0.5, { quiescence });
  return { ms: performance.now() - t0, nodes: r.nodesSearched, q: r.quiescenceNodes };
}

function main(argv: string[]): void {
  const i = argv.indexOf('--depths');
  const depths = i >= 0 ? argv[i + 1].split(',').map(Number) : [1, 2, 3, 4];
  const positions = [1, 2, 3].flatMap(seed => fixedPositions(seed, [0, 8, 16]));

  const header = ['position', 'depth', 'nodes (no q)', 'nodes (q)', 'of which q', 'ratio', 'ms (no q)', 'ms (q)'];
  const rows: string[][] = [];
  for (const { label, state } of positions) {
    for (const d of depths) {
      // Warm the JIT once so the first row isn't dominated by compilation.
      if (rows.length === 0) { measure(state, d, false); measure(state, d, true); }
      const off = measure(state, d, false);
      const on = measure(state, d, true);
      rows.push([
        label, String(d), String(off.nodes), String(on.nodes), String(on.q),
        (on.nodes / off.nodes).toFixed(2) + 'x', off.ms.toFixed(1), on.ms.toFixed(1),
      ]);
    }
  }
  const widths = header.map((h, c) => Math.max(h.length, ...rows.map(r => r[c].length)));
  const fmt = (r: string[]) => '| ' + r.map((x, c) => x.padEnd(widths[c])).join(' | ') + ' |';
  console.log(fmt(header));
  console.log('|' + widths.map(w => '-'.repeat(w + 2)).join('|') + '|');
  for (const r of rows) console.log(fmt(r));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2));
}
