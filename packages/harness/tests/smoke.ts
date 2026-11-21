import { createInitialState, applySetupPlacement, getSetupTargets } from '@the-duke/engine';
import { generateAllMoves } from '@the-duke/engine';
import assert from 'node:assert/strict';
import { RandomAgent, runMatch, runGame, makeRng } from '../src/runner';
import { MinimaxAgent } from '../src/minimax-agent';
import { parseAgentSpec, parseArgs, playMatch, formatReport, UsageError } from '../src/match';
import { moveToString, parseMove, movesEqual } from '../src/notation';
import { renderAscii, renderCoordList, renderAbilities, renderBag } from '../src/render';

// --- Notation round-trip on a real position ---
let s = createInitialState();
const rng = makeRng(99);
while (s.status === 'setup') {
  const t = getSetupTargets(s);
  s = applySetupPlacement(s, t[Math.floor(rng() * t.length)]);
}
const legal = generateAllMoves(s);
let rtFail = 0;
for (const m of legal) {
  const rt = parseMove(moveToString(m));
  if (!rt || !movesEqual(m, rt)) rtFail++;
}
console.log(`Notation round-trip: ${legal.length - rtFail}/${legal.length} ok`);

// --- Renderer sanity ---
console.log('\n--- ASCII render ---\n' + renderAscii(s));
console.log('\n--- Coord list ---\n' + renderCoordList(s));
console.log('\n--- Bag ---\n' + renderBag(s));
console.log('\n--- Abilities (first 12 lines) ---');
console.log(renderAbilities(s, s.currentPlayer).split('\n').slice(0, 12).join('\n'));

// --- Random vs random match ---
const stats = await runMatch(
  () => new RandomAgent(1, 'rand-A'),
  () => new RandomAgent(2, 'rand-B'),
  200,
  { label: 'smoke' },
);
console.log('\n--- Match: 200 games random vs random ---');
console.log(JSON.stringify(stats, null, 2));

// --- Minimax agent + match CLI plumbing ---
assert.equal(new MinimaxAgent(3).name, 'minimax-d3');
assert.deepEqual(parseAgentSpec('minimax:2'), { kind: 'minimax', depth: 2 });
assert.deepEqual(parseAgentSpec('random'), { kind: 'random' });
assert.deepEqual(parseAgentSpec('minimax-q:3'), { kind: 'minimax', depth: 3, quiescence: true });
assert.deepEqual(parseAgentSpec('minimax-noq:3'), { kind: 'minimax', depth: 3, quiescence: false });
assert.equal(new MinimaxAgent(2, { search: { quiescence: false } }).name, 'minimax-noq-d2');
assert.equal(new MinimaxAgent(2, { search: { quiescence: true } }).name, 'minimax-q-d2');
assert.equal(new MinimaxAgent(2, 'custom').name, 'custom');
for (const bad of ['minimax', 'minimax:0', 'minimax:x', 'greedy', 'minimax-q:0', 'minimax-qq:2', 'minimax-noq']) {
  assert.throws(() => parseAgentSpec(bad), UsageError, `spec '${bad}' should be rejected`);
}
assert.throws(() => parseArgs(['--a', 'random', '--b', 'random', '--bogus', '1']), UsageError);
assert.throws(() => parseArgs(['--a', 'random']), UsageError);
assert.deepEqual(
  parseArgs(['--a', 'minimax:3', '--b', 'random', '--games', '20', '--seed', '1']),
  { a: { kind: 'minimax', depth: 3 }, b: { kind: 'random' }, games: 20, seed: 1 },
);

const mmArgs = parseArgs(['--a', 'minimax:2', '--b', 'random', '--games', '4', '--seed', '1']);
assert.ok(mmArgs !== 'help');
const mmReport = await playMatch(mmArgs);
console.log('\n--- Match: 4 games minimax-d2 vs random (seed 1) ---');
console.log(formatReport(mmReport));
assert.ok(
  mmReport.stats.aWins > mmReport.stats.games / 2,
  `minimax-d2 should win the majority vs random, got ${mmReport.stats.aWins}/${mmReport.stats.games}`,
);
const mmReplay = await playMatch(mmArgs);
assert.deepEqual(mmReplay.stats, mmReport.stats, 'seeded match should be reproducible');
assert.ok(mmReport.nodes.a.searches > 0, 'minimax agent reports its searches');
assert.equal(mmReport.nodes.a.quiescenceNodes, 0, 'plain minimax:<d> follows the AI default (quiescence off)');
assert.equal(mmReport.nodes.b.searches, 0, 'random agent runs no searches');

// Quiescence on vs off in one seeded match.
const qArgs = parseArgs(['--a', 'minimax-q:1', '--b', 'minimax-noq:1', '--games', '4', '--seed', '1']);
assert.ok(qArgs !== 'help');
const qReport = await playMatch(qArgs);
console.log('\n--- Match: 4 games minimax-q-d1 vs minimax-noq-d1 (seed 1) ---');
console.log(formatReport(qReport));
assert.equal(qReport.stats.a, 'minimax-q-d1');
assert.ok(qReport.nodes.a.quiescenceNodes > 0, 'minimax-q must run quiescence');
assert.equal(qReport.nodes.b.quiescenceNodes, 0, 'minimax-noq must not run quiescence');
assert.deepEqual((await playMatch(qArgs)).stats, qReport.stats, 'q vs noq match should be reproducible');

// --- Single logged game for record shape ---
const rec = await runGame(new RandomAgent(3), new RandomAgent(4), { gameId: 'demo', seed: 5 });
console.log(`\nDemo game: winner=${rec.winner} plies=${rec.plies} firstMoves=${rec.moves.slice(0, 4).join(' | ')}`);
