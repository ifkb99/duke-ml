import { createInitialState, applySetupPlacement, getSetupTargets } from '@the-duke/engine';
import { generateAllMoves } from '@the-duke/engine';
import { RandomAgent, runMatch, runGame, makeRng } from '../src/runner';
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

// --- Single logged game for record shape ---
const rec = await runGame(new RandomAgent(3), new RandomAgent(4), { gameId: 'demo', seed: 5 });
console.log(`\nDemo game: winner=${rec.winner} plies=${rec.plies} firstMoves=${rec.moves.slice(0, 4).join(' | ')}`);
