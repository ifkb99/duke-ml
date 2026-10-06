/**
 * The network player: engine state in, engine move out.
 *
 * A draw is played honestly, as a person would: the search decides to draw
 * without knowing the tile, a tile is pulled from the bag, then a second
 * search places it. The result is one resolved engine `place` move.
 */

import type {GameMove, GameState, SerializedGameState} from '@the-duke/engine';
import {actionToMove, fromGameState} from './bridge.js';
import {chanceProbs, stepChance, stepDecision, type State} from './env.js';
import {DRAW_ACTION, N_ACTIONS} from './rules.js';
import {type Evaluate, search, type SearchResult} from './search.js';

export interface ZeroPlayerOptions {
  simulations: number;
  maxConsidered?: number;
  /** Root noise: 0 plays the search's best move; ~1 plays like self-play (weaker, varied). */
  noise?: number;
  maxBatch?: number;
  random?: () => number;
}

export interface ZeroMoveInfo {
  value: number;              // search value for the mover, in [-1, 1]
  rootValue: number;          // the network's own value
  evaluations: number;
  batches: number;
  ms: number;
  drewTile?: string;
}

function gumbel(n: number, scale: number, random: () => number): Float64Array {
  return Float64Array.from({length: n}, () => -Math.log(-Math.log(Math.max(random(), 1e-300))) * scale);
}

function onlyLegal(s: State): number {
  let found = -1;
  for (let a = 0; a < N_ACTIONS; a++) {
    if (!s.legal[a]) continue;
    if (found >= 0) return -1;
    found = a;
  }
  return found;
}

async function decide(s: State, evaluate: Evaluate, opts: ZeroPlayerOptions,
                      info: ZeroMoveInfo): Promise<number> {
  const forced = onlyLegal(s);
  if (forced >= 0) return forced;
  const random = opts.random ?? Math.random;
  const r: SearchResult = await search(s, evaluate, {
    simulations: opts.simulations, maxConsidered: opts.maxConsidered, maxBatch: opts.maxBatch,
    gumbel: opts.noise ? gumbel(N_ACTIONS, opts.noise, random) : undefined,
  });
  info.value = r.searchValue;
  info.rootValue = r.rootValue;
  info.evaluations += r.evaluations;
  info.batches += r.batches;
  return r.action;
}

export async function chooseMove(game: GameState | SerializedGameState, evaluate: Evaluate,
                                 opts: ZeroPlayerOptions): Promise<{move: GameMove; info: ZeroMoveInfo}> {
  const t0 = performance.now();
  const info: ZeroMoveInfo = {value: 0, rootValue: 0, evaluations: 0, batches: 0, ms: 0};
  let s = fromGameState(game);
  let action = await decide(s, evaluate, opts, info);
  if (action === DRAW_ACTION) {
    // Pull a tile, as the engine would, then decide where it goes.
    const probs = chanceProbs(stepDecision(s, DRAW_ACTION));
    let u = (opts.random ?? Math.random)(), tile = 0;
    while (tile < probs.length - 1 && (u -= probs[tile]) >= 0) tile++;
    s = stepChance(stepDecision(s, DRAW_ACTION), tile);
    action = await decide(s, evaluate, opts, info);
  }
  const move = actionToMove(action, s);
  if (move.type === 'place') info.drewTile = move.tileName;
  info.ms = performance.now() - t0;
  return {move, info};
}
