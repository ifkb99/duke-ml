import type {DrawOutcome, GameMove, GameState, Player} from '@the-duke/engine';
import {enumerateDrawOutcomes, findDukePosition, generatePseudoLegalMoves, getDrawTargets, isPlayerInCheck, isSquareAttackedBy, makeMove, unmakeMove} from '@the-duke/engine';
import {evaluate, evaluatePlacement, tileValue} from './evaluate.js';
import {
  TT_EXACT, TT_LOWER, TT_UPPER,
  hashState, hashAfterMove, ttStore, ttProbe, ttProbeMove, ttClear,
  encodeMove, movesMatch,
} from './utils/zobrist.js';

export interface MinimaxResult {
  move: GameMove | null;
  score: number;
  /** All nodes visited: the full-width tree plus quiescence nodes. */
  nodesSearched: number;
  /** The subset of `nodesSearched` that lies beyond the horizon (quiescence). */
  quiescenceNodes: number;
}

export interface SearchOptions {
  /**
   * Extend leaves with a capture-only quiescence search instead of returning
   * the static evaluation at depth 0. Off by default, so existing callers
   * (web worker, selfplay, neural-eval, zero minimax-server) keep the old
   * search; whether to enable it is the repo owner's call. In seeded matches
   * (harness match CLI, minimax-q:<d> vs minimax-noq:<d>) it beat the plain
   * search at every depth tried, 1-6, scoring 0.80-0.94, at 1.1-4.7x the time
   * per move. Against the plain search one ply deeper it scored 0.79 (q2 vs 3)
   * and 0.71 (q3 vs 4), but only 0.55 [0.48, 0.62] at q5 vs 6, while taking
   * 1.2-1.6x its time: the gain fades at depth 5-6, which the web uses.
   */
  quiescence?: boolean;
  /** Maximum quiescence plies beyond the horizon (default 6). */
  quiescenceDepth?: number;
}

export const DEFAULT_QUIESCENCE_DEPTH = 6;

/** Per-search state threaded through the recursion. */
interface SearchContext {
  /** Full-width nodes, including the horizon nodes where quiescence starts. */
  nodes: number;
  /** Quiescence nodes below the horizon node. */
  qNodes: number;
  /** Quiescence ply cap; 0 disables quiescence. */
  qDepth: number;
}

function makeContext(options: SearchOptions): SearchContext {
  const qDepth = (options.quiescence ?? false) ? (options.quiescenceDepth ?? DEFAULT_QUIESCENCE_DEPTH) : 0;
  if (!Number.isInteger(qDepth) || qDepth < 0) {
    throw new Error(`quiescenceDepth must be a non-negative integer, got ${options.quiescenceDepth}`);
  }
  return {nodes: 0, qNodes: 0, qDepth};
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_DEPTH = 64;
const WIN_SCORE = 99_999;
const MAX_PLACEMENTS_PER_TILE = 4;
const ASPIRATION_WINDOW = 50;

/**
 * Score for a decided game, from P1's perspective. Adding the remaining depth
 * makes a win found sooner (more depth left) worth more than one found later,
 * so the search actually converts won positions instead of wandering.
 */
function terminalScore(p1Wins: boolean, depth: number): number {
  const s = WIN_SCORE + Math.max(depth, 0);
  return p1Wins ? s : -s;
}

// ---------------------------------------------------------------------------
// Killer move table — 2 slots per depth
// ---------------------------------------------------------------------------

const killerMoves: Int32Array = new Int32Array(MAX_DEPTH * 2);

function clearKillers(): void {
  killerMoves.fill(0);
}

function storeKiller(depth: number, encoded: number): void {
  const base = depth * 2;
  if (killerMoves[base] !== encoded) {
    killerMoves[base + 1] = killerMoves[base];
    killerMoves[base] = encoded;
  }
}

function isKiller(depth: number, encoded: number): boolean {
  const base = depth * 2;
  return killerMoves[base] === encoded || killerMoves[base + 1] === encoded;
}

// ---------------------------------------------------------------------------
// Inline legality check
// ---------------------------------------------------------------------------

function isLegalAfterMake(state: GameState, mover: Player): boolean {
  if (state.status !== 'active') return true;
  const opponent = mover === 'P1' ? 'P2' : 'P1';
  const dukePos = findDukePosition(state, mover);
  return dukePos ? !isSquareAttackedBy(state, dukePos, opponent) : false;
}

// ---------------------------------------------------------------------------
// Move ordering — PV move first, then captures/strikes, then killers, then quiet
// ---------------------------------------------------------------------------

function isCapture(move: GameMove, state: GameState): boolean {
  if (move.type === 'strike') return true;
  if (move.type === 'move') return state.board[move.to.row][move.to.col] !== null;
  if (move.type === 'command') return state.board[move.targetTo.row][move.targetTo.col] !== null;
  return false;
}

function moveScore(move: GameMove, state: GameState, pvEncoded: number, depth: number): number {
  const encoded = encodeMove(move);

  // PV move from TT gets highest priority
  if (pvEncoded !== 0 && encoded === pvEncoded) return 100_000;

  if (move.type === 'strike') {
    const targetId = state.board[move.target.row][move.target.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 50_000;
    }
    return 10_000;
  }
  if (move.type === 'move') {
    const targetId = state.board[move.to.row][move.to.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 50_000;
      return 10_000;
    }
    if (isKiller(depth, encoded)) return 5_000;
    return 0;
  }
  if (move.type === 'command') {
    const targetId = state.board[move.targetTo.row][move.targetTo.col];
    if (targetId) {
      const target = state.tiles.get(targetId);
      if (target?.defName === 'Duke') return 50_000;
      return 10_000;
    }
    if (isKiller(depth, encoded)) return 5_000;
    return 0;
  }
  // Place moves — lowest priority
  return -1;
}

function orderMoves(moves: GameMove[], state: GameState, pvEncoded: number, depth: number): GameMove[] {
  return moves.sort((a, b) => moveScore(b, state, pvEncoded, depth) - moveScore(a, state, pvEncoded, depth));
}

// ---------------------------------------------------------------------------
// Expectimax: split moves into board moves and draw outcomes
// ---------------------------------------------------------------------------

function splitMoves(
  moves: GameMove[],
  state: GameState,
): {boardMoves: GameMove[]; drawOutcomes: DrawOutcome[]} {
  // Draw moves are chance decisions. Expand them into weighted per-tile
  // outcomes via the engine, which owns the bag's probability structure — this
  // keeps the expectimax model (E over drawn tile of best placement) in one
  // place and prevents any caller from treating a specific tile as choosable.
  const boardMoves = moves.filter(m => m.type !== 'draw' && m.type !== 'place');
  const drawOutcomes = enumerateDrawOutcomes(state);
  return {boardMoves, drawOutcomes};
}

// ---------------------------------------------------------------------------
// Apply + search helper with incremental hashing
// ---------------------------------------------------------------------------

function applyAndSearch(
  state: GameState,
  move: GameMove,
  mover: Player,
  depth: number,
  alpha: number,
  beta: number,
  childMaximizing: boolean,
  hash: number,
  ctx: SearchContext,
  allowNull: boolean,
): number | null {
  const undo = makeMove(state, move);
  if (!isLegalAfterMake(state, mover)) {
    unmakeMove(state, undo);
    return null;
  }
  const newHash = hashAfterMove(hash, state, undo);
  const score = minimax(state, depth, alpha, beta, childMaximizing, newHash, ctx, allowNull);
  unmakeMove(state, undo);
  return score;
}

// ---------------------------------------------------------------------------
// Evaluate chance node with static placement pruning
// Only fully searches top-N placements per drawn tile (sorted by static eval).
//
// Each outcome is searched with a full window: the node's value is an
// expectation, and averaging alpha-beta *bounds* (from cutoffs) instead of
// exact scores gives a meaningless number.
// ---------------------------------------------------------------------------

function evaluateChanceNode(
  drawOutcomes: DrawOutcome[],
  state: GameState,
  mover: Player,
  depth: number,
  maximizing: boolean,
  hash: number,
  ctx: SearchContext,
): number {
  let expectedValue = 0;

  for (const {probability, placements} of drawOutcomes) {
    let bestPlacementScore = maximizing ? -Infinity : Infinity;

    // Static pruning: sort placements by tile value, only search top N
    const sorted = placements.length > MAX_PLACEMENTS_PER_TILE
      ? placements
          .map(m => ({move: m, val: m.type === 'place' ? evaluatePlacement(state, m.tileName) : 0}))
          .sort((a, b) => b.val - a.val)
          .slice(0, MAX_PLACEMENTS_PER_TILE)
          .map(x => x.move)
      : placements;

    for (const move of sorted) {
      const score = applyAndSearch(state, move, mover, depth - 1, -Infinity, Infinity, !maximizing, hash, ctx, true);
      if (score === null) continue;
      bestPlacementScore = maximizing
        ? Math.max(bestPlacementScore, score)
        : Math.min(bestPlacementScore, score);
    }

    expectedValue += probability * bestPlacementScore;
  }

  return expectedValue;
}

// ---------------------------------------------------------------------------
// Quiescence search
//
// At the horizon, keep searching "noisy" moves until the position is quiet,
// so a leaf with a capture pending isn't scored as if nothing were hanging.
//
// Noisy = any board move that removes an enemy tile: a move/jump/slide onto an
// enemy, a strike, or a command whose destination holds an enemy (the
// commanded tile captures exactly like a move does, so the evaluation swings
// by the same material). Everything else is quiet: non-capturing moves and
// commands only reposition or flip tiles, and draws are chance nodes whose
// value is an expectation over the bag; expanding them here would bring back
// the full-width chance branching that quiescence exists to avoid.
//
// If the side to move is in check it may not stand pat (its Duke would be
// taken next move), so every legal board move is searched as an evasion. If
// there is none but a draw could block the check we fall back to the static
// score rather than expand the chance node; with no draw either, it's mate.
//
// Quiescence stores TT entries at depth 0: the full-width search (depth >= 1)
// never accepts them, and the replacement policy never lets them overwrite a
// deeper entry. Quiescence in turn accepts any full-width entry. Killers are
// left alone (they are indexed by full-width depth and only hold quiet moves).
// ---------------------------------------------------------------------------

/** Ordering for quiescence: TT move, Duke captures, then MVV-LVA, then quiet evasions. */
function quiescenceOrder(move: GameMove, state: GameState, pvEncoded: number): number {
  if (pvEncoded !== 0 && encodeMove(move) === pvEncoded) return 1e9;
  let victimSq, attackerSq;
  switch (move.type) {
    case 'move': victimSq = move.to; attackerSq = move.from; break;
    case 'strike': victimSq = move.target; attackerSq = move.from; break;
    case 'command': victimSq = move.targetTo; attackerSq = move.target; break;
    default: return -1;
  }
  const victimId = state.board[victimSq.row][victimSq.col];
  if (!victimId) return 0;
  const victim = state.tiles.get(victimId)!;
  if (victim.defName === 'Duke') return 1e8;
  const attacker = state.tiles.get(state.board[attackerSq.row][attackerSq.col]!)!;
  return 1000 + 16 * tileValue(victim.defName) - tileValue(attacker.defName);
}

function quiesce(
  state: GameState,
  qply: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  hash: number,
  ctx: SearchContext,
): number {
  // The horizon node itself (qply 0) was already counted by minimax().
  if (qply > 0) ctx.qNodes++;

  if (state.status !== 'active') return terminalScore(state.status === 'P1_wins', 0);
  if (qply >= ctx.qDepth) return evaluate(state);

  const ttHit = ttProbe(hash, 0, alpha, beta);
  if (ttHit !== null) return ttHit;

  const mover = state.currentPlayer;
  const inCheck = isPlayerInCheck(state, mover);
  const standPat = evaluate(state);
  const origAlpha = alpha;
  const origBeta = beta;
  let best = maximizing ? -Infinity : Infinity;

  if (!inCheck) {
    // Stand pat: the side to move can decline every capture.
    if (maximizing ? standPat >= beta : standPat <= alpha) return standPat;
    best = standPat;
    if (maximizing) alpha = Math.max(alpha, standPat);
    else beta = Math.min(beta, standPat);
  }

  const pvEncoded = ttProbeMove(hash);
  const candidates = generatePseudoLegalMoves(state)
    .filter(m => (inCheck ? m.type !== 'draw' && m.type !== 'place' : isCapture(m, state)))
    .map(m => ({m, order: quiescenceOrder(m, state, pvEncoded)}))
    .sort((a, b) => b.order - a.order);

  let bestEncoded = 0;
  let hasLegalMove = false;

  for (const {m: move} of candidates) {
    const undo = makeMove(state, move);
    if (!isLegalAfterMake(state, mover)) {
      unmakeMove(state, undo);
      continue;
    }
    const score = quiesce(state, qply + 1, alpha, beta, !maximizing, hashAfterMove(hash, state, undo), ctx);
    unmakeMove(state, undo);
    hasLegalMove = true;

    if (maximizing ? score > best : score < best) {
      best = score;
      bestEncoded = encodeMove(move);
    }
    if (maximizing) alpha = Math.max(alpha, best);
    else beta = Math.min(beta, best);
    if (alpha >= beta) break;
  }

  if (inCheck && !hasLegalMove) {
    return getDrawTargets(state).length > 0 ? standPat : terminalScore(!maximizing, 0);
  }

  const flag = best <= origAlpha ? TT_UPPER : best >= origBeta ? TT_LOWER : TT_EXACT;
  ttStore(hash, 0, best, flag, bestEncoded);
  return best;
}

// ---------------------------------------------------------------------------
// Minimax with alpha-beta + TT + null-move pruning + LMR + expectimax draws
// ---------------------------------------------------------------------------

function minimax(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  hash: number,
  ctx: SearchContext,
  allowNull: boolean,
): number {
  ctx.nodes++;

  if (state.status !== 'active') return terminalScore(state.status === 'P1_wins', depth);
  if (depth <= 0) {
    return ctx.qDepth > 0 ? quiesce(state, 0, alpha, beta, maximizing, hash, ctx) : evaluate(state);
  }

  // Transposition table probe
  const ttHit = ttProbe(hash, depth, alpha, beta);
  if (ttHit !== null) return ttHit;

  const mover = state.currentPlayer;
  const opponent = mover === 'P1' ? 'P2' : 'P1';
  const inCheck = isPlayerInCheck(state, mover);

  // --- Null-move pruning ---
  // Skip if in check, if null-move is disallowed (already in null-move), or at shallow depth
  if (allowNull && !inCheck && depth >= 3) {
    const R = 3;
    state.currentPlayer = opponent;
    const nullHash = hash ^ 0; // turn key XOR handled by passing !maximizing
    // Null move = opponent gets an extra turn. Flip hash turn key manually.
    const nullScore = minimax(state, depth - 1 - R, alpha, beta, !maximizing, nullHash ^ hashNullMoveDelta(), ctx, false);
    state.currentPlayer = mover;

    if (maximizing ? nullScore >= beta : nullScore <= alpha) {
      return nullScore;
    }
  }

  const allMoves = generatePseudoLegalMoves(state);

  if (allMoves.length === 0) return terminalScore(!maximizing, depth);

  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);
  const pvEncoded = ttProbeMove(hash);
  const orderedBoardMoves = orderMoves(boardMoves, state, pvEncoded, depth);

  let bestScore = maximizing ? -Infinity : Infinity;
  let bestEncoded = 0;
  let flag: number;
  let hasLegalMove = false;
  const origAlpha = alpha;
  const origBeta = beta;

  for (let i = 0; i < orderedBoardMoves.length; i++) {
    const move = orderedBoardMoves[i];
    const isCap = isCapture(move, state);

    // --- Late Move Reductions ---
    let searchDepth = depth - 1;
    const doLMR = i >= 3 && depth >= 3 && !isCap && !inCheck;
    if (doLMR) {
      searchDepth = depth - 2; // reduced search
    }

    let score = applyAndSearch(state, move, mover, searchDepth, alpha, beta, !maximizing, hash, ctx, true);
    if (score === null) continue;

    // Re-search at full depth if LMR found a promising score
    if (doLMR) {
      const promising = maximizing ? score > alpha : score < beta;
      if (promising) {
        score = applyAndSearch(state, move, mover, depth - 1, alpha, beta, !maximizing, hash, ctx, true)!;
      }
    }

    hasLegalMove = true;
    const encoded = encodeMove(move);

    if (maximizing) {
      if (score > bestScore) {
        bestScore = score;
        bestEncoded = encoded;
      }
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) {
        if (!isCap) storeKiller(depth, encoded);
        break;
      }
    } else {
      if (score < bestScore) {
        bestScore = score;
        bestEncoded = encoded;
      }
      beta = Math.min(beta, bestScore);
      if (alpha >= beta) {
        if (!isCap) storeKiller(depth, encoded);
        break;
      }
    }
  }

  // Chance nodes — search 3 plies shallower than board moves
  if (drawOutcomes.length > 0 && alpha < beta) {
    const drawDepth = Math.max(depth - 3, 0);
    const drawScore = evaluateChanceNode(drawOutcomes, state, mover, drawDepth, maximizing, hash, ctx);
    hasLegalMove = true;
    if (maximizing) {
      bestScore = Math.max(bestScore, drawScore);
      alpha = Math.max(alpha, bestScore);
    } else {
      bestScore = Math.min(bestScore, drawScore);
      beta = Math.min(beta, bestScore);
    }
  }

  if (!hasLegalMove) return terminalScore(!maximizing, depth);

  // Determine TT flag from original bounds
  if (bestScore <= origAlpha) {
    flag = TT_UPPER;
  } else if (bestScore >= origBeta) {
    flag = TT_LOWER;
  } else {
    flag = TT_EXACT;
  }

  ttStore(hash, depth, bestScore, flag, bestEncoded);
  return bestScore;
}

// Null-move hash delta: just the turn key. Reuse zobrist internals.
// We import hashState but need the turn key separately. We derive it by
// hashing an empty-ish state toggle. Instead, hardcode the known turn key
// from the PRNG sequence. For correctness, we compute it at init.
let _nullMoveTurnDelta = 0;
{
  // The turn key is deterministic from the PRNG. Recompute it here.
  let seed = 0xDEAD_BEEF;
  function rand32(): number {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  }
  const numTiles = 15; // ALL_TILES.length
  // Skip board keys: 6 * 6 * numTiles * 4 = 2160 rand32 calls
  for (let i = 0; i < 6 * 6 * numTiles * 4; i++) rand32();
  _nullMoveTurnDelta = rand32(); // the turnKey
}

function hashNullMoveDelta(): number {
  return _nullMoveTurnDelta;
}

// ---------------------------------------------------------------------------
// Public API: findBestMove with iterative deepening
// ---------------------------------------------------------------------------

/**
 * Find the best move using iterative deepening minimax with alpha-beta pruning.
 * Evaluates from P1's perspective: P1 maximizes, P2 minimizes.
 * Draw moves are treated as chance nodes (expectimax).
 *
 * `rng` (uniform in [0, 1)) samples the tile when a draw is chosen; pass a
 * seeded one for reproducible results. Call `ttClear()` between searches you
 * want to compare, since the transposition table persists across calls (and
 * always between searches with different `options`: scores stored with and
 * without quiescence are not interchangeable).
 */
export function findBestMove(
  state: GameState,
  maxDepth = 6,
  rng: () => number = Math.random,
  options: SearchOptions = {},
): MinimaxResult {
  const allMoves = generatePseudoLegalMoves(state);
  if (allMoves.length === 0) return {move: null, score: 0, nodesSearched: 0, quiescenceNodes: 0};

  const maximizing = state.currentPlayer === 'P1';
  const mover = state.currentPlayer;
  const ctx = makeContext(options);
  const rootHash = hashState(state);

  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);

  let bestMove: GameMove | null = null;
  let bestScore = maximizing ? -Infinity : Infinity;

  // One full root search at depth `d` within the window (alpha, beta).
  const searchRoot = (d: number, alpha: number, beta: number) => {
    clearKillers();

    let iterBestMove: GameMove | null = null;
    let iterBestScore = maximizing ? -Infinity : Infinity;

    const pvEncoded = ttProbeMove(rootHash);
    const orderedBoardMoves = orderMoves(boardMoves, state, pvEncoded, d);

    for (const move of orderedBoardMoves) {
      const score = applyAndSearch(state, move, mover, d - 1, alpha, beta, !maximizing, rootHash, ctx, true);
      if (score === null) continue;

      if (iterBestMove === null) iterBestMove = move;

      if (maximizing) {
        if (score > iterBestScore) {
          iterBestScore = score;
          iterBestMove = move;
        }
        alpha = Math.max(alpha, iterBestScore);
      } else {
        if (score < iterBestScore) {
          iterBestScore = score;
          iterBestMove = move;
        }
        beta = Math.min(beta, iterBestScore);
      }
      // Failed outside an aspiration window: the caller re-searches with a
      // full window. Searching on would hand children an inverted window
      // (alpha > beta), and the bound flags they store in the TT would be
      // wrong, poisoning that re-search. Only possible with a window.
      if (alpha >= beta) break;
    }

    // Evaluate draw outcomes
    if (drawOutcomes.length > 0) {
      const drawScore = evaluateChanceNode(drawOutcomes, state, mover, d, maximizing, rootHash, ctx);
      const drawIsBetter = maximizing ? drawScore > iterBestScore : drawScore < iterBestScore;

      if (drawIsBetter) {
        iterBestScore = drawScore;
        // Draw for real (uniformly from the bag), then choose the best square
        // for the revealed tile — the same order as the physical game.
        const bag = state.bags[state.currentPlayer];
        const randomTile = bag[Math.floor(rng() * bag.length)];
        const tilePlacements = drawOutcomes.find(o => o.tileName === randomTile)?.placements ?? [];

        let bestPlacement: GameMove | null = null;
        let bestPlacementScore = maximizing ? -Infinity : Infinity;

        for (const move of tilePlacements) {
          const score = applyAndSearch(state, move, mover, d - 1, -Infinity, Infinity, !maximizing, rootHash, ctx, true);
          if (score === null) continue;
          if (maximizing ? score > bestPlacementScore : score < bestPlacementScore) {
            bestPlacementScore = score;
            bestPlacement = move;
          }
        }

        if (bestPlacement) iterBestMove = bestPlacement;
      }
    }

    return {move: iterBestMove, score: iterBestScore};
  };

  // --- Iterative deepening ---
  for (let d = 1; d <= maxDepth; d++) {
    // Aspiration window around the previous iteration's score (depth 3+).
    // If the true score lands outside it, root moves return only bounds and
    // the "best" move is meaningless — so re-search with a full window.
    const useWindow = d >= 3 && bestMove !== null && Number.isFinite(bestScore);
    const lo = useWindow ? bestScore - ASPIRATION_WINDOW : -Infinity;
    const hi = useWindow ? bestScore + ASPIRATION_WINDOW : Infinity;

    let iter = searchRoot(d, lo, hi);
    if (useWindow && (iter.score <= lo || iter.score >= hi)) {
      iter = searchRoot(d, -Infinity, Infinity);
    }

    if (iter.move !== null) {
      bestMove = iter.move;
      bestScore = iter.score;
    }
  }

  return {move: bestMove, score: bestScore, nodesSearched: ctx.nodes + ctx.qNodes, quiescenceNodes: ctx.qNodes};
}

export interface ScoredMoves {
  /** Every legal board move with its exact score at `depth` (P1's perspective). */
  moves: {move: GameMove; score: number}[];
  /** Expected score of drawing a tile (P1's perspective), or null if drawing isn't possible. */
  draw: number | null;
  nodesSearched: number;
  quiescenceNodes: number;
}

/**
 * Exact minimax score of EVERY root move, not just the best one, for
 * benchmarking how much worse another player's choice is. Each root move gets
 * a full-window search (no root alpha-beta cutoffs), after a normal search one
 * ply shallower has warmed the transposition table for move ordering.
 */
export function scoreRootMoves(
  state: GameState,
  depth = 6,
  rng: () => number = Math.random,
  options: SearchOptions = {},
): ScoredMoves {
  const allMoves = generatePseudoLegalMoves(state);
  if (allMoves.length === 0) return {moves: [], draw: null, nodesSearched: 0, quiescenceNodes: 0};
  if (depth > 1) findBestMove(state, depth - 1, rng, options);

  const maximizing = state.currentPlayer === 'P1';
  const mover = state.currentPlayer;
  const ctx = makeContext(options);
  const rootHash = hashState(state);
  const {boardMoves, drawOutcomes} = splitMoves(allMoves, state);

  const moves: {move: GameMove; score: number}[] = [];
  for (const move of boardMoves) {
    clearKillers();
    const score = applyAndSearch(state, move, mover, depth - 1, -Infinity, Infinity, !maximizing, rootHash, ctx, true);
    if (score !== null) moves.push({move, score});
  }
  const draw = drawOutcomes.length > 0
    ? evaluateChanceNode(drawOutcomes, state, mover, depth, maximizing, rootHash, ctx)
    : null;
  return {moves, draw, nodesSearched: ctx.nodes + ctx.qNodes, quiescenceNodes: ctx.qNodes};
}

export { ttClear };
