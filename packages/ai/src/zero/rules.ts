/**
 * Static rule tables for the AlphaZero-style network, ported from
 * training/zero/rules.py. Same frame and numbering: squares are row * 6 + col
 * in the canonical frame of the player to move ("me": home row 5, forward is
 * toward row 0); owner 0 = me, 1 = opponent.
 *
 * Tables are flat typed arrays (1 = true); use the index helpers.
 *   MOVE_OK     [2, T, S, N, N]  owner, type, side, from, to: move/capture possible
 *   NEEDS_CLEAR [2, T, S, N, N]  ...and it needs the squares between to be empty
 *   STRIKE_OK   [2, T, S, N, N]  a strike from `from` can hit `to`
 *   CMD_DIR     [2, T, S, 8]     which neighbor directions are command squares
 *   BETWEEN     [N, N, N]        squares strictly between two squares on a line
 */

import {ALL_TILES, BAG_TILE_NAMES, BOARD_SIZE} from '@the-duke/engine';
import spec from './spec.json';

/** Constants shared with the Python training code (spec.json). */
export const SPEC = spec;

export const BOARD = BOARD_SIZE;                       // 6
export const N_SQ = BOARD * BOARD;                     // 36
export const TILE_NAMES: string[] = ALL_TILES.map(t => t.name);
export const N_TYPES = TILE_NAMES.length;              // 15
export const TILE_INDEX: Record<string, number> = Object.fromEntries(TILE_NAMES.map((n, i) => [n, i]));
export const DUKE = TILE_INDEX['Duke'];
export const FOOTMAN = TILE_INDEX['Footman'];

// Bag after setup: the full bag minus the two Footmen placed on the board.
export const INITIAL_BAG = new Int32Array(N_TYPES);
{
  const bag = [...BAG_TILE_NAMES];
  for (let k = 0; k < 2; k++) bag.splice(bag.indexOf('Footman'), 1);
  for (const name of bag) INITIAL_BAG[TILE_INDEX[name]]++;
}

// Neighbor directions, in the order used by the action encoding.
export const DIRS8 = spec.actions.dirs8 as [number, number][];
export const DIRS4 = spec.actions.dirs4 as [number, number][];

// Action space (see rules.py).
export const N_MOVE_ACTIONS = N_SQ * N_SQ;             // 1296: from * 36 + to
export const CMD_BASE = N_MOVE_ACTIONS;                // commander * 64 + src_dir * 8 + dst_dir
export const N_CMD_ACTIONS = N_SQ * 8 * 8;             // 2304
export const DRAW_ACTION = CMD_BASE + N_CMD_ACTIONS;   // 3600
export const PLACE_BASE = DRAW_ACTION + 1;             // 3601 + DIRS4 index
export const N_ACTIONS = PLACE_BASE + 4;               // 3605

export const sq = (r: number, c: number) => r * BOARD + c;
const onBoard = (r: number, c: number) => r >= 0 && r < BOARD && c >= 0 && c < BOARD;

/** FLIP[s] mirrors a square top-to-bottom; columns are not mirrored. */
export const FLIP = Int32Array.from({length: N_SQ}, (_, s) => sq(BOARD - 1 - Math.floor(s / BOARD), s % BOARD));

/** NBR8[s * 8 + d] / NBR4[s * 4 + d]: neighbor square in direction d, or -1. */
export const NBR8 = new Int32Array(N_SQ * 8).fill(-1);
export const NBR4 = new Int32Array(N_SQ * 4).fill(-1);
for (let s = 0; s < N_SQ; s++) {
  const r = Math.floor(s / BOARD), c = s % BOARD;
  DIRS8.forEach(([dr, dc], d) => { if (onBoard(r + dr, c + dc)) NBR8[s * 8 + d] = sq(r + dr, c + dc); });
  DIRS4.forEach(([dr, dc], d) => { if (onBoard(r + dr, c + dc)) NBR4[s * 4 + d] = sq(r + dr, c + dc); });
}

// --- table indexing ---
const S = 2;
/** Index into MOVE_OK / NEEDS_CLEAR / STRIKE_OK. */
export const tIdx = (owner: number, type: number, side: number, from: number, to: number) =>
  (((owner * N_TYPES + type) * S + side) * N_SQ + from) * N_SQ + to;
/** Index into CMD_DIR. */
export const cIdx = (owner: number, type: number, side: number, dir: number) =>
  ((owner * N_TYPES + type) * S + side) * 8 + dir;
/** Index into BETWEEN. */
export const bIdx = (from: number, to: number, x: number) => (from * N_SQ + to) * N_SQ + x;

export const MOVE_OK = new Uint8Array(2 * N_TYPES * S * N_SQ * N_SQ);
export const NEEDS_CLEAR = new Uint8Array(MOVE_OK.length);
export const STRIKE_OK = new Uint8Array(MOVE_OK.length);
export const CMD_DIR = new Uint8Array(2 * N_TYPES * S * 8);
export const BETWEEN = new Uint8Array(N_SQ * N_SQ * N_SQ);

function buildSide(patterns: {type: string; offsets: {dRow: number; dCol: number}[]}[]) {
  const ok = new Uint8Array(N_SQ * N_SQ);
  const path: number[][] = Array.from({length: N_SQ * N_SQ}, () => []);
  const strike = new Uint8Array(N_SQ * N_SQ);
  const cmd = new Uint8Array(8);
  const addMove = (f: number, g: number, between: number[]) => {
    // Keep the least-blocked pattern when several reach the same square.
    const k = f * N_SQ + g;
    if (!ok[k] || between.length < path[k].length) { ok[k] = 1; path[k] = between; }
  };
  for (const p of patterns) {
    for (const {dRow: dr, dCol: dc} of p.offsets) {
      if (p.type === 'command') {
        const d = DIRS8.findIndex(([a, b]) => a === dr && b === dc);
        if (d < 0) throw new Error('command squares must be adjacent');
        cmd[d] = 1;
        continue;
      }
      for (let f = 0; f < N_SQ; f++) {
        const r = Math.floor(f / BOARD), c = f % BOARD;
        if (p.type === 'step' || p.type === 'jump' || p.type === 'strike') {
          if (!onBoard(r + dr, c + dc)) continue;
          const g = sq(r + dr, c + dc);
          if (p.type === 'strike') strike[f * N_SQ + g] = 1;
          else if (p.type === 'jump') addMove(f, g, []);
          else {
            // A multi-square step can't pass through pieces on its straight line.
            const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
            const k = gcd(Math.abs(dr), Math.abs(dc));
            const between = [];
            for (let i = 1; i < k; i++) between.push(sq(r + (dr / k) * i, c + (dc / k) * i));
            addMove(f, g, between);
          }
        } else if (p.type === 'slide' || p.type === 'jump_slide') {
          const between: number[] = [];
          for (let i = 1; onBoard(r + dr * i, c + dc * i); i++) {
            const g = sq(r + dr * i, c + dc * i);
            addMove(f, g, p.type === 'slide' ? [...between] : []);
            between.push(g);
          }
        } else {
          throw new Error(`unknown move type ${p.type}`);
        }
      }
    }
  }
  for (let k = 0; k < ok.length; k++) if (ok[k] && strike[k]) throw new Error('move and strike targets overlap');
  return {ok, path, strike, cmd};
}

{
  // The opponent's tables are mine mirrored: f -> g for them is FLIP[f] -> FLIP[g]
  // for me, and their command direction (dr, dc) is my (-dr, dc).
  const mirrorDir8 = DIRS8.map(([dr, dc]) => DIRS8.findIndex(([a, b]) => a === -dr && b === dc));
  ALL_TILES.forEach((tile, t) => {
    [tile.sideA, tile.sideB].forEach((side, s) => {
      const {ok, path, strike, cmd} = buildSide(side.patterns);
      for (let f = 0; f < N_SQ; f++) {
        for (let g = 0; g < N_SQ; g++) {
          const k = f * N_SQ + g, km = FLIP[f] * N_SQ + FLIP[g];
          MOVE_OK[tIdx(0, t, s, f, g)] = ok[k];
          NEEDS_CLEAR[tIdx(0, t, s, f, g)] = path[k].length > 0 ? 1 : 0;
          STRIKE_OK[tIdx(0, t, s, f, g)] = strike[k];
          MOVE_OK[tIdx(1, t, s, f, g)] = ok[km];
          NEEDS_CLEAR[tIdx(1, t, s, f, g)] = path[km].length > 0 ? 1 : 0;
          STRIKE_OK[tIdx(1, t, s, f, g)] = strike[km];
        }
      }
      for (let d = 0; d < 8; d++) {
        CMD_DIR[cIdx(0, t, s, d)] = cmd[d];
        CMD_DIR[cIdx(1, t, s, d)] = cmd[mirrorDir8[d]];
      }
    });
  });

  for (let f = 0; f < N_SQ; f++) {
    const r = Math.floor(f / BOARD), c = f % BOARD;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const passed: number[] = [];
        for (let i = 1; onBoard(r + dr * i, c + dc * i); i++) {
          const g = sq(r + dr * i, c + dc * i);
          for (const x of passed) BETWEEN[bIdx(f, g, x)] = 1;
          passed.push(g);
        }
      }
    }
  }
}
