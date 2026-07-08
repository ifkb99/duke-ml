/**
 * Canonical string notation for Duke moves.
 *
 * Squares use algebraic style: columns a–f (col 0–5), rows 1–6 (row 0–5).
 * So "a1" = {row: 0, col: 0} (P1's back-left corner from P1's view).
 *
 * Move formats:
 *   MOVE b2 c3            — move/jump/slide/jump_slide from b2 to c3
 *   STRIKE c3 d4          — tile at c3 strikes (captures without moving) d4
 *   COMMAND c2 b2 b3      — commander at c2 moves friendly tile b2 to b3
 *   PLACE Pikeman c1      — place Pikeman from bag at c1
 *
 * Designed so LLM output is trivially parseable and log lines are grep-able.
 */
import type { Coord, GameMove } from '@the-duke/engine';

export function coordToSquare(c: Coord): string {
  return `${String.fromCharCode(97 + c.col)}${c.row + 1}`;
}

export function squareToCoord(s: string): Coord | null {
  const m = /^([a-f])([1-6])$/.exec(s.trim().toLowerCase());
  if (!m) return null;
  return { row: parseInt(m[2], 10) - 1, col: m[1].charCodeAt(0) - 97 };
}

export function moveToString(move: GameMove): string {
  switch (move.type) {
    case 'move':
      return `MOVE ${coordToSquare(move.from)} ${coordToSquare(move.to)}`;
    case 'strike':
      return `STRIKE ${coordToSquare(move.from)} ${coordToSquare(move.target)}`;
    case 'command':
      return `COMMAND ${coordToSquare(move.commander)} ${coordToSquare(move.target)} ${coordToSquare(move.targetTo)}`;
    case 'place':
      return `PLACE ${move.tileName} ${coordToSquare(move.position)}`;
  }
}

/**
 * Parse a move string. Tolerant of case, extra whitespace, and common LLM
 * decoration (arrows, "->", trailing punctuation). Returns null on failure —
 * callers decide whether that's a parse failure or an illegal move.
 */
export function parseMove(raw: string): GameMove | null {
  const cleaned = raw
    .replace(/[`*_>]/g, ' ')
    .replace(/->|→|to /gi, ' ')
    .replace(/[.,;!]+$/g, '')
    .trim();

  const tokens = cleaned.split(/\s+/);
  if (tokens.length === 0) return null;
  const verb = tokens[0].toUpperCase();

  if (verb === 'MOVE' && tokens.length >= 3) {
    const from = squareToCoord(tokens[1]);
    const to = squareToCoord(tokens[2]);
    return from && to ? { type: 'move', from, to } : null;
  }
  if (verb === 'STRIKE' && tokens.length >= 3) {
    const from = squareToCoord(tokens[1]);
    const target = squareToCoord(tokens[2]);
    return from && target ? { type: 'strike', from, target } : null;
  }
  if (verb === 'COMMAND' && tokens.length >= 4) {
    const commander = squareToCoord(tokens[1]);
    const target = squareToCoord(tokens[2]);
    const targetTo = squareToCoord(tokens[3]);
    return commander && target && targetTo
      ? { type: 'command', commander, target, targetTo }
      : null;
  }
  if (verb === 'PLACE' && tokens.length >= 3) {
    // Tile name may be any casing; normalize to TitleCase to match registry keys.
    const name = tokens[1].charAt(0).toUpperCase() + tokens[1].slice(1).toLowerCase();
    const position = squareToCoord(tokens[2]);
    return position ? { type: 'place', tileName: name, position } : null;
  }
  return null;
}

/** Structural equality — used to check a parsed move against the legal list. */
export function movesEqual(a: GameMove, b: GameMove): boolean {
  return moveToString(a) === moveToString(b);
}
