import type { TileDefinition } from './types.js';

// Shorthand helpers for readable offset definitions.
// All offsets from P1's perspective: negative dRow = "forward" (toward P2's side).
//
// Converted from dueltenants-pieces.ts coordinate system:
//   [x, y] → { dRow: -y, dCol: x }
//   (dueltenants: y-positive = forward, x-positive = right)

const F  = { dRow: -1, dCol:  0 };
const B  = { dRow:  1, dCol:  0 };
const L  = { dRow:  0, dCol: -1 };
const R  = { dRow:  0, dCol:  1 };
const FL = { dRow: -1, dCol: -1 };
const FR = { dRow: -1, dCol:  1 };
const BL = { dRow:  1, dCol: -1 };
const BR = { dRow:  1, dCol:  1 };

const F2   = { dRow: -2, dCol:  0 };
const B2   = { dRow:  2, dCol:  0 };
const L2   = { dRow:  0, dCol: -2 };
const R2   = { dRow:  0, dCol:  2 };
const F3   = { dRow: -3, dCol:  0 };

const FL2  = { dRow: -1, dCol: -2 };  // 1 forward, 2 left
const FR2  = { dRow: -1, dCol:  2 };  // 1 forward, 2 right
const F2L  = { dRow: -2, dCol: -1 };  // 2 forward, 1 left
const F2R  = { dRow: -2, dCol:  1 };  // 2 forward, 1 right
const BL2  = { dRow:  1, dCol: -2 };  // 1 backward, 2 left (unused currently but symmetric)
const BR2  = { dRow:  1, dCol:  2 };  // 1 backward, 2 right (unused currently but symmetric)
const B2L  = { dRow:  2, dCol: -1 };  // 2 backward, 1 left
const B2R  = { dRow:  2, dCol:  1 };  // 2 backward, 1 right

const F2L2 = { dRow: -2, dCol: -2 };  // diagonal forward-left ×2
const F2R2 = { dRow: -2, dCol:  2 };  // diagonal forward-right ×2
const B2L2 = { dRow:  2, dCol: -2 };  // diagonal backward-left ×2
const B2R2 = { dRow:  2, dCol:  2 };  // diagonal backward-right ×2

// --- Starting Tiles ---
// Source: dueltenants-pieces.ts LIEUTENANT

export const DUKE: TileDefinition = {
  name: 'Duke',
  sideA: {
    patterns: [
      { type: 'slide', offsets: [L, R] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'slide', offsets: [F, B] },
    ],
  },
};

// Source: dueltenants-pieces.ts FOOTMAN

export const FOOTMAN: TileDefinition = {
  name: 'Footman',
  sideA: {
    patterns: [
      { type: 'step', offsets: [F, B, L, R] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [F2, FR, BR, BL, FL] },
    ],
  },
};

// --- Bag Tiles ---
// Source: dueltenants-pieces.ts (first occurrence of each, deduped)

export const PRIEST: TileDefinition = {
  name: 'Priest',
  sideA: {
    patterns: [
      { type: 'slide', offsets: [FR, BR, FL, BL] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [FR, BR, FL, BL] },
      { type: 'jump', offsets: [F2R2, B2R2, F2L2, B2L2] },
    ],
  },
};

export const CHAMPION: TileDefinition = {
  name: 'Champion',
  sideA: {
    patterns: [
      { type: 'step', offsets: [F, R, B, L] },
      { type: 'jump', offsets: [F2, R2, B2, L2] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'jump', offsets: [F2, R2, B2, L2] },
      { type: 'strike', offsets: [F, R, B, L] },
    ],
  },
};

export const ASSASSIN: TileDefinition = {
  name: 'Assassin',
  sideA: {
    patterns: [
      { type: 'jump_slide', offsets: [F, BL, BR] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'jump_slide', offsets: [FR, FL, B] },
    ],
  },
};

export const BOWMAN: TileDefinition = {
  name: 'Bowman',
  sideA: {
    patterns: [
      { type: 'jump', offsets: [R2, L2, B2] },
      { type: 'step', offsets: [L, R, F] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [F, BL, BR] },
      { type: 'strike', offsets: [F2, FL, FR] },
    ],
  },
};

export const DRAGOON: TileDefinition = {
  name: 'Dragoon',
  sideA: {
    patterns: [
      { type: 'step', offsets: [R, L] },
      { type: 'strike', offsets: [F2L2, F2, F2R2] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [F, F2] },
      { type: 'jump', offsets: [F2L, F2R] },
      { type: 'slide', offsets: [BL, BR] },
    ],
  },
};

export const PIKEMAN: TileDefinition = {
  name: 'Pikeman',
  sideA: {
    patterns: [
      { type: 'step', offsets: [FR, F2R2, FL, F2L2] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [F, F2, B] },
      { type: 'strike', offsets: [F2L, F2R] },
    ],
  },
};

export const GENERAL: TileDefinition = {
  name: 'General',
  sideA: {
    patterns: [
      { type: 'step', offsets: [F, B, L2, R2] },
      { type: 'jump', offsets: [F2L, F2R] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [F, L2, L, R, R2] },
      { type: 'jump', offsets: [F2L, F2R] },
      { type: 'command', offsets: [BL, B, BR, L, R] },
    ],
  },
};

export const MARSHALL: TileDefinition = {
  name: 'Marshall',
  sideA: {
    patterns: [
      { type: 'jump', offsets: [F2L2, B2, F2R2] },
      { type: 'slide', offsets: [L, R] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [L2, BL, FL, L, F, FR, R, BR, R2] },
      { type: 'command', offsets: [FL, F, FR] },
    ],
  },
};

export const LONGBOWMAN: TileDefinition = {
  name: 'Longbowman',
  sideA: {
    patterns: [
      { type: 'step', offsets: [R, L, F, B] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [BR, BL] },
      { type: 'strike', offsets: [F2, F3] },
    ],
  },
};

export const KNIGHT: TileDefinition = {
  name: 'Knight',
  sideA: {
    patterns: [
      { type: 'step', offsets: [R, L, B, B2] },
      { type: 'jump', offsets: [F2R, F2L] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [B2L2, BL, BR, B2R2] },
      { type: 'slide', offsets: [F] },
    ],
  },
};

export const SEER: TileDefinition = {
  name: 'Seer',
  sideA: {
    patterns: [
      { type: 'step', offsets: [FR, FL, BL, BR] },
      { type: 'jump', offsets: [F2, R2, B2, L2] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [F, R, B, L] },
      { type: 'jump', offsets: [F2R2, B2R2, B2L2, F2L2] },
    ],
  },
};

export const WIZARD: TileDefinition = {
  name: 'Wizard',
  sideA: {
    patterns: [
      { type: 'step', offsets: [F, FR, R, BR, B, BL, L, FL] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'jump', offsets: [F2, F2R2, R2, B2R2, B2, B2L2, L2, F2L2] },
    ],
  },
};

export const RANGER: TileDefinition = {
  name: 'Ranger',
  sideA: {
    patterns: [
      { type: 'jump', offsets: [FL2, F2L, F2R, FR2] },
      { type: 'slide', offsets: [F, B] },
    ],
  },
  sideB: {
    patterns: [
      { type: 'jump', offsets: [B2L, B2R] },
      { type: 'slide', offsets: [FR, FL] },
    ],
  },
};

// --- Registry ---

export const ALL_TILES: TileDefinition[] = [
  DUKE, FOOTMAN, PRIEST, CHAMPION, ASSASSIN, BOWMAN,
  DRAGOON, PIKEMAN, GENERAL, MARSHALL, LONGBOWMAN, KNIGHT,
  SEER, WIZARD, RANGER,
];

export const TILE_REGISTRY: ReadonlyMap<string, TileDefinition> = new Map(
  ALL_TILES.map(t => [t.name, t]),
);

/**
 * Tile names that go into each player's draw bag at game start.
 * Counts from dueltenants-pieces.ts defaultBag.
 * (Duke is placed on the board, not in the bag.)
 */
export const BAG_TILE_NAMES: readonly string[] = [
  'Footman', 'Footman', 'Footman',
  'Pikeman', 'Pikeman', 'Pikeman',
  'Priest', 'Champion', 'Assassin', 'Bowman',
  'Dragoon', 'General', 'Marshall', 'Longbowman',
  'Knight', 'Seer', 'Wizard', 'Ranger',
];
