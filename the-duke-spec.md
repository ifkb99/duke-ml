# The Duke — Game Engine Spec & Implementation Plan

## Project Goal

Build a browser-playable version of the board game "The Duke" with hot-seat multiplayer and AI opponents. The project serves double duty as a portfolio piece demonstrating full-stack web development (React + Node) and ML engineering (self-play reinforcement learning).

**Stack:** React (frontend), Node/TypeScript (backend + game engine), Python/PyTorch (ML training), ONNX (inference)

**Key architectural decision:** The game engine is written in pure TypeScript with zero UI dependencies. It runs identically on server and client. This means hot-seat mode needs no backend at all, AI can run server-side or client-side, and the engine is independently testable.

---

## Phase Overview

| Phase | Deliverable | Ship? |
|-------|------------|-------|
| 1 — Game Engine | TypeScript library: state, moves, validation, win detection | No (internal) |
| 2 — Hot Seat UI | Playable 2-player game in browser, minimal UI, rules page | **Yes — push to GitHub** |
| 3 — Classical AI | Random bot → Minimax w/ alpha-beta → playable AI opponent | **Yes — this is MVP** |
| 4 — ML AI | AlphaZero-style self-play training, ONNX inference in browser | Yes — portfolio upgrade |

---

## Game Rules Summary

The Duke is a 2-player abstract strategy game on a **6×6 board**.

### Setup
- Each player starts with their **Duke** and **2 Footmen** on the board.
- Player 1 (light) places Duke on any space in their back row (row 0). Footmen go on two of the orthogonally adjacent spaces.
- Player 2 (dark) mirrors this on their back row (row 5).
- Remaining tiles go into each player's **draw bag** (randomized, hidden information).

### Turn Structure
On your turn you do **exactly one** of:
1. **Move/capture** a tile already on the board using its movement pattern
2. **Draw a tile** from your bag and place it on any empty space orthogonally adjacent to your Duke

After a tile **moves**, it **flips** to its other side (changing its movement pattern).
Tiles placed from the bag always start on **Side A**.
Tiles that use **strike** or **command** do **not** flip (they didn't move).

### Movement Types
Each side of a tile defines movement patterns using these types:

| Type | Behavior |
|------|----------|
| **Step** | Move exactly to the offset indicated. Cannot move through pieces. Captures by landing on enemy. |
| **Slide** | Move any number of squares in the direction indicated (like rook/bishop in chess). Blocked by pieces. Can capture the first enemy in path. |
| **Jump** | Move exactly to the offset indicated. **Ignores** pieces in between. Captures by landing on enemy. |
| **Strike** | **Capture at range** without moving. Remove enemy at indicated offset. Tile does **not** flip. |
| **Command** | Move a **friendly** tile (not self) that is at the indicated offset. That friendly tile moves one space in any direction. Commander does **not** flip. Commanded tile **does** flip. |

### Win Condition
**Capture the opponent's Duke.** That's it. No stalemate rules — if you can't move, you lose (though this is extremely rare in practice).

---

## Core Data Structures

### Coordinate System

```typescript
// (0,0) is top-left. Row 0 is Player 1's back row, Row 5 is Player 2's.
// This convention means Player 1's "forward" is +row, Player 2's is -row.
type Coord = { row: number; col: number };

// Movement offsets are defined from Player 1's perspective.
// When generating moves for Player 2, negate the row component.
type Offset = { dRow: number; dCol: number };
```

### Tile Definitions

```typescript
type MoveType = 'step' | 'slide' | 'jump' | 'strike' | 'command';

/**
 * A single movement capability.
 * - step/jump/strike/command: `offsets` are exact relative positions
 * - slide: `offsets` are direction vectors (e.g., {dRow:0, dCol:1} = slide right)
 */
interface MovePattern {
  type: MoveType;
  offsets: Offset[];
}

interface TileSide {
  patterns: MovePattern[];
}

interface TileDefinition {
  name: string;
  sideA: TileSide;
  sideB: TileSide;
}
```

**Example — Footman:**

```typescript
const FOOTMAN: TileDefinition = {
  name: 'Footman',
  sideA: {
    patterns: [
      { type: 'step', offsets: [
        { dRow: -1, dCol: 0 },  // forward
        { dRow: 1, dCol: 0 },   // backward
        { dRow: 0, dCol: -1 },  // left
        { dRow: 0, dCol: 1 },   // right
      ]},
    ],
  },
  sideB: {
    patterns: [
      { type: 'step', offsets: [
        { dRow: -1, dCol: 0 },  // forward
        { dRow: -1, dCol: -1 }, // forward-left
        { dRow: -1, dCol: 1 },  // forward-right
      ]},
      { type: 'slide', offsets: [
        { dRow: 1, dCol: 0 },   // backward slide
      ]},
    ],
  },
};
```

> **IMPORTANT**: All offsets are from Player 1's perspective. When computing moves
> for Player 2, multiply `dRow` by -1. This keeps tile definitions simple and
> avoids duplication.

### Tile Instance (on the board)

```typescript
type Player = 'P1' | 'P2';
type Side = 'A' | 'B';

interface TileInstance {
  defName: string;       // key into tile definition registry
  owner: Player;
  side: Side;
  position: Coord;
  id: string;            // unique instance ID, e.g. "P1-Footman-1"
}
```

### Board State

```typescript
// The board is a 6x6 grid. Each cell is either null or a tile instance ID.
// Use a Map for the tile registry to avoid stale references.

type BoardGrid = (string | null)[][]; // 6x6, stores tile instance IDs

interface GameState {
  board: BoardGrid;
  tiles: Map<string, TileInstance>;  // id -> instance
  bags: {
    P1: string[];  // tile definition names remaining in bag
    P2: string[];
  };
  currentPlayer: Player;
  turnNumber: number;
  status: 'active' | 'P1_wins' | 'P2_wins';
}
```

**Why this structure?** The `board` grid gives O(1) positional lookups ("what's at row 3, col 2?"). The `tiles` map gives O(1) lookups by ID ("where is P1's Duke?"). Both are needed constantly during move generation. The grid stores IDs (not object references) so the state is trivially serializable for AI search and network transmission.

### Move Representation

```typescript
type GameMove =
  | { type: 'place'; tileName: string; position: Coord }
  | { type: 'move'; from: Coord; to: Coord }             // includes capture-by-landing
  | { type: 'strike'; from: Coord; target: Coord }
  | { type: 'command'; commander: Coord; target: Coord; targetTo: Coord };
```

---

## Move Generation

This is the most complex part of the engine. Get this right and everything else is easy.

### `generateAllMoves(state: GameState): GameMove[]`

For the current player, generate every legal move. Two categories:

#### 1. Placement Moves
If the player's bag is non-empty:
- Find the player's Duke on the board
- For each orthogonally adjacent cell to the Duke that is **empty**:
  - For each **unique tile name** remaining in the bag (deduplicate — don't list "Footman" twice):
    - Emit `{ type: 'place', tileName, position }`

> **Design note:** In the physical game, you draw a random tile then place it.
> For the engine, we enumerate all possible placements. The AI evaluates all of them.
> For human play, you can either let the player choose what to draw (full info variant)
> or simulate the random draw. Decide this at the UI layer, not the engine layer.
> **Recommendation:** for AI training, use the full-info variant (all placements legal).
> For human vs human, consider random draw for authenticity. Either way, the engine
> supports both — just filter the placement moves at the UI layer.

#### 2. Tile Moves
For each tile the current player has on the board:
- Get its definition and current side
- For each `MovePattern` on that side:
  - Apply the appropriate generation logic:

**Step generation:**
```
for each offset in pattern.offsets:
  target = position + adjusted_offset  // adjust for player orientation
  if target is in bounds:
    if target is empty OR target has enemy tile:
      emit { type: 'move', from: position, to: target }
```

**Slide generation:**
```
for each direction in pattern.offsets:
  current = position
  loop:
    current = current + adjusted_direction
    if current is out of bounds: break
    if current has friendly tile: break
    if current is empty:
      emit { type: 'move', from: position, to: current }
    if current has enemy tile:
      emit { type: 'move', from: position, to: current }
      break  // can't slide through captures
```

**Jump generation:**
```
for each offset in pattern.offsets:
  target = position + adjusted_offset
  if target is in bounds:
    if target is empty OR target has enemy tile:
      emit { type: 'move', from: position, to: target }
  // NOTE: no obstruction check — jumps ignore intervening pieces
```

**Strike generation:**
```
for each offset in pattern.offsets:
  target = position + adjusted_offset
  if target is in bounds AND target has enemy tile:
    emit { type: 'strike', from: position, target }
  // NOTE: only legal if there IS an enemy there
```

**Command generation:**
```
for each offset in pattern.offsets:
  target = position + adjusted_offset
  if target is in bounds AND target has FRIENDLY tile (not self):
    for each orthogonal direction:
      destination = target + direction
      if destination is in bounds:
        if destination is empty OR destination has enemy tile:
          emit { type: 'command', commander: position, target, targetTo: destination }
```

### Applying a Move: `applyMove(state: GameState, move: GameMove): GameState`

**Always return a new state. Never mutate.** This is critical for AI search (minimax needs to explore and backtrack).

```
For 'place':
  - Remove one instance of tileName from current player's bag
  - Create new TileInstance (side: 'A', position: move.position)
  - Place on board

For 'move':
  - If target cell has enemy tile: remove it (capture). Check if it's the Duke → game over.
  - Move tile from `from` to `to`
  - **Flip the tile** (A→B, B→A)

For 'strike':
  - Remove enemy tile at target. Check if Duke → game over.
  - Striking tile does **NOT** flip, does **NOT** move.

For 'command':
  - Move the friendly tile at `target` to `targetTo`
  - If `targetTo` has enemy tile: capture. Check Duke.
  - **Commanded tile flips.** Commander does **NOT** flip.
```

After applying, toggle `currentPlayer` and increment `turnNumber`.

---

## Tile Definitions — Base Game

Here are the standard tiles. Each player gets one of each (except Footman ×2, which start on board). Implement these first.

Offsets below are from **Player 1's perspective** (positive dRow = toward Player 2 / "backward" for P1).

> **Notation guide:** F = forward (dRow: -1), B = backward (dRow: 1), L = (dCol: -1), R = (dCol: 1).
> Numbers indicate distance, e.g. F2 = {dRow: -2, dCol: 0}.

### Starting Tiles (on board at game start)

**Duke**
| Side A | Side B |
|--------|--------|
| Slide: F, B, L, R | Step: F, B, L, R, FL, FR, BL, BR |

**Footman** (×2 per player)
| Side A | Side B |
|--------|--------|
| Step: F, B, L, R | Step: F, FL, FR; Slide: B |

### Bag Tiles

**Pikeman**
| Side A | Side B |
|--------|--------|
| Step: F, FL, FR | Step: F2; Slide: B |

**Knight**
| Side A | Side B |
|--------|--------|
| Jump: F2L, F2R | Step: F, B; Jump: FL2, FR2 |

> Knight offsets — F2L = {dRow: -2, dCol: -1}, F2R = {dRow: -2, dCol: 1},
> FL2 = {dRow: -1, dCol: -2}, FR2 = {dRow: -1, dCol: 2}

**Sergeant**
| Side A | Side B |
|--------|--------|
| Step: FL, FR | Step: F, BL, BR |

**Longbowman**
| Side A | Side B |
|--------|--------|
| Step: F, BL, BR; Strike: F2, F3 | Slide: B, FL, FR |

**Champion**
| Side A | Side B |
|--------|--------|
| Jump: F2, B2, L2, R2 | Step: FL, FR, BL, BR; Jump: F2, B2, L2, R2 |

**Wizard**
| Side A | Side B |
|--------|--------|
| Step: FL, FR, BL, BR | Slide: FL, FR, BL, BR |

**Marshall**
| Side A | Side B |
|--------|--------|
| Slide: F, FL, FR | Jump: F2, L2, R2; Slide: B |

**General**
| Side A | Side B |
|--------|--------|
| Step: F; Slide: FL, FR, B | Slide: F, BL, BR; Step: B |

**Ranger**
| Side A | Side B |
|--------|--------|
| Slide: FL, FR, BL, BR | Slide: F, B, L, R |

**Priest**
| Side A | Side B |
|--------|--------|
| Step: F; Slide: BL, BR | Slide: FL, FR; Step: B |

**Dragoon**
| Side A | Side B |
|--------|--------|
| Slide: F; Step: BL, BR; Strike: L, R | Jump: FL, FR; Step: BL, BR; Slide: B |

**Assassin**
| Side A | Side B |
|--------|--------|
| Jump: F2, FL, FR | Jump: BL, BR; Slide: F |

**Seer**
| Side A | Side B |
|--------|--------|
| Step: F, B, FL, FR | Slide: BL, BR; Step: F, L, R |

> ⚠️ **Double-check all tile definitions against the physical game or official reference
> before finalizing.** The patterns above are from memory and may have errors. This is
> worth getting exactly right since the entire game depends on it. The official tile
> images show movement patterns as icons on a 5×5 grid centered on the tile.

---

## State Serialization (for AI & Networking)

The GameState needs to be:
1. **Serializable** — for sending over websockets, storing in history, AI evaluation
2. **Hashable** — for transposition tables in minimax

```typescript
// Compact serialization: convert to a plain object with arrays instead of Maps
interface SerializedGameState {
  board: (string | null)[][];          // 6x6 grid of tile IDs
  tiles: [string, TileInstance][];     // entries from the Map
  bags: { P1: string[]; P2: string[] };
  currentPlayer: Player;
  turnNumber: number;
  status: string;
}

function serialize(state: GameState): SerializedGameState { /* ... */ }
function deserialize(data: SerializedGameState): GameState { /* ... */ }

// For transposition table hashing, use a Zobrist-style hash:
// - Assign a random 64-bit number to each (tile_type, side, owner, row, col) combination
// - XOR them together for the full board
// - XOR in a value for current player
// This gives O(1) incremental hash updates when applying moves.
```

---

## AI Design

### Phase 3: Classical AI

**Level 0 — Random:** Pick a random legal move. Useful as a baseline and for testing.

**Level 1 — Minimax + Alpha-Beta:**

```
function minimax(state, depth, alpha, beta, maximizing):
  if depth == 0 or state.status != 'active':
    return evaluate(state)

  moves = generateAllMoves(state)
  if maximizing:
    value = -Infinity
    for move in moves:
      child = applyMove(state, move)
      value = max(value, minimax(child, depth - 1, alpha, beta, false))
      alpha = max(alpha, value)
      if alpha >= beta: break
    return value
  else:
    // symmetric for minimizing
```

**Evaluation function (heuristic):**

Start simple and iterate:

```typescript
function evaluate(state: GameState): number {
  if (state.status === 'P1_wins') return +100000;
  if (state.status === 'P2_wins') return -100000;

  let score = 0;

  // Material: count tiles on board and in bag
  // Tiles on board are worth more than tiles in bag
  for (const [id, tile] of state.tiles) {
    const value = TILE_VALUES[tile.defName] ?? 1;
    const multiplier = tile.owner === 'P1' ? 1 : -1;
    score += value * multiplier;
  }

  // Mobility: number of legal moves (compute for both sides)
  // More moves = more options = generally better
  // (This is expensive — consider skipping at lower difficulties)

  // Duke safety: penalize Duke positions with many enemy attack lines

  // Control: bonus for tiles near center of board

  return score;
}
```

**TILE_VALUES** — rough starting point (tune later):
| Tile | Value |
|------|-------|
| Duke | 0 (loss if captured, handled by win check) |
| Footman | 2 |
| Pikeman | 2 |
| Sergeant | 2 |
| Knight | 3 |
| Longbowman | 4 |
| Champion | 3 |
| Wizard | 3 |
| Marshall | 4 |
| General | 4 |
| Ranger | 4 |
| Priest | 3 |
| Dragoon | 4 |
| Assassin | 3 |
| Seer | 3 |

**Depth:** Start with depth 4. The 6×6 board with limited pieces means branching factor is moderate (maybe 15-30 moves per position), so depth 4-6 should be feasible without much optimization. Add iterative deepening and move ordering later if needed.

**Handling the draw bag (hidden info):**
The bag introduces stochasticity — you don't know what you'll draw. For minimax:
- **Option A (recommended for MVP):** Treat as full information. Both players can see both bags. This simplifies everything and the AI just evaluates all placement options.
- **Option B:** Expectiminimax — average over possible draws. More correct but significantly more complex and slower.
- Go with Option A for Phase 3. Revisit in Phase 4 if needed.

### Phase 4: ML AI (AlphaZero-style)

High-level approach — details to be planned when you get here:

**Network input (board representation as tensor):**
- 6×6 spatial dimensions
- Feature planes (channels):
  - Per tile type × per side (A/B) × per owner (own/opponent): binary plane (1 if that tile is there)
  - Current player indicator (full plane of 1s or 0s)
  - Turn number (normalized)
- Approximately ~60-80 channels × 6 × 6

**Network output:**
- **Policy head:** probability distribution over all possible moves (needs a move indexing scheme)
- **Value head:** single scalar, probability of winning from this position

**Training loop:**
1. Play games via MCTS guided by current network
2. Store (state, MCTS policy, game outcome) tuples
3. Train network to predict policy and value
4. Repeat

**Inference:**
- Train in PyTorch
- Export to ONNX
- Run in browser via onnxruntime-web, or server-side via onnxruntime-node
- MCTS runs in TypeScript, calls network for evaluation

---

## Project Structure

```
the-duke/
├── packages/
│   ├── engine/              # Pure TypeScript game engine
│   │   ├── src/
│   │   │   ├── types.ts         # All type definitions
│   │   │   ├── tiles.ts         # Tile definitions registry
│   │   │   ├── state.ts         # GameState creation, serialization
│   │   │   ├── moves.ts         # Move generation
│   │   │   ├── game.ts          # applyMove, win detection, game loop
│   │   │   └── index.ts         # Public API
│   │   └── tests/
│   │       ├── moves.test.ts    # Move generation tests (THE critical tests)
│   │       ├── game.test.ts     # Full game flow tests
│   │       └── tiles.test.ts    # Tile definition validation
│   │
│   ├── ai/                  # AI implementations
│   │   ├── src/
│   │   │   ├── random.ts        # Random move bot
│   │   │   ├── evaluate.ts      # Board evaluation heuristic
│   │   │   ├── minimax.ts       # Minimax + alpha-beta
│   │   │   └── index.ts
│   │   └── tests/
│   │
│   └── web/                 # React frontend
│       ├── src/
│       │   ├── components/
│       │   │   ├── Board.tsx         # 6x6 board rendering
│       │   │   ├── Tile.tsx          # Individual tile display (both sides)
│       │   │   ├── MoveOverlay.tsx   # Highlight legal moves
│       │   │   ├── GameControls.tsx  # New game, undo, settings
│       │   │   └── RulesPage.tsx     # How to play
│       │   ├── hooks/
│       │   │   ├── useGame.ts        # Game state management
│       │   │   └── useAI.ts          # AI move computation (web worker)
│       │   ├── App.tsx
│       │   └── main.tsx
│       └── public/
│
├── training/                # Python ML training (Phase 4)
│   ├── model.py
│   ├── self_play.py
│   ├── train.py
│   └── export_onnx.py
│
└── README.md
```

**Why a monorepo with packages?** The engine has no dependencies and can be tested in isolation. The AI package depends only on the engine. The web package depends on both. This keeps boundaries clean and lets you swap out the frontend later if needed (e.g., add a CLI interface for AI benchmarking).

Use a simple workspace setup (npm/pnpm workspaces). Don't over-engineer the build — no Nx, no Turborepo. `tsc` and Vite are plenty.

---

## Testing Strategy

**The engine is the foundation. If the engine has bugs, everything built on top is broken. Test it aggressively.**

Priority test cases for move generation:

1. **Each move type in isolation:** Set up a board with one tile, verify it generates exactly the expected moves. Cover step, slide, jump, strike, command.
2. **Slide blocking:** Friendly piece blocks slide. Enemy piece blocks slide but is capturable. Edge of board blocks slide.
3. **Jump over pieces:** Verify jumps aren't blocked by intervening pieces.
4. **Strike only on enemies:** Strike should not generate moves for empty squares.
5. **Command moves friendly:** Verify command targets friendly tiles, not enemies or self.
6. **Player orientation:** Verify P2's moves are correctly mirrored (row negation).
7. **Tile flipping:** After a step/slide/jump, tile flips. After strike/command, it doesn't.
8. **Placement adjacency:** Tiles can only be placed orthogonally adjacent to own Duke.
9. **Win detection:** Capturing the Duke ends the game immediately.
10. **Full game playthrough:** Play a scripted short game, verify each state transition.

---

## Implementation Order (within Phase 1)

1. `types.ts` — all the interfaces above
2. `tiles.ts` — define Footman, Duke, and one simple tile (Sergeant). **Don't do all tiles at once.** Get the engine working with 3 tile types first, then add the rest.
3. `state.ts` — createInitialState, serialize/deserialize
4. `moves.ts` — start with step generation only. Test it. Add slide. Test it. Add jump. Test. Etc.
5. `game.ts` — applyMove, status checking
6. Integration test: play a game programmatically using only Footmen and Dukes
7. Add remaining tile definitions one at a time, testing each

---

## Notes & Pitfalls

- **Offset orientation is the #1 source of bugs.** Every single move generation path needs to account for which player owns the tile. Write a helper like `adjustOffset(offset: Offset, player: Player): Offset` and use it everywhere.

- **Immutable state updates.** `applyMove` must return a brand new GameState. Deep clone the board grid and tiles map. This is critical for minimax correctness. If performance becomes an issue later, you can switch to an undo-stack approach, but start with immutable copies.

- **Don't render tiles as images initially.** Use colored text/symbols or simple CSS shapes showing the movement pattern. You can make it pretty later. The goal of Phase 2 is "it works" not "it's beautiful."

- **Web Worker for AI.** Minimax search will block the main thread. Run it in a Web Worker from day one so the UI stays responsive. The engine's serializability makes this trivial — post the serialized state to the worker, get a move back.

- **Git discipline.** Commit at every milestone: "engine: step move generation works", "engine: full move gen for all types", "ui: board renders", etc. This creates a visible history of progress that tells a story to anyone looking at the repo.
