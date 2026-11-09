// The tutorial's lessons: each a small position, what the person is asked to
// do there, and the text around it. Pure data and engine calls, no React, so
// lessons.test.ts can play every lesson through.
//
// The person always plays Light, at the bottom of the screen, and Dark never
// answers: after each action it is Light's turn again (keepTurn). Squares are
// written as they are labelled on the board: files a-f left to right, ranks
// 1-6 from Light's back row up.
import type {Coord, GameMove, GameState, Player, Side, TileInstance} from '@the-duke/engine';
import {
  BAG_TILE_NAMES, applyMove, applySetupPlacement, cloneState, createEmptyBoard, createInitialState,
  getSetupTargets, getStandardSetupTargets, makeMove,
} from '@the-duke/engine';
import type {LastMove} from '../hooks/useGame.js';

/** Where the person is reading: the wording follows the layout and the input. */
export interface LessonContext {
  /** Board on the left, a column beside it (else stacked: things are under the board) */
  wide: boolean;
  /** A touch screen: "tap" rather than "click" */
  touch: boolean;
}

/** Something the person did that a task may be waiting for. */
export type LessonEvent =
  | {kind: 'move'; prev: GameState; move: LastMove; next: GameState}
  | {kind: 'look'; name: string; owner: Player; from: 'board' | 'bag'};

export interface Task {
  id: string;
  label: (ctx: LessonContext) => string;
  done: (event: LessonEvent) => boolean;
}

/** One step of a lesson's worked solution (lessons.test.ts plays them). */
export type SolutionStep = LastMove | {type: 'look'; name: string; owner: Player; from: 'board' | 'bag'};

export interface Lesson {
  id: string;
  title: string;
  /**
   * Paragraphs. `**bold**`, and `[step]`, `[slide]`, `[jump]`, `[jump_slide]`,
   * `[strike]`, `[command]` or `[info]` for the symbol itself (Tutorial.tsx).
   */
  body: (ctx: LessonContext) => string[];
  tasks: Task[];
  /** Said once every task is done */
  success?: (ctx: LessonContext) => string;
  start: () => {state: GameState; lastMove?: LastMove};
  /** What the person's action leads to; keepTurn unless given */
  play?: (state: GameState, move: LastMove) => GameState;
  solution: SolutionStep[];
}

// ── Building positions ──

/** 'c3' -> {row: 2, col: 2} */
export function square(name: string): Coord {
  const col = name.charCodeAt(0) - 97;
  const row = Number(name.slice(1)) - 1;
  if (!(col >= 0 && col < 6 && row >= 0 && row < 6)) throw new Error(`Bad square ${name}`);
  return {row, col};
}

interface PositionSpec {
  /** 'Name square', or 'Name square B' for a tile showing side B */
  light: string[];
  dark: string[];
  /** Each bag holds the tiles that side hasn't put on the board yet; empty unless set */
  bags?: boolean;
  lastMove?: LastMove;
}

/** The bag a side has left once these tiles are on the board. */
function bagWithout(onBoard: string[]): string[] {
  const bag = [...BAG_TILE_NAMES];
  for (const name of onBoard) {
    const i = bag.indexOf(name);
    if (i !== -1) bag.splice(i, 1);
  }
  return bag;
}

function position({light, dark, bags = false}: PositionSpec): GameState {
  const state: GameState = {
    board: createEmptyBoard(),
    tiles: new Map(),
    bags: {P1: [], P2: []},
    currentPlayer: 'P1',
    turnNumber: 9,
    status: 'active',
  };
  const placed: Record<Player, string[]> = {P1: [], P2: []};
  let n = 0;
  for (const [owner, specs] of [['P1', light], ['P2', dark]] as const) {
    for (const spec of specs) {
      const [name, where, side = 'A'] = spec.split(' ');
      const pos = square(where);
      if (state.board[pos.row][pos.col]) throw new Error(`Two tiles on ${where}`);
      const tile: TileInstance = {id: `${owner}-${name}-t${++n}`, defName: name, owner, side: side as Side, position: pos};
      state.tiles.set(tile.id, tile);
      state.board[pos.row][pos.col] = tile.id;
      placed[owner].push(name);
    }
  }
  if (bags) state.bags = {P1: bagWithout(placed.P1), P2: bagWithout(placed.P2)};
  return state;
}

/** A lesson's start position, from a spec. */
const at = (spec: PositionSpec) => () => ({state: position(spec), lastMove: spec.lastMove});

// ── Playing ──

/** The person's action, after which it is their turn again: Dark passes. */
export function keepTurn(state: GameState, move: LastMove): GameState {
  if (move.type === 'setup') return applySetupPlacement(state, move.position);
  const next = cloneState(state);
  makeMove(next, move);
  if (next.status === 'active') next.currentPlayer = state.currentPlayer;
  return next;
}

/** Applies the person's action the way a lesson does. */
export function playLesson(lesson: Lesson, state: GameState, move: LastMove): GameState {
  return (lesson.play ?? keepTurn)(state, move);
}

/** The lesson's tasks this event completes. */
export function tasksDoneBy(lesson: Lesson, event: LessonEvent): string[] {
  return lesson.tasks.filter(t => t.done(event)).map(t => t.id);
}

// ── Reading events ──

const sameSquare = (a: Coord, b: string) => {
  const s = square(b);
  return a.row === s.row && a.col === s.col;
};

/** The tile that made a move or strike, or gave a command, as it was before. */
function actor(prev: GameState, move: GameMove): TileInstance | null {
  const from = move.type === 'move' || move.type === 'strike' ? move.from : move.type === 'command' ? move.commander : null;
  const id = from && prev.board[from.row][from.col];
  return id ? prev.tiles.get(id) ?? null : null;
}

/** A move (not setup) by Light. */
function lightMove(e: LessonEvent, test: (move: GameMove, prev: GameState, next: GameState) => boolean): boolean {
  return e.kind === 'move' && e.move.type !== 'setup' && e.prev.currentPlayer === 'P1' && test(e.move, e.prev, e.next);
}

// ── Wording ──

const tap = (c: LessonContext) => (c.touch ? 'Tap' : 'Click');
const tapLower = (c: LessonContext) => (c.touch ? 'tap' : 'click');
const besideBoard = (c: LessonContext) => (c.wide ? 'beside the board' : 'under the board');

// ── The lessons ──

/** A game some moves in, to look around */
const MIDGAME: PositionSpec = {
  light: ['Duke c1', 'Footman b1', 'Footman c3 B', 'Pikeman d2', 'Knight e1'],
  dark: ['Duke d6', 'Footman d5', 'Footman e6 B', 'Wizard c5', 'Bowman e4 B'],
  bags: true,
};

export const LESSONS: Lesson[] = [
  {
    id: 'welcome',
    title: 'Welcome to The Duke',
    body: c => [
      'The Duke is a strategy game for two, played with tiles on a 6×6 board. You play **Light**, at the bottom; **Dark** sits across from you.',
      'Each side has a **Duke**, the tile with the crown. Trap the enemy Duke so it can’t escape and you win, much like checkmate in chess.',
      `On your turn you do one thing: move one of your tiles, or draw a new one from your bag. ${tap(c)} **Next** whenever you’re ready to move on.`,
    ],
    tasks: [{
      id: 'find-duke',
      label: c => `${tap(c)} your Duke`,
      done: e => e.kind === 'look' && e.from === 'board' && e.name === 'Duke' && e.owner === 'P1',
    }],
    success: () => 'That’s your Duke. Keep it safe: lose it and you lose the game.',
    start: at(MIDGAME),
    solution: [{type: 'look', name: 'Duke', owner: 'P1', from: 'board'}],
  },
  {
    id: 'setup',
    title: 'Setting up',
    body: c => [
      'Every game starts the same way. Light places its **Duke** on any square of its back row, the bottom one, then two **Footmen** on free squares right beside or in front of it.',
      `Dark then does the same on the far side, and Light moves first. ${tap(c)} a highlighted square to place each tile.`,
      'In a real game you can choose to play either side.',
    ],
    tasks: [
      {
        id: 'duke',
        label: () => 'Place your Duke on the back row',
        done: e => e.kind === 'move' && e.move.type === 'setup' && e.prev.setupPhase === 'p1_duke',
      },
      {
        id: 'footmen',
        label: () => 'Place two Footmen next to it',
        done: e => e.kind === 'move' && e.prev.setupPhase === 'p1_footman2',
      },
    ],
    success: () => 'Dark has set up too. The other tiles wait in each side’s bag.',
    start: () => ({state: createInitialState()}),
    play: (state, move) => {
      if (move.type !== 'setup') return keepTurn(state, move);
      let next = applySetupPlacement(state, move.position);
      // Light is done: Dark sets up at once, the way the AIs do
      if (next.setupPhase === 'p2_duke') {
        next = applySetupPlacement(next, getStandardSetupTargets(next)[0]);
        next = applySetupPlacement(next, getSetupTargets(next)[0]);
        const targets = getSetupTargets(next);
        next = applySetupPlacement(next, targets[targets.length - 1]);
      }
      return next;
    },
    solution: [
      {type: 'setup', position: square('c1')},
      {type: 'setup', position: square('c2')},
      {type: 'setup', position: square('d1')},
    ],
  },
  {
    id: 'step',
    title: 'Moving and flipping',
    body: c => [
      'A tile shows how it moves on its face. The tile itself is the square in the middle; the marks around it are where it can go.',
      'A dot [step] is a **move** to that square. The way there has to be clear, and if an enemy is standing on it, you capture it.',
      `Here’s the twist: every time a tile acts, it **flips** to its other side, which moves differently. Side B tiles have marked corners. ${tap(c)} a tile to see both its sides ${besideBoard(c)}.`,
    ],
    tasks: [
      {
        id: 'move',
        label: c => `${tap(c)} your Footman, then a highlighted square`,
        done: e => lightMove(e, (m, prev) => m.type === 'move' && actor(prev, m)?.defName === 'Footman'),
      },
      {
        id: 'flip-back',
        label: () => 'Move it again, with its side B moves',
        done: e => lightMove(e, (m, prev) => {
          const tile = actor(prev, m);
          return m.type === 'move' && tile?.defName === 'Footman' && tile.side === 'B';
        }),
      },
    ],
    success: () => 'Back on side A. Every tile swaps sides like this, so think one move ahead.',
    start: at({light: ['Duke c1', 'Footman c3'], dark: ['Duke d6']}),
    solution: [
      {type: 'move', from: square('c3'), to: square('c4')},
      {type: 'move', from: square('c4'), to: square('b5')},
    ],
  },
  {
    id: 'slide',
    title: 'Sliding',
    body: () => [
      'A dot with a dark centre [slide] is a **slide**: the tile moves any number of squares that way, until the edge of the board or another tile stops it. It can capture the first enemy in its path.',
      'Your Duke slides left and right on side A, and forward and back on side B.',
    ],
    tasks: [{
      id: 'capture',
      label: () => 'Slide your Duke into the Dark Footman to capture it',
      done: e => lightMove(e, (m, prev) => m.type === 'move' && actor(prev, m)?.defName === 'Duke' && sameSquare(m.to, 'e2')),
    }],
    success: () => 'Captured! A captured tile is out of the game for good.',
    start: at({light: ['Duke a2'], dark: ['Footman e2', 'Duke f6']}),
    solution: [{type: 'move', from: square('a2'), to: square('e2')}],
  },
  {
    id: 'jump',
    title: 'Jumping',
    body: () => [
      'A ring [jump] is a **jump**: the tile lands right on that square, leaping over anything in between.',
      'One rarer mark, a ring with a dot [jump_slide], is a **jump-slide**: a slide that passes over any tiles in its way. Only the Assassin has it.',
    ],
    tasks: [{
      id: 'jump',
      label: () => 'Jump your Champion over your Footman and capture the Pikeman',
      done: e => lightMove(e, m => m.type === 'move' && sameSquare(m.from, 'c2') && sameSquare(m.to, 'c4')),
    }],
    success: () => 'Over the top. Jumps ignore whatever is in the way.',
    start: at({light: ['Duke d1', 'Champion c2', 'Footman c3'], dark: ['Pikeman c4', 'Duke f6']}),
    solution: [{type: 'move', from: square('c2'), to: square('c4')}],
  },
  {
    id: 'strike',
    title: 'Striking',
    body: () => [
      'A cross [strike] is a **strike**: capture an enemy on that square without moving there.',
      'The striker stays where it is, but it still flips.',
    ],
    tasks: [{
      id: 'strike',
      label: () => 'Strike the Dark Footman with your Bowman',
      done: e => lightMove(e, m => m.type === 'strike'),
    }],
    success: () => 'The Bowman never left its square, and it flipped to side A.',
    start: at({light: ['Duke d1', 'Bowman c2 B'], dark: ['Footman c4', 'Duke f6']}),
    solution: [{type: 'strike', from: square('c2'), target: square('c4')}],
  },
  {
    id: 'command',
    title: 'Commanding',
    body: c => [
      'A diamond [command] marks a **command** square. Instead of moving, a commander can send one of your tiles from one of its command squares to another, capturing an enemy there.',
      'The commander flips; the tile it moved does not.',
      `${tap(c)} the Marshall, then your Footman, and choose **Marshall commands Footman**. Then ${tapLower(c)} where it goes.`,
    ],
    tasks: [{
      id: 'command',
      label: () => 'Have the Marshall send your Footman onto the Pikeman',
      done: e => lightMove(e, m => m.type === 'command' && sameSquare(m.targetTo, 'd3')),
    }],
    success: () => 'The Footman made the capture and stayed on side A; the Marshall flipped.',
    start: at({light: ['Duke d1', 'Marshall c2 B', 'Footman b3'], dark: ['Pikeman d3', 'Duke f6']}),
    solution: [{type: 'command', commander: square('c2'), target: square('b3'), targetTo: square('d3')}],
  },
  {
    id: 'read',
    title: 'Reading tiles',
    body: c => c.wide ? [
      `No need to memorise anything. ${tap(c)} any tile, yours or Dark’s, and both of its sides appear beside the board: the lit one is face up, the other is what it turns into next. Where a Dark tile can go is marked in red.`,
      `**Pieces left** lists every tile still in the bags. ${tap(c)} it, then a name, to see how a tile moves before anyone draws it.`,
      `Not sure what a mark means? ${tap(c)} **What’s this?** next to the symbols.`,
    ] : [
      `No need to memorise anything. ${tap(c)} any tile, yours or Dark’s, and both of its sides appear under the board: the lit one is face up, the other is what it turns into next. Where a Dark tile can go is marked in red.`,
      `With no tile picked, the strip under the board shows the tiles still in each bag. ${tap(c)} it, then a name, to see how a tile moves before anyone draws it.`,
      `Not sure what a mark means? ${tap(c)} [info] next to a tile’s sides.`,
    ],
    tasks: [
      {
        id: 'enemy',
        label: () => 'Look at one of Dark’s tiles',
        done: e => e.kind === 'look' && e.from === 'board' && e.owner === 'P2',
      },
      {
        id: 'bag',
        label: () => 'Look at a tile that is still in a bag',
        done: e => e.kind === 'look' && e.from === 'bag',
      },
    ],
    success: () => 'Whenever you’re unsure, look. Checking what your opponent can reach is half the game.',
    start: at(MIDGAME),
    solution: [
      {type: 'look', name: 'Wizard', owner: 'P2', from: 'board'},
      {type: 'look', name: 'Assassin', owner: 'P1', from: 'bag'},
    ],
  },
  {
    id: 'draw',
    title: 'Drawing tiles',
    body: c => [
      `Instead of moving, you can **draw**: press **Draw** ${c.wide ? 'under this card' : 'at the bottom of the screen'}, and a random tile from your bag is revealed. Put it on a free square right beside your Duke, not diagonally.`,
      'New tiles always start on side A. You can’t draw when every square beside your Duke is taken, or when your bag is empty.',
    ],
    tasks: [{
      id: 'draw',
      label: () => 'Draw a tile and place it next to your Duke',
      done: e => lightMove(e, m => m.type === 'place'),
    }],
    success: () => 'A new tile joins the fight. Drawing costs a turn, so weigh it against making a move.',
    start: at({
      light: ['Duke c1', 'Footman c2', 'Footman b2'],
      dark: ['Duke d6', 'Footman d5', 'Footman c6'],
      bags: true,
    }),
    solution: [{type: 'place', tileName: 'Champion', position: square('d1')}],
  },
  {
    id: 'check',
    title: 'Check',
    body: () => [
      'Dark’s Knight just jumped, flipped, and now slides straight at your Duke. Your Duke is in **check**: it glows red.',
      'You have to get it out of danger: move the Duke, block the attack, or capture the attacker. The game won’t let you make a move that leaves your Duke open.',
    ],
    tasks: [{
      id: 'escape',
      label: () => 'Get your Duke out of check',
      done: e => lightMove(e, () => true),
    }],
    success: () => 'Safe. Moving the Duke, blocking with a tile, or capturing the Knight with your Pikeman all work.',
    start: at({
      light: ['Duke c1', 'Footman b2', 'Pikeman a2'],
      dark: ['Knight c4 B', 'Duke e6'],
      bags: true,
      lastMove: {type: 'move', from: square('b6'), to: square('c4')},
    }),
    solution: [{type: 'move', from: square('a2'), to: square('c4')}],
  },
  {
    id: 'checkmate',
    title: 'Checkmate',
    body: () => [
      'To win, **checkmate** the Dark Duke: attack it so that Dark has no move that saves it. A player with no legal move at all loses too.',
      'Dark’s Duke is boxed in by its own Footmen. Find an attack it can’t block or capture. Hint: look at the Longbowman’s other side.',
    ],
    tasks: [{
      id: 'mate',
      label: () => 'Checkmate Dark in one move',
      done: e => e.kind === 'move' && e.next.status === 'P1_wins',
    }],
    success: () => 'Checkmate! A strike can’t be blocked, and nothing of Dark’s can reach the Longbowman.',
    start: at({
      light: ['Duke c1', 'Longbowman e3'],
      dark: ['Duke f6', 'Footman e6', 'Footman f5'],
      bags: true,
    }),
    // A winning move ends the game; any other and Dark passes, so the person can try again
    play: (state, move) => {
      if (move.type === 'setup') return keepTurn(state, move);
      const next = applyMove(state, move);
      return next.status === 'P1_wins' ? next : keepTurn(state, move);
    },
    solution: [{type: 'move', from: square('e3'), to: square('f3')}],
  },
  {
    id: 'ready',
    title: 'You’re ready',
    body: c => [
      'That’s the whole game. A few more things around the board:',
      '**Undo** takes back your last move. **Threats** shades every square each side can reach. **Eval** shows the neural net’s guess at who’s winning.',
      `**New game** picks your opponent and your side. Random is a gentle first opponent; the Neural Net is the strongest. You can come back here any time with ${c.wide ? '**How to play** at the top' : 'the **?** button'}.`,
    ],
    tasks: [],
    start: at(MIDGAME),
    solution: [],
  },
];
