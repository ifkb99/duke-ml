import {describe, expect, it} from 'vitest';
import type {Coord, GameMove, GameState} from '@the-duke/engine';
import {generateAllMoves, getSetupTargets, isPlayerInCheck} from '@the-duke/engine';
import type {LastMove} from '../hooks/useGame.js';
import {LESSONS, type Lesson, type LessonEvent, playLesson, square, tasksDoneBy} from './lessons.js';

const same = (a: Coord, b: Coord) => a.row === b.row && a.col === b.col;

/** Whether the board would let the person make this move now. */
function isAllowed(state: GameState, move: LastMove): boolean {
  if (move.type === 'setup') return getSetupTargets(state).some(t => same(t, move.position));
  const legal = generateAllMoves(state);
  if (move.type === 'place') {
    return state.bags[state.currentPlayer].includes(move.tileName)
      && legal.some(m => m.type === 'draw' && same(m.position, move.position));
  }
  return legal.some(m => JSON.stringify(m) === JSON.stringify(move));
}

/** Plays the lesson's solution; returns the tasks it completed and the final position. */
function playThrough(lesson: Lesson) {
  let {state} = lesson.start();
  const done = new Set<string>();
  for (const step of lesson.solution) {
    let event: LessonEvent;
    if (step.type === 'look') {
      event = {kind: 'look', name: step.name, owner: step.owner, from: step.from};
    } else {
      expect(isAllowed(state, step), `${lesson.id}: ${JSON.stringify(step)} is not allowed`).toBe(true);
      const next = playLesson(lesson, state, step);
      event = {kind: 'move', prev: state, move: step, next};
      state = next;
    }
    for (const id of tasksDoneBy(lesson, event)) done.add(id);
  }
  return {done, state};
}

describe('tutorial lessons', () => {
  it('have unique ids', () => {
    expect(new Set(LESSONS.map(l => l.id)).size).toBe(LESSONS.length);
  });

  for (const lesson of LESSONS) {
    describe(lesson.id, () => {
      it('starts on Light’s turn, with something to do', () => {
        const {state} = lesson.start();
        expect(state.currentPlayer).toBe('P1');
        if (state.status === 'setup') {
          expect(getSetupTargets(state).length).toBeGreaterThan(0);
        } else {
          expect(state.status).toBe('active');
          expect(generateAllMoves(state).length).toBeGreaterThan(0);
          // Only the check lesson starts with a Duke in check, and never Dark's
          expect(isPlayerInCheck(state, 'P1')).toBe(lesson.id === 'check');
          expect(isPlayerInCheck(state, 'P2')).toBe(false);
        }
      });

      it('its solution completes every task', () => {
        const {done} = playThrough(lesson);
        expect([...done].sort()).toEqual(lesson.tasks.map(t => t.id).sort());
      });

      it('keeps it Light’s turn until the game is won', () => {
        const {state} = playThrough(lesson);
        if (state.status === 'active') expect(state.currentPlayer).toBe('P1');
      });
    });
  }

  const lesson = (id: string) => LESSONS.find(l => l.id === id)!;

  it('setup: Dark sets up as soon as Light has, then Light moves first', () => {
    const {state} = playThrough(lesson('setup'));
    expect(state.status).toBe('active');
    expect(state.currentPlayer).toBe('P1');
    expect([...state.tiles.values()].filter(t => t.owner === 'P2')).toHaveLength(3);
  });

  it('step: the Footman flips with each move', () => {
    const l = lesson('step');
    let {state} = l.start();
    const sides: string[] = [];
    for (const step of l.solution as GameMove[]) {
      state = playLesson(l, state, step);
      const to = step.type === 'move' ? step.to : null;
      sides.push(state.tiles.get(state.board[to!.row][to!.col]!)!.side);
    }
    expect(sides).toEqual(['B', 'A']);
  });

  it('check: every legal move gets the Duke out of check, and there are several kinds', () => {
    const l = lesson('check');
    const {state} = l.start();
    const moves = generateAllMoves(state);
    const kinds = new Set<string>();
    for (const m of moves) {
      const next = playLesson(l, state, m.type === 'draw' ? {type: 'place', tileName: state.bags.P1[0], position: m.position} : m);
      expect(isPlayerInCheck(next, 'P1')).toBe(false);
      const id = m.type === 'move' ? state.board[m.from.row][m.from.col] : null;
      kinds.add(m.type === 'draw' ? 'block by drawing' : state.tiles.get(id!)!.defName);
    }
    // The Duke steps aside, the Footman blocks, the Pikeman captures, a drawn tile blocks
    expect(kinds).toEqual(new Set(['Duke', 'Footman', 'Pikeman', 'block by drawing']));
  });

  it('checkmate: only a mating move wins; any other leaves it Light’s turn', () => {
    const l = lesson('checkmate');
    const {state} = l.start();
    let wins = 0;
    for (const m of generateAllMoves(state)) {
      if (m.type === 'draw') continue;
      const next = playLesson(l, state, m);
      if (next.status === 'P1_wins') wins++;
      else expect(next.currentPlayer).toBe('P1');
    }
    expect(wins).toBe(1);
  });

  it('command: the Marshall can command the Footman', () => {
    const {state} = lesson('command').start();
    const moves = generateAllMoves(state);
    const fromMarshall = (m: GameMove) => m.type === 'command' && same(m.commander, square('c2'));
    expect(moves.filter(fromMarshall).map(m => m.type === 'command' && m.target)).toContainEqual(square('b3'));
  });
});
