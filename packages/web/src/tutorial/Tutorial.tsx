import {useCallback, useEffect, useMemo, useState, type ReactNode} from 'react';
import type {MoveType} from '@the-duke/engine';
import {ActionBar} from '../components/ActionBar.js';
import {BagDrawer, BagStrip} from '../components/Bags.js';
import {Board} from '../components/Board.js';
import {CheckIcon, ChevronLeftIcon, ChevronRightIcon, CloseIcon, InfoIcon, RematchIcon} from '../components/Icons.js';
import {Inspector, Legend, SymbolKey, inspectorSubject} from '../components/Inspector.js';
import {LegendIcon} from '../components/MoveDiagram.js';
import {Crown} from '../components/Tile.js';
import {useGame} from '../hooks/useGame.js';
import {LESSONS, type Lesson, type LessonContext, type LessonEvent, playLesson, tasksDoneBy} from './lessons.js';
import './tutorial.css';

/** The same breakpoint as the layout's (controls.css): board left, a column beside it */
const WIDE = '(min-aspect-ratio: 5/4) and (min-width: 560px)';
const TOUCH = '(pointer: coarse)';

function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const mq = matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    onChange();
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [query]);
  return matches;
}

interface TutorialProps {
  /** Leave partway */
  onExit: () => void;
  /** Done with the last lesson: on to a real game */
  onFinish: () => void;
}

/** The interactive tutorial: one lesson at a time, each on its own small board. */
export function Tutorial({onExit, onFinish}: TutorialProps) {
  const [index, setIndex] = useState(0);
  // Bumped to start the lesson over: its board remounts with a fresh position
  const [attempt, setAttempt] = useState(0);
  const [completed, setCompleted] = useState<ReadonlySet<string>>(() => new Set());
  const wide = useMedia(WIDE);
  const touch = useMedia(TOUCH);
  const ctx = useMemo<LessonContext>(() => ({wide, touch}), [wide, touch]);
  const lesson = LESSONS[index];
  const last = index === LESSONS.length - 1;

  const go = (i: number) => {
    setIndex(i);
    setAttempt(0);
  };
  const onComplete = useCallback((id: string) => setCompleted(s => new Set(s).add(id)), []);

  return (
    <div className="app tutorial">
      <header className="topbar">
        <div className="brand">
          <Crown className="crown brand-crown" />
          <h1>The Duke</h1>
        </div>
        <span className="matchup">How to play</span>
        <button type="button" className="btn" onClick={onExit}>
          <CloseIcon /> <span className="btn-label">Exit tutorial</span>
        </button>
      </header>
      <LessonView key={`${lesson.id}-${attempt}`} lesson={lesson} ctx={ctx}
                  number={index + 1} completed={completed} onComplete={onComplete}
                  onBack={index > 0 ? () => go(index - 1) : undefined}
                  onNext={last ? onFinish : () => go(index + 1)} last={last}
                  onReset={() => setAttempt(a => a + 1)} onExit={onExit} />
    </div>
  );
}

interface LessonViewProps {
  lesson: Lesson;
  ctx: LessonContext;
  /** 1-based */
  number: number;
  completed: ReadonlySet<string>;
  onComplete: (id: string) => void;
  onBack?: () => void;
  onNext: () => void;
  last: boolean;
  onReset: () => void;
  onExit: () => void;
}

function LessonView({lesson, ctx, number, completed, onComplete, onBack, onNext, last, onReset, onExit}: LessonViewProps) {
  const [start] = useState(lesson.start);
  const [done, setDone] = useState<ReadonlySet<string>>(() => new Set());
  const [bagOpen, setBagOpen] = useState(false);
  const [keyOpen, setKeyOpen] = useState(false);

  const report = useCallback((event: LessonEvent) => {
    const ids = tasksDoneBy(lesson, event);
    if (ids.length) setDone(d => new Set([...d, ...ids]));
  }, [lesson]);

  const game = useGame('hotseat', 'P2', {
    evaluate: false,
    lesson: {
      initial: start.state,
      lastMove: start.lastMove,
      play: (state, move) => {
        const next = playLesson(lesson, state, move);
        report({kind: 'move', prev: state, move, next});
        return next;
      },
    },
  });
  const {state} = game;

  // Looking at tiles counts too (the "Reading tiles" lesson)
  const looked = game.selectedTileInstance;
  const lookedId = looked?.id;
  useEffect(() => {
    if (looked) report({kind: 'look', name: looked.defName, owner: looked.owner, from: 'board'});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lookedId, report]);
  const bagTile = game.viewingBagTile;
  const bagKey = bagTile ? `${bagTile.owner}-${bagTile.name}` : null;
  useEffect(() => {
    if (bagTile) report({kind: 'look', name: bagTile.name, owner: bagTile.owner, from: 'bag'});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bagKey, report]);

  const allDone = lesson.tasks.every(t => done.has(t.id));
  useEffect(() => {
    if (allDone) onComplete(lesson.id);
  }, [allDone, lesson.id, onComplete]);

  // As in a game: Esc closes the drawer, then the inspector; Ctrl/Cmd+Z undoes
  const {clearSelection, undo, canUndo} = game;
  const viewingBag = game.viewingBagTile !== null;
  const closeBag = useCallback(() => {
    setBagOpen(false);
    if (viewingBag) clearSelection();
  }, [viewingBag, clearSelection]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (document.querySelector('dialog[open]')) return;
      if (e.key === 'Escape') { if (bagOpen) closeBag(); else clearSelection(); }
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && canUndo) { e.preventDefault(); undo(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [bagOpen, closeBag, clearSelection, undo, canUndo]);

  const inspecting = inspectorSubject(game) !== null;
  const hasBags = state.bags.P1.length + state.bags.P2.length > 0;
  const titleId = `lesson-${lesson.id}`;
  const nextLabel = last ? 'Play a game' : allDone ? 'Next' : 'Skip';

  return (
    <main className={`layout tutorial-layout ${inspecting ? 'inspecting' : ''}`}>
      <section className="card lesson area-lesson" aria-labelledby={titleId}>
        <header className="lesson-head">
          <h2 id={titleId}>
            <span className="lesson-count">{number}/{LESSONS.length}</span>
            <span className="lesson-title">{lesson.title}</span>
          </h2>
          <span className="lesson-nav">
            <button type="button" className="icon-btn" onClick={onBack} disabled={!onBack} aria-label="Previous lesson">
              <ChevronLeftIcon />
            </button>
            {lesson.tasks.length > 0 && (
              <button type="button" className="icon-btn" onClick={onReset} aria-label="Start this lesson again" title="Start this lesson again">
                <RematchIcon />
              </button>
            )}
            <button type="button" className={`btn lesson-next ${allDone ? 'btn-gold' : ''}`} onClick={onNext}
                    aria-label={nextLabel} title={allDone ? undefined : 'Skip this lesson'}>
              <span className="lesson-next-label">{nextLabel}</span>
              {!last && <ChevronRightIcon />}
            </button>
            <button type="button" className="icon-btn lesson-exit" onClick={onExit} aria-label="Exit tutorial">
              <CloseIcon />
            </button>
          </span>
        </header>

        <div className="lesson-body">
          {lesson.body(ctx).map((p, i) => <p key={i}>{rich(p)}</p>)}
        </div>

        {lesson.tasks.length > 0 && (
          <footer className="lesson-foot" aria-live="polite">
            {allDone && lesson.success ? (
              <p className="lesson-success"><CheckIcon /> {lesson.success(ctx)}</p>
            ) : (
              <ul className="tasks">
                {lesson.tasks.map(t => (
                  <li key={t.id} className={done.has(t.id) ? 'task-done' : ''}>
                    <span className="task-box" aria-hidden>{done.has(t.id) && <CheckIcon />}</span>
                    <span>{t.label(ctx)}</span>
                    {done.has(t.id) && <span className="visually-hidden"> (done)</span>}
                  </li>
                ))}
              </ul>
            )}
          </footer>
        )}

        <ol className="lesson-dots" aria-label="Lessons">
          {LESSONS.map((l, i) => (
            <li key={l.id} className={`${i + 1 === number ? 'on' : ''} ${completed.has(l.id) ? 'done' : ''}`}
                title={l.title} aria-current={i + 1 === number ? 'step' : undefined} />
          ))}
        </ol>
      </section>

      <div className="area-board">
        <div className="board-wrap">
          <Board
            state={state}
            bottom="P1"
            humanTurn={game.humanTurn}
            selectedTile={game.selectedTile}
            legalMoves={game.legalMoves}
            onCellClick={game.onCellClick}
            drawMode={game.drawMode}
            placementTargets={game.placementTargets}
            commandTarget={game.commandTarget}
            commandDestinations={game.commandDestinations}
            commandChoice={game.commandChoice}
            onCommandChoice={game.resolveCommandChoice}
            setupTargets={game.setupTargets}
            inCheck={game.inCheck}
            viewingEnemyMoves={game.viewingEnemyMoves}
            allFriendlyTargets={game.allFriendlyTargets}
            allEnemyTargets={game.allEnemyTargets}
            lastMove={game.lastMove}
          />
        </div>
      </div>

      <div className="area-actions">
        <ActionBar game={game} />
      </div>

      <aside className="area-panel">
        {bagOpen ? <BagDrawer game={game} bottom="P1" onClose={closeBag} /> : (
          <>
            {inspecting ? <Inspector game={game} bottom="P1" /> : (
              <section className="card hint-card">
                <p>{ctx.touch ? 'Tap' : 'Click'} a tile to see how it moves. Tiles flip after every action, so each has two sides.</p>
                <Legend onOpen={() => setKeyOpen(true)} />
              </section>
            )}
            {hasBags && <BagStrip game={game} bottom="P1" onOpen={() => setBagOpen(true)} />}
          </>
        )}
      </aside>

      <SymbolKey open={keyOpen} onClose={() => setKeyOpen(false)} />
    </main>
  );
}

const MOVE_ICONS: ReadonlySet<string> = new Set<MoveType>(['step', 'slide', 'jump', 'jump_slide', 'strike', 'command']);

/** A lesson paragraph: `**bold**`, and `[step]` etc. for a move symbol or `[info]` for the info button. */
function rich(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|\[[a-z_]+\])/).filter(Boolean).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>;
    const symbol = /^\[([a-z_]+)\]$/.exec(part)?.[1];
    if (symbol && MOVE_ICONS.has(symbol)) return <LegendIcon key={i} type={symbol as MoveType} />;
    if (symbol === 'info') return <span key={i} className="inline-icon"><InfoIcon /></span>;
    return part;
  });
}
