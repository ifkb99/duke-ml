import {useCallback, useEffect, useState} from 'react';
import type {Player} from '@the-duke/engine';
import {ActionBar} from './components/ActionBar.js';
import {AppUpdate} from './components/AppUpdate.js';
import {BagDrawer, BagStrip} from './components/Bags.js';
import {Board} from './components/Board.js';
import {GameOver, NewGameDialog} from './components/Dialogs.js';
import {EvalBar} from './components/EvalBar.js';
import {PlusIcon} from './components/Icons.js';
import {Inspector, Legend, SymbolKey, inspectorSubject} from './components/Inspector.js';
import {StatusLine} from './components/StatusLine.js';
import {Crown} from './components/Tile.js';
import {useGame} from './hooks/useGame.js';
import {type GameMode, MODE_NAME, PLAYER_NAME, endReason, other} from './ui.js';
import './components/controls.css';

interface Settings {
  mode: GameMode;
  /** The person's side against an AI */
  human: Player;
  showEval: boolean;
}

const DEFAULTS: Settings = {mode: 'vs-zero', human: 'P1', showEval: true};
const STORAGE_KEY = 'duke-settings';

function loadSettings(): Settings {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    return {...DEFAULTS, ...saved};
  } catch {
    return DEFAULTS;
  }
}

export function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [newGameOpen, setNewGameOpen] = useState(false);
  const [resultDismissed, setResultDismissed] = useState(false);
  const [bagOpen, setBagOpen] = useState(false);
  const [keyOpen, setKeyOpen] = useState(false);
  const {mode, human, showEval} = settings;
  const game = useGame(mode, other(human), {evaluate: showEval});
  const {state} = game;

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(settings)); } catch { /* private mode */ }
  }, [settings]);

  useEffect(() => setResultDismissed(false), [state]);

  // Closing the bag drawer also drops the bag tile it was showing
  const {clearSelection, undo, canUndo} = game;
  const viewingBag = game.viewingBagTile !== null;
  const closeBag = useCallback(() => {
    setBagOpen(false);
    if (viewingBag) clearSelection();
  }, [viewingBag, clearSelection]);

  // Esc closes the drawer, then the inspector; Ctrl/Cmd+Z undoes
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (document.querySelector('dialog[open]')) return;
      if (e.key === 'Escape') { if (bagOpen) closeBag(); else clearSelection(); }
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && canUndo) { e.preventDefault(); undo(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [bagOpen, closeBag, clearSelection, undo, canUndo]);

  const humanSide = mode === 'hotseat' ? null : human;
  const bottom: Player = humanSide ?? 'P1';
  const over = state.status === 'P1_wins' || state.status === 'P2_wins';
  const inspecting = inspectorSubject(game) !== null;

  const start = (m: GameMode, h: Player) => {
    setSettings(s => ({...s, mode: m, human: h}));
    setNewGameOpen(false);
    game.newGame();
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Crown className="crown brand-crown" />
          <h1>The Duke</h1>
        </div>
        <span className="matchup">
          {mode === 'hotseat' ? 'Hot seat' : <>You play <b className={human === 'P1' ? 'text-light' : 'text-dark'}>{PLAYER_NAME[human]}</b> vs {MODE_NAME[mode]}</>}
        </span>
        <button type="button" className="btn" onClick={() => setNewGameOpen(true)}>
          <PlusIcon /> <span className="btn-label">New game</span>
        </button>
      </header>

      <main className={`layout ${inspecting ? 'inspecting' : ''} ${showEval ? 'with-eval' : ''}`}>
        <div className="area-status">
          <StatusLine game={game} mode={mode} human={humanSide} />
          <button type="button" className="btn status-new" onClick={() => setNewGameOpen(true)} aria-label="New game">
            <PlusIcon />
          </button>
        </div>

        <div className="area-board">
          {showEval && <EvalBar value={game.evaluation} bottom={bottom} />}
          <div className="board-wrap">
            <Board
              state={state}
              bottom={bottom}
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
            {over && !resultDismissed && (
              <GameOver winner={state.status === 'P1_wins' ? 'P1' : 'P2'} reason={endReason(state)} human={humanSide}
                        mode={mode} onRematch={game.newGame} onNewGame={() => setNewGameOpen(true)}
                        onDismiss={() => setResultDismissed(true)} />
            )}
          </div>
        </div>

        <div className="area-actions">
          <ActionBar game={game} showEval={showEval}
                     onToggleEval={() => setSettings(s => ({...s, showEval: !s.showEval}))} />
        </div>

        <aside className="area-panel">
          {bagOpen ? <BagDrawer game={game} bottom={bottom} onClose={closeBag} /> : (
            <>
              {inspecting ? <Inspector game={game} bottom={bottom} /> : (
                <section className="card hint-card">
                  <p>Tap a tile to see how it moves. Tiles flip after every action, so each has two sides.</p>
                  <Legend onOpen={() => setKeyOpen(true)} />
                </section>
              )}
              <BagStrip game={game} bottom={bottom} onOpen={() => setBagOpen(true)} />
            </>
          )}
        </aside>
      </main>

      <SymbolKey open={keyOpen} onClose={() => setKeyOpen(false)} />
      <NewGameDialog open={newGameOpen} mode={mode} human={human} onClose={() => setNewGameOpen(false)} onStart={start} />
      <AppUpdate />
    </div>
  );
}
