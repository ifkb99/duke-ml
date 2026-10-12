import {useState, useCallback, useEffect, useRef, useMemo} from 'react';
import type {Coord, GameMove, GameState, Player, SerializedGameState, SetupPhase, TileInstance} from '@the-duke/engine';
import {
  createInitialState, generateAllMoves, generateTileMoves, applyMove,
  getSetupTargets, getStandardSetupTargets, applySetupPlacement, isPlayerInCheck, serialize, deserialize,
} from '@the-duke/engine';
import type {AIWorkerRequest, AIWorkerResponse} from '../workers/ai.worker.js';
import {type GameMode, sameCoord} from '../ui.js';

const SETUP_STEP: Record<SetupPhase, string> = {
  p1_duke: 'Place your Duke on your back row',
  p1_footman1: 'Place a Footman next to your Duke',
  p1_footman2: 'Place a second Footman next to your Duke',
  p2_duke: 'Place your Duke on your back row',
  p2_footman1: 'Place a Footman next to your Duke',
  p2_footman2: 'Place a second Footman next to your Duke',
};

/** The move that produced the current position (setup placements included). */
export type LastMove = GameMove | {type: 'setup'; position: Coord};

/** A friendly tile that the selected tile could command, or that could simply be selected. */
export interface CommandChoice {coord: Coord; tileName: string}

interface HistoryEntry {state: GameState; lastMove: LastMove | null}

// The game in progress is kept in localStorage: updating the app reloads the
// page, and phones drop apps in the background. App.tsx keeps the mode and the
// person's side the same way.
const SAVE_KEY = 'duke-game';
/** Bump when a build can't continue games saved by the one before */
const SAVE_VERSION = 1;
interface SavedEntry {state: SerializedGameState; lastMove: LastMove | null}
interface SavedGame extends SavedEntry {v: number; history: SavedEntry[]}

function loadGame(): (HistoryEntry & {history: HistoryEntry[]}) | null {
  try {
    const saved: SavedGame | null = JSON.parse(localStorage.getItem(SAVE_KEY) ?? 'null');
    if (saved?.v !== SAVE_VERSION) return null;
    const entry = (e: SavedEntry): HistoryEntry => ({state: deserialize(e.state), lastMove: e.lastMove});
    return {...entry(saved), history: saved.history.map(entry)};
  } catch {
    return null;
  }
}

function saveGame(state: GameState, lastMove: LastMove | null, history: HistoryEntry[]) {
  const entry = (e: HistoryEntry): SavedEntry => ({state: serialize(e.state), lastMove: e.lastMove});
  const saved: SavedGame = {v: SAVE_VERSION, ...entry({state, lastMove}), history: history.map(entry)};
  try { localStorage.setItem(SAVE_KEY, JSON.stringify(saved)); } catch { /* private mode, or full */ }
}

const params = new URLSearchParams(location.search);
const workerFlags = {
  report: params.has('report'),
  backend: params.get('backend') ?? undefined,
  threads: Number(params.get('threads')) || undefined,
};

export interface UseGameReturn {
  state: GameState;
  /** It's a person's turn (always, in hot seat), so the board takes clicks */
  humanTurn: boolean;
  thinking: boolean;
  aiError: string | null;
  lastMove: LastMove | null;
  /** The network's estimate for Light, in [-1, 1]; null when unknown */
  evaluation: number | null;
  selectedTile: Coord | null;
  selectedTileInstance: TileInstance | null;
  viewingBagTile: {name: string; owner: Player} | null;
  legalMoves: GameMove[];
  onCellClick: (coord: Coord) => void;
  onBagTileClick: (name: string, owner: Player) => void;
  clearSelection: () => void;
  newGame: () => void;
  undo: () => void;
  canUndo: boolean;
  canDraw: boolean;
  drawMode: boolean;
  selectedDrawTile: string | null;
  placementTargets: Coord[];
  startDraw: () => void;
  commandTarget: Coord | null;
  commandDestinations: Coord[];
  commandChoice: CommandChoice | null;
  resolveCommandChoice: (choice: 'command' | 'select' | null) => void;
  setupTargets: Coord[];
  setupStep: string | null;
  inCheck: Player | null;
  /** Pseudo-legal moves of a selected enemy tile (for viewing only) */
  viewingEnemyMoves: GameMove[];
  showAllMoves: boolean;
  toggleShowAllMoves: () => void;
  allFriendlyTargets: Coord[];
  allEnemyTargets: Coord[];
}

export function useGame(mode: GameMode, aiPlayer: Player, opts: {evaluate: boolean}): UseGameReturn {
  const [saved] = useState(loadGame);
  const [state, setState] = useState<GameState>(() => saved?.state ?? createInitialState());
  const [lastMove, setLastMove] = useState<LastMove | null>(saved?.lastMove ?? null);
  const [history, setHistory] = useState<HistoryEntry[]>(saved?.history ?? []);
  const [selectedTile, setSelectedTile] = useState<Coord | null>(null);
  const [viewingBagTile, setViewingBagTile] = useState<{name: string; owner: Player} | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [selectedDrawTile, setSelectedDrawTile] = useState<string | null>(null);
  const [commandTarget, setCommandTarget] = useState<Coord | null>(null);
  const [commandChoice, setCommandChoice] = useState<CommandChoice | null>(null);
  const [showAllMoves, setShowAllMoves] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<number | null>(null);

  useEffect(() => saveGame(state, lastMove, history), [state, lastMove, history]);

  const isSetup = state.status === 'setup';
  const isAiTurn = mode !== 'hotseat' && state.currentPlayer === aiPlayer;
  const humanTurn = !isAiTurn && (isSetup || state.status === 'active');
  const allMoves = useMemo(() => (isSetup ? [] : generateAllMoves(state)), [state, isSetup]);

  const setupTargets = useMemo(() => (isSetup ? getSetupTargets(state) : []), [state, isSetup]);
  const setupStep = isSetup && state.setupPhase ? SETUP_STEP[state.setupPhase] : null;

  const selectedTileInstance = useMemo<TileInstance | null>(() => {
    if (!selectedTile) return null;
    const id = state.board[selectedTile.row][selectedTile.col];
    return id ? state.tiles.get(id) ?? null : null;
  }, [selectedTile, state]);

  const placementTargets = useMemo<Coord[]>(() => {
    if (!drawMode || !selectedDrawTile) return [];
    // Draw squares don't depend on the tile: the revealed tile goes on whichever one the player picks.
    const targets: Coord[] = [];
    for (const m of allMoves) {
      if (m.type === 'draw' && !targets.some(t => sameCoord(t, m.position))) targets.push(m.position);
    }
    return targets;
  }, [drawMode, selectedDrawTile, allMoves]);

  const commandDestinations = useMemo<Coord[]>(() => {
    if (!selectedTile || !commandTarget) return [];
    const dests: Coord[] = [];
    for (const m of allMoves) {
      if (m.type !== 'command' || !sameCoord(m.commander, selectedTile) || !sameCoord(m.target, commandTarget)) continue;
      if (!dests.some(d => sameCoord(d, m.targetTo))) dests.push(m.targetTo);
    }
    return dests;
  }, [selectedTile, commandTarget, allMoves]);

  const inCheck = useMemo<Player | null>(() => {
    if (state.status !== 'active') return null;
    if (isPlayerInCheck(state, 'P1')) return 'P1';
    if (isPlayerInCheck(state, 'P2')) return 'P2';
    return null;
  }, [state]);

  const viewingEnemyMoves = useMemo<GameMove[]>(() => {
    if (!selectedTile || drawMode || isSetup) return [];
    const id = state.board[selectedTile.row][selectedTile.col];
    const tile = id ? state.tiles.get(id) : undefined;
    if (!tile || tile.owner === state.currentPlayer) return [];
    return generateTileMoves(state, tile);
  }, [selectedTile, state, drawMode, isSetup]);

  // Threats overlay: every square each side's tiles could reach
  const {allFriendlyTargets, allEnemyTargets} = useMemo(() => {
    const friendly: Coord[] = [];
    const enemy: Coord[] = [];
    if (!showAllMoves || state.status !== 'active') return {allFriendlyTargets: friendly, allEnemyTargets: enemy};
    for (const tile of state.tiles.values()) {
      const coords = tile.owner === state.currentPlayer ? friendly : enemy;
      for (const m of generateTileMoves(state, tile)) {
        const target = m.type === 'move' ? m.to : m.type === 'strike' ? m.target : m.type === 'command' ? m.targetTo : null;
        if (target && !coords.some(c => sameCoord(c, target))) coords.push(target);
      }
    }
    return {allFriendlyTargets: friendly, allEnemyTargets: enemy};
  }, [showAllMoves, state]);

  const toggleShowAllMoves = useCallback(() => setShowAllMoves(prev => !prev), []);

  const clearSelection = useCallback(() => {
    setSelectedTile(null);
    setViewingBagTile(null);
    setCommandTarget(null);
    setCommandChoice(null);
  }, []);

  const resetSelection = useCallback(() => {
    clearSelection();
    setDrawMode(false);
    setSelectedDrawTile(null);
  }, [clearSelection]);

  const advance = useCallback((from: GameState, next: GameState, move: LastMove) => {
    setHistory(prev => [...prev, {state: from, lastMove}]);
    setState(next);
    setLastMove(move);
  }, [lastMove]);

  const newGame = useCallback(() => {
    setState(createInitialState());
    setLastMove(null);
    setHistory([]);
    setEvaluation(null);
    setAiError(null);
    setShowAllMoves(false);
    resetSelection();
  }, [resetSelection]);

  const undo = useCallback(() => {
    if (history.length === 0) return;
    // Against the AI, undo back to the person's previous turn
    let steps = 1;
    if (mode !== 'hotseat') {
      while (steps < history.length && history[history.length - steps].state.currentPlayer === aiPlayer) steps++;
    }
    const target = history[history.length - steps];
    setState(target.state);
    setLastMove(target.lastMove);
    setHistory(prev => prev.slice(0, -steps));
    setAiError(null);
    resetSelection();
  }, [history, mode, aiPlayer, resetSelection]);

  const executeMove = useCallback((move: GameMove) => {
    advance(state, applyMove(state, move), move);
    resetSelection();
  }, [state, advance, resetSelection]);

  const executeSetup = useCallback((coord: Coord) => {
    advance(state, applySetupPlacement(state, coord), {type: 'setup', position: coord});
    resetSelection();
  }, [state, advance, resetSelection]);

  const canDraw = humanTurn && !isSetup && !drawMode
    && state.bags[state.currentPlayer].length > 0 && allMoves.some(m => m.type === 'draw');

  const startDraw = useCallback(() => {
    if (!canDraw) return;
    const bag = state.bags[state.currentPlayer];
    clearSelection();
    setDrawMode(true);
    setSelectedDrawTile(bag[Math.floor(Math.random() * bag.length)]);
  }, [state, canDraw, clearSelection]);

  const onBagTileClick = useCallback((name: string, owner: Player) => {
    if (drawMode) return;
    clearSelection();
    setViewingBagTile({name, owner});
  }, [drawMode, clearSelection]);

  // ── AI worker ──
  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);
  const [workerReady, setWorkerReady] = useState(false);
  useEffect(() => {
    const worker = new Worker(new URL('../workers/ai.worker.ts', import.meta.url), {type: 'module'});
    workerRef.current = worker;
    setWorkerReady(true);
    return () => {
      worker.terminate();
      workerRef.current = null;
      setWorkerReady(false);
    };
  }, []);

  /** Posts a request; the reply comes back to onReply only if no newer request was sent since. */
  const ask = useCallback((req: Omit<AIWorkerRequest, 'id'>, onReply: (r: AIWorkerResponse) => void) => {
    const worker = workerRef.current;
    if (!worker) return () => {};
    const id = ++requestId.current;
    const onMessage = (e: MessageEvent<AIWorkerResponse>) => {
      if (e.data.id !== id) return;
      worker.removeEventListener('message', onMessage);
      onReply(e.data);
    };
    worker.addEventListener('message', onMessage);
    worker.postMessage({...req, id} satisfies AIWorkerRequest);
    return () => worker.removeEventListener('message', onMessage);
  }, []);

  // Evaluation bar: the network's quick value of every new position. Declared
  // before the AI turn so its request is queued first; the AI's own search value
  // then replaces it.
  useEffect(() => {
    if (!opts.evaluate || !workerReady) return;
    if (state.status === 'P1_wins' || state.status === 'P2_wins') {
      setEvaluation(state.status === 'P1_wins' ? 1 : -1);
      return;
    }
    if (state.status !== 'active') {
      setEvaluation(null);
      return;
    }
    const mover = state.currentPlayer;
    return ask({type: 'zeroEvaluate', state: serialize(state), backend: workerFlags.backend, threads: workerFlags.threads},
      r => { if (!r.error) setEvaluation(mover === 'P1' ? r.score : -r.score); });
  }, [state, opts.evaluate, workerReady, ask]);

  // AI turn, setup included
  useEffect(() => {
    if (!isAiTurn || !workerReady) return;

    if (isSetup && setupTargets.length > 0) {
      // A random square of the official setup (Duke in the center): the only starts the network trained on
      const timer = setTimeout(() => {
        const targets = getStandardSetupTargets(state);
        const pick = targets[Math.floor(Math.random() * targets.length)];
        advance(state, applySetupPlacement(state, pick), {type: 'setup', position: pick});
      }, 350);
      return () => clearTimeout(timer);
    }
    if (state.status !== 'active') return;

    setThinking(true);
    setAiError(null);
    const started = performance.now();
    let delay: ReturnType<typeof setTimeout> | undefined;
    const cancel = ask({
      type: mode === 'vs-random' ? 'pickRandomMove' : mode === 'vs-zero' ? 'zeroMove' : 'findBestMove',
      state: serialize(state),
      depth: mode === 'vs-minimax' ? 6 : undefined,
      budgetMs: mode === 'vs-zero' ? 3000 : undefined,   // thinking time per decision
      ...(mode === 'vs-zero' ? workerFlags : {}),
    }, ({move, score, error}) => {
      // Instant replies (random) still wait a beat, so the move can be followed
      delay = setTimeout(() => {
        setThinking(false);
        if (error || !move) {
          setAiError(error ?? 'The AI found no move');
          return;
        }
        if (mode === 'vs-zero' && opts.evaluate) setEvaluation(aiPlayer === 'P1' ? score : -score);
        advance(state, applyMove(state, move), move);
      }, Math.max(0, 400 - (performance.now() - started)));
    });
    return () => {
      cancel();
      clearTimeout(delay);
      setThinking(false);
    };
    // advance changes with lastMove, which changes only with state
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, mode, aiPlayer, isAiTurn, isSetup, setupTargets, workerReady, ask]);

  const findMove = (from: Coord, to: Coord) => allMoves.find(m =>
    (m.type === 'move' && sameCoord(m.from, from) && sameCoord(m.to, to))
    || (m.type === 'strike' && sameCoord(m.from, from) && sameCoord(m.target, to)));

  const onCellClick = useCallback((coord: Coord) => {
    setCommandChoice(null);
    if (!humanTurn) {
      // Still let people look at tiles while the AI thinks
      if (!isSetup) {
        setViewingBagTile(null);
        setSelectedTile(sel => (sameCoord(sel, coord) || !state.board[coord.row][coord.col] ? null : coord));
      }
      return;
    }

    if (isSetup) {
      if (setupTargets.some(t => sameCoord(t, coord))) executeSetup(coord);
      return;
    }

    const clickedId = state.board[coord.row][coord.col];
    const clickedTile = clickedId ? state.tiles.get(clickedId) ?? null : null;

    // Drawing: the tile is already revealed; place it on any legal draw square.
    if (drawMode && selectedDrawTile) {
      if (placementTargets.some(t => sameCoord(t, coord))) {
        executeMove({type: 'place', tileName: selectedDrawTile, position: coord});
      }
      return;
    }

    setViewingBagTile(null);

    // Command, step 2: where the commanded tile goes
    if (selectedTile && commandTarget) {
      const move = allMoves.find(m => m.type === 'command' && sameCoord(m.commander, selectedTile)
        && sameCoord(m.target, commandTarget) && sameCoord(m.targetTo, coord));
      if (move) {
        executeMove(move);
        return;
      }
      setCommandTarget(null);
      setSelectedTile(clickedTile && clickedTile.owner === state.currentPlayer ? coord : null);
      return;
    }

    if (selectedTile) {
      const move = findMove(selectedTile, coord);
      if (move) {
        executeMove(move);
        return;
      }
      // A friendly tile in command range: command it, or just select it? Ask.
      const commandable = allMoves.some(m => m.type === 'command'
        && sameCoord(m.commander, selectedTile) && sameCoord(m.target, coord));
      if (commandable && clickedTile) {
        setCommandChoice({coord, tileName: clickedTile.defName});
        return;
      }
      if (sameCoord(selectedTile, coord)) {
        setSelectedTile(null);
        return;
      }
    }

    setCommandTarget(null);
    setSelectedTile(clickedTile ? coord : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, humanTurn, selectedTile, commandTarget, allMoves, drawMode, selectedDrawTile, placementTargets,
      executeMove, executeSetup, isSetup, setupTargets]);

  const resolveCommandChoice = useCallback((choice: 'command' | 'select' | null) => {
    if (commandChoice && choice === 'command') setCommandTarget(commandChoice.coord);
    if (commandChoice && choice === 'select') setSelectedTile(commandChoice.coord);
    setCommandChoice(null);
  }, [commandChoice]);

  return {
    state,
    humanTurn,
    thinking,
    aiError,
    lastMove,
    evaluation: opts.evaluate ? evaluation : null,
    selectedTile,
    selectedTileInstance,
    viewingBagTile,
    legalMoves: allMoves,
    onCellClick,
    onBagTileClick,
    clearSelection,
    newGame,
    undo,
    canUndo: history.length > 0 && !(mode !== 'hotseat' && history.every(h => h.state.currentPlayer === aiPlayer)),
    canDraw,
    drawMode,
    selectedDrawTile,
    placementTargets,
    startDraw,
    commandTarget,
    commandDestinations,
    commandChoice,
    resolveCommandChoice,
    setupTargets,
    setupStep,
    inCheck,
    viewingEnemyMoves,
    showAllMoves,
    toggleShowAllMoves,
    allFriendlyTargets,
    allEnemyTargets,
  };
}
