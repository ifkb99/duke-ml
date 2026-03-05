import {useState, useCallback, useEffect, useRef, useMemo} from 'react';
import type {Coord, GameMove, GameState, Player, SetupPhase, TileInstance} from '@the-duke/engine';
import {
  createInitialState, generateAllMoves, generateTileMoves, applyMove,
  getSetupTargets, applySetupPlacement, isPlayerInCheck, serialize,
} from '@the-duke/engine';
import type {AIWorkerRequest, AIWorkerResponse} from '../workers/ai.worker.js';
import type {GameMode} from '../App.js';

const SETUP_LABELS: Record<SetupPhase, string> = {
  p1_duke: 'Light: Place your Duke on the back row',
  p1_footman1: 'Light: Place a Footman next to the Duke',
  p1_footman2: 'Light: Place a Footman next to the Duke',
  p2_duke: 'Dark: Place your Duke on the back row',
  p2_footman1: 'Dark: Place a Footman next to the Duke',
  p2_footman2: 'Dark: Place a Footman next to the Duke',
};

export interface UseGameReturn {
  state: GameState;
  selectedTile: Coord | null;
  selectedTileInstance: TileInstance | null;
  viewingBagTile: {name: string; owner: Player} | null;
  legalMoves: GameMove[];
  onCellClick: (coord: Coord) => void;
  onBagTileClick: (name: string, owner: Player) => void;
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
  /** Setup phase info */
  setupTargets: Coord[];
  setupLabel: string | null;
  /** Check indicator */
  inCheck: Player | null;
  /** Pseudo-legal moves for a clicked enemy tile (for viewing only) */
  viewingEnemyMoves: GameMove[];
  /** Show-all-moves toggle */
  showAllMoves: boolean;
  toggleShowAllMoves: () => void;
  allFriendlyTargets: Coord[];
  allEnemyTargets: Coord[];
}

export function useGame(mode: GameMode, aiPlayer: Player = 'P2'): UseGameReturn {
  const [state, setState] = useState<GameState>(() => createInitialState());
  const [history, setHistory] = useState<GameState[]>([]);
  const [selectedTile, setSelectedTile] = useState<Coord | null>(null);
  const [viewingBagTile, setViewingBagTile] = useState<{name: string; owner: Player} | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [selectedDrawTile, setSelectedDrawTile] = useState<string | null>(null);
  const [commandTarget, setCommandTarget] = useState<Coord | null>(null);
  const [showAllMoves, setShowAllMoves] = useState(false);
  const aiThinking = useRef(false);

  const isSetup = state.status === 'setup';
  const allMoves = isSetup ? [] : generateAllMoves(state);

  const setupTargets = useMemo(() => {
    return isSetup ? getSetupTargets(state) : [];
  }, [state, isSetup]);

  const setupLabel = isSetup && state.setupPhase ? SETUP_LABELS[state.setupPhase] : null;

  const selectedTileInstance = useMemo<TileInstance | null>(() => {
    if (!selectedTile) return null;
    const id = state.board[selectedTile.row][selectedTile.col];
    return id ? state.tiles.get(id) ?? null : null;
  }, [selectedTile, state]);

  const placementTargets = useMemo<Coord[]>(() => {
    if (!drawMode || !selectedDrawTile) return [];
    const seen = new Set<string>();
    const targets: Coord[] = [];
    for (const m of allMoves) {
      if (m.type !== 'place') continue;
      if (m.tileName !== selectedDrawTile) continue;
      const key = `${m.position.row},${m.position.col}`;
      if (!seen.has(key)) {
        seen.add(key);
        targets.push(m.position);
      }
    }
    return targets;
  }, [drawMode, selectedDrawTile, allMoves]);

  const commandDestinations = useMemo<Coord[]>(() => {
    if (!selectedTile || !commandTarget) return [];
    const seen = new Set<string>();
    const dests: Coord[] = [];
    for (const m of allMoves) {
      if (m.type !== 'command') continue;
      if (m.commander.row !== selectedTile.row || m.commander.col !== selectedTile.col) continue;
      if (m.target.row !== commandTarget.row || m.target.col !== commandTarget.col) continue;
      const key = `${m.targetTo.row},${m.targetTo.col}`;
      if (!seen.has(key)) {
        seen.add(key);
        dests.push(m.targetTo);
      }
    }
    return dests;
  }, [selectedTile, commandTarget, allMoves]);

  // Check detection
  const inCheck = useMemo<Player | null>(() => {
    if (isSetup || state.status !== 'active') return null;
    if (isPlayerInCheck(state, 'P1')) return 'P1';
    if (isPlayerInCheck(state, 'P2')) return 'P2';
    return null;
  }, [state, isSetup]);

  // Enemy tile move viewing (pseudo-legal moves for display)
  const viewingEnemyMoves = useMemo<GameMove[]>(() => {
    if (!selectedTile || drawMode || isSetup) return [];
    const id = state.board[selectedTile.row][selectedTile.col];
    if (!id) return [];
    const tile = state.tiles.get(id);
    if (!tile || tile.owner === state.currentPlayer) return [];
    return generateTileMoves(state, tile);
  }, [selectedTile, state, drawMode, isSetup]);

  // Show-all-moves: compute all squares targeted by friendly and enemy pieces
  const {allFriendlyTargets, allEnemyTargets} = useMemo(() => {
    if (!showAllMoves || isSetup || state.status !== 'active') {
      return {allFriendlyTargets: [] as Coord[], allEnemyTargets: [] as Coord[]};
    }
    const player = state.currentPlayer;
    const opponent = player === 'P1' ? 'P2' : 'P1';
    const friendlySet = new Set<string>();
    const enemySet = new Set<string>();
    const friendlyCoords: Coord[] = [];
    const enemyCoords: Coord[] = [];

    for (const tile of state.tiles.values()) {
      const moves = generateTileMoves(state, tile);
      const isOwner = tile.owner === player;
      const set = isOwner ? friendlySet : enemySet;
      const coords = isOwner ? friendlyCoords : enemyCoords;
      for (const m of moves) {
        let target: Coord | null = null;
        if (m.type === 'move') target = m.to;
        else if (m.type === 'strike') target = m.target;
        else if (m.type === 'command') target = m.targetTo;
        if (target) {
          const key = `${target.row},${target.col}`;
          if (!set.has(key)) {
            set.add(key);
            coords.push(target);
          }
        }
      }
    }
    return {allFriendlyTargets: friendlyCoords, allEnemyTargets: enemyCoords};
  }, [showAllMoves, state, isSetup]);

  const toggleShowAllMoves = useCallback(() => {
    setShowAllMoves(prev => !prev);
  }, []);

  const resetSelection = useCallback(() => {
    setSelectedTile(null);
    setViewingBagTile(null);
    setDrawMode(false);
    setSelectedDrawTile(null);
    setCommandTarget(null);
    setShowAllMoves(false);
  }, []);

  const newGame = useCallback(() => {
    setState(createInitialState());
    setHistory([]);
    resetSelection();
    aiThinking.current = false;
  }, [resetSelection]);

  const undo = useCallback(() => {
    if (history.length === 0) return;
    const stepsBack = mode !== 'hotseat' && history.length >= 2 ? 2 : 1;
    const target = history[history.length - stepsBack];
    setState(target);
    setHistory(prev => prev.slice(0, -stepsBack));
    resetSelection();
  }, [history, mode, resetSelection]);

  const executeMove = useCallback((move: GameMove) => {
    setHistory(prev => [...prev, state]);
    setState(applyMove(state, move));
    resetSelection();
  }, [state, resetSelection]);

  const executeSetup = useCallback((coord: Coord) => {
    setHistory(prev => [...prev, state]);
    setState(applySetupPlacement(state, coord));
    resetSelection();
  }, [state, resetSelection]);

  const hasPlacementMoves = allMoves.some(m => m.type === 'place');
  const canDraw = !isSetup && state.bags[state.currentPlayer].length > 0 && hasPlacementMoves;

  const startDraw = useCallback(() => {
    if (!canDraw) return;
    const bag = state.bags[state.currentPlayer];
    const drawn = bag[Math.floor(Math.random() * bag.length)];
    setDrawMode(true);
    setSelectedDrawTile(drawn);
    setSelectedTile(null);
    setViewingBagTile(null);
    setCommandTarget(null);
  }, [state, canDraw]);

  const onBagTileClick = useCallback((name: string, owner: Player) => {
    if (drawMode || isSetup) return;
    setSelectedTile(null);
    setCommandTarget(null);
    setViewingBagTile({name, owner});
  }, [drawMode, isSetup]);

  // Web Worker for AI
  const workerRef = useRef<Worker | null>(null);
  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/ai.worker.ts', import.meta.url),
      {type: 'module'},
    );
    workerRef.current = worker;
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // AI turn logic — covers both setup and normal play
  useEffect(() => {
    if (mode === 'hotseat') return;
    if (state.currentPlayer !== aiPlayer) return;
    if (aiThinking.current) return;

    // Setup phase AI — trivial, stays on main thread
    if (isSetup && setupTargets.length > 0) {
      aiThinking.current = true;
      const timer = setTimeout(() => {
        const pick = setupTargets[Math.floor(Math.random() * setupTargets.length)];
        setHistory(prev => [...prev, state]);
        setState(applySetupPlacement(state, pick));
        aiThinking.current = false;
      }, 300);
      return () => {clearTimeout(timer); aiThinking.current = false;};
    }

    // Normal play AI — offload to worker
    if (state.status !== 'active') return;
    const worker = workerRef.current;
    if (!worker) return;

    aiThinking.current = true;

    const msg: AIWorkerRequest = {
      type: mode === 'vs-random' ? 'pickRandomMove' : 'findBestMove',
      state: serialize(state),
      depth: mode === 'vs-minimax' ? 6 : undefined,
    };

    const onMessage = (e: MessageEvent<AIWorkerResponse>) => {
      const {move} = e.data;
      if (move) {
        setHistory(prev => [...prev, state]);
        setState(applyMove(state, move));
      }
      aiThinking.current = false;
    };

    worker.addEventListener('message', onMessage);
    worker.postMessage(msg);

    return () => {
      worker.removeEventListener('message', onMessage);
      aiThinking.current = false;
    };
  }, [state, mode, aiPlayer, isSetup, setupTargets]);

  const onCellClick = useCallback((coord: Coord) => {
    // Setup phase: only accept valid setup placements
    if (isSetup) {
      if (mode !== 'hotseat' && state.currentPlayer === aiPlayer) return;
      if (setupTargets.some(t => t.row === coord.row && t.col === coord.col)) {
        executeSetup(coord);
      }
      return;
    }

    if (state.status !== 'active') return;
    if (mode !== 'hotseat' && state.currentPlayer === aiPlayer) return;

    const clickedId = state.board[coord.row][coord.col];
    const clickedTile = clickedId ? state.tiles.get(clickedId) ?? null : null;

    // Draw mode: only valid placement clicks accepted
    if (drawMode && selectedDrawTile) {
      const placementMove = allMoves.find(
        m => m.type === 'place'
          && m.tileName === selectedDrawTile
          && m.position.row === coord.row && m.position.col === coord.col,
      );
      if (placementMove) {
        executeMove(placementMove);
      }
      return;
    }

    setViewingBagTile(null);

    // Step 3 of command: pick destination for the commanded tile
    if (selectedTile && commandTarget) {
      const move = allMoves.find(
        m => m.type === 'command'
          && m.commander.row === selectedTile.row && m.commander.col === selectedTile.col
          && m.target.row === commandTarget.row && m.target.col === commandTarget.col
          && m.targetTo.row === coord.row && m.targetTo.col === coord.col,
      );
      if (move) {
        executeMove(move);
        return;
      }
      setCommandTarget(null);
      if (clickedTile && clickedTile.owner === state.currentPlayer) {
        setSelectedTile(coord);
        return;
      }
      setSelectedTile(null);
      return;
    }

    // If a tile is already selected, try to move/strike or start command flow
    if (selectedTile) {
      const isCommandTarget = allMoves.some(
        m => m.type === 'command'
          && m.commander.row === selectedTile.row && m.commander.col === selectedTile.col
          && m.target.row === coord.row && m.target.col === coord.col,
      );
      if (isCommandTarget) {
        setCommandTarget(coord);
        return;
      }

      const move = allMoves.find(m => {
        if (m.type === 'move') {
          return m.from.row === selectedTile.row && m.from.col === selectedTile.col
            && m.to.row === coord.row && m.to.col === coord.col;
        }
        if (m.type === 'strike') {
          return m.from.row === selectedTile.row && m.from.col === selectedTile.col
            && m.target.row === coord.row && m.target.col === coord.col;
        }
        return false;
      });

      if (move) {
        executeMove(move);
        return;
      }

      if (selectedTile.row === coord.row && selectedTile.col === coord.col) {
        setSelectedTile(null);
        return;
      }
    }

    if (clickedTile) {
      setSelectedTile(coord);
      setCommandTarget(null);
      return;
    }

    setSelectedTile(null);
    setCommandTarget(null);
  }, [state, selectedTile, commandTarget, allMoves, drawMode, selectedDrawTile, executeMove, executeSetup, mode, aiPlayer, isSetup, setupTargets]);

  return {
    state,
    selectedTile,
    selectedTileInstance,
    viewingBagTile,
    legalMoves: allMoves,
    onCellClick,
    onBagTileClick,
    newGame,
    undo,
    canUndo: history.length > 0,
    canDraw,
    drawMode,
    selectedDrawTile,
    placementTargets,
    startDraw,
    commandTarget,
    commandDestinations,
    setupTargets,
    setupLabel,
    inCheck,
    viewingEnemyMoves,
    showAllMoves,
    toggleShowAllMoves,
    allFriendlyTargets,
    allEnemyTargets,
  };
}
