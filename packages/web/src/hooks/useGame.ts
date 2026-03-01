import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import type { Coord, GameMove, GameState, Player, TileInstance } from '@the-duke/engine';
import { createInitialState, generateAllMoves, applyMove } from '@the-duke/engine';
import { pickRandomMove, findBestMove } from '@the-duke/ai';
import type { GameMode } from '../App.js';

export interface UseGameReturn {
  state: GameState;
  selectedTile: Coord | null;
  selectedTileInstance: TileInstance | null;
  viewingBagTile: { name: string; owner: Player } | null;
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
  /**
   * When non-null, we're in the second step of a command:
   * the commander is at `selectedTile`, the friendly target to
   * move is at `commandTarget`. The board highlights valid destinations.
   */
  commandTarget: Coord | null;
  /** Valid destination squares for the commanded tile. */
  commandDestinations: Coord[];
}

export function useGame(mode: GameMode): UseGameReturn {
  const [state, setState] = useState<GameState>(() => createInitialState());
  const [history, setHistory] = useState<GameState[]>([]);
  const [selectedTile, setSelectedTile] = useState<Coord | null>(null);
  const [viewingBagTile, setViewingBagTile] = useState<{ name: string; owner: Player } | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [selectedDrawTile, setSelectedDrawTile] = useState<string | null>(null);
  const [commandTarget, setCommandTarget] = useState<Coord | null>(null);
  const aiThinking = useRef(false);

  const allMoves = generateAllMoves(state);

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

  // Valid destinations for the commanded tile (step 3 of command)
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

  const resetSelection = useCallback(() => {
    setSelectedTile(null);
    setViewingBagTile(null);
    setDrawMode(false);
    setSelectedDrawTile(null);
    setCommandTarget(null);
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

  const hasPlacementMoves = allMoves.some(m => m.type === 'place');
  const canDraw = state.bags[state.currentPlayer].length > 0 && hasPlacementMoves;

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
    if (drawMode) return;
    setSelectedTile(null);
    setCommandTarget(null);
    setViewingBagTile({ name, owner });
  }, [drawMode]);

  // AI turn logic
  useEffect(() => {
    if (state.status !== 'active') return;
    if (mode === 'hotseat') return;

    const isAiTurn = state.currentPlayer === 'P2';
    if (!isAiTurn || aiThinking.current) return;

    aiThinking.current = true;
    const timer = setTimeout(() => {
      let move: GameMove | null = null;
      if (mode === 'vs-random') {
        move = pickRandomMove(state);
      } else if (mode === 'vs-minimax') {
        const result = findBestMove(state, 3);
        move = result.move;
      }

      if (move) {
        setHistory(prev => [...prev, state]);
        setState(applyMove(state, move!));
      }
      aiThinking.current = false;
    }, 300);

    return () => {
      clearTimeout(timer);
      aiThinking.current = false;
    };
  }, [state, mode]);

  const onCellClick = useCallback((coord: Coord) => {
    if (state.status !== 'active') return;
    if (mode !== 'hotseat' && state.currentPlayer === 'P2') return;

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
      // Clicked something invalid — back out of command target selection
      setCommandTarget(null);
      // If they clicked a different own tile, select it
      if (clickedTile && clickedTile.owner === state.currentPlayer) {
        setSelectedTile(coord);
        return;
      }
      setSelectedTile(null);
      return;
    }

    // If a tile is already selected, try to move/strike or start command flow
    if (selectedTile) {
      // Check for command: clicking a friendly tile that's a valid command target
      const isCommandTarget = allMoves.some(
        m => m.type === 'command'
          && m.commander.row === selectedTile.row && m.commander.col === selectedTile.col
          && m.target.row === coord.row && m.target.col === coord.col,
      );
      if (isCommandTarget) {
        // Enter step 2: show destinations for this commanded tile
        setCommandTarget(coord);
        return;
      }

      // Check for move or strike
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

      // Clicking same tile deselects
      if (selectedTile.row === coord.row && selectedTile.col === coord.col) {
        setSelectedTile(null);
        return;
      }
    }

    // Select any tile on the board to view its moveset
    if (clickedTile) {
      setSelectedTile(coord);
      setCommandTarget(null);
      return;
    }

    setSelectedTile(null);
    setCommandTarget(null);
  }, [state, selectedTile, commandTarget, allMoves, drawMode, selectedDrawTile, executeMove, mode]);

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
  };
}
