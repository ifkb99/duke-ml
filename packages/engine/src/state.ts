import type {
  BoardGrid, GameState, SerializedGameState,
} from './types.js';
import { BAG_TILE_NAMES } from './tiles.js';

export const BOARD_SIZE = 6;

export function createEmptyBoard(): BoardGrid {
  return Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null) as (string | null)[]);
}

function shuffleArray<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function createInitialState(): GameState {
  return {
    board: createEmptyBoard(),
    tiles: new Map(),
    bags: {
      P1: shuffleArray([...BAG_TILE_NAMES]),
      P2: shuffleArray([...BAG_TILE_NAMES]),
    },
    currentPlayer: 'P1',
    turnNumber: 0,
    status: 'setup',
    setupPhase: 'p1_duke',
  };
}

export function cloneState(state: GameState): GameState {
  return {
    board: state.board.map(row => [...row]),
    tiles: new Map(
      Array.from(state.tiles.entries()).map(([id, t]) => [id, { ...t, position: { ...t.position } }]),
    ),
    bags: {
      P1: [...state.bags.P1],
      P2: [...state.bags.P2],
    },
    currentPlayer: state.currentPlayer,
    turnNumber: state.turnNumber,
    status: state.status,
    setupPhase: state.setupPhase,
  };
}

export function serialize(state: GameState): SerializedGameState {
  return {
    board: state.board.map(row => [...row]),
    tiles: Array.from(state.tiles.entries()).map(([id, t]) => [id, { ...t, position: { ...t.position } }]),
    bags: { P1: [...state.bags.P1], P2: [...state.bags.P2] },
    currentPlayer: state.currentPlayer,
    turnNumber: state.turnNumber,
    status: state.status,
    setupPhase: state.setupPhase,
  };
}

export function deserialize(data: SerializedGameState): GameState {
  return {
    board: data.board.map(row => [...row]),
    tiles: new Map(
      data.tiles.map(([id, t]) => [id, { ...t, position: { ...t.position } }]),
    ),
    bags: { P1: [...data.bags.P1], P2: [...data.bags.P2] },
    currentPlayer: data.currentPlayer,
    turnNumber: data.turnNumber,
    status: data.status,
    setupPhase: data.setupPhase,
  };
}
