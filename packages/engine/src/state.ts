import type {
  BoardGrid, Coord, GameState, Player, SerializedGameState, TileInstance,
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

function placeTile(
  board: BoardGrid,
  tiles: Map<string, TileInstance>,
  id: string,
  tile: TileInstance,
): void {
  tiles.set(id, tile);
  board[tile.position.row][tile.position.col] = id;
}

/**
 * Create the initial game state with both players' starting positions.
 *
 * P1 places Duke in their back row (row 0), P2 in row 5.
 * Each Duke gets two Footmen on orthogonally adjacent spaces.
 *
 * `dukeCol` lets callers control Duke column (default: col 2 for P1, col 3 for P2).
 */
export function createInitialState(
  p1DukeCol = 2,
  p2DukeCol = 3,
): GameState {
  const board = createEmptyBoard();
  const tiles = new Map<string, TileInstance>();

  const placeDukeAndFootmen = (
    player: Player,
    dukeRow: number,
    dukeCol: number,
  ) => {
    const dukeId = `${player}-Duke`;
    placeTile(board, tiles, dukeId, {
      defName: 'Duke',
      owner: player,
      side: 'A',
      position: { row: dukeRow, col: dukeCol },
      id: dukeId,
    });

    // Footmen on two orthogonally adjacent spaces.
    // Place one in front of (toward center) and one to the side of the Duke.
    const footmenPositions: Coord[] = player === 'P1'
      ? [
          { row: dukeRow + 1, col: dukeCol },     // in front (toward center)
          { row: dukeRow, col: dukeCol + 1 },      // to the right
        ]
      : [
          { row: dukeRow - 1, col: dukeCol },      // in front (toward center)
          { row: dukeRow, col: dukeCol - 1 },      // to the left
        ];

    footmenPositions.forEach((pos, i) => {
      const fId = `${player}-Footman-${i + 1}`;
      placeTile(board, tiles, fId, {
        defName: 'Footman',
        owner: player,
        side: 'A',
        position: pos,
        id: fId,
      });
    });
  };

  placeDukeAndFootmen('P1', 0, p1DukeCol);
  placeDukeAndFootmen('P2', 5, p2DukeCol);

  return {
    board,
    tiles,
    bags: {
      P1: shuffleArray([...BAG_TILE_NAMES]),
      P2: shuffleArray([...BAG_TILE_NAMES]),
    },
    currentPlayer: 'P1',
    turnNumber: 1,
    status: 'active',
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
  };
}
