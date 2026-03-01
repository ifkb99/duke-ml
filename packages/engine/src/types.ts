// (0,0) is top-left. Row 0 is Player 1's back row, Row 5 is Player 2's.
export type Coord = { row: number; col: number };

// Movement offsets defined from Player 1's perspective.
// When generating moves for Player 2, negate the row component.
export type Offset = { dRow: number; dCol: number };

export type MoveType = 'step' | 'slide' | 'jump' | 'jump_slide' | 'strike' | 'command';

export interface MovePattern {
  type: MoveType;
  offsets: Offset[];
}

export interface TileSide {
  patterns: MovePattern[];
}

export interface TileDefinition {
  name: string;
  sideA: TileSide;
  sideB: TileSide;
}

export type Player = 'P1' | 'P2';
export type Side = 'A' | 'B';

export interface TileInstance {
  defName: string;
  owner: Player;
  side: Side;
  position: Coord;
  id: string;
}

export type BoardGrid = (string | null)[][];

export type GameStatus = 'setup' | 'active' | 'P1_wins' | 'P2_wins';

export type SetupPhase =
  | 'p1_duke'
  | 'p1_footman1'
  | 'p1_footman2'
  | 'p2_duke'
  | 'p2_footman1'
  | 'p2_footman2';

export interface GameState {
  board: BoardGrid;
  tiles: Map<string, TileInstance>;
  bags: {
    P1: string[];
    P2: string[];
  };
  currentPlayer: Player;
  turnNumber: number;
  status: GameStatus;
  setupPhase?: SetupPhase;
}

export type GameMove =
  | { type: 'place'; tileName: string; position: Coord }
  | { type: 'move'; from: Coord; to: Coord }
  | { type: 'strike'; from: Coord; target: Coord }
  | { type: 'command'; commander: Coord; target: Coord; targetTo: Coord };

export interface SerializedGameState {
  board: (string | null)[][];
  tiles: [string, TileInstance][];
  bags: { P1: string[]; P2: string[] };
  currentPlayer: Player;
  turnNumber: number;
  status: GameStatus;
  setupPhase?: SetupPhase;
}
