import {useEffect, useMemo, useRef, useState, type CSSProperties} from 'react';
import type {Coord, GameMove, GameState, Player, TileInstance} from '@the-duke/engine';
import {BOARD_SIZE} from '@the-duke/engine';
import type {CommandChoice, LastMove} from '../hooks/useGame.js';
import {type Orientation, PLAYER_NAME, moveSquares, sameCoord, toScreen} from '../ui.js';
import {TileFace} from './Tile.js';
import './board.css';

interface BoardProps {
  state: GameState;
  bottom: Orientation;
  humanTurn: boolean;
  selectedTile: Coord | null;
  legalMoves: GameMove[];
  onCellClick: (coord: Coord) => void;
  drawMode: boolean;
  placementTargets: Coord[];
  commandTarget: Coord | null;
  commandDestinations: Coord[];
  commandChoice: CommandChoice | null;
  onCommandChoice: (choice: 'command' | 'select' | null) => void;
  setupTargets: Coord[];
  inCheck: Player | null;
  viewingEnemyMoves: GameMove[];
  allFriendlyTargets: Coord[];
  allEnemyTargets: Coord[];
  lastMove: LastMove | null;
}

type Mark = 'dot' | 'capture' | 'strike' | 'command' | 'place' | 'view-dot' | 'view-capture';

interface CellInfo {
  marks: Set<Mark>;
  classes: Set<string>;
  threat?: 'mine' | 'theirs' | 'both';
}

const key = (c: Coord) => c.row * BOARD_SIZE + c.col;
const FILES = 'abcdef';

export function Board(props: BoardProps) {
  const {state, bottom, selectedTile, legalMoves, drawMode, commandTarget, lastMove, humanTurn} = props;

  const tileAt = (c: Coord): TileInstance | null => {
    const id = state.board[c.row][c.col];
    return id ? state.tiles.get(id) ?? null : null;
  };

  const cells = useMemo(() => {
    const info: CellInfo[] = Array.from({length: BOARD_SIZE * BOARD_SIZE}, () => ({marks: new Set(), classes: new Set()}));
    const at = (c: Coord) => info[key(c)];
    const occupied = (c: Coord) => state.board[c.row][c.col] !== null;

    if (lastMove) {
      if (lastMove.type === 'setup') at(lastMove.position).classes.add('last-to');
      else {
        const sq = moveSquares(lastMove);
        if (sq.from) at(sq.from).classes.add('last-from');
        if (sq.to) at(sq.to).classes.add('last-to');
        if (sq.struck) at(sq.struck).classes.add('last-struck');
        if (sq.commander) at(sq.commander).classes.add('last-commander');
      }
    }

    if (selectedTile) at(selectedTile).classes.add('selected');
    if (commandTarget) at(commandTarget).classes.add('command-target');

    // The selected tile's options
    if (selectedTile && humanTurn && !drawMode && !commandTarget) {
      for (const m of legalMoves) {
        if (m.type === 'move' && sameCoord(m.from, selectedTile)) at(m.to).marks.add(occupied(m.to) ? 'capture' : 'dot');
        else if (m.type === 'strike' && sameCoord(m.from, selectedTile)) at(m.target).marks.add('strike');
        else if (m.type === 'command' && sameCoord(m.commander, selectedTile)) at(m.target).marks.add('command');
      }
    }
    for (const d of props.commandDestinations) at(d).marks.add(occupied(d) ? 'capture' : 'dot');
    for (const p of props.placementTargets) at(p).marks.add('place');
    for (const p of props.setupTargets) if (humanTurn) at(p).marks.add('place');

    for (const m of props.viewingEnemyMoves) {
      const t = m.type === 'move' ? m.to : m.type === 'strike' ? m.target : m.type === 'command' ? m.targetTo : null;
      if (t) at(t).marks.add(occupied(t) ? 'view-capture' : 'view-dot');
    }

    for (const t of props.allFriendlyTargets) at(t).threat = 'mine';
    for (const t of props.allEnemyTargets) at(t).threat = at(t).threat ? 'both' : 'theirs';

    if (props.inCheck) {
      for (const tile of state.tiles.values()) {
        if (tile.defName === 'Duke' && tile.owner === props.inCheck) at(tile.position).classes.add('in-check');
      }
    }
    return info;
  }, [state, selectedTile, legalMoves, drawMode, commandTarget, lastMove, humanTurn, props.commandDestinations,
      props.placementTargets, props.setupTargets, props.viewingEnemyMoves, props.allFriendlyTargets,
      props.allEnemyTargets, props.inCheck]);

  // Screen order: row by row from the top of the screen
  const squares: Coord[] = [];
  for (let sr = 0; sr < BOARD_SIZE; sr++) {
    for (let c = 0; c < BOARD_SIZE; c++) squares.push(toScreen({row: sr, col: c}, bottom));
  }

  const viewedTile = selectedTile ? tileAt(selectedTile) : null;
  const viewClass = viewedTile && viewedTile.owner === bottom ? 'view-own' : 'view-theirs';

  return (
    <div className={`board ${viewClass}`}>
      <div className="board-inner">
        <div className="board-cells" role="grid" aria-label="Board">
          {squares.map(c => {
            const s = toScreen(c, bottom);
            const cell = cells[key(c)];
            const tile = tileAt(c);
            const name = `${FILES[c.col]}${c.row + 1}`;
            const classes = ['cell', (c.row + c.col) % 2 ? 'cell-dark' : 'cell-light', ...cell.classes];
            if (cell.marks.size) classes.push('cell-target');
            return (
              <button
                key={key(c)}
                type="button"
                className={classes.join(' ')}
                onClick={() => props.onCellClick(c)}
                aria-label={`${name}${tile ? `, ${PLAYER_NAME[tile.owner]} ${tile.defName}` : ''}`}
              >
                {s.col === 0 && <span className="coord coord-rank">{c.row + 1}</span>}
                {s.row === BOARD_SIZE - 1 && <span className="coord coord-file">{FILES[c.col]}</span>}
              </button>
            );
          })}
        </div>

        <Pieces state={state} bottom={bottom} />

        <div className="board-marks" aria-hidden>
          {squares.map(c => {
            const cell = cells[key(c)];
            const s = toScreen(c, bottom);
            if (!cell.marks.size && !cell.threat && !cell.classes.has('last-struck')) return null;
            return (
              <div key={key(c)} className="mark-cell" style={{'--x': s.col, '--y': s.row} as CSSProperties}>
                {[...cell.marks].map(m => <span key={m} className={`mark mark-${m}`} />)}
                {cell.classes.has('last-struck') && <span className="mark mark-struck" />}
                {cell.threat && <span className={`threat threat-${cell.threat}`} />}
              </div>
            );
          })}
        </div>

        {props.commandChoice && (
          <ChoicePopover choice={props.commandChoice} bottom={bottom} commander={viewedTile?.defName ?? ''}
                         onChoose={props.onCommandChoice} />
        )}
      </div>
    </div>
  );
}

function ChoicePopover({choice, bottom, commander, onChoose}: {
  choice: CommandChoice; bottom: Orientation; commander: string; onChoose: (c: 'command' | 'select' | null) => void;
}) {
  const s = toScreen(choice.coord, bottom);
  const below = s.row < 2;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onChoose(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onChoose]);
  return (
    <div className={`choice ${below ? 'choice-below' : ''}`}
         style={{'--x': s.col, '--y': s.row} as CSSProperties} role="dialog" aria-label="Choose an action">
      <button type="button" className="btn btn-gold" onClick={() => onChoose('command')} autoFocus>
        <span className="choice-icon" aria-hidden>◆</span> {commander} commands {choice.tileName}
      </button>
      <button type="button" className="btn" onClick={() => onChoose('select')}>
        Select {choice.tileName}
      </button>
    </div>
  );
}

/** The tiles, absolutely placed so moves slide; captured tiles fade out where they stood. */
function Pieces({state, bottom}: {state: GameState; bottom: Orientation}) {
  const prev = useRef(state.tiles);
  const [ghosts, setGhosts] = useState<TileInstance[]>([]);
  useEffect(() => {
    const gone = [...prev.current.values()].filter(t => !state.tiles.has(t.id));
    prev.current = state.tiles;
    if (gone.length === 0 || gone.length > 2) return;   // more: a new game or a long undo
    setGhosts(gone);
    const timer = setTimeout(() => setGhosts([]), 600);
    return () => clearTimeout(timer);
  }, [state.tiles]);

  const tiles = [...state.tiles.values()].sort((a, b) => (a.id < b.id ? -1 : 1));
  return (
    <div className="board-pieces" aria-hidden>
      {ghosts.map(t => <Piece key={`ghost-${t.id}`} tile={t} bottom={bottom} ghost />)}
      {tiles.map(t => <Piece key={t.id} tile={t} bottom={bottom} />)}
    </div>
  );
}

function Piece({tile, bottom, ghost}: {tile: TileInstance; bottom: Orientation; ghost?: boolean}) {
  // The face remounts when the tile flips; after the first mount that plays the flip animation
  const mounted = useRef(false);
  useEffect(() => { mounted.current = true; }, []);
  const s = toScreen(tile.position, bottom);
  return (
    <div className={`piece ${ghost ? 'piece-ghost' : ''}`} style={{'--x': s.col, '--y': s.row} as CSSProperties}>
      <div className={`piece-body ${ghost ? '' : 'piece-enter'}`}>
        <TileFace key={tile.side} name={tile.defName} owner={tile.owner} side={tile.side} bottom={bottom}
                  flipped={mounted.current && !ghost} />
      </div>
    </div>
  );
}
