import type { Coord, GameMove, GameState } from '@the-duke/engine';
import { BOARD_SIZE } from '@the-duke/engine';
import { Tile } from './Tile.js';

interface BoardProps {
  state: GameState;
  selectedTile: Coord | null;
  legalMoves: GameMove[];
  onCellClick: (coord: Coord) => void;
  drawMode: boolean;
  placementTargets: Coord[];
  commandTarget: Coord | null;
  commandDestinations: Coord[];
}

export function Board({
  state, selectedTile, legalMoves, onCellClick,
  drawMode, placementTargets, commandTarget, commandDestinations,
}: BoardProps) {
  const at = (r: number, c: number, coord: Coord | null) =>
    coord?.row === r && coord?.col === c;

  const isCommandDest = (r: number, c: number) =>
    commandTarget && commandDestinations.some(d => d.row === r && d.col === c);

  const isLegalTarget = (r: number, c: number) => {
    if (!selectedTile || drawMode || commandTarget) return false;
    return legalMoves.some(m => {
      if (m.type === 'move') {
        return m.from.row === selectedTile.row && m.from.col === selectedTile.col
          && m.to.row === r && m.to.col === c;
      }
      if (m.type === 'strike') {
        return m.from.row === selectedTile.row && m.from.col === selectedTile.col
          && m.target.row === r && m.target.col === c;
      }
      if (m.type === 'command') {
        return m.commander.row === selectedTile.row && m.commander.col === selectedTile.col
          && m.target.row === r && m.target.col === c;
      }
      return false;
    });
  };

  const isCommandableTarget = (r: number, c: number) => {
    if (!selectedTile || drawMode || commandTarget) return false;
    return legalMoves.some(
      m => m.type === 'command'
        && m.commander.row === selectedTile.row && m.commander.col === selectedTile.col
        && m.target.row === r && m.target.col === c,
    );
  };

  const isPlacementTarget = (r: number, c: number) =>
    drawMode && placementTargets.some(p => p.row === r && p.col === c);

  const rows = Array.from({ length: BOARD_SIZE }, (_, i) => i);
  const cols = Array.from({ length: BOARD_SIZE }, (_, i) => i);

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: `repeat(${BOARD_SIZE}, 1fr)`,
      gap: '1.5px',
      background: 'rgba(201,168,76,0.12)',
      padding: '3px',
      borderRadius: 'var(--radius-lg)',
      width: '100%',
      maxWidth: '460px',
      aspectRatio: '1',
      boxShadow: '0 2px 20px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(201,168,76,0.08)',
    }}>
      {rows.map(r =>
        cols.map(c => {
          const tileId = state.board[r][c];
          const tile = tileId ? state.tiles.get(tileId) ?? null : null;
          const dark = (r + c) % 2 === 1;
          const selected = at(r, c, selectedTile);
          const cmdTarget = at(r, c, commandTarget);
          const cmdDest = isCommandDest(r, c);
          const legalTarget = isLegalTarget(r, c);
          const commandable = isCommandableTarget(r, c);
          const placeable = isPlacementTarget(r, c);

          let bg: string;
          if (selected) bg = 'var(--selected)';
          else if (cmdTarget) bg = 'rgba(201,168,76,0.3)';
          else if (cmdDest) bg = 'rgba(201,168,76,0.2)';
          else if (commandable) bg = 'rgba(201,168,76,0.15)';
          else if (legalTarget) bg = 'var(--highlight)';
          else if (placeable) bg = 'var(--accent-dim)';
          else bg = dark ? 'var(--board-dark)' : 'var(--board-light)';

          const isCorner = (r === 0 || r === BOARD_SIZE - 1) && (c === 0 || c === BOARD_SIZE - 1);
          let radius = '0';
          if (isCorner) {
            if (r === 0 && c === 0) radius = '7px 0 0 0';
            else if (r === 0 && c === BOARD_SIZE - 1) radius = '0 7px 0 0';
            else if (r === BOARD_SIZE - 1 && c === 0) radius = '0 0 0 7px';
            else radius = '0 0 7px 0';
          }

          return (
            <div
              key={`${r}-${c}`}
              onClick={() => onCellClick({ row: r, col: c })}
              style={{
                background: bg,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                position: 'relative',
                aspectRatio: '1',
                borderRadius: radius,
                transition: 'background 0.15s',
              }}
            >
              {tile && <Tile tile={tile} />}

              {!tile && cmdDest && (
                <div style={{
                  width: '34%', height: '34%', borderRadius: '50%',
                  background: 'var(--accent)', opacity: 0.5,
                }} />
              )}

              {tile && commandable && (
                <div style={{
                  position: 'absolute', inset: '2px', borderRadius: '5px',
                  border: '2px solid var(--accent)', pointerEvents: 'none',
                }} />
              )}

              {tile && cmdTarget && (
                <div style={{
                  position: 'absolute', inset: '2px', borderRadius: '5px',
                  border: '2px solid var(--accent)',
                  boxShadow: '0 0 8px rgba(201,168,76,0.4)',
                  pointerEvents: 'none',
                }} />
              )}

              {tile && cmdDest && (
                <div style={{
                  position: 'absolute', inset: '2px', borderRadius: '5px',
                  border: '2px dashed var(--accent)', pointerEvents: 'none',
                }} />
              )}

              {!tile && placeable && (
                <div style={{
                  width: '36%', height: '36%', borderRadius: '50%',
                  background: 'var(--accent)', opacity: 0.35,
                  border: '2px solid var(--accent)',
                }} />
              )}

              {legalTarget && !tile && !commandable && (
                <div style={{
                  width: '28%', height: '28%', borderRadius: '50%',
                  background: 'var(--accent)', opacity: 0.45,
                }} />
              )}

              {legalTarget && tile && !commandable && (
                <div style={{
                  position: 'absolute', inset: '2px', borderRadius: '5px',
                  border: '2px solid var(--accent)', pointerEvents: 'none',
                }} />
              )}
            </div>
          );
        }),
      )}
    </div>
  );
}
