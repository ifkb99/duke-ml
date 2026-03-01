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

  // Step 2 of command: highlight valid destinations for the commanded tile
  const isCommandDest = (r: number, c: number) =>
    commandTarget && commandDestinations.some(d => d.row === r && d.col === c);

  // Step 1: highlight legal targets (moves, strikes, and commandable friendly tiles)
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

  // Is this cell a commandable target (vs a move/strike target)?
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
      gap: '2px',
      background: '#111',
      padding: '2px',
      borderRadius: '8px',
      width: '100%',
      maxWidth: '480px',
      aspectRatio: '1',
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
          else if (cmdTarget) bg = 'rgba(255, 183, 77, 0.35)';
          else if (cmdDest) bg = 'rgba(255, 183, 77, 0.25)';
          else if (commandable) bg = 'rgba(255, 183, 77, 0.2)';
          else if (legalTarget) bg = 'var(--highlight)';
          else if (placeable) bg = 'rgba(233, 69, 96, 0.2)';
          else bg = dark ? 'var(--board-dark)' : 'var(--board-light)';

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
                transition: 'background 0.15s',
              }}
            >
              {tile && <Tile tile={tile} />}

              {/* Command destination marker (empty square) */}
              {!tile && cmdDest && (
                <div style={{
                  width: '36%',
                  height: '36%',
                  borderRadius: '50%',
                  background: '#ffb74d',
                  opacity: 0.6,
                }} />
              )}

              {/* Commandable tile border highlight */}
              {tile && commandable && (
                <div style={{
                  position: 'absolute',
                  inset: '2px',
                  borderRadius: '6px',
                  border: '2px solid #ffb74d',
                  pointerEvents: 'none',
                }} />
              )}

              {/* Command target being commanded (step 2 active) */}
              {tile && cmdTarget && (
                <div style={{
                  position: 'absolute',
                  inset: '2px',
                  borderRadius: '6px',
                  border: '2px solid #ffb74d',
                  boxShadow: '0 0 6px rgba(255,183,77,0.5)',
                  pointerEvents: 'none',
                }} />
              )}

              {/* Command destination marker (occupied by enemy — capture) */}
              {tile && cmdDest && (
                <div style={{
                  position: 'absolute',
                  inset: '2px',
                  borderRadius: '6px',
                  border: '2px dashed #ffb74d',
                  pointerEvents: 'none',
                }} />
              )}

              {/* Draw placement target */}
              {!tile && placeable && (
                <div style={{
                  width: '40%',
                  height: '40%',
                  borderRadius: '50%',
                  background: 'var(--accent)',
                  opacity: 0.4,
                  border: '2px solid var(--accent)',
                }} />
              )}

              {/* Regular move/strike target dot */}
              {legalTarget && !tile && !commandable && (
                <div style={{
                  width: '30%',
                  height: '30%',
                  borderRadius: '50%',
                  background: 'var(--accent)',
                  opacity: 0.5,
                }} />
              )}

              {/* Strike target on enemy tile */}
              {legalTarget && tile && !commandable && (
                <div style={{
                  position: 'absolute',
                  inset: '2px',
                  borderRadius: '6px',
                  border: '2px solid var(--accent)',
                  pointerEvents: 'none',
                }} />
              )}
            </div>
          );
        }),
      )}
    </div>
  );
}
