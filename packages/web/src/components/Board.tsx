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
  setupTargets: Coord[];
  inCheck: string | null;
  viewingEnemyMoves: GameMove[];
  allFriendlyTargets: Coord[];
  allEnemyTargets: Coord[];
}

export function Board({
  state, selectedTile, legalMoves, onCellClick,
  drawMode, placementTargets, commandTarget, commandDestinations,
  setupTargets, inCheck, viewingEnemyMoves,
  allFriendlyTargets, allEnemyTargets,
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

  const isEnemyMoveTarget = (r: number, c: number) => {
    if (viewingEnemyMoves.length === 0) return false;
    return viewingEnemyMoves.some(m => {
      if (m.type === 'move') return m.to.row === r && m.to.col === c;
      if (m.type === 'strike') return m.target.row === r && m.target.col === c;
      if (m.type === 'command') return m.targetTo.row === r && m.targetTo.col === c;
      return false;
    });
  };

  const isPlacementTarget = (r: number, c: number) =>
    drawMode && placementTargets.some(p => p.row === r && p.col === c);

  const isSetupTarget = (r: number, c: number) =>
    setupTargets.some(t => t.row === r && t.col === c);

  const isDukeInCheck = (r: number, c: number) => {
    if (!inCheck) return false;
    const tileId = state.board[r][c];
    if (!tileId) return false;
    const tile = state.tiles.get(tileId);
    return tile?.defName === 'Duke' && tile.owner === inCheck;
  };

  const isFriendlyAll = (r: number, c: number) =>
    allFriendlyTargets.some(t => t.row === r && t.col === c);

  const isEnemyAll = (r: number, c: number) =>
    allEnemyTargets.some(t => t.row === r && t.col === c);

  const rows = Array.from({ length: BOARD_SIZE }, (_, i) => i);
  const cols = Array.from({ length: BOARD_SIZE }, (_, i) => i);

  const showAll = allFriendlyTargets.length > 0 || allEnemyTargets.length > 0;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: `repeat(${BOARD_SIZE}, 1fr)`,
      gap: '1.5px',
      background: 'rgba(196,162,74,0.1)',
      padding: '3px',
      borderRadius: 'var(--radius-lg)',
      width: '100%',
      maxWidth: '460px',
      aspectRatio: '1',
      boxShadow: '0 2px 20px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(196,162,74,0.08)',
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
          const enemyTarget = isEnemyMoveTarget(r, c);
          const placeable = isPlacementTarget(r, c);
          const setupCell = isSetupTarget(r, c);
          const dukeCheck = isDukeInCheck(r, c);
          const friendlyAll = showAll && isFriendlyAll(r, c);
          const enemyAll = showAll && isEnemyAll(r, c);

          let bg: string;
          if (dukeCheck) bg = 'rgba(220, 60, 60, 0.4)';
          else if (setupCell) bg = 'var(--highlight)';
          else if (selected) bg = 'var(--selected)';
          else if (cmdTarget) bg = 'rgba(196,162,74,0.3)';
          else if (cmdDest) bg = 'rgba(196,162,74,0.2)';
          else if (commandable) bg = 'rgba(196,162,74,0.15)';
          else if (legalTarget) bg = 'var(--highlight)';
          else if (enemyTarget) bg = 'rgba(220, 60, 60, 0.22)';
          else if (placeable) bg = 'var(--accent-dim)';
          else if (friendlyAll && enemyAll) bg = 'rgba(180, 130, 50, 0.25)';
          else if (friendlyAll) bg = 'rgba(60, 180, 80, 0.2)';
          else if (enemyAll) bg = 'rgba(220, 60, 60, 0.18)';
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

              {dukeCheck && tile && (
                <div style={{
                  position: 'absolute', inset: '2px', borderRadius: '5px',
                  border: '2px solid rgba(220, 60, 60, 0.7)',
                  boxShadow: '0 0 10px rgba(220, 60, 60, 0.4)',
                  pointerEvents: 'none',
                }} />
              )}

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
                  boxShadow: '0 0 8px rgba(196,162,74,0.4)',
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

              {!tile && setupCell && (
                <div style={{
                  width: '36%', height: '36%', borderRadius: '50%',
                  background: 'var(--accent)', opacity: 0.4,
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

              {enemyTarget && !tile && (
                <div style={{
                  width: '28%', height: '28%', borderRadius: '50%',
                  background: 'rgba(220, 60, 60, 0.6)', opacity: 0.6,
                }} />
              )}

              {enemyTarget && tile && (
                <div style={{
                  position: 'absolute', inset: '2px', borderRadius: '5px',
                  border: '2px solid rgba(220, 60, 60, 0.6)', pointerEvents: 'none',
                }} />
              )}

              {showAll && !tile && friendlyAll && !enemyAll && (
                <div style={{
                  width: '22%', height: '22%', borderRadius: '50%',
                  background: 'rgba(60, 180, 80, 0.6)',
                }} />
              )}

              {showAll && !tile && enemyAll && !friendlyAll && (
                <div style={{
                  width: '22%', height: '22%', borderRadius: '50%',
                  background: 'rgba(220, 60, 60, 0.5)',
                }} />
              )}

              {showAll && !tile && friendlyAll && enemyAll && (
                <div style={{
                  width: '22%', height: '22%', borderRadius: '50%',
                  background: 'rgba(180, 130, 50, 0.6)',
                }} />
              )}
            </div>
          );
        }),
      )}
    </div>
  );
}
