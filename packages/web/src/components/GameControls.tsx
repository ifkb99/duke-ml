import type { GameState } from '@the-duke/engine';
import type { GameMode } from '../App.js';

interface GameControlsProps {
  mode: GameMode;
  onModeChange: (mode: GameMode) => void;
  state: GameState;
  onNewGame: () => void;
  onUndo: () => void;
  canUndo: boolean;
}

const btnStyle: React.CSSProperties = {
  padding: '0.4rem 0.9rem',
  borderRadius: '6px',
  border: '1px solid var(--surface-2)',
  background: 'var(--surface)',
  color: 'var(--text)',
  cursor: 'pointer',
  fontSize: '0.85rem',
  fontWeight: 500,
  transition: 'background 0.15s',
};

export function GameControls({
  mode, onModeChange, state, onNewGame, onUndo, canUndo,
}: GameControlsProps) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '0.5rem',
      width: '100%',
    }}>
      <div style={{
        display: 'flex',
        gap: '0.5rem',
        flexWrap: 'wrap',
        justifyContent: 'center',
      }}>
        <select
          value={mode}
          onChange={e => onModeChange(e.target.value as GameMode)}
          style={{
            ...btnStyle,
            appearance: 'none',
            paddingRight: '1.5rem',
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%238892b0' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'right 0.5rem center',
          }}
        >
          <option value="hotseat">Hot Seat (2P)</option>
          <option value="vs-random">vs Random Bot</option>
          <option value="vs-minimax">vs Minimax AI</option>
        </select>

        <button onClick={onNewGame} style={btnStyle}>
          New Game
        </button>

        <button
          onClick={onUndo}
          disabled={!canUndo}
          style={{ ...btnStyle, opacity: canUndo ? 1 : 0.4 }}
        >
          Undo
        </button>
      </div>

      <div style={{
        fontSize: '0.85rem',
        color: 'var(--text-muted)',
      }}>
        Turn {state.turnNumber} &middot;{' '}
        <span style={{ color: state.currentPlayer === 'P1' ? 'var(--p1-color)' : 'var(--p2-color)', fontWeight: 600 }}>
          {state.currentPlayer === 'P1' ? 'Light' : 'Dark'}
        </span>
        {' '}to move &middot; Bag: {state.bags[state.currentPlayer].length} tiles
      </div>
    </div>
  );
}
