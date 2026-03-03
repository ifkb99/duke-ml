import type { GameState, Player } from '@the-duke/engine';
import type { GameMode } from '../App.js';

interface GameControlsProps {
  mode: GameMode;
  onModeChange: (mode: GameMode) => void;
  aiPlayer: Player;
  onAiPlayerChange: (player: Player) => void;
  state: GameState;
  onNewGame: () => void;
  onUndo: () => void;
  canUndo: boolean;
  compact?: boolean;
}

const btn: React.CSSProperties = {
  padding: '0.35rem 0.7rem',
  borderRadius: 'var(--radius)',
  border: '1px solid var(--surface-3)',
  background: 'var(--surface)',
  color: 'var(--text)',
  cursor: 'pointer',
  fontSize: '0.78rem',
  fontWeight: 500,
  transition: 'background 0.15s, border-color 0.15s',
  letterSpacing: '0.01em',
};

export function GameControls({
  mode, onModeChange, aiPlayer, onAiPlayerChange, state, onNewGame, onUndo, canUndo, compact,
}: GameControlsProps) {
  const isVsAi = mode !== 'hotseat';
  const playerColor = state.currentPlayer === 'P1' ? 'var(--p1-color)' : 'var(--p2-color)';
  const playerLabel = state.currentPlayer === 'P1' ? 'Light' : 'Dark';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '0.35rem',
      width: '100%',
    }}>
      <div style={{
        display: 'flex',
        gap: '0.4rem',
        flexWrap: 'wrap',
        justifyContent: 'center',
        alignItems: 'center',
      }}>
        <select
          value={mode}
          onChange={e => onModeChange(e.target.value as GameMode)}
          style={{
            ...btn,
            appearance: 'none',
            paddingRight: '1.4rem',
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 12 12'%3E%3Cpath fill='%236e7086' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'right 0.45rem center',
            fontSize: compact ? '0.72rem' : '0.78rem',
          }}
        >
          <option value="hotseat">Hot Seat</option>
          <option value="vs-random">vs Random</option>
          <option value="vs-minimax">vs Minimax</option>
        </select>

        {isVsAi && (
          <select
            value={aiPlayer}
            onChange={e => onAiPlayerChange(e.target.value as Player)}
            style={{
              ...btn,
              appearance: 'none',
              paddingRight: '1.4rem',
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 12 12'%3E%3Cpath fill='%236e7086' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 0.45rem center',
              fontSize: compact ? '0.72rem' : '0.78rem',
            }}
          >
            <option value="P2">Play as Light</option>
            <option value="P1">Play as Dark</option>
          </select>
        )}

        <button onClick={onNewGame} style={btn}>New</button>
        <button
          onClick={onUndo}
          disabled={!canUndo}
          style={{ ...btn, opacity: canUndo ? 1 : 0.3 }}
        >
          Undo
        </button>
      </div>

      <div style={{
        fontSize: '0.76rem',
        color: 'var(--text-muted)',
        fontWeight: 400,
        display: 'flex',
        alignItems: 'center',
        gap: '0.4em',
      }}>
        <span>{state.status === 'setup' ? 'Setup' : `Turn ${state.turnNumber}`}</span>
        <span style={{ opacity: 0.4 }}>&middot;</span>
        <span style={{
          color: playerColor,
          fontWeight: 600,
        }}>
          {playerLabel}
        </span>
        {state.status !== 'setup' && (
          <>
            <span style={{ opacity: 0.4 }}>&middot;</span>
            <span>Bag: {state.bags[state.currentPlayer].length}</span>
          </>
        )}
      </div>
    </div>
  );
}
