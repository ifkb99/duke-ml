import type {Player} from '@the-duke/engine';
import type {UseGameReturn} from '../hooks/useGame.js';
import {type GameMode, MODE_NAME, PLAYER_NAME, endReason} from '../ui.js';

interface StatusLineProps {
  game: UseGameReturn;
  mode: GameMode;
  /** The person's side against the AI; null in hot seat */
  human: Player | null;
}

/** One line that always says what's happening and what to do next. */
export function StatusLine({game, mode, human}: StatusLineProps) {
  const {state} = game;
  const mover = state.currentPlayer;
  const ai = MODE_NAME[mode];

  let tone: 'turn' | 'thinking' | 'check' | 'win' | 'loss' | 'error' = 'turn';
  let player: Player | null = mover;
  let text: string;
  let detail: string | null = null;

  if (state.status === 'P1_wins' || state.status === 'P2_wins') {
    const winner: Player = state.status === 'P1_wins' ? 'P1' : 'P2';
    player = winner;
    tone = human && winner !== human ? 'loss' : 'win';
    text = human ? (winner === human ? 'You win!' : `${ai} wins`) : `${PLAYER_NAME[winner]} wins!`;
    detail = endReason(state) === 'captured' ? 'Duke captured' : 'Checkmate: no legal move left';
  } else if (game.aiError) {
    tone = 'error';
    text = `${ai} ran into a problem`;
    detail = game.aiError;
  } else if (state.status === 'setup') {
    text = game.humanTurn ? (game.setupStep ?? '') : `${ai} is setting up`;
    detail = human ? null : `${PLAYER_NAME[mover]} to place`;
    if (!game.humanTurn) tone = 'thinking';
  } else if (game.thinking || !game.humanTurn) {
    tone = 'thinking';
    text = `${ai} is thinking`;
  } else if (game.drawMode && game.selectedDrawTile) {
    text = `You drew ${article(game.selectedDrawTile)} — place it next to your Duke`;
    if (!human) text = `${PLAYER_NAME[mover]} drew ${article(game.selectedDrawTile)} — place it next to the Duke`;
  } else if (game.commandTarget) {
    text = 'Choose where the commanded tile goes';
  } else {
    text = human ? 'Your move' : `${PLAYER_NAME[mover]} to move`;
    detail = game.selectedTile ? null : 'Pick a tile, or draw a new one';
  }
  if (game.inCheck && state.status === 'active' && !game.aiError) {
    if (tone !== 'thinking') tone = 'check';
    const mine = human ? game.inCheck === human : game.inCheck === mover;
    detail = mine ? 'Your Duke is in check!'
      : `${human ? ai : PLAYER_NAME[game.inCheck]}’s Duke is in check`;
  }

  return (
    <div className={`status status-${tone}`} role="status" aria-live="polite">
      {player && <span className={`status-dot ${player === 'P1' ? 'dot-light' : 'dot-dark'}`} />}
      <span className="status-text">
        <span className="status-main">
          {text}
          {tone === 'thinking' && <span className="thinking-dots" aria-hidden><i /><i /><i /></span>}
        </span>
        {detail && <span className="status-detail">{detail}</span>}
      </span>
      {state.status !== 'setup' && <span className="status-turn">Turn {state.turnNumber}</span>}
    </div>
  );
}

function article(name: string) {
  return /^[AEIOU]/.test(name) ? `an ${name}` : `a ${name}`;
}
