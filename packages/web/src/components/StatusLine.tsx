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
    text = game.humanTurn ? (human ? 'Set up your side' : `${PLAYER_NAME[mover]} sets up`) : `${ai} is setting up`;
    detail = game.humanTurn ? game.setupStep : null;
    if (!game.humanTurn) tone = 'thinking';
  } else if (game.thinking || !game.humanTurn) {
    tone = 'thinking';
    text = `${ai} is thinking`;
  } else if (game.drawMode && game.selectedDrawTile) {
    text = `${human ? 'You' : PLAYER_NAME[mover]} drew ${article(game.selectedDrawTile)}`;
    detail = `Place it next to ${human ? 'your' : 'the'} Duke`;
  } else if (game.commandTarget) {
    text = 'Choose where the commanded tile goes';
  } else {
    text = human ? 'Your move' : `${PLAYER_NAME[mover]} to move`;
    // Drawing needs a tile in the bag and a free square next to the Duke
    detail = game.selectedTile ? null
      : game.canDraw ? 'Pick a tile, or draw a new one'
      : state.bags[mover].length === 0 ? 'Pick a tile to move: the bag is empty'
      : 'Pick a tile: no free square by the Duke to draw';
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
      {state.status !== 'setup' && <span className="status-count">Turn {state.turnNumber}</span>}
    </div>
  );
}

function article(name: string) {
  return /^[AEIOU]/.test(name) ? `an ${name}` : `a ${name}`;
}
