import type {UseGameReturn} from '../hooks/useGame.js';
import {BagIcon, EyeIcon, GaugeIcon, UndoIcon} from './Icons.js';

interface ActionBarProps {
  game: UseGameReturn;
  showEval: boolean;
  onToggleEval: () => void;
}

export function ActionBar({game, showEval, onToggleEval}: ActionBarProps) {
  const {state} = game;
  const bag = state.status === 'setup' ? 0 : state.bags[state.currentPlayer].length;
  const active = state.status === 'active';
  return (
    <div className="actions">
      <div className="actions-row">
      <button type="button" className="btn" onClick={game.undo} disabled={!game.canUndo} title="Take back your last move" aria-label="Undo">
        <UndoIcon /> <span className="btn-label">Undo</span>
      </button>
      <button type="button" className="btn btn-draw" onClick={game.startDraw}
              disabled={!game.canDraw}
              title={game.canDraw ? 'Pull a random tile from your bag and place it next to your Duke' : undefined}>
        <BagIcon />
        <span className="btn-label">{game.drawMode ? 'Drawn' : 'Draw'}</span>
        {active && <span className="btn-count">{bag}</span>}
      </button>
      <button type="button" className={`btn btn-toggle ${game.showAllMoves ? 'on' : ''}`} aria-pressed={game.showAllMoves}
              onClick={game.toggleShowAllMoves} disabled={!active} title="Show every square each side can reach" aria-label="Threats">
        <EyeIcon /> <span className="btn-label">Threats</span>
      </button>
      <button type="button" className={`btn btn-toggle ${showEval ? 'on' : ''}`} aria-pressed={showEval}
              onClick={onToggleEval} title="Show the neural net's estimate of who's winning" aria-label="Evaluation bar">
        <GaugeIcon /> <span className="btn-label">Eval</span>
      </button>
      </div>
    </div>
  );
}
