import {useEffect, useRef, useState, type ReactNode} from 'react';
import type {Player} from '@the-duke/engine';
import {type GameMode, MODE_NAME, PLAYER_NAME} from '../ui.js';
import {CloseIcon, PlusIcon, RematchIcon} from './Icons.js';
import {Crown} from './Tile.js';

export function Modal({open, onClose, label, children}: {open: boolean; onClose: () => void; label: string; children: ReactNode}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} className="modal" aria-label={label} onClose={onClose}
            onClick={e => { if (e.target === ref.current) onClose(); }}>
      {open && children}
    </dialog>
  );
}

const OPPONENTS: {mode: GameMode; blurb: string}[] = [
  {mode: 'vs-zero', blurb: 'Trained by self-play. Runs on your device; thinks ~3 s a move.'},
  {mode: 'vs-minimax', blurb: 'Classic search, 6 moves deep.'},
  {mode: 'vs-random', blurb: 'Plays any legal move. Good for learning the rules.'},
  {mode: 'hotseat', blurb: 'Two people, one device. Light sits at the bottom.'},
];

type SidePick = Player | 'random';

interface NewGameProps {
  open: boolean;
  mode: GameMode;
  /** The person's side in the current settings */
  human: Player;
  onClose: () => void;
  onStart: (mode: GameMode, human: Player) => void;
}

export function NewGameDialog({open, mode, human, onClose, onStart}: NewGameProps) {
  const [pickMode, setPickMode] = useState(mode);
  const [side, setSide] = useState<SidePick>(human);
  useEffect(() => {
    if (open) { setPickMode(mode); setSide(human); }
  }, [open, mode, human]);

  const start = () => {
    const h: Player = side === 'random' ? (Math.random() < 0.5 ? 'P1' : 'P2') : side;
    onStart(pickMode, h);
  };

  return (
    <Modal open={open} onClose={onClose} label="New game">
      <header className="modal-head">
        <h2><Crown className="crown modal-crown" /> New game</h2>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close"><CloseIcon /></button>
      </header>

      <fieldset className="field">
        <legend>Opponent</legend>
        <div className="options">
          {OPPONENTS.map(o => (
            <label key={o.mode} className={`option ${pickMode === o.mode ? 'on' : ''}`}>
              <input type="radio" name="opponent" value={o.mode} checked={pickMode === o.mode}
                     onChange={() => setPickMode(o.mode)} />
              <span className="option-title">{MODE_NAME[o.mode]}{o.mode === 'vs-zero' && <span className="badge badge-gold">Strongest</span>}</span>
              <span className="option-blurb">{o.blurb}</span>
            </label>
          ))}
        </div>
      </fieldset>

      {pickMode !== 'hotseat' && (
        <fieldset className="field">
          <legend>Play as</legend>
          <div className="segmented">
            {(['P1', 'P2', 'random'] as const).map(s => (
              <label key={s} className={`segment ${side === s ? 'on' : ''}`}>
                <input type="radio" name="side" value={s} checked={side === s} onChange={() => setSide(s)} />
                {s !== 'random' && <span className={`status-dot ${s === 'P1' ? 'dot-light' : 'dot-dark'}`} />}
                {s === 'random' ? 'Random' : PLAYER_NAME[s]}
                {s === 'P1' && <span className="segment-note">moves first</span>}
              </label>
            ))}
          </div>
        </fieldset>
      )}

      <footer className="modal-foot">
        <button type="button" className="btn" onClick={onClose}>Cancel</button>
        <button type="button" className="btn btn-gold btn-big" onClick={start} autoFocus>
          <PlusIcon /> Start game
        </button>
      </footer>
    </Modal>
  );
}

interface GameOverProps {
  winner: Player;
  reason: 'captured' | 'checkmate';
  /** The person's side, or null in hot seat */
  human: Player | null;
  mode: GameMode;
  onRematch: () => void;
  onNewGame: () => void;
  onDismiss: () => void;
}

/** Laid over the board; dismiss it to look at the final position. */
export function GameOver({winner, reason, human, mode, onRematch, onNewGame, onDismiss}: GameOverProps) {
  const won = human ? winner === human : true;
  const title = human ? (won ? 'Victory' : 'Defeat') : `${PLAYER_NAME[winner]} wins`;
  const ai = `the ${MODE_NAME[mode]}`;
  const loser = human ? (won ? ai : 'you') : PLAYER_NAME[winner === 'P1' ? 'P2' : 'P1'];
  const line = reason === 'checkmate'
    ? `Checkmate: ${loser} had no legal move left.`
    : human ? (won ? `You captured ${ai}’s Duke.` : `The ${MODE_NAME[mode]} captured your Duke.`)
      : `${PLAYER_NAME[winner]} captured the enemy Duke.`;
  return (
    <div className="gameover-backdrop">
      <div className={`card gameover ${won ? 'gameover-won' : 'gameover-lost'}`} role="alertdialog" aria-label={title}>
        <button type="button" className="icon-btn gameover-close" onClick={onDismiss} aria-label="Look at the board">
          <CloseIcon />
        </button>
        <Crown className="crown gameover-crown" />
        <h2>{title}</h2>
        <p>{line}</p>
        <div className="gameover-actions">
          <button type="button" className="btn btn-gold btn-big" onClick={onRematch} autoFocus><RematchIcon /> Rematch</button>
          <button type="button" className="btn btn-big" onClick={onNewGame}><PlusIcon /> New game</button>
        </div>
      </div>
    </div>
  );
}
