import {useState} from 'react';
import type {Player, Side, TileSide} from '@the-duke/engine';
import {TILE_REGISTRY} from '@the-duke/engine';
import type {UseGameReturn} from '../hooks/useGame.js';
import {MOVE_INFO, type Orientation, PLAYER_NAME, patternRowSign, sameCoord} from '../ui.js';
import {Modal} from './Dialogs.js';
import {CloseIcon, InfoIcon} from './Icons.js';
import {LegendIcon, MoveDiagram} from './MoveDiagram.js';
import {TileChip} from './Tile.js';

interface Subject {
  name: string;
  owner: Player;
  /** The side facing up; undefined for a tile still in the bag */
  side?: Side;
  badges: {text: string; tone?: 'gold' | 'owner' | 'warn'}[];
  note?: string;
  closable: boolean;
}

/** What the inspector is about right now, or null when nothing is picked. */
export function inspectorSubject(game: UseGameReturn): Subject | null {
  const {state} = game;
  if (game.drawMode && game.selectedDrawTile) {
    return {name: game.selectedDrawTile, owner: state.currentPlayer, side: 'A', closable: false,
      badges: [{text: 'Just drawn', tone: 'gold'}],
      note: 'New tiles enter on side A, on a highlighted square next to the Duke.'};
  }
  if (game.viewingBagTile) {
    const {name, owner} = game.viewingBagTile;
    const left = state.bags[owner].filter(n => n === name).length;
    return {name, owner, closable: true,
      badges: [left ? {text: `${PLAYER_NAME[owner]}’s bag${left > 1 ? ` ×${left}` : ''}`, tone: 'owner'} : {text: `None left in ${PLAYER_NAME[owner]}’s bag`}],
      note: 'Drawn tiles enter on side A.'};
  }
  const tile = game.selectedTileInstance;
  if (!tile) return null;
  if (game.commandTarget) {
    const id = state.board[game.commandTarget.row][game.commandTarget.col];
    const target = id ? state.tiles.get(id) : undefined;
    if (target) {
      return {name: target.defName, owner: target.owner, side: target.side, closable: true,
        badges: [{text: `Commanded by the ${tile.defName}`, tone: 'gold'}],
        note: 'Pick a highlighted square to send it there. The commander flips, not this tile.'};
    }
  }
  const stuck = game.humanTurn && tile.owner === state.currentPlayer && !game.legalMoves.some(m =>
    (m.type === 'move' || m.type === 'strike') ? sameCoord(m.from, tile.position)
      : m.type === 'command' && sameCoord(m.commander, tile.position));
  return {name: tile.defName, owner: tile.owner, side: tile.side, closable: true,
    badges: stuck ? [{text: 'No legal moves', tone: 'warn'}] : [{text: PLAYER_NAME[tile.owner], tone: 'owner'}],
    note: stuck ? 'No legal moves for this tile right now: it is blocked, or moving it would leave your Duke in check.'
      : 'A tile flips to its other side every time it acts: moves, strikes or commands.'};
}

/**
 * The picked tile: its name and both sides. Laid out as a short row by default
 * (under the board on a phone, in the bag drawer) and as a taller card where
 * the side column has room (controls.css). The help text lives behind the
 * info button in the short form.
 */
export function Inspector({game, bottom}: {game: UseGameReturn; bottom: Orientation}) {
  const [helpOpen, setHelpOpen] = useState(false);
  const subject = inspectorSubject(game);
  if (!subject) return null;
  const def = TILE_REGISTRY.get(subject.name);
  if (!def) return null;
  const rowSign = patternRowSign(subject.owner, bottom);
  const ownerClass = subject.owner === 'P1' ? 'owner-light' : 'owner-dark';
  const current = subject.side ?? 'A';
  return (
    <section className={`card inspector ${ownerClass}`} aria-label={`${subject.name} details`}>
      <header className="inspector-head">
        <TileChip name={subject.name} owner={subject.owner} side={subject.side} />
        <h2 className="inspector-name">{subject.name}</h2>
        <span className="badges">
          {subject.badges.map(b => <span key={b.text} className={`badge ${b.tone ? `badge-${b.tone}` : ''}`}>{b.text}</span>)}
        </span>
        <span className="inspector-tools">
          <button type="button" className="icon-btn inspector-info" onClick={() => setHelpOpen(true)} aria-label="What the symbols mean">
            <InfoIcon />
          </button>
          {subject.closable && (
            <button type="button" className="icon-btn" onClick={game.clearSelection} aria-label="Close">
              <CloseIcon />
            </button>
          )}
        </span>
      </header>
      <div className="sides">
        <SideDiagram letter="A" face={def.sideA} rowSign={rowSign}
                     state={subject.side === undefined ? 'enters' : current === 'A' ? 'up' : 'next'} />
        <SideDiagram letter="B" face={def.sideB} rowSign={rowSign}
                     state={subject.side === undefined ? 'later' : current === 'B' ? 'up' : 'next'} />
      </div>
      {subject.note && <p className="inspector-note">{subject.note}</p>}
      <Legend onOpen={() => setHelpOpen(true)} />
      <SymbolKey open={helpOpen} onClose={() => setHelpOpen(false)} note={subject.note} />
    </section>
  );
}

// [the tall card's caption, the short row's]
const SIDE_STATE = {
  up: ['Face up', 'Face up'],
  next: ['After it moves', 'After a move'],
  enters: ['Enters on this side', 'Enters'],
  later: ['After its first move', 'After a move'],
};

function SideDiagram({letter, face, rowSign, state}: {
  letter: 'A' | 'B'; face: TileSide; rowSign: number; state: keyof typeof SIDE_STATE;
}) {
  const active = state === 'up' || state === 'enters';
  return (
    <figure className={`side ${active ? 'side-active' : ''} ${letter === 'B' ? 'side-b' : ''}`}>
      <MoveDiagram side={face} rowSign={rowSign} size={7} className="side-diagram" />
      <figcaption>
        <strong><span className="side-word">Side </span>{letter}</strong>
        <span className="side-long">{SIDE_STATE[state][0]}</span>
        <span className="side-short">{SIDE_STATE[state][1]}</span>
      </figcaption>
    </figure>
  );
}

/** The move symbols in one line; tap for what each means. */
export function Legend({onOpen}: {onOpen: () => void}) {
  return (
    <button type="button" className="legend" onClick={onOpen}>
      {MOVE_INFO.map(m => (
        <span key={m.type} className="legend-item"><LegendIcon type={m.type} />{m.label}</span>
      ))}
      <span className="legend-more">What’s this?</span>
    </button>
  );
}

/** What each move symbol means, as a dialog. */
export function SymbolKey({open, onClose, note}: {open: boolean; onClose: () => void; note?: string}) {
  return (
    <Modal open={open} onClose={onClose} label="How tiles move">
      <header className="modal-head">
        <h2>How tiles move</h2>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close"><CloseIcon /></button>
      </header>
      {note && <p className="key-note">{note}</p>}
      <dl className="legend-help">
        {MOVE_INFO.map(m => (
          <div key={m.type}>
            <dt><LegendIcon type={m.type} />{m.label}</dt>
            <dd>{m.help}</dd>
          </div>
        ))}
      </dl>
    </Modal>
  );
}
