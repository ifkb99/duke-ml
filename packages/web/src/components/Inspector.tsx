import type {Player, Side, TileSide} from '@the-duke/engine';
import {TILE_REGISTRY} from '@the-duke/engine';
import type {UseGameReturn} from '../hooks/useGame.js';
import {MOVE_INFO, type Orientation, PLAYER_NAME, patternRowSign, sameCoord} from '../ui.js';
import {CloseIcon} from './Icons.js';
import {LegendIcon, MoveDiagram} from './MoveDiagram.js';
import {TileChip} from './Tile.js';

interface Subject {
  name: string;
  owner: Player;
  /** The side facing up; undefined for a tile still in the bag */
  side?: Side;
  badges: {text: string; tone?: 'gold' | 'owner'}[];
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
    return {name, owner, closable: true, badges: [{text: `${PLAYER_NAME[owner]}’s bag`, tone: 'owner'}],
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
    badges: [{text: PLAYER_NAME[tile.owner], tone: 'owner'}],
    note: stuck ? 'No legal moves for this tile right now: it is blocked, or moving it would leave your Duke in check.'
      : 'A tile flips to its other side every time it acts: moves, strikes or commands.'};
}

export function Inspector({game, bottom}: {game: UseGameReturn; bottom: Orientation}) {
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
        {subject.closable && (
          <button type="button" className="icon-btn" onClick={game.clearSelection} aria-label="Close">
            <CloseIcon />
          </button>
        )}
      </header>
      <div className="sides">
        <SideDiagram label="Side A" face={def.sideA} rowSign={rowSign}
                     state={subject.side === undefined ? 'enters' : current === 'A' ? 'up' : 'next'} />
        <SideDiagram label="Side B" face={def.sideB} rowSign={rowSign} b
                     state={subject.side === undefined ? 'later' : current === 'B' ? 'up' : 'next'} />
      </div>
      {subject.note && <p className="inspector-note">{subject.note}</p>}
      <Legend />
    </section>
  );
}

const SIDE_STATE = {up: 'Face up', next: 'After it moves', enters: 'Enters on this side', later: 'After its first move'};

function SideDiagram({label, face, rowSign, state, b}: {
  label: string; face: TileSide; rowSign: number; state: keyof typeof SIDE_STATE; b?: boolean;
}) {
  const active = state === 'up' || state === 'enters';
  return (
    <figure className={`side ${active ? 'side-active' : ''} ${b ? 'side-b' : ''}`}>
      <MoveDiagram side={face} rowSign={rowSign} size={7} className="side-diagram" />
      <figcaption>
        <strong>{label}</strong>
        <span>{SIDE_STATE[state]}</span>
      </figcaption>
    </figure>
  );
}

export function Legend() {
  return (
    <details className="legend">
      <summary>
        {MOVE_INFO.map(m => (
          <span key={m.type} className="legend-item"><LegendIcon type={m.type} />{m.label}</span>
        ))}
      </summary>
      <dl className="legend-help">
        {MOVE_INFO.map(m => (
          <div key={m.type}>
            <dt><LegendIcon type={m.type} />{m.label}</dt>
            <dd>{m.help}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
