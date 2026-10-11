import type {Player, Side} from '@the-duke/engine';
import {TILE_REGISTRY} from '@the-duke/engine';
import {type Orientation, patternRowSign, shortName} from '../ui.js';
import {MoveDiagram} from './MoveDiagram.js';
import './tile.css';

interface TileFaceProps {
  name: string;
  owner: Player;
  side: Side;
  bottom: Orientation;
  /** Plays the flip animation when it mounts (the side just changed) */
  flipped?: boolean;
}

/** A tile as it lies on the board: its moves for the side facing up, and its name. */
export function TileFace({name, owner, side, bottom, flipped}: TileFaceProps) {
  const def = TILE_REGISTRY.get(name);
  const face = def ? (side === 'B' ? def.sideB : def.sideA) : null;
  const rowSign = patternRowSign(owner, bottom);
  // Longbowman B strikes 2 and 3 squares ahead: shift its center back to fit them
  const center = name === 'Longbowman' && side === 'B' ? {row: rowSign > 0 ? 3 : 1, col: 2} : undefined;
  const classes = ['tile', owner === 'P1' ? 'tile-light' : 'tile-dark', `tile-side-${side.toLowerCase()}`,
    name === 'Duke' && 'tile-duke', flipped && 'tile-flip'].filter(Boolean).join(' ');
  return (
    <div className={classes}>
      {side === 'B' && <span className="tile-corners" aria-hidden />}
      {face && <MoveDiagram side={face} rowSign={rowSign} size={5} center={center} className="tile-diagram" />}
      <span className="tile-name">
        {name === 'Duke' && <Crown />}
        <span className="tile-name-full">{name}</span>
        <span className="tile-name-short">{name === 'Duke' ? 'Duke' : shortName(name)}</span>
      </span>
    </div>
  );
}

/** A small flat tile chip: the owner's colours and the short name (bags, inspector header). */
export function TileChip({name, owner, side}: {name: string; owner: Player; side?: Side}) {
  return (
    <span className={`tile-chip ${owner === 'P1' ? 'tile-light' : 'tile-dark'} ${side === 'B' ? 'tile-side-b' : ''}`}>
      {side === 'B' && <span className="tile-corners" aria-hidden />}
      {shortName(name)}
    </span>
  );
}

export function Crown({className = 'crown'}: {className?: string}) {
  return (
    <svg viewBox="0 0 24 16" className={className} aria-hidden>
      <path d="M2 14 4.5 4.5 9 9.5 12 2.5 15 9.5 19.5 4.5 22 14Z" fill="currentColor" />
      <circle cx="4.5" cy="3.5" r="1.6" fill="currentColor" />
      <circle cx="12" cy="1.8" r="1.6" fill="currentColor" />
      <circle cx="19.5" cy="3.5" r="1.6" fill="currentColor" />
    </svg>
  );
}
