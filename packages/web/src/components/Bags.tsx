import {useRef, type PointerEvent} from 'react';
import {BAG_TILE_NAMES, type Player} from '@the-duke/engine';
import type {UseGameReturn} from '../hooks/useGame.js';
import {type Orientation, PLAYER_NAME, other, shortName} from '../ui.js';
import {ChevronUpIcon, CloseIcon} from './Icons.js';
import {Inspector, inspectorSubject} from './Inspector.js';

/** Every kind of tile a bag starts with. Both views keep this order, so a kind always sits in the same place. */
const KINDS = [...new Set(BAG_TILE_NAMES)].sort();

function countByName(bag: string[]) {
  const counts = new Map<string, number>();
  for (const name of bag) counts.set(name, (counts.get(name) ?? 0) + 1);
  return counts;
}

const ownerClass = (p: Player) => (p === 'P1' ? 'owner-light' : 'owner-dark');
const tileClass = (p: Player) => (p === 'P1' ? 'tile-light' : 'tile-dark');

/** What's left in each bag, small: one row of mini tiles per side. Opens the drawer. */
export function BagStrip({game, bottom, onOpen}: {game: UseGameReturn; bottom: Orientation; onOpen: () => void}) {
  const {bags} = game.state;
  const sides = [other(bottom), bottom];
  return (
    <button type="button" className="card bag-strip" onClick={onOpen}
            aria-label={`Pieces left: ${sides.map(p => `${PLAYER_NAME[p]} ${bags[p].length}`).join(', ')}. Show the list`}>
      <span className="bag-strip-title">Pieces left</span>
      <span className="bag-strip-rows">
        {sides.map(p => {
          const counts = countByName(bags[p]);
          return (
            <span key={p} className={`bag-strip-row ${ownerClass(p)}`}>
              <span className={`status-dot ${p === 'P1' ? 'dot-light' : 'dot-dark'}`} />
              <span className="bag-strip-count">{bags[p].length}</span>
              {KINDS.map(name => {
                const n = counts.get(name) ?? 0;
                return (
                  <span key={name} className={`mini-tile ${n ? tileClass(p) : 'mini-tile-gone'}`}>
                    {n > 0 && shortName(name)}
                    {n > 1 && <i className="mini-tile-count">{n}</i>}
                  </span>
                );
              })}
            </span>
          );
        })}
      </span>
      <ChevronUpIcon />
    </button>
  );
}

/**
 * The bags in full: every kind of tile and how many each side still has, with
 * the picked tile's details at the bottom. A sheet over the lower screen when
 * stacked (drag its header down to close), part of the side column when wide.
 */
export function BagDrawer({game, bottom, onClose}: {game: UseGameReturn; bottom: Orientation; onClose: () => void}) {
  const {bags} = game.state;
  const sides = [bottom, other(bottom)];
  const counts = {P1: countByName(bags.P1), P2: countByName(bags.P2)};
  const viewing = game.viewingBagTile;
  const sheet = useRef<HTMLElement>(null);
  const drag = useRef<{y: number; id: number} | null>(null);

  const pick = (name: string) => {
    const owner = sides.find(p => counts[p].has(name)) ?? bottom;
    game.onBagTileClick(name, owner);
  };

  const onDown = (e: PointerEvent) => {
    const el = sheet.current;
    if (!el || getComputedStyle(el).position !== 'fixed' || (e.target as Element).closest('button')) return;
    drag.current = {y: e.clientY, id: e.pointerId};
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e: PointerEvent) => {
    if (!drag.current || !sheet.current) return;
    sheet.current.style.translate = `0 ${Math.max(0, e.clientY - drag.current.y)}px`;
  };
  const onUp = (e: PointerEvent) => {
    if (!drag.current || !sheet.current) return;
    const dy = e.clientY - drag.current.y;
    drag.current = null;
    sheet.current.style.translate = '';
    if (dy > 60) onClose();
  };

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <section ref={sheet} className="card drawer" role="dialog" aria-label="Pieces left">
        <header className="drawer-head" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
          <span className="drawer-grip" aria-hidden />
          <h2>Pieces left</h2>
          {sides.map(p => (
            <span key={p} className={`drawer-total ${ownerClass(p)}`}>
              <span className={`status-dot ${p === 'P1' ? 'dot-light' : 'dot-dark'}`} />
              {PLAYER_NAME[p]} <b>{bags[p].length}</b>
            </span>
          ))}
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close" autoFocus><CloseIcon /></button>
        </header>
        <ul className="bag-table">
          {KINDS.map(name => (
            <li key={name} className={`bag-row ${viewing?.name === name ? 'on' : ''}`}>
              <button type="button" className="bag-row-name" onClick={() => pick(name)} disabled={game.drawMode}>{name}</button>
              {sides.map(p => {
                const n = counts[p].get(name) ?? 0;
                return (
                  <button key={p} type="button" disabled={game.drawMode}
                          className={`bag-pip ${n ? tileClass(p) : 'bag-pip-gone'} ${ownerClass(p)} ${viewing?.name === name && viewing.owner === p ? 'on' : ''}`}
                          onClick={() => game.onBagTileClick(name, p)}
                          aria-label={`${name}: ${PLAYER_NAME[p]} has ${n} left`}>
                    {n || ''}
                  </button>
                );
              })}
            </li>
          ))}
        </ul>
        <div className="drawer-detail">
          {inspectorSubject(game) ? <Inspector game={game} bottom={bottom} />
            : <p className="drawer-hint">Tap a tile to see how it moves. The numbers are how many each side can still draw.</p>}
        </div>
      </section>
    </>
  );
}
