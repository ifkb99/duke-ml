import type {Player} from '@the-duke/engine';
import type {UseGameReturn} from '../hooks/useGame.js';
import {type Orientation, PLAYER_NAME, other} from '../ui.js';
import {TileChip} from './Tile.js';

/** What's left in each bag; tap a tile to see how it moves. */
export function Bags({game, bottom}: {game: UseGameReturn; bottom: Orientation}) {
  return (
    <section className="card bags" aria-label="Bags">
      <p className="bags-hint">Tiles left to draw. Tap one to see how it moves.</p>
      <div className="bags-columns">
        <Bag game={game} player={bottom} />
        <Bag game={game} player={other(bottom)} />
      </div>
    </section>
  );
}

function Bag({game, player}: {game: UseGameReturn; player: Player}) {
  const bag = game.state.bags[player];
  const counts = new Map<string, number>();
  for (const name of [...bag].sort()) counts.set(name, (counts.get(name) ?? 0) + 1);
  const viewing = game.viewingBagTile?.owner === player ? game.viewingBagTile.name : null;
  const active = game.state.currentPlayer === player && game.state.status === 'active';
  return (
    <div className={`bag ${active ? 'bag-active' : ''} ${player === 'P1' ? 'owner-light' : 'owner-dark'}`}>
      <h3 className="bag-head">
        <span className={`status-dot ${player === 'P1' ? 'dot-light' : 'dot-dark'}`} />
        {PLAYER_NAME[player]}
        <span className="bag-count">{bag.length}</span>
      </h3>
      {bag.length === 0 && <p className="bag-empty">Empty</p>}
      <ul className="bag-tiles">
        {[...counts.entries()].map(([name, count]) => (
          <li key={name}>
            <button type="button" className={`bag-tile ${viewing === name ? 'on' : ''}`}
                    onClick={() => game.onBagTileClick(name, player)} disabled={game.drawMode}>
              <TileChip name={name} owner={player} />
              <span className="bag-tile-name">{name}</span>
              {count > 1 && <span className="bag-tile-count">×{count}</span>}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
