import type { Coord, Player, TileInstance } from '@the-duke/engine';
import { TILE_REGISTRY } from '@the-duke/engine';
import { MovesetGrid } from './MovesetGrid.js';

interface TileDetailProps {
  tile: TileInstance | null;
  viewingBagTile: { name: string; owner: Player } | null;
  currentPlayer: Player;
  bagSize: number;
  canDraw: boolean;
  drawMode: boolean;
  selectedDrawTile: string | null;
  onStartDraw: () => void;
  commandTarget: Coord | null;
  commandTargetTile: TileInstance | null;
}

const btnStyle: React.CSSProperties = {
  padding: '0.4rem 0.9rem',
  borderRadius: '6px',
  border: '1px solid var(--surface-2)',
  background: 'var(--surface)',
  color: 'var(--text)',
  cursor: 'pointer',
  fontSize: '0.8rem',
  fontWeight: 500,
  transition: 'background 0.15s, border-color 0.15s',
};

export function TileDetail({
  tile, viewingBagTile, currentPlayer, bagSize, canDraw,
  drawMode, selectedDrawTile, onStartDraw,
  commandTarget, commandTargetTile,
}: TileDetailProps) {
  // Draw mode: show the randomly drawn tile + placement prompt (no cancel)
  if (drawMode && selectedDrawTile) {
    const def = TILE_REGISTRY.get(selectedDrawTile);
    if (!def) return null;
    const flip = currentPlayer === 'P1';

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent)' }}>
            Drew: {def.name}
          </span>
          <span style={{
            fontSize: '0.65rem',
            padding: '1px 6px',
            borderRadius: '4px',
            background: 'var(--surface-2)',
            color: 'var(--text-muted)',
          }}>
            Places on Side A
          </span>
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Click a highlighted square adjacent to your Duke to place
        </div>
        <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <MovesetGrid side={def.sideA} label="Side A" flipVertical={flip} />
          <MovesetGrid side={def.sideB} label="Side B" flipVertical={flip} />
        </div>
      </div>
    );
  }

  // Command mode step 2: picking destination for commanded tile
  if (commandTarget && commandTargetTile && tile) {
    const cmdDef = TILE_REGISTRY.get(commandTargetTile.defName);
    const commanderDef = TILE_REGISTRY.get(tile.defName);
    if (!cmdDef || !commanderDef) return null;
    const isP1 = tile.owner === 'P1';
    const ownerColor = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: ownerColor }}>
            {commanderDef.name}
          </span>
          <span style={{
            fontSize: '0.65rem',
            padding: '1px 6px',
            borderRadius: '4px',
            background: 'rgba(255, 183, 77, 0.2)',
            color: '#ffb74d',
          }}>
            Commanding {cmdDef.name}
          </span>
        </div>
        <div style={{ fontSize: '0.75rem', color: '#ffb74d' }}>
          Click a highlighted square to move {cmdDef.name}
        </div>
      </div>
    );
  }

  // Viewing a bag tile's moveset
  if (viewingBagTile) {
    const def = TILE_REGISTRY.get(viewingBagTile.name);
    if (!def) return null;
    const isP1 = viewingBagTile.owner === 'P1';
    const ownerColor = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';
    const flip = isP1;

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: ownerColor }}>
            {def.name}
          </span>
          <span style={{
            fontSize: '0.65rem',
            padding: '1px 6px',
            borderRadius: '4px',
            background: 'var(--surface-2)',
            color: 'var(--text-muted)',
          }}>
            In Bag
          </span>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            {isP1 ? 'Light' : 'Dark'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <MovesetGrid side={def.sideA} label="Side A (placed)" flipVertical={flip} />
          <MovesetGrid side={def.sideB} label="Side B (after flip)" flipVertical={flip} />
        </div>
      </div>
    );
  }

  // Board tile selected
  if (tile) {
    const def = TILE_REGISTRY.get(tile.defName);
    if (!def) return null;

    const side = tile.side === 'A' ? def.sideA : def.sideB;
    const otherSide = tile.side === 'A' ? def.sideB : def.sideA;
    const isP1 = tile.owner === 'P1';
    const ownerColor = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';
    const flip = isP1;
    const isOwnDuke = tile.owner === currentPlayer && tile.defName === 'Duke';
    const showDrawButton = isOwnDuke && canDraw;

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: ownerColor }}>
            {def.name}
          </span>
          <span style={{
            fontSize: '0.65rem',
            padding: '1px 6px',
            borderRadius: '4px',
            background: 'var(--surface-2)',
            color: 'var(--text-muted)',
          }}>
            Side {tile.side}
          </span>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            {isP1 ? 'Light' : 'Dark'}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <MovesetGrid side={side} label={`Current (${tile.side})`} flipVertical={flip} />
          <MovesetGrid side={otherSide} label={`After flip (${tile.side === 'A' ? 'B' : 'A'})`} flipVertical={flip} />
        </div>

        {showDrawButton && (
          <button
            onClick={onStartDraw}
            style={{
              ...btnStyle,
              background: 'var(--surface-2)',
              borderColor: 'var(--accent)',
              color: 'var(--accent)',
              fontWeight: 600,
            }}
          >
            Draw from Bag ({bagSize})
          </button>
        )}
      </div>
    );
  }

  // Nothing selected
  return (
    <div style={containerStyle}>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', padding: '16px 0' }}>
        Select a tile to view its moveset
      </div>
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: '8px',
  padding: '12px 16px',
  background: 'var(--surface)',
  borderRadius: '8px',
  border: '1px solid var(--surface-2)',
  width: '100%',
  minHeight: '80px',
};
