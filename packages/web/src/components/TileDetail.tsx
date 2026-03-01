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

function Badge({ text, color }: { text: string; color?: string }) {
  return (
    <span style={{
      fontSize: '0.6rem',
      padding: '1px 7px',
      borderRadius: '4px',
      background: color ? `${color}18` : 'var(--surface-3)',
      color: color ?? 'var(--text-muted)',
      fontWeight: 500,
      letterSpacing: '0.02em',
    }}>
      {text}
    </span>
  );
}

function TileName({ name, color }: { name: string; color: string }) {
  return (
    <span style={{
      fontSize: '0.9rem',
      fontWeight: 700,
      color,
      fontFamily: 'var(--font-display)',
      letterSpacing: '0.04em',
    }}>
      {name}
    </span>
  );
}

/** Side indicator badge — miniature of the board tile design */
function SideIndicator({ side, color }: { side: 'A' | 'B'; color: string }) {
  const isSideB = side === 'B';
  const s = 5;
  const purple = 'var(--side-b)';
  const triBase: React.CSSProperties = { position: 'absolute', width: 0, height: 0 };
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: '24px',
      height: '24px',
      borderRadius: '4px',
      background: color,
      position: 'relative',
      overflow: 'hidden',
      fontSize: '0.6rem',
      fontWeight: 700,
      color: '#ede8d8',
      letterSpacing: 0,
      boxShadow: 'inset 0 1px 1px rgba(255,255,255,0.1), inset 0 -1px 2px rgba(0,0,0,0.3)',
    }}>
      {isSideB && (
        <>
          <span style={{ ...triBase, top: 0, left: 0,
            borderLeft: `${s}px solid ${purple}`, borderBottom: `${s}px solid transparent` }} />
          <span style={{ ...triBase, top: 0, right: 0,
            borderRight: `${s}px solid ${purple}`, borderBottom: `${s}px solid transparent` }} />
          <span style={{ ...triBase, bottom: 0, left: 0,
            borderLeft: `${s}px solid ${purple}`, borderTop: `${s}px solid transparent` }} />
          <span style={{ ...triBase, bottom: 0, right: 0,
            borderRight: `${s}px solid ${purple}`, borderTop: `${s}px solid transparent` }} />
        </>
      )}
      {side}
    </span>
  );
}

function MovesetPair({ def, flip, currentSide }: {
  def: { name: string; sideA: any; sideB: any };
  flip: boolean;
  currentSide?: 'A' | 'B';
}) {
  return (
    <div style={{
      display: 'flex',
      gap: '12px',
      justifyContent: 'center',
      flexWrap: 'wrap',
    }}>
      <MovesetGrid
        side={def.sideA}
        label={currentSide === 'A' ? 'Side A — active' : currentSide ? 'Side A — after flip' : 'Side A (placed)'}
        flipVertical={flip}
        isCurrent={currentSide ? currentSide === 'A' : undefined}
      />
      <MovesetGrid
        side={def.sideB}
        label={currentSide === 'B' ? 'Side B — active' : currentSide ? 'Side B — after flip' : 'Side B (after flip)'}
        flipVertical={flip}
        isCurrent={currentSide ? currentSide === 'B' : undefined}
      />
    </div>
  );
}

export function TileDetail({
  tile, viewingBagTile, currentPlayer, bagSize, canDraw,
  drawMode, selectedDrawTile, onStartDraw,
  commandTarget, commandTargetTile,
}: TileDetailProps) {

  if (drawMode && selectedDrawTile) {
    const def = TILE_REGISTRY.get(selectedDrawTile);
    if (!def) return null;
    const flip = currentPlayer === 'P1';

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <TileName name={def.name} color="var(--accent)" />
          <Badge text="Drawn — place on Side A" color="var(--accent)" />
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          Click a highlighted square adjacent to your Duke
        </div>
        <MovesetPair def={def} flip={flip} currentSide="A" />
      </div>
    );
  }

  if (commandTarget && commandTargetTile && tile) {
    const cmdDef = TILE_REGISTRY.get(commandTargetTile.defName);
    const commanderDef = TILE_REGISTRY.get(tile.defName);
    if (!cmdDef || !commanderDef) return null;
    const ownerColor = tile.owner === 'P1' ? 'var(--p1-color)' : 'var(--p2-color)';

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <TileName name={commanderDef.name} color={ownerColor} />
          <Badge text={`Commanding ${cmdDef.name}`} color="var(--accent)" />
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--accent)' }}>
          Click a highlighted square to move {cmdDef.name}
        </div>
      </div>
    );
  }

  if (viewingBagTile) {
    const def = TILE_REGISTRY.get(viewingBagTile.name);
    if (!def) return null;
    const isP1 = viewingBagTile.owner === 'P1';
    const ownerColor = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <TileName name={def.name} color={ownerColor} />
          <Badge text="In Bag" />
          <Badge text={isP1 ? 'Light' : 'Dark'} color={ownerColor} />
        </div>
        <MovesetPair def={def} flip={isP1} />
      </div>
    );
  }

  if (tile) {
    const def = TILE_REGISTRY.get(tile.defName);
    if (!def) return null;

    const isP1 = tile.owner === 'P1';
    const ownerColor = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';
    const gradient = isP1
      ? 'linear-gradient(155deg, #0c3460, #164d8a, #2066a8)'
      : 'linear-gradient(155deg, #3d0a18, #6e1828, #8e2838)';
    const isOwnDuke = tile.owner === currentPlayer && tile.defName === 'Duke';

    return (
      <div style={containerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <TileName name={def.name} color={ownerColor} />
          <SideIndicator side={tile.side as 'A' | 'B'} color={gradient} />
          <Badge text={isP1 ? 'Light' : 'Dark'} color={ownerColor} />
        </div>

        <MovesetPair
          def={{ name: def.name, sideA: def.sideA, sideB: def.sideB }}
          flip={isP1}
          currentSide={tile.side as 'A' | 'B'}
        />

        {isOwnDuke && canDraw && (
          <button
            onClick={onStartDraw}
            style={{
              padding: '0.4rem 1rem',
              borderRadius: 'var(--radius)',
              border: '1.5px solid var(--accent)',
              background: 'transparent',
              color: 'var(--accent)',
              cursor: 'pointer',
              fontSize: '0.78rem',
              fontWeight: 600,
              fontFamily: 'var(--font-body)',
              letterSpacing: '0.02em',
              transition: 'background 0.15s',
              marginTop: '2px',
            }}
          >
            Draw from Bag ({bagSize})
          </button>
        )}
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <div style={{
        fontSize: '0.72rem',
        color: 'var(--text-muted)',
        textAlign: 'center',
        padding: '10px 0',
        fontStyle: 'italic',
      }}>
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
  borderRadius: 'var(--radius-lg)',
  border: '1px solid var(--surface-2)',
  width: '100%',
  minHeight: '60px',
};
