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

/** Side indicator — mirrors the physical tile metaphor (edge bar + corner fold for B) */
function SideIndicator({ side, color }: { side: 'A' | 'B'; color: string }) {
  const isSideB = side === 'B';
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: '22px',
      height: '22px',
      borderRadius: '4px',
      background: color,
      position: 'relative',
      overflow: 'hidden',
      fontSize: '0.55rem',
      fontWeight: 700,
      color: '#fff',
      letterSpacing: 0,
    }}>
      {/* Edge bar */}
      <span style={{
        position: 'absolute',
        left: '25%',
        right: '25%',
        height: '2px',
        borderRadius: '1px',
        background: 'rgba(255,255,255,0.5)',
        ...(isSideB ? { top: '1px' } : { bottom: '1px' }),
      }} />
      {/* Corner fold for B */}
      {isSideB && (
        <span style={{
          position: 'absolute',
          top: 0,
          right: 0,
          width: 0,
          height: 0,
          borderStyle: 'solid',
          borderWidth: '0 7px 7px 0',
          borderColor: 'transparent rgba(255,255,255,0.3) transparent transparent',
        }} />
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
      ? 'linear-gradient(135deg, #1e6cb8, #2584d8)'
      : 'linear-gradient(135deg, #b83030, #d04545)';
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
