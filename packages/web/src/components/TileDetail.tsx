import type { Coord, Player, TileInstance, MoveType, TileDefinition } from '@the-duke/engine';
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
  compact?: boolean;
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
      fontSize: '0.85rem',
      fontWeight: 700,
      color,
      fontFamily: 'var(--font-display)',
      letterSpacing: '0.04em',
    }}>
      {name}
    </span>
  );
}

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
      width: '22px',
      height: '22px',
      borderRadius: '4px',
      background: color,
      position: 'relative',
      overflow: 'hidden',
      fontSize: '0.55rem',
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

function MovesetPair({ def, flip, currentSide, cellSize, dimmed }: {
  def: TileDefinition;
  flip: boolean;
  currentSide?: 'A' | 'B';
  cellSize: number;
  dimmed?: boolean;
}) {
  return (
    <div style={{
      display: 'flex',
      gap: '10px',
      justifyContent: 'center',
      flexWrap: 'wrap',
      opacity: dimmed ? 0.25 : 1,
      transition: 'opacity 0.2s',
    }}>
      <MovesetGrid
        side={def.sideA}
        label={currentSide === 'A' ? 'Side A — active' : currentSide ? 'Side A — after flip' : 'Side A (placed)'}
        flipVertical={flip}
        isCurrent={currentSide ? currentSide === 'A' : undefined}
        cellSize={cellSize}
      />
      <MovesetGrid
        side={def.sideB}
        label={currentSide === 'B' ? 'Side B — active' : currentSide ? 'Side B — after flip' : 'Side B (after flip)'}
        flipVertical={flip}
        isCurrent={currentSide ? currentSide === 'B' : undefined}
        cellSize={cellSize}
      />
    </div>
  );
}

/* ── Legend ── */

const MOVE_COLORS: Record<MoveType | 'self', string> = {
  step: '#5a96cc',
  slide: '#8870b0',
  jump: '#50a868',
  jump_slide: '#40aab0',
  strike: '#c4586a',
  command: '#c4a24a',
  self: '#302e40',
};

const SZ = 12;

const LEGEND_ITEMS: { type: MoveType; label: string; icon: () => React.JSX.Element }[] = [
  { type: 'step', label: 'Step', icon: () => (
    <svg width={SZ} height={SZ}><circle cx={SZ/2} cy={SZ/2} r={SZ*0.32} fill={MOVE_COLORS.step} /></svg>
  )},
  { type: 'slide', label: 'Slide', icon: () => (
    <svg width={SZ} height={SZ}><circle cx={SZ/2} cy={SZ/2} r={SZ*0.32} fill={MOVE_COLORS.slide} /></svg>
  )},
  { type: 'jump', label: 'Jump', icon: () => (
    <svg width={SZ} height={SZ}><circle cx={SZ/2} cy={SZ/2} r={SZ*0.26} fill="none" stroke={MOVE_COLORS.jump} strokeWidth={1.4} /></svg>
  )},
  { type: 'jump_slide', label: 'J-Slide', icon: () => (
    <svg width={SZ} height={SZ}><circle cx={SZ/2} cy={SZ/2} r={SZ*0.26} fill="none" stroke={MOVE_COLORS.jump_slide} strokeWidth={1.4} /></svg>
  )},
  { type: 'strike', label: 'Strike', icon: () => {
    const d = SZ * 0.22;
    const c = SZ / 2;
    return (
      <svg width={SZ} height={SZ}>
        <line x1={c-d} y1={c-d} x2={c+d} y2={c+d} stroke={MOVE_COLORS.strike} strokeWidth={1.4} strokeLinecap="round" />
        <line x1={c+d} y1={c-d} x2={c-d} y2={c+d} stroke={MOVE_COLORS.strike} strokeWidth={1.4} strokeLinecap="round" />
      </svg>
    );
  }},
  { type: 'command', label: 'Command', icon: () => {
    const d = SZ * 0.24;
    const c = SZ / 2;
    return (
      <svg width={SZ} height={SZ}>
        <polygon points={`${c},${c-d} ${c+d},${c} ${c},${c+d} ${c-d},${c}`} fill={MOVE_COLORS.command} />
      </svg>
    );
  }},
];

function Legend() {
  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      gap: '2px 10px',
      justifyContent: 'center',
      fontSize: '0.56rem',
      color: 'var(--text-muted)',
    }}>
      {LEGEND_ITEMS.map(({ type, label, icon: Icon }) => (
        <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
          <Icon />
          <span>{label}</span>
        </div>
      ))}
    </div>
  );
}

/* ── Placeholder def for stable sizing ── */

const PLACEHOLDER_DEF = TILE_REGISTRY.get('Footman')!;

/* ── Main ── */

export function TileDetail({
  tile, viewingBagTile, currentPlayer, bagSize, canDraw,
  drawMode, selectedDrawTile, onStartDraw,
  commandTarget, commandTargetTile, compact,
}: TileDetailProps) {

  const cellSize = compact ? 16 : 18;

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '5px',
    padding: compact ? '6px 10px' : '8px 14px',
    background: 'var(--surface)',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--surface-2)',
    width: '100%',
    flexShrink: 0,
  };

  // Header row content + optional extras
  let headerContent: React.JSX.Element;
  let extraContent: React.JSX.Element | null = null;

  // Determine which def/flip/side to show in the grids
  let def: TileDefinition = PLACEHOLDER_DEF;
  let flip = false;
  let currentSide: 'A' | 'B' | undefined;
  let dimmed = false;

  if (drawMode && selectedDrawTile) {
    const d = TILE_REGISTRY.get(selectedDrawTile);
    if (!d) return null;
    def = d;
    flip = currentPlayer === 'P1';
    currentSide = 'A';
    headerContent = (
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <TileName name={def.name} color="var(--accent)" />
        <Badge text="Drawn — place on Side A" color="var(--accent)" />
      </div>
    );
    extraContent = (
      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
        Click a highlighted square adjacent to your Duke
      </div>
    );
  } else if (commandTarget && commandTargetTile && tile) {
    const cmdDef = TILE_REGISTRY.get(commandTargetTile.defName);
    const commanderDef = TILE_REGISTRY.get(tile.defName);
    if (!cmdDef || !commanderDef) return null;
    def = cmdDef;
    flip = commandTargetTile.owner === 'P1';
    currentSide = commandTargetTile.side as 'A' | 'B';
    const ownerColor = tile.owner === 'P1' ? 'var(--p1-color)' : 'var(--p2-color)';
    headerContent = (
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <TileName name={commanderDef.name} color={ownerColor} />
        <Badge text={`Commanding ${cmdDef.name}`} color="var(--accent)" />
      </div>
    );
    extraContent = (
      <div style={{ fontSize: '0.65rem', color: 'var(--accent)' }}>
        Click a highlighted square to move {cmdDef.name}
      </div>
    );
  } else if (viewingBagTile) {
    const d = TILE_REGISTRY.get(viewingBagTile.name);
    if (!d) return null;
    def = d;
    const isP1 = viewingBagTile.owner === 'P1';
    flip = isP1;
    const ownerColor = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';
    headerContent = (
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <TileName name={def.name} color={ownerColor} />
        <Badge text="In Bag" />
        <Badge text={isP1 ? 'Light' : 'Dark'} color={ownerColor} />
      </div>
    );
  } else if (tile) {
    const d = TILE_REGISTRY.get(tile.defName);
    if (!d) return null;
    def = d;
    const isP1 = tile.owner === 'P1';
    flip = isP1;
    currentSide = tile.side as 'A' | 'B';
    const ownerColor = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';
    const gradient = isP1
      ? 'linear-gradient(155deg, #0c3460, #164d8a, #2066a8)'
      : 'linear-gradient(155deg, #3d0a18, #6e1828, #8e2838)';
    const isOwnDuke = tile.owner === currentPlayer && tile.defName === 'Duke';
    headerContent = (
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
        <TileName name={def.name} color={ownerColor} />
        <SideIndicator side={tile.side as 'A' | 'B'} color={gradient} />
        <Badge text={isP1 ? 'Light' : 'Dark'} color={ownerColor} />
      </div>
    );
    if (isOwnDuke && canDraw) {
      extraContent = (
        <button
          onClick={onStartDraw}
          style={{
            padding: '0.3rem 0.8rem',
            borderRadius: 'var(--radius)',
            border: '1.5px solid var(--accent)',
            background: 'transparent',
            color: 'var(--accent)',
            cursor: 'pointer',
            fontSize: '0.72rem',
            fontWeight: 600,
            fontFamily: 'var(--font-body)',
            letterSpacing: '0.02em',
            transition: 'background 0.15s',
          }}
        >
          Draw from Bag ({bagSize})
        </button>
      );
    }
  } else {
    // Nothing selected — show dimmed placeholder grids
    dimmed = true;
    headerContent = (
      <div style={{
        fontSize: '0.68rem',
        color: 'var(--text-muted)',
        fontStyle: 'italic',
      }}>
        Select a tile to view its moveset
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      {headerContent}
      {extraContent}
      <MovesetPair def={def} flip={flip} currentSide={currentSide} cellSize={cellSize} dimmed={dimmed} />
      <Legend />
    </div>
  );
}
