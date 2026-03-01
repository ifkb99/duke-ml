import type { TileInstance } from '@the-duke/engine';

interface TileProps {
  tile: TileInstance;
}

const TILE_SYMBOLS: Record<string, string> = {
  Duke: 'D',
  Footman: 'Ft',
  Pikeman: 'Pk',
  Knight: 'Kn',
  Longbowman: 'Lb',
  Bowman: 'Bw',
  Champion: 'Ch',
  Wizard: 'Wz',
  Marshall: 'Ma',
  General: 'Ge',
  Ranger: 'Ra',
  Priest: 'Pr',
  Dragoon: 'Dr',
  Assassin: 'As',
  Seer: 'Se',
};

const CORNER_SIZE = 9;
const GOLD = 'var(--side-b)';

function CornerTriangles() {
  const s = CORNER_SIZE;
  const base: React.CSSProperties = { position: 'absolute', width: 0, height: 0 };
  return (
    <>
      <div style={{ ...base, top: 0, left: 0,
        borderLeft: `${s}px solid ${GOLD}`, borderBottom: `${s}px solid transparent` }} />
      <div style={{ ...base, top: 0, right: 0,
        borderRight: `${s}px solid ${GOLD}`, borderBottom: `${s}px solid transparent` }} />
      <div style={{ ...base, bottom: 0, left: 0,
        borderLeft: `${s}px solid ${GOLD}`, borderTop: `${s}px solid transparent` }} />
      <div style={{ ...base, bottom: 0, right: 0,
        borderRight: `${s}px solid ${GOLD}`, borderTop: `${s}px solid transparent` }} />
    </>
  );
}

export function Tile({ tile }: TileProps) {
  const isP1 = tile.owner === 'P1';
  const symbol = TILE_SYMBOLS[tile.defName] ?? tile.defName.slice(0, 2);
  const isDuke = tile.defName === 'Duke';
  const isSideB = tile.side === 'B';

  const bg = isP1
    ? 'linear-gradient(155deg, #0c3460 0%, #164d8a 45%, #2066a8 100%)'
    : 'linear-gradient(155deg, #3d0a18 0%, #6e1828 45%, #8e2838 100%)';

  const innerGlow = isP1
    ? 'rgba(80, 150, 220, 0.12)'
    : 'rgba(200, 80, 100, 0.12)';

  const borderColor = isDuke
    ? 'var(--accent)'
    : isP1 ? 'rgba(70,135,210,0.5)' : 'rgba(180,60,80,0.5)';

  return (
    <div style={{
      width: '88%',
      height: '88%',
      borderRadius: '5px',
      background: bg,
      border: `1.5px solid ${borderColor}`,
      boxShadow: isDuke
        ? `0 0 10px rgba(196,162,74,0.3), inset 0 1px 1px ${innerGlow}, inset 0 -2px 4px rgba(0,0,0,0.3)`
        : `0 2px 4px rgba(0,0,0,0.5), inset 0 1px 1px ${innerGlow}, inset 0 -2px 4px rgba(0,0,0,0.3)`,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '1px',
      color: '#ede8d8',
      fontFamily: 'var(--font-body)',
      fontWeight: 700,
      fontSize: 'clamp(0.55rem, 2vw, 0.82rem)',
      lineHeight: 1,
      userSelect: 'none',
      textShadow: '0 1px 3px rgba(0,0,0,0.6)',
      position: 'relative',
      overflow: 'hidden',
      letterSpacing: '0.01em',
    }}>
      {/* Inner frame */}
      <div style={{
        position: 'absolute',
        inset: '2.5px',
        borderRadius: '3px',
        border: `1px solid ${isP1 ? 'rgba(90,160,230,0.1)' : 'rgba(200,80,100,0.1)'}`,
        pointerEvents: 'none',
      }} />

      {/* Side B: gold corner triangles */}
      {isSideB && <CornerTriangles />}

      {/* Duke crown */}
      {isDuke && (
        <svg
          viewBox="0 0 24 14"
          style={{
            width: 'clamp(10px, 3.2vw, 16px)',
            height: 'auto',
            marginBottom: '-1px',
            filter: 'drop-shadow(0 1px 1px rgba(0,0,0,0.4))',
          }}
        >
          <path
            d="M1 13L4 4L8 9L12 2L16 9L20 4L23 13Z"
            fill="var(--accent)"
          />
          <circle cx="4" cy="3" r="1.5" fill="var(--accent)" />
          <circle cx="12" cy="1" r="1.5" fill="var(--accent)" />
          <circle cx="20" cy="3" r="1.5" fill="var(--accent)" />
        </svg>
      )}

      <span>{symbol}</span>
    </div>
  );
}
