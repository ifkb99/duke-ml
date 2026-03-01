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

export function Tile({ tile }: TileProps) {
  const isP1 = tile.owner === 'P1';
  const symbol = TILE_SYMBOLS[tile.defName] ?? tile.defName.slice(0, 2);
  const isDuke = tile.defName === 'Duke';
  const isSideB = tile.side === 'B';

  const gradientAngle = isSideB ? '325deg' : '145deg';
  const p1Gradient = `linear-gradient(${gradientAngle}, #1e6cb8, #2584d8)`;
  const p2Gradient = `linear-gradient(${gradientAngle}, #b83030, #d04545)`;

  const edgeColor = isDuke
    ? 'var(--accent)'
    : isP1 ? 'rgba(91,155,213,0.6)' : 'rgba(212,85,85,0.6)';

  return (
    <div style={{
      width: '86%',
      height: '86%',
      borderRadius: '5px',
      background: isP1 ? p1Gradient : p2Gradient,
      border: isDuke
        ? '2px solid var(--accent)'
        : `1.5px solid ${isP1 ? 'rgba(91,155,213,0.45)' : 'rgba(212,85,85,0.45)'}`,
      boxShadow: isDuke
        ? '0 0 8px rgba(201,168,76,0.25)'
        : '0 1px 3px rgba(0,0,0,0.4)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#fff',
      fontFamily: 'var(--font-body)',
      fontWeight: 700,
      fontSize: 'clamp(0.55rem, 2vw, 0.8rem)',
      lineHeight: 1,
      userSelect: 'none',
      textShadow: '0 1px 2px rgba(0,0,0,0.5)',
      position: 'relative',
      letterSpacing: '-0.01em',
      overflow: 'hidden',
    }}>
      {/* Edge bar: bottom for Side A, top for Side B */}
      <div style={{
        position: 'absolute',
        left: '20%',
        right: '20%',
        height: '2.5px',
        borderRadius: '1px',
        background: edgeColor,
        ...(isSideB ? { top: '1px' } : { bottom: '1px' }),
      }} />

      {/* Corner fold for Side B — a small triangle suggesting the tile is flipped */}
      {isSideB && (
        <div style={{
          position: 'absolute',
          top: 0,
          right: 0,
          width: 0,
          height: 0,
          borderStyle: 'solid',
          borderWidth: '0 10px 10px 0',
          borderColor: `transparent ${isP1 ? 'rgba(130,190,255,0.35)' : 'rgba(255,140,140,0.35)'} transparent transparent`,
        }} />
      )}

      <span>{symbol}</span>
    </div>
  );
}
