import type { TileInstance } from '@the-duke/engine';

interface TileProps {
  tile: TileInstance;
}

const TILE_SYMBOLS: Record<string, string> = {
  Duke: 'D',
  Footman: 'F',
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
  const symbol = TILE_SYMBOLS[tile.defName] ?? tile.defName[0];

  return (
    <div style={{
      width: '85%',
      height: '85%',
      borderRadius: '6px',
      background: isP1
        ? 'linear-gradient(135deg, #1565c0, #1976d2)'
        : 'linear-gradient(135deg, #c62828, #d32f2f)',
      border: `2px solid ${isP1 ? 'var(--p1-color)' : 'var(--p2-color)'}`,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#fff',
      fontWeight: 700,
      fontSize: 'clamp(0.6rem, 2vw, 0.9rem)',
      lineHeight: 1.1,
      userSelect: 'none',
      textShadow: '0 1px 2px rgba(0,0,0,0.5)',
      position: 'relative',
    }}>
      <span>{symbol}</span>
      <span style={{
        fontSize: '0.55em',
        opacity: 0.7,
        fontWeight: 400,
      }}>
        {tile.side}
      </span>
    </div>
  );
}
