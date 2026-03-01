import type { Player } from '@the-duke/engine';

interface BagPanelProps {
  player: Player;
  bag: string[];
  isActive: boolean;
  viewingTile: string | null;
  onTileClick: (name: string, owner: Player) => void;
  compact?: boolean;
}

const TILE_SHORT: Record<string, string> = {
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

export function BagPanel({
  player, bag, isActive, viewingTile, onTileClick, compact,
}: BagPanelProps) {
  const counts = new Map<string, number>();
  for (const name of bag) {
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }

  const isP1 = player === 'P1';
  const color = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';
  const label = isP1 ? 'Light' : 'Dark';
  const gradient = isP1
    ? 'linear-gradient(155deg, #0c3460, #164d8a, #2066a8)'
    : 'linear-gradient(155deg, #3d0a18, #6e1828, #8e2838)';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '5px',
      flex: compact ? '1 1 0' : undefined,
      minWidth: compact ? 0 : '130px',
      maxWidth: compact ? undefined : '150px',
      opacity: bag.length === 0 ? 0.35 : 1,
      transition: 'opacity 0.3s',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'baseline',
        gap: '6px',
        padding: '0 2px 4px',
        borderBottom: `1.5px solid ${isActive ? color : 'var(--surface-3)'}`,
        transition: 'border-color 0.3s',
      }}>
        <span style={{
          fontSize: '0.72rem',
          fontWeight: 600,
          color: isActive ? color : 'var(--text-muted)',
          letterSpacing: '0.04em',
          transition: 'color 0.3s',
        }}>
          {label}
        </span>
        <span style={{
          fontSize: '0.62rem',
          color: 'var(--text-muted)',
          fontWeight: 400,
        }}>
          {bag.length}
        </span>
      </div>

      {bag.length === 0 && (
        <div style={{
          fontSize: '0.65rem',
          color: 'var(--text-muted)',
          textAlign: 'center',
          padding: '8px 0',
          fontStyle: 'italic',
        }}>
          Empty
        </div>
      )}

      {/* Tile chips */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '3px',
      }}>
        {[...counts.entries()].map(([name, count]) => {
          const isViewing = viewingTile === name;
          const abbr = TILE_SHORT[name] ?? name.slice(0, 2);

          return (
            <div
              key={name}
              onClick={() => onTileClick(name, player)}
              title={name}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '3px',
                padding: '3px 5px',
                borderRadius: 'var(--radius)',
                fontSize: '0.62rem',
                background: isViewing ? 'var(--surface-3)' : 'var(--surface)',
                cursor: 'pointer',
                border: isViewing ? `1.5px solid ${color}` : '1.5px solid var(--surface-2)',
                transition: 'all 0.15s',
                whiteSpace: 'nowrap',
                userSelect: 'none',
              }}
            >
              <span style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '18px',
                height: '18px',
                borderRadius: '4px',
                background: gradient,
                color: '#ede8d8',
                fontSize: '0.55rem',
                fontWeight: 700,
                flexShrink: 0,
                letterSpacing: '-0.02em',
                boxShadow: 'inset 0 1px 1px rgba(255,255,255,0.1), inset 0 -1px 2px rgba(0,0,0,0.25)',
              }}>
                {abbr}
              </span>
              {!compact && (
                <span style={{
                  color: isViewing ? 'var(--text)' : 'var(--text-muted)',
                  fontWeight: isViewing ? 500 : 400,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  maxWidth: '80px',
                }}>
                  {name}
                </span>
              )}
              {count > 1 && (
                <span style={{
                  fontSize: '0.55rem',
                  color: 'var(--text-muted)',
                  fontWeight: 500,
                }}>
                  {count}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
