import type { Player } from '@the-duke/engine';

interface BagPanelProps {
  player: Player;
  bag: string[];
  isActive: boolean;
  viewingTile: string | null;
  onTileClick: (name: string, owner: Player) => void;
}

const TILE_SHORT: Record<string, string> = {
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

export function BagPanel({
  player, bag, isActive, viewingTile, onTileClick,
}: BagPanelProps) {
  const counts = new Map<string, number>();
  for (const name of bag) {
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }

  const isP1 = player === 'P1';
  const color = isP1 ? 'var(--p1-color)' : 'var(--p2-color)';
  const label = isP1 ? 'Light' : 'Dark';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '3px',
      minWidth: '72px',
      maxWidth: '84px',
      opacity: bag.length === 0 ? 0.4 : 1,
    }}>
      <div style={{
        fontSize: '0.7rem',
        fontWeight: 600,
        color: isActive ? color : 'var(--text-muted)',
        textAlign: 'center',
        paddingBottom: '2px',
        borderBottom: isActive ? `2px solid ${color}` : '1px solid var(--surface-2)',
        letterSpacing: '0.03em',
      }}>
        {label} Bag ({bag.length})
      </div>

      {bag.length === 0 && (
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textAlign: 'center', padding: '8px 0' }}>
          Empty
        </div>
      )}

      {[...counts.entries()].map(([name, count]) => {
        const isViewing = viewingTile === name;

        return (
          <div
            key={name}
            onClick={() => onTileClick(name, player)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '3px 6px',
              borderRadius: '4px',
              fontSize: '0.65rem',
              background: isViewing ? 'var(--selected)' : 'var(--surface)',
              cursor: 'pointer',
              border: isViewing ? `1px solid ${color}` : '1px solid transparent',
              transition: 'background 0.15s',
            }}
          >
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '18px',
              height: '18px',
              borderRadius: '3px',
              background: isP1
                ? 'linear-gradient(135deg, #1565c0, #1976d2)'
                : 'linear-gradient(135deg, #c62828, #d32f2f)',
              color: '#fff',
              fontSize: '0.55rem',
              fontWeight: 700,
              flexShrink: 0,
            }}>
              {TILE_SHORT[name] ?? name[0]}
            </span>
            <span style={{ color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {name}
            </span>
            {count > 1 && (
              <span style={{ color: 'var(--text-muted)', marginLeft: 'auto', fontSize: '0.6rem' }}>
                ×{count}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
