import type { MovePattern, MoveType, TileSide } from '@the-duke/engine';

interface MovesetGridProps {
  side: TileSide;
  label: string;
  /** Flip rows so "forward" points downward (for P1 pieces at the top of the board). */
  flipVertical?: boolean;
}

const GRID_SIZE = 7;
const CENTER = 3;

const MOVE_COLORS: Record<MoveType | 'self', string> = {
  step:       '#64b5f6',
  slide:      '#ba68c8',
  jump:       '#81c784',
  jump_slide: '#4dd0e1',
  strike:     '#ef5350',
  command:    '#ffb74d',
  self:       '#555',
};

const MOVE_LABELS: Record<MoveType, string> = {
  step:       'Step',
  slide:      'Slide',
  jump:       'Jump',
  jump_slide: 'J-Slide',
  strike:     'Strike',
  command:    'Command',
};

type CellInfo = { type: MoveType; distance: number } | null;

function buildGrid(side: TileSide, flip: boolean): CellInfo[][] {
  const grid: CellInfo[][] = Array.from(
    { length: GRID_SIZE },
    () => Array(GRID_SIZE).fill(null),
  );

  // When flip is true (P1 at top of board), negate dRow so "forward" points down visually.
  const rowSign = flip ? -1 : 1;

  for (const pattern of side.patterns) {
    switch (pattern.type) {
      case 'step':
      case 'jump':
      case 'strike':
      case 'command':
        for (const { dRow, dCol } of pattern.offsets) {
          const r = CENTER + dRow * rowSign;
          const c = CENTER + dCol;
          if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE) {
            grid[r][c] = { type: pattern.type, distance: 1 };
          }
        }
        break;

      case 'slide':
        for (const { dRow, dCol } of pattern.offsets) {
          for (let i = 1; i <= 3; i++) {
            const r = CENTER + dRow * rowSign * i;
            const c = CENTER + dCol * i;
            if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE) {
              grid[r][c] = { type: 'slide', distance: i };
            }
          }
        }
        break;

      case 'jump_slide':
        for (const { dRow, dCol } of pattern.offsets) {
          for (let i = 2; i <= 3; i++) {
            const r = CENTER + dRow * rowSign * i;
            const c = CENTER + dCol * i;
            if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE) {
              grid[r][c] = { type: 'jump_slide', distance: i };
            }
          }
        }
        break;
    }
  }

  return grid;
}

function usedTypes(side: TileSide): MoveType[] {
  const types = new Set<MoveType>();
  for (const p of side.patterns) types.add(p.type);
  return [...types];
}

export function MovesetGrid({ side, label, flipVertical = false }: MovesetGridProps) {
  const grid = buildGrid(side, flipVertical);
  const types = usedTypes(side);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 500 }}>{label}</div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)`,
        gap: '1px',
        width: `${GRID_SIZE * 26}px`,
        height: `${GRID_SIZE * 26}px`,
      }}>
        {grid.map((row, r) =>
          row.map((cell, c) => {
            const isCenter = r === CENTER && c === CENTER;
            const color = isCenter
              ? MOVE_COLORS.self
              : cell
                ? MOVE_COLORS[cell.type]
                : 'transparent';
            const opacity = cell?.distance === 3 ? 0.5 : cell?.distance === 2 ? 0.75 : 1;

            return (
              <div
                key={`${r}-${c}`}
                style={{
                  width: '25px',
                  height: '25px',
                  borderRadius: isCenter ? '4px' : cell ? '3px' : '0',
                  background: color,
                  opacity: isCenter ? 1 : cell ? opacity : 1,
                  border: isCenter
                    ? '2px solid #888'
                    : cell
                      ? `1px solid ${color}`
                      : '1px solid rgba(255,255,255,0.04)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.5rem',
                  color: '#fff',
                  fontWeight: 600,
                }}
              >
                {isCenter && '●'}
              </div>
            );
          }),
        )}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
        {types.map(t => (
          <div key={t} style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.6rem' }}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '2px',
              background: MOVE_COLORS[t],
            }} />
            <span style={{ color: 'var(--text-muted)' }}>{MOVE_LABELS[t]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
