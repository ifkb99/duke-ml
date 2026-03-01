import type { MoveType, TileSide } from '@the-duke/engine';

interface MovesetGridProps {
  side: TileSide;
  label: string;
  flipVertical?: boolean;
  isCurrent?: boolean;
}

const GRID_SIZE = 7;
const CENTER = 3;

const MOVE_COLORS: Record<MoveType | 'self', string> = {
  step:       '#5a96cc',
  slide:      '#8870b0',
  jump:       '#50a868',
  jump_slide: '#40aab0',
  strike:     '#c4586a',
  command:    '#c4a24a',
  self:       '#302e40',
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

export function MovesetGrid({ side, label, flipVertical = false, isCurrent }: MovesetGridProps) {
  const grid = buildGrid(side, flipVertical);
  const types = usedTypes(side);
  const cellSize = 22;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '4px',
      padding: '6px',
      borderRadius: 'var(--radius)',
      border: isCurrent
        ? '1.5px solid var(--accent)'
        : isCurrent === false
          ? '1.5px solid var(--surface-3)'
          : '1.5px solid transparent',
      background: isCurrent
        ? 'rgba(196,162,74,0.05)'
        : 'transparent',
      opacity: isCurrent === false ? 0.5 : 1,
      transition: 'opacity 0.2s, border-color 0.2s, background 0.2s',
    }}>
      <div style={{
        fontSize: '0.65rem',
        color: isCurrent ? 'var(--accent)' : 'var(--text-muted)',
        fontWeight: isCurrent ? 600 : 500,
        letterSpacing: '0.02em',
      }}>
        {label}
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${GRID_SIZE}, ${cellSize}px)`,
        gap: '1px',
      }}>
        {grid.map((row, r) =>
          row.map((cell, c) => {
            const isCenter = r === CENTER && c === CENTER;
            const color = isCenter
              ? MOVE_COLORS.self
              : cell ? MOVE_COLORS[cell.type] : 'transparent';
            const opacity = cell?.distance === 3 ? 0.45 : cell?.distance === 2 ? 0.7 : 1;

            return (
              <div
                key={`${r}-${c}`}
                style={{
                  width: `${cellSize}px`,
                  height: `${cellSize}px`,
                  borderRadius: isCenter ? '4px' : cell ? '3px' : '0',
                  background: color,
                  opacity: isCenter ? 1 : cell ? opacity : 1,
                  border: isCenter
                    ? '1.5px solid var(--text-muted)'
                    : cell
                      ? `1px solid ${color}`
                      : '1px solid rgba(255,255,255,0.03)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.45rem',
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

      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', justifyContent: 'center' }}>
        {types.map(t => (
          <div key={t} style={{
            display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.56rem',
          }}>
            <div style={{
              width: '7px', height: '7px', borderRadius: '2px',
              background: MOVE_COLORS[t],
            }} />
            <span style={{ color: 'var(--text-muted)' }}>{MOVE_LABELS[t]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
