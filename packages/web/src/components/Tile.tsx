import type { TileInstance, MoveType, TileSide } from '@the-duke/engine';
import { TILE_REGISTRY } from '@the-duke/engine';

interface TileProps {
  tile: TileInstance;
}

const GRID = 5;
const CENTER = 2;

const MOVE_COLORS: Record<MoveType | 'self', string> = {
  step: '#5a96cc',
  slide: '#8870b0',
  jump: '#50a868',
  jump_slide: '#40aab0',
  strike: '#c4586a',
  command: '#c4a24a',
  self: '#302e40',
};

const TILE_SHORT: Record<string, string> = {
  Duke: 'Duke',
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

interface CellEntry { type: MoveType; distance: number }
type Cell = CellEntry[];

function buildMiniGrid(side: TileSide, flip: boolean, centerRow: number): Cell[][] {
  const grid: Cell[][] = Array.from(
    { length: GRID },
    () => Array.from({ length: GRID }, (): Cell => []),
  );

  const rowSign = flip ? -1 : 1;

  const add = (r: number, c: number, type: MoveType, dist: number) => {
    if (r >= 0 && r < GRID && c >= 0 && c < GRID) {
      const cell = grid[r][c];
      if (!cell.some(e => e.type === type)) {
        cell.push({ type, distance: dist });
      }
    }
  };

  for (const pattern of side.patterns) {
    switch (pattern.type) {
      case 'step':
      case 'jump':
      case 'strike':
      case 'command':
        for (const { dRow, dCol } of pattern.offsets) {
          add(centerRow + dRow * rowSign, CENTER + dCol, pattern.type, 1);
        }
        break;
      case 'slide':
      case 'jump_slide':
        for (const { dRow, dCol } of pattern.offsets) {
          for (let i = 1; i <= 2; i++) {
            add(centerRow + dRow * rowSign * i, CENTER + dCol * i, pattern.type, i);
          }
        }
        break;
    }
  }

  return grid;
}

/** Longbowman side B needs shifted center so F2/F3 strikes fit */
function needsShiftedCenter(defName: string, side: 'A' | 'B'): boolean {
  return defName === 'Longbowman' && side === 'B';
}

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

/** SVG icon for a move type */
function MoveIcon({ type, distance, size }: { type: MoveType; distance: number; size: number }) {
  const color = MOVE_COLORS[type];
  const opacity = distance === 2 ? 0.6 : 1;
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.32;

  switch (type) {
    case 'step':
      return <circle cx={cx} cy={cy} r={r} fill={color} opacity={opacity} />;
    case 'slide':
      return <circle cx={cx} cy={cy} r={r} fill={color} opacity={opacity} />;
    case 'jump':
      return <circle cx={cx} cy={cy} r={r * 0.85} fill="none" stroke={color} strokeWidth={size * 0.12} opacity={opacity} />;
    case 'jump_slide':
      return <circle cx={cx} cy={cy} r={r * 0.85} fill="none" stroke={color} strokeWidth={size * 0.12} opacity={opacity} />;
    case 'strike': {
      const d = r * 0.65;
      return (
        <g opacity={opacity}>
          <line x1={cx - d} y1={cy - d} x2={cx + d} y2={cy + d} stroke={color} strokeWidth={size * 0.12} strokeLinecap="round" />
          <line x1={cx + d} y1={cy - d} x2={cx - d} y2={cy + d} stroke={color} strokeWidth={size * 0.12} strokeLinecap="round" />
        </g>
      );
    }
    case 'command': {
      const d = r * 0.7;
      return (
        <polygon
          points={`${cx},${cy - d} ${cx + d},${cy} ${cx},${cy + d} ${cx - d},${cy}`}
          fill={color}
          opacity={opacity}
        />
      );
    }
  }
}

function MiniMovesetSVG({ side, flip, centerRow }: { side: TileSide; flip: boolean; centerRow: number }) {
  const grid = buildMiniGrid(side, flip, centerRow);
  const cellSize = 10;
  const svgSize = GRID * cellSize;

  return (
    <svg
      viewBox={`0 0 ${svgSize} ${svgSize}`}
      style={{
        width: '78%',
        height: 'auto',
        aspectRatio: '1',
      }}
    >
      {/* Background fill for grid area */}
      <rect width={svgSize} height={svgSize} fill="rgba(0,0,0,0.25)" rx={1} />

      {/* Grid lines */}
      {Array.from({ length: GRID + 1 }, (_, i) => (
        <g key={`grid-${i}`}>
          <line
            x1={i * cellSize} y1={0} x2={i * cellSize} y2={svgSize}
            stroke="rgba(255,255,255,0.18)" strokeWidth={0.5}
          />
          <line
            x1={0} y1={i * cellSize} x2={svgSize} y2={i * cellSize}
            stroke="rgba(255,255,255,0.18)" strokeWidth={0.5}
          />
        </g>
      ))}

      {/* Outer border */}
      <rect width={svgSize} height={svgSize} fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth={0.8} rx={1} />

      {/* Cells */}
      {grid.map((row, r) =>
        row.map((cell, c) => {
          const x = c * cellSize;
          const y = r * cellSize;
          const isCenter = r === centerRow && c === CENTER;

          if (isCenter) {
            const pad = cellSize * 0.2;
            return (
              <rect
                key={`${r}-${c}`}
                x={x + pad}
                y={y + pad}
                width={cellSize - pad * 2}
                height={cellSize - pad * 2}
                fill={MOVE_COLORS.self}
                stroke="rgba(255,255,255,0.5)"
                strokeWidth={0.6}
                rx={0.8}
              />
            );
          }

          if (cell.length === 0) return null;

          return (
            <g key={`${r}-${c}`} transform={`translate(${x},${y})`}>
              {cell.slice(0, 2).map((entry, i) => (
                <MoveIcon key={i} type={entry.type} distance={entry.distance} size={cellSize} />
              ))}
            </g>
          );
        }),
      )}
    </svg>
  );
}

export function Tile({ tile }: TileProps) {
  const isP1 = tile.owner === 'P1';
  const isDuke = tile.defName === 'Duke';
  const isSideB = tile.side === 'B';
  const label = TILE_SHORT[tile.defName] ?? tile.defName.slice(0, 2);

  const def = TILE_REGISTRY.get(tile.defName);
  const side = def ? (isSideB ? def.sideB : def.sideA) : null;
  const flip = isP1;
  const shiftedCenter = needsShiftedCenter(tile.defName, tile.side);
  const centerRow = shiftedCenter ? (isP1 ? 1 : 3) : CENTER;

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
      gap: '0px',
      color: '#ede8d8',
      fontFamily: 'var(--font-body)',
      fontWeight: 700,
      fontSize: 'clamp(0.4rem, 1.4vw, 0.6rem)',
      lineHeight: 1,
      userSelect: 'none',
      textShadow: '0 1px 3px rgba(0,0,0,0.6)',
      position: 'relative',
      overflow: 'hidden',
      letterSpacing: '0.01em',
      padding: '2px 0',
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
            width: 'clamp(8px, 2.5vw, 13px)',
            height: 'auto',
            marginBottom: '-1px',
            filter: 'drop-shadow(0 1px 1px rgba(0,0,0,0.4))',
            flexShrink: 0,
          }}
        >
          <path d="M1 13L4 4L8 9L12 2L16 9L20 4L23 13Z" fill="var(--accent)" />
          <circle cx="4" cy="3" r="1.5" fill="var(--accent)" />
          <circle cx="12" cy="1" r="1.5" fill="var(--accent)" />
          <circle cx="20" cy="3" r="1.5" fill="var(--accent)" />
        </svg>
      )}

      {/* 5x5 Moveset grid */}
      {side && <MiniMovesetSVG side={side} flip={flip} centerRow={centerRow} />}

      {/* Tile name */}
      <span style={{ zIndex: 1, marginTop: '1px' }}>{label}</span>
    </div>
  );
}
