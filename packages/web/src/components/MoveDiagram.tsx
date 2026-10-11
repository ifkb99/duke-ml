import type {Coord, MoveType, TileSide} from '@the-duke/engine';
import {MOVE_COLOR, patternGrid} from '../ui.js';

/** One move symbol in a unit cell (0..1): the same marks on tiles, in the inspector and in the legend. */
export function MoveIcon({type, x = 0, y = 0, fade = 1}: {type: MoveType; x?: number; y?: number; fade?: number}) {
  const color = MOVE_COLOR[type];
  const cx = x + 0.5, cy = y + 0.5;
  switch (type) {
    case 'step':
      return <circle cx={cx} cy={cy} r={0.3} fill={color} opacity={fade} />;
    case 'slide':
      return (
        <g opacity={fade}>
          <circle cx={cx} cy={cy} r={0.3} fill={color} />
          <circle cx={cx} cy={cy} r={0.13} fill="var(--tile-ink)" opacity={0.55} />
        </g>
      );
    case 'jump':
    case 'jump_slide':
      return (
        <g opacity={fade}>
          <circle cx={cx} cy={cy} r={0.25} fill="none" stroke={color} strokeWidth={0.12} />
          {type === 'jump_slide' && <circle cx={cx} cy={cy} r={0.09} fill={color} />}
        </g>
      );
    case 'strike': {
      const d = 0.22;
      return (
        <g opacity={fade} stroke={color} strokeWidth={0.13} strokeLinecap="round">
          <line x1={cx - d} y1={cy - d} x2={cx + d} y2={cy + d} />
          <line x1={cx + d} y1={cy - d} x2={cx - d} y2={cy + d} />
        </g>
      );
    }
    case 'command': {
      const d = 0.26;
      return <polygon points={`${cx},${cy - d} ${cx + d},${cy} ${cx},${cy + d} ${cx - d},${cy}`} fill={color} opacity={fade} />;
    }
  }
}

export function LegendIcon({type}: {type: MoveType}) {
  return (
    <svg viewBox="0 0 1 1" className="legend-icon" aria-hidden>
      <MoveIcon type={type} />
    </svg>
  );
}

interface MoveDiagramProps {
  side: TileSide;
  /** 1: forward is up, as printed; -1: upside down (see patternRowSign) */
  rowSign: number;
  size: number;
  center?: Coord;
  className?: string;
}

/** A tile side's moves on a size x size grid, the tile itself in the middle. */
export function MoveDiagram({side, rowSign, size, center, className}: MoveDiagramProps) {
  const c = center ?? {row: (size - 1) / 2, col: (size - 1) / 2};
  const grid = patternGrid(side, rowSign, size, c);
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className={className} aria-hidden>
      <rect className="diagram-bg" width={size} height={size} />
      {Array.from({length: size - 1}, (_, i) => (
        <g key={i} className="diagram-line">
          <line x1={i + 1} y1={0} x2={i + 1} y2={size} />
          <line x1={0} y1={i + 1} x2={size} y2={i + 1} />
        </g>
      ))}
      <rect className="diagram-self" x={c.col + 0.18} y={c.row + 0.18} width={0.64} height={0.64} rx={0.1} />
      {grid.map((row, r) => row.map((cell, col) => cell.slice(0, 2).map((e, i) => (
        <MoveIcon key={`${r}-${col}-${i}`} type={e.type} x={col} y={r}
                  fade={e.distance <= 1 ? 1 : Math.max(0.35, 1 - (e.distance - 1) * 0.25)} />
      ))))}
    </svg>
  );
}
