import type {MoveType, TileSide} from '@the-duke/engine';

interface MovesetGridProps {
    side: TileSide;
    label: string;
    flipVertical?: boolean;
    isCurrent?: boolean;
}

const GRID_SIZE = 7;
const CENTER = 3;

const MOVE_COLORS: Record<MoveType | 'self', string> = {
    step: '#5a96cc',
    slide: '#8870b0',
    jump: '#50a868',
    jump_slide: '#40aab0',
    strike: '#c4586a',
    command: '#c4a24a',
    self: '#302e40',
};

const MOVE_LABELS: Record<MoveType, string> = {
    step: 'Step',
    slide: 'Slide',
    jump: 'Jump',
    jump_slide: 'J-Slide',
    strike: 'Strike',
    command: 'Command',
};

interface CellEntry {type: MoveType; distance: number}
type CellInfo = CellEntry[];

function buildGrid(side: TileSide, flip: boolean): CellInfo[][] {
    const grid: CellInfo[][] = Array.from(
        {length: GRID_SIZE},
        () => Array.from({length: GRID_SIZE}, () => []),
    );

    const rowSign = flip ? -1 : 1;

    const addEntry = (r: number, c: number, type: MoveType, distance: number) => {
        if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE) {
            const existing = grid[r][c];
            if (!existing.some(e => e.type === type)) {
                existing.push({type, distance});
            }
        }
    };

    for (const pattern of side.patterns) {
        switch (pattern.type) {
            case 'step':
            case 'jump':
            case 'strike':
            case 'command':
                for (const {dRow, dCol} of pattern.offsets) {
                    addEntry(CENTER + dRow * rowSign, CENTER + dCol, pattern.type, 1);
                }
                break;

            case 'slide':
                for (const {dRow, dCol} of pattern.offsets) {
                    for (let i = 1; i <= 3; i++) {
                        addEntry(CENTER + dRow * rowSign * i, CENTER + dCol * i, 'slide', i);
                    }
                }
                break;

            case 'jump_slide':
                for (const {dRow, dCol} of pattern.offsets) {
                    for (let i = 1; i <= 3; i++) {
                        addEntry(CENTER + dRow * rowSign * i, CENTER + dCol * i, 'jump_slide', i);
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

function SingleCell({entry, size}: {entry: CellEntry; size: number}) {
    const color = MOVE_COLORS[entry.type];
    const opacity = entry.distance === 3 ? 0.45 : entry.distance === 2 ? 0.7 : 1;
    return (
        <div style={{
            width: `${size}px`,
            height: `${size}px`,
            borderRadius: '3px',
            background: color,
            opacity,
            border: `1px solid ${color}`,
        }} />
    );
}

function DualCell({entries, size}: {entries: [CellEntry, CellEntry]; size: number}) {
    const c1 = MOVE_COLORS[entries[0].type];
    const c2 = MOVE_COLORS[entries[1].type];
    return (
        <div style={{
            width: `${size}px`,
            height: `${size}px`,
            borderRadius: '3px',
            position: 'relative',
            overflow: 'hidden',
            border: `1px solid rgba(255,255,255,0.1)`,
        }}>
            {/* Top-left triangle */}
            <div style={{
                position: 'absolute',
                inset: 0,
                background: c1,
                clipPath: 'polygon(0 0, 100% 0, 0 100%)',
            }} />
            {/* Bottom-right triangle */}
            <div style={{
                position: 'absolute',
                inset: 0,
                background: c2,
                clipPath: 'polygon(100% 0, 100% 100%, 0 100%)',
            }} />
        </div>
    );
}

export function MovesetGrid({side, label, flipVertical = false, isCurrent}: MovesetGridProps) {
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

                        if (isCenter) {
                            return (
                                <div
                                    key={`${r}-${c}`}
                                    style={{
                                        width: `${cellSize}px`,
                                        height: `${cellSize}px`,
                                        borderRadius: '4px',
                                        background: MOVE_COLORS.self,
                                        border: '1.5px solid var(--text-muted)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        fontSize: '0.45rem',
                                        color: '#fff',
                                        fontWeight: 600,
                                    }}
                                >
                                    ●
                                </div>
                            );
                        }

                        if (cell.length === 0) {
                            return (
                                <div
                                    key={`${r}-${c}`}
                                    style={{
                                        width: `${cellSize}px`,
                                        height: `${cellSize}px`,
                                        border: '1px solid rgba(255,255,255,0.03)',
                                    }}
                                />
                            );
                        }

                        if (cell.length === 1) {
                            return <SingleCell key={`${r}-${c}`} entry={cell[0]} size={cellSize} />;
                        }

                        return (
                            <DualCell
                                key={`${r}-${c}`}
                                entries={[cell[0], cell[1]]}
                                size={cellSize}
                            />
                        );
                    }),
                )}
            </div>

            <div style={{display: 'flex', gap: '6px', flexWrap: 'wrap', justifyContent: 'center'}}>
                {types.map(t => (
                    <div key={t} style={{
                        display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.56rem',
                    }}>
                        <div style={{
                            width: '7px', height: '7px', borderRadius: '2px',
                            background: MOVE_COLORS[t],
                        }} />
                        <span style={{color: 'var(--text-muted)'}}>{MOVE_LABELS[t]}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
