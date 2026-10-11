import type {CSSProperties} from 'react';
import {type Orientation, PLAYER_NAME, other} from '../ui.js';

/**
 * The network's estimate of who's winning: its value (Light's view, -1..1) as
 * each side's share of the bar, the bottom player's share from the bottom
 * (or the left, when the bar lies above the board).
 */
export function EvalBar({value, bottom}: {value: number | null; bottom: Orientation}) {
  const light = value === null ? 0.5 : (value + 1) / 2;
  const share = bottom === 'P1' ? light : 1 - light;
  const pct = Math.round(share * 100);
  const top = other(bottom);
  const label = value === null ? 'Evaluation: waiting for the first move'
    : `Neural net estimate: ${PLAYER_NAME[bottom]} ${pct}%, ${PLAYER_NAME[top]} ${100 - pct}%`;
  return (
    <div className={`eval ${value === null ? 'eval-unknown' : ''} eval-bottom-${bottom === 'P1' ? 'light' : 'dark'}`}
         style={{'--share': share} as CSSProperties} role="meter" aria-valuemin={0} aria-valuemax={100}
         aria-valuenow={pct} aria-label={label} title={label}>
      <div className="eval-fill" />
      <div className="eval-mid" />
      {value !== null && (
        <>
          <span className="eval-num eval-num-bottom">{pct}</span>
          <span className="eval-num eval-num-top">{100 - pct}</span>
        </>
      )}
    </div>
  );
}
