import type {SerializedGameState, GameMove} from '@the-duke/engine';
import {deserialize} from '@the-duke/engine';
import {findBestMove, pickRandomMove} from '@the-duke/ai';

export interface AIWorkerRequest {
  type: 'findBestMove' | 'pickRandomMove';
  state: SerializedGameState;
  depth?: number;
}

export interface AIWorkerResponse {
  move: GameMove | null;
  score: number;
  nodesSearched: number;
  error?: string;
}

self.onmessage = (e: MessageEvent<AIWorkerRequest>) => {
  try {
    const {type, state: serialized, depth} = e.data;
    const state = deserialize(serialized);

    if (type === 'pickRandomMove') {
      const move = pickRandomMove(state);
      self.postMessage({move, score: 0, nodesSearched: 0} satisfies AIWorkerResponse);
      return;
    }

    const d = depth ?? 4;
    const t0 = performance.now();
    const result = findBestMove(state, d);
    const elapsed = performance.now() - t0;
    console.log(`[AI] depth=${d} nodes=${result.nodesSearched} time=${elapsed.toFixed(0)}ms score=${result.score}`);
    self.postMessage({
      move: result.move,
      score: result.score,
      nodesSearched: result.nodesSearched,
    } satisfies AIWorkerResponse);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[AI Worker Error]', err);
    self.postMessage({move: null, score: 0, nodesSearched: 0, error: msg} satisfies AIWorkerResponse);
  }
};
