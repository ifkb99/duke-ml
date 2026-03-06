import type {SelfPlayResult, SelfPlayOptions} from '@the-duke/ai';
import {selfPlay} from '@the-duke/ai';

export interface SelfPlayWorkerRequest {
  type: 'runGames';
  count: number;
  options?: SelfPlayOptions;
}

export interface SelfPlayWorkerResponse {
  type: 'gameComplete';
  result: SelfPlayResult;
  gameIndex: number;
}

export interface SelfPlayWorkerDone {
  type: 'allComplete';
  totalGames: number;
  totalSamples: number;
  p1Wins: number;
  p2Wins: number;
  draws: number;
}

self.onmessage = (e: MessageEvent<SelfPlayWorkerRequest>) => {
  const {count, options} = e.data;
  let totalSamples = 0;
  let p1Wins = 0;
  let p2Wins = 0;
  let draws = 0;

  for (let i = 0; i < count; i++) {
    const result = selfPlay(options);
    totalSamples += result.samples.length;

    if (result.outcome === 1) p1Wins++;
    else if (result.outcome === -1) p2Wins++;
    else draws++;

    self.postMessage({
      type: 'gameComplete',
      result,
      gameIndex: i,
    } satisfies SelfPlayWorkerResponse);
  }

  self.postMessage({
    type: 'allComplete',
    totalGames: count,
    totalSamples,
    p1Wins,
    p2Wins,
    draws,
  } satisfies SelfPlayWorkerDone);
};
