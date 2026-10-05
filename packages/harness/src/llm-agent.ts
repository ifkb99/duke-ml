/**
 * LLM agent hitting any OpenAI-compatible chat endpoint — vLLM, llama.cpp
 * server, or a hosted API. Point baseUrl at your local Qwen:
 *
 *   new LlmAgent({ baseUrl: 'http://localhost:8000/v1', model: 'qwen3.6-35b', ... })
 *
 * Ablation knobs (the experimental variables):
 *   representation: 'ascii' | 'coords'     — board rendering
 *   legalMoves:     boolean                — include legal move list in prompt
 *   cot:            boolean                — ask for reasoning before the move
 *
 * Protocol per move:
 *   1. Build prompt, call model (up to maxRetries on illegal/unparseable).
 *   2. Retries append an error message so the model sees what it did wrong.
 *   3. On exhaustion, fall back to a uniform-random legal move and record it.
 *      (Fallback keeps games finishable; the fallback RATE is itself a metric.)
 */
import type { GameMove, GameState } from '@the-duke/engine';
import { renderAscii, renderCoordList, renderAbilities, renderBag } from './render';
import { moveToString, parseMove, movesEqual } from './notation';
import { makeRng, type Agent, type MoveResult } from './runner';

export interface LlmAgentConfig {
  baseUrl: string;               // e.g. http://localhost:8000/v1
  model: string;
  apiKey?: string;               // 'local' is fine for local servers
  name?: string;
  representation: 'ascii' | 'coords';
  legalMoves: boolean;
  cot: boolean;
  temperature?: number;
  maxRetries?: number;           // retries per move on illegal/unparseable
  seed?: number;                 // for the random fallback
}

const RULES_PREAMBLE = `You are playing The Duke, a 6x6 tile-based strategy game.
Rules summary:
- Each tile has two sides with different movement patterns. EVERY TIME a tile moves, strikes, or issues a command, it FLIPS to its other side (a commanded tile does NOT flip). Track sides carefully: a tile's legal moves next turn come from its NEW side.
- Win by capturing the enemy Duke, or by leaving the opponent with no legal moves.
- You may also PLACE a new tile from your bag on an empty square orthogonally adjacent to your Duke (this does not flip anything).
- Move types: step (blocked by pieces on the path), slide (any distance until blocked), jump (ignores intervening pieces), jump-slide (slides, jumping over blockers), strike (captures at a square WITHOUT moving), command (moves a friendly piece standing on one of the commander's command squares to another of its command squares; only the commander flips).
- Squares are written like c3 (columns a-f, rows 1-6). You cannot make a move that leaves your own Duke capturable ("in check").`;

const FORMAT_NO_COT = `Respond with ONLY the move, nothing else, in one of these exact formats:
MOVE <from> <to>
STRIKE <from> <target>
COMMAND <commander> <piece> <destination>
PLACE <TileName> <square>`;

const FORMAT_COT = `First reason briefly about the position (threats to your Duke, capture opportunities, what sides your tiles will flip to). Then output your final move on its own line, prefixed with exactly "FINAL: ", e.g.
FINAL: MOVE c2 c3`;

export class LlmAgent implements Agent {
  readonly name: string;
  private cfg: Required<Pick<LlmAgentConfig, 'temperature' | 'maxRetries'>> & LlmAgentConfig;
  private rng: () => number;

  constructor(cfg: LlmAgentConfig) {
    this.cfg = { temperature: 0.3, maxRetries: 2, ...cfg };
    this.name = cfg.name
      ?? `${cfg.model}[${cfg.representation}${cfg.legalMoves ? '+legal' : ''}${cfg.cot ? '+cot' : ''}]`;
    this.rng = makeRng(cfg.seed ?? 7);
  }

  private buildPrompt(state: GameState, legal: GameMove[]): string {
    const board = this.cfg.representation === 'ascii'
      ? `Board (row 6 at top):\n${renderAscii(state)}`
      : renderCoordList(state);

    const sections = [
      `You are ${state.currentPlayer}. It is your turn (turn ${state.turnNumber}).`,
      board,
      `Your tile abilities (current sides):\n${renderAbilities(state, state.currentPlayer)}`,
      renderBag(state),
    ];
    if (this.cfg.legalMoves) {
      sections.push(`Legal moves:\n${legal.map(m => `  ${moveToString(m)}`).join('\n')}`);
    }
    sections.push(this.cfg.cot ? FORMAT_COT : FORMAT_NO_COT);
    return sections.join('\n\n');
  }

  private extractMoveText(response: string): string {
    if (!this.cfg.cot) return response.trim();
    // Take the last FINAL: line; fall back to last non-empty line.
    const finals = [...response.matchAll(/FINAL:\s*(.+)/gi)];
    if (finals.length > 0) return finals[finals.length - 1][1].trim();
    const lines = response.split('\n').map(l => l.trim()).filter(Boolean);
    return lines[lines.length - 1] ?? '';
  }

  private async complete(messages: { role: string; content: string }[]): Promise<string> {
    const res = await fetch(`${this.cfg.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.cfg.apiKey ? { Authorization: `Bearer ${this.cfg.apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: this.cfg.model,
        messages,
        temperature: this.cfg.temperature,
        max_tokens: this.cfg.cot ? 1200 : 60,
      }),
    });
    if (!res.ok) throw new Error(`LLM endpoint ${res.status}: ${await res.text()}`);
    const data = await res.json() as { choices: { message: { content: string } }[] };
    return data.choices[0]?.message?.content ?? '';
  }

  async getMove(state: GameState, legal: GameMove[]): Promise<MoveResult> {
    const start = Date.now();
    let illegalAttempts = 0;
    let parseFailures = 0;
    let lastRaw = '';

    const messages: { role: string; content: string }[] = [
      { role: 'system', content: RULES_PREAMBLE },
      { role: 'user', content: this.buildPrompt(state, legal) },
    ];

    for (let attempt = 0; attempt <= this.cfg.maxRetries; attempt++) {
      lastRaw = await this.complete(messages);
      const candidate = parseMove(this.extractMoveText(lastRaw));

      if (!candidate) {
        parseFailures++;
        messages.push({ role: 'assistant', content: lastRaw });
        messages.push({ role: 'user', content: `Could not parse a move from your response. ${this.cfg.cot ? 'End with a line "FINAL: <move>".' : FORMAT_NO_COT}` });
        continue;
      }

      const match = legal.find(m => movesEqual(m, candidate));
      if (match) {
        return {
          move: match,
          meta: { illegalAttempts, parseFailures, latencyMs: Date.now() - start, rawResponse: lastRaw },
        };
      }

      illegalAttempts++;
      messages.push({ role: 'assistant', content: lastRaw });
      messages.push({
        role: 'user',
        content: `"${moveToString(candidate)}" is not a legal move in this position. Choose a legal move.${this.cfg.legalMoves ? '' : ' Re-check piece positions, current tile sides, and whether your Duke is left in check.'}`,
      });
    }

    // Exhausted retries — random legal fallback, recorded as such.
    const fallback = legal[Math.floor(this.rng() * legal.length)];
    return {
      move: fallback,
      meta: {
        illegalAttempts, parseFailures,
        fellBackToRandom: true,
        latencyMs: Date.now() - start,
        rawResponse: lastRaw,
      },
    };
  }
}
