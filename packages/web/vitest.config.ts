import {fileURLToPath} from 'node:url';
import {defineConfig} from 'vitest/config';

// Kept apart from vite.config.ts, whose PWA plugin has no business in tests.
// The engine comes from its source, so the tests don't need it built first.
export default defineConfig({
  resolve: {
    alias: {'@the-duke/engine': fileURLToPath(new URL('../engine/src/index.ts', import.meta.url))},
  },
  test: {include: ['src/**/*.test.ts']},
});
