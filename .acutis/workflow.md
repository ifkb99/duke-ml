---
model: opus
effort: high
setup:
  - npm ci --prefer-offline --no-audit --no-fund
  - npm run build -w packages/engine
gates:
  - name: build engine + ai
    run: npm run build -w packages/engine && npm run build -w packages/ai
  - npm test -w packages/engine
  - npm test -w packages/ai
  - name: harness smoke
    run: npm test -w packages/harness
permissions:
  allow: ["Bash(npm test:*)", "Bash(npm run:*)", "Bash(npx tsc:*)", "Bash(npx vitest:*)", "Bash(npx tsx:*)"]
max_attempts: 2
permission_timeout_s: 1800
land: trunk
---
Ticket #{{ticket.id}}: {{ticket.title}}

{{ticket.body}}

About this repo:
- npm workspaces in `packages/` (engine, ai, harness, web). `@the-duke/ai` imports `@the-duke/engine` from its `dist`, so rebuild the engine (`npm run build -w packages/engine`) after changing it.
- The TypeScript AI is the focus. The Python/JAX training in `training/` and the browser model files in `packages/web/public/models/` are out of scope. Don't edit them, `deploy/`, or `.github/`.
- There is no network in this environment. Don't install packages.
- Keep changes focused on the ticket. Explain in your summary what you measured.
