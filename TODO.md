# TODO

## Now: installable and offline (PWA)
- built on branch `web/pwa`; what's left needs real phones, on the deployed site (the LAN test server's self-signed certificate rules out a service worker):
  - install it on Android and on iOS Safari; check standalone mode (safe areas, the no-scroll phone layout)
  - play vs Neural Net in airplane mode
  - deploy a second build and take the "A new version is ready" update mid-game

## Most important
- rules page for new players, then a short interactive tutorial
  - first: move/command space only allows command atm
- multiplayer over websocket: a small room server and a link to share

## Show what the net is thinking
- hint button
- top moves with their share of the search
- move list and replay (the notation in `packages/harness` already exists)
- review after the game: eval graph, biggest mistakes marked

## AI in the web app
- difficulty levels (simulations and temperature)
- draw moves take two full budgets (~6 s on the phone): reuse the subtree or split the budget
- cache the CPU/WebGPU calibration per device
- TimeBudget undershoots (2.0-2.9 s of 3 s)
- the rest is under "Next steps" in `training/zero/EXPERIMENTS.md`

## Telling the story
- rewrite the README: it still says alpha-beta is next and that the AI code is handmade
- a "how it was made" page: MuZero retired, big1 teacher, distillation, 4-bit network, CPU/WebGPU routing
  - charts from `runs/`: strength by generation, download size against strength

## Safety net and housekeeping
- address console errors on web
- `packages/web` has no tests; one Playwright test that plays a whole game against the net
- run the Python parity tests in CI (only the npm tests run on push)
- nginx.conf and vite.config.ts still set COOP/COEP for threads the app no longer uses
- remove leftovers: `main.py`, `cursor_reinforcement_learning_for_the_d.md`, the root `docker-compose.yml`, `training/supervised_v1_baseline`

## ML
- next experiments are in `training/zero/EXPERIMENTS.md` ("Next steps")

## Probably done (delete if so)
- enhance UI for desktop and mobile
- ml time! muZero and then onnx
- ml: look into semantic ifs for some moves, i.e. split a decision into meaningful steps the search can weigh separately: is drawing worth it at all, then which piece to move, then where (and maybe more, e.g. attack vs reposition)
