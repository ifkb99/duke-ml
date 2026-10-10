# Diagnostics

One-off experiments used to debug training (see ../EXPERIMENTS.md for results).
Run from the repo root, e.g.
`CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python training/zero/diagnostics/heldout.py ckpt_002200`.
They are quick scripts, not polished tools; most read checkpoints from `runs/az1/`.

| script | question |
|---|---|
| `sims_eval.py` | does more search (simulations) help against minimax? |
| `ladder.py` | do later checkpoints beat earlier ones? |
| `search_vs_policy.py` | how far are search targets from the policy; does search beat the raw policy? |
| `bellman.py` | is v(s) consistent with the values one move later? |
| `heldout.py` | value loss on fresh self-play games never trained on |
| `ztruth.py` | do value targets match the true winner? |
| `gen_games.py` | generate a fresh self-play dataset from a checkpoint |
| `learnability.py` | can a fresh net learn the value from a dataset (game-level split)? `ABLATE=features|planes` |
