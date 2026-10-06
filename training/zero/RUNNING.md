# Running training: start, stop, resume

All commands run from the repository root. `az2` is the run name used below;
use any name you like, since each run lives in its own `runs/<name>/` folder.

## Start

```bash
training/zero/run.sh az2
```

That's it. The script:

- runs training **in the background, detached**, so closing the terminal (or
  Claude Code) doesn't stop it;
- **restarts it automatically** 30 seconds after a crash, resuming from the last
  checkpoint (you'll see `[run.sh] train.py exited with ... restarting` in the log);
- keeps it **off CPU core 20 (CPUs 10 and 11)**, which is faulty on this
  machine (see "Troubleshooting");
- writes everything to `runs/az2/`.

A brand-new run starts with ~11 minutes of minimax-vs-minimax games and a few
minutes of pretraining on them (the "imitation warm-start"), then self-play.

## Watch

```bash
tail -f runs/az2/train.log                 # live log (Ctrl-C stops watching, not training)
grep -E "eval|after imitation" runs/az2/train.log   # just the evaluation results
```

An evaluation line looks like:

```
[22:31:07] eval 50: random 1.00, minimax2 0.81, minimax4 0.47
```

Each number is the network's **score** against that opponent over 64 games:
1 point per win, ½ per draw, divided by the number of games. **0.5 means evenly
matched**, and above 0.5 means it's winning more than losing.

Every 100 iterations there's also a **ladder** line: the network against
itself from 100 iterations earlier. Above 0.5 means it improved. This keeps
working after minimax becomes too easy.

```
[11:02:13] ladder 300 vs 200: 0.61
```

Other useful numbers in the `iter` lines:

| field | meaning |
|---|---|
| `moves/s` | self-play speed (searched moves per second, both GPUs) |
| `policy_loss` | how far the network's instinct is from what the search decided (lower = closer) |
| `value_loss` | how well it predicts its value targets, on samples it trains on |
| `sp_game_plies` | average self-play game length |
| `fresh_value_corr_mc` | **the key health number**: correlation between the value head and who actually won, on new games it hasn't trained on yet. Should climb well above 0 (az1 sat at ~0.05 all night, which meant it was learning nothing) |
| `fresh_value_mse_mc` vs `fresh_mc_baseline` | value error on new games vs always guessing 0; the first should be clearly lower |
| `fresh_policy_kl` | how far search targets are from the network's policy; it should shrink as the policy learns |
| `fresh_prior_entropy`, `fresh_target_entropy` | how spread out the policy and the search targets are (uniform over 9 moves ≈ 2.2) |
| `fresh_top1_agree` | how often the policy's favourite move is the search's favourite |
| `fresh_target_corr_mc` | how well the n-step value targets agree with the real game result |

Everything is also in `runs/az2/metrics.jsonl`, one JSON object per line, for
plotting.

Is it running?

```bash
ps -o pid,etime,cmd -p $(cat runs/az2/pid)   # prints a line with elapsed time if alive
nvidia-smi                                   # both GPUs should show a python process
```

## Stop

```bash
kill -- -$(cat runs/az2/pid)
```

The `-` in front of the number matters. It stops the whole **process group**:
the restart loop, the trainer and all 30 minimax helper processes. (Without it
you'd only kill the restart loop, and the trainer would keep running.)

**What you lose by stopping:** at most ~7 minutes. The resume point
(`latest.pkl`) is saved at every evaluation, i.e. every 25 iterations. The replay
buffer (recent self-play games) isn't saved; it refills in a few minutes after
resuming.

If you stop during the warm-start, the minimax games are kept
(`imitation_data.npz`) and won't be replayed.

## Resume

Run the **same start command with the same name**:

```bash
training/zero/run.sh az2
```

It finds `runs/az2/latest.pkl` and continues from that iteration. You can stop
and resume as often as you like, e.g. stop to free the GPUs for a game and
resume afterwards, or across reboots.

Settings when resuming: the **network size always comes from the checkpoint**.
Other flags (learning rate, simulations, number of games...) are taken from
the command line, so pass the same ones you started with unless you mean to change them.

## Start over

Either pick a new name (`training/zero/run.sh az2`), or delete the old run first:

```bash
kill -- -$(cat runs/az2/pid)      # if it's still running
rm -rf runs/az2
training/zero/run.sh az2
```

## Changing settings

Any `train.py` option can be added after the run name:

```bash
training/zero/run.sh az2 --simulations 64 --channels 128 --eval-depths 2 4 6
```

The ones you're most likely to want:

| flag | default | what it does |
|---|---|---|
| `--simulations` | 32 | search effort per move (stronger, but slower self-play) |
| `--channels`, `--blocks` | 96, 6 | network size (new runs only) |
| `--games`, `--learner-games` | 872, 792 | parallel self-play games on GPU 1 and GPU 0 |
| `--sample-reuse` | 1 | how many times each new position is trained on on average |
| `--value-nstep` | 10 | value target looks this many steps ahead then uses the search's estimate (0 = final result) |
| `--eval-every` | 25 | iterations between evaluations and checkpoints |
| `--eval-depths` | 2 4 | minimax depths to evaluate against every `--eval-every` iterations |
| `--slow-eval-depths`, `--slow-eval-every` | 6, 100 | deeper (slower) minimax, evaluated less often; shows up as a second `eval` line |
| `--imitation-games` | 3000 | warm-start games (0 = pure self-play from scratch); training on them stops early when held-out loss stops improving |
| `--ladder-every` | 100 | iterations between ladder matches against the earlier network (0 = off) |
| `--lr` | 3e-4 | learning rate |

See all of them with `uv run python -m training.zero.train --help`.

Environment variables for `run.sh`:

| variable | default | purpose |
|---|---|---|
| `DUKE_CPUS` | `0-9,12-31` | CPUs training may use (excludes the faulty core) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.6` | share of each GPU's memory JAX may take |

## Files in `runs/az2/`

| file | what |
|---|---|
| `train.log` | everything printed, including crash traces |
| `metrics.jsonl` | machine-readable record of everything, one JSON object per line, by `kind`: `start` (config, git commit, resumes), `imitation` (warm-start losses), `train` (every 5 iterations), `eval`, `ladder` |
| `diagnostics/` | results of one-off experiments (see `training/zero/EXPERIMENTS.md`) |
| `latest.pkl` | resume point (weights + optimizer state + iteration) |
| `ckpt_000200.pkl`, ... | permanent snapshots every 200 iterations |
| `imitation.pkl` | the network right after the warm-start |
| `imitation_data.npz` | the warm-start minimax games |
| `config.json` | settings the run was started with |
| `pid` | process-group id used by the stop command |

`runs/.jax_cache/` holds compiled GPU programs shared by all runs, so restarts
skip most compilation. It's safe to delete; it just rebuilds.

## Troubleshooting

**`[run.sh] train.py exited with 139`** (segfault) in the log. Check which CPU it
happened on:

```bash
journalctl -k | grep -E "segfault|traps:" | tail
```

If it says `likely on CPU 10` or `CPU 11`, that's the faulty core again
(something wasn't pinned). Anything else is a new problem worth looking at:
the lines just above in `train.log` have a Python stack trace.

**`[minimax_pool] ... restarting worker`**: a minimax helper process died and
was replaced automatically. Harmless unless it happens constantly.

**Out of GPU memory** when starting something else alongside training: lower
`XLA_PYTHON_CLIENT_MEM_FRACTION` when you start the run, or run the other job
with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

**Before rebooting:** stop the run. After rebooting, start it again with the same name.
