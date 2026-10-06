"""
Mirror runs/<run>/metrics.jsonl to Weights & Biases.

Training only writes metrics.jsonl; this separate process uploads it, so a
network problem or a W&B outage can never stall or crash training, and old
runs can be backfilled the same way.

    uv run python -m training.zero.wandb_sync az2                 # upload what's there, exit
    uv run python -m training.zero.wandb_sync az3-net --follow    # keep uploading new lines
    training/zero/wandb_follow.sh az3-net az3-sims                # --follow, detached

Each run becomes the W&B run `<run>` in project `duke-ml`. Progress is kept
in runs/<run>/.wandb_offset (bytes of metrics.jsonl already uploaded), so
rerunning never uploads a line twice. Delete that file (and the W&B run) to
upload a run from scratch.

X axes: `iteration`, `games` (self-play games so far), `hours` (time spent
training, excluding gaps such as stops and reboots) and `gpu_hours` (hours ×
GPUs used, the fair axis for comparing runs on different hardware).
"""

import argparse
import json
import time
from pathlib import Path

import wandb

RUNS = Path(__file__).resolve().parents[2] / 'runs'
PROJECT = 'duke-ml'
GAP = 15 * 60          # a pause longer than this between records counts as downtime


class Clock:
    """Training hours and GPU-hours, skipping downtime between records."""

    def __init__(self, gpus):
        self.gpus, self.last, self.hours, self.gpu_hours = gpus, None, 0.0, 0.0

    def tick(self, t):
        if self.last is not None and 0 < t - self.last < GAP:
            dh = (t - self.last) / 3600
            self.hours += dh
            self.gpu_hours += dh * self.gpus
        self.last = t
        return {'hours': self.hours, 'gpu_hours': self.gpu_hours}


def convert(rec, state):
    """One metrics.jsonl record -> dict to log, or None."""
    kind = rec.get('kind')
    if kind == 'start':
        state['gpus'] = rec.get('gpus', state['gpus'])
        state['clock'].gpus = state['gpus']
        return {'events/start': 1, 'iteration': rec.get('iteration', 0)}
    out = state['clock'].tick(rec['time'])
    if kind == 'train':
        state['games'] = rec.get('games', state['games'])
        out.update({k: v for k, v in rec.items()
                    if isinstance(v, (int, float)) and k not in ('time', 'iteration', 'games')})
        out['iteration'], out['games'] = rec['iteration'], state['games']
        return out
    if kind == 'eval':
        out.update({f"eval/{r['b']}": r['score'] for r in rec['results']})
        out.update({f"eval_plies/{r['b']}": r['mean_plies'] for r in rec['results'] if 'mean_plies' in r})
        out['iteration'], out['games'] = rec['iteration'], state['games']
        return out
    if kind == 'ladder':
        out['ladder/score_vs_earlier'] = rec['results']['score']
        out['ladder/iterations_apart'] = rec['iteration'] - rec['opponent']
        out['iteration'], out['games'] = rec['iteration'], state['games']
        return out
    if kind == 'imitation':
        return {f'imitation/{k}': v for k, v in rec.items() if k not in ('time', 'kind')}
    return None


def run_config(run_dir: Path):
    for line in (run_dir / 'metrics.jsonl').open():
        rec = json.loads(line)
        if rec.get('kind') == 'start' and rec.get('config'):
            return rec['config'], rec.get('commit')
    cfg = run_dir / 'config.json'
    return (json.loads(cfg.read_text()) if cfg.exists() else {}), None


def sync(run: str, follow: bool, gpus: int, poll: float):
    run_dir = RUNS / run
    metrics = run_dir / 'metrics.jsonl'
    offset_file = run_dir / '.wandb_offset'
    offset = int(offset_file.read_text()) if offset_file.exists() else 0
    config, commit = run_config(run_dir)

    wb = wandb.init(project=PROJECT, name=run, id=run.replace('/', '-'), resume='allow',
                    config={**config, 'gpus': gpus, 'commit': commit}, dir=run_dir)
    wb.define_metric('iteration')
    wb.define_metric('imitation/step')
    for prefix in ('eval/*', 'eval_plies/*', 'ladder/*', 'fresh_*', 'sp_*', 'events/*',
                   'policy_*', 'value_*', 'grad_norm', 'moves_per_s', 'replay', 'wait_s',
                   'games', 'hours', 'gpu_hours'):
        wb.define_metric(prefix, step_metric='iteration')
    wb.define_metric('imitation/*', step_metric='imitation/step')

    # Replay everything before `offset` without uploading, to rebuild the
    # running clock and game count.
    state = {'clock': Clock(gpus), 'games': 0, 'gpus': gpus}
    with metrics.open('rb') as f:
        for line in f.read(offset).splitlines():
            if line.strip():
                convert(json.loads(line), state)

    try:
        while True:
            with metrics.open('rb') as f:
                f.seek(offset)
                chunk = f.read()
            # Only whole lines: training may be halfway through writing one.
            complete = chunk[:chunk.rfind(b'\n') + 1]
            n = 0
            for line in complete.splitlines():
                if line.strip():
                    row = convert(json.loads(line), state)
                    if row:
                        wb.log(row)
                        n += 1
            offset += len(complete)
            offset_file.write_text(str(offset))
            if n:
                print(f'[{time.strftime("%H:%M:%S")}] {run}: uploaded {n} records', flush=True)
            if not follow:
                break
            time.sleep(poll)
    finally:
        wb.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--follow', action='store_true', help='keep uploading new lines')
    p.add_argument('--gpus', type=int, default=None,
                   help='GPUs the run used (default: from its start records, else 2)')
    p.add_argument('--poll', type=float, default=60.0, help='seconds between checks with --follow')
    a = p.parse_args()
    sync(a.run, a.follow, a.gpus or 2, a.poll)


if __name__ == '__main__':
    main()
