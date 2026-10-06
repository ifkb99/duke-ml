"""
Evaluator: rates every checkpoint of every run on one common scale.

Runs next to training (its own process, never touches the training runs) and
watches runs/<run>/ for new checkpoints (imitation.pkl and ckpt_*.pkl). For
each one, with the SAME search for everyone (32 simulations; a run's own
training evals use its own setting, which makes e.g. a 64-simulation run look
stronger than its network is):

  1. benchmark: move quality on the fixed position set (benchmark.py)
  2. calibration: 256 self-play games; does the value head's confidence match
     how often that side actually wins, overall and by game phase?
  3. rating games: 128 games vs the run's previous checkpoint and 128 vs the
     closest-rated checkpoint of another run. Together with the training
     evals against minimax (same parameters at checkpoint iterations), all
     games are fit to one Elo scale with minimax depth 2 fixed at 0.

Results: runs/ratings/{games.jsonl,results.jsonl,ratings.json} and the W&B run
`evaluator` (project duke-ml), with per-run metrics on `<run>/...` panels and
an Elo table/chart refreshed after every checkpoint.

    uv run python -m training.zero.evaluator az2 az3-net az3-sims [--once]
    training/zero/evaluate.sh az2 az3-net az3-sims        # detached, restarts on crash
"""

import argparse
import json
import math
import time
import zlib
from pathlib import Path

import jax
import numpy as np

from training.zero import arena as AR
from training.zero import benchmark as BM
from training.zero import env as E
from training.zero import replay as RP
from training.zero import search as S
from training.zero import selfplay as SP
from training.zero.network import make_forward

RUNS = Path(__file__).resolve().parents[2] / 'runs'
OUT = RUNS / 'ratings'
SIMS = 32
ANCHOR = 'minimax2'
MATCH_GAMES = 128
CALIBRATION_GAMES = 256


def player_name(run, iteration):
    return f'{run}:{iteration}'


# ---------------------------------------------------------------------------
# Elo fit
# ---------------------------------------------------------------------------

def fit_elo(games, anchor=ANCHOR, prior_sd=800.0, iters=2000):
    """Maximum-likelihood Elo from match results (draws count half).

    games: list of {'a', 'b', 'wins', 'draws', 'losses'} (from a's view).
    A weak Gaussian prior keeps players with only wins (e.g. vs random)
    finite. Returns {player: (elo, standard error)}.
    """
    players = sorted({g['a'] for g in games} | {g['b'] for g in games})
    idx = {p: i for i, p in enumerate(players)}
    a = np.array([idx[g['a']] for g in games]); b = np.array([idx[g['b']] for g in games])
    n = np.array([g['wins'] + g['draws'] + g['losses'] for g in games], float)
    s = np.array([g['wins'] + 0.5 * g['draws'] for g in games], float)
    k = math.log(10) / 400
    r = np.zeros(len(players))
    for _ in range(iters):                     # Newton-free gradient ascent; small problem
        p = 1 / (1 + np.exp(-k * (r[a] - r[b])))
        g = np.zeros_like(r)
        np.add.at(g, a, k * (s - n * p)); np.add.at(g, b, -k * (s - n * p))
        g -= r / prior_sd ** 2
        h = np.zeros_like(r)
        np.add.at(h, a, k * k * n * p * (1 - p)); np.add.at(h, b, k * k * n * p * (1 - p))
        h += 1 / prior_sd ** 2
        r += g / h
        if anchor in idx:
            r -= r[idx[anchor]]
    # Standard errors from the Fisher information (anchor fixed).
    p = 1 / (1 + np.exp(-k * (r[a] - r[b])))
    H = np.zeros((len(r), len(r)))
    w = k * k * n * p * (1 - p)
    np.add.at(H, (a, a), w); np.add.at(H, (b, b), w); np.add.at(H, (a, b), -w); np.add.at(H, (b, a), -w)
    H += np.eye(len(r)) / prior_sd ** 2
    free = [i for i, q in enumerate(players) if q != anchor]
    cov = np.linalg.inv(H[np.ix_(free, free)])
    se = np.zeros(len(r)); se[free] = np.sqrt(np.diag(cov))
    return {q: (float(r[i]), float(se[i])) for q, i in idx.items()}


# ---------------------------------------------------------------------------
# Per-checkpoint measurements
# ---------------------------------------------------------------------------

def calibration(forward, params, key, sp, games=CALIBRATION_GAMES):
    """Self-play games with this network (`sp` = its make_selfplay); value head vs actual results."""
    states = jax.vmap(E.init)(jax.random.split(key, games))
    asm, parts = RP.Assembler(games), []
    for i in range(14):                        # 448 moves per slot
        states, rec = sp(params, states, jax.random.fold_in(key, i))
        rec = jax.device_get(rec)
        rec.pop('search_value')
        out = asm.add(rec)
        if out is not None:
            parts.append(out)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    obs = jax.jit(jax.vmap(E.observe_arrays))
    vs = []
    for i in range(0, len(d['z']), 2048):
        a = [jax.numpy.asarray(d[k][i:i + 2048], jax.numpy.int32) for k in ('tile', 'owner', 'side', 'bags', 'pending')]
        vs.append(np.asarray(forward(params, *obs(*a, jax.numpy.asarray(d['ply'][i:i + 2048])))[1], np.float32))
    v, z, ply = np.concatenate(vs), d['z_mc'], d['ply']
    out = {'value_corr': float(np.corrcoef(v, z)[0, 1]),
           'value_mse': float(np.mean((v - z) ** 2)), 'predict0_mse': float(np.mean(z ** 2))}
    for name, lo, hi in (('opening', 0, 20), ('middle', 20, 60), ('late', 60, 120), ('endgame', 120, E.MAX_PLY)):
        m = (ply >= lo) & (ply < hi)
        if m.sum() > 200 and np.std(z[m]) > 0:
            out[f'value_corr_{name}'] = float(np.corrcoef(v[m], z[m])[0, 1])
    # Calibration: in each bin of predicted value, the average actual result.
    bins = np.linspace(-1, 1, 11)
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum() >= 50:
            table.append([float((lo + hi) / 2), float(v[m].mean()), float(z[m].mean()), int(m.sum())])
    # Expected calibration error: average |predicted - actual| over bins, weighted.
    if table:
        t = np.array(table)
        out['calibration_error'] = float(np.sum(np.abs(t[:, 1] - t[:, 2]) * t[:, 3]) / t[:, 3].sum())
    return out, table


def play_match(a, b, games, key):
    res, _ = AR.play(a, b, games, key)
    return {'a': a.name, 'b': b.name, 'wins': res['wins'], 'draws': res['draws'],
            'losses': res['losses'], 'mean_plies': res['mean_plies']}


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def run_timeline(run):
    """iteration -> (cumulative games, gpu_hours) from metrics.jsonl, like wandb_sync."""
    from training.zero.wandb_sync import Clock, convert
    st = {'clock': Clock(2), 'games': 0, 'games_base': 0, 'raw_games': 0, 'gpus': 2}
    gpus = 1 if run.startswith('az3') else 2   # az3 runs predate the recorded GPU count
    st['clock'].gpus = st['gpus'] = gpus
    timeline = {0: (0, 0.0)}
    for line in (RUNS / run / 'metrics.jsonl').open():
        rec = json.loads(line)
        convert(rec, st)
        if 'iteration' in rec and rec.get('kind') in ('train', 'eval'):
            timeline.setdefault(rec['iteration'], (st['games'], st['clock'].gpu_hours))
    return timeline


def run_config(run):
    p = RUNS / run / 'config.json'
    return json.loads(p.read_text()) if p.exists() else {}


def minimax_games(run, iteration):
    """Training evals at a checkpoint iteration (same parameters) as rating games.

    Only for runs whose training evals used the evaluator's search budget: a
    64-simulation run's evals measure 64-simulation play, not the 32-simulation
    player rated here."""
    if run_config(run).get('simulations', SIMS) != SIMS:
        return []
    out = []
    for line in (RUNS / run / 'metrics.jsonl').open():
        rec = json.loads(line)
        if rec.get('kind') == 'eval' and rec.get('iteration') == iteration:
            for r in rec['results']:
                if r['b'] in ('random',) or r['b'].startswith('minimax'):
                    out.append({'a': player_name(run, iteration), 'b': r['b'], 'wins': r['wins'],
                                'draws': r['draws'], 'losses': r['losses'], 'note': 'training eval'})
    return out


STRIDE = {}     # run -> only checkpoints whose iteration is a multiple of this


def checkpoints(run):
    d = RUNS / run
    found = []
    if (d / 'imitation.pkl').exists():
        found.append((0, d / 'imitation.pkl'))
    for p in sorted(d.glob('ckpt_*.pkl')):
        it = int(p.stem.split('_')[1])
        if it % STRIDE.get(run, 1) == 0:
            found.append((it, p))
    return found


def load_jsonl(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def append_jsonl(path, rec):
    with path.open('a') as f:
        f.write(json.dumps(rec) + '\n')


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, runs, wb=None):
        self.runs = runs
        OUT.mkdir(parents=True, exist_ok=True)
        self.games_path, self.results_path = OUT / 'games.jsonl', OUT / 'results.jsonl'
        self.done = {(r['run'], r['iteration']) for r in load_jsonl(self.results_path)}
        self.bench = BM.load_benchmark()
        self.wb = wb
        self.agents = {}
        self.archs = {}
        self.jobs = 0
        if wb is not None:
            for run in runs:
                wb.define_metric(f'{run}/iteration')
                wb.define_metric(f'{run}/*', step_metric=f'{run}/gpu_hours')

    def arch(self, graphdef, config):
        """Compiled functions shared by every checkpoint of one network shape."""
        key = (config['channels'], config['blocks'])
        if key not in self.archs:
            fwd = make_forward(graphdef)
            self.archs[key] = {'forward': fwd, 'search': jax.jit(S.make_search(fwd, SIMS, 16, 16)),
                               'selfplay': SP.make_selfplay(S.make_search(fwd, SIMS, 16, 16), unroll=32)}
        return self.archs[key]

    def agent(self, run, iteration, path):
        name = player_name(run, iteration)
        if name not in self.agents:
            from training.zero.train import load_params
            graphdef, params, ckpt = load_params(path)
            arch = self.arch(graphdef, ckpt['config'])
            # Parameters stay on the host; each job moves them to its GPU.
            self.agents[name] = (AR.Agent('net', name=name, params=jax.device_get(params),
                                          search=arch['search']), arch)
        return self.agents[name]

    def pending(self):
        todo = []
        for run in self.runs:
            for it, path in checkpoints(run):
                if (run, it) not in self.done:
                    todo.append((run, it, path))
        # Oldest first within each run; interleave runs.
        todo.sort(key=lambda x: (x[1], x[0]))
        return todo

    def ratings(self):
        games = load_jsonl(self.games_path)
        return fit_elo(games) if games else {}

    def evaluate(self, run, it, path):
        t0 = time.time()
        dev = jax.devices()[self.jobs % len(jax.devices())]   # alternate GPUs: fair to both training runs
        self.jobs += 1
        with jax.default_device(dev):
            agent, arch = self.agent(run, it, path)
            agent.params = jax.device_put(agent.params, dev)
            me = agent.name
            key = jax.random.key(zlib.crc32(me.encode()))

            bench = BM.score_network(arch['forward'], agent.params, self.bench, SIMS, search=arch['search'])
            calib, table = calibration(arch['forward'], agent.params, jax.random.fold_in(key, 1), arch['selfplay'])

            new_games = [g for g in minimax_games(run, it)]
            prev = [(i, p) for i, p in checkpoints(run) if i < it]
            if prev:
                pi, pp = prev[-1]
                opp, _ = self.agent(run, pi, pp)
                opp.params = jax.device_put(opp.params, dev)
                new_games.append(play_match(agent, opp, MATCH_GAMES, jax.random.fold_in(key, 2)))
            # Closest-rated checkpoint of another run, judged by a provisional fit.
            provisional = fit_elo(load_jsonl(self.games_path) + new_games) if new_games else {}
            others = [(r, i, p) for r in self.runs if r != run for i, p in checkpoints(r)
                      if (r, i) in self.done and player_name(r, i) in provisional]
            if others:
                if me in provisional:
                    gap = lambda x: abs(provisional[player_name(x[0], x[1])][0] - provisional[me][0])
                else:   # no games yet (e.g. its minimax evals used other settings): nearest iteration
                    gap = lambda x: abs(x[1] - it)
                r, i, p = min(others, key=gap)
                opp, _ = self.agent(r, i, p)
                opp.params = jax.device_put(opp.params, dev)
                new_games.append(play_match(agent, opp, MATCH_GAMES, jax.random.fold_in(key, 3)))
        for g in new_games:
            append_jsonl(self.games_path, {**g, 'time': time.time()})

        elo = self.ratings()
        games, gpu_hours = run_timeline(run).get(it, (None, None))
        result = {'run': run, 'iteration': it, 'player': me, 'games': games, 'gpu_hours': gpu_hours,
                  'elo': elo.get(me, (None, None))[0], 'elo_se': elo.get(me, (None, None))[1],
                  **{f'bench_{k}': v for k, v in bench.items()}, **calib,
                  'calibration_table': table, 'seconds': time.time() - t0, 'time': time.time()}
        append_jsonl(self.results_path, result)
        self.done.add((run, it))
        self.write_ratings(elo)
        print(f"[{time.strftime('%H:%M:%S')}] {me}: elo {result['elo']:.0f}±{result['elo_se']:.0f}, "
              f"search acc {bench['search_acc']:.3f}, blunders {bench['search_blunder']:.3f}, "
              f"value corr {calib['value_corr']:.3f} ({result['seconds']:.0f}s)", flush=True)
        self.log_wandb(result, elo)

    def write_ratings(self, elo):
        rows = []
        results = {(r['run'], r['iteration']): r for r in load_jsonl(self.results_path)}
        for name, (e, se) in sorted(elo.items(), key=lambda x: -x[1][0]):
            run, _, it = name.rpartition(':')
            r = results.get((run, int(it))) if run else None
            rows.append({'player': name, 'elo': e, 'se': se, 'run': run or name,
                         'iteration': int(it) if run else None,
                         'gpu_hours': r['gpu_hours'] if r else None, 'games': r['games'] if r else None})
        (OUT / 'ratings.json').write_text(json.dumps(rows, indent=1))

    def log_wandb(self, result, elo):
        if self.wb is None:
            return
        import wandb
        run = result['run']
        row = {f'{run}/{k}': v for k, v in result.items()
               if isinstance(v, (int, float)) and v is not None and k not in ('time', 'seconds')}
        self.wb.log(row)
        # Ratings move as more games come in, so the full table is re-logged.
        rows = json.loads((OUT / 'ratings.json').read_text())
        table = wandb.Table(columns=['player', 'run', 'iteration', 'gpu_hours', 'games', 'elo', 'se'],
                            data=[[r['player'], r['run'], r['iteration'], r['gpu_hours'], r['games'],
                                   r['elo'], r['se']] for r in rows])
        series = {}
        for r in rows:
            if r['iteration'] is not None and r['gpu_hours'] is not None:
                series.setdefault(r['run'], []).append((r['gpu_hours'], r['elo']))
        keys = sorted(series)
        xs = [[x for x, _ in sorted(series[k])] for k in keys]
        ys = [[y for _, y in sorted(series[k])] for k in keys]
        log = {'ratings/table': table}
        if keys:
            log['ratings/elo_vs_gpu_hours'] = wandb.plot.line_series(
                xs=xs, ys=ys, keys=keys, title='Elo vs GPU-hours (minimax depth 2 = 0)', xname='GPU-hours')
        for name in ('random', 'minimax4', 'minimax6'):
            if name in elo:
                log[f'ratings/{name}'] = elo[name][0]
        self.wb.log(log)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--once', action='store_true', help='evaluate what exists, then exit')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--poll', type=float, default=120)
    p.add_argument('--stride', nargs='*', default=[], metavar='RUN=N',
                   help='only rate checkpoints of RUN at multiples of N iterations (e.g. az1=1000)')
    a = p.parse_args()
    for item in a.stride:
        run, n = item.split('=')
        STRIDE[run] = int(n)
    jax.config.update('jax_compilation_cache_dir', str(RUNS / '.jax_cache'))
    wb = None
    if not a.no_wandb:
        import wandb
        wb = wandb.init(project='duke-ml', name='evaluator', id='evaluator', resume='allow', dir=OUT,
                        config={'simulations': SIMS, 'match_games': MATCH_GAMES, 'anchor': ANCHOR})
    ev = Evaluator(a.runs, wb)
    try:
        while True:
            todo = ev.pending()
            for run, it, path in todo:
                ev.evaluate(run, it, path)
            if a.once:
                break
            if not todo:
                time.sleep(a.poll)
    finally:
        if wb is not None:
            wb.finish()


if __name__ == '__main__':
    main()
