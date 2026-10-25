"""Boundary extension, randomized repeated timings, and training-only audit."""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from acv import simulate, make_fold, CompressedFold
from boundary_acv import adaptive, fit_box, active_step


METHODS = ['lbfgs', 'newton', 'frozen', 'one_step', 'adaptive', 'interior_only']


def run_method(name, folds, starts):
    if name in ('adaptive', 'interior_only'):
        return adaptive(folds, starts, allow_boundary=name == 'adaptive')
    scores, parameters, metadata = [], [], []
    for m, fs in enumerate(folds):
        for f in fs:
            if name in ('lbfgs', 'newton'):
                theta, ev, meta = fit_box(f, starts[m], method=name)
            else:
                theta = starts[m].copy(); ev = f.evaluate(theta)
                if name == 'one_step':
                    theta = active_step(f, theta, ev); ev = f.evaluate(theta)
                meta = {}
            mu = f.prediction(theta, ev)[0]
            scores.append(float(np.mean((f.test[1]-mu)**2)))
            parameters.append(theta); metadata.append(meta)
    return np.array(scores).reshape(len(folds), -1), np.array(parameters), metadata


def benchmark(folds, starts, setup_seconds, seed, timing_reps=5, audit=True):
    # Untimed warm-up also provides the canonical deterministic result.
    canonical = {name: run_method(name, folds, starts) for name in METHODS}
    rng = np.random.default_rng(seed+700000)
    times = []
    for repeat in range(timing_reps):
        for position, name in enumerate(rng.permutation(METHODS)):
            before = sum(f.evaluations for fs in folds for f in fs)
            t0 = perf_counter()
            result = run_method(name, folds, starts)
            elapsed = perf_counter()-t0
            times.append({'repeat': repeat, 'position': position, 'method': name,
                          'seconds': elapsed, 'evaluations':
                          sum(f.evaluations for fs in folds for f in fs)-before})
            if name in ('adaptive', 'interior_only'):
                assert result[0] == canonical[name][0]
    time_frame = pd.DataFrame(times)
    med = time_frame.groupby('method').seconds.median()
    ref_loss, ref_theta, ref_meta = canonical['newton']
    cv = ref_loss.mean(axis=1); target = int(np.argmin(cv))
    winner, states, lower, upper, rounds = canonical['adaptive']
    iwinner, istates, _, _, _ = canonical['interior_only']
    rows, audit_losses = [], []
    for index, s in enumerate(states):
        c = s.cert; reference = ref_loss[s.model, s.group]
        root_distance = np.nan; inside = True; active_match = True
        if c['valid']:
            free = c['free']; active = c['active']
            root_distance = float(np.linalg.norm(np.linalg.solve(c['S'][free],
                                      (ref_theta[index]-s.theta)[free])))
            active_match = bool(np.all(ref_theta[index, active] == 0))
            inside = active_match and root_distance <= c['outer_radius']+1e-7
        row = {'model': s.model, 'group_index': s.group, 'steps': s.steps,
               'local_valid': c['valid'], 'certificate_kind': c['reason'],
               'numerical_fallback': s.numerical, 'fallback_reason': s.fallback_reason,
               'lower': c['lo'], 'upper': c['hi'], 'reference_loss': reference,
               'containment_gap': max(c['lo']-reference, reference-c['hi'], 0.),
               'contains_reference': c['lo']-1e-8 <= reference <= c['hi']+1e-8,
               'reference_in_local_ball': inside, 'active_set_match': active_match,
               'root_distance': root_distance, 'root_radius': c.get('outer_radius', np.nan),
               'active_count': len(c.get('active', [])),
               'minimum_active_margin': min(c.get('active_lower', []), default=np.nan),
               'reference_gradient': ref_meta[index]['projected_gradient']}
        for j in range(3):
            row[f'theta_{j}'] = float(s.theta[j])
            row[f'reference_theta_{j}'] = float(ref_theta[index, j])
        if audit:
            scale = max(float(np.var(s.fold.y)), 1e-4)
            fits = [fit_box(s.fold, scale*np.array(v), method='lbfgs')
                    for v in ([.5, .25, .25], [.8, .1, .1], [.1, .45, .45])]
            theta, ev, meta = min(fits, key=lambda x: x[1].value)
            loss = float(np.mean((s.fold.test[1]-s.fold.prediction(theta, ev)[0])**2))
            audit_losses.append(loss)
            row.update(training_audit_loss=loss,
                training_audit_contains=c['lo']-1e-8 <= loss <= c['hi']+1e-8,
                training_audit_objective_spread=max(f[1].value for f in fits)-min(f[1].value for f in fits),
                training_audit_gradient=meta['projected_gradient'])
        rows.append(row)
    reference_time = min(med['lbfgs'], med['newton'])
    summary = {'reference_winner': target, 'adaptive_winner': winner,
               'adaptive_agrees': winner == target, 'interior_only_agrees': iwinner == target,
               'fold_count': len(states), 'local_certificates': sum(s.cert['valid'] for s in states),
               'active_face_certificates': sum(s.cert['reason'] == 'active_face' for s in states),
               'double_active_certificates': sum(len(s.cert.get('active', [])) == 2 for s in states),
               'numerical_fallbacks': sum(s.numerical for s in states),
               'interior_only_fallbacks': sum(s.numerical for s in istates),
               'refinement_rounds': rounds, 'common_seconds': setup_seconds,
               'speedup_total': (setup_seconds+reference_time)/(setup_seconds+med['adaptive']),
               'speedup_kernel': reference_time/med['adaptive'],
               'interior_only_speedup_total': (setup_seconds+reference_time)/(setup_seconds+med['interior_only']),
               'reference_difference': float(np.max(np.abs(canonical['lbfgs'][0]-ref_loss))),
               'containment_failures': sum(not r['contains_reference'] for r in rows),
               'root_mismatches': sum(not r['reference_in_local_ball'] for r in rows),
               'max_reference_gradient': max(r['reference_gradient'] for r in rows),
               'timing_repeats': timing_reps}
    for name in METHODS:
        summary[name+'_seconds'] = med[name]
    for name in ('frozen', 'one_step'):
        v = canonical[name][0].mean(axis=1)
        summary[name+'_agrees'] = int(np.argmin(v)) == target
        summary[name+'_max_cv_error'] = float(np.max(np.abs(v-cv)))
        for m in range(len(folds)):
            summary[f'{name}_cv_{m}'] = v[m]
    if audit:
        acv = np.array(audit_losses).reshape(len(folds), -1).mean(axis=1)
        summary['training_audit_agrees'] = int(np.argmin(acv)) == target
        summary['training_audit_failures'] = sum(not r['training_audit_contains'] for r in rows)
    for m in range(len(folds)):
        summary[f'cv_{m}'] = cv[m]
        summary[f'lower_{m}'] = lower[m]
        summary[f'upper_{m}'] = upper[m]
    return summary, rows, times


def simulation(regime, seed, na, nb):
    if regime not in ('zero_a', 'zero_both'):
        return simulate(seed, regime, na=na, nb=nb)
    X, _, a, b = simulate(seed, 'balanced', na=na, nb=nb)
    rng = np.random.default_rng(seed+900000)
    y = 1+1.2*X[:, 1]+.75*X[:, 2]
    if regime == 'zero_a':
        y += rng.normal(scale=np.sqrt(.55), size=nb)[b]
    y += rng.normal(size=len(y))
    return X, y, a, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--timing-reps', type=int, default=5)
    p.add_argument('--na', type=int, default=12)
    p.add_argument('--nb', type=int, default=10)
    p.add_argument('--regimes', nargs='+', default=['balanced', 'sparse', 'unbalanced', 'near_tie', 'boundary', 'zero_a', 'zero_both'])
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[1]/'results'/'boundary_v2'))
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, audits, timings = [], [], []
    for ri, regime in enumerate(args.regimes):
        for repetition in range(args.reps):
            seed = 20261025+1000*ri+repetition
            X, y, a, b = simulation(regime, seed, args.na, args.nb)
            columns = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
            t0 = perf_counter()
            starts = [fit_box(CompressedFold(X[:, cols], y, a, b), [1., .7, .5])[0] for cols in columns]
            folds = [[make_fold(X, y, a, b, cols, k, 'compressed') for k in np.unique(a)] for cols in columns]
            setup = perf_counter()-t0
            row, ar, tr = benchmark(folds, starts, setup, seed, args.timing_reps)
            row.update(regime=regime, seed=seed, n=len(y), groups=len(np.unique(a)))
            rows.append(row)
            audits.extend(dict(regime=regime, seed=seed, **x) for x in ar)
            timings.extend(dict(regime=regime, seed=seed, **x) for x in tr)
            pd.DataFrame(rows).to_csv(out/'runs.csv', index=False)
            pd.DataFrame(audits).to_csv(out/'fold_audit.csv', index=False)
            pd.DataFrame(timings).to_csv(out/'timings.csv', index=False)
            print(f'{regime} {repetition+1}/{args.reps}: local={row["local_certificates"]} '
                  f'face={row["active_face_certificates"]} fallback={row["numerical_fallbacks"]} '
                  f'agree={row["adaptive_agrees"]} speedup={row["speedup_total"]:.2f} '
                  f'fail={row["containment_failures"]}/{row["training_audit_failures"]}', flush=True)
    env = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
           'pandas': pd.__version__, 'platform': platform.platform(), 'parameters': vars(args),
           'scope': 'Float64 local KKT and root checks; true zero random variances; no global proof.',
           'timing': 'One warm-up plus randomized-order repeats; medians; common setup included once in each total.',
           'audit': 'Three training-only initializations per fold, excluded from timing.'}
    (out/'environment.json').write_text(json.dumps(env, indent=2))
    print(pd.DataFrame(rows).groupby('regime').agg(agreement=('adaptive_agrees', 'mean'),
          median_speedup=('speedup_total', 'median'), faces=('active_face_certificates', 'sum'),
          fallback=('numerical_fallbacks', 'sum'), failures=('containment_failures', 'sum')).to_string())


if __name__ == '__main__':
    main()
