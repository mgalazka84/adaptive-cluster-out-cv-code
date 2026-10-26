"""Cost of completing the sole unresolved fit-then-certify-once comparison.

The extra work starts with the already computed practical terminal evaluation.
It does not rerun the practical fits or add shared preparation time. Timings
are a separate seven-repetition diagnostic, not the original benchmark.
"""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from run_accuracy_benchmark import problem, run
from boundary_acv import active_step, face_certificate, same_root

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'accuracy_v2' / 'summary'

def complete(folds, theta, evaluations, initial):
    M, G = len(folds), len(folds[0])
    fs = [f for row in folds for f in row]
    th = [x.copy() for x in theta]
    ev = list(evaluations)
    certs = [dict(c) for c in initial]
    updates = []
    def refine(i):
        old = certs[i] if certs[i]['valid'] else None
        th[i] = active_step(fs[i], th[i], ev[i])
        ev[i] = fs[i].evaluate(th[i])
        c = face_certificate(fs[i], th[i], ev[i], loss_bound='directional')
        updates.append(dict(model=i//G, group_index=i%G, valid=c['valid'], reason=c['reason']))
        if old and c['valid']:
            assert same_root(old, c)
            c['lo'], c['hi'] = max(old['lo'], c['lo']), min(old['hi'], c['hi'])
            assert c['lo'] <= c['hi']
        certs[i] = c
    for i in range(M*G):
        while not certs[i]['valid']:
            assert len(updates) < 30
            refine(i)
    while True:
        lo = np.array([c['lo'] for c in certs]).reshape(M, G).mean(axis=1)
        hi = np.array([c['hi'] for c in certs]).reshape(M, G).mean(axis=1)
        best = int(np.argmin(hi)); other = np.min(np.delete(lo, best))
        guard = 1e-10+1e-10*max(abs(hi[best]), abs(other))
        if hi[best]+guard < other:
            return dict(winner=best, updates=updates, lower=lo.tolist(), upper=hi.tolist(),
                        separation=float(other-hi[best]), guard=float(guard))
        candidates = np.flatnonzero(lo <= hi[best]+guard)
        i = max((i for i in range(M*G) if i//G in candidates),
                key=lambda i: certs[i]['hi']-certs[i]['lo'])
        assert len(updates) < 30
        refine(i)

def main():
    folds, starts, _ = problem('zero_both', 20267031)
    once = run('certify_once', folds, starts)
    assert once['certified_winner'] is None and once['valid_fits'] == 35
    evaluations = [f.evaluate(th) for f, th in zip([f for row in folds for f in row], once['theta'])]
    initial_failures = [dict(model=i//12, group_index=i%12, reason=c['reason'])
                        for i, c in enumerate(once['certs']) if not c['valid']]
    result = complete(folds, once['theta'], evaluations, once['certs'])
    seconds = []
    for _ in range(7):
        begin = perf_counter()
        repeat = complete(folds, once['theta'], evaluations, once['certs'])
        seconds.append(perf_counter()-begin)
        assert repeat == result
    assert result['winner'] == run('newton', folds, starts)['winner']
    result.update(suite='original', regime='zero_both', seed=20267031,
                  initial_valid_fits=35, total_fits=36, initial_failures=initial_failures,
                  completion_seconds=seconds, median_completion_seconds=float(np.median(seconds)),
                  scope='incremental completion from cached practical terminal evaluations; excludes prior fits and setup')
    (OUT / 'certify_once_completion.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
