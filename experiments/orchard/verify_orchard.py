"""Independent 60-digit audit on the zero-column-variance face.

Run beside orchard_data.py, OrchardSprays.csv and results/orchard_folds.csv.
The profile calculation uses only the observed data and model definitions.
Archived estimates and intervals are read only after an independent refit.
"""
import argparse
import csv
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from orchard_data import read_data, write_rows

HERE = Path(__file__).resolve().parent


def matrix(array):
    return mp.matrix(np.asarray(array, dtype=float).tolist())


def dot(v):
    return mp.fdot(v, v)


def trace(a):
    return sum(a[i, i] for i in range(a.rows))


def profile_fit(design, y, row, col, train):
    x, z = matrix(design[train]), matrix(y[train])
    n, p = x.rows, x.cols
    levels = np.unique(row[train])
    row_train = matrix(row[train, None] == levels)
    row_test = matrix(row[~train, None] == levels)
    col_train = matrix(col[train, None] == np.unique(col[train]))
    sizes = [int(sum(row_train[:, j])) for j in range(row_train.cols)]
    if len(set(sizes)) != 1:
        raise ValueError('This profile shortcut assumes balanced retained row sizes.')
    m = sizes[0]
    a, b, c = x.T*x, x.T*z, dot(z)
    rx, ry = row_train.T*x, row_train.T*z
    d, e, f = rx.T*rx, rx.T*ry, dot(ry)

    def evaluate(lam):
        t = lam/(1+m*lam)
        dt = 1/(1+m*lam)**2
        inverse = (a-t*d)**-1
        h = b-t*e
        beta = inverse*h
        rss = c-t*f-(h.T*beta)[0]
        u = ry-rx*beta
        score = (len(levels)*m/(1+m*lam)-dt*trace(inverse*d)-(n-p)*dt*dot(u)/rss)/2
        return score, rss/(n-p), beta, u, inverse

    score0 = evaluate(mp.mpf(0))[0]
    if score0 >= 0:
        ratio = mp.mpf(0)
    else:
        low, high = mp.mpf(0), mp.mpf(1)/64
        while evaluate(high)[0] < 0:
            low, high = high, 2*high
            if high > 2**30:
                raise ArithmeticError('Could not bracket a stationary point on this face.')
        try:
            ratio = mp.findroot(lambda lam: evaluate(lam)[0], (low, high), tol=mp.mpf('1e-48'), verify=True)
        except (ValueError, ZeroDivisionError):
            ratio = -1
        if not low <= ratio <= high:
            for _ in range(200):
                mid = (low+high)/2
                if evaluate(mid)[0] < 0:
                    low = mid
                else:
                    high = mid
            ratio = (low+high)/2
    score, variance, beta, u, inverse = evaluate(ratio)
    t = ratio/(1+m*ratio)
    row_col = row_train.T*col_train
    xc = x.T*col_train-t*rx.T*row_col
    cy = col_train.T*z-t*row_col.T*ry-xc.T*beta
    trace_c = n-t*dot(row_col)-trace(inverse*(xc*xc.T))
    col_score = (trace_c/variance-dot(cy)/variance**2)/2
    row_score = score/variance
    if col_score <= 0 or (ratio == 0 and row_score <= 0) or (ratio > 0 and abs(row_score) > mp.mpf('1e-40')):
        raise ArithmeticError('The full-model first-order conditions do not support the proposed face.')
    prediction = matrix(design[~train])*beta+t*row_test*u
    return variance, ratio*variance, row_score, col_score, dot(matrix(y[~train])-prediction)/prediction.rows


def main(data, recorded, output):
    mp.mp.dps = 60
    y, row, col, designs = read_data(data)
    # Archive the original binary responses for diagnostics at the last bit.
    log_path = data.with_name('response_binary64.hex')
    if log_path.exists():
        recorded_logs = np.array([float.fromhex(line) for line in log_path.read_text().splitlines()])
        if recorded_logs.shape != y.shape or not np.allclose(y, recorded_logs, rtol=2e-15, atol=0):
            raise ValueError('Archived binary responses do not match the input data.')
        y = recorded_logs
    with recorded.open(encoding='utf-8', newline='') as stream:
        archive = {(f['target'], f['model'], int(f['held_out'])): f for f in csv.DictReader(stream)}
    rows, maxima = [], []
    for target, block in [('row', row), ('column', col)]:
        for name, design in designs.items():
            for group in np.unique(block):
                ve, vr, gr, gc, loss = profile_fit(design, y, row, col, block != group)
                f = archive[target, name, int(group)]
                lo, hi = [mp.mpf(float(f[k])) for k in ('lo','hi')]
                excess = max(mp.mpf(0), lo-loss, loss-hi)
                same = float(f['col_var']) == 0 and ((float(f['row_var']) == 0) == (vr == 0))
                maxima.append(excess)
                rows.append(dict(target=target, model=name, held_out=int(group), residual_var=mp.nstr(ve, 50),
                           row_var=mp.nstr(vr, 50), column_var=0, row_score=mp.nstr(gr, 28), column_score=mp.nstr(gc, 28),
                           loss=mp.nstr(loss, 50), same_face=same, positive_width=hi>lo, raw_contained=excess==0,
                           endpoint_excess=mp.nstr(excess, 12)))
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/'orchard_profile60.csv',rows)
    summary = dict(decimal_precision=mp.mp.dps, mpmath=mp.__version__, n_fits=len(rows), matched_faces=sum(r['same_face'] for r in rows),
              nonzero_widths=sum(r['positive_width'] for r in rows), covered_nonzero=sum(r['positive_width'] and r['raw_contained'] for r in rows),
              covered_zero_widths=sum(not r['positive_width'] and r['raw_contained'] for r in rows),
              max_excess=mp.nstr(max(maxima),12), min_column_kkt=mp.nstr(min(mp.mpf(r['column_score']) for r in rows),12))
    (output/'orchard_profile60.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE/'OrchardSprays.csv')
    p.add_argument('--recorded', type=Path, default=HERE/'results'/'orchard_folds.csv')
    p.add_argument('--output', type=Path, default=HERE/'results')
    args = p.parse_args()
    main(args.data,args.recorded,args.output)
