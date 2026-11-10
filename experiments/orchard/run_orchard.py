"""Repeat the 48 OrchardSprays REML fits and audit their local loss intervals.

Run in the directory containing this script and OrchardSprays.csv. The solver modules
are loaded from the code directory at the root of this repository.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

from orchard_data import read_data, write_rows

HERE = Path(__file__).resolve().parent


def run(solver, data_path, output):
    sys.path.insert(0, str(solver.resolve()))
    from acv import Fold, CompressedFold
    from boundary_acv import fit_box, training_start
    from postfit_validation import postfit_validate

    y, row, col, designs = read_data(data_path)
    fold_records, score_records, targets = [], [], []
    for target, block in [('row', row), ('column', col)]:
        groups = np.unique(block)
        problems = []
        for design in designs.values():
            folds = []
            for group in groups:
                select = block == group
                held = (design[select], y[select], row[select], col[select])
                folds.append(CompressedFold(design[~select], y[~select], row[~select], col[~select], held))
            problems.append(folds)
        answer = postfit_validate(problems)
        winner = list(designs)[answer['winner']] if answer['winner'] is not None else 'unresolved'
        targets.append(dict(target=target, winner=winner, first_accepted=answer['initial_valid_fits'],
                            first_separated=answer['initial_resolved'], locally_resolved=answer['resolved_by_local_bounds'],
                            extra_updates=sum(s.steps for s in answer['states']), fallbacks=answer['numerical_fallbacks']))
        for state in answer['states']:
            f, cert = state.fold, state.cert
            dense = Fold(f.X, f.y, f.a, f.b, f.test)
            ref, ev, meta = fit_box(dense, training_start(dense), method='newton')
            pred = dense.prediction(ref, ev)[0]
            loss = float(np.mean((dense.test[1] - pred) ** 2))
            active = np.flatnonzero(state.theta == 0)
            free = np.flatnonzero(state.theta > 0)
            distance = float(np.linalg.norm(np.linalg.solve(cert['S'][free], ref[free] - state.theta[free]))) if cert['valid'] else None
            margin = cert.get('active_lower', [])
            fold_records.append(dict(target=target, model=list(designs)[state.model], held_out=int(groups[state.group]),
                checks=state.checks, updates=state.steps, numerical_fallback=bool(state.numerical),
                status=cert['reason'], residual_var=state.theta[0], row_var=state.theta[1], col_var=state.theta[2],
                active_count=len(active), min_active_gradient_lower=float(min(margin)) if len(margin) else '',
                loss=cert['mse'], lo=cert['lo'], hi=cert['hi'], dense_loss=loss,
                dense_residual_var=ref[0], dense_row_var=ref[1], dense_col_var=ref[2],
                same_face=np.array_equal(active, np.flatnonzero(ref == 0)), dense_projected_gradient=meta['projected_gradient'],
                uniqueness_radius=cert.get('outer_radius', ''), dense_root_distance=distance,
                dense_in_uniqueness_ball=distance is not None and distance < cert['outer_radius'],
                endpoint_excess=max(cert['lo']-loss, loss-cert['hi'], 0.0)))
        for idx, name in enumerate(designs):
            rows = [f for f in fold_records if (f['target'], f['model']) == (target, name)]
            score_records.append(dict(target=target, model=name, strict_score=float(np.mean([x['dense_loss'] for x in rows])),
                lower=answer['lower'][idx], upper=answer['upper'][idx], zero_row=sum(x['row_var']==0 for x in rows),
                zero_column=sum(x['col_var']==0 for x in rows), both_zero=sum(x['active_count']==2 for x in rows),
                any_zero=sum(x['active_count']>0 for x in rows), n_folds=len(rows)))
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/'orchard_folds.csv', fold_records)
    write_rows(output/'orchard_scores.csv', score_records)
    summary = dict(data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                   python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, targets=targets,
                   n_fits=len(fold_records), boundary_fits=sum(f['active_count']>0 for f in fold_records),
                   double_zero=sum(f['active_count']==2 for f in fold_records),
                   min_active_lower=min(f['min_active_gradient_lower'] for f in fold_records if f['active_count']),
                   same_faces=sum(f['same_face'] for f in fold_records),
                   dense_roots_in_ball=sum(f['dense_in_uniqueness_ball'] for f in fold_records),
                   raw_contained=sum(f['endpoint_excess']==0 for f in fold_records),
                   max_raw_excess=float(max(f['endpoint_excess'] for f in fold_records)),
                   max_dense_projected_gradient=float(max(f['dense_projected_gradient'] for f in fold_records)))
    (output/'orchard_summary.json').write_text(json.dumps(summary, indent=2, default=lambda x: x.item())+'\n', encoding='utf-8')
    print((output/'orchard_summary.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, default=HERE.parents[1] / 'code')
    parser.add_argument('--data', type=Path, default=HERE/'OrchardSprays.csv')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    args = parser.parse_args()
    run(args.solver, args.data, args.output)
