"""80-digit local reference correction at the single smaller-ball discrepancy.

This is a numerical estimate in the fixed binary64 compressed model, not an
enclosure of QR/compression errors or a directed-rounding proof.
"""
from pathlib import Path
import json
import numpy as np
import mpmath as mp
from accuracy_io import read_csv
from audit_constant_precision import matrix, vec
from run_accuracy_benchmark import problem
from boundary_acv import face_certificate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'accuracy_v2'

def main():
    mp.mp.dps = 80
    d = read_csv(BASE / 'original' / 'fold_audit.csv')
    row = d.loc[d.parameter_ratio.idxmin()]
    folds, _, _ = problem(row.regime, int(row.seed))
    fold = folds[int(row.model)][int(row.group_index)]
    center = np.array([row[f'theta_{j}'] for j in range(3)])
    reference = np.array([row[f'reference_theta_{j}'] for j in range(3)])
    cert = face_certificate(fold, center)
    free = list(map(int, cert['free']))
    cs = [matrix(c) for c in fold.C]; z = vec(fold.z)
    def derivatives(th):
        C = mp.zeros(len(z))
        for t, c in zip(th, cs): C += t*c
        inv = C**-1; alpha = inv*z
        tr = lambda A: sum(A[i,i] for i in range(A.rows))
        g = mp.matrix([mp.mpf('.5')*(tr(inv*c)-(alpha.T*c*alpha)[0]) for c in cs])
        g[0] += mp.mpf('.5')*fold.d0/th[0]-mp.mpf('.5')*mp.mpf(fold.s0)/th[0]**2
        h = mp.matrix(3)
        for i in range(3):
            for j in range(3):
                h[i,j] = -mp.mpf('.5')*tr(inv*cs[i]*inv*cs[j])+(alpha.T*cs[i]*inv*cs[j]*alpha)[0]
        h[0,0] += -mp.mpf('.5')*fold.d0/th[0]**2+mp.mpf(fold.s0)/th[0]**3
        return g, h
    th = vec(reference); Sf = matrix(cert['S'][free])
    g, h = derivatives(th)
    first = mp.lu_solve(mp.matrix([[h[i,j] for j in free] for i in free]), mp.matrix([g[i] for i in free]))
    first_error = mp.norm(mp.lu_solve(Sf, first))
    corrections = []
    for _ in range(8):
        g, h = derivatives(th)
        step = mp.lu_solve(mp.matrix([[h[i,j] for j in free] for i in free]), mp.matrix([g[i] for i in free]))
        corrections.append(float(mp.norm(step)))
        for j, i in enumerate(free): th[i] -= step[j]
        if mp.norm(step) < mp.mpf('1e-65'): break
    g, _ = derivatives(th)
    ref_error = mp.norm(mp.lu_solve(Sf, mp.matrix([th[i]-mp.mpf(reference[i]) for i in free])))
    center_error = mp.norm(mp.lu_solve(Sf, mp.matrix([th[i]-mp.mpf(center[i]) for i in free])))
    old_error = float(row.parameter_error)
    out = dict(suite='original', regime=str(row.regime), seed=int(row.seed),
               model=int(row.model), group_index=int(row.group_index), digits=80,
               rho=float(cert['radius']), binary64_reference_distance=old_error,
               binary64_radius_gap=old_error-float(cert['radius']),
               reference_error_first_newton_estimate=float(first_error),
               reference_to_high_precision_root=float(ref_error),
               center_to_high_precision_root=float(center_error),
               high_precision_root_inside_rho=bool(center_error <= mp.mpf(cert['radius'])),
               free_gradient_max=float(max(abs(g[i]) for i in free)),
               active_gradient_min=float(min(g[int(i)] for i in cert['active'])),
               newton_correction_norms=corrections,
               scope='fixed binary64 compressed arrays and whitening transform; no directed rounding')
    (BASE / 'summary' / 'reference_precision.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
