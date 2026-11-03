"""Practical training-only fit, one local test, then unresolved-only repair.

This is the recommended post-fit procedure. The archived adaptive() API and
its timing outputs are unchanged. Tests use binary64, not directed rounding.
The 12/20 work limits count Newton updates AFTER the practical fit. They are
engineering limits, not separately tuned or benchmarked hybrid budgets.
"""
import numpy as np
from boundary_acv import State, fit_box, face_certificate, training_start


def postfit_validate(folds, *, weights=None, initial_budget=12, max_steps=20,
                     selection_atol=1e-10, selection_rtol=1e-10, starts=None):
    """Return winner, intervals, states and explicit local-bound/fallback status.

    Defaults use each fold's training responses only. Explicit model starts
    are supported solely for reproducing archived warm-start examples; a
    caller must not interpret those as training-only evaluation.
    """
    M = len(folds)
    if M < 2 or not folds[0] or any(len(row) != len(folds[0]) for row in folds):
        raise ValueError('Require at least two models with equally many nonempty folds')
    G = len(folds[0])
    w = np.full(G, 1 / G) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (G,) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('Weights must be finite, nonnegative and sum to one')
    if not all(isinstance(b, (int, np.integer)) for b in (initial_budget, max_steps)) or not 1 <= initial_budget <= max_steps:
        raise ValueError('Require integer budgets 1 <= initial_budget <= max_steps')
    if any(not np.isfinite(t) or t < 0 for t in (selection_atol, selection_rtol)):
        raise ValueError('Tie tolerances must be finite and nonnegative')
    if starts is not None and len(starts) != M:
        raise ValueError('Explicit starts must have one entry per model')
    states, fit_metadata = [], []
    for m, row in enumerate(folds):
        for k, fold in enumerate(row):
            start = training_start(fold) if starts is None else np.asarray(starts[m]).copy()
            theta, ev, meta = fit_box(fold, start, gtol=1e-6, ftol=1e-9,
                                      polish=False, terminal_hessian=True)
            cert = face_certificate(fold, theta, ev, loss_bound='directional')
            states.append(State(fold, theta, m, k, ev=ev, cert=cert, checks=1,
                                anchor=cert if cert['valid'] else None,
                                initial_budget=initial_budget, max_steps=max_steps))
            fit_metadata.append(meta)
    initial_valid = sum(s.cert['valid'] for s in states)

    def bounds():
        lo = np.array([s.cert['lo'] for s in states]).reshape(M, G) @ w
        hi = np.array([s.cert['hi'] for s in states]).reshape(M, G) @ w
        best = int(np.argmin(hi))
        other = np.min(np.delete(lo, best))
        guard = selection_atol + selection_rtol * max(abs(hi[best]), abs(other))
        return lo, hi, best, guard, bool(hi[best] + guard < other)

    initial_resolved = initial_valid == len(states) and bounds()[-1]
    # A failed first test has no interval: repair it before aggregation.
    for s in states:
        while 'lo' not in s.cert:
            s.refine()
    rounds = 0
    while True:
        lo, hi, best, guard, separated = bounds()
        active = np.flatnonzero(lo <= hi[best] + guard)
        choices = [s for s in states if s.model in active and not s.numerical]
        if separated or not choices:
            return dict(winner=best if separated else None, lower=lo, upper=hi,
                        states=states, fit_metadata=fit_metadata,
                        initial_valid_fits=initial_valid, initial_resolved=bool(initial_resolved),
                        refinement_rounds=rounds,
                        resolved_by_local_bounds=bool(separated and all(s.cert['valid'] for s in states)),
                        numerical_fallbacks=sum(s.numerical for s in states))
        s = max(choices, key=lambda s: w[s.group] * (s.cert['hi'] - s.cert['lo']))
        s.refine()
        while 'lo' not in s.cert:
            s.refine()
        rounds += 1
