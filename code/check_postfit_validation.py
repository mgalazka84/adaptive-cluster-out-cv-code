"""Functional checks, not a new timing study, for the post-fit procedure."""
import json
from pathlib import Path
import numpy as np
from boundary_acv import training_start, fit_box, face_certificate
from postfit_validation import postfit_validate
from run_accuracy_benchmark import problem, run


def main():
    folds, _, _ = problem('balanced', 74003000, na=8, nb=6)
    result = postfit_validate(folds)
    independent = []
    for row in folds:
        for f in row:
            t, e, _ = fit_box(f, training_start(f), gtol=1e-6, ftol=1e-9,
                              polish=False, terminal_hessian=True)
            independent.append(face_certificate(f, t, e))
    assert all(c['valid'] for c in independent)
    assert result['initial_resolved'] and result['resolved_by_local_bounds']
    assert all(s.steps == 0 and s.checks == 1 for s in result['states'])
    np.testing.assert_allclose(result['lower'], np.array([c['lo'] for c in independent]).reshape(3, 8).mean(axis=1), rtol=0, atol=1e-15)
    first = dict(valid_fits=result['initial_valid_fits'], winner=result['winner'],
                 additional_updates=sum(s.steps for s in result['states']))

    # Reproduce the previously archived unresolved example, explicitly using
    # its full-data starts. This is a regression check, not new held-out data.
    archived, starts, _ = problem('zero_both', 20267031)
    once = run('certify_once', archived, starts)
    assert once['valid_fits'] == 35 and once['certified_winner'] is None
    repaired = postfit_validate(archived, starts=starts)
    assert repaired['resolved_by_local_bounds'] and not repaired['initial_resolved']
    assert sum(s.steps for s in repaired['states']) == 1
    assert repaired['winner'] == run('newton', archived, starts)['winner']
    repaired_summary = dict(initial_valid_fits=35, total_fits=36,
                            additional_updates=1, winner=repaired['winner'])

    # Identical candidates must remain tied; a budget-completed point value
    # must never be relabelled as an accepted local-bound result.
    tied = postfit_validate([folds[0], folds[0]], initial_budget=1, max_steps=1)
    assert tied['winner'] is None and not tied['resolved_by_local_bounds']
    assert tied['numerical_fallbacks'] == 16
    record = dict(scope='functional regression checks; no new performance experiment',
                  training_only_first_stage=first,
                  archived_full_data_start_repair=repaired_summary,
                  identical_candidates=dict(winner=None, fallback_fits=16),
                  checks_passed=True)
    out = Path(__file__).resolve().parents[1] / 'results' / 'postfit_validation'
    out.mkdir(exist_ok=True)
    (out / 'checks.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
