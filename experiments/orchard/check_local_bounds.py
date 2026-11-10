"""Independently refit OrchardSprays and replay the first local test.

This dense implementation uses full error contrasts throughout. Its floating-
point results can differ in the last digits from the compressed implementation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, cho_solve, eigh, qr, solve_triangular
from scipy.optimize import minimize

from orchard_data import read_data, write_rows

HERE = Path(__file__).resolve().parent


def symwhiten(factor, matrix):
    """Return L^-1 M L^-T for a lower triangular matrix L."""
    left = solve_triangular(factor, matrix, lower=True)
    return solve_triangular(factor, left.T, lower=True).T


def refit_and_check(design, response, row, column, train):
    x0, x1 = design[train], design[~train]
    y0, y1 = response[train], response[~train]
    n, p = x0.shape
    train_cov = np.array([
        np.eye(n),
        np.equal.outer(row[train], row[train]),
        np.equal.outer(column[train], column[train]),
    ], dtype=float)
    cross_cov = np.array([
        np.zeros((len(y1), n)),
        np.equal.outer(row[~train], row[train]),
        np.equal.outer(column[~train], column[train]),
    ], dtype=float)
    contrast = qr(x0, mode='full')[0][:, p:]
    components = contrast.T @ train_cov @ contrast
    z = contrast.T @ y0
    scale = max(np.var(y0), 1.e-5)

    def objective(relative):
        covariance = np.einsum('i,ijk->jk', scale * relative, components)
        chol = cholesky(covariance, lower=True)
        w = solve_triangular(chol, z, lower=True)
        inverse = cho_solve((chol, True), np.eye(n - p))
        v = cho_solve((chol, True), z)
        gradient = .5 * (np.einsum('ab,iab->i', inverse, components)
                         - np.einsum('a,iab,b->i', v, components, v))
        return np.log(np.diag(chol)).sum() + np.dot(w, w) / 2, scale * gradient

    fit = minimize(
        objective, [.5, .25, .25], jac=True, method='L-BFGS-B',
        bounds=[(1.e-10 / scale, None), (0, None), (0, None)],
        options={'ftol': 1.e-9, 'gtol': 1.e-6 * scale, 'maxls': 40, 'maxiter': 500},
    )
    theta = scale * fit.x
    covariance = np.einsum('i,ijk->jk', theta, components)
    chol = cholesky(covariance, lower=True)
    w = solve_triangular(chol, z, lower=True)
    derivatives = np.array([symwhiten(chol, part) for part in components])
    dw = derivatives @ w
    gradient = (np.trace(derivatives, axis1=1, axis2=2) - dw @ w) / 2
    hessian = dw @ dw.T - np.einsum('iab,jab->ij', derivatives, derivatives) / 2

    # Build the inverse square root only on the free variance coordinates.
    free = theta > 0
    eigenvalues, vectors = eigh(hessian[np.ix_(free, free)])
    if eigenvalues[0] <= 1.e-10:
        raise ArithmeticError('Insufficient curvature on the estimated face')
    geometry = np.zeros((len(theta), sum(free)))
    geometry[free] = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    eta = np.linalg.norm(geometry.T @ gradient)
    radius = max(2 * eta, 1.e-12)
    if not np.all(theta[free] > radius * np.linalg.norm(geometry[free], axis=1)):
        raise ArithmeticError('A free variance reaches the boundary')
    directions = np.einsum('ij,iab->jab', geometry, derivatives)
    a = np.linalg.norm([np.linalg.norm(direction, 2) for direction in directions])
    t = a * radius
    if t >= 1:
        raise ArithmeticError('Covariance perturbation exceeds unit radius')
    w_norm = np.linalg.norm(w)
    b = np.linalg.norm((directions @ w).T, 2) + a * t * w_norm / (1 - t)
    lipschitz = (np.sum(directions**2) * a / (1 - t)**3
                 + 3 * b * b * a / (1 - t)**2)
    if lipschitz * radius >= 1 or eta + lipschitz * radius**2 / 2 > radius:
        raise ArithmeticError('Contraction or self-map condition failed')
    root = 2 * eta / (1 + np.sqrt(1 - 2 * lipschitz * eta))

    # Test both one-sided score bounds for every zero variance.
    t = a * root
    delta = t / (1 - t)
    for j in np.flatnonzero(~free):
        trace = np.trace(derivatives[j])
        largest_eigenvalue = eigh(derivatives[j], eigvals_only=True)[-1]
        perturbation = np.sqrt(largest_eigenvalue) * delta * w_norm
        low1 = (trace / (1 + t) - (np.sqrt(max(0., w @ dw[j])) + perturbation)**2) / 2
        low2 = gradient[j] - (delta * trace + 2 * np.linalg.norm(dw[j]) * delta * w_norm
                              + largest_eigenvalue * (delta * w_norm)**2) / 2
        if max(low1, low2) <= 0:
            raise ArithmeticError('Unresolved active gradient')

    # Transfer the parameter enclosure to the held-out loss.
    weight = solve_triangular(chol, contrast.T, lower=True).T
    training_covariance = np.einsum('i,ijk->jk', theta, train_cov)
    inverse_x = np.linalg.solve(training_covariance, x0)
    predictor = (x1 @ np.linalg.solve(x0.T @ inverse_x, inverse_x.T)
                 + np.einsum('i,ijk->jk', theta, cross_cov) @ weight @ weight.T)
    sensitivity = np.einsum('ij,iab->jab', geometry, cross_cov - predictor @ train_cov) @ weight
    residual = y1 - predictor @ y0
    e = np.linalg.norm(residual)
    linear = (sensitivity @ w).T
    remainder = np.linalg.norm(sensitivity) * a * root**2 * w_norm / (1 - a * root)
    displacement = root * np.linalg.norm(linear, 2) + remainder
    gamma = root * np.linalg.norm(residual @ linear) + e * remainder
    lower = max(max(0., e - displacement)**2, max(0., e * e - 2 * gamma)) / len(y1)
    upper = min((e + displacement)**2, e * e + 2 * gamma + displacement * displacement) / len(y1)
    return dict(row_var=theta[1], column_var=theta[2], loss=e * e / len(y1), lower=lower, upper=upper)


def main(data, output):
    y, row, column, designs = read_data(data)
    records, scores, summary = [], [], {}
    for target, block in [('row', row), ('column', column)]:
        bounds = {}
        for model, design in designs.items():
            fits = []
            for group in np.unique(block):
                fit = refit_and_check(design, y, row, column, block != group)
                records.append(dict(target=target, model=model, group=int(group), **fit))
                fits.append(fit)
            lower = np.mean([fit['lower'] for fit in fits])
            upper = np.mean([fit['upper'] for fit in fits])
            bounds[model] = (lower, upper)
            scores.append(dict(target=target, model=model, lower=lower, upper=upper))
        best = min(bounds, key=lambda model: bounds[model][1])
        best_upper = bounds[best][1]
        other_lower = min(interval[0] for model, interval in bounds.items() if model != best)
        guard = 1e-10 + 1e-10 * max(abs(other_lower), abs(best_upper))
        summary[target] = best if best_upper + guard < other_lower else 'unresolved'
    summary.update(
        fits=len(records),
        zero_column=sum(fit['column_var'] == 0 for fit in records),
        both_zero=sum(fit['column_var'] == 0 and fit['row_var'] == 0 for fit in records),
    )
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'dense_replay_folds.csv', records)
    write_rows(output / 'dense_replay_scores.csv', scores)
    (output / 'dense_replay.json').write_text(json.dumps(summary, indent=2, default=int) + '\n')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'OrchardSprays.csv')
    parser.add_argument('--output', type=Path, default=HERE / 'dense_replay')
    args = parser.parse_args()
    main(args.data, args.output)
