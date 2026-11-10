"""Time the complete audit of default OrchardSprays fits, including repairs."""
from time import perf_counter

import numpy as np
import pandas as pd

from cases import HERE, CompressedFold, cases
from default_fits import package_fit
from boundary_acv import active_step, face_certificate


def timed(key, inputs, repeat):
    before = perf_counter()
    theta, _, reported_convergence, _ = package_fit(inputs)
    fitted = perf_counter()
    fold = CompressedFold(*inputs)
    ev = fold.evaluate(theta)
    cert = face_certificate(fold, theta, ev)
    first = perf_counter()
    steps = 0
    while not cert["valid"] and steps < 20:
        theta = active_step(fold, theta, ev)
        ev = fold.evaluate(theta)
        cert = face_certificate(fold, theta, ev)
        steps += 1
    finished = perf_counter()
    assert cert["valid"]
    return dict(**key, repeat=repeat, optimizer_converged=reported_convergence, steps=steps,
                package_ms=1000*(fitted-before), first_ms=1000*(first-fitted),
                repair_ms=1000*(finished-first), audit_including_repair_ms=1000*(finished-fitted))


if __name__ == "__main__":
    selected = [(k, f) for k, f, keep in cases("orchard") if keep]
    timed(*selected[0], 0)
    rng = np.random.default_rng(14862)
    rows = []
    for repeat in range(1,4):
        for index in rng.permutation(len(selected)):
            rows.append(timed(*selected[index], repeat))
    data = pd.DataFrame(rows).sort_values(["target","model_index","fold","repeat"])
    data.to_csv(HERE/"results/orchard_repair_timing.csv", index=False)
    per_fit = data.groupby(["target","model","fold"])[["package_ms","first_ms","repair_ms","audit_including_repair_ms"]].median()
    print(per_fit.median().to_string())
    print("Median paired complete overhead", (per_fit.audit_including_repair_ms / per_fit.package_ms).median())
