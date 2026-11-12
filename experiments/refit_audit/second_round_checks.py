"""Summarize simple local diagnostics, paired costs, and budget counts."""
import json
import numpy as np
import pandas as pd

from cases import HERE, ROOT, CompressedFold, cases
from boundary_acv import projected_gradient

OUT = HERE / "results"


def basic_diagnostics():
    recorded = pd.read_csv(OUT / "default_folds.csv", float_precision="round_trip")
    recorded = recorded[recorded.dataset.eq("orchard")].set_index(["target", "model", "fold"])
    rows = []
    for key, inputs, _ in cases("orchard"):
        source = recorded.loc[(key["target"], key["model"], key["fold"])]
        theta = source[["residual", "var_a", "var_b"]].to_numpy(dtype=float)
        fold = CompressedFold(*inputs)
        ev = fold.evaluate(theta)
        free = np.flatnonzero(theta > 0)
        minimum = float(np.linalg.eigvalsh(ev.hessian[np.ix_(free, free)])[0])
        pg = float(np.linalg.norm(projected_gradient(theta, ev.gradient), np.inf))
        rows.append(dict(**key, projected_gradient_norm=pg, free_hessian_minimum=minimum,
                         gradient_flag=pg > 1e-6, curvature_flag=minimum <= 0,
                         either_flag=pg > 1e-6 or minimum <= 0,
                         original_failure=source.reason, original_converged=bool(source.converged)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "orchard_basic_diagnostics.csv", index=False)
    return dict(fits=len(frame), gradient_flags=int(frame.gradient_flag.sum()),
                curvature_flags=int(frame.curvature_flag.sum()),
                either_flags=int(frame.either_flag.sum()),
                min_gradient=float(frame.projected_gradient_norm.min()),
                max_gradient=float(frame.projected_gradient_norm.max()),
                gradient_tolerance=1e-6, curvature_threshold=0,
                coordinates="Original, unscaled variance coordinates; strictly positive components are free.")


def timing_spread():
    out = {}
    for filename in ["default_timing.csv", "orchard_repair_timing.csv"]:
        data = pd.read_csv(OUT / filename, float_precision="round_trip")
        data = data[data.dataset.eq("orchard")]
        grouped = data.groupby(["target", "model", "fold"]).package_ms
        assert len(grouped) == 18 and grouped.size().eq(3).all()
        medians = grouped.median()
        spread = (grouped.max() - grouped.min()) / medians
        out[filename] = {
            "fits": 18, "repetitions": 3, "package_ms": float(medians.median()),
            "within_fit_relative_range_median": float(spread.median()),
            "within_fit_relative_range_q25": float(spread.quantile(.25)),
            "within_fit_relative_range_q75": float(spread.quantile(.75)),
            "package_median_by_repetition": data.groupby("repeat").package_ms.median().to_dict()
        }
    runs = pd.read_csv(ROOT / "results/training_timing/runs.csv", float_precision="round_trip")
    sim = runs[runs.kind.eq("large_fresh")]
    pair = sim.strict_total_ratio / sim.bound_once_total_ratio
    out["simulation_strict_over_checked_practical"] = {
        "datasets": len(pair), "median": float(pair.median()),
        "minimum": float(pair.min()), "maximum": float(pair.max())}
    return out


def budget_design():
    runs = pd.read_csv(ROOT / "results/revision_study/runs.csv")
    intercept = runs[runs.kind.eq("intercept") & runs.setting.isin(["default", "budget_4_7"])]
    assert intercept.groupby("setting").size().eq(8).all()
    design = []
    for row in intercept.itertuples():
        regime = ["balanced", "boundary", "zero_both", "near_tie"][row.seed % 4]
        design.append(dict(seed=row.seed, regime=regime, setting=row.setting, components=3,
                           n_observations=240, candidates=3, held_out_groups=12,
                           fits=row.folds, passed=row.valid, fallbacks=row.fallbacks))
    frame = pd.DataFrame(design)
    frame.to_csv(OUT / "budget_design.csv", index=False)
    assert frame.fits.eq(3*12).all()
    return {setting: {"datasets": len(g), "fits": int(g.fits.sum()), "passed": int(g.passed.sum()),
                      "fallbacks": int(g.fallbacks.sum())} for setting, g in frame.groupby("setting")}


if __name__ == "__main__":
    result = dict(basic_diagnostics=basic_diagnostics(), timing=timing_spread(), budgets=budget_design())
    (OUT / "second_round_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
