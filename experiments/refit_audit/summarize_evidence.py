"""Join the independently fitted losses to the training-only intervals and tabulate comparators."""
import json

import numpy as np
import pandas as pd

from cases import HERE, ROOT, cases


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def strict_overlap(out):
    local = read(ROOT / "results/postfit_validation/lexdec_folds.csv").rename(
        columns={"kind": "target", "group": "group_index"})
    independent = read(ROOT / "results/lexdec/statsmodels_audit.csv")
    keys = ["target", "model", "group_index"]
    joined = independent.merge(local, on=keys, validate="one_to_one", how="outer", indicator=True)
    labels = {(k["target"], k["model_index"], k["fold"]): k["group_label"]
              for k, _, _ in cases("lexdec")}
    assert len(joined) == len(labels) == 300 and joined["_merge"].eq("both").all()
    assert all(labels[(r.target, r.model, r.group_index)] == str(r.group) for r in joined.itertuples())
    assert np.max(abs(joined.reference_loss - joined.strict_loss)) < 5e-12
    assert joined[["valid", "active_match", "reference_in_uniqueness_ball"]].all().all()
    joined["inside"] = joined.external_loss.between(joined.lower, joined.upper)
    width = joined.upper - joined.lower
    assert (width > 0).all()
    joined["nearer_endpoint_fraction"] = np.minimum(joined.external_loss - joined.lower,
                                                  joined.upper - joined.external_loss) / width
    joined = joined[keys + ["group", "lower", "upper", "strict_loss", "external_loss",
                            "external_converged", "inside", "nearer_endpoint_fraction"]]
    joined.to_csv(out / "strict_training_overlap.csv", index=False)
    return {"targets": joined.groupby("target").inside.agg(["sum", "count"]).to_dict("index"),
            "statuses": joined.groupby("external_converged").agg(
                total=("inside", "size"), inside=("inside", "sum"),
                minimum_fraction=("nearer_endpoint_fraction", "min"),
                median_fraction=("nearer_endpoint_fraction", "median")).to_dict("index")}


def simulation(out):
    geometry = read(out / "training_geometry.csv.gz")
    results = []
    for seed, folds in geometry.groupby("seed"):
        assert folds.groupby("model").size().eq(30).all()
        scores = folds.groupby("model").reference_loss.mean().sort_values()
        winner, second = scores.index[:2]
        pair = folds[folds.model.isin([winner, second])]
        loss = pair.pivot(index="fold", columns="model", values="reference_loss")
        half_width = pair.groupby("model").half_width.mean().sum()
        paired_se = (loss[second] - loss[winner]).std(ddof=1) / np.sqrt(len(loss))
        results.append(dict(seed=seed, winner=winner, runner_up=second, gap=scores[second]-scores[winner],
                            numerical_half_width=half_width, fold_se_proxy=paired_se, ratio=half_width/paired_se))
    frame = pd.DataFrame(results)
    frame.to_csv(out / "simulation_scales.csv", index=False)
    runs = read(ROOT / "results/training_timing/runs.csv")
    runs["check_practical"] = (runs.common_seconds + runs.bound_once_seconds) / (runs.common_seconds + runs.practical_seconds)
    ratios = runs.groupby("kind")[["strict_total_ratio", "practical_total_ratio", "bound_once_total_ratio",
                                   "check_practical"]].median().to_dict("index")
    return {"datasets": len(frame), "median_scale_ratio": frame.ratio.median(),
            "maximum_scale_ratio": frame.ratio.max(), "paired_timing_ratios": ratios,
            "training_practical_and_check_agree": int((runs.practical_agrees & runs.bound_once_agrees).sum())}


def default_summary(out):
    full = read(out / "default_folds.csv")
    assert full.produced.all()
    aggregate = full.groupby(["dataset","target","model"])[["mse","reference_loss"]].mean().reset_index()
    aggregate.to_csv(out / "default_aggregate.csv", index=False)
    reversals = []
    for (ds, target, label), group in full.groupby(["dataset","target","group_label"]):
        default = group.loc[group.mse.idxmin(), "model"]
        strict = group.loc[group.reference_loss.idxmin(), "model"]
        if default != strict:
            reversals.append(dict(dataset=ds, target=target, label=label, default=default, strict=strict))
    timing = read(out / "default_timing.csv")
    keys = ["dataset", "target", "model", "fold"]
    assert timing.groupby(keys).size().eq(3).all()
    per_fit = timing.groupby(keys)[["package_ms","first_check_ms"]].median()
    per_fit["check_over_fit"] = per_fit.first_check_ms / per_fit.package_ms
    return {"within_fold_changes": reversals,
            "timing": per_fit.groupby("dataset").median().to_dict("index")}


if __name__ == "__main__":
    out = HERE / "results"
    summary = {"strict_training_overlap": strict_overlap(out), "simulation": simulation(out),
               "defaults": default_summary(out)}
    (out / "evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
