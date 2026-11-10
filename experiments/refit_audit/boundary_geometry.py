"""Evaluate the same tightness diagnostics on the 48 training-only OrchardSprays fits."""
import numpy as np
import pandas as pd

from cases import HERE, CompressedFold, cases
from training_diagnostics import one


if __name__ == "__main__":
    frame = pd.DataFrame([dict(**key, **one(CompressedFold(*inputs)))
                          for key, inputs, _ in cases("orchard")])
    frame.to_csv(HERE / "results/orchard_geometry.csv", index=False)
    subset = frame[frame.usable & frame.accepted & frame.same_face & frame.in_ball]
    print("Accepted, with active components, above error threshold:",
          len(frame[frame.accepted]), len(frame[frame.active_count.gt(0)]), len(subset))
    if len(subset):
        ratio = subset.half_width / subset.loss_error
        print("Half-width/error quantiles", ratio.quantile([.5, .95, 1.]).to_dict())
        print("Condition range", np.min(subset.hessian_condition), np.max(subset.hessian_condition))
