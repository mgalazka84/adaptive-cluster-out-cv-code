"""Round-two diagnostics: every test, tolerance-free containment and radii."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from accuracy_io import read_csv
BASE=Path(__file__).resolve().parents[1]/'results'/'accuracy_v2'
OUT=BASE/'summary';OUT.mkdir(exist_ok=True)
def allfiles(name):
    return pd.concat([read_csv(BASE/s/name) for s in ['original','scale','fresh','lexdec']],ignore_index=True)
F=allfiles('fold_audit.csv');T=allfiles('certificate_tests.csv');R=allfiles('runs.csv');D=allfiles('scheduler_decisions.csv')
V=F[F.local_valid]
def quant(x):return x.quantile([0,.05,.5,.95,1]).to_dict()
ratios=V[V.loss_ratio.notna()]
out=dict(test_count=len(T),accepted_tests=int(T.valid.sum()),rejection_reasons=T[~T.valid].reason.value_counts().to_dict(),
         Lr_accepted=quant(T[T.valid].Lr),Lr_rejected=quant(T[~T.valid].Lr),
         Lr_unavailable=int(T.Lr.isna().sum()),parameter_ratio=quant(V.parameter_ratio),
         parameter_ratios_below_one=int((V.parameter_ratio<1).sum()),
         rho_failures=int(V.reference_in_rho.eq(False).sum()),r_failures=int(V.reference_in_r.eq(False).sum()),
         loss_failures_no_tolerance=int((~F.contained_no_tolerance).sum()),
         loss_failures_absolute_tolerance=int((~F.contained).sum()),
         containment_excess_max=float(F.containment_excess.max()),
         positive_width_violation_max=float(F.loc[F.upper>F.lower,'containment_excess_relative_width'].max()),
         zero_width_violations=int(((F.upper==F.lower)&~F.contained_no_tolerance).sum()),
         violation_active_counts=F[~F.contained_no_tolerance].active_count.value_counts().to_dict(),
         directional_loss_ratio=quant(V.loss_ratio),
         norm_loss_ratio_same_centres=quant(ratios.norm_loss_half_width/ratios.loss_error),
         norm_to_directional_width=quant(ratios.norm_loss_half_width/ratios.loss_half_width),
         directional_refinements=int(R.refinement_rounds.sum()),norm_refinements=int(R.norm_refinement_rounds.sum()),
         tie_guard_binding_events=int(R.tie_guard_binding_events.sum()),tie_guard=quant(D.guard),
         loss_magnitudes=quant(F.reference_loss),interval_width=quant(F.upper-F.lower),
         floor_uses=int((V.outer_radius==1e-12).sum()),self_map_max_ratio=float((V.self_map/V.outer_radius).max()),
         reporting_to_original_radius=quant(V.reporting_radius/V.outer_radius),
         reporting_radius=quant(V.reporting_radius),
         old_tube_epsilon=quant(V.tube_epsilon),larger_tube_epsilon=quant(V.larger_tube_epsilon),
         larger_tube_free_radius=quant(V.larger_tube_free_radius),
         failed_test_after_anchor=int((T.had_anchor&~T.valid).sum()))
(OUT/'directional_summary.json').write_text(json.dumps(out,indent=2))
T.groupby(['valid','reason']).agg(tests=('step','size'),Lr_min=('Lr','min'),Lr_median=('Lr','median'),Lr_max=('Lr','max')).to_csv(OUT/'test_failures.csv')
rows=[]
for (suite,regime),d in R.groupby(['suite','regime'],sort=False):
    rows.append(dict(suite=suite,regime=regime,datasets=len(d),norm_rounds=int(d.norm_refinement_rounds.sum()),directional_rounds=int(d.refinement_rounds.sum()),
        norm_kernel_median=d.norm_only_seconds.median(),directional_kernel_median=d.adaptive_seconds.median(),
        paired_kernel_ratio=(d.norm_only_seconds/d.adaptive_seconds).median(),
        practical_faster=int((d.practical_speedup_total<1).sum()),default_faster=int((d.default_speedup_total<1).sum())))
pd.DataFrame(rows).to_csv(OUT/'directional_comparison.csv',index=False)
print(json.dumps(out,indent=2))
