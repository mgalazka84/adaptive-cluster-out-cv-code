"""Analyze all prespecified training-only timing jobs without exclusions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'training_timing'
def read(name):
    p=OUT/(name+'.csv')
    return pd.read_csv(p if p.exists() else p.with_suffix('.csv.gz'))
def main():
    plan=json.loads((OUT/'plan.json').read_text())
    r,t,f,c=[read(x) for x in ['runs','timings','folds','tests']]
    assert len(r)==22 and len(f)==5100
    assert set(r[r.kind=='large_fresh'].seed)==set(plan['simulation_seeds'])
    assert set(r.kind)=={'large_fresh','new_subject','new_word'}
    assert len(t)==(20*3+2*7)*7
    assert not r.duplicated(['kind','seed']).any()
    assert not f.duplicated(['kind','seed','model','group']).any()
    for (kind,seed,rep),z in t.groupby(['kind','seed','repeat']):
        assert set(z.method)==set(plan['methods']) and len(z)==7
        assert set(z.position)==set(range(7))
    # Recompute medians and ratios from repetition-level records.
    for row in r.itertuples():
        med=t[(t.kind==row.kind)&(t.seed==row.seed)].groupby('method').seconds.median()
        for method in plan['methods']:
            assert np.isclose(med[method],getattr(row,method+'_seconds'),rtol=1e-12,atol=1e-12)
        for method in ['practical','default_lbfgs','bound_once','norm_only']:
            q=(row.common_seconds+med[method])/(row.common_seconds+med.adaptive)
            assert np.isclose(q,getattr(row,method+'_total_ratio'),rtol=1e-12)
    groups=[];work=[];spread=[]
    for kind,z in r.groupby('kind',sort=False):
        q=dict(kind=kind,comparisons=len(z),fits=int(z.folds.sum()),
          adaptive_valid=int(z.adaptive_valid.sum()),fallbacks=int(z.fallbacks.sum()),
          anchor_fallbacks=int(z.anchor_fallbacks.sum()),bound_once_resolves=int(z.bound_once_resolves.sum()),
          bound_once_valid=int(z.bound_once_valid.sum()),norm_rounds=int(z.norm_rounds.sum()),
          adaptive_rounds=int(z.adaptive_rounds.sum()),common_ms=float(1000*z.common_seconds.median()))
        for col in r.columns:
            if col.endswith('_ratio'):q[col]=float(z[col].median())
            if col.endswith('_seconds') and col!='common_seconds':q[col.replace('_seconds','_ms')]=float(1000*z[col].median())
        for method in plan['methods']:
            q[method+'_faster']=int((z[method+'_seconds']<z.adaptive_seconds).sum())
            work.append(dict(kind=kind,method=method,iterations_per_fit=float(z[method+'_iterations'].sum()/z.folds.sum()),
                tests_per_fit=float(z[method+'_checks'].sum()/z.folds.sum()),
                disagreement_count=int((~z[method+'_agrees']).sum()),
                max_cv_error=float(z[method+'_max_cv_error'].max())))
        groups.append(q)
    for (kind,seed,method),z in t.groupby(['kind','seed','method']):
        spread.append(dict(kind=kind,seed=int(seed),method=method,relative_range=float((z.seconds.max()-z.seconds.min())/z.seconds.median())))
    summary=pd.DataFrame(groups);summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(work).to_csv(OUT/'work_summary.csv',index=False)
    sp=pd.DataFrame(spread);sp.to_csv(OUT/'repetition_ranges.csv',index=False)
    valid=f[f.valid];width=valid.upper-valid.lower
    check=dict(comparisons=len(r),fits=len(f),timed_runs=len(t),
      all_methods_all_choices_agree=bool(r[[m+'_agrees' for m in plan['methods']]].all().all()),
      repeated_choices_identical=bool(t.repeat_agrees.all()),
      valid_final=int(f.valid.sum()),fallbacks=int(f.fallback.sum()),
      anchor_fallbacks=int(r.anchor_fallbacks.sum()),
      strict_references_in_uniqueness_region=bool(valid.root_in_r.all()),
      reference_active_sets_match=bool(valid.active_match.all()),
      positive_width_misses=int(((width>0)&(valid.excess>0)).sum()),
      zero_width_misses=int(((width==0)&(valid.excess>0)).sum()),
      max_raw_excess=float(valid.excess.max()),
      max_strict_loss_difference=float(abs(f.newton_loss-f.lbfgs_loss).max()),
      bound_once_resolved=int(r.bound_once_resolves.sum()),
      bound_once_valid=int(r.bound_once_valid.sum()),
      adaptive_rounds=int(r.adaptive_rounds.sum()),norm_rounds=int(r.norm_rounds.sum()),
      local_test_attempts=len(c),
      all_practical_faster=bool((r.practical_seconds<r.adaptive_seconds).all()),
      all_bound_once_faster=bool((r.bound_once_seconds<r.adaptive_seconds).all()),
      all_strict_faster=bool((r.strict_kernel_ratio<1).all()))
    check['relative_range_by_method']={m:{'median':float(z.relative_range.median()),'p95':float(z.relative_range.quantile(.95))} for m,z in sp.groupby('method')}
    (OUT/'audit_summary.json').write_text(json.dumps(check,indent=2)+'\n')
    print(summary[['kind','adaptive_ms','practical_ms','bound_once_ms','strict_total_ratio','practical_total_ratio','default_lbfgs_total_ratio','bound_once_total_ratio']].to_string(index=False))
    print(json.dumps(check,indent=2))
if __name__=='__main__':main()
