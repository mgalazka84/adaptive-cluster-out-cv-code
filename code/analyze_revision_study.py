"""Summaries and real-data figure, retaining every planned seed and setting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_revision_study import ROOT,OUT

def main():
    runs=pd.read_csv(OUT/'runs.csv',float_precision='round_trip');foldpath=OUT/'folds.csv'
    if not foldpath.exists():foldpath=OUT/'folds.csv.gz'
    folds=pd.read_csv(foldpath,float_precision='round_trip')
    tab=runs.groupby(['kind','setting']).agg(comparisons=('seed','size'),agreements=('agrees','sum'),
       final_pass=('valid','sum'),fallbacks=('fallbacks','sum'),checks=('checks','sum'),steps=('steps','sum'),
       active_tests_not_separated=('active_not_separated','sum'))
    tab.to_csv(OUT/'sensitivity_summary.csv')
    f=folds[folds.setting=='default'];starts=pd.read_csv(OUT/'starts.csv');ex=pd.read_csv(OUT/'external_timings.csv')
    paired=ex.groupby(['target','model','group_index','method']).seconds.median().unstack()
    losses=ex.pivot(index=['target','model','group_index','repeat'],columns='method',values='loss')
    allrows=[]
    for suite in ['original','scale','fresh','lexdec']:
        path=ROOT/'results'/'accuracy_v2'/suite/'runs.csv'
        if not path.exists():path=path.with_suffix('.csv.gz')
        r=pd.read_csv(path,float_precision='round_trip')
        for _,a in r.iterrows():
            ids=sorted([int(c[3:]) for c in r.columns if c.startswith('cv_') and c[3:].isdigit() and pd.notna(a[c])],key=lambda j:a[f'cv_{j}'])
            i,j=ids[:2];N=((a[f'upper_{i}']-a[f'lower_{i}'])+(a[f'upper_{j}']-a[f'lower_{j}']))/2
            allrows.append(dict(suite=suite,seed=a.seed,gap=a.cv_gap,proxy=a.paired_fold_se_proxy,half_width=N,
               num_ge_proxy=N>=a.paired_fold_se_proxy,gap_gt_proxy=a.cv_gap>a.paired_fold_se_proxy))
    scales=pd.DataFrame(allrows);scales.to_csv(OUT/'numerical_vs_fold_variation.csv',index=False)
    summary={'default_fits':len(f),'default_pass':int(f.valid.sum()),'default_model_agreements':int(runs[runs.setting=='default'].agrees.sum()),
      'active_second_branch_larger':int(f.second_branch_wins.sum()),'active_coordinate_tests':int(f.branch_checks.sum()),
      'positive_width_misses':int(((f.upper>f.lower)&(f.excess>0)).sum()),'zero_width_misses':int(((f.upper==f.lower)&(f.excess>0)).sum()),
      'max_excess':float(f.excess.max()),'loss_ratio_quantiles':f.loss_ratio.quantile([.5,.95,1]).to_dict(),
      'start_active_mismatches':int(starts.active_mismatches.sum()),'start_max_loss_difference':float(starts.max_loss_difference.max()),
      'external_seconds_medians':ex.groupby('method').seconds.median().to_dict(),
      'paired_bound_once_over_practical':float((paired.bound_once/paired.practical).median()),
      'paired_added_seconds':float((paired.bound_once-paired.practical).median()),
      'external_max_loss_difference':float(abs(losses.practical-losses.statsmodels).max()),
      'numerical_scale_ge_proxy':int(scales.num_ge_proxy.sum()),'scale_comparisons':len(scales),
      'numerical_to_proxy_quantiles':(scales.half_width/scales.proxy).quantile([.5,.95,1]).to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    trace=pd.read_csv(OUT/'lexdec_trace.csv',float_precision='round_trip')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(8,2.7),layout='constrained')
    for ax,(target,t) in zip(axs,trace.groupby('target',sort=False)):
        t=t[t.valid];width=t.upper-t.lower;pos=width>0
        ax.plot(t.step[pos],width[pos],'-o',color='#16647a',lw=1.5,ms=4)
        if (~pos).any():ax.scatter(t.step[~pos],np.full((~pos).sum(),1e-17),marker='v',color='#ae4b36',label='zero width')
        ax.set_yscale('log');ax.set_ylim(5e-18,1e-3);ax.set_xlim(.7,8.3);ax.set_xticks(range(1,9))
        ax.axvspan(.7,2.5,color='#e7e7e7');ax.text(1.55,2e-5,'no accepted\nbound',ha='center',fontsize=8)
        ax.set_title('New participant' if target=='new_subject' else 'New word',loc='left',fontweight='bold')
        ax.set_xlabel('Projected Newton step');ax.set_ylabel('Accepted loss interval width')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if (~pos).any():ax.legend(frameon=False,fontsize=8,loc='upper right')
    fig.savefig(OUT/'Fig8.pdf');fig.savefig(OUT/'Fig8.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
