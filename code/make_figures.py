"""Generate figures and numeric tables only from saved experiment outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter


root=Path(__file__).resolve().parents[1]
(root/'results/figures').mkdir(parents=True, exist_ok=True)
(root/'results/tables').mkdir(parents=True, exist_ok=True)
r=pd.read_csv(root/'results/compressed/runs.csv')
f=pd.read_csv(root/'results/compressed/fold_audit.csv')
order=['balanced','sparse','unbalanced','near_tie','boundary']
labels=['Balanced','Sparse','Unbalanced','Weak signal','Near boundary']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':150,'savefig.dpi':200})
fig,axs=plt.subplots(1,2,figsize=(10,3.65),gridspec_kw={'width_ratios':[1.2,1]})
vals=[r.loc[r.regime==x,'speedup_total'].values for x in order]
bp=axs[0].boxplot(vals,tick_labels=labels,patch_artist=True,widths=.55,
                  medianprops={'color':'#082f49','linewidth':1.5},showfliers=True)
for box in bp['boxes']:box.set_facecolor('#b6dae7')
axs[0].axhline(1,color='#b45309',linestyle='--',linewidth=1)
axs[0].set_ylabel('Full refit time / adaptive time')
axs[0].set_title('(a) End-to-end speedup, 20 datasets per setting')
axs[0].tick_params(axis='x',labelrotation=25)
rates=[100*f.loc[f.regime==x,'local_valid'].mean() for x in order]
axs[1].barh(labels,rates,color=['#0e7490']*4+['#b45309'])
axs[1].invert_yaxis();axs[1].set_xlim(0,112)
axs[1].set_xlabel('Fold fits with a local analytic enclosure (%)')
axs[1].set_title('(b) Local validation versus numerical fallback')
for i,v in enumerate(rates):axs[1].text(v+1,i,f'{v:.1f}%',va='center',fontsize=9)
fig.tight_layout(w_pad=2)
fig.savefig(root/'results/figures/pilot_summary.pdf',bbox_inches='tight')
fig.savefig(root/'results/figures/pilot_summary.png',bbox_inches='tight')
plt.close(fig)

# A pre-specified dataset index found in the audit: one Newton step reverses
# the winner. Display actual reference scores and final certified intervals.
row=r.loc[r.seed==20261031].iloc[0]
fig,ax=plt.subplots(figsize=(6.8,2.7))
models=np.array([1,2])
for i,m in enumerate(models):
    lo,hi=row[f'lower_{m}'],row[f'upper_{m}'];mid=(lo+hi)/2
    ax.errorbar(mid,i,xerr=np.array([[mid-lo],[hi-mid]]),fmt='o',color='#0e7490',capsize=5,label='Final local enclosure' if i==0 else None)
    ax.plot(row[f'cv_{m}'],i,'|',color='#b45309',markersize=15,markeredgewidth=2,label='Full-refit CV' if i==0 else None)
ax.set_yticks([0,1],['Model 2','Model 3'])
ax.set_xlabel('Mean held-out squared error (equal group weights)')
ax.set_title('A close comparison resolved by local error bounds')
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.set_ylim(-.55,1.7);ax.legend(loc='upper left',fontsize=8,frameon=False)
fig.tight_layout();fig.savefig(root/'results/figures/close_comparison.pdf',bbox_inches='tight')
fig.savefig(root/'results/figures/close_comparison.png',bbox_inches='tight');plt.close(fig)

rows=[]
for regime,label in zip(order,labels):
    z=r[r.regime==regime];a=f[f.regime==regime]
    rows.append(f'{label} & {len(z)} & {int(z.adaptive_agrees.sum())}/{len(z)} & '
                f'{int(a.local_valid.sum())}/{len(a)} & {z.speedup_total.median():.2f} & '
                f'{int((~z.frozen_agrees).sum())} & {int((~z.one_step_agrees).sum())} \\\\')
table_head=r'''\begin{tabular}{lrrrrrr}
\toprule
Setting & Runs & Correct & Local folds & Speedup & Frozen & 1-step\\
 & & choices & / all folds & median & errors & errors\\
\midrule
'''
(root/'results/tables/results_table.tex').write_text(table_head+'\n'.join(rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n')
summary={'datasets':len(r),'fold_fits':len(f),'local_enclosures':int(f.local_valid.sum()),
         'fallbacks':int(f.numerical_fallback.sum()),'adaptive_correct':int(r.adaptive_agrees.sum()),
         'frozen_incorrect':int((~r.frozen_agrees).sum()),'one_step_incorrect':int((~r.one_step_agrees).sum()),
         'interval_failures':int((~f.contains_reference).sum()),
         'root_mismatches':int(r.root_mismatches.sum()),
         'all_local_datasets':int((r.numerical_fallbacks==0).sum()),
         'median_speedup_nonboundary':float(r[r.regime!='boundary'].speedup_total.median()),
         'median_speedup_boundary':float(r[r.regime=='boundary'].speedup_total.median())}
(root/'results/summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
