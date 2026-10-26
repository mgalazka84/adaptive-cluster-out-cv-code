"""Summaries and numerical-study figures from the default-rule benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from accuracy_io import read_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results'/'accuracy_v2';OUT=BASE/'summary'
OUT.mkdir(exist_ok=True)
METHODS=['newton','lbfgs','practical','default_lbfgs','loose','certify_once','adaptive','norm_only','interior_only','frozen','one_step']
R=pd.concat([read_csv(BASE/s/'runs.csv') for s in ['original','scale','fresh','lexdec']],ignore_index=True)
F=pd.concat([read_csv(BASE/s/'fold_audit.csv') for s in ['original','scale','fresh','lexdec']],ignore_index=True)
MIC=pd.concat([read_csv(BASE/s/'microbenchmarks.csv').assign(suite=s) for s in ['original','scale','fresh','lexdec']],ignore_index=True)
rows=[]
for (suite,regime),d in R.groupby(['suite','regime'],sort=False):
 row=dict(suite=suite,regime=regime,datasets=len(d),n=int(d.n.median()),models=int(d.models.iloc[0]),folds=int(d.fold_count.sum()),
          faces=int(d.active_face_certificates.sum()),fallbacks=int(d.numerical_fallbacks.sum()),once_resolves=int(d.cert_once_resolves.sum()),
          once_valid=int(d.cert_once_valid_fits.sum()),gap_median=d.cv_gap.median(),se_proxy_median=d.paired_fold_se_proxy.median(),gap_se_median=d.gap_to_se_proxy.median())
 for c in ['strict_speedup_total','practical_speedup_total','default_speedup_total','interior_only_speedup_total','common_seconds','adaptive_seconds','adaptive_iterations','adaptive_checks']:
  row[c]=d[c].mean() if c.endswith(('iterations','checks')) else d[c].median()
 for method in METHODS:row[method+'_errors']=int((~d[method+'_agrees']).sum())
 rows.append(row)
pd.DataFrame(rows).to_csv(OUT/'regime_summary.csv',index=False)
rows=[]
for suite,d in R.groupby('suite',sort=False):
 for method in METHODS:
  rows.append(dict(suite=suite,method=method,kernel_median_seconds=d[method+'_seconds'].median(),mean_iterations=np.average(d[method+'_iterations'],weights=d.fold_count),
                   mean_checks=np.average(d[method+'_checks'],weights=d.fold_count),disagreements=int((~d[method+'_agrees']).sum()),
                   total_time_median=(d.common_seconds+d[method+'_seconds']).median()))
pd.DataFrame(rows).to_csv(OUT/'method_summary.csv',index=False)
rows=[]
for (suite,regime),d in F[F.local_valid].groupby(['suite','regime'],sort=False):
 for active,z in d.groupby('active_count'):
  row=dict(suite=suite,regime=regime,active=int(active),fits=len(z),parameter_censored=int(z.parameter_ratio.isna().sum()),loss_censored=int(z.loss_ratio.isna().sum()))
  for c in ['parameter_ratio','loss_ratio','Lr','a','f2','b0','b_r','c','c0','active_margin_fraction','tube_epsilon']:
   for q in [.05,.5,.95]:row[c+'_'+str(q)]=z[c].quantile(q)
  row['quadratic_L_fraction']= (z.quadratic_L/(z.trace_L+z.quadratic_L)).median()
  rows.append(row)
pd.DataFrame(rows).to_csv(OUT/'tightness.csv',index=False)
valid=F[F.local_valid].copy()
distance=valid[['reference_theta_1','reference_theta_2']].min(axis=1)/valid.reference_theta_0
valid['distance_bin']=pd.cut(distance,[-1,0,.01,.1,float('inf')],labels=['zero','(0,.01]','(.01,.1]','>.1'])
valid.groupby('distance_bin',observed=True).agg(fits=('seed','size'),parameter_ratio=('parameter_ratio','median'),loss_ratio=('loss_ratio','median')).to_csv(OUT/'boundary_distance.csv')
errors=[]
for _,r in R.iterrows():
 d=F[(F.suite==r.suite)&(F.seed==r.seed)&(F.regime==r.regime)]
 for method in ['frozen','one_step','loose','practical','default_lbfgs']:
  if not r[method+'_agrees']:
   means=d.groupby('model')[method+'_loss'].mean();chosen=int(means.idxmin());best=int(r.reference_winner)
   pairing=d.pivot(index='group_index',columns='model',values='reference_loss')
   delta=pairing[chosen]-pairing[best]
   errors.append(dict(suite=r.suite,regime=r.regime,seed=int(r.seed),method=method,chosen=chosen,reference=best,
           reference_pair_gap=delta.mean(),approximate_pair_gap=means[chosen]-means[best],se_proxy=delta.std(ddof=1)/np.sqrt(len(delta)),
           adaptive_low_best=r[f'lower_{best}'],adaptive_high_best=r[f'upper_{best}'],adaptive_low_chosen=r[f'lower_{chosen}'],adaptive_high_chosen=r[f'upper_{chosen}']))
pd.DataFrame(errors).to_csv(OUT/'selection_disagreements.csv',index=False)
# Independent software fits are reused; the intervals and reference values are current.
e=read_csv(ROOT/'results'/'lexdec'/'statsmodels_audit.csv').rename(columns={'target':'regime','reference_loss':'external_old_reference'})
e=e.merge(F[F.suite=='lexdec'],on=['regime','model','group_index'],validate='one_to_one')
e['endpoint_fraction']=np.minimum(e.external_loss-e.lower,e.upper-e.external_loss)/(e.upper-e.lower)
e['inside_current_interval']=(e.external_loss>=e.lower)&(e.external_loss<=e.upper)
e.to_csv(OUT/'external_endpoint_audit.csv',index=False)
# Difference scales for each real-data candidate relative to M3.
rows=[]
for regime,d in F[F.suite=='lexdec'].groupby('regime'):
 z=d.pivot(index='group_index',columns='model',values='reference_loss')
 for m in z.columns:
  v=z[m]-z[2]
  rows.append(dict(regime=regime,model=m,mean_difference=v.mean(),fold_sd=v.std(ddof=1),se_proxy=v.std(ddof=1)/np.sqrt(len(v)),groups=len(v)))
pd.DataFrame(rows).to_csv(OUT/'lexdec_differences.csv',index=False)
# Concise global audit.
checks=dict(runs=len(R),folds=len(F),selection_failures=int((~R.adaptive_agrees).sum()),containment_failures=int(R.containment_failures.sum()),
 root_mismatches=int(R.root_mismatches.sum()),training_start_selection_failures=int((~R.training_start_agrees).sum()),
 training_start_active_mismatches=int(R.training_start_active_mismatches.sum()),max_projected_gradient=F.reference_gradient.max(),
 max_strict_loss_discrepancy=R.independent_ref_max_loss_difference.max(),min_contraction_margin=1-F.Lr.max(),
 parameter_ratio_quantiles=F.parameter_ratio.quantile([.05,.5,.95]).to_dict(),loss_ratio_quantiles=F.loss_ratio.quantile([.05,.5,.95]).to_dict(),
 parameter_censored=int(F[F.local_valid].parameter_ratio.isna().sum()),loss_censored=int(F[F.local_valid].loss_ratio.isna().sum()),
 active_fraction_quantiles=F.active_margin_fraction.quantile([0,.05,.5,.95]).to_dict(),
 external_all_inside=bool(e.inside_current_interval.all()),external_max_loss_error=float(abs(e.external_loss-e.reference_loss).max()))
(OUT/'audit_summary.json').write_text(json.dumps(checks,indent=2))
MIC['certificate_step_ratio']=MIC.certificate_seconds/MIC.newton_seconds
MIC.groupby('suite')[['newton_seconds','certificate_seconds','certificate_step_ratio']].median().to_csv(OUT/'micro_summary.csv')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.titleweight':'bold'})
colors=['#176B87','#7E57A0','#D88327','#37846C']
# Figure 5: cost and accuracy, no structurally forced fallback histogram.
fig,ax=plt.subplots(1,3,figsize=(9,3.6),layout='constrained')
labels=['Dev.\n(n≈240)','Scale\n(n=960)','Fresh\n(n=240)','Fresh\n(n=1440)']
sets=[R[R.suite=='original'],R[R.suite=='scale'],R[(R.suite=='fresh')&(R.regime!='many_levels')],R[(R.suite=='fresh')&(R.regime=='many_levels')]]
for j,col in enumerate(['strict_speedup_total','practical_speedup_total','default_speedup_total']):
 ax[0].plot(range(4),[d[col].median() for d in sets],marker='o',label=['Strict refit','Practical refit','Default tolerances'][j],color=colors[j])
ax[0].axhline(1,color='gray',ls=':');ax[0].set_xticks(range(4),labels,fontsize=9);ax[0].set_ylabel('Refit / adaptive total time');ax[0].set_title('A  Refit tolerance',loc='left');ax[0].legend(fontsize=9)
for j,r in enumerate(['boundary','zero_a','zero_both']):
 d=R[(R.suite=='original')&(R.regime==r)]
 ax[1].plot([0,1],[d.interior_only_speedup_total.median(),d.strict_speedup_total.median()],marker='o',color=colors[j],label=['Near-zero A','Zero A','Zero A and B'][j])
ax[1].axhline(1,color='gray',ls=':');ax[1].set_xticks([0,1],['Interior test','With face test']);ax[1].set_ylabel('Strict refit / total time');ax[1].set_title('B  Boundary ablation',loc='left');ax[1].legend(fontsize=9)
for j,d in enumerate(sets):
 ax[2].scatter(d.paired_fold_se_proxy,d.cv_gap,s=12,alpha=.65,label=labels[j].replace('\n',' '),color=colors[j])
lims=[1e-5,1];ax[2].plot(lims,lims,ls=':',color='gray');ax[2].set(xscale='log',yscale='log',xlabel='Paired fold SE proxy',ylabel='Top-two reference CV gap');ax[2].set_title('C  Gap and fold variation',loc='left');ax[2].legend(fontsize=9,loc='upper left')
fig.savefig(OUT/'Fig5.pdf',bbox_inches='tight');plt.close(fig)
# Figure 6: quantitative conservatism, censored ratios omitted explicitly in caption.
fig,ax=plt.subplots(1,3,figsize=(9,3.6),layout='constrained')
fsets=[F[F.suite=='original'],F[F.suite=='scale'],F[(F.suite=='fresh')&(F.regime!='many_levels')],F[(F.suite=='fresh')&(F.regime=='many_levels')],F[F.suite=='lexdec']]
lab=['Dev.\n240','Scale\n960','Fresh\n240','Fresh\n1440','lexdec\n1659']
vals=[d.parameter_ratio.dropna().to_numpy() for d in fsets]
ax[0].boxplot(vals,showfliers=False,whis=(5,95),widths=.55,patch_artist=True,boxprops={'facecolor':'#CDE4EB'},medianprops={'color':'#174E63'})
ax[0].set_xticks(range(1,6),lab,fontsize=10);ax[0].axhline(1,color='gray',ls=':');ax[0].set_ylabel('Radius / observed error');ax[0].set_title('A  Parameter enclosure',loc='left')
for offset,col,color,label in [(-.17,'norm_loss_half_width','#E6C9A4','Norm'),(.17,'loss_half_width','#CDE4EB','Directional')]:
 vals=[(d.loc[d.loss_ratio.notna(),col]/d.loc[d.loss_ratio.notna(),'loss_error']).to_numpy() for d in fsets]
 ax[1].boxplot(vals,positions=np.arange(1,6)+offset,showfliers=False,whis=(5,95),widths=.28,patch_artist=True,boxprops={'facecolor':color},medianprops={'color':'#174E63'})
 ax[1].plot([],[],lw=6,color=color,label=label)
ax[1].set_xticks(range(1,6),lab,fontsize=10);ax[1].set_yscale('log');ax[1].axhline(1,color='gray',ls=':');ax[1].set_ylabel('Half-width / observed error');ax[1].set_title('B  Paired loss bounds',loc='left');fig.legend(*ax[1].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=9)
for active,color in [(1,colors[0]),(2,colors[2])]:
 z=np.sort(F[F.active_count==active].active_margin_fraction.dropna());ax[2].plot(z,np.arange(1,len(z)+1)/len(z),label=f'{active} active variance'+('s' if active>1 else ''),color=color)
ax[2].set(xlabel='Active lower bound / derivative',ylabel='Empirical cumulative fraction',xlim=(0,1));ax[2].set_title('C  Boundary margins',loc='left');ax[2].legend(fontsize=10)
fig.savefig(OUT/'Fig6.pdf',bbox_inches='tight');plt.close(fig)
# Figure 7: real-data runtime and paired differences.
fig,ax=plt.subplots(1,3,figsize=(9,3.6),layout='constrained')
for j,regime in enumerate(['new_subject','new_word']):
 d=R[(R.suite=='lexdec')&(R.regime==regime)].iloc[0];methods=['newton','practical','default_lbfgs','certify_once','adaptive'];yy=np.arange(len(methods));kern=[d[m+'_seconds'] for m in methods]
 ax[j].barh(yy,[d.common_seconds]*len(methods),color='#BFCBD2',label='Shared setup');ax[j].barh(yy,kern,left=d.common_seconds,color=colors[0],label='Fit and score');ax[j].set_yticks(yy,['Strict Newton','Practical','Default tol.','Certify once','Adaptive'],fontsize=10);ax[j].invert_yaxis();ax[j].set_xlabel('Seconds');ax[j].set_title(['A  New participant','B  New word'][j],loc='left')
fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=9)
for j,regime in enumerate(['new_subject','new_word']):
 d=F[(F.suite=='lexdec')&(F.regime==regime)];z=d.pivot(index='group_index',columns='model',values='reference_loss');delta=z[1]-z[2]
 ax[2].scatter(np.full(len(delta),j)+np.linspace(-.12,.12,len(delta)),delta,s=10,alpha=.65,color=colors[j]);ax[2].scatter([j],[delta.mean()],marker='D',s=40,color='black')
ax[2].axhline(0,color='gray',ls=':');ax[2].set_xticks([0,1],['Participant','Word']);ax[2].set_ylabel('Fold loss difference: M2 − M3');ax[2].set_title('C  Paired differences',loc='left')
fig.savefig(OUT/'Fig7.pdf',bbox_inches='tight');plt.close(fig)
print(json.dumps(checks,indent=2))
