"""Generate extension figures and exact tables from saved study output."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
(ROOT/'results/figures').mkdir(parents=True, exist_ok=True)
(ROOT/'results/tables').mkdir(parents=True, exist_ok=True)
ORDER = ['balanced', 'sparse', 'unbalanced', 'near_tie', 'boundary', 'zero_a', 'zero_both']
LABELS = ['Balanced', 'Sparse', 'Unbalanced', 'Weak signal', 'Near-zero A', 'Zero A', 'Zero A and B']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'font.family': 'DejaVu Sans'})


def main():
    r = pd.read_csv(ROOT/'results/boundary_v2/runs.csv')
    a = pd.read_csv(ROOT/'results/boundary_v2/fold_audit.csv')
    real = pd.read_csv(ROOT/'results/lexdec/runs.csv')
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
    yy = np.arange(len(ORDER))
    q = np.array([r[r.regime == k].speedup_total.quantile([.25, .5, .75]) for k in ORDER])
    ax[0].errorbar(q[:, 1], yy, xerr=np.array([q[:, 1]-q[:, 0], q[:, 2]-q[:, 1]]),
                  fmt='o', color='#087f8c', capsize=4, markersize=6)
    ax[0].axvline(1, color='.6', linestyle='--', linewidth=1)
    ax[0].set_yticks(yy, LABELS); ax[0].invert_yaxis()
    ax[0].set_xlabel('Total speedup\n(median and interquartile range)')
    ax[0].set_title('A. Repeated timings', loc='left', weight='bold', fontsize=10)
    ax[0].grid(axis='x', alpha=.15)
    ab = r.groupby('regime')[['interior_only_fallbacks', 'numerical_fallbacks']].sum().reindex(ORDER)
    ax[1].barh(yy-.16, ab.interior_only_fallbacks, height=.3, color='#d98548', label='Interior-only test')
    ax[1].barh(yy+.16, ab.numerical_fallbacks, height=.3, color='#087f8c', label='Active-face test')
    ax[1].set_yticks(yy, LABELS); ax[1].invert_yaxis(); ax[1].set_xlim(0, 760)
    ax[1].set_xlabel('Numerical fallbacks\n(out of 720 fits per setting)')
    ax[1].set_title('B. Matched boundary-test ablation', loc='left', weight='bold', fontsize=10)
    for i, v in enumerate(ab.interior_only_fallbacks):
        ax[1].text(v+8, i-.16, str(v), va='center', fontsize=9)
    ax[1].legend(loc='upper right', fontsize=9, frameon=False)
    for ext in ['pdf', 'png']:
        fig.savefig(ROOT/f'results/figures/boundary_extension.{ext}', bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    for axis, (_, row), title in zip(ax, real.iterrows(), ['New participant (21 folds)', 'New word (79 folds)']):
        common = row.common_seconds
        ref = min(row.lbfgs_seconds, row.newton_seconds)
        kernel = [ref, row.adaptive_seconds]
        axis.bar([0, 1], [common, common], color='#b9c4cc', width=.5, label='Shared preparation')
        axis.bar([0, 1], kernel, bottom=common, color='#087f8c', width=.5, label='Refit and score')
        total = common+np.array(kernel)
        for i, v in enumerate(total):
            axis.text(i, v+.025*max(total), f'{v:.3f} s', ha='center', fontsize=10)
        axis.set_ylim(0, max(total)*1.2)
        axis.set_xticks([0, 1], ['Full refit', 'Adaptive'])
        axis.set_title(title, loc='left', weight='bold', fontsize=10)
        axis.set_ylabel('Total elapsed time (seconds)')
        axis.legend(loc='upper right', fontsize=8, frameon=False)
    for ext in ['pdf', 'png']:
        fig.savefig(ROOT/f'results/figures/lexdec_runtime.{ext}', bbox_inches='tight')
    plt.close(fig)
    table = ['\\begin{tabular}{lrrrrrr}', '\\toprule',
             'Setting & Datasets & Face fits & Fallbacks & Speedup & Frozen errors & One-step errors \\\\', '\\midrule']
    for key, label in zip(ORDER, LABELS):
        x = r[r.regime == key]
        table.append(f'{label} & {len(x)} & {int(x.active_face_certificates.sum())} & '
                     f'{int(x.numerical_fallbacks.sum())} & {x.speedup_total.median():.2f} & '
                     f'{int((~x.frozen_agrees).sum())} & {int((~x.one_step_agrees).sum())} \\\\')
    table += ['\\bottomrule', '\\end{tabular}']
    (ROOT/'results/tables/boundary_results_table.tex').write_text('\n'.join(table))
    table = ['\\begin{tabular}{llrrr}', '\\toprule',
             'Target & Model & Full CV & Adaptive lower & Adaptive upper \\\\', '\\midrule']
    for _, x in real.iterrows():
        for m in range(3):
            target = ('New participant' if x.target == 'new_subject' else 'New word') if m == 0 else ''
            table.append(f'{target} & M{m+1} & {x[f"cv_{m}"]:.8f} & '
                         f'{x[f"lower_{m}"]:.8f} & {x[f"upper_{m}"]:.8f} \\\\')
    table += ['\\bottomrule', '\\end{tabular}']
    (ROOT/'results/tables/lexdec_results_table.tex').write_text('\n'.join(table))
    summary = {'datasets': len(r), 'selected_model_agreement': int(r.adaptive_agrees.sum()),
               'fits': len(a), 'local_enclosures': int(a.local_valid.sum()),
               'active_faces': int((a.certificate_kind == 'active_face').sum()),
               'double_active': int((a.active_count == 2).sum()),
               'fallbacks': int(a.numerical_fallback.sum()),
               'all_local_datasets': int((r.numerical_fallbacks == 0).sum()),
               'reference_loss_max_difference': float(r.reference_difference.max()),
               'max_reference_gradient': float(r.max_reference_gradient.max()),
               'max_training_start_objective_spread': float(a.training_audit_objective_spread.max()),
               'per_regime_speedup_median': r.groupby('regime').speedup_total.median().to_dict(),
               'real_data': real[['target', 'speedup_total', 'speedup_kernel', 'local_certificates', 'numerical_fallbacks']].to_dict(orient='records')}
    (ROOT/'results/extension_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
