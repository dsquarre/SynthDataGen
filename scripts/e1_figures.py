"""Figure for E1 (training-size experiment): relative MAE change d vs training size, per dataset and regressor.

    python scripts/e1_figures.py SUMMARY.csv OUT.png [--exploratory DATASET_NAME ...]

SUMMARY.csv is the output of e1_analyze.py (all datasets in one file). Lines show the median d over draws for
each generator; filled markers are cells with a pre-registered Benefit/Harm verdict, hollow markers are
"no detectable effect". The shaded band is the +/-5% practical-effect threshold.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

FONT = 'Liberation Serif'
DPI = 200
WIDE_SIZE = (12, 6.2)
GEN_LABEL = {'CART': 'Sequential CART', 'GaussianCopula': 'Gaussian Copula', 'CTGAN': 'CTGAN',
             'CopulaGAN': 'CopulaGAN', 'TVAE': 'TVAE'}
GEN_COLOR = {'CART': '#3b4a8a', 'GaussianCopula': '#2a9d8f', 'CTGAN': '#e76f51', 'CopulaGAN': '#9b5de5', 'TVAE': '#e9b949'}
DS_LABEL = {'frp_confined_concrete': 'FRP-confined concrete (832)',
            'cfst_rectangular_hss': 'Rectangular HSS CFST (255)',
            'cfst_circular_hss': 'Circular HSS CFST (93), exploratory'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('summary')
    ap.add_argument('out')
    ap.add_argument('--number', default='14')
    a = ap.parse_args()
    s = pd.read_csv(a.summary)
    datasets = [d for d in DS_LABEL if d in set(s.dataset)]
    regs = ['XGBoost', 'RandomForest']
    fig, axes = plt.subplots(len(regs), len(datasets), figsize=WIDE_SIZE, sharey=True, squeeze=False)
    for j, ds in enumerate(datasets):
        for i, reg in enumerate(regs):
            ax = axes[i][j]
            ax.axhspan(-5, 5, color='0.9', zorder=0)
            ax.axhline(0, color='0.3', lw=0.8, zorder=1)
            for gen in GEN_LABEL:
                g = s[(s.dataset == ds) & (s.regressor == reg) & (s.generator == gen)].sort_values('n')
                if g.empty:
                    continue
                y = 100 * g.median_d
                ax.plot(g.n, y, color=GEN_COLOR[gen], lw=1.6, label=GEN_LABEL[gen], zorder=2)
                sig = g.verdict != 'No detectable effect'
                ax.scatter(g.n[sig], y[sig], color=GEN_COLOR[gen], s=34, zorder=3)
                ax.scatter(g.n[~sig], y[~sig], facecolor='white', edgecolor=GEN_COLOR[gen], s=34, zorder=3)
            ax.set_xscale('log')
            ns = sorted(s[s.dataset == ds].n.unique())
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            ax.minorticks_off()
            ax.grid(axis='y', color='0.85', lw=0.6)
            if i == 0:
                ax.set_title(DS_LABEL[ds], fontsize=11, weight='bold')
            if i == len(regs) - 1:
                ax.set_xlabel('Real training rows (n)')
            if j == 0:
                ax.set_ylabel(f'{reg}\nmedian MAE change (%)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    cap = (f'Figure {a.number}: Change in test MAE from 1:1 synthetic augmentation, by training size.',
           ' Median over 20 random train/test draws of (MAE augmented − MAE real-only) / MAE real-only; below zero '
           'means augmentation reduced error. Filled markers: Benefit or Harm under the pre-registered rule '
           '(|median| ≥ 5% and Holm-adjusted one-sided Wilcoxon p < 0.05); hollow: no detectable effect. '
           'Grey band: ±5%. The circular CFST panel was added after registration and is exploratory.')
    fig.text(0.015, 0.09, cap[0], family=FONT, weight='bold', size=10.5, va='top', wrap=True)
    fig.text(0.015, 0.06, cap[1].strip(), family=FONT, size=10, va='top', wrap=True)
    fig.savefig(a.out, dpi=DPI)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
