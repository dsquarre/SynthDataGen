"""Regenerate the presentation figures from one pipeline run.

Usage:
    python scripts/make_deck_figures.py RUN_DIR OUT_DIR --ratio-label "1:1" [--compare-run OTHER_RUN --compare-label "11:1"]

Reads RUN_DIR/{evaluation_report,regression_report,augmented_cv_report}.xlsx and the
imputed real / synthetic data files of that run. Captions state only what the plotted numbers show.
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_SIZE = (12, 9)   # 1152 x 864 px at 96 dpi, same canvas as the original deck figures
WIDE_SIZE = (12, 6.2)  # regression / ratio / classification: matches their picture boxes on the slides
PCA_SIZE = (12, 6.93)  # PCA slides' picture box aspect
DPI = 96
GEN_ORDER = ['cart', 'gaussian', 'ctgan', 'copulagan', 'tvae']
GEN_LABEL = {'cart': 'Sequential CART', 'gaussian': 'Gaussian Copula', 'ctgan': 'CTGAN',
             'copulagan': 'CopulaGAN', 'tvae': 'TVAE'}
SYNTH_TO_GEN = {'cart': 'cart', 'gaussian copula': 'gaussian', 'ctgan': 'ctgan',
                'copulagan': 'copulagan', 'tvae': 'tvae'}
IMP_LABEL = {'iter': 'Iterative Imputation', 'knn': 'K-Nearest Neighbors Imputation',
             'mi': 'Missing Indicator Imputation'}
AUG = 'FoldLocal_TrainAugmented_TestReal'
BASE = 'Real_Only_TestReal'

sns.set_theme(style='whitegrid')


def caption(fig, number, bold, body, y=0.035):
    """Serif caption under the plot area, like the original figures."""
    text = textwrap.fill(f'Figure {number}: {bold}. {body}', 175)
    lines = text.split('\n')
    first, rest = lines[0], '\n'.join(lines[1:])
    cut = len(f'Figure {number}: {bold}.')
    fig.text(0.015, y, '', fontsize=1)  # anchor
    # Bold lead-in on the first line, regular for the remainder.
    fig.text(0.015, y + 0.0 if not rest else y + 0.022 * len(lines[1:]),
             first[:cut], family='Liberation Serif', weight='bold', fontsize=10.5, va='bottom')
    renderer = fig.canvas.get_renderer()
    lead = fig.texts[-1]
    bb = lead.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.004, bb.y0, first[cut:], family='Liberation Serif', fontsize=10.5, va='bottom')
    if rest:
        fig.text(0.015, bb.y0 - 0.004, rest, family='Liberation Serif', fontsize=10.5, va='top',
                 linespacing=1.25)


def synth_gen(name):
    return SYNTH_TO_GEN[name.rsplit('_', 1)[0]]


# ---------------------------------------------------------------- fidelity
def fidelity_figure(ev, imp, number, out):
    d = ev[ev.Imputation == imp].set_index('Generator').reindex(GEN_ORDER)
    labels = [GEN_LABEL[g] for g in GEN_ORDER]
    panels = [('cov_score', 'Covariance Similarity (Higher is Better)', 'Covariance Score', 'viridis', 'max'),
              ('wd_score', 'Wasserstein Distance (Lower is Better)', 'Distance', 'magma', 'min'),
              ('fd_score', 'Fréchet Distance (Lower is Better)', 'Distance', 'rocket', 'min'),
              ('mmd_score', 'Maximum Mean Discrepancy (Lower is Better)', 'MMD Score', 'crest', 'min')]
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, dpi=DPI)
    fig.suptitle(f'Comparative Validation Metrics ({IMP_LABEL[imp]})', fontsize=15, weight='bold', y=0.985)
    best = {}
    for ax, (col, title, ylab, pal, direction) in zip(axes.flat, panels):
        sns.barplot(x=labels, y=d[col].values, hue=labels, palette=pal, legend=False, ax=ax)
        ax.set_title(title, fontsize=11.5, weight='bold')
        ax.set_ylabel(ylab)
        ax.tick_params(axis='x', rotation=45)
        for t in ax.get_xticklabels():
            t.set_ha('right')
        idx = d[col].idxmax() if direction == 'max' else d[col].idxmin()
        tied = np.isclose(d[col].max(), d[col].min(), rtol=1e-3)
        best[col] = (None if tied else GEN_LABEL[idx], d.loc[idx, col])
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    def phrase(col, what, fmt):
        name, val = best[col]
        if name is None:
            return f'{what} is the same for all generators ({val:{fmt}})'
        return f'{what}: {name} ({val:{fmt}})'
    body = (phrase('cov_score', 'Best covariance similarity', '.3f') + '; '
            + phrase('wd_score', 'lowest Wasserstein distance', '.2f') + '; '
            + phrase('fd_score', 'lowest Fréchet distance', ',.0f') + '; '
            + phrase('mmd_score', 'lowest MMD', '.4f').replace('lowest MMD is', 'MMD is') + '. '
            'Each generator was gated with its own thresholds (see methodology), so these values are not a like-for-like comparison.')
    caption(fig, number, f'Comparative validation metrics of synthetic data generation models under {IMP_LABEL[imp]}', body)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- PCA
def _data_dir(run):
    """generated/ in the new layout, datasets/ in runs made before the split."""
    g = os.path.join(run, 'generated')
    return g if os.path.isdir(g) else os.path.join(run, 'datasets')


def pca_figure(run, imp, number, out):
    """Same method as plot.py: standardise pooled real + synthetic numeric columns, 2-component PCA."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    ddir = _data_dir(run)
    real = pd.read_excel(os.path.join(ddir, f'{imp}real_data.xlsx'), index_col=0)
    fig = plt.figure(figsize=PCA_SIZE, dpi=DPI)
    slots = [(0.05, 0.575), (0.37, 0.575), (0.69, 0.575), (0.21, 0.18), (0.53, 0.18)]
    for (x, y), g in zip(slots, GEN_ORDER):
        synth = pd.read_excel(os.path.join(ddir, f'{imp}{g}_data.xlsx'))
        cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c]) and c in synth.columns
                and pd.api.types.is_numeric_dtype(synth[c])]
        r_, s_ = real[cols].dropna(), synth[cols].dropna()
        z = StandardScaler().fit_transform(pd.concat([r_, s_]))
        pca = PCA(n_components=2).fit(z)
        pc = pca.transform(z)
        ax = fig.add_axes((x, y, 0.27, 0.3))
        ax.scatter(pc[len(r_):, 0], pc[len(r_):, 1], s=14, alpha=0.55, color='#ff7f0e', edgecolor='w', lw=0.3, label='Synthetic')
        ax.scatter(pc[:len(r_), 0], pc[:len(r_), 1], s=18, alpha=0.8, color='#1f77b4', edgecolor='w', lw=0.3, label='Real')
        ax.set_title(f'{GEN_LABEL[g]}  (PC1+PC2: {pca.explained_variance_ratio_.sum() * 100:.1f}%)', fontsize=10.5, weight='bold')
        ax.set_xlabel('PC 1', fontsize=9)
        ax.set_ylabel('PC 2', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle(f'PCA Projection: Real vs Synthetic ({IMP_LABEL[imp]})', fontsize=15, weight='bold', y=0.985)
    caption(fig, number, f'Principal Component Analysis (PCA) projections under {IMP_LABEL[imp]}',
            'Real (blue) and synthetic (orange) records, standardised and projected with a two-component PCA fitted on the pooled '
            'real and synthetic numeric columns, one panel per generator. The percentage is the variance explained by the two '
            'components; it depends on the pooled data and is not itself a fidelity score.')
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- regression
def paired_wins(folds, imp, reg, gen):
    f = folds[(folds.Imputation == imp) & (folds.Regressor == reg)].copy()
    f['gen'] = f.Synthetic_Model.map(synth_gen)
    f = f[f.gen == gen]
    p = f.pivot_table(index='Fold', columns='Evaluation', values='mae')
    return int((p[AUG] < p[BASE]).sum()), len(p)


def regression_figure(summ, folds, imp, reg, number, ratio_label, out):
    s = summ[(summ.Imputation == imp) & (summ.Regressor == reg)].copy()
    s['gen'] = s.Synthetic_Model.map(synth_gen)
    labels = [GEN_LABEL[g].replace('Sequential ', '') for g in GEN_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE, dpi=DPI)
    x = np.arange(len(GEN_ORDER))
    w = 0.38
    f = folds[(folds.Imputation == imp) & (folds.Regressor == reg)].copy()
    f['gen'] = f.Synthetic_Model.map(synth_gen)

    def iqr_err(evaluation, metric, means):
        q = f[f.Evaluation == evaluation].groupby('gen')[metric].quantile([0.25, 0.75]).unstack().reindex(GEN_ORDER)
        lo = np.clip(means.values - q[0.25].values, 0, None)
        hi = np.clip(q[0.75].values - means.values, 0, None)
        return np.vstack([lo, hi])

    for ax, (metric, title, ylab, color) in zip(axes, [('mae', 'Mean Absolute Error (Lower is Better)', 'MAE (kN)', '#2ca02c'),
                                                        ('r2', 'R² Score (Higher is Better)', 'R² Score', '#1f77b4')]):
        aug = s[s.Evaluation == AUG].set_index('gen').reindex(GEN_ORDER)
        base = s[s.Evaluation == BASE].set_index('gen').reindex(GEN_ORDER)
        ax.bar(x - w / 2, aug[f'{metric}_mean'], w, yerr=iqr_err(AUG, metric, aug[f'{metric}_mean']), color='#dddddd',
               ecolor='#777777', capsize=3, label=f'{ratio_label} Augmented')
        ax.bar(x + w / 2, base[f'{metric}_mean'], w, yerr=iqr_err(BASE, metric, base[f'{metric}_mean']), color=color,
               ecolor='#555555', capsize=3, label='Baseline (Real Only)')
        if metric == 'r2':
            ax.set_ylim(top=1.0)
        ax.set_xticks(x, labels, rotation=30, ha='right')
        ax.set_title(title, fontsize=11.5, weight='bold', pad=30)
        ax.set_ylabel(ylab)
        ax.grid(axis='x', visible=False)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle(f'Downstream Predictive Performance - {reg} ({IMP_LABEL[imp]})', fontsize=15, weight='bold', y=0.975)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.78, bottom=0.33, wspace=0.2)
    base_mae = s[s.Evaluation == BASE].mae_mean.iloc[0]
    aug_mae = s[s.Evaluation == AUG].set_index('gen').mae_mean
    better = [GEN_LABEL[g] for g in GEN_ORDER if aug_mae[g] < base_mae]
    wins = {g: paired_wins(folds, imp, reg, g) for g in GEN_ORDER}
    win_txt = ', '.join(f'{GEN_LABEL[g].replace("Sequential ", "")} {w}/{n}' for g, (w, n) in wins.items())
    body = (f'Bars show mean over repeated 5-fold CV folds; error bars span the interquartile range of the fold scores (the mean can lie outside it when a few folds are very poor). '
            f'Baseline MAE {base_mae:.2f} kN. Augmentation gave a lower mean MAE for '
            f'{len(better)} of 5 generators{(" (" + ", ".join(better) + ")") if better else ""}. '
            f'Folds where augmentation lowered MAE (paired, same test fold): {win_txt}.')
    caption(fig, number, f'Downstream predictive performance of the {reg} regressor under {IMP_LABEL[imp]}', body, y=0.03)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- ratio comparison
def ratio_figure(runs, number, out):
    """Mean MAE change (augmented - real only), averaged over imputations, per generator and ratio."""
    rows = []
    for label, run in runs:
        s = pd.read_excel(os.path.join(run, 'regression_report.xlsx'), 'summary')
        s['gen'] = s.Synthetic_Model.map(synth_gen)
        p = s.pivot_table(index=['Imputation', 'Regressor', 'gen'], columns='Evaluation', values='mae_mean').reset_index()
        p['delta'] = p[AUG] - p[BASE]
        g = p.groupby(['Regressor', 'gen']).delta.mean().reset_index()
        g['Ratio'] = label
        rows.append(g)
    d = pd.concat(rows)
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE, dpi=DPI, sharey=True)
    labels = [GEN_LABEL[g].replace('Sequential ', '') for g in GEN_ORDER]
    x = np.arange(len(GEN_ORDER))
    w = 0.38
    colors = ['#4c72b0', '#dd8452']
    for ax, reg in zip(axes, ['XGBoost', 'RandomForest']):
        for i, (label, _) in enumerate(runs):
            v = d[(d.Regressor == reg) & (d.Ratio == label)].set_index('gen').reindex(GEN_ORDER).delta
            bars = ax.bar(x + (i - 0.5) * w, v, w, color=colors[i], label=f'{label} augmentation')
            ax.bar_label(bars, fmt='%+.2f', fontsize=8, padding=2)
        ax.axhline(0, color='#333333', lw=1)
        ax.set_xticks(x, labels, rotation=45, ha='right')
        ax.set_title(reg, fontsize=11.5, weight='bold', pad=30)
        ax.set_ylabel('Δ MAE vs Real Only (kN), lower is better')
        ax.grid(axis='x', visible=False)
        ax.margins(y=0.15)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle('Effect of Augmentation Ratio on Prediction Error', fontsize=15, weight='bold', y=0.97)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.3, wspace=0.12)
    caption(fig, number, 'Change in mean absolute error from adding synthetic data, by augmentation ratio',
            'Bars show augmented-minus-baseline MAE, averaged over the three imputation branches; below zero means augmentation '
            f'reduced error. {runs[0][0]} adds as many synthetic rows as real training rows; {runs[1][0]} adds a fixed 500 synthetic rows per fold.',
            y=0.045)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- classification
def classification_figure(run, number, ratio_label, out):
    c = pd.read_excel(os.path.join(run, 'augmented_cv_report.xlsx'), 'summary')
    c['gen'] = c.Synthetic_Model.map(synth_gen)
    g = c.groupby(['Classifier', 'Evaluation', 'gen']).balanced_accuracy_mean.mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE, dpi=DPI, sharey=True)
    labels = [GEN_LABEL[x].replace('Sequential ', '') for x in GEN_ORDER]
    x = np.arange(len(GEN_ORDER))
    w = 0.38
    for ax, clf in zip(axes, ['XGBoost', 'RandomForest']):
        aug = g[(g.Classifier == clf) & (g.Evaluation == AUG)].set_index('gen').reindex(GEN_ORDER).balanced_accuracy_mean
        base = g[(g.Classifier == clf) & (g.Evaluation == BASE)].set_index('gen').reindex(GEN_ORDER).balanced_accuracy_mean
        ax.bar(x - w / 2, aug, w, color='#dddddd', label=f'{ratio_label} Augmented (+ROS on real)')
        ax.bar(x + w / 2, base, w, color='#8172b3', label='Baseline (Real Only, +ROS)')
        ax.set_xticks(x, labels, rotation=45, ha='right')
        ax.set_title(clf, fontsize=11.5, weight='bold', pad=48)
        ax.set_ylabel('Balanced accuracy')
        ax.set_ylim(0, 1)
        ax.grid(axis='x', visible=False)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=9.5)
    fig.suptitle('Section Classification (Circular vs Square): Augmented vs Real Only', fontsize=15, weight='bold', y=0.97)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.3, wspace=0.12)
    caption(fig, number, 'Balanced accuracy for classifying cross-section shape',
            'Mean over repeated stratified 5-fold CV and the three imputation branches. Random oversampling is applied to the real '
            'training fold only; synthetic rows are generated class-balanced, with no duplicated rows.', y=0.045)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--ratio-label', default='1:1')
    ap.add_argument('--compare-run')
    ap.add_argument('--compare-label', default='11:1')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = pd.read_excel(os.path.join(a.run, 'evaluation_report.xlsx'))
    summ = pd.read_excel(os.path.join(a.run, 'regression_report.xlsx'), 'summary')
    folds = pd.read_excel(os.path.join(a.run, 'regression_report.xlsx'), 'fold_scores')
    n = 1
    for imp in ['iter', 'knn', 'mi']:
        fidelity_figure(ev, imp, n, os.path.join(a.out, f'fig{n:02d}_fidelity_{imp}.png')); n += 1
    for imp in ['iter', 'knn', 'mi']:
        pca_figure(a.run, imp, n, os.path.join(a.out, f'fig{n:02d}_pca_{imp}.png')); n += 1
    for reg in ['XGBoost', 'RandomForest']:
        for imp in ['iter', 'knn', 'mi']:
            regression_figure(summ, folds, imp, reg, n, a.ratio_label, os.path.join(a.out, f'fig{n:02d}_reg_{reg}_{imp}.png')); n += 1
    if a.compare_run:
        ratio_figure([(a.ratio_label, a.run), (a.compare_label, a.compare_run)], n, os.path.join(a.out, f'fig{n:02d}_ratio.png')); n += 1
    classification_figure(a.run, n, a.ratio_label, os.path.join(a.out, f'fig{n:02d}_classification.png'))
    print('wrote', n, 'figures to', a.out)


if __name__ == '__main__':
    main()
