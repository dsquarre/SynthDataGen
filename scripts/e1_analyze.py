"""Apply the pre-registered E1 decision rule (PLAN.md, 2026-11-14) to e1_training_size.py output.

    python scripts/e1_analyze.py RESULTS.csv [RESULTS2.csv ...] --out SUMMARY.csv

Per dataset x size x generator x regressor:
  d = (MAE_augmented - MAE_real) / MAE_real per draw (paired on the same draw and test set)
  Benefit: median d <= -0.05 and one-sided Wilcoxon (augmented < real) Holm-adjusted p < 0.05
  Harm:    median d >= +0.05 and one-sided Wilcoxon (augmented > real) Holm-adjusted p < 0.05
  Holm family: all sizes x generators x regressors within a dataset (each direction adjusted separately).
Main claim supported only if the same generator shows Benefit at n <= 50 on every dataset for >= 1 regressor.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

THRESH = 0.05
ALPHA = 0.05


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def analyze(res):
    base = res[res.generator == 'RealOnly'][['dataset', 'draw', 'n', 'regressor', 'mae']].rename(columns={'mae': 'mae_real'})
    aug = res[res.generator != 'RealOnly'].merge(base, on=['dataset', 'draw', 'n', 'regressor'])
    aug['d'] = (aug.mae - aug.mae_real) / aug.mae_real
    rows = []
    for (ds, n, gen, reg), g in aug.groupby(['dataset', 'n', 'generator', 'regressor']):
        ok = g[~g.failed & g.d.notna()]
        diff = ok.mae - ok.mae_real
        if len(ok) >= 2 and np.any(diff != 0):
            p_less = wilcoxon(ok.mae, ok.mae_real, alternative='less').pvalue
            p_greater = wilcoxon(ok.mae, ok.mae_real, alternative='greater').pvalue
        else:
            p_less = p_greater = 1.0
        rows.append(dict(dataset=ds, n=n, generator=gen, regressor=reg, draws_used=len(ok),
                         failures=int(g.failed.sum()), median_d=ok.d.median(), mean_d=ok.d.mean(),
                         aug_better_draws=int((diff < 0).sum()), mae_real_median=ok.mae_real.median(),
                         mae_aug_median=ok.mae.median(), p_benefit=p_less, p_harm=p_greater))
    s = pd.DataFrame(rows)
    s['p_benefit_holm'] = s.groupby('dataset').p_benefit.transform(holm)
    s['p_harm_holm'] = s.groupby('dataset').p_harm.transform(holm)
    s['verdict'] = np.where((s.median_d <= -THRESH) & (s.p_benefit_holm < ALPHA), 'Benefit',
                            np.where((s.median_d >= THRESH) & (s.p_harm_holm < ALPHA), 'Harm', 'No detectable effect'))
    return s


def main_claim(s):
    datasets = sorted(s.dataset.unique())
    small = s[(s.n <= 50) & (s.verdict == 'Benefit')]
    support = []
    for (gen, reg), g in small.groupby(['generator', 'regressor']):
        if set(g.dataset) == set(datasets):
            support.append(f'{gen} + {reg}')
    return support


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = pd.concat([pd.read_csv(f) for f in a.results], ignore_index=True)
    s = analyze(res)
    s.to_csv(a.out, index=False)
    pd.set_option('display.width', 200)
    print(s[['dataset', 'n', 'generator', 'regressor', 'draws_used', 'median_d', 'aug_better_draws',
             'p_benefit_holm', 'p_harm_holm', 'verdict']].round(4).to_string(index=False))
    print('\nVerdict counts:', s.verdict.value_counts().to_dict())
    sup = main_claim(s)
    print('Main claim ("augmentation helps in the small-data regime"):',
          'SUPPORTED by ' + ', '.join(sup) if sup else 'NOT supported (no generator shows Benefit at n <= 50 on all datasets)')
