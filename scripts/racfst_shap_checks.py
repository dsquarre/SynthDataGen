"""Descriptive checks that qualify the RACFST SHAP results (exploratory, not pre-registered).

    OMP_NUM_THREADS=6 nice -n 10 python scripts/racfst_shap_checks.py --out runs/2026-11-23_SHAP

1. Series table: what differs between the 7 series. 2. Within-series associations of Pu, which remove differences
between series. 3. Whether the engineered ratios change real-only XGBoost error (same folds, with and without them).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from metadata_manager import generate_metadata  # noqa: E402
from metrics import _impute_fold_train_test, _prepare_regression_features, apply_reconstruction_rules  # noqa: E402

SEED = 2026


def cv(df, metadata, rules, target, branch, splits, use_derived):
    out = []
    for tr, te in splits:
        a, b = _impute_fold_train_test(df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True), metadata,
                                       branch, target, SEED)
        if use_derived:
            a, b = apply_reconstruction_rules(a, rules), apply_reconstruction_rules(b, rules)
        Xa, Xb = _prepare_regression_features(a.drop(columns=[target]), b.drop(columns=[target]))
        m = xgb.XGBRegressor(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                             objective='reg:squarederror', random_state=SEED, n_jobs=6).fit(Xa, a[target].astype(float))
        out.append(mean_absolute_error(b[target].astype(float), m.predict(Xb)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(PROJECT, 'configs', 'racfst.json'), encoding='utf-8'))
    target = cfg['regression_target']
    raw = pd.read_excel(os.path.join(PROJECT, cfg['data']))
    df = raw.drop(columns=[cfg['id_column']])
    series = df.groupby(['Section', 'D_mm', 't_mm']).ngroup().values
    d = df.assign(series=series, D_t=df.D_mm / df.t_mm, L_D=df.L2_mm / (2 * df.D_mm))
    d['steel_area_mm2'] = np.where(d.Section == 'Circular', np.pi * d.t_mm * (d.D_mm - d.t_mm),
                                   4 * d.t_mm * (d.D_mm - d.t_mm))
    tab = d.groupby('series').agg(Section=('Section', 'first'), D=('D_mm', 'first'), t=('t_mm', 'first'),
                                  L2_min=('L2_mm', 'min'), L2_max=('L2_mm', 'max'), D_t=('D_t', 'first'), tests=('n', 'size'),
                                  fy_min=('fy_MPa', 'min'), fy_max=('fy_MPa', 'max'), n_min=('n', 'min'), n_max=('n', 'max'),
                                  r_min=('r_pct', 'min'), r_max=('r_pct', 'max'), fcu_min=('fcu_MPa', 'min'),
                                  fcu_max=('fcu_MPa', 'max'), As=('steel_area_mm2', 'first'), Pu_mean=(target, 'mean'),
                                  Pu_min=(target, 'min'), Pu_max=(target, 'max')).round(2)
    tab.to_csv(os.path.join(a.out, 'series_table.csv'))
    print(tab.to_string())
    print('\nSeries-level Spearman with mean Pu (7 points):')
    for c in ('t', 'D', 'D_t', 'As'):
        r = spearmanr(tab[c], tab.Pu_mean)
        print(f'  {c}: rho={r.statistic:.2f} p={r.pvalue:.3f}')

    rows = []
    for c in ('n', 'r_pct', 'fcu_MPa', 'fy_MPa'):
        x = d[c] - d.groupby('series')[c].transform('mean')
        y = d[target] - d.groupby('series')[target].transform('mean')
        keep = x.abs() > 1e-9
        r = spearmanr(x[keep], y[keep])
        each = []
        for _, g in d.groupby('series'):
            if g[c].nunique() >= 3:
                each.append(spearmanr(g[c], g[target]).statistic)
        rows.append({'feature': c, 'within_rho': r.statistic, 'p_uncorrected': r.pvalue, 'rows_varying': int(keep.sum()),
                     'series_with_3plus_levels': len(each), 'series_positive': int(np.sum(np.array(each) > 0))})
    within = pd.DataFrame(rows)
    within.to_csv(os.path.join(a.out, 'within_series_associations.csv'), index=False)
    print('\nWithin-series association with Pu\n', within.round(3).to_string())
    print('\nn vs Pu by n band, relative to the series mean Pu:')
    rel = d[target] / d.groupby('series')[target].transform('mean')
    print(rel.groupby(pd.cut(d.n, [0, .2, .4, .6, .85])).agg(['count', 'mean']).round(3).to_string())
    print('\nfcu-r within series, Spearman of centred values:',
          round(spearmanr(d.fcu_MPa - d.groupby('series').fcu_MPa.transform('mean'),
                          d.r_pct - d.groupby('series').r_pct.transform('mean')).statistic, 2))
    print('n-r within series, Spearman of centred values:',
          round(spearmanr(d.n - d.groupby('series').n.transform('mean'),
                          d.r_pct - d.groupby('series').r_pct.transform('mean')).statistic, 2))

    metadata = generate_metadata(df)
    rules = [{'target_column': k, 'expression': v} for k, v in cfg['derived'].items()]
    random_splits = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(df))
    loso = list(LeaveOneGroupOut().split(df, groups=series))
    res = []
    for branch in ('iter', 'knn', 'missing'):
        for scheme, splits in (('random', random_splits), ('loso', loso)):
            w, wo = cv(df, metadata, rules, target, branch, splits, True), cv(df, metadata, rules, target, branch, splits, False)
            p = wilcoxon(w, wo).pvalue if np.any(w != wo) else 1.0
            res.append({'branch': branch, 'scheme': scheme, 'mae_raw_plus_engineered': w.mean(), 'mae_raw_only': wo.mean(),
                        'engineered_lower_folds': int((w < wo).sum()), 'folds': len(w), 'wilcoxon_p_uncorrected': p})
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(a.out, 'engineered_feature_ablation.csv'), index=False)
    print('\nEngineered-feature ablation\n', res.round(3).to_string())


if __name__ == '__main__':
    main()
