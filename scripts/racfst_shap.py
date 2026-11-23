"""Exploratory TreeSHAP analysis of the real-only XGBoost model for RACFST Pu (not pre-registered).

    OMP_NUM_THREADS=6 nice -n 10 python scripts/racfst_shap.py --out runs/2026-11-23_SHAP

Uses the E0 fold procedure (repeated 5-fold CV, fold-local imputation, derived ratios rebuilt after imputation) and
XGBoost's exact TreeSHAP (pred_contribs). SHAP values are computed on held-out rows only. A leave-one-series-out run
shows which explanations survive when the model cannot see sister specimens. SHAP explains the fitted model; it does
not establish a physical cause, and correlated inputs can share credit.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from metadata_manager import generate_metadata  # noqa: E402
from metrics import _impute_fold_train_test, _prepare_regression_features, apply_reconstruction_rules  # noqa: E402

SEED = 2026
BRANCHES = {'iter': 'Iterative', 'knn': 'KNN', 'missing': 'Median (MI branch)'}
PUBLISHED_REAL_XGB_MAE = {'iter': 6.36, 'knn': 5.14, 'missing': 4.97}

# Written before the first run, from general mechanics. +1: larger value should raise Pu; -1: lower it.
# Inputs without one clear single-variable expectation are omitted (n is non-monotonic; mix quantities act jointly).
EXPECTED_SIGN = {'D_mm': 1, 't_mm': 1, 'fy_MPa': 1, 'fcu_MPa': 1, 'L_D': -1, 'D_t': -1, 'r_pct': -1, 'w_c': -1}
GROUPS = {
    'Section geometry': ['Section', 'D_mm', 't_mm', 'D_t'],
    'Member length': ['L2_mm', 'L_D'],
    'Steel strength': ['fy_MPa'],
    'Axial load': ['n'],
    'Concrete and mix': ['fcu_MPa', 'r_pct', 'cement_kgm3', 'sand_kgm3', 'nca_kgm3', 'rca_kgm3', 'water_kgm3', 'w_c'],
}


def model():
    return xgb.XGBRegressor(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                            objective='reg:squarederror', random_state=SEED, n_jobs=6)


def run(df, metadata, rules, target, branch, splits):
    """Return held-out predictions and SHAP rows, with the one-hot Section columns summed into one feature."""
    rows, fold_mae = [], []
    for fold, (tr, te) in enumerate(splits):
        train, test = _impute_fold_train_test(df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True),
                                              metadata, branch, target, SEED)
        train, test = apply_reconstruction_rules(train, rules), apply_reconstruction_rules(test, rules)
        y_tr, y_te = train[target].astype(float), test[target].astype(float)
        X_tr, X_te = _prepare_regression_features(train.drop(columns=[target]), test.drop(columns=[target]))
        m = model().fit(X_tr, y_tr)
        pred = m.predict(X_te)
        fold_mae.append(mean_absolute_error(y_te, pred))
        contrib = m.get_booster().predict(xgb.DMatrix(X_te), pred_contribs=True)
        gap = np.abs(contrib.sum(axis=1) - pred).max()
        if gap > 1e-2:
            raise RuntimeError(f'SHAP additivity failed: {gap}')
        shap = pd.DataFrame(contrib[:, :-1], columns=X_te.columns)
        sec = [c for c in shap.columns if c.startswith('Section_')]
        shap['Section'] = shap[sec].sum(axis=1)
        shap = shap.drop(columns=sec)
        values = X_te.drop(columns=sec).add_prefix('value_')
        values['value_Section'] = (test['Section'].astype(str).values == 'Circular').astype(int)
        out = pd.concat([shap, values], axis=1)
        out['row'], out['fold'], out['actual'], out['pred'] = te, fold, y_te.values, pred
        rows.append(out)
    return pd.concat(rows, ignore_index=True), float(np.mean(fold_mae))


def summarise(res, features):
    imp = res[features].abs().mean()
    table = pd.DataFrame({'mean_abs_shap_kN': imp, 'share_pct': 100 * imp / imp.sum()})
    table['rank'] = table.mean_abs_shap_kN.rank(ascending=False).astype(int)
    table['direction_rho'] = [spearmanr(res['value_' + f], res[f]).statistic if res['value_' + f].nunique() > 1 and
                              res[f].nunique() > 1 else np.nan for f in features]
    fold_rank = res.groupby('fold')[features].apply(lambda g: g.abs().mean().rank(ascending=False))
    table['rank_fold_q10'] = fold_rank.quantile(.1)
    table['rank_fold_q90'] = fold_rank.quantile(.9)
    return table.sort_values('rank')


def groups(res):
    g = pd.DataFrame({k: res[v].sum(axis=1) for k, v in GROUPS.items()})
    imp = g.abs().mean()
    return pd.DataFrame({'mean_abs_shap_kN': imp, 'share_pct': 100 * imp / imp.sum()}).sort_values('share_pct', ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(PROJECT, 'configs', 'racfst.json'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(a.config, encoding='utf-8'))
    target = cfg['regression_target']
    df = pd.read_excel(os.path.join(PROJECT, cfg['data'])).drop(columns=[cfg['id_column']])
    metadata = generate_metadata(df)
    rules = [{'target_column': k, 'expression': v} for k, v in cfg['derived'].items()]
    features = [c for c in df.columns if c != target] + list(cfg['derived'])
    series = df.groupby(['Section', 'D_mm', 't_mm']).ngroup().values
    random_splits = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(df))
    loso_splits = list(LeaveOneGroupOut().split(df, groups=series))

    report = {'series': int(len(set(series))), 'branches': {}}
    for branch, label in BRANCHES.items():
        for scheme, splits in (('random', random_splits), ('loso', loso_splits)):
            res, mae = run(df, metadata, rules, target, branch, splits)
            res.to_csv(os.path.join(a.out, f'shap_rows_{branch}_{scheme}.csv'), index=False)
            table = summarise(res, features)
            table.to_csv(os.path.join(a.out, f'importance_{branch}_{scheme}.csv'))
            grp = groups(res)
            grp.to_csv(os.path.join(a.out, f'groups_{branch}_{scheme}.csv'))
            entry = report['branches'].setdefault(branch, {'label': label})
            entry[scheme] = {'mean_fold_mae': mae, 'pooled_r2': float(r2_score(res.actual, res.pred)),
                             'order': table.index.tolist(), 'groups': grp.share_pct.round(1).to_dict()}
            if scheme == 'random':
                entry['published_real_xgb_mae'] = PUBLISHED_REAL_XGB_MAE[branch]
                entry['sign_check'] = {f: {'expected': s, 'rho': float(table.loc[f, 'direction_rho']),
                                           'agrees': bool(np.sign(table.loc[f, 'direction_rho']) == s)}
                                       for f, s in EXPECTED_SIGN.items()}
            print(f'\n=== {label} | {scheme} | MAE {mae:.2f} kN | pooled R2 {entry[scheme]["pooled_r2"]:.2f}')
            print(table.round(2).to_string())
            print(grp.round(2).to_string())
    x = df.copy()
    x = apply_reconstruction_rules(x, rules).drop(columns=[target, 'Section'])
    corr = x.corr(method='spearman')
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
    strong = pairs[pairs.abs() >= .8].sort_values(key=np.abs, ascending=False)
    strong.to_csv(os.path.join(a.out, 'strong_feature_correlations.csv'))
    report['strong_correlations'] = {f'{i}|{j}': round(float(v), 2) for (i, j), v in strong.items()}
    per_series = df.assign(series=series).groupby('series')[[c for c in features if c in df.columns and c != 'Section']].nunique()
    per_series.to_csv(os.path.join(a.out, 'unique_values_per_series.csv'))
    print('\nStrong correlations\n', strong.round(2).to_string())
    print('\nUnique values per series\n', per_series.to_string())
    json.dump(report, open(os.path.join(a.out, 'shap_summary.json'), 'w'), indent=1)
    print('\nSIGN CHECK', json.dumps({b: {k: (round(v['rho'], 2), v['agrees']) for k, v in report['branches'][b]['sign_check'].items()} for b in BRANCHES}))


if __name__ == '__main__':
    main()
