"""E1: does synthetic augmentation help at small training sizes? (design pre-registered in PLAN.md, 2026-11-14)

Run from an empty run folder (SDV generators write sdv_metadata.json to the working directory):
    python /path/to/scripts/e1_training_size.py --data DATA.xlsx --target COL --sizes 25 50 100 \
        --draws 20 --out results.csv [--generators CART GaussianCopula CTGAN CopulaGAN TVAE]

The first column of DATA.xlsx is an ID and is dropped (same convention as run_all.py).
Results are appended per draw, so an interrupted run resumes where it stopped.
"""
import argparse
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402
import xgboost as xgb                                   # noqa: E402
from sklearn.ensemble import RandomForestRegressor      # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from metadata_manager import generate_metadata, save_as_sdv_json  # noqa: E402
from metrics import (_impute_fold_train_test, _build_fold_constraints,  # noqa: E402
                     _prepare_regression_features, generate_valid_samples_adaptive)
from run_all import DEFAULT_GENERATION_SETTINGS, build_aligned_metadata_for_df  # noqa: E402
from sdv_all import run_gaussian_copula, run_ctgan, run_copula_gan, run_tvae  # noqa: E402
from synthpop.synth import run_cart                     # noqa: E402

BASE_SEED = 2026
TEST_FRACTION = 0.25
SDV = {'GaussianCopula': run_gaussian_copula, 'CTGAN': run_ctgan, 'CopulaGAN': run_copula_gan, 'TVAE': run_tvae}
ALL_GENERATORS = ['CART'] + list(SDV)


JOBS = 6


def regressors(seed):
    return {
        'XGBoost': xgb.XGBRegressor(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.9,
                                    colsample_bytree=0.9, objective='reg:squarederror', random_state=seed, n_jobs=JOBS),
        'RandomForest': RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=JOBS),
    }


def score(train, test, target, seed):
    """Fit both regressors on train, return {name: (mae, rmse, r2)} on test."""
    y_tr = train[target].astype(float).values
    y_te = test[target].astype(float).values
    X_tr, X_te = _prepare_regression_features(train.drop(columns=[target]), test.drop(columns=[target]))
    out = {}
    for name, model in regressors(seed).items():
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        out[name] = (mean_absolute_error(y_te, p), float(np.sqrt(mean_squared_error(y_te, p))), r2_score(y_te, p))
    return out


def generate(gen, train, n, metadata, constraints):
    meta = build_aligned_metadata_for_df(train, metadata)
    if gen == 'CART':
        func = lambda d, k, p: run_cart(d, k, p, build_aligned_metadata_for_df(d, meta), persist=False)
    else:
        def func(d, k, p, mf=SDV[gen]):
            save_as_sdv_json(build_aligned_metadata_for_df(d, meta), path='sdv_metadata.json')
            return mf(d, k, p)
    s = DEFAULT_GENERATION_SETTINGS
    _, synth = generate_valid_samples_adaptive(
        model_func=func, imp_df=train, real_reference_df=train, target_size=n, prefix='e1', metadata=meta,
        constraints=constraints, ros_target_column=None, rejection_thresholds=None,   # no statistical gate
        min_batch_size=s['min_batch_size'], max_batch_size=s['max_batch_size'],
        max_total_generated_multiplier=s['max_total_generated_multiplier'],
        max_failed_attempts=s['max_failed_attempts'], acceptance_rate_floor=s['acceptance_rate_floor'])
    return synth[train.columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', required=True)
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--generators', nargs='+', default=ALL_GENERATORS)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=6, help='worker threads for the regressors (laptop limit: 6)')
    a = ap.parse_args()
    global JOBS
    JOBS = a.jobs

    df = pd.read_excel(a.data)
    df = df.drop(columns=df.columns[0]).reset_index(drop=True)
    metadata = generate_metadata(df)
    dataset = os.path.splitext(os.path.basename(a.data))[0]
    done = set(pd.read_csv(a.out).draw.unique()) if os.path.exists(a.out) else set()

    for r in range(1, a.draws + 1):
        if r in done:
            continue
        seed = BASE_SEED + r
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
        n_test = int(round(TEST_FRACTION * len(df)))
        test_raw = df.iloc[order[:n_test]].reset_index(drop=True)
        pool = df.iloc[order[n_test:]].reset_index(drop=True)          # shuffled once; sizes are nested prefixes
        rows = []
        for n in a.sizes:
            if n > len(pool):
                raise ValueError(f'size {n} exceeds training pool {len(pool)}')
            train, test = _impute_fold_train_test(pool.iloc[:n].reset_index(drop=True), test_raw.copy(),
                                                  metadata, 'iter', a.target, seed)
            constraints = _build_fold_constraints(train, [])
            base = score(train, test, a.target, seed)
            for reg, (mae, rmse, r2) in base.items():
                rows.append(dict(dataset=dataset, draw=r, n=n, generator='RealOnly', regressor=reg,
                                 mae=mae, rmse=rmse, r2=r2, n_synth=0, gen_seconds=0.0, failed=False))
            for gen in a.generators:
                t0 = time.time()
                try:
                    synth = generate(gen, train, n, metadata, constraints)
                    res = score(pd.concat([train, synth], ignore_index=True), test, a.target, seed)
                    failed = False
                except Exception as exc:   # pre-registered: failed draws are excluded for that generator
                    print(f'  [{dataset}] draw {r} n={n} {gen} failed: {exc}')
                    res = {reg: (np.nan, np.nan, np.nan) for reg in base}
                    synth, failed = [], True
                for reg, (mae, rmse, r2) in res.items():
                    rows.append(dict(dataset=dataset, draw=r, n=n, generator=gen, regressor=reg, mae=mae,
                                     rmse=rmse, r2=r2, n_synth=len(synth), gen_seconds=time.time() - t0,
                                     failed=failed))
            print(f'[{dataset}] draw {r}/{a.draws} n={n} done', flush=True)
        pd.DataFrame(rows).to_csv(a.out, mode='a', header=not os.path.exists(a.out), index=False)


if __name__ == '__main__':
    main()
