import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import re
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import GaussianMixture
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import warnings
from plot import Plot
from ros import apply_ros, drop_constraint_violators
warnings.filterwarnings('ignore')

METRIC_DIRECTIONS = {
    'Marginal': 'max',
    'boundary_score': 'max',
    'corr_score': 'max',
    'cov_score': 'max',
    'violation_rate': 'min',
    'DCR_mean': 'max',
    'DCR_min': 'max',
    'NNDR_mean': 'max',
    'wd_score': 'min',
    'mmd_score': 'min',
    'fd_score': 'min',
    'log_cluster_score': 'max',
    'propensity_score': 'max'
}

DEFAULT_RANDOM_SEED = 2026
AUTO_BOUND_PATTERN = re.compile(r'^\s*c\d+\s*(>=|<=)\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$')


def _encode_train_test_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame, metadata: dict, excluded_cols=None):
    excluded = set(excluded_cols or [])
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    encoders = {}

    for col, dtype in metadata.items():
        if col in excluded or dtype != 'categorical':
            continue
        if col not in train_encoded.columns or col not in test_encoded.columns:
            continue

        train_series = train_encoded[col].astype('category')
        categories = train_series.cat.categories
        encoders[col] = categories
        train_encoded[col] = train_series.cat.codes.replace(-1, np.nan)

        test_categorical = pd.Categorical(test_encoded[col], categories=categories)
        test_encoded[col] = pd.Series(test_categorical.codes, index=test_encoded.index).replace(-1, np.nan)

    return train_encoded, test_encoded, encoders


def _decode_categorical_with_encoders(df: pd.DataFrame, encoders: dict):
    decoded = df.copy()
    for col, categories in encoders.items():
        if col not in decoded.columns:
            continue
        series = decoded[col].round().fillna(-1)
        vals = series.values.copy()
        valid_mask = vals >= 0
        vals[valid_mask] = np.clip(vals[valid_mask], 0, len(categories) - 1)
        decoded[col] = pd.Categorical.from_codes(vals.astype(int), categories=categories).astype(object)
    return decoded


def _impute_categorical_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, metadata: dict, random_state: int):
    train = train_df.copy()
    test = test_df.copy()

    numeric_cols = [
        col for col, dtype in metadata.items()
        if dtype == 'numerical' and col in train.columns and col in test.columns
    ]
    categorical_cols = [
        col for col, dtype in metadata.items()
        if dtype == 'categorical' and col in train.columns and col in test.columns
    ]

    for target_col in categorical_cols:
        known_train = train[train[target_col].notna()]

        if known_train.empty:
            mode_vals = pd.concat([train[target_col], test[target_col]], axis=0).mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else 'Missing'
            train[target_col] = train[target_col].fillna(fill_value)
            test[target_col] = test[target_col].fillna(fill_value)
            continue

        if not numeric_cols:
            mode_vals = known_train[target_col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else 'Missing'
            train[target_col] = train[target_col].fillna(fill_value)
            test[target_col] = test[target_col].fillna(fill_value)
            continue

        X_known = known_train[numeric_cols].copy()
        y_known = known_train[target_col].copy()
        fill_values = X_known.median()
        X_known = X_known.fillna(fill_values)

        if y_known.nunique(dropna=True) <= 1:
            fill_value = y_known.iloc[0]
            train[target_col] = train[target_col].fillna(fill_value)
            test[target_col] = test[target_col].fillna(fill_value)
            continue

        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_known, y_known)

        for frame in (train, test):
            missing_mask = frame[target_col].isna()
            if not missing_mask.any():
                continue
            X_missing = frame.loc[missing_mask, numeric_cols].copy().fillna(fill_values)
            frame.loc[missing_mask, target_col] = model.predict(X_missing)

    return train, test


def _impute_fold_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
    imputation_name: str,
    target_column: str,
    random_state: int
):
    train, test = _impute_categorical_train_test(train_df, test_df, metadata, random_state)

    feature_cols = [col for col in train.columns if col != target_column]
    if not feature_cols:
        return train, test

    if imputation_name in {'iter', 'knn'}:
        train_encoded, test_encoded, encoders = _encode_train_test_categorical(
            train[feature_cols], test[feature_cols], metadata
        )

        if imputation_name == 'iter':
            imputer = IterativeImputer(random_state=random_state)
        else:
            imputer = KNNImputer(n_neighbors=5)

        train_values = imputer.fit_transform(train_encoded)
        test_values = imputer.transform(test_encoded)

        train_imputed = pd.DataFrame(train_values, columns=feature_cols, index=train.index)
        test_imputed = pd.DataFrame(test_values, columns=feature_cols, index=test.index)

        train_imputed = _decode_categorical_with_encoders(train_imputed, encoders)
        test_imputed = _decode_categorical_with_encoders(test_imputed, encoders)
    else:
        train_imputed = train[feature_cols].copy()
        test_imputed = test[feature_cols].copy()

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(train_imputed[col]):
                fill_value = train_imputed[col].median()
                train_imputed[col] = train_imputed[col].fillna(fill_value)
                test_imputed[col] = test_imputed[col].fillna(fill_value)
            else:
                mode_vals = train_imputed[col].mode(dropna=True)
                fill_value = mode_vals.iloc[0] if not mode_vals.empty else 'Missing'
                train_imputed[col] = train_imputed[col].fillna(fill_value)
                test_imputed[col] = test_imputed[col].fillna(fill_value)

    train_final = train.copy()
    test_final = test.copy()
    for col in feature_cols:
        train_final[col] = train_imputed[col]
        test_final[col] = test_imputed[col]

    return train_final, test_final


def _build_fold_constraints(train_df: pd.DataFrame, base_constraints: list):
    custom_constraints = [
        con for con in (base_constraints or [])
        if not AUTO_BOUND_PATTERN.match(str(con).strip())
    ]

    fold_constraints = []
    for idx, col in enumerate(train_df.columns):
        if pd.api.types.is_numeric_dtype(train_df[col]):
            col_min = train_df[col].min()
            col_max = train_df[col].max()
            fold_constraints.append(f'c{idx} >= {col_min}')
            fold_constraints.append(f'c{idx} <= {col_max}')

    fold_constraints.extend(custom_constraints)
    return fold_constraints


def _normalize_formula_expression(expression: str) -> str:
    return re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'c\1', expression)


def _evaluate_formula_expression(expression: str, reference_df: pd.DataFrame) -> pd.Series:
    if reference_df.empty:
        raise ValueError('Reference dataframe is empty.')

    normalized_expr = _normalize_formula_expression(expression.strip())
    parsed = ast.parse(normalized_expr, mode='eval')
    column_refs = {f'c{i}': reference_df.iloc[:, i] for i in range(reference_df.shape[1])}

    def _eval_ast(ast_node):
        if isinstance(ast_node, ast.Expression):
            return _eval_ast(ast_node.body)

        if isinstance(ast_node, ast.BinOp):
            left_val = _eval_ast(ast_node.left)
            right_val = _eval_ast(ast_node.right)

            if isinstance(ast_node.op, ast.Add):
                return left_val + right_val
            if isinstance(ast_node.op, ast.Sub):
                return left_val - right_val
            if isinstance(ast_node.op, ast.Mult):
                return left_val * right_val
            if isinstance(ast_node.op, ast.Div):
                return left_val / right_val
            if isinstance(ast_node.op, ast.Pow):
                return left_val ** right_val
            if isinstance(ast_node.op, ast.Mod):
                return left_val % right_val

            raise ValueError('Unsupported operator in formula.')

        if isinstance(ast_node, ast.UnaryOp):
            operand = _eval_ast(ast_node.operand)
            if isinstance(ast_node.op, ast.UAdd):
                return operand
            if isinstance(ast_node.op, ast.USub):
                return -operand
            raise ValueError('Unsupported unary operator in formula.')

        if isinstance(ast_node, ast.Name):
            if ast_node.id not in column_refs:
                raise ValueError(f"Unknown column reference '{ast_node.id}'.")
            return column_refs[ast_node.id]

        if isinstance(ast_node, ast.Constant) and isinstance(ast_node.value, (int, float)):
            return float(ast_node.value)

        raise ValueError('Unsupported expression syntax.')

    evaluated = _eval_ast(parsed)
    if np.isscalar(evaluated):
        return pd.Series([evaluated] * len(reference_df), index=reference_df.index)
    return pd.Series(evaluated, index=reference_df.index)


def _apply_reconstruction_rules(df: pd.DataFrame, reconstruction_rules=None) -> pd.DataFrame:
    if not reconstruction_rules:
        return df

    base_reference = df.copy()
    reconstructed_df = df.copy()

    for rule in reconstruction_rules:
        target_column = rule.get('target_column')
        expression = rule.get('expression')
        if not target_column or not expression:
            continue
        try:
            reconstructed_df[target_column] = _evaluate_formula_expression(expression, base_reference)
        except Exception as exc:
            print(f"Warning: could not apply reconstruction formula '{target_column}={expression}': {exc}")

    return reconstructed_df


def summarize_metrics(metric_output: dict) -> dict:
    row = {}
    if 'marginals' in metric_output and metric_output['marginals']:
        row['Marginal'] = float(np.mean(list(metric_output['marginals'].values())))

    for key, value in metric_output.items():
        if key == 'marginals':
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            row[key] = float(value)
    return row


def evaluate_threshold_gate(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    metadata: dict,
    constraints,
    thresholds: dict | None = None,
    metric_directions: dict | None = None
):
    metric_output = eval(real_df, synth_df, metadata, constraints)
    metric_row = summarize_metrics(metric_output)

    if not thresholds:
        return True, {}, metric_row

    directions = metric_directions or METRIC_DIRECTIONS
    failed = {}

    for metric_name, threshold in thresholds.items():
        if metric_name not in metric_row:
            failed[metric_name] = {'reason': 'missing_metric', 'threshold': threshold}
            continue

        metric_value = metric_row[metric_name]
        direction = directions.get(metric_name, 'max')
        if direction == 'min':
            passed = metric_value <= threshold
        else:
            passed = metric_value >= threshold

        if not passed:
            failed[metric_name] = {
                'value': metric_value,
                'threshold': threshold,
                'direction': direction
            }

    return len(failed) == 0, failed, metric_row

# =====================================================================
# 1. 1D STATISTICAL FIDELITY METRICS
# =====================================================================

def ks_complement(real_col: pd.Series, synth_col: pd.Series) -> float:
    """
    Kolmogorov-Smirnov Complement. Measures the maximum distance between 
    two continuous empirical Cumulative Distribution Functions (CDFs).
    Returns: 1.0 (perfect match) to 0.0 (no match).
    """
    stat, _ = ks_2samp(real_col.dropna(), synth_col.dropna())
    return 1.0 - stat

def tv_complement(real_col: pd.Series, synth_col: pd.Series) -> float:
    """
    Total Variation Complement. For categorical columns (e.g., Square vs Circle).
    Returns: 1.0 (perfect frequency match) to 0.0.
    """
    real_freq = real_col.value_counts(normalize=True)
    synth_freq = synth_col.value_counts(normalize=True)
    
    # Align indexes to compare same categories
    all_cats = set(real_freq.index).union(set(synth_freq.index))
    real_freq = real_freq.reindex(list(all_cats), fill_value=0.0)
    synth_freq = synth_freq.reindex(list(all_cats), fill_value=0.0)
    
    tv_distance = 0.5 * np.sum(np.abs(real_freq - synth_freq))
    return 1.0 - tv_distance

def marginal_distribution(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> dict:
    """
    Single Column Check. Applies KS for numerical and TV for categorical data.
    """
    results = {}
    for col, dtype in metadata.items():
        if col in real_df.columns and col in synth_df.columns:
            if dtype == 'numerical':
                results[col] = ks_complement(real_df[col], synth_df[col])
            elif dtype == 'categorical':
                results[col] = tv_complement(real_df[col], synth_df[col])
    return results

def boundary_adherence(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> float:
    """
    Calculates the percentage of synthetic numerical data that stays within 
    the physical [min, max] boundaries of the real experimental data.
    """
    adherence_scores = []
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    for col in num_cols:
        min_val, max_val = real_df[col].min(), real_df[col].max()
        valid_mask = (synth_df[col] >= min_val) & (synth_df[col] <= max_val)
        adherence_scores.append(valid_mask.mean())
        
    return np.mean(adherence_scores) if adherence_scores else 1.0

# =====================================================================
# 2. MULTIVARIATE & ENGINEERING METRICS
# =====================================================================

def correlation_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> float:
    """
    Measures the similarity between the correlation matrices of real and synthetic data.
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    if len(num_cols) < 2: return 1.0
    
    real_corr = real_df[num_cols].corr().fillna(0).values
    synth_corr = synth_df[num_cols].corr().fillna(0).values
    
    # Matrix norm difference normalized to [0, 1]
    diff = np.abs(real_corr - synth_corr)
    return 1.0 - (np.sum(diff) / (len(num_cols) ** 2 - len(num_cols)))

def constraints_violation_rate(synth_df: pd.DataFrame, constraints: list) -> float:
    """
    Evaluates what percentage of synthetic rows VIOLATE the provided Pandas rules.
    Lower is better (0.0 means perfect adherence to engineering rules).
    """
    from ros import drop_constraint_violators
    good = drop_constraint_violators(synth_df, constraints)
    violation_rate = 1.0 - (len(good) / len(synth_df))
    return violation_rate

def covariance_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> float:
    """
    Measures how well the synthetic data preserves the Covariance Matrix of the real data.
    Returns a score from 0.0 (complete failure) to 1.0 (perfect match).
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    # Standardize the data first so variables with large numbers (like Load) 
    # don't overshadow variables with small numbers (like D/t ratio)
    real_std = (real_df[num_cols] - real_df[num_cols].mean()) / real_df[num_cols].std()
    synth_std = (synth_df[num_cols] - real_df[num_cols].mean()) / real_df[num_cols].std()
    
    # Calculate Covariance Matrices
    cov_real = np.cov(real_std.fillna(0), rowvar=False)
    cov_synth = np.cov(synth_std.fillna(0), rowvar=False)
    
    # Calculate Frobenius Norm (the geometric distance between the two matrices)
    norm_diff = np.linalg.norm(cov_real - cov_synth, ord='fro')
    norm_real = np.linalg.norm(cov_real, ord='fro')
    
    # Normalize to a 0 to 1 scale
    similarity = max(0.0, 1.0 - (norm_diff / norm_real))
    return float(similarity)
# =====================================================================
# 3. PRIVACY & OVERFITTING METRICS
# =====================================================================


def calculate_dcr_nndr(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict):
    """
    Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR).
    Ensures the GAN isn't just copying real experimental specimens.
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    # Standardize data for fair distance calculation
    real_num = (real_df[num_cols] - real_df[num_cols].mean()) / real_df[num_cols].std()
    synth_num = (synth_df[num_cols] - real_df[num_cols].mean()) / real_df[num_cols].std()
    
    # Find 2 nearest neighbors in REAL data for every SYNTHETIC point
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn.fit(real_num.fillna(0))
    distances, _ = nn.kneighbors(synth_num.fillna(0))
    
    # DCR: Distance to the absolute closest real point
    dcr = distances[:, 0]
    
    # NNDR: Ratio of distance to 1st closest vs 2nd closest (checks for memorization)
    # Add tiny epsilon to avoid division by zero
    nndr = distances[:, 0] / (distances[:, 1] + 1e-8)
    
    return {
        'DCR_mean': np.mean(dcr),
        'DCR_min': np.min(dcr), # If 0, model perfectly copied a specimen
        'NNDR_mean': np.mean(nndr)
    }

# =====================================================================
# 4. ADVANCED GAN DISTRIBUTION METRICS
# =====================================================================

def feature_wasserstein_distance(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> float:
    """
    Earth Mover's Distance. Averages the 1D WD across all numerical columns.
    Lower is better. Represents the 'work' required to transform synthetic physics into real physics.
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    wd_scores = [wasserstein_distance(real_df[col].dropna(), synth_df[col].dropna()) for col in num_cols]
    return np.mean(wd_scores)



def max_mean_discrepancy(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> float:
    """
    Maximum Mean Discrepancy (MMD). Uses an RBF Kernel to check if the high-dimensional
    multivariate shapes of the datasets match. Lower is better.
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    X = real_df[num_cols].fillna(0).values
    Y = synth_df[num_cols].fillna(0).values
    
    XX = rbf_kernel(X, X)
    YY = rbf_kernel(Y, Y)
    XY = rbf_kernel(X, Y)
    
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return float(mmd)

def frechet_distance(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> float:
    """
    Fréchet Distance (Multivariate Gaussian approximation). 
    Used heavily in GANs (similar to FID but for tabular data). Lower is better.
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    mu_real, sigma_real = real_df[num_cols].mean().values, real_df[num_cols].cov().values
    mu_synth, sigma_synth = synth_df[num_cols].mean().values, synth_df[num_cols].cov().values
    
    diff = mu_real - mu_synth
    
    # Calculate sqrt of dot product of covariances
    covmean, _ = sqrtm(sigma_real.dot(sigma_synth), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fd = diff.dot(diff) + np.trace(sigma_real + sigma_synth - 2.0 * covmean)
    return float(fd)

def log_cluster(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict) -> float:
    """
    Log Cluster Metric. Fits a Gaussian Mixture Model on the REAL data to find 
    "structural clusters", then calculates the Log-Likelihood of the SYNTHETIC data.
    Higher is better (means synthetic data falls neatly into the real physical clusters).
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    X_real = real_df[num_cols].fillna(0).values
    X_synth = synth_df[num_cols].fillna(0).values
    
    # Number of components usually kept low for N=60 datasets
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=DEFAULT_RANDOM_SEED)
    gmm.fit(X_real)
    
    # Mean log-likelihood of synthetic data under the real distribution
    return float(np.mean(gmm.score_samples(X_synth)))

def propensity_score(real_df,synth_df,metadata):
    real = real_df.copy()
    fake = synth_df.copy()
    
    # Drop duplicates in Real data to prevent ROS leakage (train/test memorization)
    real = real.drop_duplicates()
    
    real['Target'] = 1
    fake['Target'] = 0
    
    # Align columns to ensure consistency
    common_cols = [c for c in real.columns if c in fake.columns and c != 'Target']
    
    # Align precision to prevent trivial leaks (e.g. 12 vs 12.0000001)
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(real[col]):
            real_vals = real[col].dropna()
            # Check if real data is effectively integer (e.g. 5.0, 12.0)
            if len(real_vals) > 0 and (real_vals % 1 == 0).all():
                fake[col] = fake[col].round()
                real[col] = real[col].round()
            else:
                fake[col] = fake[col].round(4)
                real[col] = real[col].round(4)
                
    data = pd.concat([real[common_cols + ['Target']], fake[common_cols + ['Target']]], ignore_index=True)
    
    # Step 2: Preprocess and Split
    X = data.drop(columns=['Target'])
    y = data['Target']
    
    # Handle Missing Values to prevent "NaN vs Value" leakage
    # If real data has NaNs and synthetic doesn't, XGBoost sees it instantly.
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            if not X[col].mode().empty:
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna("Missing")

    # Binning Strategy to prevent trivial distribution leaks (spikes vs smooth)
    # We bin continuous features into 10 quantiles. This forces the discriminator
    # to look at the overall shape (histogram) rather than exact values.
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 20:
            # Use rank to handle non-unique edges (like many 0s)
            X[col] = pd.qcut(X[col].rank(method='first'), q=10, labels=False)

    # One-hot encoding for categorical columns
    categorical_cols = [col for col, dtype in metadata.items() if dtype == 'categorical' and col in X.columns]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, dtype=int)
    
    # Split 75/25
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=DEFAULT_RANDOM_SEED,
        stratify=y
    )
    
    # Step 3: Train the "Disposable" XGBoost Detective
    # Constraint: max_depth=2 and fewer trees to prevent overfitting/memorization (more lenient)
    model = xgb.XGBClassifier(
        max_depth=1, n_estimators=30, eval_metric='logloss', random_state=DEFAULT_RANDOM_SEED
    )
    model.fit(X_train, y_train)
    
    '''# Debugging: Print Feature Importance to identify leaks
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n[Propensity Debug] Top features used to distinguish Real vs Fake:")
        for f in range(min(3, X.shape[1])):
            print(f"  {X.columns[indices[f]]}: {importances[indices[f]]:.4f}")
    '''
    # Step 4: Predict and Score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    score = 1 - 2 * abs(0.5 - auc)
    #print(f"[Propensity Debug] AUC: {auc:.4f} | Score: {score:.4f}")
    return score


def eval(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict, constraints):
    output = {}
    output['marginals'] = marginal_distribution(real_df, synth_df, metadata)
    output['boundary_score'] = boundary_adherence(real_df, synth_df, metadata)
    output['corr_score'] = correlation_similarity(real_df, synth_df, metadata)
    output['cov_score'] = covariance_similarity(real_df, synth_df, metadata)
    output['violation_rate'] = constraints_violation_rate(synth_df, constraints)
    dcr_nndr = calculate_dcr_nndr(real_df, synth_df, metadata)
    output.update(dcr_nndr)
    output['wd_score'] = feature_wasserstein_distance(real_df, synth_df, metadata)
    output['mmd_score'] = max_mean_discrepancy(real_df, synth_df, metadata)
    output['fd_score'] = frechet_distance(real_df, synth_df, metadata)
    output['log_cluster_score'] = log_cluster(real_df, synth_df, metadata)
    output['propensity_score'] = propensity_score(real_df,synth_df,metadata)
    return output


def generate_valid_samples_adaptive(
    model_func,
    imp_df: pd.DataFrame,
    real_reference_df: pd.DataFrame,
    target_size: int,
    prefix: str,
    metadata: dict,
    constraints,
    ros_target_column: str | None = None,
    rejection_thresholds: dict | None = None,
    min_batch_size: int = 2000,
    max_batch_size: int = 5000,
    max_total_generated_multiplier: int = 20,
    max_failed_attempts: int = 8,
    acceptance_rate_floor: float = 0.05,
    verbose: bool = False
):
    valid_data = pd.DataFrame()
    total_generated = 0
    consecutive_failed_attempts = 0
    total_failed_attempts = 0
    estimated_acceptance_rate = 1.0
    model_name = ''
    batch_iteration = 0
    MAX_ITERATIONS = 500  # Hard iteration limit to prevent infinite loops

    max_total_generated = max(int(target_size * max_total_generated_multiplier), min_batch_size)

    while (
        len(valid_data) < target_size
        and consecutive_failed_attempts < max_failed_attempts
        and total_generated < max_total_generated
        and batch_iteration < MAX_ITERATIONS
    ):
        remaining = target_size - len(valid_data)
        effective_rate = max(estimated_acceptance_rate, acceptance_rate_floor)
        requested_batch = int(np.ceil(remaining / effective_rate))
        batch_size = max(min_batch_size, requested_batch)
        batch_size = min(batch_size, max_batch_size)

        total_remaining_budget = max_total_generated - total_generated
        if total_remaining_budget <= 0:
            break
        batch_size = min(batch_size, total_remaining_budget)
        if batch_size <= 0:
            break

        batch_iteration += 1
        if batch_iteration > MAX_ITERATIONS:
            break

        model_name, temp_df = model_func(imp_df, batch_size, prefix)
        total_generated += batch_size

        if ros_target_column:
            temp_df = apply_ros(temp_df, ros_target_column)
        temp_df = drop_constraint_violators(temp_df, constraints)

        observed_acceptance_rate = len(temp_df) / batch_size if batch_size else 0.0
        if temp_df.empty:
            consecutive_failed_attempts += 1
            total_failed_attempts += 1
            estimated_acceptance_rate = max(
                acceptance_rate_floor,
                0.5 * estimated_acceptance_rate + 0.5 * observed_acceptance_rate
            )
            if verbose:
                print(
                    f"Rejected empty synthetic batch ({prefix}); generated={total_generated}, "
                    f"consecutive_failures={consecutive_failed_attempts}, total_failures={total_failed_attempts}"
                )
            # Force exit if too many consecutive empty batches (indicates fundamental constraint issue)
            if consecutive_failed_attempts >= max(5, int(max_failed_attempts * 0.3)):
                if verbose:
                    print(f"  [TIMEOUT] Too many consecutive empty batches. Exiting generation for {prefix}.")
                break
            continue

        keep_batch = True
        failed_metrics = {}
        metric_snapshot = {}
        if rejection_thresholds:
            keep_batch, failed_metrics, metric_snapshot = evaluate_threshold_gate(
                real_reference_df,
                temp_df,
                metadata,
                constraints,
                thresholds=rejection_thresholds
            )

        if not keep_batch:
            consecutive_failed_attempts += 1
            total_failed_attempts += 1
            estimated_acceptance_rate = max(
                acceptance_rate_floor,
                0.5 * estimated_acceptance_rate
            )
            if verbose:
                failed_names = ', '.join(sorted(failed_metrics.keys()))
                print(
                    f"Rejected synthetic batch ({prefix}) due to thresholds: {failed_names}; "
                    f"generated={total_generated}, consecutive_failures={consecutive_failed_attempts}, "
                    f"total_failures={total_failed_attempts}"
                )
            continue

        estimated_acceptance_rate = max(
            acceptance_rate_floor,
            0.5 * estimated_acceptance_rate + 0.5 * observed_acceptance_rate
        )
        consecutive_failed_attempts = 0
        if verbose and metric_snapshot:
            print(
                f"Accepted synthetic batch ({prefix}) with Marginal={metric_snapshot.get('Marginal', np.nan):.3f}, "
                f"propensity={metric_snapshot.get('propensity_score', np.nan):.3f}, kept={len(temp_df)}"
            )

        valid_data = pd.concat([valid_data, temp_df.head(remaining)], ignore_index=True)

    if len(valid_data) < target_size:
        raise RuntimeError(
            f"Could not reach target size {target_size}. Accepted {len(valid_data)} rows after "
            f"generating {total_generated} rows with {total_failed_attempts} total failed attempts "
            f"({consecutive_failed_attempts} consecutive at stop). Batch iterations: {batch_iteration}/{MAX_ITERATIONS}."
        )

    return model_name, valid_data.iloc[:target_size].reset_index(drop=True)


def _prepare_classification_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str):
    X_train = train_df.drop(columns=[target_column]).copy()
    X_test = test_df.drop(columns=[target_column]).copy()
    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()

    for column in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[column]):
            fill_value = X_train[column].median()
            X_train[column] = X_train[column].fillna(fill_value)
            X_test[column] = X_test[column].fillna(fill_value)
        else:
            mode_vals = X_train[column].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else 'Missing'
            X_train[column] = X_train[column].fillna(fill_value)
            X_test[column] = X_test[column].fillna(fill_value)

    X_train = pd.get_dummies(X_train, dtype=int)
    X_test = pd.get_dummies(X_test, dtype=int)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    y_train = y_train.fillna('__MISSING_TARGET__').astype(str)
    y_test = y_test.fillna('__MISSING_TARGET__').astype(str)

    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train, y_test], axis=0))
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return X_train, X_test, y_train_encoded, y_test_encoded


def run_repeated_augmented_cv(
    real_df: pd.DataFrame,
    target_column: str,
    imputation_name: str,
    synthetic_model_name: str,
    model_func,
    metadata: dict,
    constraints,
    synthetic_target_size: int,
    rejection_thresholds: dict | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = DEFAULT_RANDOM_SEED,
    min_batch_size: int = 2000,
    max_batch_size: int = 5000,
    max_total_generated_multiplier: int = 20,
    max_failed_attempts: int = 8,
    acceptance_rate_floor: float = 0.05,
    synthetic_train_ratio: float = 1.0,
    use_full_synthetic_target: bool = False,
    reconstruction_rules=None
):
    if target_column not in real_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    real_base = real_df.copy()
    real_base = real_base[real_base[target_column].notna()].reset_index(drop=True)

    if real_base.empty:
        return pd.DataFrame(), pd.DataFrame()

    class_counts = real_base[target_column].value_counts()
    if class_counts.shape[0] < 2 or class_counts.min() < n_splits:
        return pd.DataFrame(), pd.DataFrame()

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    total_folds = n_splits * n_repeats

    fold_rows = []
    y_full = real_base[target_column].astype(str)
    split_indices = splitter.split(real_base.drop(columns=[target_column]), y_full)

    for fold_idx, (train_idx, test_idx) in enumerate(split_indices, start=1):
        print(
            f"[Classification CV] {imputation_name} | {synthetic_model_name} | "
            f"fold {fold_idx}/{total_folds} started"
        )
        real_train_raw = real_base.iloc[train_idx].reset_index(drop=True)
        real_test_raw = real_base.iloc[test_idx].reset_index(drop=True)

        real_train, real_test = _impute_fold_train_test(
            train_df=real_train_raw,
            test_df=real_test_raw,
            metadata=metadata,
            imputation_name=imputation_name,
            target_column=target_column,
            random_state=random_state
        )
        real_train = _apply_reconstruction_rules(real_train, reconstruction_rules)
        real_test = _apply_reconstruction_rules(real_test, reconstruction_rules)
        fold_constraints = _build_fold_constraints(real_train, constraints)
        if use_full_synthetic_target:
            fold_target_size = synthetic_target_size
        else:
            fold_target_size = max(1, int(np.ceil(len(real_train) * synthetic_train_ratio)))
            fold_target_size = min(fold_target_size, synthetic_target_size)

        try:
            _, synthetic_fold = generate_valid_samples_adaptive(
                model_func=model_func,
                imp_df=real_train,
                real_reference_df=real_train,
                target_size=fold_target_size,
                prefix=f'{imputation_name}_fold{fold_idx}',
                metadata=metadata,
                constraints=fold_constraints,
                ros_target_column=None,
                rejection_thresholds=rejection_thresholds,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                max_total_generated_multiplier=max_total_generated_multiplier,
                max_failed_attempts=max_failed_attempts,
                acceptance_rate_floor=acceptance_rate_floor,
                verbose=False
            )
        except Exception as exc:
            print(
                f"[Classification CV] {imputation_name} | {synthetic_model_name} | "
                f"fold {fold_idx}/{total_folds} skipped during synthetic generation: {exc}"
            )
            continue

        synthetic_fold = _apply_reconstruction_rules(synthetic_fold, reconstruction_rules)

        real_train_ros = apply_ros(real_train, target_column)
        synthetic_ros = apply_ros(synthetic_fold, target_column)

        augmented_train = pd.concat([real_train_ros, synthetic_ros], ignore_index=True)

        X_train, X_test, y_train, y_test = _prepare_classification_data(
            augmented_train,
            real_test,
            target_column
        )

        n_classes = len(np.unique(y_train))

        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1
            )
        }

        if n_classes > 2:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                random_state=random_state,
                n_jobs=-1
            )
        else:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=-1
            )

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            row = {
                'Imputation': imputation_name,
                'Synthetic_Model': synthetic_model_name,
                'Classifier': model_name,
                'Fold': fold_idx,
                'accuracy': accuracy_score(y_test, predictions),
                'balanced_accuracy': balanced_accuracy_score(y_test, predictions),
                'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0)
            }

            auc_value = np.nan
            if hasattr(model, 'predict_proba') and len(np.unique(y_test)) > 1:
                probabilities = model.predict_proba(X_test)
                try:
                    if probabilities.shape[1] == 2:
                        auc_value = roc_auc_score(y_test, probabilities[:, 1])
                    else:
                        auc_value = roc_auc_score(y_test, probabilities, multi_class='ovr')
                except Exception:
                    auc_value = np.nan

            row['roc_auc'] = auc_value
            fold_rows.append(row)

        print(
            f"[Classification CV] {imputation_name} | {synthetic_model_name} | "
            f"fold {fold_idx}/{total_folds} completed"
        )

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_df = (
        fold_df
        .groupby(['Imputation', 'Synthetic_Model', 'Classifier'], as_index=False)
        .agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            balanced_accuracy_mean=('balanced_accuracy', 'mean'),
            balanced_accuracy_std=('balanced_accuracy', 'std'),
            f1_macro_mean=('f1_macro', 'mean'),
            f1_macro_std=('f1_macro', 'std'),
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            folds=('Fold', 'count')
        )
    )

    return summary_df, fold_df


def run_repeated_augmented_regression_cv(
    real_df: pd.DataFrame,
    target_column: str,
    imputation_name: str,
    synthetic_model_name: str,
    model_func,
    metadata: dict,
    constraints,
    synthetic_target_size: int,
    rejection_thresholds: dict | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = DEFAULT_RANDOM_SEED,
    min_batch_size: int = 2000,
    max_batch_size: int = 5000,
    max_total_generated_multiplier: int = 20,
    max_failed_attempts: int = 8,
    acceptance_rate_floor: float = 0.05,
    synthetic_train_ratio: float = 1.0,
    use_full_synthetic_target: bool = False,
    reconstruction_rules=None
):
    if target_column not in real_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    real_base = real_df.copy()
    target_values = pd.to_numeric(real_base[target_column], errors='coerce')
    real_base = real_base[target_values.notna()].reset_index(drop=True)

    if len(real_base) < n_splits:
        return pd.DataFrame(), pd.DataFrame()

    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    total_folds = n_splits * n_repeats

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(real_base), start=1):
        print(
            f"[Regression CV] {imputation_name} | {synthetic_model_name} | "
            f"fold {fold_idx}/{total_folds} started"
        )
        real_train_raw = real_base.iloc[train_idx].reset_index(drop=True)
        real_test_raw = real_base.iloc[test_idx].reset_index(drop=True)

        real_train, real_test = _impute_fold_train_test(
            train_df=real_train_raw,
            test_df=real_test_raw,
            metadata=metadata,
            imputation_name=imputation_name,
            target_column=target_column,
            random_state=random_state
        )
        real_train = _apply_reconstruction_rules(real_train, reconstruction_rules)
        real_test = _apply_reconstruction_rules(real_test, reconstruction_rules)
        fold_constraints = _build_fold_constraints(real_train, constraints)
        if use_full_synthetic_target:
            fold_target_size = synthetic_target_size
        else:
            fold_target_size = max(1, int(np.ceil(len(real_train) * synthetic_train_ratio)))
            fold_target_size = min(fold_target_size, synthetic_target_size)

        try:
            _, synthetic_fold = generate_valid_samples_adaptive(
                model_func=model_func,
                imp_df=real_train,
                real_reference_df=real_train,
                target_size=fold_target_size,
                prefix=f'{imputation_name}_regfold{fold_idx}',
                metadata=metadata,
                constraints=fold_constraints,
                ros_target_column=None,
                rejection_thresholds=rejection_thresholds,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                max_total_generated_multiplier=max_total_generated_multiplier,
                max_failed_attempts=max_failed_attempts,
                acceptance_rate_floor=acceptance_rate_floor,
                verbose=False
            )
        except Exception as exc:
            print(
                f"[Regression CV] {imputation_name} | {synthetic_model_name} | "
                f"fold {fold_idx}/{total_folds} skipped during synthetic generation: {exc}"
            )
            continue

        synthetic_fold = _apply_reconstruction_rules(synthetic_fold, reconstruction_rules)

        if target_column not in synthetic_fold.columns:
            continue

        augmented_train = pd.concat([real_train, synthetic_fold], ignore_index=True)

        # --- Common test set ---
        y_test = pd.to_numeric(real_test[target_column], errors='coerce')
        X_test_raw = real_test.drop(columns=[target_column])
        test_mask = y_test.notna()
        X_test_raw = X_test_raw.loc[test_mask].reset_index(drop=True)
        y_test = y_test.loc[test_mask].reset_index(drop=True)

        if len(X_test_raw) == 0:
            continue

        # --- Augmented training set (real + synthetic) ---
        y_train_aug = pd.to_numeric(augmented_train[target_column], errors='coerce')
        X_train_aug_raw = augmented_train.drop(columns=[target_column])
        aug_mask = y_train_aug.notna()
        X_train_aug_raw = X_train_aug_raw.loc[aug_mask].reset_index(drop=True)
        y_train_aug = y_train_aug.loc[aug_mask].reset_index(drop=True)

        # --- Real-only training set (baseline) ---
        y_train_real = pd.to_numeric(real_train[target_column], errors='coerce')
        X_train_real_raw = real_train.drop(columns=[target_column])
        real_mask = y_train_real.notna()
        X_train_real_raw = X_train_real_raw.loc[real_mask].reset_index(drop=True)
        y_train_real = y_train_real.loc[real_mask].reset_index(drop=True)

        if len(X_train_aug_raw) < 2 or len(X_train_real_raw) < 2:
            continue

        X_train_aug, X_test_aug = _prepare_regression_features(X_train_aug_raw, X_test_raw.copy())
        X_train_real, X_test_real = _prepare_regression_features(X_train_real_raw, X_test_raw.copy())

        training_scenarios = [
            ('FoldLocal_TrainAugmented_TestReal',
             f'{imputation_name}_{synthetic_model_name}_augmented',
             X_train_aug, X_test_aug, y_train_aug),
            ('Real_Only_TestReal',
             f'{imputation_name}_real_only',
             X_train_real, X_test_real, y_train_real),
        ]

        for eval_label, dataset_label, X_tr, X_te, y_tr in training_scenarios:
            models = {
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective='reg:squarederror',
                    random_state=random_state,
                    n_jobs=-1
                ),
                'RandomForest': RandomForestRegressor(
                    n_estimators=400,
                    random_state=random_state,
                    n_jobs=-1
                )
            }

            for regressor_name, model in models.items():
                model.fit(X_tr, y_tr)
                predictions = model.predict(X_te)

                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)
                denominator = np.clip(np.abs(y_test.values), 1e-8, None)
                mape = float(np.mean(np.abs((y_test.values - predictions) / denominator)) * 100)

                fold_rows.append({
                    'Dataset': dataset_label,
                    'Imputation': imputation_name,
                    'Synthetic_Model': synthetic_model_name,
                    'Evaluation': eval_label,
                    'Regressor': regressor_name,
                    'Fold': fold_idx,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape
                })

        print(
            f"[Regression CV] {imputation_name} | {synthetic_model_name} | "
            f"fold {fold_idx}/{total_folds} completed"
        )

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_df = (
        fold_df
        .groupby(['Dataset', 'Imputation', 'Synthetic_Model', 'Evaluation', 'Regressor'], as_index=False)
        .agg(
            mae_mean=('mae', 'mean'),
            mae_std=('mae', 'std'),
            rmse_mean=('rmse', 'mean'),
            rmse_std=('rmse', 'std'),
            r2_mean=('r2', 'mean'),
            r2_std=('r2', 'std'),
            mape_mean=('mape', 'mean'),
            mape_std=('mape', 'std'),
            folds=('Fold', 'count')
        )
    )

    return summary_df, fold_df


def save_metric_comparison_plots(results_df: pd.DataFrame, output_dir: str = 'plots', model_name: str = 'cart'):
    if results_df.empty:
        return []

    required_columns = {'Imputation', 'Generator'}
    if not required_columns.issubset(results_df.columns):
        return []

    model_rows = results_df[results_df['Generator'].str.lower() == model_name.lower()].copy()
    if model_rows.empty:
        return []

    os.makedirs(output_dir, exist_ok=True)

    excluded_columns = {'Model', 'Imputation', 'Generator'}
    metric_columns = [
        column for column in model_rows.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(model_rows[column])
    ]

    saved_files = []
    for metric_name in metric_columns:
        metric_df = model_rows[['Imputation', metric_name]].dropna()
        if metric_df.empty:
            continue

        agg_df = metric_df.groupby('Imputation', as_index=False)[metric_name].mean()
        if agg_df.shape[0] < 2:
            continue

        color_map = plt.get_cmap('tab20')
        technique_colors = [color_map(i % color_map.N) for i in range(len(agg_df))]

        plt.figure(figsize=(9, 5))
        plt.bar(agg_df['Imputation'], agg_df[metric_name], color=technique_colors)
        plt.title(f'{model_name.upper()} comparison by imputation: {metric_name}')
        plt.xlabel('Imputation Technique')
        plt.ylabel(metric_name)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        safe_metric = ''.join(ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in metric_name)
        output_path = os.path.join(output_dir, f'{model_name.lower()}_imputation_compare_{safe_metric}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        saved_files.append(output_path)

    return saved_files


def _prepare_regression_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.copy()
    X_test = test_df.copy()

    for column in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[column]):
            fill_value = X_train[column].median()
            X_train[column] = X_train[column].fillna(fill_value)
            X_test[column] = X_test[column].fillna(fill_value)
        else:
            mode_vals = X_train[column].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else 'Missing'
            X_train[column] = X_train[column].fillna(fill_value).astype(str)
            X_test[column] = X_test[column].fillna(fill_value).astype(str)

    X_train = pd.get_dummies(X_train, dtype=int)
    X_test = pd.get_dummies(X_test, dtype=int)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    return X_train, X_test


def run_repeated_regression_cv(
    df: pd.DataFrame,
    target_column: str,
    dataset_name: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = DEFAULT_RANDOM_SEED
):
    if target_column not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    work_df = df.copy()
    y = pd.to_numeric(work_df[target_column], errors='coerce')
    X = work_df.drop(columns=[target_column])

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if len(X) < n_splits:
        return pd.DataFrame(), pd.DataFrame()

    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train_raw = X.iloc[train_idx].copy()
        X_test_raw = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train, X_test = _prepare_regression_features(X_train_raw, X_test_raw)

        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='reg:squarederror',
                random_state=random_state,
                n_jobs=-1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1
            )
        }

        for regressor_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            denominator = np.clip(np.abs(y_test.values), 1e-8, None)
            mape = float(np.mean(np.abs((y_test.values - predictions) / denominator)) * 100)

            fold_rows.append({
                'Dataset': dataset_name,
                'Regressor': regressor_name,
                'Fold': fold_idx,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            })

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_df = (
        fold_df
        .groupby(['Dataset', 'Regressor'], as_index=False)
        .agg(
            mae_mean=('mae', 'mean'),
            mae_std=('mae', 'std'),
            rmse_mean=('rmse', 'mean'),
            rmse_std=('rmse', 'std'),
            r2_mean=('r2', 'mean'),
            r2_std=('r2', 'std'),
            mape_mean=('mape', 'mean'),
            mape_std=('mape', 'std'),
            folds=('Fold', 'count')
        )
    )

    return summary_df, fold_df


def run_repeated_xgb_regression_cv(
    df: pd.DataFrame,
    target_column: str,
    dataset_name: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 47
):
    return run_repeated_regression_cv(
        df=df,
        target_column=target_column,
        dataset_name=dataset_name,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )


def save_regression_metric_plots(regression_summary_df: pd.DataFrame, output_dir: str = 'plots'):
    required_columns = {'Dataset', 'Regressor'}
    if regression_summary_df.empty or not required_columns.issubset(regression_summary_df.columns):
        return []

    os.makedirs(output_dir, exist_ok=True)

    metric_columns = [
        column for column in ['mae_mean', 'rmse_mean', 'r2_mean', 'mape_mean']
        if column in regression_summary_df.columns
    ]
    if not metric_columns:
        return []

    has_evaluation = 'Evaluation' in regression_summary_df.columns
    has_imputation = 'Imputation' in regression_summary_df.columns
    has_synthetic_model = 'Synthetic_Model' in regression_summary_df.columns
    regressors = sorted(regression_summary_df['Regressor'].dropna().unique())
    synthetic_models = sorted(regression_summary_df['Synthetic_Model'].dropna().unique()) if has_synthetic_model else [None]

    # Canonical order: real-only first so it appears left in every group
    eval_order = ['Real_Only_TestReal', 'FoldLocal_TrainAugmented_TestReal']
    eval_labels = {
        'Real_Only_TestReal': 'Real Only',
        'FoldLocal_TrainAugmented_TestReal': 'Augmented (Real + Synthetic)',
    }
    eval_colors = {
        'Real_Only_TestReal': '#aec7e8',
        'FoldLocal_TrainAugmented_TestReal': '#1f77b4',
    }

    saved_files = []

    # Create separate plots for each synthetic model
    for synth_model in synthetic_models:
        if synth_model is not None:
            filtered_df = regression_summary_df[regression_summary_df['Synthetic_Model'] == synth_model].copy()
            plot_title_suffix = f' ({synth_model})'
            plot_file_prefix = f'{synth_model.lower()}_'
        else:
            filtered_df = regression_summary_df.copy()
            plot_title_suffix = ''
            plot_file_prefix = ''

        if filtered_df.empty:
            continue

        for metric_name in metric_columns:
            n_regressors = len(regressors)
            fig, axes = plt.subplots(1, n_regressors, figsize=(7 * n_regressors, 6), sharey=False)
            if n_regressors == 1:
                axes = [axes]

            for ax, regressor in zip(axes, regressors):
                reg_df = filtered_df[
                    filtered_df['Regressor'] == regressor
                ].copy()

                if reg_df.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(regressor, fontsize=12, fontweight='bold')
                    continue

                if has_evaluation and has_imputation:
                    imputations = sorted(reg_df['Imputation'].dropna().unique())
                    present_evals = [e for e in eval_order if e in reg_df['Evaluation'].values]
                    if not present_evals:
                        present_evals = sorted(reg_df['Evaluation'].dropna().unique())

                    x = np.arange(len(imputations))
                    n_evals = len(present_evals)
                    bar_width = 0.7 / n_evals

                    for i, eval_name in enumerate(present_evals):
                        eval_df = reg_df[reg_df['Evaluation'] == eval_name]
                        values = [
                            eval_df.loc[eval_df['Imputation'] == imp, metric_name].mean()
                            if imp in eval_df['Imputation'].values else 0.0
                            for imp in imputations
                        ]
                        offset = (i - n_evals / 2 + 0.5) * bar_width
                        ax.bar(
                            x + offset, values, bar_width,
                            label=eval_labels.get(eval_name, eval_name),
                            color=eval_colors.get(eval_name, '#7f7f7f')
                        )

                    ax.set_xticks(x)
                    ax.set_xticklabels(imputations, fontsize=10)
                else:
                    agg = reg_df[['Dataset', metric_name]].dropna()
                    if not agg.empty:
                        ax.bar(agg['Dataset'], agg[metric_name], color='#1f77b4')
                        ax.set_xticklabels(agg['Dataset'], rotation=30, ha='right', fontsize=9)

                ax.set_title(regressor, fontsize=12, fontweight='bold')
                ax.set_xlabel('Imputation')
                ax.set_ylabel(metric_name)
                ax.legend(fontsize=9)
                ax.grid(axis='y', linestyle='--', alpha=0.4)

            fig.suptitle(f'Regression Comparison: {metric_name}{plot_title_suffix}', fontsize=13, fontweight='bold')
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'regression_compare_{plot_file_prefix}{metric_name}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            saved_files.append(output_path)

    return saved_files


def evaluate_all(name,real_df: pd.DataFrame, outputs, metadata, constraints):
    results_list = []
    for key, synth_df in outputs.items():
        metrics = eval(real_df, synth_df, metadata, constraints)
        row = summarize_metrics(metrics)
        row['Model'] = key
        row['Imputation'] = name
        row['Generator'] = key.split('_')[0] if '_' in key else key
        results_list.append(row)
        Plot(f'plots/{name}{key}',real_df, synth_df, metadata)

    return results_list
