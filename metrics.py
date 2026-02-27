import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
from plot import Plot
warnings.filterwarnings('ignore')

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
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=43)
    gmm.fit(X_real)
    
    # Mean log-likelihood of synthetic data under the real distribution
    return float(np.mean(gmm.score_samples(X_synth)))

def propensity_score(real_df,synth_df,metadata):
    real = real_df.copy()
    fake = synth_df.copy()
    
    real['Target'] = 1
    fake['Target'] = 0
    
    # Align columns to ensure consistency
    common_cols = [c for c in real.columns if c in fake.columns and c != 'Target']
    data = pd.concat([real[common_cols + ['Target']], fake[common_cols + ['Target']]], ignore_index=True)
    
    # Step 2: Preprocess and Split
    X = data.drop(columns=['Target'])
    y = data['Target']
    
    # One-hot encoding for categorical columns
    categorical_cols = [col for col, dtype in metadata.items() if dtype == 'categorical' and col in X.columns]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, dtype=int)
    
    # Split 75/25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)
    
    # Step 3: Train the "Disposable" XGBoost Detective
    # Constraint: max_depth=2 and fewer trees to prevent overfitting/memorization (more lenient)
    model = xgb.XGBClassifier(
        max_depth=2, n_estimators=50, eval_metric='logloss', random_state=1001
    )
    model.fit(X_train, y_train)
    
    # Step 4: Predict and Score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return 1-2*abs(0.5-roc_auc_score(y_test, y_pred_proba))


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

def evaluate_all(name,real_df: pd.DataFrame, outputs, metadata, constraints):
    results_list = []
    for key, synth_df in outputs.items():
        metrics = eval(real_df, synth_df, metadata, constraints)
        row = {'Model': name + key}
        if 'marginals' in metrics:
            row['Marginal'] = np.mean(list(metrics['marginals'].values()))
            
        for k, v in metrics.items():
            if k != 'marginals' and isinstance(v, (int, float)):
                row[k] = v
        results_list.append(row)
        Plot(f'plots/{name}{key}',real_df, synth_df, metadata)

    return results_list
