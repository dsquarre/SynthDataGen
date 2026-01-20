# metrics.py

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, iqr

# ------------------------------------------------------------------------------
# Coverage Metrics
# ------------------------------------------------------------------------------

def category_coverage(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Measure the proportion of categories present in the real data that are
    also present in the synthetic data.

    Args:
        real (pd.Series): Real (categorical) data.
        synthetic (pd.Series): Synthetic (categorical) data.

    Returns:
        float: Ratio (0 to 1) of real categories that are found in the synthetic data.
    """
    real_cats = set(real.dropna().unique())
    synth_cats = set(synthetic.dropna().unique())
    if not real_cats:
        return 1.0
    return len(real_cats.intersection(synth_cats)) / len(real_cats)


def range_coverage(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Measure the proportion of the real data's numerical range that is covered by the
    synthetic data. If the data is datetime or timedelta, convert it to seconds.
    
    Args:
        real (pd.Series): Real numerical data.
        synthetic (pd.Series): Synthetic numerical data.
    
    Returns:
        float: The ratio of the intersection length of the ranges to the real range.
    """
    real_min, real_max = real.min(), real.max()
    synth_min, synth_max = synthetic.min(), synthetic.max()
    
    # If the data is datetime, convert to seconds since epoch.
    if isinstance(real_min, pd.Timestamp):
        real_min = real_min.value / 1e9  # convert nanoseconds to seconds
        real_max = real_max.value / 1e9
        synth_min = synth_min.value / 1e9
        synth_max = synth_max.value / 1e9
    # If the data is timedelta, convert to total seconds.
    elif isinstance(real_min, pd.Timedelta):
        real_min = real_min.total_seconds()
        real_max = real_max.total_seconds()
        synth_min = synth_min.total_seconds()
        synth_max = synth_max.total_seconds()
    
    if real_max == real_min:
        return 1.0
    intersection = max(0, min(real_max, synth_max) - max(real_min, synth_min))
    return intersection / (real_max - real_min)


# ------------------------------------------------------------------------------
# Adherence Metrics
# ------------------------------------------------------------------------------

def boundary_adherence(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Measure the fraction of synthetic numerical values that lie within the boundaries
    of the real data.

    Args:
        real (pd.Series): Real numerical data.
        synthetic (pd.Series): Synthetic numerical data.

    Returns:
        float: The fraction (0 to 1) of synthetic values within [real_min, real_max].
    """
    real_min, real_max = real.min(), real.max()
    adherence = ((synthetic >= real_min) & (synthetic <= real_max)).mean()
    return adherence


def category_adherence(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Measure the fraction of synthetic categorical values that are present in the set
    of real categories.

    Args:
        real (pd.Series): Real categorical data.
        synthetic (pd.Series): Synthetic categorical data.

    Returns:
        float: The fraction (0 to 1) of synthetic values that are among the real categories.
    """
    real_cats = set(real.dropna().unique())
    if not real_cats:
        return 1.0
    adherence = synthetic.dropna().apply(lambda x: x in real_cats).mean()
    return adherence

# ------------------------------------------------------------------------------
# Distribution/Shape Comparison Metrics
# ------------------------------------------------------------------------------

def ks_complement(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Compute the complement of the Kolmogorov-Smirnov statistic comparing the
    real and synthetic data distributions.

    Args:
        real (pd.Series): Real numerical data.
        synthetic (pd.Series): Synthetic numerical data.

    Returns:
        float: 1 - KS statistic (ranges between 0 and 1, where 1 means identical distributions).
    """
    real_clean = real.dropna()
    synthetic_clean = synthetic.dropna()
    if len(real_clean) == 0 or len(synthetic_clean) == 0:
        return 0.0
    ks_stat, _ = ks_2samp(real_clean, synthetic_clean)
    return 1 - ks_stat


def tv_complement(real_series: pd.Series, synthetic_series: pd.Series) -> float:
    """
    Computes the TVComplement score between a real and a synthetic categorical column.
    
    TVD is defined as:
        TVD = 1/2 * sum(|R_ω - S_ω|) for all categories ω in the union of both series.
        
    The TVComplement score is:
        score = 1 - TVD
        
    Parameters
    ----------
    real_series : pd.Series
        Categorical data from the real dataset.
    synthetic_series : pd.Series
        Categorical data from the synthetic dataset.
    
    Returns
    -------
    float
        The TVComplement score (between 0 and 1).
    """
    # Compute normalized frequency distributions (probabilities)
    real_freq = real_series.value_counts(normalize=True)
    synthetic_freq = synthetic_series.value_counts(normalize=True)
    
    # Get the union of categories present in both series
    all_categories = real_freq.index.union(synthetic_freq.index)
    
    # Reindex to ensure both distributions have the same categories, fill missing with 0
    real_freq = real_freq.reindex(all_categories, fill_value=0)
    synthetic_freq = synthetic_freq.reindex(all_categories, fill_value=0)
    
    # Calculate Total Variation Distance (TVD)
    tvd = 0.5 * np.abs(real_freq - synthetic_freq).sum()
    
    # Compute TVComplement: higher score means higher similarity
    tv_complement_score = 1 - tvd
    
    return tv_complement_score


# ------------------------------------------------------------------------------
# Statistical Similarity Metrics
# ------------------------------------------------------------------------------

def statistic_similarity(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Compare basic statistics (mean, standard deviation, and median) of the real and
    synthetic data and return an average similarity score between 0 and 1 (1 means perfect similarity).
    
    If the data is datetime or timedelta, it is converted to a numeric representation (seconds).
    
    Args:
        real (pd.Series): Real data.
        synthetic (pd.Series): Synthetic data.
    
    Returns:
        float: Similarity score between 0 and 1.
    """
    real_clean = real.dropna()
    synthetic_clean = synthetic.dropna()
    if len(real_clean) == 0 or len(synthetic_clean) == 0:
        return 0.0

    eps = 1e-8  # small constant to avoid division by zero
    
    # Convert datetime/timedelta to numeric values (in seconds)
    if np.issubdtype(real_clean.dtype, np.datetime64):
        real_vals = real_clean.astype('int64') / 1e9
        synth_vals = synthetic_clean.astype('int64') / 1e9
    elif np.issubdtype(real_clean.dtype, np.timedelta64):
        # Use the .dt accessor if available
        if hasattr(real_clean, 'dt'):
            real_vals = real_clean.dt.total_seconds()
            synth_vals = synthetic_clean.dt.total_seconds()
        else:
            real_vals = real_clean.astype('int64') / 1e9
            synth_vals = synthetic_clean.astype('int64') / 1e9
    else:
        real_vals = real_clean
        synth_vals = synthetic_clean

    stats = ['mean', 'std', 'median']
    real_stats = {
        'mean': real_vals.mean(),
        'std': real_vals.std(),
        'median': real_vals.median()
    }
    synth_stats = {
        'mean': synth_vals.mean(),
        'std': synth_vals.std(),
        'median': synth_vals.median()
    }
    
    similarities = []
    for stat in stats:
        diff = abs(real_stats[stat] - synth_stats[stat])
        denom = abs(real_stats[stat]) + eps
        sim = 1 - (diff / denom)
        sim = max(0, min(1, sim))
        similarities.append(sim)
    return np.mean(similarities)



def missing_value_similarity(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Compare the proportion of missing values (NaNs) in the real and synthetic data.

    Args:
        real (pd.Series): Real data.
        synthetic (pd.Series): Synthetic data.

    Returns:
        float: 1 minus the absolute difference in missing value proportions (ranges from 0 to 1).
    """
    real_missing = real.isna().mean()
    synth_missing = synthetic.isna().mean()
    return 1 - abs(real_missing - synth_missing)

