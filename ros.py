import pandas as pd
import numpy as np

from scipy.stats import chi2
import numpy as np
import pandas as pd
from scipy.stats import chi2

def drop_covariance_violators(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict, confidence_level=0.99) -> pd.DataFrame:
    """
    Drops synthetic rows that violate the multivariate covariance structure of the real data.
    Uses Mahalanobis distance and Chi-Square distribution bounds.
    """
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    if not num_cols:
        return synth_df
        
    # 1. Get the "Ground Truth" mean and covariance from the REAL data
    real_num = real_df[num_cols].dropna()
    
    # Keep the mean as a Pandas Series (with column names) for fillna
    mean_series = real_num.mean() 
    # Extract the raw numbers for matrix math
    mean_array = mean_series.values 
    
    cov_real = np.cov(real_num, rowvar=False)
    
    # Calculate the inverse of the covariance matrix
    try:
        inv_cov_real = np.linalg.inv(cov_real)
    except np.linalg.LinAlgError:
        print("Warning: Real covariance matrix is singular. Cannot apply Mahalanobis filter.")
        return synth_df
        
    # 2. Calculate the threshold (Chi-Square critical value)
    degrees_of_freedom = len(num_cols)
    threshold = chi2.ppf(confidence_level, degrees_of_freedom)
    
    # 3. Check every SYNTHETIC row
    # Use the Pandas Series so fillna() knows which column gets which mean
    synth_num = synth_df[num_cols].fillna(mean_series) 
    valid_mask = []
    
    for index, row in synth_num.iterrows():
        # Difference between synthetic point and real center (using raw array)
        diff = row.values - mean_array 
        
        # Mahalanobis Distance Formula
        mahalanobis_sq = np.dot(np.dot(diff.T, inv_cov_real), diff)
        
        # Keep row if it falls inside the physical threshold
        valid_mask.append(mahalanobis_sq <= threshold)
        
    filtered_synth_df = synth_df[valid_mask].copy()
    
    dropped_count = len(synth_df) - len(filtered_synth_df)
    
    return filtered_synth_df

def get_constraints(real_df: pd.DataFrame):
    """Extract min/max constraints for numerical columns from real data."""
    constraints = []
    for i, col in enumerate(real_df.columns):
        if pd.api.types.is_numeric_dtype(real_df[col].dtype):
            constraints.append(f'c{i} >= {real_df[col].min()}')
            constraints.append(f'c{i} <= {real_df[col].max()}')
    return constraints

def add_constraints(real_df: pd.DataFrame):
    """Interactive loop to add custom constraints."""
    con = get_constraints(real_df)
    print("\nAdd constraints as an expression with column numbers (e.g., c0 > c1 * 1.5)")
    while True:
        ent = input("Enter constraint (or 'q' to exit): ").strip()
        if ent.lower() == 'q':
            break
        if ent:
            con.append(ent)
    return con

def drop_constraint_violators(df: pd.DataFrame, constraints: list):
    """
    Remove synthetic rows that violate constraints using Vectorized Pandas Eval.
    """
    # 1. Map 'c0', 'c1' to actual column names wrapped in backticks 
    # Backticks are required in Pandas if your column names have spaces (e.g. `span length`)
    col_map = {f'c{i}': f'`{col}`' for i, col in enumerate(df.columns)}
    # 2. Sort keys by length descending so 'c10' is replaced before 'c1'
    sorted_keys = sorted(col_map.keys(), key=len, reverse=True)    
    valid_mask = pd.Series(True, index=df.index)
    
    for con in constraints:
        parsed_con = con
        # Replace 'c#' with actual column names
        for key in sorted_keys:
            parsed_con = parsed_con.replace(key, col_map[key])
            
        try:
            # df.eval parses the string and evaluates it in C, returning a boolean array
            condition_mask = df.eval(parsed_con)
            # Use bitwise AND to keep only rows that satisfy ALL constraints
            valid_mask = valid_mask & condition_mask
            
        except Exception as e:
            print(f"Warning: Failed to evaluate constraint '{con}' -> '{parsed_con}'. Error: {e}")
            
    filtered_df = df[valid_mask].copy()    
    return filtered_df

def apply_ros(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Actual Random Oversampling (ROS). 
    Duplicates minority class rows until they match the majority class count.
    """
    if target_col not in df.columns:
        print(f"Column '{target_col}' not found. Skipping ROS.")
        return df
    max_size = df[target_col].value_counts().max()
    balanced_dfs = []
    for class_name, group in df.groupby(target_col):
        # Sample with replacement to bring this category up to max_size
        oversampled_group = group.sample(n=max_size, replace=True, random_state=42)
        balanced_dfs.append(oversampled_group)
        
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

import pandas as pd

def apply_multi_col_ros(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Applies Random Oversampling across one or multiple categorical columns 
    by creating a combined cross-distribution target.
    """
    df_ros = df.copy()
    
    # 1. Automatically identify all categorical columns from metadata
    cat_cols = [col for col, dtype in metadata.items() if dtype == 'categorical' and col in df_ros.columns]
    
    if not cat_cols:
        print("No categorical columns found for ROS.")
        return df_ros
        
    print(f"Applying ROS on columns: {cat_cols}")
    
    # 2. If only ONE categorical column, do standard ROS
    if len(cat_cols) == 1:
        target_col = cat_cols[0]
        max_size = df_ros[target_col].value_counts().max()
        balanced_dfs = []
        for class_name, group in df_ros.groupby(target_col):
            oversampled_group = group.sample(n=max_size, replace=True, random_state=42)
            balanced_dfs.append(oversampled_group)
            
        return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        
    # 3. If MULTIPLE categorical columns, create a "Composite Class"
    # Example: 'Square' + '_' + 'Crushing' -> 'Square_Crushing'
    temp_target = '_ros_combined_class'
    
    # Convert all categorical columns to strings and join them with an underscore
    df_ros[temp_target] = df_ros[cat_cols].astype(str).agg('_'.join, axis=1)
    
    print(f"Original composite distribution:\n{df_ros[temp_target].value_counts()}")
    
    # Find the maximum count among all existing combinations
    max_size = df_ros[temp_target].value_counts().max()
    
    balanced_dfs = []
    
    # Balance based on the composite combinations
    for class_name, group in df_ros.groupby(temp_target):
        oversampled_group = group.sample(n=max_size, replace=True, random_state=42)
        balanced_dfs.append(oversampled_group)
        
    # Combine, shuffle, and drop the temporary composite column
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df = balanced_df.drop(columns=[temp_target])
    
    print(f"\nNew dataset size after Multi-Column ROS: {len(balanced_df)} rows (Balanced across all combinations)")
    return balanced_df