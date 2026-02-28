import pandas as pd
import numpy as np

from scipy.stats import chi2
import numpy as np
import pandas as pd
from scipy.stats import chi2

import numpy as np
import pandas as pd
from scipy.stats import chi2

def drop_covariance_violators(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict, threshold_percentile=0.9999999999) -> pd.DataFrame:
    """
    Drops synthetic rows that fall outside the statistical Mahalanobis boundary of the real data.
    Uses a relaxed threshold to prevent massive rejection rates on high-dimensional small datasets.
    """
    # 1. Identify purely numerical columns present in BOTH dataframes
    # (This prevents crashes if deterministic columns were dropped from real_df)
    num_cols = [col for col, dtype in metadata.items() 
                if dtype in ['numerical', 'integer', 'float'] 
                and col in real_df.columns 
                and col in synth_df.columns]
                
    if len(num_cols) < 2:
        return synth_df
        
    real_num = real_df[num_cols].to_numpy(dtype=float)
    synth_num = synth_df[num_cols].to_numpy(dtype=float)
    
    # 2. Calculate Real Mean and Covariance
    mean_real = np.mean(real_num, axis=0)
    cov_real = np.cov(real_num, rowvar=False)
    
    try:
        # Use pseudo-inverse to handle degenerate dimensions
        inv_cov_real = np.linalg.pinv(cov_real)
    except np.linalg.LinAlgError:
        return synth_df
        
    # 3. Calculate Mahalanobis Distance for every synthetic row
    diff = synth_num - mean_real
    
    # Fast vectorized Mahalanobis calculation: diag( (x-\mu) @ \Sigma^{-1} @ (x-\mu)^T )
    left_term = np.dot(diff, inv_cov_real)
    mahal_distances = np.sum(left_term * diff, axis=1)
    
    # 4. Set the Boundary Threshold
    # Using the Chi-Square distribution to find the cutoff distance.
    # We use 0.999 to be highly forgiving, allowing the GANs to explore new structural combinations
    threshold = chi2.ppf(threshold_percentile, df=len(num_cols))
    
    # 5. Keep only the rows inside the boundary
    valid_mask = mahal_distances <= threshold
    
    return synth_df[valid_mask].copy()

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
    temp_df = df.copy()
    temp_df.columns = [f'c{i}' for i in range(len(df.columns))]
    valid_mask = pd.Series(True, index=df.index)
    
    for con in constraints:
        try:
            # df.eval parses the string and evaluates it in C, returning a boolean array
            condition_mask = temp_df.eval(con)
            # Use bitwise AND to keep only rows that satisfy ALL constraints
            valid_mask = valid_mask & condition_mask
            
        except Exception as e:
            print(f"Warning: Failed to evaluate constraint '{con}'. Error: {e}")
            
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