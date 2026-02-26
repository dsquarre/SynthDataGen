"""Random Oversampling (ROS) with constraint validation for synthetic data augmentation."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def get_numerical_constraints(real_df: pd.DataFrame) -> Dict[str, tuple]:
    """Extract min/max constraints for numerical columns from real data."""
    constraints = {}
    for col in real_df.columns:
        if pd.api.types.is_numeric_dtype(real_df[col].dtype):
            constraints[col] = (real_df[col].min(), real_df[col].max())
    return constraints


def validate_constraints(synth_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Remove synthetic rows that violate numerical constraints from real data.
    
    Returns filtered synthetic DataFrame with only rows within constraint ranges.
    """
    constraints = get_numerical_constraints(real_df)
    mask = pd.Series([True] * len(synth_df))
    
    for col, (min_val, max_val) in constraints.items():
        if col in synth_df.columns:
            col_mask = (synth_df[col] >= min_val) & (synth_df[col] <= max_val)
            mask = mask & col_mask
    
    filtered_df = synth_df[mask].reset_index(drop=True)
    return filtered_df


def apply_custom_constraints(synth_df: pd.DataFrame, custom_constraints: List = None) -> pd.DataFrame:
    """Apply custom user-defined constraints to synthetic data."""
    if not custom_constraints:
        return synth_df
    
    result = synth_df.copy()
    for constraint in custom_constraints:
        before_len = len(result)
        result = constraint.validate(result)
        removed = before_len - len(result)
        if removed > 0:
            print(f"  Custom constraint '{constraint.name}': removed {removed} rows")
    return result


def apply_ros(df: pd.DataFrame, target_column: str = None, sampling_strategy: str = 'auto') -> pd.DataFrame:
    """Apply Random Oversampling using imbalanced-learn.
    
    If target_column is provided and is CATEGORICAL, oversample minority classes.
    If target_column is CONTINUOUS (numerical), use simple random oversampling (50% boost).
    Otherwise, randomly oversample the entire dataset by 50%.
    """
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError:
        raise ImportError("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
    
    if target_column and target_column in df.columns:
        # Check if target is categorical or continuous
        is_continuous = pd.api.types.is_numeric_dtype(df[target_column].dtype)
        
        # For continuous targets or targets with too many unique values, use simple random oversampling
        if is_continuous or df[target_column].nunique() > 20:
            # Simple random oversampling: duplicate 50% of rows
            num_new_rows = int(len(df) * 0.5)
            new_rows = df.sample(n=num_new_rows, replace=True, random_state=42)
            result_df = pd.concat([df, new_rows], ignore_index=True)
            return result_df.reset_index(drop=True)
        else:
            # Use class-based RandomOverSampler for discrete/categorical targets
            X = df.drop(columns=[target_column])
            y = df[target_column]
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_ros, y_ros = ros.fit_resample(X, y)
            result_df = X_ros.copy()
            result_df[target_column] = y_ros
            return result_df.reset_index(drop=True)
    else:
        # Random oversampling: duplicate 50% of rows
        num_new_rows = int(len(df) * 0.5)
        new_rows = df.sample(n=num_new_rows, replace=True, random_state=42)
        result_df = pd.concat([df, new_rows], ignore_index=True)
        return result_df


def apply_ros_with_constraints(synth_df: pd.DataFrame, real_df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """Apply ROS to synthetic data and then validate constraints against real data.
    
    Returns augmented synthetic dataset with only valid rows.
    """
    # Apply ROS
    ros_df = apply_ros(synth_df, target_column=target_column)
    
    # Validate constraints
    validated_df = validate_constraints(ros_df, real_df)
    
    return validated_df


def apply_ros_with_all_constraints(synth_df: pd.DataFrame, real_df: pd.DataFrame, custom_constraints: List = None, target_column: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply ROS and both range + custom constraints to synthetic data.
    
    Returns tuple of (validated_df, metadata_dict) where metadata includes:
    - initial_rows: rows after ROS
    - final_rows: rows after all validations
    - rejection_rate: % of data filtered out
    - passed_failsafe: whether rejection_rate is acceptable
    """
    # Apply ROS
    ros_df = apply_ros(synth_df, target_column=target_column)
    initial_count = len(ros_df)
    
    # Validate range constraints
    validated_df = validate_constraints(ros_df, real_df)
    after_range = len(validated_df)
    
    # Apply custom constraints
    if custom_constraints:
        validated_df = apply_custom_constraints(validated_df, custom_constraints)
    
    final_count = len(validated_df)
    rejection_rate = 1 - (final_count / initial_count) if initial_count > 0 else 0
    passed_failsafe = rejection_rate < 0.5  # Fail if >50% rejected
    
    metadata = {
        'initial_rows': initial_count,
        'final_rows': final_count,
        'rejection_rate': rejection_rate,
        'passed_failsafe': passed_failsafe,
        'rows_filtered': initial_count - final_count
    }
    
    return validated_df, metadata


def regenerate_until_target(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    custom_constraints: List = None,
    target_column: str = None,
    target_size: int = None,
    max_regenerations: int = 5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Regenerate synthetic data via ROS until target size is reached.
    
    If target_size is None, defaults to 2× real data size.
    If most data is rejected, automatically generates more until target is reached.
    
    Args:
        synth_df: Initial synthetic dataset
        real_df: Real dataset (for constraint validation)
        custom_constraints: User-defined constraints
        target_column: Column for ROS balancing
        target_size: Target number of rows (default: 2 × len(real_df))
        max_regenerations: Max retry attempts (default: 5)
        verbose: Print progress messages
    
    Returns:
        Tuple of (final_df, metadata_dict) where metadata includes regeneration history
    """
    if target_size is None:
        target_size = len(real_df) * 2
    
    if verbose:
        print(f"\n  Failsafe Mode: Target size = {target_size} rows")
    
    all_valid_data = []
    regeneration_history = []
    attempt = 0
    total_valid = 0
    
    current_synth = synth_df.copy()
    
    while total_valid < target_size and attempt < max_regenerations:
        attempt += 1
        if verbose:
            print(f"  Attempt {attempt}: Processing {len(current_synth)} rows...")
        
        validated_df, metadata = apply_ros_with_all_constraints(
            current_synth, real_df, custom_constraints, target_column
        )
        
        regeneration_history.append({
            'attempt': attempt,
            'input_rows': len(current_synth),
            'output_rows': len(validated_df),
            'rejection_rate': metadata['rejection_rate']
        })
        
        all_valid_data.append(validated_df)
        total_valid = sum(len(df) for df in all_valid_data)
        
        if verbose:
            print(f"    → {len(validated_df)} valid rows (cumulative: {total_valid}/{target_size})")
        
        # If still below target and have attempts left, generate more
        if total_valid < target_size and attempt < max_regenerations:
            # Generate fresh synthetic data by upsampling
            current_synth = synth_df.sample(n=len(synth_df) * 2, replace=True, random_state=42 + attempt)
    
    # Combine all valid data and trim to target size
    if all_valid_data:
        final_df = pd.concat(all_valid_data, ignore_index=True)
        if len(final_df) > target_size:
            final_df = final_df.sample(n=target_size, random_state=42)
        final_df = final_df.reset_index(drop=True)
    else:
        final_df = pd.DataFrame()
    
    failsafe_metadata = {
        'regeneration_attempts': attempt,
        'max_attempts': max_regenerations,
        'target_size': target_size,
        'final_rows': len(final_df),
        'total_regenerations': len(regeneration_history),
        'regeneration_history': regeneration_history,
        'target_met': len(final_df) >= target_size
    }
    
    if verbose:
        print(f"  Final result: {len(final_df)} rows (target was {target_size})")
        print(f"  Target met: {'✓ YES' if failsafe_metadata['target_met'] else '✗ NO - insufficient valid data'}")
    
    return final_df, failsafe_metadata
