from metadata_manager import generate_metadata, interactive_edit_metadata, save_as_sdv_json
import pandas as pd
import numpy as np
import os
from synthpop.synth import run_cart
#from sdv_all import run_copula_gan,run_ctgan, run_gaussian_copula, run_tvae
from ctab_gan_plus import ctabganplus
from metrics import (
    evaluate_all,
    run_repeated_augmented_cv,
    run_repeated_augmented_regression_cv,
    save_metric_comparison_plots,
    save_regression_metric_plots,
    generate_valid_samples_adaptive
)
from ros import apply_ros,add_constraints
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, MissingIndicator
from sklearn.ensemble import RandomForestClassifier

GLOBAL_RANDOM_SEED = 2026


DEFAULT_REJECTION_THRESHOLDS = {
    'Marginal': 0.70,
    'boundary_score': 0.99,
    'corr_score': 0.65,
    'cov_score': 0.40,
    'violation_rate': 0.001,
    'propensity_score': 0.15
}

DEFAULT_GENERATION_SETTINGS = {
    'min_batch_size': 500,
    'max_batch_size': 24000,
    'max_total_generated_multiplier': 120,
    'max_failed_attempts': 60,
    'acceptance_rate_floor': 0.01,
    'model_retry_attempts': 3,
    'fold_synthetic_ratio': 1.0,
    'use_full_fold_synthetic_target': False
}


def parse_metric_thresholds(input_text: str, defaults: dict):
    if not input_text.strip():
        return defaults.copy()

    parsed = {}
    for chunk in input_text.split(','):
        entry = chunk.strip()
        if not entry:
            continue

        if '=' not in entry:
            print(f"Skipping malformed threshold entry: {entry}")
            continue

        metric_name, raw_value = entry.split('=', 1)
        metric_name = metric_name.strip()
        raw_value = raw_value.strip()

        try:
            parsed[metric_name] = float(raw_value)
        except ValueError:
            print(f"Skipping non-numeric threshold: {entry}")

    if not parsed:
        return defaults.copy()

    merged = defaults.copy()
    merged.update(parsed)
    return merged

def impute_categorical_rf(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df_imputed = df.copy()
    
    numeric_cols = [col for col, dtype in metadata.items() if dtype == 'numerical' and col in df_imputed.columns]
    categorical_cols = [col for col, dtype in metadata.items() if dtype == 'categorical' and col in df_imputed.columns]
    
    if not numeric_cols:
        return df_imputed

    for target_col in categorical_cols:
        known_data = df_imputed[df_imputed[target_col].notna()]
        missing_data = df_imputed[df_imputed[target_col].isna()]
        
        if missing_data.empty:
            continue
            
        X_train = known_data[numeric_cols].copy()
        y_train = known_data[target_col].copy()
        X_missing = missing_data[numeric_cols].copy()
        
        fill_values = X_train.median()
        X_train = X_train.fillna(fill_values)
        X_missing = X_missing.fillna(fill_values)

        model = RandomForestClassifier(n_estimators=100, random_state=GLOBAL_RANDOM_SEED)
        model.fit(X_train, y_train)
        
        predicted_categories = model.predict(X_missing)
        df_imputed.loc[df_imputed[target_col].isna(), target_col] = predicted_categories
        
    return df_imputed

def encode_categorical(df, metadata):
    df_encoded = df.copy()
    encoders = {}
    for col, dtype in metadata.items():
        if dtype == 'categorical' and col in df_encoded.columns:
            series = df_encoded[col].astype('category')
            encoders[col] = series.cat.categories
            df_encoded[col] = series.cat.codes.replace(-1, np.nan)
    return df_encoded, encoders

def decode_categorical(df, encoders):
    df_decoded = df.copy()
    for col, categories in encoders.items():
        if col in df_decoded.columns:
            series = df_decoded[col].round().fillna(-1)
            vals = series.values.copy()
            valid_mask = vals >= 0
            vals[valid_mask] = np.clip(vals[valid_mask], 0, len(categories) - 1)
            df_decoded[col] = pd.Categorical.from_codes(vals.astype(int), categories=categories)
            df_decoded[col] = df_decoded[col].astype(object)
    return df_decoded

def knn_imputation(df: pd.DataFrame,metadata ) -> pd.DataFrame:
    """Impute missing values using KNN."""
    df_encoded, encoders = encode_categorical(df, metadata)
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(data_imputed, columns=df.columns, index=df.index)
    return decode_categorical(df_imputed, encoders)

def missing_indicator_imputation(df: pd.DataFrame) -> pd.DataFrame:
	"""Impute missing using MissingIndicator."""
    
	indicator = MissingIndicator(missing_values=np.nan)
	indicator.fit(df)
	indicator_cols = indicator.get_feature_names_out()
	df_indicator = pd.DataFrame(indicator.transform(df), columns=indicator_cols, index=df.index)	
	df_filled = df.copy()
	for col in df_filled.columns:
		if pd.api.types.is_numeric_dtype(df_filled[col]):
			df_filled[col] = df_filled[col].fillna(df_filled[col].median())
		else:
			if not df_filled[col].mode().empty:
				df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])

	df_imputed = pd.concat([df_filled, df_indicator], axis=1)
	return df_imputed

def iterative_imputation(df: pd.DataFrame,metadata) -> pd.DataFrame:
    """Impute missing using IterativeImputer."""
    df_encoded, encoders = encode_categorical(df, metadata)
    imputer = IterativeImputer(random_state=GLOBAL_RANDOM_SEED)
    data_imputed = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(data_imputed, columns=df.columns, index=df.index)
    return decode_categorical(df_imputed, encoders)

def generate_valid_samples(
    model_func,
    imp_df,
    real_reference_df,
    target_size,
    prefix,
    metadata,
    constraints,
    target_column,
    rejection_thresholds=None,
    max_iterations=8,
    generation_settings=None
):
    settings = DEFAULT_GENERATION_SETTINGS.copy()
    if generation_settings:
        settings.update(generation_settings)

    return generate_valid_samples_adaptive(
        model_func=model_func,
        imp_df=imp_df,
        real_reference_df=real_reference_df,
        target_size=target_size,
        prefix=prefix,
        metadata=metadata,
        constraints=constraints,
        ros_target_column=target_column,
        rejection_thresholds=rejection_thresholds,
        min_batch_size=settings['min_batch_size'],
        max_batch_size=settings['max_batch_size'],
        max_total_generated_multiplier=settings['max_total_generated_multiplier'],
        max_failed_attempts=max(settings['max_failed_attempts'], max_iterations),
        acceptance_rate_floor=settings['acceptance_rate_floor'],
        verbose=True
    )


def main():
    # Ensure output directories exist
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    filename = input("Enter dataset filename in datasets/ folder (default: data.csv): ").strip()
    if not filename:
        filename = 'data.csv'

    filepath = os.path.join('datasets', filename)
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    print(f"Loading {filename}...")
    if filename.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    try:
        df.drop(columns=df.columns[0], inplace=True)
    except Exception:
        pass
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    df.columns = df.columns.str.strip()
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    # Remove leading underscores
    df.columns = [col.lstrip('_') for col in df.columns]
    # Remove trailing underscores
    df.columns = [col.rstrip('_') for col in df.columns]

    drop_indices = input("\nEnter column indices to drop (comma-separated) or press Enter to continue: ").strip()
    if drop_indices:
        try:
            indices = [int(x.strip()) for x in drop_indices.split(',') if x.strip()]
            cols_to_drop = [df.columns[i] for i in indices if 0 <= i < len(df.columns)]
            if cols_to_drop:
                print(f"Dropping: {cols_to_drop}")
                df.drop(columns=cols_to_drop, inplace=True)
        except ValueError:
            print("Invalid input. No columns dropped.")

    Dn = max(len(df), 500)
    print(f"Target synthetic dataset size: {Dn}")
    
    print("Generating metadata...")
    metadata = generate_metadata(df)

    print("You can edit column types now. Use index and 'n' or 'c' (e.g. '3 n'). Type 'done' when finished.")
    metadata = interactive_edit_metadata(metadata)
    ros_index = int(input("Enter ROS target column index (or '-1' to skip ROS/classification evaluation): "))
    target_column = df.columns[ros_index] if ros_index != -1 else None
    regression_index = int(input("Enter regression output column index for regression models (or '-1' to skip regression): "))
    regression_target_column = df.columns[regression_index] if regression_index != -1 else None
    print("Saving SDV-compatible metadata to sdv_metadata.json")
    save_as_sdv_json(metadata, path='sdv_metadata.json')

    raw_df_for_cv = df.copy()
    df = impute_categorical_rf(df, metadata)
    #print(df.head())
    imputations = {
        'iter': iterative_imputation(df, metadata), 
        'knn': knn_imputation(df, metadata), 
        'mi': missing_indicator_imputation(df)
    }
    
    constraints = add_constraints(df)

    print("\nDefault rejection thresholds:")
    for metric_name, threshold in DEFAULT_REJECTION_THRESHOLDS.items():
        print(f"  {metric_name} = {threshold}")

    threshold_input = input(
        "Enter custom thresholds as metric=value pairs (comma-separated), or press Enter for defaults: "
    ).strip()
    rejection_thresholds = parse_metric_thresholds(threshold_input, DEFAULT_REJECTION_THRESHOLDS)
    regression_rejection_thresholds = {
        metric: threshold
        for metric, threshold in rejection_thresholds.items()
        if metric != 'propensity_score'
    }
    print(f"Using rejection thresholds: {rejection_thresholds}")
    print(f"Using regression thresholds (privacy removed): {regression_rejection_thresholds}")

    repeats_input = input("Repeated 5-fold CV repeats (default: 3): ").strip()
    try:
        cv_repeats = int(repeats_input) if repeats_input else 3
    except ValueError:
        cv_repeats = 3

    fold_target_mode_input = input(
        "Fold synthetic size mode for CV ('1' = 1:1 with train fold, '2' = full Dn per fold, default: 1): "
    ).strip().lower()
    use_full_fold_synthetic_target = fold_target_mode_input in {'2', 'full', 'dn', 'all'}
    fold_mode_label = 'full Dn per fold' if use_full_fold_synthetic_target else '1:1 with train fold'
    print(f"Using fold synthetic sizing mode: {fold_mode_label}")

    all_results = []
    cv_summary_frames = []
    cv_fold_frames = []
    regression_summary_frames = []
    regression_fold_frames = []
    for impute_name, imputed_df in imputations.items():
        #print(imp_df.head())
        outputs = {}
        generation_settings = DEFAULT_GENERATION_SETTINGS.copy()
        generation_settings['use_full_fold_synthetic_target'] = use_full_fold_synthetic_target
        imp_df_real = imputed_df.copy()
        imp_df_for_generation = apply_ros(imp_df_real, target_column) if target_column else imp_df_real.copy()
            
        # Update metadata for this imputation (e.g. if MI added columns)
        current_metadata = metadata.copy()
        for col in imp_df_for_generation.columns:
            if col not in current_metadata:
                current_metadata[col] = 'categorical'
        
        save_as_sdv_json(current_metadata, path='sdv_metadata.json')
        #print(imp_df.head())
        models = {
            'CART': lambda d, n, pfx, persist=True: run_cart(d, n, pfx, current_metadata, persist=persist),
            #'Gaussian Copula': run_gaussian_copula,
            #'CTGAN': run_ctgan,
            #'CopulaGAN': run_copula_gan,
            #'TVAE': run_tvae
        }
        
        for model_display_name, model_func in models.items():
            print(f"[{impute_name}] Running {model_display_name}...")

            max_model_attempts = max(1, int(generation_settings.get('model_retry_attempts', 1)))
            model_generated = False

            for attempt_idx in range(1, max_model_attempts + 1):
                try:
                    generation_func = lambda d, n, pfx, mf=model_func: mf(d, n, pfx, persist=False)
                    name, valid_df = generate_valid_samples(
                        model_func=generation_func,
                        imp_df=imp_df_for_generation,
                        real_reference_df=imp_df_real,
                        target_size=Dn,
                        prefix=f"{impute_name}_attempt{attempt_idx}",
                        metadata=current_metadata,
                        constraints=constraints,
                        target_column=target_column,
                        rejection_thresholds=rejection_thresholds,
                        generation_settings=generation_settings
                    )
                    outputs[f"{name}_{impute_name}"] = valid_df
                    model_generated = True
                    if attempt_idx > 1:
                        print(f"[{impute_name}] {model_display_name} succeeded on retry {attempt_idx}/{max_model_attempts}.")
                    break
                except Exception as e:
                    if attempt_idx < max_model_attempts:
                        print(
                            f"[{impute_name}] {model_display_name} attempt {attempt_idx}/{max_model_attempts} failed: {e}. "
                            "Retrying..."
                        )
                    else:
                        print(f"Skipping {model_display_name} due to error after {max_model_attempts} attempts: {e}")

            if not model_generated:
                continue

        if impute_name == 'mi':
            indicator_cols = [col for col in imp_df_for_generation.columns if col not in metadata]
            if indicator_cols:
                print(f"Removing missing indicator columns: {indicator_cols}")
                imp_df_real = imp_df_real.drop(columns=indicator_cols)
                imp_df_for_generation = imp_df_for_generation.drop(columns=indicator_cols)
                for key in outputs:
                    outputs[key] = outputs[key].drop(columns=indicator_cols, errors='ignore')
                for col in indicator_cols:
                    if col in current_metadata:
                        del current_metadata[col]

        for key, synth_df in outputs.items():
            generator_name = key.split('_')[0] if '_' in key else key
            imp_df_for_generation.to_excel(f'datasets/{impute_name}real_data.xlsx', index=True)
            synth_df.to_excel(f'datasets/{impute_name}{generator_name}_data.xlsx', index=False)

        print(f"Generating comparative metrics report for {impute_name} imputations...")
        # Evaluate this batch
        batch_results = evaluate_all(impute_name, imp_df_real, outputs, current_metadata, constraints)
        all_results.extend(batch_results)

        if regression_target_column and regression_target_column in raw_df_for_cv.columns:
            for model_display_name, model_func in models.items():
                regression_generation_func = lambda d, n, pfx: run_cart(d, n, pfx, metadata, persist=False)
                reg_summary, reg_folds = run_repeated_augmented_regression_cv(
                    real_df=raw_df_for_cv,
                    target_column=regression_target_column,
                    imputation_name=impute_name,
                    synthetic_model_name=f"{model_display_name.lower()}_{impute_name}",
                    model_func=regression_generation_func,
                    metadata=metadata,
                    constraints=constraints,
                    synthetic_target_size=Dn,
                    rejection_thresholds=regression_rejection_thresholds,
                    n_splits=5,
                    n_repeats=cv_repeats,
                    random_state=GLOBAL_RANDOM_SEED,
                    min_batch_size=generation_settings['min_batch_size'],
                    max_batch_size=generation_settings['max_batch_size'],
                    max_total_generated_multiplier=generation_settings['max_total_generated_multiplier'],
                    max_failed_attempts=generation_settings['max_failed_attempts'],
                    acceptance_rate_floor=generation_settings['acceptance_rate_floor'],
                    synthetic_train_ratio=generation_settings['fold_synthetic_ratio'],
                    use_full_synthetic_target=generation_settings['use_full_fold_synthetic_target']
                )
                if not reg_summary.empty:
                    reg_summary['Target_Column'] = regression_target_column
                    regression_summary_frames.append(reg_summary)
                if not reg_folds.empty:
                    reg_folds['Target_Column'] = regression_target_column
                    regression_fold_frames.append(reg_folds)

        if target_column:
            for model_display_name, model_func in models.items():
                cv_generation_func = lambda d, n, pfx: run_cart(d, n, pfx, metadata, persist=False)
                cv_summary, cv_folds = run_repeated_augmented_cv(
                    real_df=raw_df_for_cv,
                    target_column=target_column,
                    imputation_name=impute_name,
                    synthetic_model_name=f"{model_display_name.lower()}_{impute_name}",
                    model_func=cv_generation_func,
                    metadata=metadata,
                    constraints=constraints,
                    synthetic_target_size=Dn,
                    rejection_thresholds=rejection_thresholds,
                    n_splits=5,
                    n_repeats=cv_repeats,
                    random_state=GLOBAL_RANDOM_SEED,
                    min_batch_size=generation_settings['min_batch_size'],
                    max_batch_size=generation_settings['max_batch_size'],
                    max_total_generated_multiplier=generation_settings['max_total_generated_multiplier'],
                    max_failed_attempts=generation_settings['max_failed_attempts'],
                    acceptance_rate_floor=generation_settings['acceptance_rate_floor'],
                    synthetic_train_ratio=generation_settings['fold_synthetic_ratio'],
                    use_full_synthetic_target=generation_settings['use_full_fold_synthetic_target']
                )
                if not cv_summary.empty:
                    cv_summary_frames.append(cv_summary)
                if not cv_folds.empty:
                    cv_fold_frames.append(cv_folds)

    #print("Skipping CTAB-GAN+ model for now...")
    
    if all_results:
        metrics_df = pd.DataFrame(all_results).round(3)
        metrics_df.to_excel('evaluation_report.xlsx', index=False)
        save_metric_comparison_plots(metrics_df, output_dir='plots', model_name='cart')

    if cv_summary_frames:
        cv_summary_df = pd.concat(cv_summary_frames, ignore_index=True).round(4)
        cv_folds_df = pd.concat(cv_fold_frames, ignore_index=True).round(4) if cv_fold_frames else pd.DataFrame()
        with pd.ExcelWriter('augmented_cv_report.xlsx') as writer:
            cv_summary_df.to_excel(writer, sheet_name='summary', index=False)
            if not cv_folds_df.empty:
                cv_folds_df.to_excel(writer, sheet_name='fold_scores', index=False)

    if regression_summary_frames:
        regression_summary_df = pd.concat(regression_summary_frames, ignore_index=True).round(4)
        regression_folds_df = pd.concat(regression_fold_frames, ignore_index=True).round(4) if regression_fold_frames else pd.DataFrame()
        with pd.ExcelWriter('regression_report.xlsx') as writer:
            regression_summary_df.to_excel(writer, sheet_name='summary', index=False)
            if not regression_folds_df.empty:
                regression_folds_df.to_excel(writer, sheet_name='fold_scores', index=False)
        save_regression_metric_plots(regression_summary_df, output_dir='plots')

    print("\nAll done. Outputs and reports written to current directory.")

if __name__ == '__main__':
    main()
