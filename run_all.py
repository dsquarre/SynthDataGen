from metadata_manager import generate_metadata, interactive_edit_metadata, save_as_sdv_json
import pandas as pd
import numpy as np
import os
import ast
import re
from synthpop.synth import run_cart
from sdv_all import run_copula_gan,run_ctgan, run_gaussian_copula, run_tvae
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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


def build_aligned_metadata_for_df(df: pd.DataFrame, base_metadata: dict) -> dict:
    aligned = {}
    for col in df.columns:
        if col in base_metadata:
            aligned[col] = base_metadata[col]
        else:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                aligned[col] = 'numerical'
            else:
                aligned[col] = 'categorical'
    return aligned


def run_sdv_model_with_aligned_metadata(model_func, df: pd.DataFrame, num_rows: int, output_prefix: str, base_metadata: dict):
    aligned_metadata = build_aligned_metadata_for_df(df, base_metadata)
    save_as_sdv_json(aligned_metadata, path='sdv_metadata.json')
    return model_func(df, num_rows, output_prefix)


def _normalize_formula_expression(expression: str) -> str:
    return re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'c\1', expression)


def _evaluate_formula_expression(expression: str, reference_df: pd.DataFrame) -> pd.Series:
    if reference_df.empty:
        raise ValueError("Reference dataframe is empty.")

    normalized_expr = _normalize_formula_expression(expression.strip())
    node = ast.parse(normalized_expr, mode='eval')
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

            raise ValueError("Unsupported operator in formula.")

        if isinstance(ast_node, ast.UnaryOp):
            operand = _eval_ast(ast_node.operand)
            if isinstance(ast_node.op, ast.UAdd):
                return operand
            if isinstance(ast_node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator in formula.")

        if isinstance(ast_node, ast.Name):
            if ast_node.id not in column_refs:
                raise ValueError(f"Unknown column reference '{ast_node.id}'.")
            return column_refs[ast_node.id]

        if isinstance(ast_node, ast.Constant) and isinstance(ast_node.value, (int, float)):
            return float(ast_node.value)

        raise ValueError("Unsupported expression syntax.")

    evaluated = _eval_ast(node)
    if np.isscalar(evaluated):
        return pd.Series([evaluated] * len(reference_df), index=reference_df.index)
    return pd.Series(evaluated, index=reference_df.index)


def collect_reconstruction_formulas(reference_df: pd.DataFrame):
    print("\nOptional: define reconstruction formulas to add derived columns in exported files.")
    print("Enter a formula expression, then provide the new column name.")
    print("You can also use shorthand '<new_column>=<formula>'.")
    print("Examples: 12/13, c12/(c13+0.001), ratio=12/13")
    print("Note: integer tokens are treated as column indices. Type 'done' when finished.")

    rules = []
    auto_idx = 1

    while True:
        user_input = input("Formula expression (or 'done'): ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'done':
            break

        if '=' in user_input:
            target_column, expression = user_input.split('=', 1)
            target_column = target_column.strip()
            expression = expression.strip()
        else:
            expression = user_input
            default_name = f"derived_{auto_idx}"
            target_column = input(
                f"New column name (default: {default_name}): "
            ).strip() or default_name

        if not target_column or not expression:
            print("Invalid formula. Please provide both target column and expression.")
            continue

        try:
            preview_series = _evaluate_formula_expression(expression, reference_df)
        except Exception as exc:
            print(f"Failed to parse formula '{user_input}': {exc}")
            continue

        if np.isinf(preview_series.to_numpy(dtype=float, na_value=np.nan)).any():
            print(f"Warning: formula '{target_column}={expression}' produced infinity values in preview.")

        rules.append({'target_column': target_column, 'expression': expression})
        auto_idx += 1
        print(f"Stored formula: {target_column} = {expression}")

    if rules:
        print(f"Captured {len(rules)} reconstruction formula(s).")
    else:
        print("No reconstruction formulas captured.")

    return rules


def apply_reconstruction_formulas(df: pd.DataFrame, rules):
    if not rules:
        return df

    base_reference = df.copy()
    enriched_df = df.copy()

    for rule in rules:
        target_column = rule['target_column']
        expression = rule['expression']
        try:
            enriched_df[target_column] = _evaluate_formula_expression(expression, base_reference)
        except Exception as exc:
            print(f"Warning: could not apply reconstruction formula '{target_column}={expression}': {exc}")

    return enriched_df

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

    print("\nColumns after drop:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    reconstruction_rules = collect_reconstruction_formulas(df)

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

    cv_threshold_input = input(
        "Apply metric-threshold rejection during CV synthetic generation? (y/N): "
    ).strip().lower()
    use_cv_thresholds = cv_threshold_input in {'y', 'yes', '1', 'true'}
    print(f"CV threshold gating is {'enabled' if use_cv_thresholds else 'disabled'}.")

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
            'Gaussian Copula': run_gaussian_copula,
            'CTGAN': run_ctgan,
            'CopulaGAN': run_copula_gan,
            'TVAE': run_tvae
        }
        
        for model_display_name, model_func in models.items():
            print(f"[{impute_name}] Running {model_display_name}...")

            max_model_attempts = max(1, int(generation_settings.get('model_retry_attempts', 1)))
            model_generated = False

            for attempt_idx in range(1, max_model_attempts + 1):
                try:
                    # Only CART accepts persist parameter; other models don't
                    if model_display_name == 'CART':
                        generation_func = lambda d, n, pfx, mf=model_func: mf(d, n, pfx, persist=False)
                        model_thresholds = rejection_thresholds
                    elif model_display_name == 'Gaussian Copula':
                        generation_func = lambda d, n, pfx, mf=model_func: mf(d, n, pfx)
                        # Lower cov_score and propensity_score for CTGAN due to observed performance
                        model_thresholds = rejection_thresholds.copy()
                        model_thresholds['corr_score'] = 0.40  # Lowered from 0.99
                        model_thresholds['cov_score'] = 0.20 # Lowered from 0.40
                        model_thresholds['propensity_score'] = 0  # Lowered from 0.15
                    else:
                        generation_func = lambda d, n, pfx, mf=model_func: mf(d, n, pfx)
                        # Lower cov_score and propensity_score for non-CART models
                        model_thresholds = rejection_thresholds.copy()
                        model_thresholds['corr_score'] = 0.18  # Lowered from 0.99
                        model_thresholds['cov_score'] = 0.05  # Lowered from 0.40
                        model_thresholds['propensity_score'] = 0  # Lowered from 0.15

                    
                    name, valid_df = generate_valid_samples(
                        model_func=generation_func,
                        imp_df=imp_df_for_generation,
                        real_reference_df=imp_df_real,
                        target_size=Dn,
                        prefix=f"{impute_name}_attempt{attempt_idx}",
                        metadata=current_metadata,
                        constraints=constraints,
                        target_column=target_column,
                        rejection_thresholds=model_thresholds,
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

        reconstructed_eval_real = apply_reconstruction_formulas(imp_df_real, reconstruction_rules)
        reconstructed_outputs = {
            key: apply_reconstruction_formulas(synth_df, reconstruction_rules)
            for key, synth_df in outputs.items()
        }
        reconstructed_metadata = build_aligned_metadata_for_df(reconstructed_eval_real, current_metadata)

        for key, synth_df in outputs.items():
            generator_name = key.split('_')[0] if '_' in key else key
            export_real_df = apply_reconstruction_formulas(imp_df_for_generation, reconstruction_rules)
            export_synth_df = reconstructed_outputs.get(key, synth_df)
            export_real_df.to_excel(f'datasets/{impute_name}real_data.xlsx', index=True)
            export_synth_df.to_excel(f'datasets/{impute_name}{generator_name}_data.xlsx', index=False)

        print(f"Generating comparative metrics report for {impute_name} imputations...")
        # Evaluate this batch
        batch_results = evaluate_all(
            impute_name,
            reconstructed_eval_real,
            reconstructed_outputs,
            reconstructed_metadata,
            constraints
        )
        all_results.extend(batch_results)

        if regression_target_column and regression_target_column in raw_df_for_cv.columns:
            for model_display_name, model_func in models.items():
                if model_display_name == 'CART':
                    regression_generation_func = (
                        lambda d, n, pfx, bm=current_metadata:
                        run_cart(d, n, pfx, build_aligned_metadata_for_df(d, bm), persist=False)
                    )
                else:
                    regression_generation_func = (
                        lambda d, n, pfx, mf=model_func, bm=current_metadata:
                        run_sdv_model_with_aligned_metadata(mf, d, n, pfx, bm)
                    )

                if use_cv_thresholds:
                    if model_display_name == 'CART':
                        reg_model_thresholds = regression_rejection_thresholds
                    else:
                        reg_model_thresholds = regression_rejection_thresholds.copy()
                        reg_model_thresholds['cov_score'] = 0.20
                        reg_model_thresholds['corr_score'] = 0.18
                else:
                    reg_model_thresholds = None
                
                reg_summary, reg_folds = run_repeated_augmented_regression_cv(
                    real_df=raw_df_for_cv,
                    target_column=regression_target_column,
                    imputation_name=impute_name,
                    synthetic_model_name=f"{model_display_name.lower()}_{impute_name}",
                    model_func=regression_generation_func,
                    metadata=metadata,
                    constraints=constraints,
                    synthetic_target_size=Dn,
                    rejection_thresholds=reg_model_thresholds,
                    n_splits=5,
                    n_repeats=cv_repeats,
                    random_state=GLOBAL_RANDOM_SEED,
                    min_batch_size=generation_settings['min_batch_size'],
                    max_batch_size=generation_settings['max_batch_size'],
                    max_total_generated_multiplier=generation_settings['max_total_generated_multiplier'],
                    max_failed_attempts=generation_settings['max_failed_attempts'],
                    acceptance_rate_floor=generation_settings['acceptance_rate_floor'],
                    synthetic_train_ratio=generation_settings['fold_synthetic_ratio'],
                    use_full_synthetic_target=generation_settings['use_full_fold_synthetic_target'],
                    reconstruction_rules=reconstruction_rules
                )
                if not reg_summary.empty:
                    reg_summary['Target_Column'] = regression_target_column
                    regression_summary_frames.append(reg_summary)
                if not reg_folds.empty:
                    reg_folds['Target_Column'] = regression_target_column
                    regression_fold_frames.append(reg_folds)

        if target_column:
            for model_display_name, model_func in models.items():
                if model_display_name == 'CART':
                    cv_generation_func = (
                        lambda d, n, pfx, bm=current_metadata:
                        run_cart(d, n, pfx, build_aligned_metadata_for_df(d, bm), persist=False)
                    )
                else:
                    cv_generation_func = (
                        lambda d, n, pfx, mf=model_func, bm=current_metadata:
                        run_sdv_model_with_aligned_metadata(mf, d, n, pfx, bm)
                    )

                if use_cv_thresholds:
                    if model_display_name == 'CART':
                        cv_model_thresholds = rejection_thresholds
                    else:
                        cv_model_thresholds = rejection_thresholds.copy()
                        cv_model_thresholds['cov_score'] = 0.20
                        cv_model_thresholds['corr_score'] = 0.18
                        cv_model_thresholds['propensity_score'] = 0.08
                else:
                    cv_model_thresholds = None
                
                cv_summary, cv_folds = run_repeated_augmented_cv(
                    real_df=raw_df_for_cv,
                    target_column=target_column,
                    imputation_name=impute_name,
                    synthetic_model_name=f"{model_display_name.lower()}_{impute_name}",
                    model_func=cv_generation_func,
                    metadata=metadata,
                    constraints=constraints,
                    synthetic_target_size=Dn,
                    rejection_thresholds=cv_model_thresholds,
                    n_splits=5,
                    n_repeats=cv_repeats,
                    random_state=GLOBAL_RANDOM_SEED,
                    min_batch_size=generation_settings['min_batch_size'],
                    max_batch_size=generation_settings['max_batch_size'],
                    max_total_generated_multiplier=generation_settings['max_total_generated_multiplier'],
                    max_failed_attempts=generation_settings['max_failed_attempts'],
                    acceptance_rate_floor=generation_settings['acceptance_rate_floor'],
                    synthetic_train_ratio=generation_settings['fold_synthetic_ratio'],
                    use_full_synthetic_target=generation_settings['use_full_fold_synthetic_target'],
                    reconstruction_rules=reconstruction_rules
                )
                if not cv_summary.empty:
                    cv_summary_frames.append(cv_summary)
                if not cv_folds.empty:
                    cv_fold_frames.append(cv_folds)

    #print("Skipping CTAB-GAN+ model for now...")
    
    if all_results:
        metrics_df = pd.DataFrame(all_results).round(3)
        metrics_df.to_excel('evaluation_report.xlsx', index=False)
        # Generate comparison plots for all generators/models
        all_generators = metrics_df['Generator'].unique()
        for gen_name in sorted(all_generators):
            save_metric_comparison_plots(metrics_df, output_dir='plots', model_name=gen_name)

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
