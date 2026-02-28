from metadata_manager import generate_metadata, interactive_edit_metadata, save_as_sdv_json
import pandas as pd
import numpy as np
import os
from synthpop.synth import run_cart
from sdv_all import run_copula_gan,run_ctgan, run_gaussian_copula, run_tvae
from ctab_gan_plus import ctabganplus
from metrics import evaluate_all
from ros import apply_ros,add_constraints,drop_covariance_violators,drop_constraint_violators
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, MissingIndicator
from sklearn.ensemble import RandomForestClassifier

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

        model = RandomForestClassifier(n_estimators=100, random_state=47)
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
            vals = series.values
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
    imputer = IterativeImputer(random_state=67)
    data_imputed = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(data_imputed, columns=df.columns, index=df.index)
    return decode_categorical(df_imputed, encoders)

def generate_valid_samples(model_func, imp_df, target_size, prefix, metadata, constraints,target_column):
    """
    Continually generates synthetic data in doubling batch sizes until 
    the requested target_size is met after all physical constraints are applied.
    """
    valid_data = pd.DataFrame()
    iteration = 0
    model_name = ""
    
    while len(valid_data) < target_size:
        n_samples = (2 ** iteration) * target_size
        #print(f"Generating batch size of {n_samples}.. Current valid : {len(valid_data)}")
        model_name, temp_df = model_func(imp_df, n_samples, prefix)    
        if target_column:
            temp_df = apply_ros(temp_df, target_column)
        temp_df = drop_constraint_violators(temp_df, constraints)
        #temp_df = drop_covariance_violators(imp_df, temp_df, metadata)
        
        valid_data = pd.concat([valid_data, temp_df], ignore_index=True)
        iteration += 1
        
    return model_name, valid_data


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
    n = int(input("Which one is the most important column for physical constraints? (type column index, or '-1')> "))
    target_column = df.columns[n] if n!=-1 else None
    print("Saving SDV-compatible metadata to sdv_metadata.json")
    save_as_sdv_json(metadata, path='sdv_metadata.json')
    
    df = impute_categorical_rf(df, metadata)
    #print(df.head())
    imputations = {
        'iter': iterative_imputation(df, metadata), 
        'knn': knn_imputation(df, metadata), 
        'mi': missing_indicator_imputation(df)
    }
    
    constraints = add_constraints(df)

    all_results = []
    for impute_name, imp_df in imputations.items():
        #print(imp_df.head())
        outputs = {}
        if target_column:    
            imp_df = apply_ros(imp_df, target_column)
            
        # Update metadata for this imputation (e.g. if MI added columns)
        current_metadata = metadata.copy()
        for col in imp_df.columns:
            if col not in current_metadata:
                current_metadata[col] = 'categorical'
        
        save_as_sdv_json(current_metadata, path='sdv_metadata.json')
        #print(imp_df.head())
        models = {
            'CART': lambda d, n, pfx: run_cart(d, n, pfx, current_metadata),
            'Gaussian Copula': run_gaussian_copula,
            'CTGAN': run_ctgan,
            'CopulaGAN': run_copula_gan,
            'TVAE': run_tvae
        }
        
        for model_display_name, model_func in models.items():
            print(f"[{impute_name}] Running {model_display_name}...")
            
            try:
                name, valid_df = generate_valid_samples(
                    model_func=model_func,
                    imp_df=imp_df,
                    target_size=Dn,
                    prefix=impute_name,
                    metadata=current_metadata,
                    constraints=constraints,
                    target_column=target_column
                )
                outputs[f"{name}_{impute_name}"] = valid_df
            except Exception as e:
                print(f"Skipping {model_display_name} due to error: {e}")

        if impute_name == 'mi':
            indicator_cols = [col for col in imp_df.columns if col not in metadata]
            if indicator_cols:
                print(f"Removing missing indicator columns: {indicator_cols}")
                imp_df = imp_df.drop(columns=indicator_cols)
                for key in outputs:
                    outputs[key] = outputs[key].drop(columns=indicator_cols, errors='ignore')
                for col in indicator_cols:
                    if col in current_metadata:
                        del current_metadata[col]

        print(f"Generating comparative metrics report for {impute_name} imputations...")
        # Evaluate this batch
        batch_results = evaluate_all(impute_name, imp_df, outputs, current_metadata, constraints)
        all_results.extend(batch_results)

    #print("Skipping CTAB-GAN+ model for now...")
    
    if all_results:
        pd.DataFrame(all_results).round(3).to_excel('evaluation_report.xlsx', index=False)

    print("\nAll done. Outputs and reports written to current directory.")

if __name__ == '__main__':
    main()
