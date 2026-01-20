import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.experimental import enable_iterative_imputer  # For MICE and EM
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from .data_processor import DataProcessor
import warnings


class MissingDataHandler:
    """Detects missingness type (MCAR, MAR, MNAR) and applies automatic imputation."""

    def __init__(self):
        self.imputers = {}

    @staticmethod
    def get_column_dtypes(data) -> dict:
        """
        Returns a dictionary mapping column names to abstract data types
        that are compatible with the processor.
        
        The mapping is as follows:
        - float64, float32, int64, int32 -> "numerical"
        - bool -> "boolean"
        - datetime64[...] -> "datetime"
        - timedelta64[...] -> "timedelta"
        - All others (e.g., object) -> "categorical"
        """
        def map_dtype(dtype: str) -> str:
            if dtype in ['float64', 'float32', 'int64', 'int32']:
                return "numerical"
            elif dtype == 'bool':
                return "boolean"
            elif 'datetime' in dtype:
                return "datetime"
            elif 'timedelta' in dtype:
                return "timedelta"
            else:
                return "categorical"
        
        if isinstance(data, pd.DataFrame):
            return {col: map_dtype(str(dtype)) for col, dtype in data.dtypes.items()}
        elif isinstance(data, np.ndarray) and data.dtype.names is not None:
            return {name: map_dtype(str(data.dtype.fields[name][0])) for name in data.dtype.names}
        else:
            raise TypeError("Data must be a pandas DataFrame or a structured numpy array.")

    def encode_predictors(
        self, df: pd.DataFrame, drop_cols: list = None
    ) -> pd.DataFrame:
        """
        Encodes all columns in the DataFrame so that they are numeric.
        Optionally, drops specified columns (e.g., the target column).

        Steps:
        1. Extract numeric columns.
        2. Convert datetime columns to Unix timestamp (numeric).
        3. Convert timedelta columns to total seconds as float.
        4. For categorical columns (object, category), create dummy variables.
        5. For boolean columns, convert to int (0/1).
        6. Concatenate everything and fill any remaining NaNs with each column's median.

        Args:
            df (pd.DataFrame): Input DataFrame.
            drop_cols (list): List of column names to drop (optional).

        Returns:
            pd.DataFrame: DataFrame with only numeric values and no missing entries.
        """
        df_work = df.copy()
        if drop_cols is not None:
            df_work = df_work.drop(columns=drop_cols)

        # 1. Extract numeric columns.
        num_df = df_work.select_dtypes(include=[np.number], exclude = ["timedelta64[ns]"]).copy()

        # 2. Convert datetime columns to Unix timestamp (numeric).
        datetime_cols = df_work.select_dtypes(include=["datetime64[ns]"])
        if not datetime_cols.empty:
            datetime_numeric = datetime_cols.apply(
                lambda col: col.astype(np.int64) // 10**9
            )
            num_df = pd.concat([num_df, datetime_numeric], axis=1)

        # 3. Convert timedelta columns to total seconds (as float).
        timedelta_cols = df_work.select_dtypes(include=["timedelta64[ns]"])
        if not timedelta_cols.empty:
            timedelta_numeric = pd.DataFrame({
                col: timedelta_cols[col].dt.total_seconds() for col in timedelta_cols.columns
            }, index=df_work.index)
            num_df = pd.concat([num_df, timedelta_numeric], axis=1)


        # 4. Encode categorical columns using get_dummies.
        cat_df = df_work.select_dtypes(include=["object", "category"])
        if not cat_df.empty:
            dummies = pd.get_dummies(cat_df, drop_first=True)
        else:
            dummies = pd.DataFrame(index=df_work.index)

        # 5. Handle boolean columns: convert them to int explicitly.
        bool_df = df_work.select_dtypes(include=["bool"]).astype(int)

        # 6. Concatenate all predictors and fill any remaining missing values with the median.
        result_df = pd.concat([num_df, dummies, bool_df], axis=1)
        result_df = result_df.apply(lambda col: col.fillna(0), axis=0)
        return result_df

    def detect_missingness(self, dfc: pd.DataFrame) -> dict:
        """Detects missingness type for each column, handling multiple data types."""
        df = dfc.copy()
        missingness = {}
        for col in df.columns:
            missing_values = df[col].isna().sum()
            if missing_values == 0:
                continue  # No missing values → Skip detection
            col_type = df[col].dtype

            # **Categorical Data Handling (object, category)**
            if col_type == "object" or df[col].nunique() < 10:
                observed_counts = df[col].dropna().value_counts()
                if len(observed_counts) > 1:
                    _, p_value = stats.chisquare(observed_counts)
                    if p_value > 0.05:
                        missingness[col] = "MCAR"
                        continue

                missing_mask = df[col].isna().astype(int)
                # Use our helper to encode all predictors (drop the target col)
                encoded_data = self.encode_predictors(df, drop_cols=[col])
                model = LogisticRegression()
                model.fit(encoded_data, missing_mask)
                if model.score(encoded_data, missing_mask) > 0.6:
                    missingness[col] = "MAR"
                    continue
                missingness[col] = "MNAR"
                continue

            # **Numerical Data Handling (int, float)**
            elif np.issubdtype(col_type, np.number):
                _, p_value = stats.shapiro(df[col].dropna())
                if p_value > 0.05:
                    missingness[col] = "MCAR"
                    continue
                missing_mask = df[col].isna().astype(int)
                observed_data = self.encode_predictors(df, drop_cols=[col])
                model = LogisticRegression()
                model.fit(observed_data, missing_mask)
                if model.score(observed_data, missing_mask) > 0.6:
                    missingness[col] = "MAR"
                    continue
                observed_values = df[col].dropna()
                missing_rows = df[col].isna()
                if missing_rows.sum() > 0:
                    encoded_missing_vals = self.encode_predictors(df.loc[missing_rows, df.columns != col])
                    missing_vals = encoded_missing_vals.mean(axis=1)
                    _, p_value = stats.ks_2samp(observed_values, missing_vals)
                    if p_value < 0.05:
                        missingness[col] = "MNAR"
                        continue
                missingness[col] = "MAR"
                continue

            # **Boolean Data Handling (bool)**
            elif np.issubdtype(col_type, np.bool_):
                bool_as_int = df[col].astype(float)
                _, p_value = stats.chisquare(bool_as_int.value_counts())
                if p_value > 0.05:
                    missingness[col] = "MCAR"
                    continue
                missingness[col] = "MNAR"
                continue

            # **Datetime Handling (datetime64)**
            elif np.issubdtype(col_type, np.datetime64):
                timestamps = df[col].dropna().astype(int) // 10**9
                _, p_value = stats.shapiro(timestamps)
                if p_value > 0.05:
                    missingness[col] = "MCAR"
                    continue
                missing_mask = df[col].isna().astype(int)
                observed_data = self.encode_predictors(df, drop_cols=[col])
                model = LogisticRegression()
                model.fit(observed_data, missing_mask)
                if model.score(observed_data, missing_mask) > 0.6:
                    missingness[col] = "MAR"
                    continue
                missingness[col] = "MNAR"
                continue

            # **Timedelta Handling (timedelta64)**
            elif np.issubdtype(col_type, np.timedelta64):
                durations = df[col].dropna().dt.total_seconds()
                _, p_value = stats.shapiro(durations)
                if p_value > 0.05:
                    missingness[col] = "MCAR"
                    continue
                missingness[col] = "MNAR"
                continue

        return missingness

    def apply_imputation(self, df: pd.DataFrame, missingness: dict) -> pd.DataFrame:
        """Automatically applies imputation based on missingness type and column data type."""
        df = df.copy()
        metadata = self.get_column_dtypes(df)
        processor = DataProcessor(metadata)
        processed_data = processor.preprocess(df)
        imputer = IterativeImputer(random_state=5842, max_iter=50)
        df_iterative = pd.DataFrame(imputer.fit_transform(processed_data), columns= df.columns)
        for col, mtype in missingness.items():
            if df[col].isna().sum() == 0:
                continue

            # --- Categorical Data (object, category or few unique values) ---
            if (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_categorical_dtype(df[col])
                or (df[col].nunique() < 10)
            ):
                if mtype == "MCAR":
                    df.fillna({col: df[col].mode()[0]}, inplace=True)
                elif mtype == "MAR":
                    # Use get_dummies encoding for categorical data
                    le = LabelEncoder()
                    non_missing = df[col].dropna()
                    le.fit(non_missing)
                    predictor_cols = [c for c in df.columns if c != col]
                    predictors = df_iterative[predictor_cols].copy()
                    df_copy = df.copy()
                    df_copy[f"{col}_encoded"] = df_copy[col].apply(lambda x: le.transform([x])[0] if pd.notna(x) else np.nan)
    
                    # Combine predictors and the encoded target.
                    combined = pd.concat([predictors, df_copy[[f"{col}_encoded"]]], axis=1)
                    # Impute missing values using IterativeImputer.
                    imputer = IterativeImputer(random_state=42)
                    imputed_array = imputer.fit_transform(combined)
                    imputed_df = pd.DataFrame(imputed_array, columns=combined.columns, index=df.index)

                    # Extract the imputed encoded target column.
                    imputed_encoded = imputed_df[f"{col}_encoded"]
                    imputed_encoded = imputed_encoded.round().astype(int)
                    min_code = 0
                    max_code = len(le.classes_) - 1
                    imputed_encoded = imputed_encoded.clip(lower=min_code, upper=max_code)
                    # Decode back to the original categorical labels.
                    imputed_categories = le.inverse_transform(imputed_encoded)
                    df[col] = imputed_categories
                elif mtype == "MNAR":
                    df[col].fillna("Missing", inplace=True)

            # --- Numerical Data ---
            elif pd.api.types.is_numeric_dtype(df[col]):
                if mtype == "MCAR":
                    imputer = SimpleImputer(strategy="mean")
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                elif mtype in ["MAR", "MNAR"]:
                    imputer = IterativeImputer(random_state=42)
                    df[col] = imputer.fit_transform(df[[col]]).ravel()

            # --- Boolean Data ---
            elif pd.api.types.is_bool_dtype(df[col]):
                if mtype == "MCAR":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif mtype in ["MAR", "MNAR"]:
                    numeric_vals = df[col].astype(float)
                    imputer = IterativeImputer(random_state=42)
                    imputed = imputer.fit_transform(numeric_vals.values.reshape(-1, 1))
                    df[col] = np.rint(imputed).astype(bool).flatten()

            # --- Datetime Data ---
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                numeric_series = df[col].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
                if mtype == "MCAR":
                    imputer = SimpleImputer(strategy="median")
                elif mtype in ["MAR", "MNAR"]:
                    imputer = IterativeImputer(random_state=42)
                imputed_numeric = imputer.fit_transform(
                    numeric_series.values.reshape(-1, 1)
                )
                df[col] = pd.to_datetime(imputed_numeric.flatten(), unit='s')

            # --- Timedelta Data ---
            elif pd.api.types.is_timedelta64_dtype(df[col]):
                numeric_series = df[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else np.nan).values.reshape(-1, 1)
                if mtype == "MCAR":
                    imputer = SimpleImputer(strategy="median" )
                elif mtype in ["MAR", "MNAR"]:
                    imputer = IterativeImputer(random_state=42)
                imputed_numeric = imputer.fit_transform(numeric_series)
                df[col] = pd.to_timedelta(imputed_numeric.flatten(), unit="s")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df
