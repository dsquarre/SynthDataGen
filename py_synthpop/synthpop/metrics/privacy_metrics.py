# privacy_metrics.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class DisclosureProtection:
    """
    A class to compute the disclosure protection metric for synthetic data.

    This metric measures the proportion of synthetic records that are too similar
    (within a defined threshold) to real records, posing a disclosure risk.

    Parameters
    ----------
    real_data : pd.DataFrame
        A DataFrame containing the real data. Supports both numerical and categorical features.
    synthetic_data : pd.DataFrame
        A DataFrame containing the synthetic data (with the same structure as real_data).
    threshold : float, optional
        A distance threshold under which a synthetic record is considered a potential disclosure risk.
        If not provided, it is computed as the 10th percentile of the nearest-neighbor distances among real records.
    """

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, threshold: float = None):
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.threshold = threshold
        
        # Preprocess data for distance computation
        self.real_data, self.synthetic_data = self._preprocess_data(self.real_data, self.synthetic_data)
        
        # Compute distance threshold if not provided
        self._compute_threshold()

    def _preprocess_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Preprocess both real and synthetic datasets:
        - Standardize numerical columns
        - One-hot encode categorical columns
        - Align columns to ensure consistency
        """

        # Identify numerical and categorical columns
        categorical_cols = real_data.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()

        # One-Hot Encode Categorical Columns
        if categorical_cols:
            encoder = OneHotEncoder(sparse_output=True, drop="first", handle_unknown="ignore")
            real_cats = encoder.fit_transform(real_data[categorical_cols])
            synthetic_cats = encoder.transform(synthetic_data[categorical_cols])

            # Convert to DataFrame
            real_cat_df = pd.DataFrame(real_cats.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
            synthetic_cat_df = pd.DataFrame(synthetic_cats.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

            # Drop original categorical columns and replace with encoded versions
            real_data = real_data.drop(columns=categorical_cols)
            synthetic_data = synthetic_data.drop(columns=categorical_cols)
            real_data = pd.concat([real_data, real_cat_df], axis=1)
            synthetic_data = pd.concat([synthetic_data, synthetic_cat_df], axis=1)

        # Standardize numerical features
        if numerical_cols:
            scaler = MinMaxScaler()
            real_data[numerical_cols] = scaler.fit_transform(real_data[numerical_cols])
            synthetic_data[numerical_cols] = scaler.transform(synthetic_data[numerical_cols])

        # Align columns (in case some categories exist in one dataset but not the other)
        real_data, synthetic_data = real_data.align(synthetic_data, join="left", axis=1, fill_value=0)

        return real_data, synthetic_data

    def _compute_threshold(self):
        """
        Compute the threshold if not provided. Uses the 10th percentile of the nearest-neighbor
        distances among real records (excluding self-distance).
        """
        if self.threshold is None:
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(self.real_data)
            distances, _ = nn.kneighbors(self.real_data)
            self.threshold = np.percentile(distances[:, 1], 10)  # Exclude self-distance

    def score(self) -> float:
        """
        Compute the disclosure protection score.

        Returns
        -------
        float
            Disclosure protection score between 0 and 1.
        """
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.real_data)
        distances, _ = nn.kneighbors(self.synthetic_data)
        distances = distances.flatten()
        risk_count = np.sum(distances < self.threshold)
        risk_rate = risk_count / len(distances)
        return 1 - risk_rate  # Higher score means better protection

    def report(self) -> dict:
        """
        Generate a detailed report of the disclosure protection metric.

        Returns
        -------
        dict
            A dictionary containing the threshold, risk rate, and the final disclosure protection score.
        """
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.real_data)
        distances, _ = nn.kneighbors(self.synthetic_data)
        distances = distances.flatten()
        risk_count = np.sum(distances < self.threshold)
        risk_rate = risk_count / len(distances)
        score = 1 - risk_rate

        return {
            "threshold": self.threshold,
            "risk_rate": risk_rate,
            "disclosure_protection_score": score
        }
