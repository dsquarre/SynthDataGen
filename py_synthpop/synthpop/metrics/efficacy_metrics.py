# efficacy_metrics.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    accuracy_score, 
    f1_score
)
from sklearn.model_selection import train_test_split

class EfficacyMetrics:
    """
    A class to compute efficacy metrics comparing real and synthetic datasets
    for downstream predictive tasks. The idea is to train a predictive model on
    synthetic data and evaluate its performance on real data. The type of metrics
    computed depends on the task:
    
      - For regression (when the target is numerical):
            * Mean Squared Error (MSE)
            * Mean Absolute Error (MAE)
            * R^2 Score
            
      - For classification (when the target is categorical/boolean):
            * Accuracy Score
            * Weighted F1 Score

    Parameters
    ----------
    task : str, optional (default='regression')
        The predictive task type. Must be either 'regression' or 'classification'.
    target_column : str
        The name of the target column. Must exist in both real and synthetic data.
    test_size : float, optional (default=0.3)
        (Optional) Proportion of the real data to be used for testing.
        (Note: In the default approach we train on all synthetic data and test on full real data.)
    random_state : int, optional (default=42)
        Random seed for reproducibility.
    """
    
    def __init__(self, task='regression', target_column=None, test_size=0.3, random_state=42):
        if task not in ['regression', 'classification']:
            raise ValueError("Task must be either 'regression' or 'classification'.")
        if target_column is None:
            raise ValueError("A target column must be specified.")
            
        self.task = task
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
        """
        Evaluate the efficacy of synthetic data by training a model on synthetic data
        and testing its performance on real data.

        Args:
            real_df (pd.DataFrame): The real dataset.
            synthetic_df (pd.DataFrame): The synthetic dataset.

        Returns:
            dict: A dictionary of performance metrics.
        """
        # Verify that the target column exists in both datasets.
        if self.target_column not in real_df.columns or self.target_column not in synthetic_df.columns:
            raise ValueError("The target column must exist in both real and synthetic datasets.")

        # Separate features and target.
        X_syn = synthetic_df.drop(columns=[self.target_column])
        y_syn = synthetic_df[self.target_column]
        X_real = real_df.drop(columns=[self.target_column])
        y_real = real_df[self.target_column]

        # Handle categorical encoding only if it's a classification task
        
        categorical_cols = X_syn.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols:
            X_syn = pd.get_dummies(X_syn, columns=categorical_cols, drop_first=True)
            X_real = pd.get_dummies(X_real, columns=categorical_cols, drop_first=True)

            # Align columns in case of different categorical levels between real and synthetic data
            X_syn, X_real = X_syn.align(X_real, join='left', axis=1, fill_value=0)

        # Model Training and Evaluation
        if self.task == 'regression':
            model = LinearRegression()
            model.fit(X_syn, y_syn)
            predictions = model.predict(X_real)
            mse = mean_squared_error(y_real, predictions)
            mae = mean_absolute_error(y_real, predictions)
            r2 = r2_score(y_real, predictions)
            return {
                "mse": mse,
                "mae": mae,
                "r2": r2
            }
        else:  # classification
            model = DecisionTreeClassifier(random_state=self.random_state)
            model.fit(X_syn, y_syn)
            predictions = model.predict(X_real)
            accuracy = accuracy_score(y_real, predictions)
            f1 = f1_score(y_real, predictions, average='weighted')
            return {
                "accuracy": accuracy,
                "f1_score": f1
            }

