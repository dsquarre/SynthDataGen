# metrics_report.py

import pandas as pd
import numpy as np
from .single_columns_metrics import (
    category_coverage,
    range_coverage,
    boundary_adherence,
    category_adherence,
    ks_complement,
    tv_complement,
    statistic_similarity,
    missing_value_similarity
)

class MetricsReport:
    """
    A class to produce a report comparing real and synthetic datasets with respect
    to data validity and data structure.

    The report computes the following metrics for each column:
      - For numerical (or datetime/timedelta) columns:
          * Range Coverage: Proportion of the real data's range covered by the synthetic data.
          * Boundary Adherence: Fraction of synthetic values within the real data's min/max.
          * KS Complement: 1 minus the Kolmogorov-Smirnov statistic.
          * TV Complement: 1 minus the Total Variation distance computed over histograms.
          * Statistic Similarity: Similarity of mean, std, and median.
          * Missing Value Similarity: Similarity in the proportion of missing values.
      - For categorical (or boolean) columns:
          * Category Coverage: Proportion of real categories found in synthetic data.
          * Category Adherence: Fraction of synthetic values that are valid real categories.
          * Missing Value Similarity.

    Optionally, you may provide a metadata dictionary mapping column names to abstract types.
    If metadata is not provided, the type is inferred from the pandas dtype.
    """
    
    def __init__(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, metadata: dict = None):
        """
        Args:
            real_df (pd.DataFrame): The real dataset.
            synthetic_df (pd.DataFrame): The synthetic dataset.
            metadata (dict, optional): Mapping from column names to types (e.g., "numerical",
                "categorical", "boolean", "datetime", "timedelta"). If not provided, types are inferred.
        """
        self.real_df = real_df
        self.synthetic_df = synthetic_df
        # If no metadata is provided, infer types based on the dtype string.
        if metadata is None:
            metadata = {}
            for col in real_df.columns:
                dtype = str(real_df[col].dtype)
                if "float" in dtype or "int" in dtype:
                    metadata[col] = "numerical"
                elif "datetime" in dtype:
                    metadata[col] = "datetime"
                elif "timedelta" in dtype:
                    metadata[col] = "timedelta"
                elif "bool" in dtype:
                    metadata[col] = "boolean"
                else:
                    metadata[col] = "categorical"
        self.metadata = metadata

    def generate_report(self) -> pd.DataFrame:
        """
        Generate a report comparing the real and synthetic datasets.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a column in the data and
            contains computed metrics. Non-applicable metrics are marked as 'N/A'.
        """
        report_data = []
        for col in self.real_df.columns:
            col_type = self.metadata.get(col, "numerical")
            real = self.real_df[col]
            synthetic = self.synthetic_df[col]
            col_report = {"column": col, "type": col_type}
            
            # Missing value similarity applies to all columns.
            col_report["missing_value_similarity"] = missing_value_similarity(real, synthetic)
            
            # For numerical/datetime/timedelta columns, compute numerical metrics and mark categorical metrics as 'N/A'
            if col_type in ["numerical", "datetime", "timedelta"]:
                col_report["range_coverage"] = range_coverage(real, synthetic)
                col_report["boundary_adherence"] = boundary_adherence(real, synthetic)
                col_report["ks_complement"] = ks_complement(real, synthetic)
                col_report["tv_complement"] = "N/A"
                col_report["statistic_similarity"] = statistic_similarity(real, synthetic)
                col_report["category_coverage"] = "N/A"
                col_report["category_adherence"] = "N/A"
            
            # For categorical/boolean columns, compute categorical metrics and mark numerical metrics as 'N/A'
            elif col_type in ["categorical", "boolean"]:
                col_report["range_coverage"] = "N/A"
                col_report["boundary_adherence"] = "N/A"
                col_report["ks_complement"] = "N/A"
                col_report["tv_complement"] = tv_complement(real, synthetic)
                col_report["statistic_similarity"] = "N/A"
                col_report["category_coverage"] = category_coverage(real, synthetic)
                col_report["category_adherence"] = category_adherence(real, synthetic)
            
            else:
                col_report["note"] = "Unknown type; metrics not computed"
            
            report_data.append(col_report)
        return pd.DataFrame(report_data)
