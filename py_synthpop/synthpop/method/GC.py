import inspect
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy
import copulas.univariate
from copulas import multivariate
from sklearn.preprocessing import OneHotEncoder
from synthpop.method.helpers import (
    validate_numerical_distributions,
    warn_missing_numerical_distributions,
    flatten_dict,
    unflatten_dict,
)

LOGGER = logging.getLogger(__name__)

class BaseSingleTableSynthesizer:
    """
    Base class for single table synthesizers.
    
    Args:
        metadata (dict): Dictionary mapping column names to their types.
        enforce_min_max_values (bool): Whether to clip reversed numerical values to the observed min/max. Defaults to True.
        enforce_rounding (bool): Whether to round numerical columns during reverse transformation. Defaults to True.
        locales (Union[List[str], str]): Default locale(s) to use. Defaults to "en_US".
    """
    def __init__(
        self,
        metadata: Dict[str, str],
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        locales: Union[List[str], str] = "en_US",
    ) -> None:
        self.metadata = metadata
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        if isinstance(locales, str):
            self.locales = [locales]
        else:
            self.locales = locales


class GaussianCopulaMethod(BaseSingleTableSynthesizer):
    # Mapping of distribution name (lowercase) to copulas univariate classes.
    _DISTRIBUTIONS: Dict[str, Any] = {
        "norm": copulas.univariate.GaussianUnivariate,
        "beta": copulas.univariate.BetaUnivariate,
        "truncnorm": copulas.univariate.TruncatedGaussian,
        "gamma": copulas.univariate.GammaUnivariate,
        "uniform": copulas.univariate.UniformUnivariate,
        "gaussian_kde": copulas.univariate.GaussianKDE,
    }
    # Maximum iterations for correlation matrix adjustment
    _MAX_CORR_ITERATIONS: int = 10

    @classmethod
    def get_distribution_class(cls, distribution: str) -> Any:
        """
        Return the corresponding distribution class from copulas.univariate.

        Args:
            distribution (str): A string representing a copulas univariate distribution.

        Returns:
            The corresponding copulas univariate class.
        """
        if not isinstance(distribution, str):
            raise ValueError(f"Distribution specification must be a string, got {type(distribution)}")
        # Allow case-insensitive matching.
        distribution_key = distribution.lower()
        if distribution_key not in cls._DISTRIBUTIONS:
            error_message = (
                f"Invalid distribution specification '{distribution}'. "
                f"Valid options: {list(cls._DISTRIBUTIONS.keys())}"
            )
            raise ValueError(error_message)
        return cls._DISTRIBUTIONS[distribution_key]

    def __init__(
        self,
        metadata: Dict[str, str],
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        locales: Union[List[str], str] = "en_US",
        numerical_distributions: Optional[Dict[str, str]] = None,
        default_distribution: Optional[str] = None,
    ) -> None:
        super().__init__(metadata, enforce_min_max_values, enforce_rounding, locales)
        # Validate numerical distributions using metadata keys.
        validate_numerical_distributions(numerical_distributions, list(self.metadata.keys()))
        self.default_distribution: str = default_distribution or "beta"
        self._default_distribution = self.get_distribution_class(self.default_distribution)
        self._set_numerical_distributions(numerical_distributions)
        self._num_rows: Optional[int] = None
        self._model: Optional[Any] = None
        self._fitted: bool = False

    def _set_numerical_distributions(self, numerical_distributions: Optional[Dict[str, str]]) -> None:
        """
        Sets the numerical distributions to be used during model initialization.
        """
        self.numerical_distributions = numerical_distributions or {}
        self._numerical_distributions = {
            field: self.get_distribution_class(distribution)
            for field, distribution in self.numerical_distributions.items()
        }

    def _learn_num_rows(self, processed_data: pd.DataFrame) -> int:
        """
        Learn the number of rows from the processed data.
        """
        return len(processed_data)

    def _get_numerical_distributions(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a complete dictionary of numerical distributions for all columns in the data.
        """
        numerical_distributions = deepcopy(self._numerical_distributions)
        for column in processed_data.columns:
            if column not in numerical_distributions:
                numerical_distributions[column] = self._default_distribution
        return numerical_distributions

    def _initialize_model(self, numerical_distributions: Dict[str, Any]) -> Any:
        """
        Initialize the GaussianMultivariate model with the given numerical distributions.
        """
        return multivariate.GaussianMultivariate(distribution=numerical_distributions)

    def _fit_model(self, processed_data: pd.DataFrame) -> None:
        """
        Fit the GaussianMultivariate model on the processed data.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="scipy")
            self._model.fit(processed_data)

    
    def fit(self, processed_data: pd.DataFrame) -> None:
        """
        Public API method to fit the Gaussian Copula model on processed data.
        
        Args:
            processed_data (pd.DataFrame): Data that has been preprocessed.
        """
        warn_missing_numerical_distributions(self.numerical_distributions, list(processed_data.columns))
        self._num_rows = self._learn_num_rows(processed_data)
        numerical_distributions = self._get_numerical_distributions(processed_data)
        self._model = self._initialize_model(numerical_distributions)
        self._fit_model(processed_data)
        self._fitted = True

    def sample(self, num_rows: int, conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Public API method to sample synthetic data from the fitted model.
        
        Args:
            num_rows (int): Number of rows to sample.
            conditions (Optional[Dict[str, Any]]): Optional conditions for sampling.
            
        Returns:
            pd.DataFrame: A DataFrame containing the synthetic samples.
        """
        if not self._fitted or self._model is None:
            raise ValueError("Model is not fitted yet. Please call fit() before sampling.")
        return self._model.sample(num_rows, conditions=conditions)

    def get_learned_distributions(self) -> Dict[str, Any]:
        """
        Get the marginal distributions used by the Gaussian Copula.
        
        Returns:
            Dict[str, Any]: A dictionary mapping column names to the distribution name and learned parameters.
            
        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self._fitted or self._model is None:
            raise ValueError("Distributions have not been learned yet. Please fit your model first using 'fit()'.")
        if not hasattr(self._model, "to_dict") or not self._model.to_dict():
            return {}
        parameters = self._model.to_dict()
        columns = parameters.get("columns", [])
        univariates = deepcopy(parameters.get("univariates", []))
        learned_distributions: Dict[str, Any] = {}
        valid_columns = self._get_valid_columns_from_metadata(columns)
        for column, learned_params in zip(columns, univariates):
            if column in valid_columns:
                distribution = self.numerical_distributions.get(column, self.default_distribution)
                learned_params.pop("type", None)
                learned_distributions[column] = {
                    "distribution": distribution,
                    "learned_parameters": learned_params,
                }
        return learned_distributions

    def _get_valid_columns_from_metadata(self, columns: List[str]) -> List[str]:
        """
        Extract valid columns based on the metadata.
        
        Args:
            columns (List[str]): List of column names.
            
        Returns:
            List[str]: Valid column names found in metadata.
        """
        valid_columns: List[str] = []
        for column in columns:
            for valid_column in self.metadata.keys():
                if column.startswith(valid_column):
                    valid_columns.append(column)
                    break
        return valid_columns

    def _get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the copula model.
        
        Returns:
            Dict[str, Any]: A flattened dictionary containing copula parameters.
        """
        # Ensure univariates are in their base instance form if applicable.
        for univariate in self._model.univariates:
            if isinstance(univariate, copulas.univariate.Univariate):
                univariate = univariate._instance
        params = self._model.to_dict()
        correlation = []
        for index, row in enumerate(params.get("correlation", [])[1:]):
            correlation.append(row[: index + 1])
        params["correlation"] = correlation
        params["univariates"] = dict(zip(params.get("columns", []), params.get("univariates", [])))
        params["num_rows"] = self._num_rows
        return flatten_dict(params)
    
    @classmethod
    def _get_nearest_correlation_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        """
        Find the nearest Positive Semi-definite (PSD) correlation matrix.
        Iteratively adjust negative eigenvalues up to a maximum number of iterations.
        
        Args:
            matrix (np.ndarray): Input correlation matrix.
            
        Returns:
            np.ndarray: Adjusted correlation matrix that is PSD and has ones on the diagonal.
        """
        eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
        iterations = 0
        identity = np.identity(len(matrix))
        while np.any(eigenvalues < 0) and iterations < cls._MAX_CORR_ITERATIONS:
            # Set negative eigenvalues to zero.
            eigenvalues[eigenvalues < 0] = 0
            matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Force ones on the diagonal.
            matrix = matrix - np.diag(np.diag(matrix)) + np.identity(len(matrix))
            max_value = np.abs(matrix).max()
            if max_value > 1:
                matrix /= max_value
            eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
            iterations += 1
        if iterations >= cls._MAX_CORR_ITERATIONS and np.any(eigenvalues < 0):
            LOGGER.warning("Correlation matrix did not converge to PSD within maximum iterations.")
        return matrix

    def _set_parameters(self, parameters: Dict[str, Any], default_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Set copula model parameters based on a flattened parameter dictionary.
        
        Args:
            parameters (Dict[str, Any]): Flattened dictionary of model parameters.
            default_params (Optional[Dict[str, Any]]): Default parameters to fall back on if provided.
        """
        if default_params is not None:
            default_params = unflatten_dict(default_params)
        else:
            default_params = {}
        parameters = unflatten_dict(parameters)
        if "num_rows" in parameters:
            num_rows = parameters.pop("num_rows")
            self._num_rows = 0 if pd.isna(num_rows) else max(0, int(round(num_rows)))
        if parameters:
            parameters = self._rebuild_gaussian_copula(parameters, default_params)
            self._model = multivariate.GaussianMultivariate.from_dict(parameters)
            self._fitted = True

    def _rebuild_gaussian_copula(self, model_parameters: Dict[str, Any], default_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Rebuild the model parameters to recreate a Gaussian Multivariate instance.
        
        Args:
            model_parameters (Dict[str, Any]): Restructured model parameters.
            default_params (Optional[Dict[str, Any]]): Fallback parameters if sampled parameters are invalid.
        
        Returns:
            Dict[str, Any]: Model parameters ready for GaussianMultivariate instantiation.
        """
        if default_params is None:
            default_params = {}
        columns: List[str] = []
        univariates: List[Dict[str, Any]] = []
        for column, univariate in model_parameters.get("univariates", {}).items():
            columns.append(column)
            if column in self._numerical_distributions:
                univariate_type = self._numerical_distributions[column]
            else:
                univariate_type = self.get_distribution_class(self.default_distribution)
            univariate["type"] = univariate_type
            model = getattr(univariate_type, "MODEL_CLASS", None)
            if model and hasattr(model, "_argcheck"):
                try:
                    # Extract the parameters required for _argcheck.
                    arg_names = list(inspect.signature(model._argcheck).parameters.keys())
                    to_check = {parameter: univariate[parameter] for parameter in arg_names if parameter in univariate}
                    if not model._argcheck(**to_check):
                        if "univariates" in default_params and column in default_params["univariates"]:
                            LOGGER.info(
                                f"Invalid parameters for column '{column}', falling back to default parameters."
                            )
                            univariate = default_params["univariates"][column]
                            univariate["type"] = univariate_type
                        else:
                            LOGGER.debug(f"Column '{column}' has invalid parameters.")
                except Exception as e:
                    LOGGER.error(f"Error during parameter check for column '{column}': {e}")
            else:
                LOGGER.debug(f"Univariate for column '{column}' does not have an _argcheck method.")
            if "scale" in univariate:
                univariate["scale"] = max(0, univariate["scale"])
            univariates.append(univariate)
        model_parameters["univariates"] = univariates
        model_parameters["columns"] = columns
        correlation = model_parameters.get('correlation')
        if correlation:
            model_parameters['correlation'] = (
                self._rebuild_correlation_matrix(correlation))
        else:
            model_parameters['correlation'] = [[1.0]]
        return model_parameters

    @classmethod
    def _rebuild_correlation_matrix(cls, triangular_correlation: List[List[float]]) -> List[List[float]]:
        """
        Rebuild a valid correlation matrix from its lower triangular part.
        
        Args:
            triangular_correlation (List[List[float]]): Lower triangular values (excluding the diagonal).
            
        Returns:
            List[List[float]]: Reconstructed and adjusted full correlation matrix.
        """
        size = len(triangular_correlation) + 1
        left = np.zeros((size, size))
        right = np.zeros((size, size))
        for idx, values in enumerate(triangular_correlation):
            extended_values = values + [0.0] * (size - idx - 1)
            left[idx + 1, :] = extended_values
            right[:, idx + 1] = extended_values
        correlation = left + right
        max_value = np.abs(correlation).max()
        if max_value > 1:
            correlation /= max_value
        correlation += np.identity(size)
        adjusted_corr = cls._get_nearest_correlation_matrix(correlation)
        return adjusted_corr.tolist()
