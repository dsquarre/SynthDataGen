import numpy as np
import pandas as pd
from scipy.stats import mode, iqr


def proper(X_df=None, y_df=None, random_state=None):
    sample_indicies = y_df.sample(frac=1, replace=True, random_state=random_state).index
    y_df = y_df.loc[sample_indicies]

    if X_df is None:
        return y_df

    else:
        X_df = X_df.loc[sample_indicies]
        return X_df, y_df


def smooth(dtype, y_synth, y_real_min, y_real_max):
    # Ensure y_synth is numeric (float) before proceeding.
    y_synth = np.asarray(y_synth, dtype=float)

    indices = [True for _ in range(len(y_synth))]

    # Exclude from smoothing if frequency for a single value is higher than 70%
    y_synth_mode = mode(y_synth)
    if y_synth_mode.count / len(y_synth) > 0.7:
        indices = np.logical_and(indices, y_synth != y_synth_mode.mode)

    # Exclude from smoothing if data are top-coded - approximate check
    y_synth_sorted = np.sort(y_synth)
    top_coded = 10 * np.abs(y_synth_sorted[-2]) < np.abs(y_synth_sorted[-1] - y_synth_sorted[-2])
    if top_coded:
        indices = np.logical_and(indices, y_synth != y_real_max)

    # Compute bandwidth using the provided formula
    bw = 0.9 * len(y_synth[indices]) ** (-1/5) * np.minimum(np.std(y_synth[indices]), iqr(y_synth[indices]) / 1.34)

    # Apply smoothing: for values flagged by indices, sample from a normal distribution
    y_synth[indices] = np.array([np.random.normal(loc=value, scale=bw) for value in y_synth[indices]])
    if not top_coded:
        y_real_max += bw
    y_synth[indices] = np.clip(y_synth[indices], y_real_min, y_real_max)
    if dtype == 'int':
        y_synth[indices] = y_synth[indices].astype(int)

    return y_synth



def validate_numerical_distributions(numerical_distributions, metadata_columns):
    """Validate ``numerical_distributions``.

    Raise an error if it's not None or dict, or if its columns are not present in the metadata.

    Args:
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used.
        metadata_columns (list):
            Columns present in the metadata.
    """
    if numerical_distributions:
        if not isinstance(numerical_distributions, dict):
            raise TypeError('numerical_distributions can only be None or a dict instance.')

        invalid_columns = numerical_distributions.keys() - set(metadata_columns)
        if invalid_columns:
            raise SynthesizerInputError(
                'Invalid column names found in the numerical_distributions dictionary '
                f'{invalid_columns}. The column names you provide must be present '
                'in the metadata.'
            )
        
def warn_missing_numerical_distributions(numerical_distributions, processed_data_columns):
    """Raise an `UserWarning` when numerical distribution columns don't exist anymore."""
    unseen_columns = numerical_distributions.keys() - set(processed_data_columns)
    for column in unseen_columns:
        warnings.warn(
            f"Cannot use distribution '{numerical_distributions[column]}' for column "
            f"'{column}' because the column is not statistically modeled.",
            UserWarning,
        )

def flatten_array(nested, prefix=''):
    """Flatten an array as a dict.

    Args:
        nested (list, numpy.array):
            Iterable to flatten.
        prefix (str):
            Name to append to the array indices. Defaults to ``''``.

    Returns:
        dict:
            Flattened array.
    """
    result = {}
    for index in range(len(nested)):
        prefix_key = '__'.join([prefix, str(index)]) if len(prefix) else str(index)

        value = nested[index]
        if isinstance(value, (list, np.ndarray)):
            result.update(flatten_array(value, prefix=prefix_key))

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix=prefix_key))

        else:
            result[prefix_key] = value

    return result


def flatten_dict(nested, prefix=''):
    """Flatten a dictionary.

    This method returns a flatten version of a dictionary, concatenating key names with
    double underscores.

    Args:
        nested (dict):
            Original dictionary to flatten.
        prefix (str):
            Prefix to append to key name. Defaults to ``''``.

    Returns:
        dict:
            Flattened dictionary.
    """
    result = {}

    for key, value in nested.items():
        prefix_key = '__'.join([prefix, str(key)]) if len(prefix) else key

        if key in IGNORED_DICT_KEYS and not isinstance(value, (dict, list)):
            continue

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix_key))

        elif isinstance(value, (np.ndarray, list)):
            result.update(flatten_array(value, prefix_key))

        else:
            result[prefix_key] = value

    return result

def unflatten_dict(flat):
    """Transform a flattened dict into its original form.

    Args:
        flat (dict):
            Flattened dict.

    Returns:
        dict:
            Nested dict (if corresponds)
    """
    unflattened = {}

    for key, value in sorted(flat.items(), key=_key_order):
        if '__' in key:
            key, subkey = key.split('__', 1)
            subkey, name = subkey.rsplit('__', 1)

            if name.isdigit():
                column_index = int(name)
                row_index = int(subkey)

                array = unflattened.setdefault(key, [])

                if len(array) == row_index:
                    row = []
                    array.append(row)
                elif len(array) == row_index + 1:
                    row = array[row_index]
                else:
                    # This should never happen
                    raise ValueError('There was an error unflattening the extension.')

                if len(row) == column_index:
                    row.append(value)
                else:
                    # This should never happen
                    raise ValueError('There was an error unflattening the extension.')

            else:
                subdict = unflattened.setdefault(key, {})
                if subkey.isdigit() and key != 'univariates':
                    subkey = int(subkey)

                inner = subdict.setdefault(subkey, {})
                inner[name] = value

        else:
            unflattened[key] = value

    return unflattened



def extract_metadata(df: pd.DataFrame) -> dict:
    """
    Extract metadata from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        dict: A dictionary where keys are column names and values are column types.
    """
    return {col: str(df[col].dtype) for col in df.columns}



