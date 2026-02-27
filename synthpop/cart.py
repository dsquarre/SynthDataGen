import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
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

class CARTMethod:
    """
    Sequential CART Synthesizer.
    Generates data column-by-column. 
    Column 1 is sampled from the marginal distribution.
    Column N is predicted using a Decision Tree trained on Columns 1 through (N-1).
    """
    def __init__(self, metadata, smoothing=False, proper=False, minibucket=5, random_state=None, tree_params=None):
        self.metadata = metadata
        self.smoothing = smoothing
        self.proper = proper
        self.minibucket = minibucket
        self.random_state = random_state
        self.tree_params = tree_params or {}
        
        self.models = {}         
        self.leaf_values = {}    
        self.y_bounds = {}       
        self.columns_ = []       # To strictly store the sequential order of columns
        self.marginal_dist = []  # To store the real distribution for the very first column
        self.fitted = False
        self._train_data = None  

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits models sequentially.
        Col 1: Stores marginal distribution.
        Col N: Trains a CART model using Cols 1 to N-1 as features (X).
        """
        self._train_data = data.copy()
        self.columns_ = data.columns.tolist()
        
        # 1. Store the exact empirical distribution for the first column
        first_col = self.columns_[0]
        self.marginal_dist = data[first_col].values
        
        # 2. Train sequential decision trees for all subsequent columns
        for i in range(1, len(self.columns_)):
            target_col = self.columns_[i]
            predictor_cols = self.columns_[:i] # Use ONLY previously generated columns
            
            X = data[predictor_cols]
            y = data[target_col]
            
            if self.proper:
                X, y = proper(X_df=X, y_df=y, random_state=self.random_state)
                
            dtype = self.metadata.get(target_col, "numerical")
            
            # Initialize appropriate tree type
            if dtype in ["numerical", "datetime", "timedelta"]:
                model = DecisionTreeRegressor(min_samples_leaf=self.minibucket, random_state=self.random_state, **self.tree_params)
                self.y_bounds[target_col] = (np.min(y.to_numpy()), np.max(y.to_numpy()))
            elif dtype in ["categorical", "boolean"]:
                model = DecisionTreeClassifier(min_samples_leaf=self.minibucket, random_state=self.random_state, **self.tree_params)
            else:
                warnings.warn(f"Unknown data type for column '{target_col}', defaulting to regressor.")
                model = DecisionTreeRegressor(min_samples_leaf=self.minibucket, random_state=self.random_state, **self.tree_params)
                self.y_bounds[target_col] = (np.min(y.to_numpy()), np.max(y.to_numpy()))
                
            X_np = X.to_numpy()
            y_np = y.to_numpy()
            
            # Train the model
            model.fit(X_np, y_np)
            self.models[target_col] = model
            
            # Map training data to leaves for the sampling phase later
            leaves = model.apply(X_np)
            df_leaves = pd.DataFrame({'leaf': leaves, 'y': y_np})
            leaf_dict = df_leaves.groupby('leaf')['y'].apply(lambda arr: arr.values).to_dict()
            self.leaf_values[target_col] = leaf_dict
            
        self.fitted = True

    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        Generates synthetic data piece-by-piece.
        """
        if not self.fitted:
            raise ValueError("The model must be fitted before generating synthetic data.")
            
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Create an empty dataframe to hold our new synthetic rows
        synthetic_data = pd.DataFrame(index=range(num_rows))
        
        # Step 1: Generate the first column randomly from its marginal distribution
        first_col = self.columns_[0]
        synthetic_data[first_col] = np.random.choice(self.marginal_dist, size=num_rows, replace=True)
        
        # Step 2: Loop through the remaining columns and predict them sequentially
        for i in range(1, len(self.columns_)):
            target_col = self.columns_[i]
            predictor_cols = self.columns_[:i] # Look at what we just generated
            
            # The input (X) is the freshly generated synthetic columns
            X_test = synthetic_data[predictor_cols].to_numpy()
            model = self.models[target_col]
            dtype = self.metadata.get(target_col, "numerical")
            
            # Find which leaf each synthetic row falls into
            leaves_pred = model.apply(X_test)
            y_pred = np.empty(num_rows, dtype=object)
            
            leaf_indices = pd.DataFrame({'leaf': leaves_pred, 'index': range(num_rows)}) \
                             .groupby('leaf')['index'].apply(list).to_dict()
                             
            # For each leaf, randomly sample a Y-value from the original training data that lived in that leaf
            for leaf, indices in leaf_indices.items():
                if leaf in self.leaf_values[target_col]:
                    samples = np.random.choice(self.leaf_values[target_col][leaf], size=len(indices), replace=True)
                else:
                    # Fallback if a synthetic combination creates a path unseen in training
                    samples = model.predict(X_test[indices])
                    
                for idx_enum, idx in enumerate(indices):
                    y_pred[idx] = samples[idx_enum]
                    
            # Apply Gaussian noise (smoothing) if requested
            if self.smoothing and dtype in ["numerical", "datetime", "timedelta"]:
                y_real_min, y_real_max = self.y_bounds[target_col]
                y_pred = smooth(dtype, y_pred, y_real_min, y_real_max)
                
            # Add the newly predicted column to the dataset so the NEXT column can use it
            synthetic_data[target_col] = y_pred
            
        return synthetic_data