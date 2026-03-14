import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def Plot(name, real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict):
    """
    Generates a 2D PCA scatter plot comparing Real vs Synthetic data distributions.
    """
    # 1. Select only numerical columns for math
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    # Use .copy() to avoid SettingWithCopyWarning later
    real_num = real_df[num_cols].dropna().copy()
    synth_num = synth_df[num_cols].dropna().copy()
    
    # 2. Combine the data and add a label column
    real_label = f'Real'
    synth_label = f'Synthetic'
    
    real_num['Data Type'] = real_label
    synth_num['Data Type'] = synth_label
    combined_data = pd.concat([real_num, synth_num], axis=0)
    
    # 3. Standardize the data
    features = combined_data.drop('Data Type', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 4. Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Use .loc to safely add results back
    combined_data['PCA Component 1'] = pca_result[:, 0]
    combined_data['PCA Component 2'] = pca_result[:, 1]
    
    # 5. Plotting
    plt.figure(figsize=(10, 7))
    
    # FIXED: The dictionary syntax and f-string were malformed in the original
    palette_colors = {real_label: '#1f77b4', synth_label: '#ff7f0e'}
    
    sns.scatterplot(
        x='PCA Component 1', 
        y='PCA Component 2',
        hue='Data Type',
        palette=palette_colors,
        alpha=0.6, 
        data=combined_data,
        s=80,
        edgecolor='w', # Adds a white border to dots to make them pop
        linewidth=0.5
    )
    
    # 6. Formatting
    variance_explained = sum(pca.explained_variance_ratio_) * 100
    plt.title(f'PCA: Real vs Synthetic Structural Data ({name})\n(Captures {variance_explained:.1f}% of Physical Variance)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1 (Primary Physics Trend)', fontsize=12)
    plt.ylabel('Principal Component 2 (Secondary Physics Trend)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='', fontsize=12)
    plt.tight_layout()
    
    # Save the image
    plt.savefig(f'{name}_PCA.png', dpi=300)
    print(f"Plot saved as {name}_PCA.png")
    # plt.show()
