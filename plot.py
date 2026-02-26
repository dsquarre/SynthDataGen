import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def Plot(name,real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: dict):
    """
    Generates a 2D PCA scatter plot comparing Real vs Synthetic data distributions.
    """
    # 1. Select only numerical columns for math
    num_cols = [c for c, d in metadata.items() if d == 'numerical' and c in real_df.columns]
    
    real_num = real_df[num_cols].dropna()
    synth_num = synth_df[num_cols].dropna()
    
    # 2. Combine the data and add a label column
    real_num['Data Type'] = 'Real Experimental'
    synth_num['Data Type'] = 'Synthetic (AI)'
    combined_data = pd.concat([real_num, synth_num], axis=0)
    
    # 3. Standardize the data (CRITICAL: Load is in kN, Span is in mm. 
    # StandardScaler puts them on the same scale so PCA isn't biased by large numbers).
    features = combined_data.drop('Data Type', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 4. Perform PCA (Reduce 10D to 2D)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Add the 2D coordinates back to our dataframe
    combined_data['PCA Component 1'] = pca_result[:, 0]
    combined_data['PCA Component 2'] = pca_result[:, 1]
    
    # 5. Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='PCA Component 1', 
        y='PCA Component 2',
        hue='Data Type',
        palette={'Real Experimental': '#1f77b4', 'Synthetic (AI)': '#ff7f0e'},
        alpha=0.7, # Make dots slightly transparent to see overlaps
        data=combined_data,
        s=80 # Dot size
    )
    
    # 6. Formatting for the presentation
    variance_explained = sum(pca.explained_variance_ratio_) * 100
    plt.title(f'PCA: Real vs Synthetic Structural Data\n(Captures {variance_explained:.1f}% of Physical Variance)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1 (Primary Physics Trend)', fontsize=12)
    plt.ylabel('Principal Component 2 (Secondary Physics Trend)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='', fontsize=12)
    plt.tight_layout()
    
    # Save the image so you can put it in your PPT
    plt.savefig(f'{name}.png', dpi=300)
    #plt.show()