"""
Parallel PCA Analysis Module
Implements PCA part of Forkosh's parallel approach
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
file_path = r"C:\Users\andre\Desktop\LMT dim reduction toolkit\data\behavior_stats_intervals_to_analize\merged_analysis_behavior_stats_intervals.csv"  # Adjust path if needed
df = pd.read_csv(file_path)
behavior_cols = [col for col in df.columns if col not in ['mouse_id', 'interval_start']]

# Extract features and target labels
X = df[behavior_cols].values
y = df['mouse_id'].values



def analyze_pca_parallel(X, y):
    """
    Parallel PCA analysis following Forkosh approach
    
    Args:
        X: Input features matrix
        y: Target labels (mouse IDs)
        
    Returns:
        dict: PCA analysis results containing:
            - transformed_space: Data in PCA space
            - components: Principal components
            - eigenvalues: Corresponding eigenvalues
            - explained_variance_ratio: Explained variance ratio per component
            - stability_score: Stability of components
            - n_components: Number of significant components found
            - component_overlaps: Distribution overlap for each component
            - significant_components: Indices of components with < 5% overlap
            - feature_mask: Mask of selected features
    """
    # 1. Data preprocessing
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #X_scaled = X
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.05)
    X_filtered = selector.fit_transform(X_scaled)
    
    # Quantile normalization for Gaussian-like distributions
   # qt = QuantileTransformer(output_distribution='normal')
   # X_transformed = qt.fit_transform(X_filtered)
    X_transformed = X_filtered
    # 2. Initial PCA with all possible components
    pca = PCA()  # No component limit - let the data decide
    X_pca = pca.fit_transform(X_transformed)
    
    # 3. Determine significant components based on distribution overlap
    overlaps = []
    for comp_idx in range(X_pca.shape[1]):
        comp_overlaps = []
        for mouse1_idx, mouse1 in enumerate(np.unique(y)):
            for mouse2 in np.unique(y)[mouse1_idx+1:]:
                data1 = X_pca[y == mouse1, comp_idx]
                data2 = X_pca[y == mouse2, comp_idx]
                
                # Calculate distribution overlap
                hist1, bins = np.histogram(data1, bins=50, density=True)
                hist2, _ = np.histogram(data2, bins=bins, density=True)
                overlap = np.minimum(hist1, hist2).sum() * (bins[1] - bins[0])
                comp_overlaps.append(overlap)
        
        overlaps.append(np.mean(comp_overlaps))
    
    # Keep components with < 5% overlap
    significant_components = np.where(np.array(overlaps) < 0.05)[0]
    n_stable = len(significant_components)
    
    if n_stable == 0:
        # If no components meet the overlap criterion, use components with lowest overlap
        # that explain at least 80% of variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        min_components = np.where(cumsum >= 0.80)[0][0] + 1
        significant_components = np.argsort(overlaps)[:min_components]
        n_stable = len(significant_components)
    
    # 4. Final PCA with selected components
    X_pca_final = X_pca[:, significant_components]
    components_final = pca.components_[significant_components]
    eigenvalues_final = pca.explained_variance_[significant_components]
    explained_var_final = pca.explained_variance_ratio_[significant_components]
    
    # 5. Calculate stability scores
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X_pca_final[y == mouse]
        if mouse_data.shape[0] > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    
    return {
        'transformed_space': X_pca_final,
        'components': components_final,
        'eigenvalues': eigenvalues_final,
        'explained_variance_ratio': explained_var_final,
        'stability_score': np.mean(stability_scores),
        'n_components': n_stable,
        'component_overlaps': overlaps,
        'significant_components': significant_components,
        'feature_mask': selector.get_support()  # For tracking which features were kept
    } 

# Run PCA analysis
results = analyze_pca_parallel(X, y)

# Print output summary
print("Number of Significant Components:", results['n_components'])
print("Explained Variance Ratio:", results['explained_variance_ratio'])
print("Stability Score:", results['stability_score'])

# Convert transformed space to DataFrame and save
df_transformed = pd.DataFrame(results['transformed_space'])
df_transformed.to_csv("pca_transformed.csv", index=False)

# Save PCA components
df_components = pd.DataFrame(results['components'])
df_components.to_csv("pca_components.csv", index=False)

# Save Eigenvalues
df_eigenvalues = pd.DataFrame(results['eigenvalues'], columns=["Eigenvalues"])
df_eigenvalues.to_csv("pca_eigenvalues.csv", index=False)

# Save Component Overlaps
df_overlaps = pd.DataFrame(results['component_overlaps'], columns=["Overlap"])
df_overlaps.to_csv("pca_component_overlaps.csv", index=False)

print("Results saved as CSV files.")
