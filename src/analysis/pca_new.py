import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

file_path = r"C:\Users\andre\Desktop\LMT dim reduction toolkit\src\data\behavior_stats_intervals_to_analize\merged_analysis_behavior_stats_intervals.csv"
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
            - cumulative_variance: Cumulative explained variance
            - scree_components: Components selected using scree plot criteria
    """
    # 1. Data preprocessing
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
    X_scaled = X
    selector = VarianceThreshold(threshold=0.1)
    X_filtered = selector.fit_transform(X_scaled)

    # Convert X_filtered back to DataFrame to use Pandas methods
    X_filtered_df = pd.DataFrame(X_filtered, columns=[col for col, mask in zip(behavior_cols, selector.get_support()) if mask])

    # Step 3: Drop derived columns (example)
    X_filtered = X_filtered_df.drop(columns=[col for col in X_filtered_df.columns if "mean" in col or "median" in col])

    # Step 4: Remove correlated features
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_filtered = X_filtered.drop(columns=to_drop)

    # Convert X_filtered back to NumPy array for PCA
    X_filtered = X_filtered.values

    # 2. Initial PCA with all possible components
    pca = PCA()
    X_pca = pca.fit_transform(X_filtered)
    
    # === NEW SCREE PLOT ANALYSIS SECTION ===
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Create scree plot visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, 
            alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, 
             where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% explained variance')
    plt.axvline(x=np.argmax(cumulative_variance >= 0.8)+1, color='g', linestyle='--')
    plt.legend()
    plt.title('Scree Plot with Cumulative Variance')
    plt.savefig('scree_plot.png')
    plt.close()
    
    # Determine components using scree plot criteria
    # Method 1: Kaiser criterion (eigenvalues > 1)
    kaiser_components = np.where(pca.explained_variance_ > 1)[0]
    
    # Method 2: Elbow point detection (95% of max gradient)
    gradients = np.gradient(pca.explained_variance_ratio_)
    elbow_point = np.argmax(gradients < 0.05*gradients.max())  # Find first major drop
    
    # Method 3: Variance threshold (80% cumulative)
    variance_threshold_components = np.where(cumulative_variance <= 0.8)[0]
    
    # Combine criteria (you can modify this priority)
    significant_components = np.unique(np.concatenate([
        kaiser_components,
        [elbow_point],
        variance_threshold_components
    ]))
    
    # 3. Calculate overlaps using Wasserstein distance (more robust than histogram overlap)
    overlaps = []
    for comp_idx in range(X_pca.shape[1]):
        comp_overlaps = []
        for mouse1_idx, mouse1 in enumerate(np.unique(y)):
            for mouse2 in np.unique(y)[mouse1_idx+1:]:
                data1 = X_pca[y == mouse1, comp_idx]
                data2 = X_pca[y == mouse2, comp_idx]
                
                # Calculate Wasserstein distance between distributions
                overlap = wasserstein_distance(data1, data2)
                comp_overlaps.append(overlap)
        
        overlaps.append(np.mean(comp_overlaps))
    
    # Modified component selection with fallback
    if len(significant_components) == 0:
        # Fallback to original overlap method
        significant_components = np.where(np.array(overlaps) < 0.2)[0]
    
    # 4. Final PCA with selected components
    X_pca_final = X_pca[:, significant_components]
    components_final = pca.components_[significant_components]
    eigenvalues_final = pca.explained_variance_[significant_components]
    explained_var_final = pca.explained_variance_ratio_[significant_components]
    
    # 5. Calculate stability scores per component
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X_pca_final[y == mouse]
        if mouse_data.shape[0] > 1:
            # Calculate correlation matrix only if there are at least two samples for the mouse
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)  # Fill diagonal with NaN to avoid self-correlation
            stability_scores.append(np.nanmean(corr_matrix))  # Mean of off-diagonal correlations
        else:
            # If only one sample, skip stability calculation for this mouse
            stability_scores.append(np.nan)  # or some other placeholder if preferred

    # Now, take the mean stability score excluding NaNs
    stability_score = np.nanmean(stability_scores)
    
    # Add cumulative variance to results
    return {
        'transformed_space': X_pca_final,
        'components': components_final,
        'eigenvalues': eigenvalues_final,
        'explained_variance_ratio': explained_var_final,
        'stability_score': (stability_score),
        'n_components': len(significant_components),
        'component_overlaps': overlaps,
        'significant_components': significant_components,
        'feature_mask': selector.get_support(),
        'cumulative_variance': cumulative_variance,
        'scree_components': {
            'kaiser': kaiser_components,
            'elbow': elbow_point,
            'variance_threshold': variance_threshold_components
        }
    }

# Run PCA analysis
results = analyze_pca_parallel(X, y)

# Print output summary
print("Number of Significant Components:", results['n_components'])
print("Explained Variance Ratio:", results['explained_variance_ratio'])
print("Cumulative Variance:", results['cumulative_variance'])
print("Stability Score:", results['stability_score'])

# Save results to CSV files
df_transformed = pd.DataFrame(results['transformed_space'])
df_transformed.to_csv("pca_transformed.csv", index=False)

df_components = pd.DataFrame(results['components'])
df_components.to_csv("pca_components.csv", index=False)

df_eigenvalues = pd.DataFrame(results['eigenvalues'], columns=["Eigenvalues"])
df_eigenvalues.to_csv("pca_eigenvalues.csv", index=False)

df_overlaps = pd.DataFrame(results['component_overlaps'], columns=["Overlap"])
df_overlaps.to_csv("pca_component_overlaps.csv", index=False)

print("Results saved as CSV files.")
