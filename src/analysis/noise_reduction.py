import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
# Load data
df = pd.read_csv(r"C:\Users\andre\Desktop\LMT dim reduction toolkit\data\behavior_stats_intervals_to_analize\merged_analysis_behavior_stats_intervals.csv")
print(df.columns.shape)
behavior_cols = [col for col in df.columns if col not in ['mouse_id', 'interval_start']]

# Extract features and target labels
X = df[behavior_cols].values
y = df['mouse_id'].values

# Step 2: Remove near-zero variance columns
selector = VarianceThreshold(threshold=0.1)
X = selector.fit_transform(X)
X = pd.DataFrame(X, columns=[col for col, mask in zip(behavior_cols, selector.get_support()) if mask])
print(X.columns.shape)

# Step 3: Drop derived columns (example)
X = X.drop(columns=[col for col in X.columns if "mean" in col or "median" in col])

# Step 4: Remove correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)
print(X.columns.shape)



# 2. Initial PCA with all possible components
pca = PCA()
X_pca = pca.fit_transform(X)
# Select the number of components to explain 95% of the variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

# Transform the data using the selected number of components
pca_95 = PCA(n_components=n_components_95)
X_pca_95 = pca_95.fit_transform(X)

# Print the results
print(f"Number of components to explain 95% of the variance: {n_components_95}")
print(f"Cumulative explained variance ratio: {cumulative_explained_variance[:n_components_95]}")

#
# Print PCA results
print("Explained variance ratio per component:", pca.explained_variance_ratio_)
print("Cumulative explained variance ratio:", np.cumsum(pca.explained_variance_ratio_))
print("Principal components:", pca.components_.shape)
print("Explained variance per component:", pca.explained_variance_)
print("Singular values:", pca.singular_values_)

# Optional: Print the transformed data in PCA space
print("Transformed data in PCA space (first 5 rows):\n", X_pca[:5])

import matplotlib.pyplot as plt

#
# Plot the first two principal components
# Create a color palette for the mouse IDs
unique_mouse_ids = np.unique(y)
palette = sns.color_palette("tab20", len(unique_mouse_ids))
color_map = {mouse_id: palette[i] for i, mouse_id in enumerate(unique_mouse_ids)}
colors = [color_map[mouse_id] for mouse_id in y]

# Plot the first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_95[:, 0], X_pca_95[:, 1], c=colors, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA: First Two Principal Components')

# Create a legend for the mouse IDs

plt.grid(True)
plt.show()