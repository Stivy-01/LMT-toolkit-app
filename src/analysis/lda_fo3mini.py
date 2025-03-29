import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import gaussian_kde

# -----------------------------
# Step 1: Data Input (No Normalization)
# -----------------------------

# Load the behavioral data from a CSV file.
# Assumes the CSV file "behavior_data.csv" contains columns:
# 'mouse_id', 'interval_start', and additional columns for behaviors (with their own names).
data = pd.read_csv("behavior_data.csv")

# Determine the list of behavior columns by excluding known identifier columns.
identifier_cols = ['mouse_id', 'interval_start']  # Add other non-behavior columns if needed.
behavior_cols = [col for col in data.columns if col not in identifier_cols]

# Create two dictionaries:
#   - data_dict: maps each mouse_id to its daily behavior vectors (each row is a behavior vector).
#   - n_days: maps each mouse_id to the number of unique days (using the "interval_start" column).
data_dict = {}
n_days = {}
for mouse_id, group in data.groupby('mouse_id'):
    vectors = group[behavior_cols].to_numpy()  # shape: (num_days, num_behaviors)
    data_dict[mouse_id] = vectors
    n_days[mouse_id] = group['interval_start'].nunique()

# Determine the dimensionality (number of behavior columns)
dim = len(behavior_cols)

# -----------------------------
# Step 2: Compute the Variability Matrices
# -----------------------------

# a. Compute Within-Individual Variability (Sigma_w)
# Sigma_w = sum_m sum_d (x_m,d - mean_x_m)(x_m,d - mean_x_m)^T
Sigma_w = np.zeros((dim, dim))
for mouse_id, day_vectors in data_dict.items():
    mean_vector = np.mean(day_vectors, axis=0)
    for vector in day_vectors:
        diff = (vector - mean_vector).reshape(dim, 1)
        Sigma_w += diff @ diff.T

# b. Compute Between-Individual Variability (Sigma_b)
# We use: Sigma_b = sum_m n_m * (mean_m - mu)(mean_m - mu)^T,
# where n_m is the number of unique days for mouse m.
all_vectors = np.concatenate(list(data_dict.values()), axis=0)
mu = np.mean(all_vectors, axis=0)

Sigma_b = np.zeros((dim, dim))
for mouse_id, day_vectors in data_dict.items():
    n_m = n_days[mouse_id]
    mean_vector = np.mean(day_vectors, axis=0)
    diff = (mean_vector - mu).reshape(dim, 1)
    Sigma_b += n_m * (diff @ diff.T)

# -----------------------------
# Step 3: Implement LDA: Solve the Eigenvalue Problem
# -----------------------------

# Solve the generalized eigenvalue problem: Sigma_w^{-1} Sigma_b
Sigma_w_inv = np.linalg.inv(Sigma_w)
mat = Sigma_w_inv @ Sigma_b

eigenvalues, eigenvectors = np.linalg.eig(mat)
# Sort eigenvectors in descending order of eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]  # each column is an eigenvector of length dim

# -----------------------------
# Step 4: Selection of Significant Identity Domains (IDs)
# -----------------------------

def compute_overlap(eigenvector, data_dict):
    """
    Compute the average percentage overlap for a given eigenvector (ID) across all mice.
    
    For each mouse, the behavior vectors are projected onto the eigenvector
    to obtain a distribution of scores. The overlap between pairs of mice
    is estimated by calculating the area of intersection of their estimated
    density functions (using Gaussian KDE). The function returns the average
    overlap percentage across all pairs.
    
    Args:
        eigenvector (numpy.ndarray): The eigenvector representing a potential ID.
        data_dict (dict): Dictionary containing behavioral data for each mouse.
    
    Returns:
        float: Average overlap percentage between mouse score distributions.
    """
    # Project each mouse's behavior vectors onto the eigenvector
    scores = {}
    for mouse_id, behavior_vectors in data_dict.items():
        scores[mouse_id] = np.dot(behavior_vectors, eigenvector)
    
    # Compute pairwise overlaps using Gaussian KDE estimates
    overlaps = []
    mouse_pairs = list(combinations(scores.keys(), 2))
    
    for mouse1, mouse2 in mouse_pairs:
        kde1 = gaussian_kde(scores[mouse1])
        kde2 = gaussian_kde(scores[mouse2])
        
        # Determine a common range for evaluation
        range_min = min(np.min(scores[mouse1]), np.min(scores[mouse2]))
        range_max = max(np.max(scores[mouse1]), np.max(scores[mouse2]))
        score_range = np.linspace(range_min, range_max, 1000)
        
        density1 = kde1(score_range)
        density2 = kde2(score_range)
        
        # Calculate the area of overlap between the two density functions
        overlap_area = np.trapz(np.minimum(density1, density2), score_range)
        overlaps.append(overlap_area)
    
    # Average the overlap areas and convert to percentage
    average_overlap = np.mean(overlaps) * 100
    return average_overlap

# Use an overlap threshold of 5%
overlap_threshold = 5  # 5% threshold
significant_eigenvectors = []

# Check each eigenvector for overlap
for i in range(eigenvectors.shape[1]):
    eigvec = eigenvectors[:, i]
    overlap = compute_overlap(eigvec, data_dict)
    print(f"Eigenvector {i+1}: overlap = {overlap:.2f}%")
    if overlap < overlap_threshold:
        significant_eigenvectors.append(eigvec)

# Form the EID matrix from significant eigenvectors
EID = np.array(significant_eigenvectors)

# -----------------------------
# Step 5: Projection: Compute ID Scores for Each Mouse
# -----------------------------

# Compute the mean behavioral vector for each mouse.
mouse_mean_vectors = {}
for mouse_id, day_vectors in data_dict.items():
    mouse_mean_vectors[mouse_id] = np.mean(day_vectors, axis=0)

# Project each mouse's mean behavior vector onto the EID matrix.
# This gives each mouse a score vector with length equal to the number of significant IDs.
id_scores = {}
for mouse_id, mean_vector in mouse_mean_vectors.items():
    score = EID @ mean_vector  # (K, dim) dot (dim,) => (K,), where K = number of significant IDs.
    id_scores[mouse_id] = score

# -----------------------------
# Step 6: Output: Relationship of Each Mouse with the IDs
# -----------------------------

print("Number of significant Identity Domains (IDs):", EID.shape[0])
for mouse_id, score in id_scores.items():
    print(f"Mouse {mouse_id}: ID scores = {score}")

# -----------------------------
# Step 7: Create a CSV File with Mouse Score IDs for Plotting
# -----------------------------

output_data = []
for mouse_id, scores in id_scores.items():
    row = {'mouse_id': mouse_id}
    for idx, score in enumerate(scores, start=1):
        row[f'ID_{idx}'] = score
    output_data.append(row)

df_scores = pd.DataFrame(output_data)
df_scores.to_csv("mouse_id_scores.csv", index=False)
print("CSV file 'mouse_id_scores.csv' has been created with mouse IDs and their corresponding ID scores.")
