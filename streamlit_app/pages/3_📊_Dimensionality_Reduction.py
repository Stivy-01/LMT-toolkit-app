import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import gaussian_kde
import io
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Add the streamlit_app directory to the path to import utils
streamlit_app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(streamlit_app_path)

# Add the project root to the path to access src
project_path = os.path.dirname(os.path.dirname(streamlit_app_path))
sys.path.append(project_path)

from utils.analysis_utils import preprocess_features, perform_pca, perform_lda, get_feature_importance
from utils.visualization_utils import (
    plot_pca_results, 
    plot_lda_results, 
    plot_explained_variance, 
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_feature_distribution
)

# Page configuration
st.set_page_config(
    page_title="Dimensionality Reduction - LMT Toolkit",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'features' not in st.session_state:
    st.session_state.features = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'id_scores_df' not in st.session_state:
    st.session_state.id_scores_df = None
if 'significant_eigenvectors' not in st.session_state:
    st.session_state.significant_eigenvectors = None

# -----------------------------
# FO3mini LDA Identity Domains Function - From lda_fo3mini.py
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
    # Ensure eigenvector is real-valued
    eigenvector = np.real(eigenvector)
    
    # Project each mouse's behavior vectors onto the eigenvector
    scores = {}
    for mouse_id, behavior_vectors in data_dict.items():
        scores[mouse_id] = np.dot(behavior_vectors, eigenvector)
    
    # Compute pairwise overlaps using Gaussian KDE estimates
    overlaps = []
    mouse_pairs = list(combinations(scores.keys(), 2))
    
    if not mouse_pairs:
        return 100.0  # If there's only one mouse, return 100% overlap
    
    for mouse1, mouse2 in mouse_pairs:
        # Skip if either mouse has fewer than 2 data points (KDE requirement)
        if len(scores[mouse1]) < 2 or len(scores[mouse2]) < 2:
            continue
            
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
    if overlaps:
        average_overlap = np.mean(overlaps) * 100
    else:
        average_overlap = 100.0
    
    return average_overlap

def perform_fo3mini_lda(data, overlap_threshold=5):
    """
    Perform the FO3mini LDA Identity Domains analysis.
    
    Args:
        data (pandas.DataFrame): DataFrame containing the behavior data
        overlap_threshold (float): Threshold for determining significant identity domains
        
    Returns:
        tuple: (id_scores_df, significant_eigenvectors, overlaps, behavior_cols)
    """
    # Step 1: Process the data
    identifier_cols = ['mouse_id', 'interval_start']  # Add other non-behavior columns if needed
    behavior_cols = [col for col in data.columns if col not in identifier_cols]
    
    # Check for NaN or infinite values in the data
    if data[behavior_cols].isna().any().any() or np.isinf(data[behavior_cols].to_numpy()).any():
        # Clean the data by replacing NaN with 0 and infinite values with large finite values
        data_cleaned = data.copy()
        data_cleaned[behavior_cols] = data_cleaned[behavior_cols].fillna(0)
        data_cleaned[behavior_cols] = data_cleaned[behavior_cols].replace([np.inf, -np.inf], [1e10, -1e10])
    else:
        data_cleaned = data
    
    # Create the data structures
    data_dict = {}
    n_days = {}
    for mouse_id, group in data_cleaned.groupby('mouse_id'):
        vectors = group[behavior_cols].to_numpy()  # shape: (num_days, num_behaviors)
        data_dict[mouse_id] = vectors
        n_days[mouse_id] = group['interval_start'].nunique()
    
    # Determine the dimensionality
    dim = len(behavior_cols)
    
    # Step 2: Compute the Variability Matrices
    # a. Within-Individual Variability (Sigma_w)
    Sigma_w = np.zeros((dim, dim))
    for mouse_id, day_vectors in data_dict.items():
        mean_vector = np.mean(day_vectors, axis=0)
        for vector in day_vectors:
            diff = (vector - mean_vector).reshape(dim, 1)
            Sigma_w += diff @ diff.T
    
    # Check Sigma_w for numerical stability
    if np.isnan(Sigma_w).any() or np.isinf(Sigma_w).any():
        raise ValueError("Numerical instability detected in variance matrix computation. Please check your input data.")
    
    # b. Between-Individual Variability (Sigma_b)
    all_vectors = np.concatenate(list(data_dict.values()), axis=0)
    mu = np.mean(all_vectors, axis=0)
    
    Sigma_b = np.zeros((dim, dim))
    for mouse_id, day_vectors in data_dict.items():
        n_m = n_days[mouse_id]
        mean_vector = np.mean(day_vectors, axis=0)
        diff = (mean_vector - mu).reshape(dim, 1)
        Sigma_b += n_m * (diff @ diff.T)
    
    # Check Sigma_b for numerical stability
    if np.isnan(Sigma_b).any() or np.isinf(Sigma_b).any():
        raise ValueError("Numerical instability detected in between-group variance matrix. Please check your input data.")
    
    # Step 3: Implement LDA: Solve the Eigenvalue Problem
    try:
        Sigma_w_inv = np.linalg.inv(Sigma_w)
        mat = Sigma_w_inv @ Sigma_b
        
        # Check the matrix product for numerical stability
        if np.isnan(mat).any() or np.isinf(mat).any():
            raise np.linalg.LinAlgError("Matrix contains NaN or infinite values after multiplication")
            
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudoinverse
        Sigma_w_inv = np.linalg.pinv(Sigma_w)
        mat = Sigma_w_inv @ Sigma_b
        
        # Check again after using pseudoinverse
        if np.isnan(mat).any() or np.isinf(mat).any():
            # If still unstable, try adding a small regularization term
            epsilon = 1e-10
            Sigma_w_reg = Sigma_w + epsilon * np.eye(dim)
            Sigma_w_inv = np.linalg.pinv(Sigma_w_reg)
            mat = Sigma_w_inv @ Sigma_b
            
            # Final check
            if np.isnan(mat).any() or np.isinf(mat).any():
                raise ValueError("Cannot compute stable eigenvalues due to numerical issues in the data")
    
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    # Sort eigenvectors in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]  # each column is an eigenvector of length dim
    
    # Convert complex eigenvalues and eigenvectors to real
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Step 4: Selection of Significant Identity Domains (IDs)
    significant_eigenvectors = []
    overlaps = []
    
    # Check each eigenvector for overlap
    for i in range(eigenvectors.shape[1]):
        eigvec = eigenvectors[:, i]
        overlap = compute_overlap(eigvec, data_dict)
        overlaps.append(overlap)
        if overlap < overlap_threshold:
            significant_eigenvectors.append(eigvec)
    
    # Form the EID matrix from significant eigenvectors
    EID = np.array(significant_eigenvectors)
    
    # Step 5: Projection: Compute ID Scores for Each Mouse
    # Compute the mean behavioral vector for each mouse
    mouse_mean_vectors = {}
    for mouse_id, day_vectors in data_dict.items():
        mouse_mean_vectors[mouse_id] = np.mean(day_vectors, axis=0)
    
    # Project each mouse's mean behavior vector onto the EID matrix
    id_scores = {}
    for mouse_id, mean_vector in mouse_mean_vectors.items():
        if EID.shape[0] > 0:  # Only if we have significant eigenvectors
            score = EID @ mean_vector
            # Ensure the scores are real values
            score = np.real(score)
            id_scores[mouse_id] = score
        else:
            id_scores[mouse_id] = np.array([])
    
    # Step 6: Create a DataFrame with mouse ID scores
    output_data = []
    for mouse_id, scores in id_scores.items():
        row = {'mouse_id': mouse_id}
        for idx, score in enumerate(scores, start=1):
            row[f'ID_{idx}'] = score
        output_data.append(row)
    
    id_scores_df = pd.DataFrame(output_data)
    
    return id_scores_df, EID, overlaps, behavior_cols

st.title("📊 Dimensionality Reduction")
st.markdown("""
Apply dimensionality reduction techniques to extracted behavioral features.
This page allows you to analyze your data using PCA, LDA, and specialized LDA methods.
""")

# Add options for direct data upload
st.header("Data Source")
data_source = st.radio(
    "Select data source",
    ["Use extracted features from previous step", "Upload CSV file"],
    index=0
)

if data_source == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            features_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded data with {len(features_df)} rows and {len(features_df.columns)} columns")
            st.session_state.features = features_df
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            st.stop()
elif 'features' not in st.session_state or st.session_state.features is None:
    st.warning("No features available. Please extract features in the Feature Extraction page first or upload a CSV.")
    st.stop()

# Get the features
features_df = st.session_state.features

# Ensure features_df is a valid DataFrame before proceeding
if features_df is None:
    st.warning("Feature data is not valid. Please extract features or upload valid data.")
    st.stop()

st.success(f"Using data for {len(features_df)} rows with {len(features_df.columns)} columns")

# --- Method Selection ---
st.header("Dimensionality Reduction Method")
method = st.radio(
    "Select method",
    ["Principal Component Analysis (PCA)", 
     "Linear Discriminant Analysis (LDA)",
     "FO3mini (LDA Identity Domains)"],
    index=0
)

# --- Common Parameters ---
st.header("Parameters")

# Create columns layout
col1, col2 = st.columns(2)

with col1:
    # Parameters specific to each method
    if method == "FO3mini (LDA Identity Domains)":
        # For FO3mini, we only need the overlap threshold
        overlap_threshold = st.slider(
            "Overlap threshold (%)",
            min_value=1,
            max_value=50,
            value=5,
            help="Percentage overlap threshold for determining significant identity domains. Lower values result in more distinct components."
        )
        
        # Inform about automatic component selection
        st.info("The number of components will be automatically determined based on the overlap threshold.")
        
        # Normalization is not needed for FO3mini
        normalize = False
    else:
        # Number of components for PCA and standard LDA
        n_components = st.slider(
            "Number of components",
            min_value=2,
            max_value=min(20, len(features_df.columns) - 5),  # Limit to avoid too many components
            value=3
        )
        
        # Normalization option
        normalize = st.checkbox("Normalize features", value=True)

with col2:
    # Display options
    if method == "Principal Component Analysis (PCA)":
        # For PCA, we can color by categorical variables
        metadata_columns = ['genotype', 'sex', 'strain', 'setup']
        available_columns = [col for col in metadata_columns if col in features_df.columns]
        
        if available_columns:
            color_by = st.selectbox("Color points by", ["None"] + available_columns)
            color_column = None if color_by == "None" else color_by
        else:
            color_column = None
            st.info("No metadata columns available for coloring")
            
        # No target column needed for PCA
        target_column = None
        
    elif method == "Linear Discriminant Analysis (LDA)":
        # For LDA, we need a target column
        metadata_columns = ['genotype', 'sex', 'strain', 'setup']
        
        # Include mouse_id as a primary target option for LDA
        potential_target_columns = ['mouse_id'] + metadata_columns
        
        # Also check for interval columns that could be used as categorical variables
        interval_columns = [col for col in features_df.columns if 'interval' in col.lower()]
        potential_target_columns += interval_columns
        
        # Find columns with at least 2 unique values
        available_targets = [col for col in potential_target_columns 
                            if col in features_df.columns and features_df[col].nunique() > 1]
        
        if available_targets:
            # Add an explanation about using mouse_id vs other variables
            st.info("""
            **Target Variable Selection Guide:**
            - **mouse_id**: Use this to differentiate between individual mice (each mouse becomes a separate class)
            - **genotype/sex/strain**: Use these to group mice by biological characteristics
            - **interval columns**: Use these to analyze behavioral differences across time periods
            """)
            
            target_column = st.selectbox("Target variable for LDA", available_targets)
            
            # If mouse_id is selected, explain what this means
            if target_column == 'mouse_id':
                # Show info about mouse classes
                mouse_ids = features_df['mouse_id'].unique()
                st.success(f"Using {len(mouse_ids)} mice as separate classes for LDA")
                
                if 'interval_start' in features_df.columns:
                    intervals_per_mouse = features_df.groupby('mouse_id')['interval_start'].nunique()
                    avg_intervals = intervals_per_mouse.mean()
                    min_intervals = intervals_per_mouse.min()
                    
                    st.write(f"Average observations per mouse: {avg_intervals:.1f} intervals")
                    
                    if min_intervals < 2:
                        st.warning(f"Some mice have fewer than 2 interval observations, which may affect LDA performance.")
            
            # If an interval column is selected, show a warning that intervals are usually observations, not classes
            elif target_column in interval_columns:
                st.warning("""
                **Note about interval columns**: 
                Typically, intervals represent different observations for the same mouse, not separate classes.
                Using intervals as classes will group data by time period rather than by mouse identity.
                If you want to distinguish between individual mice, use 'mouse_id' as the target variable instead.
                """)
                
                # Still allow interval-based classification if the user insists
                st.info(f"Using {target_column} as a categorical variable for LDA.")
                
                # Preview the unique values in this column
                unique_values = features_df[target_column].unique()
                sample_values = unique_values[:min(5, len(unique_values))]
                st.write(f"Sample values: {', '.join(str(v) for v in sample_values)}")
                
                # Option to simplify interval labels
                simplify_intervals = st.checkbox("Simplify interval labels", value=False)
                
                if simplify_intervals:
                    # Keep the existing interval simplification code
                    if 'interval_start' in features_df.columns:
                        # Make a copy to avoid modifying the original dataframe
                        features_df = features_df.copy()
                        
                        # First, determine the base category (Day/Night) for each interval
                        def determine_base_category(interval_str):
                            if '19:00' in str(interval_str) or '20:00' in str(interval_str) or '7 PM' in str(interval_str) or '8 PM' in str(interval_str):
                                return 'Night'
                            elif '7:00' in str(interval_str) or '8:00' in str(interval_str) or '7 AM' in str(interval_str) or '8 AM' in str(interval_str):
                                return 'Day'
                            else:
                                # Try to detect if it's evening or morning based on hour
                                try:
                                    hour = int(str(interval_str).split(':')[0][-2:])
                                    return 'Night' if hour >= 19 or hour < 7 else 'Day'
                                except:
                                    return str(interval_str)
                        
                        # Apply base categorization
                        features_df['base_category'] = features_df[target_column].apply(determine_base_category)
                        
                        # Create a function to add sequential numbering to each category
                        def add_sequential_numbering():
                            # Order the intervals chronologically
                            ordered_intervals = features_df[target_column].unique()
                            try:
                                # Try to sort if they're datetime-like
                                ordered_intervals = sorted(ordered_intervals)
                            except:
                                # If sorting fails, use the order they appear in the data
                                pass
                            
                            # Map from interval to sequence number
                            day_count = 1
                            night_count = 1
                            interval_map = {}
                            
                            for interval in ordered_intervals:
                                base_cat = determine_base_category(interval)
                                if base_cat == 'Day':
                                    interval_map[interval] = f"Day{day_count}"
                                    day_count += 1
                                elif base_cat == 'Night':
                                    interval_map[interval] = f"Night{night_count}"
                                    night_count += 1
                                else:
                                    # For intervals that don't match Day/Night patterns
                                    interval_map[interval] = base_cat
                                    
                            return interval_map
                        
                        # Create the mapping
                        interval_mapping = add_sequential_numbering()
                        
                        # Apply the mapping to create the new categorical column
                        features_df['interval_category'] = features_df[target_column].map(interval_mapping)
                        
                        # Display information about the created categories
                        st.success(f"Created numbered interval categories from {target_column}")
                        categories = features_df['interval_category'].unique()
                        st.write(f"Categories: {', '.join(sorted(categories))}")
                        
                        # Show mapping between original intervals and new categories
                        mapping_df = pd.DataFrame({
                            'Original Interval': list(interval_mapping.keys()),
                            'Category': list(interval_mapping.values())
                        }).sort_values('Category')
                        st.write("Interval mapping:")
                        st.dataframe(mapping_df)
                        
                        # Update target column to use the new one
                        target_column = 'interval_category'
            
            # For LDA, we always color by the target
            color_column = target_column
        else:
            st.error("No suitable categorical variables found for LDA. Please use PCA instead.")
            st.write("LDA requires categorical variables with at least 2 distinct values.")
            st.write("Available columns: " + ", ".join(features_df.columns.tolist()))
            target_column = None
            color_column = None
            st.stop()
    
    elif method == "FO3mini (LDA Identity Domains)":
        # For FO3mini, we need to confirm the data structure
        if 'mouse_id' not in features_df.columns:
            st.error("The data must contain a 'mouse_id' column for FO3mini analysis.")
            st.stop()
            
        if 'interval_start' not in features_df.columns:
            st.warning("The data should contain an 'interval_start' column for optimal FO3mini analysis.")
        
        # Check if we have enough mice and days
        mouse_count = features_df['mouse_id'].nunique()
        
        if 'interval_start' in features_df.columns:
            avg_days = features_df.groupby('mouse_id')['interval_start'].nunique().mean()
            st.info(f"Data contains {mouse_count} mice with an average of {avg_days:.1f} days per mouse.")
        else:
            st.info(f"Data contains {mouse_count} mice (interval information not found).")
        
        # No target column needed for FO3mini
        target_column = None
        color_column = None

# --- Feature Selection ---
st.header("Feature Selection")

# Get behavioral features
behavior_features = [col for col in features_df.columns if col.startswith('count_') or col.startswith('avg_duration_')]

# Option to select specific features
use_all_features = st.checkbox("Use all behavioral features", value=True)

if not use_all_features:
    selected_features = st.multiselect(
        "Select features to include",
        behavior_features,
        default=behavior_features[:min(10, len(behavior_features))]
    )
    
    if not selected_features:
        st.warning("No features selected. Using all features.")
        selected_features = behavior_features
else:
    selected_features = behavior_features

# Always exclude metadata columns except target
exclude_columns = ['animal_id']
for col in ['genotype', 'sex', 'strain', 'setup']:
    if col in features_df.columns and col != target_column:
        exclude_columns.append(col)

# --- Run Analysis Button ---
st.header("Run Analysis")

if st.button("Run Dimensionality Reduction", type="primary"):
    with st.spinner(f"Running {method}..."):
        try:
            # Preprocess features
            X, y, feature_names, scaler = preprocess_features(
                features_df, 
                target_column=target_column,
                exclude_columns=exclude_columns,
                normalize=normalize
            )
            
            # Store in session state
            st.session_state.processed_data = {
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'target_column': target_column,
                'color_column': color_column
            }
            
            # Perform dimensionality reduction
            if method == "Principal Component Analysis (PCA)":
                # Run PCA
                transformed_data, model, explained_variance = perform_pca(X, n_components=n_components)
                
                # Store results in session state
                st.session_state.processed_data['method'] = 'PCA'
                st.session_state.processed_data['transformed_data'] = transformed_data
                st.session_state.processed_data['model'] = model
                st.session_state.processed_data['explained_variance'] = explained_variance
                
                # Display results
                st.success(f"PCA completed successfully with {n_components} components")
                
                # Create columns layout for plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # PCA Scatter Plot
                    st.subheader("PCA Results")
                    pca_fig = plot_pca_results(
                        transformed_data, 
                        labels=y, 
                        feature_df=features_df, 
                        color_column=color_column
                    )
                    st.plotly_chart(pca_fig, use_container_width=True)
                
                with col2:
                    # Explained Variance Plot
                    st.subheader("Explained Variance")
                    variance_fig = plot_explained_variance(
                        explained_variance,
                        title="PCA Explained Variance Ratio"
                    )
                    st.plotly_chart(variance_fig, use_container_width=True)
                
                # Feature Importance
                st.subheader("Feature Importance")
                importance_df = get_feature_importance(model, feature_names)
                
                # Create tabs for each component
                importance_tabs = st.tabs([f"Component {i+1}" for i in range(min(3, n_components))])
                
                importance_figs = plot_feature_importance(importance_df, n_features=min(20, len(feature_names)))
                
                for i, tab in enumerate(importance_tabs):
                    if i < len(importance_figs):
                        component_name = f"Component {i+1}"
                        with tab:
                            st.plotly_chart(importance_figs[component_name], use_container_width=True)
                
                # Correlation Heatmap
                st.subheader("Feature Correlation Heatmap")
                corr_fig = plot_correlation_heatmap(features_df, exclude_cols=exclude_columns)
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Save PCA results
                st.subheader("Save Results")
                
                # Combine PCA results with original data
                pca_results = pd.DataFrame(
                    transformed_data, 
                    columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])]
                )
                pca_results.index = features_df.index
                
                # Add metadata columns
                for col in ['animal_id', 'genotype', 'sex', 'strain', 'setup']:
                    if col in features_df.columns:
                        pca_results[col] = features_df[col]
                
                # Display results
                st.dataframe(pca_results)
                
                # Download button
                csv = pca_results.to_csv(index=True)
                st.download_button(
                    label="Download PCA results as CSV",
                    data=csv,
                    file_name="pca_results.csv",
                    mime="text/csv"
                )
                
            elif method == "Linear Discriminant Analysis (LDA)":
                # Run LDA
                transformed_data, model, explained_variance = perform_lda(X, y, n_components=n_components)
                
                # Check if LDA was successful
                if transformed_data is None:
                    st.error("LDA failed. Please check the error messages above.")
                    st.stop()
                
                # Store results in session state
                st.session_state.processed_data['method'] = 'LDA'
                st.session_state.processed_data['transformed_data'] = transformed_data
                st.session_state.processed_data['model'] = model
                st.session_state.processed_data['explained_variance'] = explained_variance
                
                # Display results
                st.success(f"LDA completed successfully with {n_components} components")
                
                # Create columns layout for plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # LDA Scatter Plot
                    st.subheader("LDA Results")
                    lda_fig = plot_lda_results(
                        transformed_data, 
                        labels=y, 
                        feature_df=features_df, 
                        color_column=color_column
                    )
                    st.plotly_chart(lda_fig, use_container_width=True)
                
                with col2:
                    # If explained variance is available, show it
                    if explained_variance is not None:
                        st.subheader("Explained Variance")
                        variance_fig = plot_explained_variance(
                            explained_variance,
                            title="LDA Explained Variance Ratio"
                        )
                        st.plotly_chart(variance_fig, use_container_width=True)
                    else:
                        # Show feature distribution by target
                        st.subheader(f"Feature Distribution by {target_column}")
                        # Find the most discriminative feature
                        if len(feature_names) > 0:
                            feature_to_plot = st.selectbox(
                                "Select feature to plot", 
                                feature_names
                            )
                            dist_fig = plot_feature_distribution(
                                features_df, 
                                feature_to_plot, 
                                group_by=target_column
                            )
                            st.plotly_chart(dist_fig, use_container_width=True)
                
                # Feature Importance
                st.subheader("Feature Importance")
                importance_df = get_feature_importance(model, feature_names)
                
                if not importance_df.empty:
                    # Create tabs for each component
                    importance_tabs = st.tabs([f"Component {i+1}" for i in range(min(3, n_components))])
                    
                    importance_figs = plot_feature_importance(importance_df, n_features=min(20, len(feature_names)))
                    
                    for i, tab in enumerate(importance_tabs):
                        if i < len(importance_figs):
                            component_name = f"Component {i+1}"
                            with tab:
                                st.plotly_chart(importance_figs[component_name], use_container_width=True)
                else:
                    st.info("Feature importance information not available for this LDA implementation")
                
                # Correlation Heatmap
                st.subheader("Feature Correlation Heatmap")
                corr_fig = plot_correlation_heatmap(features_df, exclude_cols=exclude_columns)
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Save LDA results
                st.subheader("Save Results")
                
                # Combine LDA results with original data
                lda_results = pd.DataFrame(
                    transformed_data, 
                    columns=[f'LD{i+1}' for i in range(transformed_data.shape[1])]
                )
                lda_results.index = features_df.index
                
                # Add metadata columns
                for col in ['animal_id', 'genotype', 'sex', 'strain', 'setup']:
                    if col in features_df.columns:
                        lda_results[col] = features_df[col]
                
                # Display results
                st.dataframe(lda_results)
                
                # Download button
                csv = lda_results.to_csv(index=True)
                st.download_button(
                    label="Download LDA results as CSV",
                    data=csv,
                    file_name="lda_results.csv",
                    mime="text/csv"
                )
                
            elif method == "FO3mini (LDA Identity Domains)":
                # Run FO3mini LDA Identity Domains
                try:
                    # Check if there's a clean division of mice
                    if 'mouse_id' not in features_df.columns:
                        st.error("Mouse ID column not found. Please ensure your data contains a 'mouse_id' column.")
                        st.stop()
                        
                    if 'interval_start' not in features_df.columns:
                        st.error("Interval start column not found. Please ensure your data contains an 'interval_start' column.")
                        st.stop()
                    
                    # Check for minimum data requirements
                    if len(features_df['mouse_id'].unique()) < 2:
                        st.error("FO3mini analysis requires data from at least 2 different mice.")
                        st.stop()
                        
                    for mouse_id, group in features_df.groupby('mouse_id'):
                        if len(group) < 2:
                            st.error(f"Mouse {mouse_id} has less than 2 observations, which is insufficient for variance calculation.")
                            st.stop()
                
                    id_scores_df, EID, overlaps, behavior_cols = perform_fo3mini_lda(features_df, overlap_threshold=overlap_threshold)
                    
                    # Store results in session state
                    st.session_state.processed_data['method'] = 'FO3mini'
                    st.session_state.processed_data['id_scores_df'] = id_scores_df
                    st.session_state.processed_data['EID'] = EID
                    st.session_state.processed_data['overlaps'] = overlaps
                    st.session_state.processed_data['behavior_cols'] = behavior_cols
                    
                    # Display results
                    num_ids = EID.shape[0]
                    st.success(f"FO3mini analysis completed with {num_ids} significant Identity Domains (IDs) identified")
                
                except ValueError as e:
                    st.error(f"Error in FO3mini analysis: {str(e)}")
                    st.info("Try removing columns with missing or infinite values, or columns with very little variance.")
                    st.stop()
                except np.linalg.LinAlgError as e:
                    st.error(f"Linear algebra error: {str(e)}")
                    st.info("This usually happens with ill-conditioned matrices. Try removing highly correlated features or features with very little variance.")
                    st.stop()
                except Exception as e:
                    st.error(f"Unexpected error in FO3mini analysis: {str(e)}")
                    st.info("Please check your data for issues or try with a different dataset.")
                    st.stop()
                
                # Display a summary of the results
                st.subheader("Analysis Summary")
                st.write(f"- Total eigenvectors analyzed: {len(overlaps)}")
                st.write(f"- Significant Identity Domains (IDs) found: {num_ids}")
                st.write(f"- Overlap threshold used: {overlap_threshold}%")
                
                if num_ids == 0:
                    st.warning("No significant Identity Domains found. Try lowering the overlap threshold.")
                    
                # Create expandable section for detailed overlap information
                with st.expander("Detailed Overlap Information", expanded=False):
                    # Create a DataFrame for the overlaps
                    overlap_df = pd.DataFrame({
                        'Eigenvector': range(1, len(overlaps) + 1),
                        'Overlap (%)': overlaps,
                        'Significant': [overlap < overlap_threshold for overlap in overlaps]
                    })
                    
                    # Display the overlap DataFrame
                    st.dataframe(overlap_df)
                    
                    # Create a bar chart of overlaps
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(
                        overlap_df['Eigenvector'], 
                        overlap_df['Overlap (%)'],
                        color=[('green' if sig else 'gray') for sig in overlap_df['Significant']]
                    )
                    
                    # Add a horizontal line for the threshold
                    ax.axhline(y=overlap_threshold, color='red', linestyle='--', alpha=0.7)
                    ax.text(0.02, overlap_threshold + 1, f'Threshold: {overlap_threshold}%', 
                            color='red', fontsize=10)
                    
                    # Style the plot
                    ax.set_xlabel('Eigenvector')
                    ax.set_ylabel('Overlap (%)')
                    ax.set_title('Overlap Percentage for Each Eigenvector')
                    ax.grid(True, alpha=0.3)
                    
                    # Show the plot
                    st.pyplot(fig)
                
                # Only display ID scores if we have significant IDs
                if num_ids > 0:
                    # Create columns layout for ID score plots
                    st.subheader("Identity Domain (ID) Scores")
                    
                    # If we have more than one ID, create a scatter plot
                    if num_ids >= 2:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Create a scatter plot of the first two IDs
                        scatter = ax.scatter(
                            id_scores_df['ID_1'], 
                            id_scores_df['ID_2'],
                            s=100, 
                            alpha=0.7
                        )
                        
                        # Add mouse IDs as labels
                        for i, txt in enumerate(id_scores_df['mouse_id']):
                            ax.annotate(
                                txt, 
                                (id_scores_df['ID_1'].iloc[i], id_scores_df['ID_2'].iloc[i]),
                                xytext=(5, 5), 
                                textcoords='offset points'
                            )
                        
                        # Style the plot
                        ax.set_xlabel('ID 1')
                        ax.set_ylabel('ID 2')
                        ax.set_title('Mouse Positions in ID Space')
                        ax.grid(True, alpha=0.3)
                        
                        # Show the plot
                        st.pyplot(fig)
                        
                        # If we have more than 2 IDs, offer pairwise plots
                        if num_ids > 2:
                            with st.expander("View additional ID pairwise plots", expanded=False):
                                # Create a dropdown to select which IDs to plot
                                id_cols = [f'ID_{i+1}' for i in range(num_ids)]
                                col1, col2 = st.columns(2)
                                with col1:
                                    x_id = st.selectbox("X-axis ID", id_cols, index=0)
                                with col2:
                                    # Default to the second ID
                                    default_y = min(1, len(id_cols)-1)
                                    y_id = st.selectbox("Y-axis ID", id_cols, index=default_y)
                                
                                # Create the selected pairwise plot
                                fig, ax = plt.subplots(figsize=(10, 8))
                                
                                # Create the scatter plot
                                scatter = ax.scatter(
                                    id_scores_df[x_id], 
                                    id_scores_df[y_id],
                                    s=100, 
                                    alpha=0.7
                                )
                                
                                # Add mouse IDs as labels
                                for i, txt in enumerate(id_scores_df['mouse_id']):
                                    ax.annotate(
                                        txt, 
                                        (id_scores_df[x_id].iloc[i], id_scores_df[y_id].iloc[i]),
                                        xytext=(5, 5), 
                                        textcoords='offset points'
                                    )
                                
                                # Style the plot
                                ax.set_xlabel(x_id)
                                ax.set_ylabel(y_id)
                                ax.set_title(f'Mouse Positions in {x_id} vs {y_id} Space')
                                ax.grid(True, alpha=0.3)
                                
                                # Show the plot
                                st.pyplot(fig)
                        
                        # For just one ID, create a bar chart
                        elif num_ids == 1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Sort by ID_1 value
                            sorted_df = id_scores_df.sort_values('ID_1')
                            
                            # Create a bar chart
                            bars = ax.bar(
                                range(len(sorted_df)), 
                                sorted_df['ID_1'],
                                tick_label=sorted_df['mouse_id']
                            )
                            
                            # Style the plot
                            ax.set_xlabel('Mouse ID')
                            ax.set_ylabel('ID 1 Score')
                            ax.set_title('Mouse Scores for Identity Domain 1')
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45, ha='right')
                            
                            # Show the plot
                            st.pyplot(fig)
                        
                        # Display feature importance for each ID
                        st.subheader("Feature Importance for Identity Domains")
                        
                        with st.expander("View feature importance", expanded=False):
                            # Select which ID to show feature importance for
                            if num_ids > 1:
                                selected_id = st.selectbox(
                                    "Select Identity Domain", 
                                    [f"ID {i+1}" for i in range(num_ids)],
                                    index=0
                                )
                                id_index = int(selected_id.split(" ")[1]) - 1
                            else:
                                id_index = 0
                            
                            # Get the eigenvector for the selected ID
                            eigenvector = EID[id_index]
                            
                            # Create a DataFrame of feature importance
                            importance_df = pd.DataFrame({
                                'Feature': behavior_cols,
                                'Importance': np.abs(eigenvector)
                            }).sort_values('Importance', ascending=False)
                            
                            # Display the top features
                            st.write(f"Top features contributing to {selected_id if num_ids > 1 else 'ID 1'}:")
                            st.dataframe(importance_df.head(15))
                            
                            # Create a bar chart of the top features
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Plot the top 15 features
                            top_features = importance_df.head(15).sort_values('Importance')
                            ax.barh(top_features['Feature'], top_features['Importance'])
                            
                            # Style the plot
                            ax.set_xlabel('Importance (absolute value)')
                            ax.set_title(f'Top Features for {selected_id if num_ids > 1 else "ID 1"}')
                            ax.grid(True, alpha=0.3)
                            
                            # Show the plot
                            st.pyplot(fig)
                        
                        # Save FO3mini results
                        st.subheader("Save Results")
                        
                        # Display ID scores
                        st.dataframe(id_scores_df)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download ID scores
                            csv = id_scores_df.to_csv(index=False)
                            st.download_button(
                                label="Download ID scores as CSV",
                                data=csv,
                                file_name="id_scores.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Download overlap data
                            overlap_csv = pd.DataFrame({
                                'Eigenvector': range(1, len(overlaps) + 1),
                                'Overlap (%)': overlaps,
                                'Significant': [overlap < overlap_threshold for overlap in overlaps]
                            }).to_csv(index=False)
                            
                            st.download_button(
                                label="Download overlap data as CSV",
                                data=overlap_csv,
                                file_name="overlap_data.csv",
                                mime="text/csv"
                            )
                
        except Exception as e:
            st.error(f"Error performing dimensionality reduction: {str(e)}")
            st.exception(e)

# Display existing results if available
elif 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    st.header("Previous Analysis Results")
    
    processed_data = st.session_state.processed_data
    method = processed_data.get('method', '')
    
    if method:
        st.info(f"Using previously calculated {method} results")
        
        # Display basic visualization of previous results
        transformed_data = processed_data.get('transformed_data')
        model = processed_data.get('model')
        explained_variance = processed_data.get('explained_variance')
        y = processed_data.get('y')
        feature_names = processed_data.get('feature_names', [])
        color_column = processed_data.get('color_column')
        
        if transformed_data is not None:
            # Create columns layout for plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter Plot
                st.subheader(f"{method} Results")
                if method == 'PCA':
                    fig = plot_pca_results(
                        transformed_data, 
                        labels=y, 
                        feature_df=features_df, 
                        color_column=color_column
                    )
                elif method == 'LDA':
                    fig = plot_lda_results(
                        transformed_data, 
                        labels=y, 
                        feature_df=features_df, 
                        color_column=color_column
                    )
                else:
                    # For FO3mini method, get id_scores_df from session state
                    id_scores_df = processed_data.get('id_scores_df')
                    if id_scores_df is not None:
                        fig = plot_feature_distribution(
                            id_scores_df, 
                            'ID_1', 
                            group_by='mouse_id'
                        )
                    else:
                        st.warning("ID scores data not found. Please rerun the analysis.")
                        fig = None
                
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Explained Variance Plot
                if explained_variance is not None:
                    st.subheader("Explained Variance")
                    variance_fig = plot_explained_variance(
                        explained_variance,
                        title=f"{method} Explained Variance Ratio"
                    )
                    st.plotly_chart(variance_fig, use_container_width=True)

            # Additional plots are available in the full analysis
            st.markdown("Run the analysis again to see detailed feature importance and correlation plots")

# --- Next Steps ---
st.markdown("---")
st.header("Next Steps")
st.markdown("""
After running dimensionality reduction, you can:
1. Explore the visualizations to identify patterns in the data
2. Download the results for further analysis
3. Try different parameters or methods to compare results
4. Use the feature importance plots to identify key behavioral features
""")

# Display footer
st.markdown("---")
st.markdown("© 2025 LMT Dimensionality Reduction Toolkit") 