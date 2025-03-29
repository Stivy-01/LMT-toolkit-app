# analysis_demo_timechunks.py
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from itertools import combinations

# Time chunk configuration
TIME_CHUNKS = {
    'EarlyNight': (19, 22),
    'MidNight1': (22, 1),
    'MidNight2': (1, 4),
    'LateNight': (4, 7)
}

def create_time_chunks(df):
    """Group hourly data into temporal chunks"""
    # Extract date and hour from interval_start
    df['interval_start'] = pd.to_datetime(df['interval_start'])
    df['date'] = df['interval_start'].dt.date
    df['hour'] = df['interval_start'].dt.hour

    chunked_data = []
    
    for (mouse_id, date), group in df.groupby(['mouse_id', 'date']):
        for chunk_name, (start, end) in TIME_CHUNKS.items():
            mask = (
                (group['hour'] >= start) & 
                (group['hour'] < end) if start < end else 
                (group['hour'] >= start) | (group['hour'] < end)
            )
            chunk = group[mask].copy()
            if not chunk.empty:
                agg = chunk.drop(['mouse_id', 'date', 'interval_start', 'hour'], axis=1).sum()
                agg['mouse_id'] = mouse_id
                agg['date'] = date
                agg['time_chunk'] = chunk_name
                chunked_data.append(agg)
    
    return pd.DataFrame(chunked_data).reset_index(drop=True)

def preprocess_data(df):
    """Create 3D tensor with robust preprocessing"""
    chunked_df = create_time_chunks(df)
    
    # Feature selection setup
    features = [col for col in chunked_df.columns 
                if col not in ['mouse_id', 'date', 'time_chunk']]
    mice = chunked_df['mouse_id'].unique()
    chunks = list(TIME_CHUNKS.keys())
    
    # Initialize tensor with NaN handling
    tensor = np.full((len(mice), len(chunks), len(features)), np.nan)
    
    # Populate tensor with chunk means
    for i, mouse in enumerate(mice):
        mouse_data = chunked_df[chunked_df['mouse_id'] == mouse]
        for j, chunk in enumerate(chunks):
            chunk_data = mouse_data[mouse_data['time_chunk'] == chunk][features]
            if len(chunk_data) >= 3:  # Minimum samples per chunk
                tensor[i,j] = chunk_data.mean().values
                
    # Two-stage imputation
    mouse_medians = np.nanmedian(tensor, axis=1, keepdims=True)
    global_median = np.nanmedian(tensor)
    tensor = np.where(np.isnan(tensor), mouse_medians, tensor)
    tensor = np.where(np.isnan(tensor), global_median, tensor)
    
    # Apply robust scaling across all dimensions
    original_shape = tensor.shape
    scaler = RobustScaler()
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])
    tensor_scaled = scaler.fit_transform(tensor_2d)
    tensor = tensor_scaled.reshape(original_shape)
    
    return tensor, mice

def analyze_temporal_identity(df):
    """Main analysis pipeline with stability calculation"""
    X, mice = preprocess_data(df)
    
    # Feature selection based on temporal variance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temporal_variance = np.var(X, axis=(0,1))
    
    valid_features = temporal_variance > 1e-6
    if np.sum(valid_features) < 2:
        raise ValueError("Insufficient temporal variation in features")
    
    X_filtered = X[..., valid_features]

    # Regularized PCA
    pca = PCA(n_components=min(3, X_filtered.shape[-1]))
    X_pca = pca.fit_transform(X_filtered.reshape(-1, X_filtered.shape[-1]))
    X_pca = X_pca.reshape(X_filtered.shape[0], X_filtered.shape[1], -1)

    # Stability calculation
    stability_scores = []
    for mouse_data in X_pca:
        if mouse_data.shape[0] >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_matrix = np.corrcoef(mouse_data.reshape(-1, mouse_data.shape[1]))
                valid_corrs = corr_matrix[np.isfinite(corr_matrix)]
                
            if len(valid_corrs) > 1:
                distances = 1 - np.abs(valid_corrs)
                stability = 1 - np.nanmean(distances)
                if not np.isnan(stability):
                    stability_scores.append(stability)

    # Post-processing
    valid_scores = [s for s in stability_scores if not np.isnan(s)]
    identity_space = X_pca.mean(axis=1)[:, :min(3, X_pca.shape[-1])]
    
    # Standardize projections
    if identity_space.size > 0:
        identity_space = (identity_space - identity_space.mean(axis=0)) / identity_space.std(axis=0)
    
    return np.mean(valid_scores) if valid_scores else 0.0, identity_space

if __name__ == "__main__":
    hourly_df = pd.read_csv('merged_for_lda_hourly.csv')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg_stability, identity_space = analyze_temporal_identity(hourly_df)
    
    print(f"Temporal Stability Score: {avg_stability:.3f}")
    print("Identity Space Projections:")
    print(identity_space[0] if identity_space.size > 0 else "No valid projections")