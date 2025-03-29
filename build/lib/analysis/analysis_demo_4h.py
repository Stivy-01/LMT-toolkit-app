# analysis_demo_optimized_with_timechunks.py
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# Biological time chunk configuration
TIME_CHUNKS = {
    'EarlyNight': (19, 22),  # 7 PM - 10 PM
    'MidNight1': (22, 1),    # 10 PM - 1 AM
    'MidNight2': (1, 4),     # 1 AM - 4 AM 
    'LateNight': (4, 7)      # 4 AM - 7 AM
}
MIN_SAMPLES_PER_CHUNK = 3    # Minimum observations for valid chunk

def create_time_chunks(df):
    """Create biologically relevant time chunks with validation"""
    df['interval_start'] = pd.to_datetime(df['interval_start'])
    df['date'] = df['interval_start'].dt.date
    df['hour'] = df['interval_start'].dt.hour
    
    chunked_data = []
    
    # Daily biological grouping
    for (mouse_id, date), group in df.groupby(['mouse_id', 'date']):
        daily_chunks = []
        
        for chunk_name, (start, end) in TIME_CHUNKS.items():
            # Circadian-aware time masking
            if start < end:
                mask = (group['hour'] >= start) & (group['hour'] < end)
            else:
                mask = (group['hour'] >= start) | (group['hour'] < end)
            
            chunk = group[mask]
            if len(chunk) >= MIN_SAMPLES_PER_CHUNK:
                # Biological feature aggregation
                agg = chunk.drop(['mouse_id', 'date', 'interval_start', 'hour'], axis=1).mean()
                agg['time_chunk'] = chunk_name
                daily_chunks.append(agg.to_frame().T)
        
        if daily_chunks:
            daily_df = pd.concat(daily_chunks)
            daily_df[['mouse_id', 'date']] = mouse_id, date
            chunked_data.append(daily_df)
    
    return pd.concat(chunked_data).reset_index(drop=True)

def preprocess_data(df):
    """Biological preprocessing pipeline with time chunks"""
    chunked_df = create_time_chunks(df)
    features = [col for col in chunked_df.columns 
               if col not in ['mouse_id', 'date', 'time_chunk']]
    
    # Create 3D biological observation matrix
    mice = chunked_df['mouse_id'].unique()
    chunks = list(TIME_CHUNKS.keys())
    X = np.full((len(mice), len(chunks), len(features)), np.nan)
    
    # Mouse-specific biological data organization
    for i, mouse in enumerate(mice):
        mouse_data = chunked_df[chunked_df['mouse_id'] == mouse]
        for j, chunk in enumerate(chunks):
            chunk_data = mouse_data[mouse_data['time_chunk'] == chunk][features]
            X[i,j] = chunk_data.mean() if not chunk_data.empty else np.nan
    
    # Reshape for machine learning
    X_2d = X.reshape(-1, len(features))
    y = np.repeat(mice, len(chunks))
    
    # Biologically informed imputation
    df_2d = pd.DataFrame(X_2d, columns=features)
    df_2d = df_2d.groupby(y, group_keys=False).apply(
        lambda x: x.fillna(x.mean())  # Mouse-specific mean imputation
    )
    
    # Feature selection for biological relevance
    selector = VarianceThreshold(threshold=0.1)
    return selector.fit_transform(df_2d), y

class IdentityDomainAnalyzer:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.components_ = None

    def _compute_scatter_matrices(self, X, y):
        unique_classes = np.unique(y)
        n_features = X.shape[1]
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        overall_mean = np.mean(X, axis=0)
        
        for cls in unique_classes:
            X_cls = X[y == cls]
            if len(X_cls) > 1:
                Sw += np.cov(X_cls.T) * (len(X_cls)-1)
            if len(X_cls) > 0:
                mean_diff = (np.mean(X_cls, axis=0) - overall_mean).reshape(-1, 1)
                Sb += len(X_cls) * (mean_diff @ mean_diff.T)
                
        return Sw, Sb

    def fit(self, X, y):
        Sw, Sb = self._compute_scatter_matrices(X, y)
        Sw_reg = (Sw + Sw.T) / 2 + 1e-3 * np.eye(Sw.shape[0])
        Sb = (Sb + Sb.T) / 2
        
        eig_vals, eig_vecs = eigh(Sb, Sw_reg, check_finite=False)
        order = np.argsort(eig_vals)[::-1]
        self.components_ = eig_vecs[:, order[:self.n_components]].T
        return self

    def transform(self, X):
        return X @ self.components_.T

def analyze_identity_domain(X, y):
    ida = IdentityDomainAnalyzer(n_components=4)
    ida.fit(X, y)
    X_ids = ida.transform(X)
    
    stability_scores = []
    for mouse in np.unique(y):
        mouse_data = X_ids[y == mouse]
        if mouse_data.shape[0] > 1:
            corr_matrix = np.abs(np.corrcoef(mouse_data, rowvar=False))
            np.fill_diagonal(corr_matrix, np.nan)
            stability_scores.append(np.nanmean(corr_matrix))
    
    return np.mean(stability_scores), X_ids

if __name__ == "__main__":
    # Load biological data
    df = pd.read_csv('merged_for_lda_hourly.csv')
    
    # Preprocess with time chunks
    X, y = preprocess_data(df)
    
    # Standardization pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Remove correlated biological features
    corr_matrix = pd.DataFrame(X_scaled).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_filtered = np.delete(X_scaled, to_drop, axis=1)
    print(f"Removed {len(to_drop)} highly correlated features")
    
    # Dimensionality reduction preserving biological variance
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_filtered)
    print(f"PCA reduced dimensions from {X_filtered.shape[1]} to {X_pca.shape[1]}")
    
    # Biological identity analysis
    avg_stability, identity_space = analyze_identity_domain(X_pca, y)
    
    print(f"\nAverage Temporal Stability Score: {avg_stability:.3f}")
    print("Identity Space Projections (first 5 entries):")
    print(identity_space[:5].round(2))