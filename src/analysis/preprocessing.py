"""
Preprocessing Module
==================

Handles data preprocessing and feature selection for behavioral analysis.
Includes:
- Data cleaning and normalization
- Feature selection
- Correlation filtering
- Standardization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit

class BehaviorPreprocessor:
    """
    Preprocesses behavioral data for analysis.
    Handles data cleaning, normalization, and feature selection.
    """
    
    def __init__(self, correlation_threshold=0.95, variance_threshold=0.1):
        """
        Initialize preprocessor.
        
        Args:
            correlation_threshold: Threshold for removing highly correlated features
            variance_threshold: Threshold for removing low variance features
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.qt = QuantileTransformer(output_distribution='normal')
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        self.feature_mask = None
        self.dropped_features = None
        self.feature_names = None
        self.mouse_id_mapping = None
        
    def _handle_datetime(self, df):
        """Handle datetime columns and ensure proper interval structure."""
        if 'interval_start' in df.columns:
            df['interval_start'] = pd.to_datetime(df['interval_start'])
            df['interval_id'] = df['interval_start'].dt.strftime('%Y%m%d_%H')
            df['timestamp'] = df['interval_start'].astype('int64') // 10**9
        return df
    
    def _normalize_by_interval(self, df, feature_cols):
        """
        Normalize features within each interval.
        
        Args:
            df: DataFrame with interval_id column
            feature_cols: List of behavioral feature columns
            
        Returns:
            DataFrame with normalized features
        """
        normalized_df = df.copy()
        
        # Group by interval and normalize each feature
        for interval_id in df['interval_id'].unique():
            interval_mask = df['interval_id'] == interval_id
            interval_data = df.loc[interval_mask, feature_cols]
            
            # Quantile normalization per interval
            normalized = self.qt.fit_transform(interval_data)
            normalized_df.loc[interval_mask, feature_cols] = normalized
            
        return normalized_df
    
    def _clean_data(self, df):
        """Clean and prepare data."""
        # Get behavioral feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['mouse_id', 'date', 'interval_start', 'interval_id', 'timestamp']]
        
        # Convert to numeric and handle any non-numeric columns
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        
        # Extract features and IDs
        X = df[feature_cols].copy()
        
        # Convert mouse IDs to numeric values
        try:
            # First try direct conversion to numeric
            mouse_ids = pd.to_numeric(df['mouse_id'], errors='coerce')
            if mouse_ids.isna().any():
                # If any NaN values, try extracting numbers from strings
                mouse_ids = pd.to_numeric(df['mouse_id'].astype(str).str.extract('(\d+)', expand=False))
            
            # Convert to integers explicitly
            mouse_ids = mouse_ids.astype(np.int64)
            
        except Exception as e:
            print(f"Warning: Error converting mouse IDs: {e}")
            # Fallback: create sequential IDs
            unique_mice = df['mouse_id'].unique()
            mouse_id_to_num = {mid: i for i, mid in enumerate(sorted(unique_mice))}
            mouse_ids = pd.Series([mouse_id_to_num[mid] for mid in df['mouse_id']], dtype=np.int64)
        
        # Ensure we have valid numeric IDs
        if mouse_ids.isna().any():
            raise ValueError("Failed to convert mouse IDs to numeric values")
            
        # Create sequential mouse numbers (0 to n-1)
        unique_mice = np.sort(mouse_ids.unique())  # Sort to maintain consistent ordering
        mouse_id_to_num = {int(mid): i for i, mid in enumerate(unique_mice)}  # Ensure integer keys
        mouse_numbers = np.array([mouse_id_to_num[int(mid)] for mid in mouse_ids], dtype=np.int64)
        
        # Store the mapping for later reference
        self.mouse_id_mapping = {int(k): int(v) for k, v in mouse_id_to_num.items()}  # Ensure integer values
        
        # Fill NaN values with column means
        X = X.fillna(X.mean())
        
        return X.values, mouse_numbers, None
    
    def _remove_correlated_features(self, X):
        """Remove highly correlated features."""
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        self.dropped_features = to_drop
        X_filtered = np.delete(X, [list(upper.columns).index(c) for c in to_drop], axis=1)
        return X_filtered
    
    def _create_cross_validation_splits(self, interval_ids, n_splits=5):
        """
        Create time-series aware cross-validation splits.
        
        Args:
            interval_ids: Series of interval identifiers
            n_splits: Number of cross-validation splits
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        unique_intervals = np.unique(interval_ids)
        
        splits = []
        for train_idx, test_idx in tscv.split(unique_intervals):
            train_intervals = unique_intervals[train_idx]
            test_intervals = unique_intervals[test_idx]
            
            train_mask = interval_ids.isin(train_intervals)
            test_mask = interval_ids.isin(test_intervals)
            
            splits.append((
                np.where(train_mask)[0],
                np.where(test_mask)[0]
            ))
        
        return splits
    
    def fit_transform(self, df):
        """
        Preprocess data for analysis.
        
        Args:
            df: DataFrame containing behavioral data
            
        Returns:
            tuple: (preprocessed_features, mouse_numbers, cv_splits)
        """
        # Clean and normalize data
        X, mouse_numbers, _ = self._clean_data(df)
        
        # Store original feature names
        self.feature_names = [col for col in df.columns 
                             if col not in ['mouse_id', 'date', 'interval_start', 'interval_id', 'timestamp']]
        
        # Print mouse ID information
        unique_numbers = np.unique(mouse_numbers)
        print("\nMouse ID Summary:")
        print(f"Total samples: {len(mouse_numbers)}")
        print(f"Unique mice: {len(unique_numbers)}")
        print("\nSamples per mouse:")
        for mouse_num in unique_numbers:
            count = np.sum(mouse_numbers == mouse_num)
            original_id = list(self.mouse_id_mapping.keys())[mouse_num]  # Get original ID
            print(f"Mouse {original_id} (Mouse #{mouse_num + 1}): {count} samples")
        print()
        
        # Standardize the features
        X = self.scaler.fit_transform(X)
        
        # Remove low variance features
        X_var = self.variance_selector.fit_transform(X)
        self.feature_mask = self.variance_selector.get_support()
        
        # Remove highly correlated features
        X_final = self._remove_correlated_features(X_var)
        
        print(f"Preprocessing summary:")
        print(f"- Original features: {X.shape[1]}")
        print(f"- After variance threshold: {X_var.shape[1]}")
        print(f"- After correlation filtering: {X_final.shape[1]}")
        print(f"- Number of samples: {X_final.shape[0]}")
        print(f"- Number of unique mice: {len(unique_numbers)}")
        
        return X_final, mouse_numbers, None
    
    def get_feature_names(self):
        """Get names of selected features."""
        if self.feature_names is None:
            return None
        
        selected_features = np.array(self.feature_names)[self.feature_mask]
        final_features = [f for f in selected_features if f not in self.dropped_features]
        
        return final_features

def preprocess_data(df, correlation_threshold=0.95, variance_threshold=0.1):
    """
    Convenience function for preprocessing behavioral data.
    
    Args:
        df: DataFrame containing behavioral data
        correlation_threshold: Threshold for removing highly correlated features
        variance_threshold: Threshold for removing low variance features
        
    Returns:
        tuple: (preprocessed_features, mouse_numbers, cv_splits, feature_names)
    """
    preprocessor = BehaviorPreprocessor(
        correlation_threshold=correlation_threshold,
        variance_threshold=variance_threshold
    )
    
    X, mouse_numbers, cv_splits = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names()
    
    return X, mouse_numbers, cv_splits, feature_names 