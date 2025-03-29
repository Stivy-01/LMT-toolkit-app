# analysis_demo.py (OPTIMIZED VERSION)
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from pathlib import Path
from tkinter import filedialog, simpledialog, messagebox
import tkinter as tk
import os
import sys

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def get_analysis_type():
    """Get analysis type from user via GUI."""
    root = tk.Tk()
    root.withdraw()
    
    table_type = simpledialog.askinteger(
        "Analysis Type",
        "Choose analysis type (enter a number):\n\n" +
        "1: Hourly Analysis\n" +
        "   - behavior_hourly\n\n" +
        "2: Interval Analysis\n" +
        "   - behavior_stats_intervals\n\n" +
        "3: Daily Analysis\n" +
        "   - BEHAVIOR_STATS",
        minvalue=1, maxvalue=3
    )
    
    if not table_type:
        raise ValueError("No analysis type selected")
    
    # Map analysis type to table name
    table_mapping = {
        1: 'behavior_hourly',
        2: 'behavior_stats_intervals',
        3: 'BEHAVIOR_STATS'
    }
    
    return table_type, table_mapping[table_type]

def select_csv_file(table_name):
    """Select CSV file from the appropriate analysis directory."""
    data_dir = project_root / 'data'
    analysis_dir = data_dir / f"{table_name}_to_analyze"
    
    if not analysis_dir.exists():
        print(f"\n❌ Error: Analysis directory not found at {analysis_dir}")
        print("Please ensure the CSV file exists in the correct directory.")
        return None
        
    # Check for the specific file first
    default_file = analysis_dir / f"merged_analysis_{table_name}.csv"
    if default_file.exists():
        return str(default_file)
    
    # If specific file not found, show file dialog
    root = tk.Tk()
    root.withdraw()
    
    csv_path = filedialog.askopenfilename(
        title="Select CSV file to analyze",
        initialdir=analysis_dir,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    return csv_path if csv_path else None

def preprocess_data(df):
    # Handle datetime
    if 'interval_start' in df.columns:
        df['interval_start'] = pd.to_datetime(df['interval_start'])
        df['timestamp'] = df['interval_start'].astype('int64') // 10**9
        df = df.drop('interval_start', axis=1)
    
    # Convert to numeric and clean
    non_target_cols = [col for col in df.columns if col not in ['mouse_id', 'date']]
    df[non_target_cols] = df[non_target_cols].apply(pd.to_numeric, errors='coerce')
    
    X = df.drop(['mouse_id', 'date'], axis=1, errors='ignore').dropna(axis=1, how='all')
    X = X.fillna(X.mean())
    
    # Initial feature selection
    selector = VarianceThreshold(threshold=0.1)
    return selector.fit_transform(X), df['mouse_id']

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
        
        # Enhanced regularization
        Sw_reg = (Sw + Sw.T) / 2 + 1e-3 * np.eye(Sw.shape[0])  # Increased regularization
        Sb = (Sb + Sb.T) / 2
        
        # Eigen decomposition
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

def save_results(results_dict, output_dir, base_filename):
    """Save analysis results to the output directory."""
    # Save numerical results
    results_df = pd.DataFrame({
        'Metric': ['Average ID stability score', 'Original dimensions', 'PCA dimensions'],
        'Value': [
            results_dict['avg_stability'],
            results_dict['original_dims'],
            results_dict['pca_dims']
        ]
    })
    results_df.to_csv(output_dir / f"{base_filename}_metrics.csv", index=False)
    
    # Save identity space data with mouse_id
    identity_space_df = pd.DataFrame(
        results_dict['identity_space'],
        columns=[f'ID_Component_{i+1}' for i in range(results_dict['identity_space'].shape[1])]
    )
    # Add mouse_id column
    identity_space_df.insert(0, 'mouse_id', results_dict['mouse_ids'])
    identity_space_df.to_csv(output_dir / f"{base_filename}_identity_space.csv", index=False)

if __name__ == "__main__":
    try:
        # Get analysis type and corresponding table name
        analysis_type, table_name = get_analysis_type()
        
        # Select CSV file to analyze
        print("\nSelect CSV file to analyze...")
        csv_path = select_csv_file(table_name)
        if not csv_path:
            sys.exit(1)
            
        # Create output directory with correct spelling
        output_dir = project_root / 'data' / f"{table_name}_analyzed"
        output_dir.mkdir(exist_ok=True)
        base_filename = Path(csv_path).stem
        
        # Read and process data
        print(f"Reading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        X, y = preprocess_data(df)
        
        # Standardization pipeline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Remove highly correlated features
        corr_matrix = pd.DataFrame(X_scaled).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X_filtered = np.delete(X_scaled, to_drop, axis=1)
        print(f"Removed {len(to_drop)} highly correlated features")
        
        # Dimensionality reduction
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_filtered)
        print(f"PCA reduced dimensions from {X_filtered.shape[1]} to {X_pca.shape[1]}")
        
        # Final analysis
        avg_stability, identity_space = analyze_identity_domain(X_pca, y)
        
        # Save results
        results = {
            'avg_stability': avg_stability,
            'original_dims': X_filtered.shape[1],
            'pca_dims': X_pca.shape[1],
            'identity_space': identity_space,
            'mouse_ids': y  # Add mouse_ids to results dictionary
        }
        save_results(results, output_dir, base_filename)
        
        print(f"\n✅ Analysis complete! Results saved in: {output_dir}")
        print(f"Average ID stability score: {avg_stability:.3f}")
        print(f"Files generated:")
        print(f"- {base_filename}_metrics.csv")
        print(f"- {base_filename}_identity_space.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
