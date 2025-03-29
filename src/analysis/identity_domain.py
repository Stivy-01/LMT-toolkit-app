"""
Identity Domain Analyzer Module
=============================

Implements the core Identity Domain Analysis functionality using Fisher-Rao discriminant.
This is the fundamental component for behavioral identity analysis following Forkosh's approach.
"""

import numpy as np
from scipy.linalg import eigh

class IdentityDomainAnalyzer:
    """
    Identity Domain Analysis using Fisher-Rao discriminant.
    Maximizes the ratio of between-class to within-class variance.
    
    Attributes:
        components_: Learned identity domain components (after fitting)
        eigenvalues_: Eigenvalues corresponding to the components
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.components_ = None
        self.eigenvalues_ = None

    def _compute_scatter_matrices(self, X, y):
        """
        Compute between-class and within-class scatter matrices.
        
        Args:
            X: Feature matrix
            y: Class labels (mouse IDs)
            
        Returns:
            tuple: (within-class scatter matrix, between-class scatter matrix)
        """
        unique_classes = np.unique(y)
        n_features = X.shape[1]
        
        # Initialize scatter matrices
        Sw = np.zeros((n_features, n_features))  # Within-class scatter
        Sb = np.zeros((n_features, n_features))  # Between-class scatter
        
        # Calculate overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Compute scatter matrices
        for cls in unique_classes:
            X_cls = X[y == cls]
            
            # Within-class scatter
            if len(X_cls) > 1:
                Sw += np.cov(X_cls.T) * (len(X_cls)-1)
            
            # Between-class scatter
            if len(X_cls) > 0:
                mean_diff = (np.mean(X_cls, axis=0) - overall_mean).reshape(-1, 1)
                Sb += len(X_cls) * (mean_diff @ mean_diff.T)
                
        return Sw, Sb

    def fit(self, X, y, max_components=None):
        """
        Fit the Identity Domain Analyzer to the data.
        
        Args:
            X: Feature matrix
            y: Class labels (mouse IDs)
            max_components: Maximum number of components to compute (default: min(n_features, n_classes-1))
            
        Returns:
            self: The fitted analyzer
        """
        # Compute scatter matrices
        Sw, Sb = self._compute_scatter_matrices(X, y)
        
        # Add regularization to within-class scatter matrix
        # This helps with numerical stability
        Sw_reg = (Sw + Sw.T) / 2 + 1e-3 * np.eye(Sw.shape[0])
        
        # Ensure symmetry in between-class scatter matrix
        Sb = (Sb + Sb.T) / 2
        
        # Solve generalized eigenvalue problem
        # This maximizes the Fisher-Rao discriminant criterion
        eig_vals, eig_vecs = eigh(Sb, Sw_reg, check_finite=False)
        
        # Sort eigenvectors by eigenvalues in descending order
        order = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]
        
        # Store all eigenvalues
        self.eigenvalues_ = eig_vals
        
        # If max_components not specified, use theoretical maximum
        if max_components is None:
            max_components = min(X.shape[1], len(np.unique(y))-1)
        
        # Store all components up to max_components
        self.components_ = eig_vecs[:, :max_components].T
        
        return self

    def transform(self, X, n_components=None):
        """
        Transform data into identity domain space.
        
        Args:
            X: Feature matrix to transform
            n_components: Number of components to use (default: all available components)
            
        Returns:
            array: Transformed data in identity domain space
        """
        if n_components is None:
            return X @ self.components_.T
        else:
            return X @ self.components_[:n_components].T

    def fit_transform(self, X, y, max_components=None, n_components=None):
        """
        Fit the analyzer and transform the data in one step.
        
        Args:
            X: Feature matrix
            y: Class labels (mouse IDs)
            max_components: Maximum number of components to compute
            n_components: Number of components to use in transformation
            
        Returns:
            array: Transformed data in identity domain space
        """
        self.fit(X, y, max_components)
        return self.transform(X, n_components) 