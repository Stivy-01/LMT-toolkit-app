"""
LMT Analysis Package
===================

This package provides tools for behavioral identity analysis using PCA and LDA approaches.
It implements both sequential and parallel analysis methods following Forkosh et al.

Main Components:
- Sequential Analysis (PCA followed by LDA)
- Parallel Analysis (PCA and LDA separately)
- Statistical Analysis Tools
- Identity Domain Analysis Core
"""

from .davids_score import analyze_dominance
from .dimensionality_reduction import analyze_behavior

__version__ = '0.1.0'
__author__ = 'Andrea Stivala'
__email__ = 'andreastivala.as@gmail.com' 