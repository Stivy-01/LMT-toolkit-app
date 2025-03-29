# -*- coding: utf-8 -*-
"""
LMT Dimensionality Reduction Analysis Toolkit
===========================================

This toolkit provides advanced analysis tools for Laboratory Mouse Tracker (LMT) data,
focusing on dimensionality reduction techniques for behavioral analysis. It includes
modules for data preprocessing, behavior analysis, visualization, and database management.

The toolkit supports multiple temporal resolutions:
- 12-hour intervals (night cycle)
- 4-hour chunks (circadian phases)
- Hourly resolution

Main components:
- Preprocessing: Event filtering and data cleaning
- Behavior Analysis: Feature extraction and processing
- Dimensionality Reduction: LDA/PCA analysis
- Visualization: Interactive plots and data exploration
"""

import os
import sys
import logging
from pathlib import Path

# Package metadata
__version__ = "0.1.0"
__author__ = 'Andrea Stivala'
__email__ = 'andreastivala.as@gmail.com'
__package_name__ = "LMT Dimensionality Reduction Analysis Toolkit"

def initialize_logging():
    """Initialize logging configuration"""
    try:
        from .config import config
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_level = logging.DEBUG if config.settings['DEBUG'] else logging.INFO
        log_file = os.path.join(config.settings['logs'], 'lmt.log')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create logger for the package
        logger = logging.getLogger('lmt')
        logger.setLevel(log_level)
        
        return logger
    except Exception as e:
        print(f"Error initializing logging: {e}")
        raise

# Initialize package
logger = initialize_logging()
logger.info(f"{__package_name__} v{__version__} initialized")
logger.debug(f"Running in {os.getenv('LMT_ENV', 'development')} mode")

# Define public interface
__all__ = [
    'config',
    '__version__',
    '__author__',
    '__email__',
    '__package_name__'
]

# Import config module
from . import config

# Lazy load other modules when needed
def __getattr__(name):
    """Lazy load modules when they are first accessed"""
    if name in ['analysis', 'behavior', 'database', 'preprocessing', 'utils', 'visualization']:
        import importlib
        module = importlib.import_module(f'.{name}', __package__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__package__}' has no attribute '{name}'") 