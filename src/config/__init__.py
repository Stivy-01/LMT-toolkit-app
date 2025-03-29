# -*- coding: utf-8 -*-
"""
Configuration module for LMT Analysis package.
Handles environment settings and global configurations.
"""

import os
from pathlib import Path

class Config:
    def __init__(self):
        self.settings = {
            'ENV': os.getenv('LMT_ENV', 'development'),
            'PROJECT_ROOT': str(Path(__file__).parent.parent.parent),
            'DATA_DIR': str(Path(__file__).parent.parent.parent / 'data'),
            'DEBUG': os.getenv('LMT_DEBUG', 'True').lower() == 'true'
        }
        
        # Initialize paths
        self._init_paths()
        
    def _init_paths(self):
        """Initialize necessary paths and create directories if they don't exist"""
        paths = {
            'data': self.settings['DATA_DIR'],
            'cache': os.path.join(self.settings['DATA_DIR'], 'cache'),
            'output': os.path.join(self.settings['DATA_DIR'], 'output'),
            'logs': os.path.join(self.settings['DATA_DIR'], 'logs')
        }
        
        # Create directories if they don't exist
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            
        self.settings.update(paths)

# Create singleton instance
config = Config()

__all__ = ['config'] 