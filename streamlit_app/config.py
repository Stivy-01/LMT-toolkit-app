import os

# Path to your data directory 
DATA_DIR = r"F:\LMT 2024\SQLITE FILES\lmt analysis"

# Default database to load on startup (relative path from DATA_DIR)
# Set to None to disable auto-loading, or set to a specific file name:
# e.g., "my_database.sqlite" or "subfolder/my_database.sqlite"
DEFAULT_DATABASE = None 

def validate_data_directory():
    """
    Validate that the data directory exists and is accessible.
    
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message is a string
    """
    if not os.path.exists(DATA_DIR):
        return False, f"Data directory does not exist: {DATA_DIR}"
    
    if not os.access(DATA_DIR, os.R_OK):
        return False, f"Data directory is not readable: {DATA_DIR}"
    
    return True, f"Data directory is valid: {DATA_DIR}" 