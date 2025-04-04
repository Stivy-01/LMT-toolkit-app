import os

# Set the data directory path
DATA_DIR = r"F:\LMT 2024\SQLITE FILES\lmt analysis"

# Function to check if the directory exists and is accessible
def validate_data_directory():
    if not os.path.exists(DATA_DIR):
        return False, f"Data directory not found: {DATA_DIR}"
    if not os.access(DATA_DIR, os.R_OK):
        return False, f"Data directory not readable: {DATA_DIR}"
    return True, "Data directory is valid and accessible" 