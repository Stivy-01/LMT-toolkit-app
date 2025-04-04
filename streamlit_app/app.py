import streamlit as st
import os
import sys
import sqlite3
import pandas as pd

# Add the project directory to the path to import utils
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
streamlit_app_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(streamlit_app_path)

# Import from config and utils
from config import DATA_DIR, validate_data_directory, DEFAULT_DATABASE
from utils.db_direct_access import get_available_databases, connect_to_database, get_tables
from modules.db_utils import check_lmt_database_structure, get_database_statistics

# Function to automatically connect to a database on startup
def auto_connect_database():
    """
    Automatically connect to a database found in the data directory.
    If DEFAULT_DATABASE is set in config.py, it will connect to that specific database.
    Otherwise, it will connect to the first valid database found.
    This allows the app to start with a database already loaded.
    """
    if st.session_state.db_path is not None:
        # Already connected to a database
        return
    
    # Validate data directory
    is_valid, message = validate_data_directory()
    if not is_valid:
        st.warning(f"Data directory issue: {message}")
        return
    
    # Get available databases
    available_dbs = get_available_databases()
    if not available_dbs:
        st.info(f"No database files found in {DATA_DIR} or its subfolders")
        return
    
    # Determine which database to connect to
    if DEFAULT_DATABASE is not None and DEFAULT_DATABASE in available_dbs:
        # Use the default database specified in config
        db_name = DEFAULT_DATABASE
        st.info(f"Using default database from config: {db_name}")
    else:
        # Use the first available database
        db_name = available_dbs[0]
        st.info(f"Auto-connecting to first available database: {db_name}")
    
    db_path = os.path.join(DATA_DIR, db_name)
    
    try:
        # Create a connection
        conn = connect_to_database(db_name)
        
        # Check if it's a valid LMT database
        is_valid_lmt, tables, structure_message = check_lmt_database_structure(conn)
        
        # Set session state variables
        st.session_state.db_path = db_path
        st.session_state.valid_db = is_valid_lmt
        st.session_state.tables = tables
        
        # Get database statistics
        st.session_state.db_stats = get_database_statistics(conn, tables)
        
        # Close connection
        conn.close()
        
        st.success(f"🔄 Connected to database: {db_name}")
        
    except Exception as e:
        st.error(f"Failed to auto-connect to database: {str(e)}")

# Set page configuration
st.set_page_config(
    page_title="LMT Dimensionality Reduction Toolkit",
    page_icon="🐭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables if they don't exist
if 'db_path' not in st.session_state:
    st.session_state.db_path = None
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'valid_db' not in st.session_state:
    st.session_state.valid_db = False
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'features' not in st.session_state:
    st.session_state.features = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'db_stats' not in st.session_state:
    st.session_state.db_stats = {}

# Attempt to auto-connect to a database on startup
auto_connect_database()

# Main page header
st.title("LMT Dimensionality Reduction Toolkit")
st.markdown("""
This application provides a user-friendly interface for analyzing and visualizing 
mouse behavior data collected using the Live Mouse Tracker system.
""")

# Display database connection status
if st.session_state.db_path:
    st.success(f"Connected to database: {st.session_state.db_path}")
    st.write(f"Status: {'Valid LMT database ✅' if st.session_state.valid_db else 'Invalid LMT database ❌'}")
    if st.session_state.valid_db:
        st.write(f"Available tables: {', '.join(st.session_state.tables)}")
        
        # Display database statistics if available
        if st.session_state.db_stats:
            with st.expander("Database Statistics", expanded=False):
                for table, count in st.session_state.db_stats.items():
                    st.write(f"Table `{table}`: {count:,} rows")
else:
    st.info("No database connected. Please go to the Database Management page to connect to a database.")

# Main page content
st.markdown("""
## 🔍 Overview

The LMT Dimensionality Reduction Toolkit is designed to analyze behavioral data from the 
Live Mouse Tracker system. This Streamlit app provides an interactive user interface 
to make the analysis toolkit accessible to researchers without requiring extensive 
programming knowledge.

## 📚 Available Tools

Use the sidebar to navigate between different tools:

1. **Database Management**: Connect to LMT databases, explore tables, and run SQL queries
2. **Feature Extraction**: Extract behavioral features from the database
3. **Dimensionality Reduction**: Apply PCA and LDA to reduce dimensions 
4. **Visualization**: Visualize the results with interactive plots

## 🚀 Getting Started

To get started:
1. Navigate to the **Database Management** page
2. Connect to your LMT database using the file path option
3. Once connected, you can extract features and analyze your data

## 📚 References

- [Live Mouse Tracker Project](https://github.com/fdechaumont/lmt-analysis)
- [Forkosh et al., 2019](https://www.nature.com/articles/s41593-019-0516-y) - Identity domains capture individual differences from across the behavioral repertoire
""")

# Display footer
st.markdown("---")
st.markdown("© 2025 LMT Dimensionality Reduction Toolkit") 