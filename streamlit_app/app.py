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