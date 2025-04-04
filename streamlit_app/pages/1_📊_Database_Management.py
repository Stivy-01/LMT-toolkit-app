import streamlit as st
import os
import sys
import pandas as pd
import sqlite3
import datetime
import tkinter as tk
from tkinter import filedialog
import io
import time

# Add the streamlit_app directory to the path to import utils
streamlit_app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(streamlit_app_path)

# Add the project root to path to access src
project_path = os.path.dirname(os.path.dirname(streamlit_app_path))
sys.path.append(project_path)

from utils.db_utils import (
    validate_db_path, 
    normalize_path, 
    get_db_connection, 
    check_lmt_database_structure,
    get_table_info,
    execute_query
)
# Import the new utility functions
from utils.sqlite_table_enhancer import remove_columns, delete_rows
from utils.id_update import update_mouse_id, get_animal_columns
from utils.db_direct_access import get_available_databases, get_tables, get_table_data, get_table_schema
from config import DATA_DIR, validate_data_directory

# Utility functions for CSV processing
def remove_columns_csv(df, columns_to_keep):
    """Remove columns from a DataFrame, keeping only specified columns"""
    try:
        # Keep only the selected columns
        df_new = df[columns_to_keep].copy()
        return True, df_new, f"Removed {len(df.columns) - len(columns_to_keep)} columns from CSV"
    except Exception as e:
        return False, df, f"Error removing columns: {str(e)}"

def delete_rows_csv(df, column_name, value):
    """Delete rows from a DataFrame based on column value"""
    try:
        # Count rows before deletion
        original_count = len(df)
        
        # Delete rows where column matches value
        df_new = df[df[column_name] != value].copy()
        
        # Count deleted rows
        deleted_count = original_count - len(df_new)
        
        return True, df_new, f"Deleted {deleted_count} rows from CSV"
    except Exception as e:
        return False, df, f"Error deleting rows: {str(e)}"

# Set page title
st.set_page_config(
    page_title="Database Management - LMT Toolkit",
    page_icon="📊",
    layout="wide"
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
if 'db_stats' not in st.session_state:
    st.session_state.db_stats = {}

# Predefined SQL queries from event_filtered.py
EXCLUDED_BEHAVIORS = [
    'Detection', 'Head detected', 'Look down', 'MACHINE LEARNING ASSOCIATION',
    'RFID ASSIGN ANONYMOUS TRACK', 'RFID MATCH', 'RFID MISMATCH', 'Water Stop', 'Water Zone'
]

# SQL to add metadata columns to ANIMAL table
def get_add_animal_metadata_sql():
    return [
        """
        ALTER TABLE ANIMAL
        ADD COLUMN SEX TEXT;
        """,
        """
        ALTER TABLE ANIMAL
        ADD COLUMN AGE INTEGER;
        """,
        """
        ALTER TABLE ANIMAL
        ADD COLUMN GENOTYPE TEXT;
        """,
        """
        ALTER TABLE ANIMAL
        ADD COLUMN SETUP TEXT;
        """
    ]

# SQL to create EVENT_FILTERED table
CREATE_EVENT_FILTERED_TABLE_SQL = """
DROP TABLE IF EXISTS EVENT_FILTERED;
CREATE TABLE EVENT_FILTERED (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idanimalA INTEGER,
    idanimalB INTEGER,
    idanimalC INTEGER,
    idanimalD INTEGER,
    name TEXT,
    startframe INTEGER,
    endframe INTEGER,
    duration INTEGER,
    duration_seconds REAL,
    event_start_datetime DATETIME
);
"""

# SQL to insert merged events
def get_insert_merged_events_sql():
    excluded = ", ".join(f"'{b}'" for b in EXCLUDED_BEHAVIORS)
    return f"""
    INSERT INTO EVENT_FILTERED (idanimalA, idanimalB, idanimalC, idanimalD, name, startframe, endframe)
    WITH ordered_events AS (
        SELECT
            idanimalA,
            idanimalB,
            idanimalC,
            idanimalD,
            name,
            startframe,
            endframe,
            LAG(endframe) OVER (
                PARTITION BY idanimalA, idanimalB, idanimalC, idanimalD, name 
                ORDER BY startframe
            ) AS prev_endframe
        FROM EVENT
        WHERE name NOT IN ({excluded})
    ),
    grouped_events AS (
        SELECT
            idanimalA,
            idanimalB,
            idanimalC,
            idanimalD,
            name,
            startframe,
            endframe,
            SUM(CASE WHEN startframe - COALESCE(prev_endframe, startframe) > 30 THEN 1 ELSE 0 END) 
                OVER (PARTITION BY idanimalA, idanimalB, idanimalC, idanimalD, name ORDER BY startframe) AS group_id
        FROM ordered_events
    )
    SELECT
        idanimalA,
        idanimalB,
        idanimalC,
        idanimalD,
        name,
        MIN(startframe) AS startframe,
        MAX(endframe) AS endframe
    FROM grouped_events
    GROUP BY idanimalA, idanimalB, idanimalC, idanimalD, name, group_id
    HAVING MAX(endframe) - MIN(startframe) >= 6;
    """

# SQL to update timestamps
def get_update_timestamps_sql(exp_start, fps=30.0):
    return f"""
    UPDATE EVENT_FILTERED
    SET
        duration = endframe - startframe,
        duration_seconds = ROUND((endframe - startframe) / {fps}, 2),
        event_start_datetime = datetime(
            '{exp_start.isoformat()}', 
            '+' || CAST(ROUND(startframe / {fps}) AS INTEGER) || ' seconds'
        );
    """

# SQL to add animal metadata to EVENT_FILTERED
def get_event_metadata_sql(letter):
    return [
        f"""
        ALTER TABLE EVENT_FILTERED
        ADD COLUMN GENOTYPE_{letter} TEXT;
        """,
        f"""
        UPDATE EVENT_FILTERED
        SET GENOTYPE_{letter} = (SELECT GENOTYPE FROM ANIMAL WHERE ANIMAL.ID = EVENT_FILTERED.idanimal{letter})
        WHERE idanimal{letter} IS NOT NULL;
        """,
        f"""
        ALTER TABLE EVENT_FILTERED
        ADD COLUMN SETUP_{letter} TEXT;
        """,
        f"""
        UPDATE EVENT_FILTERED
        SET SETUP_{letter} = (SELECT SETUP FROM ANIMAL WHERE ANIMAL.ID = EVENT_FILTERED.idanimal{letter})
        WHERE idanimal{letter} IS NOT NULL;
        """
    ]

st.title("📊 Database Management")
st.markdown("""
This page allows you to manage and explore your SQLite databases for the LMT Toolkit.
The app will access databases directly from your configured data directory.
""")

# Check if data directory is valid
is_valid, message = validate_data_directory()
if not is_valid:
    st.error(message)
    st.error(f"Please check your data directory configuration in config.py: {DATA_DIR}")
    st.stop()

st.success(f"Using data directory: {DATA_DIR}")

# Get available databases
available_dbs = get_available_databases()
if not available_dbs:
    st.warning(f"No database files (.db, .sqlite, .sqlite3) found in {DATA_DIR}")
    st.info("Please add your database files to this directory")
else:
    # Database selection
    selected_db = st.selectbox("Select Database", available_dbs, format_func=lambda x: x)
    
    # Get tables in selected database
    try:
        tables = get_tables(selected_db)
        
        if not tables:
            st.warning(f"No tables found in database {selected_db}")
        else:
            # Create tabs for different database views
            tab1, tab2 = st.tabs(["Table Explorer", "Schema Viewer"])
            
            with tab1:
                # Table selection
                selected_table = st.selectbox("Select Table", tables)
                
                # Row limit to prevent loading too much data
                limit = st.slider("Maximum rows to display", 10, 5000, 1000)
                
                if st.button("View Table Data"):
                    with st.spinner(f"Loading data from {selected_table}..."):
                        try:
                            start_time = time.time()
                            df = get_table_data(selected_db, selected_table, limit)
                            load_time = time.time() - start_time
                            
                            st.success(f"Data loaded successfully in {load_time:.2f} seconds")
                            st.write(f"Showing {len(df)} rows (limited to {limit})")
                            st.dataframe(df)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download data as CSV",
                                data=csv,
                                file_name=f"{selected_table}.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error loading table data: {e}")
            
            with tab2:
                # Schema viewer
                selected_table_schema = st.selectbox("Select Table for Schema", tables, key="schema_table")
                
                if st.button("View Schema"):
                    try:
                        schema = get_table_schema(selected_db, selected_table_schema)
                        
                        # Create a DataFrame for display
                        schema_df = pd.DataFrame(schema)
                        st.write(f"Schema for table: {selected_table_schema}")
                        st.dataframe(schema_df)
                    except Exception as e:
                        st.error(f"Error loading schema: {e}")
        
    except Exception as e:
        st.error(f"Error accessing database {selected_db}: {e}")

# Alternative upload option (keeping this for compatibility)
st.markdown("---")
st.header("Alternative: Upload a Database File")
st.info("You can also upload a SQLite database file directly if it's not in your configured directory.")

uploaded_file = st.file_uploader("Upload SQLite Database", type=["db", "sqlite", "sqlite3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_uploaded.db")
    
    with open(temp_db_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Database uploaded successfully: {uploaded_file.name}")
    
    # Connect to the uploaded database
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    if not tables:
        st.warning("No tables found in the uploaded database")
    else:
        st.write("Tables in the database:")
        selected_table = st.selectbox("Select Table", tables, key="uploaded_table")
        
        if st.button("View Uploaded Data"):
            # Get the data
            data = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 1000", conn)
            st.dataframe(data)
    
    conn.close()

# Display footer
st.markdown("---")

# -----------------------------
# Database Merging Section
# -----------------------------
st.header("🔄 Merge Databases for Analysis")
st.markdown("""
This tool allows you to merge behavior statistics from multiple databases into a single analysis database.
Select the type of analysis and the source databases to create a merged database for further analysis.
""")

# Function to get table mapping
def get_table_mapping():
    """Get mapping of analysis types to their required tables."""
    return {
        1: ['behavior_hourly', 'group_events_hourly'],
        2: ['behavior_stats_intervals', 'multi_mouse_events_intervals'],
        3: ['BEHAVIOR_STATS', 'MULTI_MOUSE_EVENTS']
    }

# Function to find table in database regardless of case
def find_table_in_db(conn, table_name):
    """Find a table in database regardless of case."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? COLLATE NOCASE", (table_name,))
    result = cursor.fetchone()
    return result[0] if result else None

# Function to merge databases
def merge_databases(db_paths, table_names, output_db, progress_callback=None):
    """Merge specified tables from multiple databases into one."""
    output_text = []
    output_text.append("Starting database merge...")
    
    # Create new database
    with sqlite3.connect(output_db) as target_conn:
        # Process each table
        for table_idx, table_name in enumerate(table_names):
            output_text.append(f"\nProcessing table: {table_name}")
            all_data = []
            
            # Collect data from each source database
            for db_idx, db_path in enumerate(db_paths):
                if progress_callback:
                    # Update progress: (tables done + fraction of this table) / total tables
                    progress = (table_idx + (db_idx / len(db_paths))) / len(table_names)
                    progress_callback(progress)
                
                try:
                    with sqlite3.connect(db_path) as source_conn:
                        # Find actual table name in this database
                        actual_table = find_table_in_db(source_conn, table_name)
                        if not actual_table:
                            output_text.append(f"Table {table_name} not found in {os.path.basename(db_path)}")
                            continue
                            
                        # Read the data
                        df = pd.read_sql_query(f'SELECT * FROM "{actual_table}"', source_conn)
                        output_text.append(f"Read {len(df)} rows from {os.path.basename(db_path)}")
                        
                        all_data.append(df)
                        
                except Exception as e:
                    output_text.append(f"Error reading from {os.path.basename(db_path)}: {str(e)}")
                    continue
            
            if not all_data:
                output_text.append(f"No data found for table {table_name}")
                continue
                
            # Combine all dataframes
            combined_df = pd.concat(all_data, ignore_index=True)
            output_text.append(f"\nBefore deduplication: {len(combined_df)} total rows")
            
            # For interval tables, we need to handle the merge carefully
            if 'interval_start' in combined_df.columns and 'mouse_id' in combined_df.columns:
                # Remove exact duplicates
                combined_df = combined_df.drop_duplicates()
                output_text.append(f"After removing exact duplicates: {len(combined_df)} rows")
                
                output_text.append("\nFinal combined data:")
                output_text.append(f"Total rows: {len(combined_df)}")
                output_text.append(f"Unique intervals: {combined_df['interval_start'].nunique()}")
                output_text.append(f"Unique mice: {combined_df['mouse_id'].nunique()}")
            
            # Save to new database
            combined_df.to_sql(table_name, target_conn, if_exists='replace', index=False)
            output_text.append(f"\nSaved table {table_name} to merged database")
            
            # Export to CSV
            csv_dir = os.path.join(os.path.dirname(output_db), "csv_export")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"merged_analysis_{table_name}.csv")
            combined_df.to_csv(csv_path, index=False)
            output_text.append(f"Exported to CSV: {csv_path}")
            
            # Verify the saved data
            verification_df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', target_conn)
            output_text.append(f"\nVerification - rows in saved table: {len(verification_df)}")
    
    if progress_callback:
        progress_callback(1.0)  # Mark as complete
        
    return "\n".join(output_text)

# Create a container for the merge tool
with st.expander("Merge Database Tool", expanded=True):
    # Analysis type selection
    st.subheader("Step 1: Select Analysis Type")
    analysis_descriptions = {
        1: "Hourly Analysis (behavior_hourly, group_events_hourly)",
        2: "Interval Analysis - 12-hour (behavior_stats_intervals, multi_mouse_events_intervals)",
        3: "Daily Analysis (BEHAVIOR_STATS, MULTI_MOUSE_EVENTS)"
    }
    
    analysis_type = st.radio(
        "Choose analysis type:",
        list(analysis_descriptions.keys()),
        format_func=lambda x: analysis_descriptions[x]
    )
    
    # Database selection
    st.subheader("Step 2: Select Source Databases")
    
    # Option to upload databases
    upload_method = st.radio(
        "How would you like to select databases?",
        ["Select files", "Use connected database"]
    )
    
    db_paths = []
    
    if upload_method == "Select files":
        # Display selected files
        if 'selected_db_files' not in st.session_state:
            st.session_state.selected_db_files = []
            
        # Show currently selected files
        if st.session_state.selected_db_files:
            st.write("Selected databases:")
            for i, file_path in enumerate(st.session_state.selected_db_files):
                st.code(f"{i+1}. {os.path.basename(file_path)}")
            db_paths = st.session_state.selected_db_files
        
        # Direct file path input - use full width instead of columns
        st.subheader("Add Database Files")
        
        # Use a container with custom styling for the input section
        with st.container():
            # Initialize the file path in session state if it doesn't exist
            if 'file_path_input' not in st.session_state:
                st.session_state.file_path_input = ""
            
            # Use custom CSS to make buttons larger
            st.markdown("""
            <style>
            div.stButton > button {
                font-size: 16px;
                padding: 12px 24px;
                height: auto;
            }
            .button-spacing {
                margin-top: 24px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create two columns with better proportions for input and button
            input_col, button_col = st.columns([3, 1])
            
            with input_col:
                file_path = st.text_input(
                    "Enter full database file path (.sqlite or .db file):",
                    value=st.session_state.file_path_input,
                    key="file_path_entry",
                    placeholder="C:/path/to/your/database.sqlite"
                )
            
            # Add file path button in the second column for better alignment
            with button_col:
                # Add vertical spacing to move the button down
                st.markdown('<div class="button-spacing"></div>', unsafe_allow_html=True)
                add_button = st.button("Add File", key="add_file_path", use_container_width=True)
        
        # Handle the add file button click outside the columns for cleaner code
        if add_button:
            # Strip quotes from the beginning and end of the file path if present
            cleaned_file_path = file_path.strip('"\'')
            
            if cleaned_file_path and os.path.exists(cleaned_file_path) and cleaned_file_path.lower().endswith(('.sqlite', '.db')):
                if 'selected_db_files' not in st.session_state:
                    st.session_state.selected_db_files = []
                
                # Only add if not already in the list
                if cleaned_file_path not in st.session_state.selected_db_files:
                    st.session_state.selected_db_files.append(cleaned_file_path)
                    st.session_state.file_path_input = ""  # Clear the input
                    st.rerun()
                else:
                    st.warning("This file is already in the list.")
            else:
                st.error("Invalid file path. Please enter a valid path to a .sqlite or .db file.")
        
        # Clear selection button - centered and with better styling
        if st.session_state.selected_db_files:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Clear All Files", key="clear_selection", type="secondary", use_container_width=True):
                    st.session_state.selected_db_files = []
                    st.rerun()
            
        # Status message
        if db_paths:
            st.success(f"Selected {len(db_paths)} database files")
        else:
            st.info("Click 'Browse Files' to select database files for merging")
    else:
        # Use the currently connected database
        if st.session_state.db_path:
            db_paths = [st.session_state.db_path]
            st.success(f"Using connected database: {os.path.basename(st.session_state.db_path)}")
        else:
            st.warning("No database currently connected. Please connect a database or select files.")
    
    # Output settings
    st.subheader("Step 3: Output Settings")
    
    # Default output name
    default_output_name = "merged_analysis.sqlite"
    if db_paths:
        output_dir = os.path.dirname(db_paths[0])
    else:
        output_dir = os.path.dirname(streamlit_app_path)
    
    output_name = st.text_input("Output database name:", default_output_name)
    output_path = os.path.join(output_dir, output_name)
    
    # Merge button
    if st.button("Merge Databases", type="primary", disabled=not db_paths):
        if len(db_paths) > 0:
            # Get required tables for selected analysis
            tables = get_table_mapping()[analysis_type]
            
            # Store tables in session state for later use with download buttons
            st.session_state.merged_tables = tables
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Define progress callback
            def update_progress(progress):
                progress_bar.progress(progress)
            
            # Execute merge
            try:
                with st.spinner(f"Merging {len(db_paths)} databases..."):
                    merge_result = merge_databases(db_paths, tables, output_path, update_progress)
                    
                st.success("Database merge completed successfully!")
                
                # Store merge results in session state for display outside the expander
                st.session_state.merge_result = merge_result
                st.session_state.merged_output_path = output_path
                st.session_state.merged_output_name = output_name
                
            except Exception as e:
                st.error(f"Error during database merge: {str(e)}")
        else:
            st.error("No database files selected for merging")

# Display merge results (outside the merge tool expander)
if 'merge_result' in st.session_state and st.session_state.merge_result:
    # Add a divider for visual separation
    st.markdown("---")
    
    # Show merge details
    st.subheader("Merge Results")
    st.text(st.session_state.merge_result)
    
    # Provide download links
    st.subheader("Download Results")
    
    # For the sqlite database
    with open(st.session_state.merged_output_path, "rb") as f:
        st.download_button(
            label="Download Merged Database",
            data=f,
            file_name=st.session_state.merged_output_name,
            mime="application/octet-stream"
        )
    
    # For CSVs
    output_dir = os.path.dirname(st.session_state.merged_output_path)
    csv_dir = os.path.join(output_dir, "csv_export")
    
    # Get the tables that were merged (stored in session state)
    if 'merged_tables' not in st.session_state:
        # If not in session state, get from the analysis type
        tables = get_table_mapping().get(analysis_type, [])
        st.session_state.merged_tables = tables
    
    for table in st.session_state.merged_tables:
        csv_path = os.path.join(csv_dir, f"merged_analysis_{table}.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    label=f"Download {table} CSV",
                    data=f,
                    file_name=f"merged_analysis_{table}.csv",
                    mime="text/csv",
                    key=f"download_{table}"
                )

st.markdown("---")

# -----------------------------
# Table to CSV Converter
# -----------------------------
st.header("📄 Export Database Tables to CSV")
st.markdown("""
Convert database tables to CSV files for easy analysis in spreadsheet software or other tools.
Select a database file, choose which table to export, and download the CSV file.
""")

with st.expander("Table to CSV Converter", expanded=True):
    # Database selection
    st.subheader("Step 1: Select Database")
    
    # Option to use current or select a different database
    csv_db_selection = st.radio(
        "Database source:",
        ["Use connected database", "Select different database"],
        key="csv_db_source"
    )
    
    csv_db_path = None
    
    if csv_db_selection == "Use connected database":
        if st.session_state.db_path:
            csv_db_path = st.session_state.db_path
            st.success(f"Using connected database: {os.path.basename(csv_db_path)}")
        else:
            st.warning("No database currently connected. Please connect a database first or select a different database.")
    else:
        # Create file selector
        if st.button("Browse Database File", key="browse_csv_db"):
            # Create tkinter window and hide it
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Database File",
                filetypes=(("SQLite files", "*.sqlite;*.db"), ("All files", "*.*"))
            )
            
            if file_path:  # If file was selected
                st.session_state.csv_db_path = file_path
                # Force rerun to refresh the UI
                st.rerun()
        
        # Use the saved path if available
        if hasattr(st.session_state, 'csv_db_path'):
            csv_db_path = st.session_state.csv_db_path
            st.success(f"Selected database: {os.path.basename(csv_db_path)}")
    
    # Table selection and export
    if csv_db_path:
        st.subheader("Step 2: Select Table to Export")
        
        # Create a connection to the database
        try:
            conn = sqlite3.connect(csv_db_path)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                # Display table selection
                selected_table = st.selectbox("Select table to export:", tables, key="csv_export_table")
                
                if selected_table:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM '{selected_table}'")
                    row_count = cursor.fetchone()[0]
                    
                    # Get column info
                    cursor.execute(f"PRAGMA table_info('{selected_table}')")
                    columns = [info[1] for info in cursor.fetchall()]
                    
                    # Display table info
                    st.info(f"Table: {selected_table} ({row_count} rows, {len(columns)} columns)")
                    
                    # Sample data preview
                    cursor.execute(f"SELECT * FROM '{selected_table}' LIMIT 5")
                    sample_data = cursor.fetchall()
                    
                    if sample_data:
                        sample_df = pd.DataFrame(sample_data, columns=columns)
                        st.write("Sample data:")
                        st.dataframe(sample_df)
                        
                        # Export button
                        if st.button("Export to CSV", key="export_to_csv"):
                            # Show progress message
                            with st.spinner(f"Exporting {selected_table} to CSV..."):
                                # Get data
                                df = pd.read_sql_query(f"SELECT * FROM '{selected_table}'", conn)
                                
                                # Prepare CSV
                                csv = df.to_csv(index=False)
                                
                                # Generate filename
                                filename = f"{selected_table}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                
                                # Show download button
                                st.success(f"Exported {len(df)} rows to CSV")
                                st.download_button(
                                    label="Download CSV File",
                                    data=csv,
                                    file_name=filename,
                                    mime="text/csv"
                                )
                    else:
                        st.warning("Table is empty.")
            else:
                st.warning("No tables found in the database.")
            
            # Close connection
            conn.close()
            
        except Exception as e:
            st.error(f"Error accessing database: {str(e)}")

st.markdown("© 2025 LMT Dimensionality Reduction Toolkit") 