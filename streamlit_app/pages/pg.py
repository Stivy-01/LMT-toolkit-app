"""
Database Management Page for the LMT Toolkit.
This unified page contains all database functionality while using modular components.
"""

import streamlit as st
import os
import sys
import pandas as pd
import datetime
import tkinter as tk
from tkinter import filedialog
import io
import sqlite3
import zipfile

# Add the streamlit_app directory to the path to import modules
streamlit_app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(streamlit_app_path)

# Add the project root to path to access src
project_path = os.path.dirname(streamlit_app_path)
sys.path.append(project_path)

# Import config for data directory access
from config import DATA_DIR, validate_data_directory
from utils.db_direct_access import get_available_databases, connect_to_database, get_tables

# Import all modular components
from modules.db_utils import (
    validate_db_path, normalize_path, get_db_connection, 
    check_lmt_database_structure, get_table_info, execute_query,
    get_database_statistics
)
from modules.sql_generators import (
    EXCLUDED_BEHAVIORS, get_add_animal_metadata_sql, get_insert_merged_events_sql,
    get_update_timestamps_sql, get_event_metadata_sql, CREATE_EVENT_FILTERED_TABLE_SQL,
    get_query_templates, get_basic_query_templates, get_advanced_query_templates
)
from modules.csv_processors import (
    remove_columns_csv, delete_rows_csv, csv_to_sql, sql_to_csv,
    export_multiple_tables_to_csv, get_csv_preview, filter_csv_by_column_value,
    sort_csv_by_column, add_column_csv
)
from modules.table_operations import (
    remove_columns, delete_rows, add_column, rename_table,
    copy_table, get_table_row_count, export_table, import_csv_to_table,
    update_column_values
)
from modules.event_processing import (
    create_event_filtered_table, insert_merged_events, update_event_timestamps,
    add_animal_metadata_to_events, get_event_statistics, process_all_events,
    get_behaviors_list, get_behavior_events, get_behavior_statistics,
    get_animal_list as get_filtered_animal_list, get_animal_events,
    get_animal_behavior_distribution, get_time_range, get_events_in_frame_range
)
from modules.database_merger import (
    get_table_mapping, merge_databases, export_tables_to_csv, find_table_in_db
)
from modules.animal_manager import (
    update_mouse_id, get_animal_columns, update_animal_metadata, get_animal_list,
    add_animal, delete_animal, get_animal_events, get_animal_metadata_columns
)

# Import UI components
from components.database_management import (
    render_connection_section,
    render_sql_query_section,
    render_event_processing_section
)

# Helper function to get a fresh database connection
def get_connection():
    """
    Create a fresh connection to the database rather than using one stored in session state.
    This avoids the thread issues in Streamlit by creating a new connection for each operation.
    
    Returns:
        sqlite3.Connection or None: Database connection or None if it couldn't be established
    """
    if st.session_state.db_path is None:
        return None
        
    try:
        # Always create a fresh connection
        conn = get_db_connection(st.session_state.db_path)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

# Set page title
st.set_page_config(
    page_title="Database Management - LMT Toolkit",
    page_icon="📊",
    layout="wide"
)

# Store the connection function in session state to be used by components
st.session_state.get_connection = get_connection

# Initialize session state variables if they don't exist
if 'db_path' not in st.session_state:
    st.session_state.db_path = None
if 'valid_db' not in st.session_state:
    st.session_state.valid_db = False
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'db_stats' not in st.session_state:
    st.session_state.db_stats = {}
if 'selected_db_files' not in st.session_state:
    st.session_state.selected_db_files = []
if 'merged_output_path' not in st.session_state:
    st.session_state.merged_output_path = None
if 'merged_tables' not in st.session_state:
    st.session_state.merged_tables = []
if 'merge_result' not in st.session_state:
    st.session_state.merge_result = None
if 'csv_df' not in st.session_state:
    st.session_state.csv_df = None
if 'csv_filename' not in st.session_state:
    st.session_state.csv_filename = None
if 'csv_modified' not in st.session_state:
    st.session_state.csv_modified = False

st.title("📊 Database Management")
st.markdown("""
This all-in-one tool allows you to connect to your LMT database, explore its structure, 
run SQL queries, process events, and more - all from a unified interface.
""")

# Display database connection status at the top if connected
if st.session_state.db_path:
    st.success(f"Connected to database: {st.session_state.db_path}")
    if st.session_state.valid_db:
        st.write("✅ Valid LMT database structure detected")
        st.write(f"Found tables: {', '.join(st.session_state.tables)}")
        
        # Display database statistics if available
        if st.session_state.db_stats:
            with st.expander("Database Statistics", expanded=False):
                for table, count in st.session_state.db_stats.items():
                    st.write(f"Table `{table}`: {count:,} rows")
    else:
        st.warning("⚠️ Database structure does not match expected LMT format")
        if 'structure_message' in st.session_state:
            st.info(st.session_state.structure_message)
        st.write(f"Found tables: {', '.join(st.session_state.tables)}")

# Create main tabs for the application
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Database Connection", 
    "SQL Queries", 
    "Event Processing",
    "Table Operations", 
    "Database Merging",
    "CSV Operations"
])

# Tab 1: Database Connection - Use modular component
with tab1:
    st.header("Connect to Database")
    st.markdown("""
    Connect to your LMT database using a file path, by uploading a database file, or 
    by selecting from available databases in your configured data directory.
    This is the first step to analyze your behavioral data using the LMT Toolkit.
    """)
    
    # Create tabs for different database connection methods
    connect_tab1, connect_tab2, connect_tab3 = st.tabs([
        "Connect via Data Directory", 
        "Connect via File Path", 
        "Connect via File Upload"
    ])
    
    # Connect Tab 1: Connect via Data Directory
    with connect_tab1:
        st.subheader("Connect via Data Directory")
        st.markdown("""
        Use this method to select databases from your configured data directory.
        This method allows you to quickly access your existing database files without uploading them.
        """)
        
        # Validate data directory
        is_valid, message = validate_data_directory()
        if is_valid:
            st.success(f"Found configured data directory: {DATA_DIR}")
            
            # Get available databases
            available_dbs = get_available_databases()
            if not available_dbs:
                st.warning(f"No database files (.db, .sqlite, .sqlite3) found in {DATA_DIR} or its subfolders")
                st.info("Please add your database files to this directory or a subfolder")
            else:
                # Database selection
                selected_db = st.selectbox(
                    "Select Database", 
                    available_dbs, 
                    format_func=lambda x: x
                )
                
                # Connect button for data directory method
                if st.button("Connect to Database", key="connect_dir_button"):
                    try:
                        # Get the full path to the database
                        db_path = os.path.join(DATA_DIR, selected_db)
                        st.session_state.db_path = db_path
                        
                        # Attempt to connect to the database
                        conn = get_db_connection(db_path)
                        
                        # Check if it's a valid LMT database
                        is_valid_lmt, tables, structure_message = check_lmt_database_structure(conn)
                        st.session_state.valid_db = is_valid_lmt
                        st.session_state.tables = tables
                        st.session_state.structure_message = structure_message
                        
                        # Show success message with database information
                        st.success(f"Successfully connected to database: {selected_db}")
                        
                        # Get database statistics and store in session state
                        st.session_state.db_stats = get_database_statistics(conn, tables)
                        
                        # Always close the connection when done
                        conn.close()
                        
                        # Display information about tables
                        if is_valid_lmt:
                            st.write("✅ Valid LMT database structure detected")
                        else:
                            st.warning("⚠️ Database structure does not match expected LMT format")
                            st.info(structure_message)
                            
                        st.write(f"Found tables: {', '.join(tables)}")
                        
                        # Display database statistics
                        st.subheader("Database Statistics")
                        for table, count in st.session_state.db_stats.items():
                            st.write(f"Table `{table}`: {count:,} rows")
                        
                    except Exception as e:
                        st.error(f"Failed to connect to the database: {str(e)}")
                        st.session_state.valid_db = False
        else:
            st.error(message)
            st.info("Please check your configuration in the config.py file.")
    
    # Connect Tab 2: Connect via File Path
    with connect_tab2:
        st.subheader("Connect via File Path")
        st.markdown("""
        Use this method for databases larger than Streamlit's 200MB upload limit. 
        Enter the full path to your SQLite database file.
        """)
        
        # Input for database path
        db_path = st.text_input(
            "Enter the path to your SQLite database file:",
            help="Example: C:/Users/username/data/lmt_database.sqlite or /home/username/data/lmt_database.sqlite"
        )
        
        # Path format converter - Helpful for Windows users
        with st.expander("Path Format Help", expanded=False):
            st.markdown("""
            ### Path Format Converter
            
            If you're having trouble with file paths, especially on Windows:
            
            - Windows paths use backslashes: `C:\\Users\\username\\data\\lmt_database.sqlite`
            - Unix-style paths use forward slashes: `C:/Users/username/data/lmt_database.sqlite`
            
            The app will attempt to normalize your path, but it's best to use forward slashes (/) even on Windows.
            
            ### Common Issues
            
            - **Spaces in path**: Enclose the entire path in quotes if it contains spaces
            - **Network drives**: Use the full UNC path (e.g., `\\\\server\\share\\file.sqlite`)
            - **Permission issues**: Make sure you have read access to the file
            """)
        
        # Connect button for file path method
        if st.button("Connect to Database", key="connect_path_button"):
            if not db_path:
                st.error("Please enter a database path")
            else:
                # Normalize the path
                normalized_path = normalize_path(db_path)
                st.session_state.db_path = normalized_path
                
                # Validate the database path
                is_valid_path, path_message = validate_db_path(normalized_path)
                
                if is_valid_path:
                    try:
                        # Attempt to connect to the database
                        conn = get_db_connection(normalized_path)
                        
                        # Check if it's a valid LMT database
                        is_valid_lmt, tables, structure_message = check_lmt_database_structure(conn)
                        st.session_state.valid_db = is_valid_lmt
                        st.session_state.tables = tables
                        st.session_state.structure_message = structure_message
                        
                        # Show success message with database information
                        st.success(f"Successfully connected to database: {normalized_path}")
                        
                        # Get database statistics and store in session state
                        st.session_state.db_stats = get_database_statistics(conn, tables)
                        
                        # Always close the connection when done
                        conn.close()
                        
                        # Display information about tables
                        if is_valid_lmt:
                            st.write("✅ Valid LMT database structure detected")
                        else:
                            st.warning("⚠️ Database structure does not match expected LMT format")
                            st.info(structure_message)
                            
                        st.write(f"Found tables: {', '.join(tables)}")
                        
                        # Display database statistics
                        st.subheader("Database Statistics")
                        for table, count in st.session_state.db_stats.items():
                            st.write(f"Table `{table}`: {count:,} rows")
                            
                    except Exception as e:
                        st.error(f"Failed to connect to the database: {str(e)}")
                        st.session_state.valid_db = False
                else:
                    st.error(path_message)
    
    # Connect Tab 3: Connect via File Upload
    with connect_tab3:
        st.subheader("Connect via File Upload")
        st.markdown("""
        Use this method for smaller databases (under 200MB).
        Upload your SQLite database file directly.
        """)
        
        uploaded_file = st.file_uploader("Upload SQLite database", type=['sqlite', 'db'])
        
        if uploaded_file is not None:
            # Create a temporary file to store the uploaded database
            temp_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_db.sqlite")
            
            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.db_path = temp_db_path
            
            try:
                # Attempt to connect to the database
                conn = get_db_connection(temp_db_path)
                
                # Check if it's a valid LMT database
                is_valid_lmt, tables, structure_message = check_lmt_database_structure(conn)
                st.session_state.valid_db = is_valid_lmt
                st.session_state.tables = tables
                st.session_state.structure_message = structure_message
                
                # Show success message with database information
                st.success(f"Successfully connected to uploaded database")
                
                # Get database statistics and store in session state
                st.session_state.db_stats = get_database_statistics(conn, tables)
                
                # Always close the connection when done
                conn.close()
                
                # Display information about tables
                if is_valid_lmt:
                    st.write("✅ Valid LMT database structure detected")
                else:
                    st.warning("⚠️ Database structure does not match expected LMT format")
                    st.info(structure_message)
                    
                st.write(f"Found tables: {', '.join(tables)}")
                
                # Display database statistics
                st.subheader("Database Statistics")
                for table, count in st.session_state.db_stats.items():
                    st.write(f"Table `{table}`: {count:,} rows")
                    
            except Exception as e:
                st.error(f"Failed to connect to the uploaded database: {str(e)}")
                st.session_state.valid_db = False
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)

# Tab 2: SQL Queries - Use modular component
with tab2:
    render_sql_query_section()

# Tab 3: Event Processing - Use modular component
with tab3:
    render_event_processing_section()

# Tab 4: Table Operations
with tab4:
    st.header("Table Operations")
    st.markdown("""
    Manage database tables by adding or removing columns, deleting rows,
    updating animal IDs, and adding metadata.
    """)
    
    # Check if a database is connected
    if st.session_state.db_path is None:
        st.warning("Please connect to a database first using the 'Database Connection' tab")
    else:
        # Create subtabs for different operations
        subtab1, subtab2, subtab3 = st.tabs(["Animal Management", "Table Editing", "Table Export/Import"])
        
        # Subtab 1: Animal Management
        with subtab1:
            st.subheader("Animal Management")
            st.markdown("""
            Update animal IDs and metadata across all relevant tables.
            This ensures consistency when animal identifiers need to change.
            """)
            
            # Get a fresh connection
            conn = get_connection()
            if conn is None:
                st.error("Failed to connect to the database")
            else:
                try:
                    # Check if ANIMAL table exists
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ANIMAL'")
                    animal_table_exists = cursor.fetchone() is not None
                    
                    if not animal_table_exists:
                        st.error("The ANIMAL table does not exist in this database.")
                    else:
                        # Get list of animals
                        animals = get_animal_list(conn)
                        
                        # Animal ID update section
                        st.subheader("Update Animal ID")
                        st.markdown("""
                        Update an animal's ID across all tables in the database.
                        This will update references in all tables that contain animal ID columns.
                        """)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if animals:
                                old_id = st.selectbox("Select animal to update", animals, format_func=lambda x: f"Animal {x}")
                            else:
                                st.warning("No animals found in the database")
                                old_id = None
                        with col2:
                            new_id = st.number_input("New animal ID", min_value=1, step=1)
                        
                        # Dry run option
                        dry_run = st.checkbox("Dry run (preview changes without modifying database)", value=True)
                        
                        if old_id and st.button("Update Animal ID", key="update_animal_id_btn"):
                            if old_id == new_id:
                                st.warning("New ID is the same as the old ID. No changes needed.")
                            else:
                                try:
                                    # Close current connection before using the function that creates its own connection
                                    conn.close()
                                    success, stats, message = update_mouse_id(st.session_state.db_path, old_id, new_id, dry_run)
                                    
                                    if success:
                                        if dry_run:
                                            st.info("Dry run successful. The following changes would be made:")
                                        else:
                                            st.success(f"Successfully updated animal ID from {old_id} to {new_id}")
                                        
                                        # Display statistics
                                        for table, count in stats.items():
                                            if count > 0:
                                                st.write(f"Table {table}: {count} rows {'would be' if dry_run else 'were'} updated")
                                    else:
                                        st.error(message)
                                except Exception as e:
                                    st.error(f"Error updating animal ID: {str(e)}")
                        
                        # Animal metadata section
                        st.subheader("Update Animal Metadata")
                        st.markdown("""
                        Add or update metadata for animals, such as genotype, sex, age, and setup information.
                        """)
                        
                        # Get a fresh connection
                        conn = get_connection()
                        if conn is None:
                            st.error("Failed to connect to the database")
                        else:
                            # Get metadata columns
                            metadata_columns = get_animal_metadata_columns(conn)
                            
                            if not metadata_columns:
                                if st.button("Add Metadata Columns to ANIMAL Table", key="add_metadata_cols_btn"):
                                    try:
                                        # Get SQL for adding metadata columns
                                        add_metadata_sql = get_add_animal_metadata_sql()
                                        
                                        # Execute each SQL statement
                                        success = True
                                        for sql in add_metadata_sql:
                                            try:
                                                cursor.execute(sql)
                                                conn.commit()
                                            except sqlite3.Error:
                                                # Column might already exist, which is fine
                                                pass
                                        
                                        st.success("Metadata columns added to ANIMAL table")
                                        st.info("Please select an animal to update its metadata")
                                        
                                        # Refresh metadata columns
                                        metadata_columns = get_animal_metadata_columns(conn)
                                    except Exception as e:
                                        st.error(f"Error adding metadata columns: {str(e)}")
                            
                            if metadata_columns:
                                # Animal selection
                                if animals:
                                    selected_animal = st.selectbox("Select animal to update metadata", animals, format_func=lambda x: f"Animal {x}", key="metadata_animal_select")
                                    
                                    if selected_animal:
                                        try:
                                            # Get current metadata
                                            cursor.execute(f"SELECT * FROM ANIMAL WHERE ID = {selected_animal}")
                                            animal_data = cursor.fetchone()
                                            
                                            if animal_data:
                                                # Get column names
                                                cursor.execute(f"PRAGMA table_info(ANIMAL)")
                                                columns = [info[1] for info in cursor.fetchall()]
                                                
                                                animal_dict = {columns[i]: animal_data[i] for i in range(len(columns))}
                                                
                                                # Form for updating metadata
                                                st.subheader(f"Update Metadata for Animal {selected_animal}")
                                                
                                                # Create input fields for each metadata column
                                                metadata_updates = {}
                                                
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    if "GENOTYPE" in columns:
                                                        genotype = st.text_input("Genotype", value=animal_dict.get("GENOTYPE", ""))
                                                        metadata_updates["GENOTYPE"] = genotype
                                                    
                                                    if "AGE" in columns:
                                                        age = st.number_input("Age (days)", value=int(animal_dict.get("AGE", 0)) if animal_dict.get("AGE") else 0, min_value=0)
                                                        metadata_updates["AGE"] = age
                                                
                                                with col2:
                                                    if "SEX" in columns:
                                                        sex = st.selectbox("Sex", ["", "M", "F"], index=0 if not animal_dict.get("SEX") else (1 if animal_dict.get("SEX") == "M" else 2))
                                                        metadata_updates["SEX"] = sex
                                                    
                                                    if "SETUP" in columns:
                                                        setup = st.text_input("Setup", value=animal_dict.get("SETUP", ""))
                                                        metadata_updates["SETUP"] = setup
                                                
                                                if st.button("Update Metadata", key="update_metadata_btn"):
                                                    try:
                                                        # Close current connection before using the function that creates its own connection
                                                        conn.close()
                                                        success, message = update_animal_metadata(st.session_state.db_path, selected_animal, metadata_updates)
                                                        
                                                        if success:
                                                            st.success(f"Successfully updated metadata for Animal {selected_animal}")
                                                        else:
                                                            st.error(message)
                                                    except Exception as e:
                                                        st.error(f"Error updating metadata: {str(e)}")
                                        except Exception as e:
                                            st.error(f"Error retrieving animal data: {str(e)}")
                            
                            # Add new animal section
                            st.subheader("Add New Animal")
                            st.markdown("""
                            Add a new animal to the database with metadata.
                            """)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                new_animal_id = st.number_input("New Animal ID", min_value=1, step=1, key="new_animal_id_input")
                            with col2:
                                rfid = st.text_input("RFID", key="new_animal_rfid_input")
                            
                            # Metadata inputs
                            if metadata_columns:
                                new_metadata = {}
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if "GENOTYPE" in metadata_columns:
                                        genotype = st.text_input("Genotype", key="new_animal_genotype")
                                        new_metadata["GENOTYPE"] = genotype
                                    
                                    if "AGE" in metadata_columns:
                                        age = st.number_input("Age (days)", min_value=0, key="new_animal_age")
                                        new_metadata["AGE"] = age
                                
                                with col2:
                                    if "SEX" in metadata_columns:
                                        sex = st.selectbox("Sex", ["", "M", "F"], key="new_animal_sex")
                                        new_metadata["SEX"] = sex
                                    
                                    if "SETUP" in metadata_columns:
                                        setup = st.text_input("Setup", key="new_animal_setup")
                                        new_metadata["SETUP"] = setup
                            
                            if st.button("Add Animal", key="add_animal_btn"):
                                if new_animal_id:
                                    try:
                                        # Check if animal ID already exists
                                        cursor.execute(f"SELECT ID FROM ANIMAL WHERE ID = {new_animal_id}")
                                        if cursor.fetchone():
                                            st.error(f"Animal ID {new_animal_id} already exists in the database")
                                        else:
                                            # Close current connection before using the function that creates its own connection
                                            conn.close()
                                            success, message = add_animal(st.session_state.db_path, new_animal_id, rfid, new_metadata if 'new_metadata' in locals() else {})
                                            
                                            if success:
                                                st.success(f"Successfully added Animal {new_animal_id}")
                                            else:
                                                st.error(message)
                                    except Exception as e:
                                        st.error(f"Error adding animal: {str(e)}")
                                else:
                                    st.error("Please enter a valid Animal ID")
                            
                            # Delete animal section
                            st.subheader("Delete Animal")
                            st.markdown("""
                            Remove an animal from the database.
                            This will only remove the animal from the ANIMAL table, not from event tables.
                            """)
                            
                            if animals:
                                animal_to_delete = st.selectbox("Select animal to delete", animals, format_func=lambda x: f"Animal {x}", key="delete_animal_select")
                                
                                delete_confirm = st.checkbox("I understand this action cannot be undone", key="delete_animal_confirm")
                                
                                if delete_confirm and st.button("Delete Animal", key="delete_animal_btn"):
                                    try:
                                        # Close current connection before using the function that creates its own connection
                                        conn.close()
                                        success, message = delete_animal(st.session_state.db_path, animal_to_delete)
                                        
                                        if success:
                                            st.success(message)
                                        else:
                                            st.error(message)
                                    except Exception as e:
                                        st.error(f"Error deleting animal: {str(e)}")
                            else:
                                st.warning("No animals found to delete")
                            
                            # Always close the connection when done
                            if conn:
                                conn.close()
                except Exception as e:
                    st.error(f"Error in Animal Management: {str(e)}")
                    if conn:
                        conn.close()
        
        # Subtab 2: Table Editing
        with subtab2:
            st.subheader("Table Editing")
            st.markdown("""
            Edit database tables by adding/removing columns, deleting rows, or renaming tables.
            """)
            
            # Get a fresh connection
            conn = get_connection()
            if conn is None:
                st.error("Failed to connect to the database")
            else:
                try:
                    # Get list of tables
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    if not tables:
                        st.warning("No tables found in the database")
                    else:
                        # Table selection
                        selected_table = st.selectbox("Select a table to edit", tables, key="edit_table_select")
                        
                        if selected_table:
                            # Get table info
                            cursor.execute(f"PRAGMA table_info({selected_table})")
                            columns = [info[1] for info in cursor.fetchall()]
                            
                            # Show sample data
                            with st.expander("View sample data", expanded=True):
                                sample_df = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 5", conn)
                                st.dataframe(sample_df)
                            
                            # Table operations 
                            operations = st.radio(
                                "Select operation",
                                ["Remove Columns", "Delete Rows", "Add Column", "Rename Table", "Copy Table"],
                                key="table_operation_radio"
                            )
                            
                            if operations == "Remove Columns":
                                st.subheader("Remove Columns")
                                st.markdown("""
                                Select columns to keep. All other columns will be removed.
                                This creates a new table without the unwanted columns.
                                """)
                                
                                if columns:
                                    columns_to_keep = st.multiselect("Select columns to keep", columns, default=columns)
                                    
                                    if len(columns_to_keep) == 0:
                                        st.error("You must select at least one column to keep")
                                    elif len(columns_to_keep) == len(columns):
                                        st.info("All columns are selected. No columns will be removed.")
                                    else:
                                        if st.button("Remove Columns", key="remove_columns_btn"):
                                            try:
                                                # Close current connection before using the function that creates its own connection
                                                conn.close()
                                                success, message = remove_columns(st.session_state.db_path, selected_table, columns_to_keep)
                                                
                                                if success:
                                                    st.success(message)
                                                else:
                                                    st.error(message)
                                            except Exception as e:
                                                st.error(f"Error removing columns: {str(e)}")
                                else:
                                    st.warning("No columns found in the selected table")
                            
                            elif operations == "Delete Rows":
                                st.subheader("Delete Rows")
                                st.markdown("""
                                Delete rows that match specific criteria.
                                Use SQL WHERE clause syntax to specify the condition.
                                """)
                                
                                where_clause = st.text_input(
                                    "WHERE clause (SQL syntax)",
                                    help="Example: id > 100 or name = 'test'"
                                )
                                
                                if st.button("Preview Affected Rows", key="preview_delete_rows_btn"):
                                    if where_clause:
                                        try:
                                            preview_df = pd.read_sql(f"SELECT * FROM {selected_table} WHERE {where_clause} LIMIT 10", conn)
                                            row_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {selected_table} WHERE {where_clause}", conn).iloc[0, 0]
                                            
                                            st.write(f"This will delete {row_count} rows. Preview of affected rows:")
                                            st.dataframe(preview_df)
                                            
                                            if row_count > 0:
                                                delete_confirm = st.checkbox("I understand this action cannot be undone", key="delete_rows_confirm")
                                                
                                                if delete_confirm and st.button("Delete Rows", key="delete_rows_btn"):
                                                    try:
                                                        # Close current connection before using the function that creates its own connection
                                                        conn.close()
                                                        success, message = delete_rows(st.session_state.db_path, selected_table, where_clause)
                                                        
                                                        if success:
                                                            st.success(message)
                                                        else:
                                                            st.error(message)
                                                    except Exception as e:
                                                        st.error(f"Error deleting rows: {str(e)}")
                                        except Exception as e:
                                            st.error(f"Error in preview: {str(e)}")
                                    else:
                                        st.error("Please enter a WHERE clause")
                            
                            elif operations == "Add Column":
                                st.subheader("Add Column")
                                st.markdown("""
                                Add a new column to the table with optional default value.
                                """)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    column_name = st.text_input("Column Name", key="add_column_name")
                                with col2:
                                    column_type = st.selectbox("Column Type", ["TEXT", "INTEGER", "REAL", "BLOB"], key="add_column_type")
                                with col3:
                                    default_value = st.text_input("Default Value (optional)", key="add_column_default")
                                
                                if st.button("Add Column", key="add_column_btn"):
                                    if column_name:
                                        try:
                                            # Close current connection before using the function that creates its own connection
                                            conn.close()
                                            success, message = add_column(st.session_state.db_path, selected_table, column_name, column_type, default_value if default_value else None)
                                            
                                            if success:
                                                st.success(message)
                                            else:
                                                st.error(message)
                                        except Exception as e:
                                            st.error(f"Error adding column: {str(e)}")
                                    else:
                                        st.error("Please enter a column name")
                            
                            elif operations == "Rename Table":
                                st.subheader("Rename Table")
                                st.markdown("""
                                Rename the selected table.
                                """)
                                
                                new_table_name = st.text_input("New Table Name", key="rename_table_name")
                                
                                if st.button("Rename Table", key="rename_table_btn"):
                                    if new_table_name:
                                        if new_table_name in tables:
                                            st.error(f"Table '{new_table_name}' already exists in the database")
                                        else:
                                            try:
                                                # Close current connection before using the function that creates its own connection
                                                conn.close()
                                                success, message = rename_table(st.session_state.db_path, selected_table, new_table_name)
                                                
                                                if success:
                                                    st.success(message)
                                                else:
                                                    st.error(message)
                                            except Exception as e:
                                                st.error(f"Error renaming table: {str(e)}")
                                    else:
                                        st.error("Please enter a new table name")
                            
                            elif operations == "Copy Table":
                                st.subheader("Copy Table")
                                st.markdown("""
                                Create a copy of the selected table with a new name.
                                """)
                                
                                new_table_name = st.text_input("New Table Name", key="copy_table_name")
                                copy_data = st.checkbox("Copy data (uncheck to create empty table with same structure)", value=True)
                                
                                if st.button("Copy Table", key="copy_table_btn"):
                                    if new_table_name:
                                        if new_table_name in tables:
                                            st.error(f"Table '{new_table_name}' already exists in the database")
                                        else:
                                            try:
                                                # Close current connection before using the function that creates its own connection
                                                conn.close()
                                                success, message = copy_table(st.session_state.db_path, selected_table, new_table_name, copy_data)
                                                
                                                if success:
                                                    st.success(message)
                                                else:
                                                    st.error(message)
                                            except Exception as e:
                                                st.error(f"Error copying table: {str(e)}")
                                    else:
                                        st.error("Please enter a new table name")
                                        
                    # Always close the connection when done
                    conn.close()
                except Exception as e:
                    st.error(f"Error in Table Editing: {str(e)}")
                    if conn:
                        conn.close()
        
        # Subtab 3: Table Export/Import
        with subtab3:
            st.subheader("Table Export/Import")
            st.markdown("""
            Export tables to CSV files or import CSV data into tables.
            """)
            
            # Get a fresh connection
            conn = get_connection()
            if conn is None:
                st.error("Failed to connect to the database")
            else:
                try:
                    # Get list of tables
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    if not tables:
                        st.warning("No tables found in the database")
                    else:
                        # Export section
                        st.subheader("Export Table to CSV")
                        selected_table = st.selectbox("Select a table to export", tables, key="export_table_select")
                        
                        if selected_table:
                            # Get row count
                            row_count = get_table_row_count(conn, selected_table)
                            st.write(f"Table contains {row_count:,} rows")
                            
                            # Limit option
                            limit = st.number_input("Limit rows (0 for all rows)", min_value=0, value=0, key="export_limit_input")
                            
                            if st.button("Export to CSV", key="export_table_btn"):
                                try:
                                    # Close current connection before using the function that creates its own connection
                                    conn.close()
                                    success, csv_data, message = export_table(st.session_state.db_path, selected_table, limit if limit > 0 else None)
                                    
                                    if success:
                                        st.success(message)
                                        
                                        # Create download button
                                        st.download_button(
                                            label="Download CSV",
                                            data=csv_data,
                                            file_name=f"{selected_table}.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.error(message)
                                except Exception as e:
                                    st.error(f"Error exporting table: {str(e)}")
                        
                        # Import section
                        st.subheader("Import CSV to Table")
                        st.markdown("""
                        Import data from a CSV file into a new or existing table.
                        """)
                        
                        import_option = st.radio(
                            "Import option",
                            ["Create new table", "Import to existing table"],
                            key="import_option_radio"
                        )
                        
                        if import_option == "Create new table":
                            new_table_name = st.text_input("New table name", key="import_new_table_name")
                        else:
                            existing_table = st.selectbox("Select existing table", tables, key="import_existing_table")
                        
                        uploaded_file = st.file_uploader("Upload CSV file", type="csv", key="import_csv_file")
                        
                        if uploaded_file is not None:
                            # Preview the uploaded CSV
                            try:
                                df = pd.read_csv(uploaded_file)
                                st.write("CSV Preview (first 5 rows):")
                                st.dataframe(df.head(5))
                                
                                # CSV info
                                st.write(f"CSV has {len(df):,} rows and {len(df.columns)} columns")
                                
                                # Import options
                                if st.session_state.db_path is not None:
                                    st.subheader("Import Options")
                                    
                                    # Table name input
                                    table_name = st.text_input("Table name", value=os.path.splitext(uploaded_file.name)[0].upper(), key="import_table_name")
                                    
                                    # Import button
                                    if st.button("Import to Database", key="import_csv_db_btn"):
                                        if table_name:
                                            try:
                                                # Save to temp CSV
                                                temp_csv_path = os.path.join(os.path.dirname(st.session_state.db_path), "temp_import.csv")
                                                df.to_csv(temp_csv_path, index=False)
                                                
                                                success, message = csv_to_sql(temp_csv_path, st.session_state.db_path, table_name)
                                                
                                                # Clean up temp file
                                                if os.path.exists(temp_csv_path):
                                                    os.remove(temp_csv_path)
                                                
                                                if success:
                                                    st.success(message)
                                                else:
                                                    st.error(message)
                                            except Exception as e:
                                                st.error(f"Error importing CSV: {str(e)}")
                                        else:
                                            st.error("Please enter a table name")
                                    else:
                                        st.warning("Please connect to a database first to import CSV data")
                            except Exception as e:
                                st.error(f"Error reading CSV file: {str(e)}")
                    
                    # Always close the connection when done
                    conn.close()
                except Exception as e:
                    st.error(f"Error in Table Export/Import: {str(e)}")
                    if conn:
                        conn.close()

# Tab 5: Database Merging
with tab5:
    st.header("Database Merging")
    st.markdown("""
    Merge multiple databases into a single database for combined analysis.
    This is useful when you have multiple experimental databases that need to be analyzed together.
    """)
    
    # Database merging doesn't always require an active connection to the current database
    st.subheader("Select Databases to Merge")
    
    # Allow users to add multiple database paths
    st.markdown("""
    Enter the paths to the databases you want to merge. You can add multiple databases.
    The resulting merged database will contain selected tables from all source databases.
    """)
    
    # Initialize session state for selected databases
    if 'selected_db_files' not in st.session_state:
        st.session_state.selected_db_files = []
    
    # Database selection
    col1, col2 = st.columns([3, 1])
    with col1:
        db_path = st.text_input("Enter database path", key="merge_db_path")
    with col2:
        if st.button("Add Database", key="add_db_btn"):
            if db_path:
                # Normalize path
                norm_path = normalize_path(db_path)
                
                # Validate path
                is_valid, message = validate_db_path(norm_path)
                if is_valid:
                    if norm_path not in st.session_state.selected_db_files:
                        st.session_state.selected_db_files.append(norm_path)
                        st.success(f"Added database: {norm_path}")
                    else:
                        st.warning("This database is already in the list")
                else:
                    st.error(message)
            else:
                st.error("Please enter a database path")
    
    # Upload option
    uploaded_file = st.file_uploader("Or upload a database file", type=['sqlite', 'db'])
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded database
        temp_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"temp_merge_{uploaded_file.name}")
        
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if temp_db_path not in st.session_state.selected_db_files:
            st.session_state.selected_db_files.append(temp_db_path)
            st.success(f"Added uploaded database: {uploaded_file.name}")
    
    # Display selected databases
    if st.session_state.selected_db_files:
        st.subheader("Selected Databases")
        for i, db_file in enumerate(st.session_state.selected_db_files):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"{i+1}. {db_file}")
            with col2:
                if st.button("Remove", key=f"remove_db_{i}"):
                    st.session_state.selected_db_files.pop(i)
                    st.experimental_rerun()
        
        # Option to clear all
        if st.button("Clear All Databases", key="clear_all_dbs"):
            st.session_state.selected_db_files = []
            st.experimental_rerun()
    else:
        st.info("No databases selected. Add at least two databases to merge.")
    
    # Select tables to merge if we have at least two databases
    if len(st.session_state.selected_db_files) >= 2:
        st.subheader("Select Tables to Merge")
        
        # Get tables from all databases
        all_tables = set()
        table_db_map = {}
        
        for db_path in st.session_state.selected_db_files:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    all_tables.add(table)
                    if table not in table_db_map:
                        table_db_map[table] = []
                    table_db_map[table].append(os.path.basename(db_path))
                
                conn.close()
            except Exception as e:
                st.error(f"Error getting tables from {db_path}: {str(e)}")
        
        # Display tables with database presence information
        if all_tables:
            st.markdown("Select which tables you want to include in the merged database:")
            
            # Group tables by presence in all databases
            common_tables = [table for table in all_tables if len(table_db_map[table]) == len(st.session_state.selected_db_files)]
            partial_tables = [table for table in all_tables if table not in common_tables]
            
            # Select all option for common tables
            select_all_common = st.checkbox("Select all common tables", value=True)
            
            # Common tables
            st.markdown("### Common Tables (present in all databases)")
            common_tables_selected = []
            for table in sorted(common_tables):
                default = select_all_common
                selected = st.checkbox(f"{table}", value=default, key=f"table_select_{table}")
                if selected:
                    common_tables_selected.append(table)
            
            # Partial tables
            if partial_tables:
                st.markdown("### Partial Tables (present in some databases)")
                partial_tables_selected = []
                for table in sorted(partial_tables):
                    presence = f"(Present in {len(table_db_map[table])}/{len(st.session_state.selected_db_files)} databases)"
                    selected = st.checkbox(f"{table} {presence}", key=f"table_select_{table}")
                    if selected:
                        partial_tables_selected.append(table)
                
                selected_tables = common_tables_selected + partial_tables_selected
            else:
                selected_tables = common_tables_selected
            
            # Store selected tables in session state
            st.session_state.merged_tables = selected_tables
            
            # Output database path
            st.subheader("Output Settings")
            
            output_dir = os.path.dirname(st.session_state.selected_db_files[0])
            default_output = os.path.join(output_dir, "merged_database.sqlite")
            
            output_path = st.text_input("Output database path", value=default_output, key="merge_output_path")
            st.session_state.merged_output_path = output_path
            
            # Export options
            export_csv = st.checkbox("Also export tables to CSV files", value=False, key="export_csv_checkbox")
            export_dir = None
            
            if export_csv:
                default_csv_dir = os.path.join(os.path.dirname(output_path), "csv_export")
                export_dir = st.text_input("CSV export directory", value=default_csv_dir, key="csv_export_dir")
            
            # Merge button
            if st.button("Merge Databases", key="merge_db_button", disabled=len(selected_tables) == 0):
                if len(selected_tables) == 0:
                    st.error("Please select at least one table to merge")
                else:
                    try:
                        # Show progress bar
                        progress_bar = st.progress(0)
                        
                        # Create a function for progress updates
                        def update_progress(progress):
                            progress_bar.progress(progress)
                        
                        with st.spinner("Merging databases..."):
                            success, stats, message = merge_databases(
                                st.session_state.selected_db_files,
                                selected_tables,
                                output_path,
                                update_progress
                            )
                            
                            if success:
                                st.success("✅ Databases merged successfully!")
                                
                                # Store result for display
                                st.session_state.merge_result = {
                                    'success': True,
                                    'stats': stats,
                                    'message': message,
                                    'output_path': output_path
                                }
                                
                                # Export to CSV if requested
                                if export_csv and export_dir:
                                    with st.spinner("Exporting tables to CSV..."):
                                        csv_success, csv_files, csv_message = export_tables_to_csv(
                                            [output_path],
                                            selected_tables,
                                            export_dir
                                        )
                                        
                                        if csv_success:
                                            st.success(f"✅ Tables exported to CSV in {export_dir}")
                                            st.session_state.merge_result['csv_files'] = csv_files
                                        else:
                                            st.error(f"Error exporting to CSV: {csv_message}")
                            else:
                                st.error(f"Error merging databases: {message}")
                                st.session_state.merge_result = {
                                    'success': False,
                                    'message': message
                                }
                    except Exception as e:
                        st.error(f"Error during merge: {str(e)}")
            
            # Display merge results if available
            if 'merge_result' in st.session_state and st.session_state.merge_result:
                result = st.session_state.merge_result
                
                if result['success']:
                    st.subheader("Merge Results")
                    st.write(f"Output database: {result['output_path']}")
                    
                    # Display statistics
                    stats = result['stats']
                    st.write(f"Tables processed: {stats['tables_processed']}")
                    
                    if 'rows_merged' in stats:
                        st.subheader("Rows Merged")
                        for table, count in stats['rows_merged'].items():
                            st.write(f"Table `{table}`: {count:,} rows")
                    
                    # Display errors if any
                    if 'errors' in stats and stats['errors']:
                        st.subheader("Errors")
                        for error in stats['errors']:
                            st.error(error)
                    
                    # Display CSV export info if available
                    if 'csv_files' in result:
                        st.subheader("CSV Export")
                        for csv_file in result['csv_files']:
                            st.write(f"- {csv_file}")
                    
                    # Option to connect to the merged database
                    if st.button("Connect to Merged Database", key="connect_merged_db"):
                        st.session_state.db_path = result['output_path']
                        
                        # Get fresh connection to check the database
                        conn = get_connection()
                        if conn is not None:
                            # Check if it's a valid LMT database
                            is_valid_lmt, tables, structure_message = check_lmt_database_structure(conn)
                            st.session_state.valid_db = is_valid_lmt
                            st.session_state.tables = tables
                            st.session_state.structure_message = structure_message
                            
                            # Get database statistics
                            st.session_state.db_stats = get_database_statistics(conn, tables)
                            
                            # Close the connection
                            conn.close()
                            
                            st.success(f"Now connected to merged database: {result['output_path']}")
                            st.experimental_rerun()
                else:
                    st.error(f"Merge failed: {result['message']}")
        else:
            st.warning("No tables found in the selected databases")
    elif len(st.session_state.selected_db_files) == 1:
        st.warning("Please add at least one more database to merge")
        
    # Help information
    with st.expander("How Database Merging Works", expanded=False):
        st.markdown("""
        ### Database Merging Process
        
        1. **Table Detection**: The system identifies tables present in each database
        2. **Data Collection**: For each selected table, data is collected from all databases
        3. **Consolidation**: Data is combined into a single table in the output database
        4. **Mapping**: If table names differ slightly between databases, the system attempts to map them
        
        ### Best Practices
        
        - Ensure tables have the same structure across databases for best results
        - Common tables (present in all databases) are most reliable to merge
        - Partial tables can still be merged, but may have missing data
        - For large databases, the process may take several minutes
        """)

# Tab 6: CSV Operations
with tab6:
    st.header("CSV Operations")
    st.markdown("""
    Process CSV files or export database tables to CSV format.
    This allows you to work with your data in spreadsheet software or other analysis tools.
    """)
    
    # Create subtabs for different CSV operations
    csv_tab1, csv_tab2, csv_tab3 = st.tabs(["Import/Export CSV", "Edit CSV", "Batch Export"])
    
    # CSV Tab 1: Import/Export CSV
    with csv_tab1:
        st.subheader("Import/Export CSV")
        st.markdown("""
        Import CSV files to your database or export tables to CSV format.
        """)
        
        # Export tables section
        if st.session_state.db_path is not None:
            st.subheader("Export Database Tables to CSV")
            
            # Get a fresh connection
            conn = get_connection()
            if conn is None:
                st.error("Failed to connect to the database")
            else:
                try:
                    # Get list of tables
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    if tables:
                        # Let the user select tables to export
                        selected_tables = st.multiselect("Select tables to export", tables, key="export_tables_multiselect")
                        
                        if selected_tables:
                            # Export options
                            export_dir = st.text_input(
                                "Export directory", 
                                value=os.path.join(os.path.dirname(st.session_state.db_path), "csv_export"),
                                key="table_export_dir"
                            )
                            
                            # Export button
                            if st.button("Export Selected Tables", key="export_tables_btn"):
                                try:
                                    # Close current connection before using function that creates its own connection
                                    conn.close()
                                    
                                    # Make sure export directory exists
                                    os.makedirs(export_dir, exist_ok=True)
                                    
                                    success, exported_files, message = export_multiple_tables_to_csv(
                                        [st.session_state.db_path],
                                        selected_tables,
                                        export_dir
                                    )
                                    
                                    if success:
                                        st.success(f"Successfully exported {len(exported_files)} tables to CSV")
                                        
                                        # Display exported files
                                        for file_path in exported_files:
                                            file_name = os.path.basename(file_path)
                                            st.write(f"- {file_name}")
                                            
                                        # Offer to download as ZIP if multiple files
                                        if len(exported_files) > 1:
                                            # Create a zip file
                                            zip_path = os.path.join(export_dir, "tables_export.zip")
                                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                                for file_path in exported_files:
                                                    zipf.write(file_path, os.path.basename(file_path))
                                            
                                            # Read the zip file for download
                                            with open(zip_path, "rb") as f:
                                                zip_data = f.read()
                                            
                                            st.download_button(
                                                label="Download All as ZIP",
                                                data=zip_data,
                                                file_name="tables_export.zip",
                                                mime="application/zip"
                                            )
                                    else:
                                        st.error(message)
                                except Exception as e:
                                    st.error(f"Error exporting tables: {str(e)}")
                    else:
                        st.warning("No tables found in the database")
                    
                    # Close the connection when done
                    conn.close()
                except Exception as e:
                    st.error(f"Error listing tables: {str(e)}")
                    if conn:
                        conn.close()
        
        # Import CSV section
        st.subheader("Import CSV to Database")
        st.markdown("""
        Upload a CSV file and import it into a new database table.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv", key="import_csv_file")
        
        if uploaded_file is not None:
            try:
                # Read and preview the CSV
                df = pd.read_csv(uploaded_file)
                
                # Store in session state for use in other tabs
                st.session_state.csv_df = df
                st.session_state.csv_filename = uploaded_file.name
                st.session_state.csv_modified = False
                
                # Display preview
                st.write("CSV Preview (first 5 rows):")
                st.dataframe(df.head(5))
                
                # CSV info
                st.write(f"CSV has {len(df):,} rows and {len(df.columns)} columns")
                
                # Import options
                if st.session_state.db_path is not None:
                    st.subheader("Import Options")
                    
                    # Table name input
                    table_name = st.text_input("Table name", value=os.path.splitext(uploaded_file.name)[0].upper(), key="import_table_name")
                    
                    # Import button
                    if st.button("Import to Database", key="import_csv_db_btn"):
                        if table_name:
                            try:
                                # Save to temp CSV
                                temp_csv_path = os.path.join(os.path.dirname(st.session_state.db_path), "temp_import.csv")
                                df.to_csv(temp_csv_path, index=False)
                                
                                success, message = csv_to_sql(temp_csv_path, st.session_state.db_path, table_name)
                                
                                # Clean up temp file
                                if os.path.exists(temp_csv_path):
                                    os.remove(temp_csv_path)
                                
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"Error importing CSV: {str(e)}")
                        else:
                            st.error("Please enter a table name")
                else:
                    st.warning("Please connect to a database first to import CSV data")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # CSV Tab 2: Edit CSV
    with csv_tab2:
        st.subheader("Edit CSV")
        st.markdown("""
        Modify CSV files by removing columns, filtering rows, or adding calculated columns.
        """)
        
        if 'csv_df' in st.session_state and st.session_state.csv_df is not None:
            # Display current CSV info
            df = st.session_state.csv_df
            filename = st.session_state.csv_filename
            
            st.write(f"Editing: {filename}")
            st.write(f"Current data: {len(df):,} rows, {len(df.columns)} columns")
            
            if st.session_state.csv_modified:
                st.info("CSV has been modified since upload")
            
            # Show preview
            with st.expander("Show Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Edit operations
            operations = st.radio(
                "Select operation",
                ["Remove Columns", "Filter Rows", "Sort Data", "Add Calculated Column"],
                key="csv_edit_operation"
            )
            
            if operations == "Remove Columns":
                st.subheader("Remove Columns")
                
                # Let the user select columns to keep
                columns_to_keep = st.multiselect("Select columns to keep", df.columns.tolist(), default=df.columns.tolist(), key="csv_cols_to_keep")
                
                if st.button("Apply Column Removal", key="apply_col_removal"):
                    if len(columns_to_keep) == 0:
                        st.error("You must select at least one column to keep")
                    elif len(columns_to_keep) == len(df.columns):
                        st.info("All columns are selected. No columns will be removed.")
                    else:
                        try:
                            # Remove columns
                            columns_to_remove = [col for col in df.columns if col not in columns_to_keep]
                            new_df = remove_columns_csv(df, columns_to_keep)
                            
                            # Update session state
                            st.session_state.csv_df = new_df
                            st.session_state.csv_modified = True
                            
                            st.success(f"Removed {len(columns_to_remove)} columns: {', '.join(columns_to_remove)}")
                            st.dataframe(new_df.head(10))
                        except Exception as e:
                            st.error(f"Error removing columns: {str(e)}")
            
            elif operations == "Filter Rows":
                st.subheader("Filter Rows")
                st.markdown("""
                Filter rows based on column values.
                Only rows matching the filter criteria will be kept.
                """)
                
                # Select column to filter on
                filter_col = st.selectbox("Select column to filter on", df.columns.tolist(), key="filter_column")
                
                if filter_col:
                    # Get unique values for the column
                    if df[filter_col].dtype == 'object' or df[filter_col].dtype == 'string':
                        # For text columns, show unique values
                        unique_values = df[filter_col].unique()
                        if len(unique_values) <= 20:  # Only show if reasonably small
                            filter_value = st.selectbox("Select value to keep", ["(All)"] + list(unique_values), key="filter_value")
                            if filter_value != "(All)" and st.button("Apply Filter", key="apply_text_filter"):
                                try:
                                    new_df = filter_csv_by_column_value(df, filter_col, filter_value)
                                    
                                    # Update session state
                                    st.session_state.csv_df = new_df
                                    st.session_state.csv_modified = True
                                    
                                    st.success(f"Filtered rows: kept {len(new_df):,} of {len(df):,} rows where {filter_col} = '{filter_value}'")
                                    st.dataframe(new_df.head(10))
                                except Exception as e:
                                    st.error(f"Error filtering rows: {str(e)}")
                        else:
                            # Too many unique values to show as dropdown
                            filter_value = st.text_input("Enter value to keep", key="text_filter_value")
                            if filter_value and st.button("Apply Filter", key="apply_text_filter_input"):
                                try:
                                    new_df = filter_csv_by_column_value(df, filter_col, filter_value)
                                    
                                    # Update session state
                                    st.session_state.csv_df = new_df
                                    st.session_state.csv_modified = True
                                    
                                    st.success(f"Filtered rows: kept {len(new_df):,} of {len(df):,} rows where {filter_col} = '{filter_value}'")
                                    st.dataframe(new_df.head(10))
                                except Exception as e:
                                    st.error(f"Error filtering rows: {str(e)}")
                    else:
                        # For numeric columns, provide a range
                        min_val = float(df[filter_col].min())
                        max_val = float(df[filter_col].max())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            range_min = st.number_input("Minimum value", value=min_val, key="filter_min")
                        with col2:
                            range_max = st.number_input("Maximum value", value=max_val, key="filter_max")
                        
                        if st.button("Apply Range Filter", key="apply_range_filter"):
                            try:
                                new_df = df[(df[filter_col] >= range_min) & (df[filter_col] <= range_max)].copy()
                                
                                # Update session state
                                st.session_state.csv_df = new_df
                                st.session_state.csv_modified = True
                                
                                st.success(f"Filtered rows: kept {len(new_df):,} of {len(df):,} rows where {filter_col} is between {range_min} and {range_max}")
                                st.dataframe(new_df.head(10))
                            except Exception as e:
                                st.error(f"Error filtering rows: {str(e)}")
            
            elif operations == "Sort Data":
                st.subheader("Sort Data")
                st.markdown("""
                Sort the data by one or more columns.
                """)
                
                # Select column to sort by
                sort_col = st.selectbox("Select column to sort by", df.columns.tolist(), key="sort_column")
                sort_ascending = st.checkbox("Sort ascending (unchecked = descending)", value=True, key="sort_direction")
                
                if st.button("Apply Sorting", key="apply_sorting"):
                    try:
                        new_df = sort_csv_by_column(df, sort_col, ascending=sort_ascending)
                        
                        # Update session state
                        st.session_state.csv_df = new_df
                        st.session_state.csv_modified = True
                        
                        st.success(f"Data sorted by '{sort_col}' in {'ascending' if sort_ascending else 'descending'} order")
                        st.dataframe(new_df.head(10))
                    except Exception as e:
                        st.error(f"Error sorting data: {str(e)}")
            
            elif operations == "Add Calculated Column":
                st.subheader("Add Calculated Column")
                st.markdown("""
                Add a new column based on calculations from existing columns.
                You can use simple expressions or select from common operations.
                """)
                
                new_col_name = st.text_input("New column name", key="new_col_name")
                
                calculation_type = st.radio(
                    "Calculation type",
                    ["Simple operation", "Formula expression"],
                    key="calc_type"
                )
                
                if calculation_type == "Simple operation":
                    # Simple operations between two columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        first_col = st.selectbox("First column", df.columns.tolist(), key="first_col")
                    
                    with col2:
                        operation = st.selectbox("Operation", ["+", "-", "*", "/", "mean", "max", "min"], key="operation")
                    
                    if operation in ["+", "-", "*", "/"]:
                        with col3:
                            second_col = st.selectbox("Second column", df.columns.tolist(), key="second_col")
                    
                    if st.button("Add Column", key="add_simple_col"):
                        if not new_col_name:
                            st.error("Please enter a name for the new column")
                        else:
                            try:
                                # Perform calculation
                                if operation == "+":
                                    result = df[first_col] + df[second_col]
                                elif operation == "-":
                                    result = df[first_col] - df[second_col]
                                elif operation == "*":
                                    result = df[first_col] * df[second_col]
                                elif operation == "/":
                                    result = df[first_col] / df[second_col]
                                elif operation == "mean":
                                    result = df[first_col].mean()
                                elif operation == "max":
                                    result = df[first_col].max()
                                elif operation == "min":
                                    result = df[first_col].min()
                                
                                # Add new column
                                new_df = add_column_csv(df, new_col_name, result)
                                
                                # Update session state
                                st.session_state.csv_df = new_df
                                st.session_state.csv_modified = True
                                
                                st.success(f"Added new column '{new_col_name}'")
                                st.dataframe(new_df.head(10))
                            except Exception as e:
                                st.error(f"Error adding column: {str(e)}")
                
                else:  # Formula expression
                    st.markdown("""
                    Enter a formula using Python expressions. You can use column names in your formula.
                    
                    Examples:
                    - `df['column1'] + df['column2']`
                    - `df['column1'] * 2`
                    - `df['column1'].str.len()`
                    """)
                    
                    formula = st.text_area("Formula expression", key="formula_expr")
                    
                    if st.button("Add Column", key="add_formula_col"):
                        if not new_col_name:
                            st.error("Please enter a name for the new column")
                        elif not formula:
                            st.error("Please enter a formula expression")
                        else:
                            try:
                                # Evaluate the formula
                                result = eval(formula)
                                
                                # Add new column
                                new_df = add_column_csv(df, new_col_name, result)
                                
                                # Update session state
                                st.session_state.csv_df = new_df
                                st.session_state.csv_modified = True
                                
                                st.success(f"Added new column '{new_col_name}'")
                                st.dataframe(new_df.head(10))
                            except Exception as e:
                                st.error(f"Error evaluating formula: {str(e)}")
            
            # Save changes section
            st.subheader("Save Changes")
            
            save_format = st.radio("Save format", ["CSV", "Excel", "Database Table"], key="save_format")
            
            if save_format == "CSV":
                if st.button("Download CSV", key="download_csv_btn"):
                    try:
                        csv_data = df.to_csv(index=False)
                        filename_base = os.path.splitext(filename)[0]
                        
                        st.download_button(
                            label="Download CSV file",
                            data=csv_data,
                            file_name=f"{filename_base}_modified.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error creating CSV: {str(e)}")
            
            elif save_format == "Excel":
                if st.button("Download Excel", key="download_excel_btn"):
                    try:
                        # Create an in-memory Excel file
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df.to_excel(writer, sheet_name="Data", index=False)
                        
                        filename_base = os.path.splitext(filename)[0]
                        
                        st.download_button(
                            label="Download Excel file",
                            data=output.getvalue(),
                            file_name=f"{filename_base}_modified.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error creating Excel file: {str(e)}")
            
            elif save_format == "Database Table" and st.session_state.db_path is not None:
                table_name = st.text_input("Table name", value=os.path.splitext(filename)[0].upper(), key="save_table_name")
                
                if st.button("Save to Database", key="save_to_db_btn"):
                    if table_name:
                        try:
                            # Save to temp CSV
                            temp_csv_path = os.path.join(os.path.dirname(st.session_state.db_path), "temp_save.csv")
                            df.to_csv(temp_csv_path, index=False)
                            
                            success, message = csv_to_sql(temp_csv_path, st.session_state.db_path, table_name)
                            
                            # Clean up temp file
                            if os.path.exists(temp_csv_path):
                                os.remove(temp_csv_path)
                            
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"Error saving to database: {str(e)}")
                    else:
                        st.error("Please enter a table name")
        else:
            st.info("Please upload a CSV file in the 'Import/Export CSV' tab first")
    
    # CSV Tab 3: Batch Export
    with csv_tab3:
        st.subheader("Batch Export")
        st.markdown("""
        Export multiple tables from the database to CSV files at once.
        Useful for backing up your database or preparing data for external analysis.
        """)
        
        if st.session_state.db_path is not None:
            # Get a fresh connection
            conn = get_connection()
            if conn is None:
                st.error("Failed to connect to the database")
            else:
                try:
                    # Get list of tables
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    if tables:
                        # Let the user select all tables
                        select_all = st.checkbox("Select all tables", key="select_all_tables")
                        
                        if select_all:
                            selected_tables = tables
                        else:
                            selected_tables = st.multiselect("Select tables to export", tables, key="batch_export_tables")
                        
                        if selected_tables:
                            st.write(f"Selected {len(selected_tables)} tables for export")
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                export_dir = st.text_input(
                                    "Export directory", 
                                    value=os.path.join(os.path.dirname(st.session_state.db_path), "csv_export"),
                                    key="batch_export_dir"
                                )
                            
                            with col2:
                                file_prefix = st.text_input("File prefix (optional)", key="file_prefix")
                            
                            # File options
                            export_format = st.radio(
                                "Export format",
                                ["CSV files", "Excel file (one sheet per table)"],
                                key="export_format"
                            )
                            
                            # Export button
                            if st.button("Export Tables", key="batch_export_btn"):
                                try:
                                    # Make sure export directory exists
                                    os.makedirs(export_dir, exist_ok=True)
                                    
                                    if export_format == "CSV files":
                                        # Close current connection before using function that creates its own connection
                                        conn.close()
                                        
                                        success, exported_files, message = export_multiple_tables_to_csv(
                                            [st.session_state.db_path],
                                            selected_tables,
                                            export_dir,
                                            prefix=file_prefix
                                        )
                                        
                                        if success:
                                            st.success(f"Successfully exported {len(exported_files)} tables to CSV")
                                            
                                            # Display exported files
                                            for file_path in exported_files:
                                                file_name = os.path.basename(file_path)
                                                st.write(f"- {file_name}")
                                                
                                            # Offer to download as ZIP
                                            zip_path = os.path.join(export_dir, "tables_export.zip")
                                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                                for file_path in exported_files:
                                                    zipf.write(file_path, os.path.basename(file_path))
                                            
                                            # Read the zip file for download
                                            with open(zip_path, "rb") as f:
                                                zip_data = f.read()
                                            
                                            st.download_button(
                                                label="Download All as ZIP",
                                                data=zip_data,
                                                file_name="tables_export.zip",
                                                mime="application/zip"
                                            )
                                        else:
                                            st.error(message)
                                    else:  # Excel format
                                        # Export to Excel with one sheet per table
                                        excel_path = os.path.join(export_dir, f"{file_prefix if file_prefix else 'export'}_tables.xlsx")
                                        
                                        # Create a writer
                                        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                                            for table in selected_tables:
                                                # Read table
                                                df = pd.read_sql(f"SELECT * FROM {table}", conn)
                                                
                                                # Clean sheet name (Excel has 31 char limit and no special chars)
                                                sheet_name = table[:31].replace('/', '_').replace('\\', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_').replace(':', '_')
                                                
                                                # Write to Excel
                                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                                        
                                        st.success(f"Successfully exported {len(selected_tables)} tables to Excel file")
                                        
                                        # Offer to download
                                        with open(excel_path, "rb") as f:
                                            excel_data = f.read()
                                        
                                        st.download_button(
                                            label="Download Excel File",
                                            data=excel_data,
                                            file_name=os.path.basename(excel_path),
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                except Exception as e:
                                    st.error(f"Error exporting tables: {str(e)}")
                        else:
                            st.info("Please select at least one table to export")
                    else:
                        st.warning("No tables found in the database")
                    
                    # Close the connection when done
                    if conn:
                        conn.close()
                except Exception as e:
                    st.error(f"Error listing tables: {str(e)}")
                    if conn:
                        conn.close()
        else:
            st.warning("Please connect to a database first to export tables")

# Display footer
st.markdown("---")
st.markdown("""
### About Database Management

This unified tool provides a complete workflow for managing LMT databases:

1. **Database Connection**: Connect to existing databases or upload new ones
2. **SQL Queries**: Explore your data with custom SQL queries
3. **Event Processing**: Filter and process behavioral events for analysis
4. **Table Operations**: Modify database tables and update animal metadata
5. **Database Merging**: Combine data from multiple experiments
6. **CSV Operations**: Process and export data to CSV format

For more information and tutorials, visit the [LMT Toolkit documentation](#).
""")

st.markdown("© 2025 LMT Dimensionality Reduction Toolkit") 