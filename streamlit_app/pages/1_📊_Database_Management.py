import streamlit as st
import os
import sys
import pandas as pd
import sqlite3
import datetime
import tkinter as tk
from tkinter import filedialog
import io

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
Connect to your LMT database, explore its structure, and run SQL queries. 
This page supports both direct file path access and file uploads.
""")

# Display database connection status at the top if connected
if st.session_state.db_path:
    st.success(f"Connected to database: {st.session_state.db_path}")
    if st.session_state.valid_db:
        st.write("✅ Valid LMT database structure detected")
        st.write(f"Found tables: {', '.join(st.session_state.tables)}")
        
        # Display database statistics if available
        if st.session_state.db_stats:
            st.subheader("Database Statistics")
            for table, count in st.session_state.db_stats.items():
                st.write(f"Table `{table}`: {count:,} rows")
    else:
        st.warning("⚠️ Database structure does not match expected LMT format")
        if 'structure_message' in st.session_state:
            st.info(st.session_state.structure_message)
        st.write(f"Found tables: {', '.join(st.session_state.tables)}")

# Create tabs for different database connection methods
tab1, tab2, tab3, tab4 = st.tabs(["Connect via File Path", "Connect via File Upload", "Run SQL Queries", "Data Processing"])

# Tab 1: Connect via File Path
with tab1:
    st.header("Connect via File Path")
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
    st.expander("Path Format Help").markdown("""
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
        if db_path:
            # Normalize the path
            normalized_path = normalize_path(db_path)
            st.session_state.db_path = normalized_path
            
            # Validate the database path
            is_valid_path, path_message = validate_db_path(normalized_path)
            
            if is_valid_path:
                try:
                    # Attempt to connect to the database
                    conn = get_db_connection(normalized_path)
                    st.session_state.db_connection = conn
                    
                    # Check if it's a valid LMT database
                    is_valid_lmt, tables, structure_message = check_lmt_database_structure(conn)
                    st.session_state.valid_db = is_valid_lmt
                    st.session_state.tables = tables
                    st.session_state.structure_message = structure_message
                    
                    # Show success message with database information
                    st.success(f"Successfully connected to database: {normalized_path}")
                    
                    if is_valid_lmt:
                        st.write("✅ Valid LMT database structure detected")
                        st.write(f"Found tables: {', '.join(tables)}")
                    else:
                        st.warning("⚠️ Database structure does not match expected LMT format")
                        st.info(structure_message)
                        st.write(f"Found tables: {', '.join(tables)}")
                        
                    # Display database statistics and store in session state
                    st.subheader("Database Statistics")
                    db_stats = {}
                    for table in tables:
                        row_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0, 0]
                        st.write(f"Table `{table}`: {row_count:,} rows")
                        db_stats[table] = row_count
                    
                    # Store statistics in session state for persistence
                    st.session_state.db_stats = db_stats
                    
                except Exception as e:
                    st.error(f"Failed to connect to the database: {str(e)}")
                    st.session_state.db_connection = None
                    st.session_state.valid_db = False
            else:
                st.error(path_message)
        else:
            st.error("Please enter a database path")

# Tab 2: Connect via File Upload
with tab2:
    st.header("Connect via File Upload")
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
            st.session_state.db_connection = conn
            
            # Check if it's a valid LMT database
            is_valid_lmt, tables, structure_message = check_lmt_database_structure(conn)
            st.session_state.valid_db = is_valid_lmt
            st.session_state.tables = tables
            st.session_state.structure_message = structure_message
            
            # Show success message with database information
            st.success(f"Successfully connected to uploaded database")
            
            if is_valid_lmt:
                st.write("✅ Valid LMT database structure detected")
                st.write(f"Found tables: {', '.join(tables)}")
            else:
                st.warning("⚠️ Database structure does not match expected LMT format")
                st.info(structure_message)
                st.write(f"Found tables: {', '.join(tables)}")
                
            # Display database statistics and store in session state
            st.subheader("Database Statistics")
            db_stats = {}
            for table in tables:
                row_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0, 0]
                st.write(f"Table `{table}`: {row_count:,} rows")
                db_stats[table] = row_count
            
            # Store statistics in session state for persistence
            st.session_state.db_stats = db_stats
                
        except Exception as e:
            st.error(f"Failed to connect to the uploaded database: {str(e)}")
            st.session_state.db_connection = None
            st.session_state.valid_db = False
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

# Tab 3: Run SQL Queries
with tab3:
    st.header("Run SQL Queries")
    
    if st.session_state.db_connection is None:
        st.warning("Please connect to a database first (using the 'Connect via File Path' or 'Connect via File Upload' tab)")
    else:
        st.markdown("""
        Run SQL queries against your connected database. 
        You can explore the data and extract specific information.
        """)
        
        # Display table structure for reference
        if st.session_state.tables:
            with st.expander("Table Structure Reference"):
                for table in st.session_state.tables:
                    st.subheader(f"Table: {table}")
                    table_info = get_table_info(st.session_state.db_connection, table)
                    st.table(table_info)
        
        # SQL query input
        sql_query = st.text_area(
            "Enter your SQL query",
            height=150,
            help="Example: SELECT * FROM ANIMAL LIMIT 10"
        )
        
        if st.button("Run Query"):
            if sql_query:
                try:
                    # Execute the query and get the results
                    result_df, message = execute_query(st.session_state.db_connection, sql_query)
                    
                    if result_df is not None:
                        st.success(message)
                        st.dataframe(result_df)
                        
                        # Option to download the results as CSV
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info(message)
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
            else:
                st.error("Please enter a SQL query")

# Tab 4: Data Processing
with tab4:
    st.header("Data Processing")
    
    if st.session_state.db_connection is None:
        st.warning("Please connect to a database first (using the 'Connect via File Path' or 'Connect via File Upload' tab)")
    else:
        # Section 1: Update ANIMAL Table with Metadata
        st.subheader("1. Update ANIMAL Table Metadata")
        st.markdown("""
        Add metadata columns to ANIMAL table (SEX, AGE, GENOTYPE, SETUP) and provide an interface to edit them.
        """)
        
        # Button to add metadata columns to ANIMAL table
        add_animal_metadata_btn = st.button("Add Metadata Columns to ANIMAL Table")
        
        if add_animal_metadata_btn:
            try:
                cursor = st.session_state.db_connection.cursor()
                sql_statements = get_add_animal_metadata_sql()
                
                for sql in sql_statements:
                    try:
                        cursor.execute(sql)
                    except sqlite3.OperationalError as e:
                        # Ignore "duplicate column" errors
                        if "duplicate column" not in str(e):
                            raise e
                
                st.session_state.db_connection.commit()
                st.success("✅ Added metadata columns to ANIMAL table successfully!")
                
                # Update tables list
                if "ANIMAL" not in st.session_state.tables:
                    st.session_state.tables.append("ANIMAL")
                    
            except Exception as e:
                st.error(f"Error adding metadata columns: {str(e)}")
        
        # Display ANIMAL table for editing
        if "ANIMAL" in st.session_state.tables:
            with st.expander("Edit ANIMAL Metadata", expanded=True):
                try:
                    # Get ANIMAL data
                    animal_df = pd.read_sql("SELECT * FROM ANIMAL", st.session_state.db_connection)
                    
                    if len(animal_df) > 0:
                        # Display current animal data
                        st.subheader("Current Animal Data")
                        st.dataframe(animal_df)
                        
                        # Let user select an animal to edit
                        animal_ids = animal_df['ANIMAL'].tolist() if 'ANIMAL' in animal_df.columns else animal_df['ID'].tolist()
                        selected_animal = st.selectbox("Select animal to edit:", animal_ids)
                        
                        # Get ID column name
                        id_column = 'ANIMAL' if 'ANIMAL' in animal_df.columns else 'ID'
                        
                        # Create form for editing
                        with st.form("edit_animal_form"):
                            col1, col2 = st.columns(2)
                            
                            # Check if each column exists and get current value if it does
                            with col1:
                                sex_value = ""
                                if 'SEX' in animal_df.columns:
                                    current_sex = animal_df.loc[animal_df[id_column] == selected_animal, 'SEX'].values
                                    sex_value = current_sex[0] if len(current_sex) > 0 and pd.notna(current_sex[0]) else ""
                                sex = st.text_input("Sex:", value=sex_value)
                                
                                age_value = ""
                                if 'AGE' in animal_df.columns:
                                    current_age = animal_df.loc[animal_df[id_column] == selected_animal, 'AGE'].values
                                    age_value = current_age[0] if len(current_age) > 0 and pd.notna(current_age[0]) else ""
                                age = st.text_input("Age:", value=age_value)
                            
                            with col2:
                                genotype_value = ""
                                if 'GENOTYPE' in animal_df.columns:
                                    current_genotype = animal_df.loc[animal_df[id_column] == selected_animal, 'GENOTYPE'].values
                                    genotype_value = current_genotype[0] if len(current_genotype) > 0 and pd.notna(current_genotype[0]) else ""
                                genotype = st.text_input("Genotype:", value=genotype_value)
                                
                                setup_value = ""
                                if 'SETUP' in animal_df.columns:
                                    current_setup = animal_df.loc[animal_df[id_column] == selected_animal, 'SETUP'].values
                                    setup_value = current_setup[0] if len(current_setup) > 0 and pd.notna(current_setup[0]) else ""
                                setup = st.text_input("Setup:", value=setup_value)
                            
                            submit_button = st.form_submit_button("Update Animal Metadata")
                        
                        if submit_button:
                            try:
                                cursor = st.session_state.db_connection.cursor()
                                
                                # Update metadata in ANIMAL table
                                update_statement = f"""
                                UPDATE ANIMAL
                                SET SEX = ?,
                                    AGE = ?,
                                    GENOTYPE = ?,
                                    SETUP = ?
                                WHERE {id_column} = ?
                                """
                                
                                cursor.execute(update_statement, (sex, age, genotype, setup, selected_animal))
                                st.session_state.db_connection.commit()
                                st.success(f"✅ Updated metadata for Animal {selected_animal}!")
                                
                                # Refresh animal data
                                animal_df = pd.read_sql("SELECT * FROM ANIMAL", st.session_state.db_connection)
                                st.dataframe(animal_df)
                                
                            except Exception as e:
                                st.error(f"Error updating animal metadata: {str(e)}")
                    else:
                        st.warning("No animals found in the ANIMAL table.")
                    
                except Exception as e:
                    st.error(f"Error retrieving ANIMAL table data: {str(e)}")
        
        # Section 2: EVENT_FILTERED Creation and Event Processing
        st.divider()
        st.subheader("2. Process Events")
        st.markdown("""
        Create EVENT_FILTERED table with merged events. The system will:
        1. Unify events with the same name that are ≤30 frames (1 second) apart
        2. Filter out events with duration <6 frames (<0.2 seconds)
        3. Calculate durations and add timestamps
        """)
        
        # Experiment start time picker
        st.subheader("Experiment Start Time")
        st.markdown("Set the experiment start time for timestamp calculations")
        
        # Let user select date and time
        col1, col2 = st.columns(2)
        with col1:
            exp_date = st.date_input("Experiment Date", value=datetime.date.today())
        with col2:
            exp_time = st.time_input("Experiment Time", value=datetime.time(9, 0))
        
        # Combine date and time
        exp_start = datetime.datetime.combine(exp_date, exp_time)
        st.write(f"Selected experiment start: {exp_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # FPS input
        fps = st.number_input("Frames Per Second (FPS)", value=30.0, min_value=1.0, step=0.1)
        
        # Display excluded behaviors
        with st.expander("Excluded Behaviors"):
            st.write("The following behaviors will be excluded from processing:")
            for behavior in EXCLUDED_BEHAVIORS:
                st.write(f"- {behavior}")
        
        # Single button to process events
        process_events_btn = st.button("Process Events", help="Create EVENT_FILTERED table and process events")
        
        if process_events_btn:
            try:
                cursor = st.session_state.db_connection.cursor()
                
                # Step 1: Create table
                st.write("📊 Creating EVENT_FILTERED table...")
                cursor.executescript(CREATE_EVENT_FILTERED_TABLE_SQL)
                
                # Step 2: Insert merged events
                st.write("📥 Inserting merged events...")
                insert_sql = get_insert_merged_events_sql()
                cursor.execute(insert_sql)
                event_count = cursor.rowcount
                
                # Step 3: Update timestamps
                st.write("⏱️ Updating timestamps...")
                update_sql = get_update_timestamps_sql(exp_start, fps)
                cursor.execute(update_sql)
                
                # Step 4: Add event metadata
                st.write("🧬 Adding event metadata...")
                for letter in ['A', 'B', 'C', 'D']:
                    sql_statements = get_event_metadata_sql(letter)
                    for sql in sql_statements:
                        try:
                            cursor.execute(sql)
                        except sqlite3.OperationalError as e:
                            # Ignore "duplicate column" errors
                            if "duplicate column" not in str(e):
                                raise e
                
                st.session_state.db_connection.commit()
                
                # Update table list if needed
                if "EVENT_FILTERED" not in st.session_state.tables:
                    st.session_state.tables.append("EVENT_FILTERED")
                
                st.success(f"""
                ✅ Event processing completed successfully!
                - Created EVENT_FILTERED table
                - Inserted {event_count} merged events
                - Updated timestamps based on experiment start: {exp_start.strftime('%Y-%m-%d %H:%M:%S')}
                - Added animal metadata columns
                
                Event processing logic applied:
                - Merged events of the same type separated by ≤30 frames (1 second)
                - Filtered out events with duration <6 frames (0.2 seconds)
                - Added timestamp calculations based on experiment start time
                """)
                
                # Show sample results
                sample_df = pd.read_sql("SELECT * FROM EVENT_FILTERED LIMIT 10", st.session_state.db_connection)
                st.subheader("Sample Results")
                st.dataframe(sample_df)
                
                # Show event statistics
                with st.expander("View EVENT_FILTERED Table Statistics", expanded=True):
                    try:
                        count_df = pd.read_sql("SELECT COUNT(*) as count FROM EVENT_FILTERED", st.session_state.db_connection)
                        event_count = count_df.iloc[0, 0]
                        st.write(f"Total events in EVENT_FILTERED: {event_count:,}")
                        
                        behavior_counts = pd.read_sql(
                            "SELECT name, COUNT(*) as count FROM EVENT_FILTERED GROUP BY name ORDER BY count DESC", 
                            st.session_state.db_connection
                        )
                        st.subheader("Behavior Counts")
                        st.dataframe(behavior_counts)
                        
                        # Bar chart of behavior counts
                        st.bar_chart(behavior_counts.set_index('name'))
                        
                    except Exception as e:
                        st.error(f"Error retrieving table statistics: {str(e)}")
                
            except Exception as e:
                st.error(f"Error during event processing: {str(e)}")

        # Section 3: Table Enhancement and Mouse ID Updates
        st.divider()
        st.subheader("3. Advanced Database Operations")
        
        col1, col2 = st.columns(2)
        
        # Column Management
        with col1:
            st.markdown("### Remove Columns")
            st.markdown("Select a table and specify which columns to keep. All other columns will be removed.")
            
            # Table selector for column management
            table_for_columns = st.selectbox(
                "Select table for column management:",
                st.session_state.tables,
                key="table_for_columns"
            )
            
            # Get columns for the selected table
            columns = []
            if table_for_columns:
                try:
                    table_info = get_table_info(st.session_state.db_connection, table_for_columns)
                    columns = [row['Column Name'] for row in table_info.to_dict('records')]
                    
                    # Multi-select for columns to keep
                    columns_to_keep = st.multiselect(
                        "Select columns to keep (all others will be removed):",
                        columns,
                        default=columns[:1]  # Default to keeping the first column (usually ID)
                    )
                    
                    if st.button("Remove Unselected Columns"):
                        if not columns_to_keep:
                            st.error("You must select at least one column to keep!")
                        else:
                            try:
                                # Call the remove_columns function
                                success, message = remove_columns(
                                    st.session_state.db_path, 
                                    table_for_columns, 
                                    columns_to_keep
                                )
                                
                                if success:
                                    st.success(message)
                                    # Refresh the table info
                                    table_info = get_table_info(st.session_state.db_connection, table_for_columns)
                                    st.dataframe(table_info)
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"Error removing columns: {str(e)}")
                except Exception as e:
                    st.error(f"Error retrieving table columns: {str(e)}")
        
        # Row Management
        with col2:
            st.markdown("### Delete Rows")
            st.markdown("Delete rows from a table based on specific ID values.")
            
            # Table selector for row deletion
            table_for_rows = st.selectbox(
                "Select table for row deletion:",
                st.session_state.tables,
                key="table_for_rows"
            )
            
            # Get ID columns for the selected table
            id_columns = []
            if table_for_rows:
                try:
                    table_info = get_table_info(st.session_state.db_connection, table_for_rows)
                    # Identify potential ID columns (look for ID in name)
                    id_columns = [row['Column Name'] for row in table_info.to_dict('records') 
                                 if 'id' in row['Column Name'].lower() or 'animal' in row['Column Name'].lower()]
                    
                    # If no ID columns found, show all columns
                    if not id_columns:
                        id_columns = [row['Column Name'] for row in table_info.to_dict('records')]
                    
                    # Select ID column
                    id_column = st.selectbox(
                        "Select ID column for filtering rows:",
                        id_columns,
                        key="id_column_for_deletion"
                    )
                    
                    # Input for ID value
                    id_value = st.text_input(
                        f"Enter {id_column} value to delete rows:",
                        key="id_value_for_deletion"
                    )
                    
                    # Add a preview option
                    if id_value and st.button("Preview Rows to Delete"):
                        try:
                            # Show preview of rows that will be deleted
                            preview_query = f"SELECT * FROM {table_for_rows} WHERE {id_column} = ?"
                            cursor = st.session_state.db_connection.cursor()
                            cursor.execute(preview_query, (id_value,))
                            preview_rows = cursor.fetchall()
                            
                            if preview_rows:
                                # Get column names
                                columns = [description[0] for description in cursor.description]
                                # Convert to dataframe
                                preview_df = pd.DataFrame(preview_rows, columns=columns)
                                st.write(f"Found {len(preview_rows)} rows that will be deleted:")
                                st.dataframe(preview_df)
                            else:
                                st.warning(f"No rows found with {id_column} = {id_value}")
                        except Exception as e:
                            st.error(f"Error previewing rows: {str(e)}")
                    
                    if id_value and st.button("Delete Rows", key="delete_rows_button"):
                        try:
                            # Call the delete_rows function
                            success, message = delete_rows(
                                st.session_state.db_path,
                                table_for_rows,
                                id_column,
                                id_value
                            )
                            
                            if success:
                                st.success(message)
                                # Show updated row count
                                count_query = f"SELECT COUNT(*) as count FROM {table_for_rows}"
                                count_result = pd.read_sql(count_query, st.session_state.db_connection)
                                st.write(f"Current row count in {table_for_rows}: {count_result.iloc[0, 0]}")
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"Error deleting rows: {str(e)}")
                except Exception as e:
                    st.error(f"Error retrieving table information: {str(e)}")
        
        # Section for Mouse ID Updates
        st.markdown("### Update Mouse IDs")
        st.markdown("""
        This tool allows you to update a mouse ID across all related tables in the database.
        It will update the ID in the ANIMAL table and all related tables with idanimal fields.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input for old mouse ID
            old_id = st.text_input(
                "Old Mouse ID:",
                help="Enter the current ID of the mouse that needs to be updated"
            )
        
        with col2:
            # Input for new mouse ID
            new_id = st.text_input(
                "New Mouse ID:",
                help="Enter the new ID to assign to this mouse"
            )
        
        # Dry run option
        dry_run = st.checkbox(
            "Dry Run (preview changes without modifying database)",
            value=True,
            help="Check this to see what would be updated without making actual changes"
        )
        
        if old_id and new_id and st.button("Update Mouse ID"):
            try:
                with st.spinner("Processing ID update..."):
                    # Check if IDs are valid integers
                    try:
                        old_id_int = int(old_id)
                        new_id_int = int(new_id)
                    except ValueError:
                        st.error("Mouse IDs must be integers")
                        st.stop()
                    
                    # Use StringIO to capture stdout
                    import io
                    from contextlib import redirect_stdout
                    
                    output = io.StringIO()
                    with redirect_stdout(output):
                        # Call the update_mouse_id function
                        update_mouse_id(st.session_state.db_path, old_id_int, new_id_int, dry_run)
                    
                    # Display the output
                    output_text = output.getvalue()
                    if dry_run:
                        st.info("Dry Run Results (no changes made to database):")
                    else:
                        st.success("ID Update Completed:")
                    
                    # Format the output as a code block
                    st.code(output_text)
                    
                    # If not a dry run, check the ANIMAL table
                    if not dry_run:
                        # Verify the change in the ANIMAL table
                        animal_query = "SELECT * FROM ANIMAL WHERE ID = ?"
                        new_animal = pd.read_sql(animal_query, st.session_state.db_connection, params=(new_id_int,))
                        
                        if not new_animal.empty:
                            st.write("Updated animal record:")
                            st.dataframe(new_animal)
                        else:
                            st.warning(f"Could not find animal with new ID {new_id_int} in ANIMAL table after update.")
                
            except Exception as e:
                st.error(f"Error updating mouse ID: {str(e)}")

        # Section for CSV File Operations
        st.divider()
        st.subheader("4. CSV File Operations")
        st.markdown("""
        This section allows you to manipulate CSV files directly without importing them into a database.
        You can remove columns and delete rows from CSV files and then download the modified files.
        """)

        # Initialize session state for CSV operations
        if 'csv_df' not in st.session_state:
            st.session_state.csv_df = None
        if 'csv_filename' not in st.session_state:
            st.session_state.csv_filename = None
        if 'csv_modified' not in st.session_state:
            st.session_state.csv_modified = False

        # CSV file upload
        uploaded_csv = st.file_uploader("Upload a CSV file", type="csv", key="csv_uploader")
        
        if uploaded_csv is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_csv)
                
                # Store the DataFrame in session state
                st.session_state.csv_df = df
                st.session_state.csv_filename = uploaded_csv.name
                st.session_state.csv_modified = False
                
                # Show success message
                st.success(f"Successfully loaded CSV: {uploaded_csv.name}")
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Preview the data
                with st.expander("Preview data", expanded=True):
                    st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
        
        # If a CSV file is loaded, show the operations
        if st.session_state.csv_df is not None:
            st.subheader(f"Operations for {st.session_state.csv_filename}")
            
            # Create tabs for different operations
            csv_tab1, csv_tab2 = st.tabs(["Remove Columns", "Delete Rows"])
            
            # Tab 1: Remove Columns
            with csv_tab1:
                st.markdown("### Remove Columns")
                st.markdown("Select columns to keep. All other columns will be removed.")
                
                # Get columns from the DataFrame
                columns = list(st.session_state.csv_df.columns)
                
                # Multi-select for columns to keep
                columns_to_keep = st.multiselect(
                    "Select columns to keep (all others will be removed):",
                    columns,
                    default=columns[:1]  # Default to keeping the first column
                )
                
                if st.button("Remove Unselected Columns", key="csv_remove_columns"):
                    if not columns_to_keep:
                        st.error("You must select at least one column to keep!")
                    else:
                        try:
                            # Call the remove_columns_csv function
                            success, new_df, message = remove_columns_csv(
                                st.session_state.csv_df, 
                                columns_to_keep
                            )
                            
                            if success:
                                # Update the DataFrame in session state
                                st.session_state.csv_df = new_df
                                st.session_state.csv_modified = True
                                st.success(message)
                                
                                # Show the updated DataFrame
                                st.dataframe(new_df.head(10))
                                st.info(f"New shape: {new_df.shape[0]} rows × {new_df.shape[1]} columns")
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"Error removing columns: {str(e)}")
            
            # Tab 2: Delete Rows
            with csv_tab2:
                st.markdown("### Delete Rows")
                st.markdown("Delete rows based on column value matches.")
                
                # Get columns from the DataFrame
                columns = list(st.session_state.csv_df.columns)
                
                # Select column for filtering
                filter_column = st.selectbox(
                    "Select column for filtering rows:",
                    columns,
                    key="csv_filter_column"
                )
                
                # Get unique values from the selected column
                unique_values = st.session_state.csv_df[filter_column].unique()
                if len(unique_values) > 100:
                    # Text input if too many unique values
                    filter_value = st.text_input(
                        f"Enter {filter_column} value to delete rows (many unique values):",
                        key="csv_filter_value_input"
                    )
                else:
                    # Dropdown if manageable number of unique values
                    filter_value = st.selectbox(
                        f"Select {filter_column} value to delete rows:",
                        unique_values,
                        key="csv_filter_value_select"
                    )
                
                # Add a preview option
                if filter_column and filter_value and st.button("Preview Rows to Delete", key="csv_preview_delete"):
                    try:
                        # Show preview of rows that will be deleted
                        preview_df = st.session_state.csv_df[st.session_state.csv_df[filter_column] == filter_value]
                        
                        if not preview_df.empty:
                            st.write(f"Found {len(preview_df)} rows that will be deleted:")
                            st.dataframe(preview_df.head(10))
                            if len(preview_df) > 10:
                                st.info(f"Showing first 10 of {len(preview_df)} matching rows")
                        else:
                            st.warning(f"No rows found with {filter_column} = {filter_value}")
                    except Exception as e:
                        st.error(f"Error previewing rows: {str(e)}")
                
                if filter_column and filter_value and st.button("Delete Rows", key="csv_delete_rows"):
                    try:
                        # Call the delete_rows_csv function
                        success, new_df, message = delete_rows_csv(
                            st.session_state.csv_df,
                            filter_column,
                            filter_value
                        )
                        
                        if success:
                            # Update the DataFrame in session state
                            st.session_state.csv_df = new_df
                            st.session_state.csv_modified = True
                            st.success(message)
                            
                            # Show the updated DataFrame
                            st.dataframe(new_df.head(10))
                            st.info(f"New shape: {new_df.shape[0]} rows × {new_df.shape[1]} columns")
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Error deleting rows: {str(e)}")
            
            # Download the modified CSV
            if st.session_state.csv_modified:
                st.subheader("Download Modified CSV")
                
                # Generate a new filename
                basename = os.path.splitext(st.session_state.csv_filename)[0]
                new_filename = f"{basename}_modified.csv"
                
                # Convert the DataFrame to CSV
                csv_data = st.session_state.csv_df.to_csv(index=False)
                
                # Create a download button
                st.download_button(
                    label="Download Modified CSV",
                    data=csv_data,
                    file_name=new_filename,
                    mime="text/csv"
                )
        else:
            st.info("Upload a CSV file to perform operations on it")

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