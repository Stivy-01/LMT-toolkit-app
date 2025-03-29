import os
import sqlite3
import threading
import pandas as pd

# Connection pool for thread-safety
_connection_pool = {}
_thread_local = threading.local()

def normalize_path(path):
    """
    Normalize a file path to handle different formats (Windows backslashes, forward slashes, etc.)
    
    Args:
        path (str): The file path to normalize
        
    Returns:
        str: Normalized file path
    """
    if path is None:
        return None
        
    # Replace backslashes with forward slashes
    path = path.replace('\\', '/')
    
    # Handle escaped backslashes
    path = path.replace('//', '/')
    
    # Normalize path separators
    path = os.path.normpath(path)
    
    return path

def validate_db_path(db_path):
    """
    Validate that a database file exists and is accessible
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message is a string
    """
    # Check if the path is None or empty
    if not db_path:
        return False, "Database path is empty"
    
    # Check if the file exists
    if not os.path.exists(db_path):
        return False, f"Database file not found: {db_path}"
    
    # Check if the file is readable
    if not os.access(db_path, os.R_OK):
        return False, f"Database file is not readable: {db_path}"
    
    # Check if it's a SQLite database file
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version();")
        version = cursor.fetchone()
        conn.close()
        
        if version:
            return True, f"Valid SQLite database (SQLite version: {version[0]})"
        else:
            return False, "File exists but does not appear to be a valid SQLite database"
    
    except sqlite3.Error as e:
        return False, f"SQLite error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error validating database: {str(e)}"

def get_db_connection(db_path):
    """
    Get a thread-safe database connection
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        sqlite3.Connection: Database connection object
    """
    # Normalize the path
    normalized_path = normalize_path(db_path)
    
    # Get current thread ID
    thread_id = threading.get_ident()
    
    # Use thread-specific connection
    thread_key = f"{normalized_path}_{thread_id}"
    if thread_key in _connection_pool:
        return _connection_pool[thread_key]
    
    # Create new connection for this thread
    conn = sqlite3.connect(normalized_path, check_same_thread=False)
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    # Configure connection to return rows as dictionaries
    conn.row_factory = sqlite3.Row
    
    # Store in connection pool
    _connection_pool[thread_key] = conn
    return conn

def check_lmt_database_structure(conn):
    """
    Check if the database has the expected LMT structure
    
    Args:
        conn (sqlite3.Connection): Database connection
        
    Returns:
        tuple: (is_valid, tables, message) where is_valid is a boolean, 
               tables is a list of available tables, and message is a description
    """
    cursor = conn.cursor()
    
    # Get all tables in the database (case-insensitive)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_raw = [row[0] for row in cursor.fetchall()]
    
    # Normalize table names to uppercase for case-insensitive comparison
    tables = [table.upper() for table in tables_raw]
    
    # Preserve original table names for display
    tables_display = tables_raw.copy()
    
    # Standard tables in LMT database
    expected_structure = ['ANIMAL', 'EVENT', 'DETECTION', 'FRAME', 'LOG', 'sqlite_sequence']
    
    # Check for minimum required tables (flexible for different naming conventions)
    has_animal = 'ANIMAL' in tables
    has_event = 'EVENT' in tables or 'EVENTS' in tables
    
    # Prepare messages about the database structure
    if has_animal and has_event:
        message = "Database contains the minimum required tables (ANIMAL and EVENT)."
        is_valid = True
    else:
        missing_tables = []
        if not has_animal:
            missing_tables.append("ANIMAL")
        if not has_event:
            missing_tables.append("EVENT/EVENTS")
        
        message = f"Database is missing required tables: {', '.join(missing_tables)}. " \
                 f"A valid LMT database should contain at least ANIMAL and EVENT tables."
        is_valid = False
    
    # List expected tables that were found
    found_expected = [table for table in expected_structure if table in tables]
    if found_expected:
        message += f"\nFound expected tables: {', '.join(found_expected)}"
    
    # List additional tables that weren't expected
    additional_tables = [table for table in tables_display if table.upper() not in [t.upper() for t in expected_structure]]
    if additional_tables:
        message += f"\nAdditional tables found: {', '.join(additional_tables)}"
    
    return is_valid, tables_display, message

def get_table_info(conn, table_name):
    """
    Get information about a table's structure
    
    Args:
        conn (sqlite3.Connection): Database connection
        table_name (str): Name of the table
        
    Returns:
        pandas.DataFrame: Table structure information
    """
    try:
        # Get column information
        query = f"PRAGMA table_info({table_name});"
        df = pd.read_sql_query(query, conn)
        
        # Rename columns for clarity
        df = df.rename(columns={
            'cid': 'Column ID',
            'name': 'Column Name',
            'type': 'Data Type',
            'notnull': 'Not Null',
            'dflt_value': 'Default Value',
            'pk': 'Primary Key'
        })
        
        # Convert boolean columns to Yes/No for better display
        df['Not Null'] = df['Not Null'].apply(lambda x: 'Yes' if x == 1 else 'No')
        df['Primary Key'] = df['Primary Key'].apply(lambda x: 'Yes' if x == 1 else 'No')
        
        return df
    
    except Exception as e:
        # Return an empty DataFrame with error message
        return pd.DataFrame({'Error': [f"Failed to get table info: {str(e)}"]})

def execute_query(conn, query):
    """
    Execute a SQL query and return the results
    
    Args:
        conn (sqlite3.Connection): Database connection
        query (str): SQL query to execute
        
    Returns:
        tuple: (result_df, message) where result_df is a pandas DataFrame and message is a string
    """
    try:
        # Check if the query is a SELECT query or other type (e.g., INSERT, UPDATE, DELETE)
        is_select = query.strip().upper().startswith('SELECT')
        
        if is_select:
            # Execute SELECT query and return results as DataFrame
            df = pd.read_sql_query(query, conn)
            rows = len(df)
            message = f"Query executed successfully. {rows:,} rows returned."
            return df, message
        else:
            # Execute non-SELECT query and return None with success message
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            rows_affected = cursor.rowcount
            message = f"Query executed successfully. {rows_affected:,} rows affected."
            return None, message
    
    except Exception as e:
        # Re-raise the exception to be caught by the calling function
        raise Exception(f"Error executing query: {str(e)}")

def close_all_connections():
    """
    Close all database connections in the connection pool
    """
    global _connection_pool
    
    for conn_key in list(_connection_pool.keys()):
        try:
            _connection_pool[conn_key].close()
        except:
            pass  # Ignore errors when closing connections
    
    _connection_pool = {} 