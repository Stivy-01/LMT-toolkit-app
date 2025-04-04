import os
import sqlite3
import pandas as pd
import glob
import sys

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def get_available_databases():
    """Find all SQLite database files in the data directory and its subdirectories"""
    db_files = []
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(DATA_DIR):
        # Check each file for SQLite extensions
        for file in files:
            if file.endswith(('.db', '.sqlite', '.sqlite3')):
                # Get the full path
                full_path = os.path.join(root, file)
                # Get the relative path from DATA_DIR
                rel_path = os.path.relpath(full_path, DATA_DIR)
                db_files.append(rel_path)
    
    # Sort the files for better display
    db_files.sort()
    return db_files

def connect_to_database(db_name):
    """Connect to a database in the data directory or its subdirectories"""
    db_path = os.path.join(DATA_DIR, db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    return sqlite3.connect(db_path)

def get_tables(db_name):
    """Get list of tables in the specified database"""
    conn = connect_to_database(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    conn.close()
    return tables

def query_database(db_name, query):
    """Execute a query on the specified database and return results as DataFrame"""
    conn = connect_to_database(db_name)
    result = pd.read_sql(query, conn)
    conn.close()
    return result

def get_table_data(db_name, table_name, limit=1000):
    """Get data from a specific table in the database"""
    return query_database(db_name, f"SELECT * FROM {table_name} LIMIT {limit}")

def get_table_schema(db_name, table_name):
    """Get schema information for a specific table"""
    conn = connect_to_database(db_name)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    conn.close()
    
    # Convert to a more readable format
    columns = []
    for col in schema:
        columns.append({
            'cid': col[0],
            'name': col[1],
            'type': col[2],
            'notnull': col[3],
            'default_value': col[4],
            'pk': col[5]
        })
    
    return columns 