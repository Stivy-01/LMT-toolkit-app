# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import our modules
import pandas as pd
import sqlite3
from tkinter import filedialog, simpledialog, messagebox
import tkinter as tk
from src.utils.db_selector import get_db_path
from src.utils.database_utils import (
    get_db_connection,
    setup_metadata_table,
    is_source_processed,
    record_merge,
    merge_tables,
    get_table_mapping,
    validate_schema
)

DEFAULT_MERGED_DB = "merged_analysis.sqlite"

def conversion_to_csv(conn, db_path, table_name):
    """Convert specified table from database to CSV"""
    try:
        # Generate CSV path based on database path
        csv_path = os.path.splitext(db_path)[0] + ".csv"
        
        # Load data into pandas DataFrame
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Successfully created CSV at: {csv_path}")
        return True
    except Exception as e:
        print(f"❌ CSV conversion error: {str(e)}")
        return False

def get_columns(conn, db_path, table_name):
    """Fetch column names from a table in an attached database."""
    cursor = conn.cursor()
    columns = []
    try:
        cursor.execute("ATTACH DATABASE ? AS source_db", (db_path,))
        cursor.execute(f"PRAGMA source_db.table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Warning: Could not read columns from {db_path} - {str(e)}")
    finally:
        try:
            cursor.execute("DETACH DATABASE source_db")
        except:
            pass
    return columns

def verify_database(db_path, table_type):
    """Verify if a database contains the required tables for given analysis type.
    
    Returns:
        tuple: (is_valid, found_table_names)
            - is_valid: boolean indicating if all required tables were found
            - found_table_names: dict mapping lowercase table names to their actual names in the DB
    """
    try:
        conn = sqlite3.connect(db_path)
        required_tables = get_table_mapping()[table_type]
        
        # Get list of all tables in the database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\nChecking database: {Path(db_path).name}")
        print("Available tables:", ", ".join(existing_tables))
        print("Required tables:", ", ".join(required_tables))
        
        # Convert everything to lowercase for case-insensitive comparison
        existing_tables_lower = [t.lower() for t in existing_tables]
        required_tables_lower = set()  # Use set for faster lookup
        
        # Build set of required tables (lowercase)
        for table in required_tables:
            required_tables_lower.add(table.lower())
        
        # Check if all required tables exist (case-insensitive)
        missing_tables = []
        found_tables = {}  # Map of lowercase to actual table name
        
        # First, map existing tables
        for table in existing_tables:
            found_tables[table.lower()] = table
        
        # Then check for missing tables
        for table_lower in required_tables_lower:
            if table_lower not in found_tables:
                missing_tables.append(table_lower)
        
        if missing_tables:
            print(f"Missing tables: {', '.join(missing_tables)}")
            return False, {}
            
        # Return both status and found table names
        actual_tables = {t.lower(): found_tables[t.lower()] for t in required_tables_lower}
        return True, actual_tables
        
    except sqlite3.Error as e:
        print(f"Error checking database {Path(db_path).name}: {str(e)}")
        return False, {}
    finally:
        if 'conn' in locals():
            conn.close()

def create_target_tables(conn, table_type, source_db, table_names_map):
    """Create target tables with schema from source database."""
    source_conn = sqlite3.connect(source_db)
    cursor = conn.cursor()
    
    try:
        # Get the actual table names from the mapping
        required_tables_lower = {t.lower() for t in get_table_mapping()[table_type]}
        
        # Check which tables need to be created
        existing_tables = set()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cursor.fetchall():
            existing_tables.add(row[0].lower())
        
        for table_lower in required_tables_lower:
            actual_table_name = table_names_map[table_lower]
            
            # Only create table if it doesn't exist
            if actual_table_name.lower() not in existing_tables:
                # Get schema from source
                source_cursor = source_conn.cursor()
                source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (actual_table_name,))
                create_sql = source_cursor.fetchone()[0]
                
                # Create table in target using the same name as source
                print(f"Creating table: {actual_table_name}")
                conn.execute(create_sql)
            else:
                print(f"Table already exists: {actual_table_name}")
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error creating tables: {str(e)}")
        return False
    finally:
        source_conn.close()

def main():
    try:
        # Get analysis type from user
        root = tk.Tk()
        root.withdraw()
        
        table_type = simpledialog.askinteger(
            "Analysis Type",
            "Choose analysis type (enter a number):\n\n" +
            "1: Hourly Analysis\n" +
            "   - behavior_hourly\n" +
            "   - group_events_hourly\n\n" +
            "2: Interval Analysis\n" +
            "   - behavior_stats_intervals\n" +
            "   - MULTI_MOUSE_EVENTS\n\n" +
            "3: Daily Analysis\n" +
            "   - BEHAVIOR_STATS\n" +
            "   - MULTI_MOUSE_EVENTS",
            minvalue=1, maxvalue=3
        )
        
        if not table_type:
            raise ValueError("No analysis type selected")
        
        # Select source databases
        print("Select source databases or folder containing databases...")
        db_paths = get_db_path()
        if not db_paths:
            raise ValueError("No databases selected")
        
        # Verify all selected databases
        valid_dbs = []
        table_names_map = {}  # Store table names for each valid database
        for db_path in db_paths:
            is_valid, found_tables = verify_database(db_path, table_type)
            if is_valid:
                valid_dbs.append(db_path)
                table_names_map[db_path] = found_tables
            else:
                print(f"Skipping invalid database: {Path(db_path).name}")
        
        if not valid_dbs:
            raise ValueError("No valid databases selected")
        
        # Check for existing merged database or create new one
        default_path = Path(valid_dbs[0]).parent / DEFAULT_MERGED_DB
        if default_path.exists():
            use_existing = messagebox.askyesno(
                "Existing Database",
                f"Found existing merged database at {default_path}. Use it?"
            )
            if use_existing:
                merged_db_path = default_path
            else:
                merged_db_path = filedialog.asksaveasfilename(
                    title="Save Merged Database As",
                    defaultextension=".sqlite",
                    initialfile=DEFAULT_MERGED_DB,
                    filetypes=[("SQLite Databases", "*.sqlite"), ("All Files", "*.*")]
                )
        else:
            merged_db_path = str(default_path)
        
        if not merged_db_path:
            raise ValueError("No output location selected")
        
        # Connect to or create merged database
        conn = sqlite3.connect(merged_db_path)
        
        # Setup metadata tracking
        setup_metadata_table(conn)
        
        # Always ensure tables exist, whether new or existing database
        create_target_tables(conn, table_type, valid_dbs[0], table_names_map[valid_dbs[0]])
        
        # Process each source database
        processed_count = 0
        for db_path in valid_dbs:
            if is_source_processed(conn, db_path, table_type):
                print(f"Skipping already processed: {Path(db_path).name}")
                continue
                
            print(f"Processing: {Path(db_path).name}")
            if merge_tables(conn, db_path, table_type, table_names_map[db_path]):
                record_merge(conn, db_path, table_type)
                processed_count += 1
        
        conn.commit()
        print(f"\n✅ Successfully processed {processed_count} new databases")
        print(f"Merged database location: {merged_db_path}")
        
        # Show table statistics using actual table names from first database
        first_db_tables = table_names_map[valid_dbs[0]]
        print("\nTable Statistics:")
        for table_lower in first_db_tables:
            actual_table = first_db_tables[table_lower]
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM [{actual_table}]").fetchone()[0]
                print(f"Records in {actual_table}: {count}")
            except sqlite3.Error as e:
                print(f"Error getting count for {actual_table}: {str(e)}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()