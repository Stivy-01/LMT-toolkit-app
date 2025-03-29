import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# Basic database operations
def create_backup(source_db, backup_db):
    """Create database backup"""
    con = sqlite3.connect(source_db)
    bck = sqlite3.connect(backup_db)
    con.backup(bck)
    bck.close()
    con.close()

def get_db_connection(db_path):
    """Create database connection"""
    return sqlite3.connect(db_path)

def verify_table_structure(conn):
    """Verify created tables have correct structure"""
    tables_to_verify = ['BEHAVIOR_STATS', 'MULTI_MOUSE_EVENTS']

    for table in tables_to_verify:
        try:
            pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)
            print(f"✓ Table {table} exists and is accessible")
        except Exception as e:
            print(f"✗ Error accessing {table}: {str(e)}")

# Schema and column management
def get_table_columns(conn, table_name):
    """Get column names from a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def validate_schema(source_conn, target_conn, table_name):
    """Validate schema compatibility between source and target tables."""
    source_cols = set(get_table_columns(source_conn, table_name))
    target_cols = set(get_table_columns(target_conn, table_name))
    return {
        'match': source_cols == target_cols,
        'missing_in_target': source_cols - target_cols,
        'missing_in_source': target_cols - source_cols
    }

def add_missing_columns(conn, table_name, missing_columns):
    """Add missing columns to target table."""
    cursor = conn.cursor()
    for column in missing_columns:
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column}")
        except sqlite3.OperationalError:
            # Column might have been added by another process
            pass
    conn.commit()

# Merge metadata management
def setup_metadata_table(conn):
    """Create metadata tracking table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS merge_metadata (
            source_path TEXT,
            table_type TEXT,
            merge_date TIMESTAMP,
            PRIMARY KEY (source_path, table_type)
        )
    """)
    conn.commit()

def is_source_processed(conn, source_path, table_type):
    """Check if a source database has already been processed for given table type."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 1 FROM merge_metadata 
        WHERE source_path = ? AND table_type = ?
    """, (str(source_path), table_type))
    return cursor.fetchone() is not None

def record_merge(conn, source_path, table_type):
    """Record successful merge in metadata table."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO merge_metadata (source_path, table_type, merge_date)
        VALUES (?, ?, ?)
    """, (str(source_path), table_type, datetime.now().isoformat()))
    conn.commit()

# Table type mapping and merge operations
def get_table_mapping():
    """Get mapping of analysis types to their required tables."""
    return {
        1: ['behavior_hourly', 'group_events_hourly'],
        2: ['behavior_stats_intervals', 'multi_mouse_events_intervals'],  # Changed to match actual table names
        3: ['BEHAVIOR_STATS', 'MULTI_MOUSE_EVENTS']
    }

def merge_tables(target_conn, source_path, table_type, table_names_map):
    """Merge tables from source database into target database."""
    source_conn = sqlite3.connect(source_path)
    cursor = target_conn.cursor()
    
    try:
        # Get the required tables for this type
        required_tables_lower = {t.lower() for t in get_table_mapping()[table_type]}
        
        # Process each required table using its actual name in the source
        for table_lower in required_tables_lower:
            actual_table_name = table_names_map[table_lower]
            print(f"\nProcessing table: {actual_table_name}")
            
            # Validate schema
            schema_status = validate_schema(source_conn, target_conn, actual_table_name)
            if not schema_status['match']:
                # Add missing columns to target if any
                if schema_status['missing_in_target']:
                    add_missing_columns(target_conn, actual_table_name, schema_status['missing_in_target'])
            
            # Read data from source and target
            try:
                source_df = pd.read_sql_query(f'SELECT * FROM "{actual_table_name}"', source_conn)
                print(f"Read {len(source_df)} rows from source")
                
                try:
                    target_df = pd.read_sql_query(f'SELECT * FROM "{actual_table_name}"', target_conn)
                    print(f"Read {len(target_df)} rows from target")
                except:
                    target_df = pd.DataFrame()  # Empty DataFrame if table doesn't exist
                    print("No existing data in target")
                
                # Show data distribution before merge
                if 'interval_start' in source_df.columns and 'mouse_id' in source_df.columns:
                    print("\nSource data stats:")
                    print(f"Unique intervals: {source_df['interval_start'].nunique()}")
                    print(f"Unique mice: {source_df['mouse_id'].nunique()}")
                    if not target_df.empty:
                        print("\nTarget data stats:")
                        print(f"Unique intervals: {target_df['interval_start'].nunique()}")
                        print(f"Unique mice: {target_df['mouse_id'].nunique()}")
                
                # Combine data
                combined_df = pd.concat([target_df, source_df], ignore_index=True)
                print(f"\nCombined into {len(combined_df)} rows")
                
                # Remove exact duplicates
                combined_df = combined_df.drop_duplicates()
                print(f"After removing duplicates: {len(combined_df)} rows")
                
                # Show final stats
                if 'interval_start' in combined_df.columns and 'mouse_id' in combined_df.columns:
                    print("\nFinal data stats:")
                    print(f"Unique intervals: {combined_df['interval_start'].nunique()}")
                    print(f"Unique mice: {combined_df['mouse_id'].nunique()}")
                
                # Replace existing data with merged data
                cursor.execute(f'DROP TABLE IF EXISTS "{actual_table_name}"')
                combined_df.to_sql(actual_table_name, target_conn, if_exists='replace', index=False)
                
                # Verify the merge
                verification_df = pd.read_sql_query(f'SELECT * FROM "{actual_table_name}"', target_conn)
                print(f"\nVerification - rows in final table: {len(verification_df)}")
                
            except Exception as e:
                print(f"Error during merge for {actual_table_name}: {str(e)}")
                raise
        
        target_conn.commit()
        return True
        
    except sqlite3.Error as e:
        print(f"Error during merge: {str(e)}")
        target_conn.rollback()
        return False
    finally:
        source_conn.close() 