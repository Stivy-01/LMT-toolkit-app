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
        2: ['behavior_stats_intervals', 'MULTI_MOUSE_EVENTS',  # Original names
           'BEHAVIOR_STATS_INTERVALS', 'Multi_Mouse_Events'],  # Common variations
        3: ['BEHAVIOR_STATS', 'MULTI_MOUSE_EVENTS',
           'behavior_stats', 'Multi_Mouse_Events']
    }

def merge_tables(target_conn, source_path, table_type, table_names_map):
    """Merge tables from source database into target database.
    
    Args:
        target_conn: Connection to the target database
        source_path: Path to the source database
        table_type: Type of analysis (1, 2, or 3)
        table_names_map: Dictionary mapping lowercase table names to actual names in the source DB
    """
    source_conn = sqlite3.connect(source_path)
    cursor = target_conn.cursor()
    
    try:
        # Attach source database with a unique alias to avoid conflicts
        db_name = Path(source_path).stem.replace(" ", "_").replace("-", "_")
        safe_alias = f"source_{db_name}"
        
        # First detach if already attached (cleanup from previous errors)
        try:
            cursor.execute(f"DETACH DATABASE IF EXISTS {safe_alias}")
        except:
            pass
            
        # Attach the source database
        cursor.execute(f"ATTACH DATABASE ? AS {safe_alias}", (source_path,))
        
        # Get the required tables for this type
        required_tables_lower = {t.lower() for t in get_table_mapping()[table_type]}
        
        # Process each required table using its actual name in the source
        for table_lower in required_tables_lower:
            actual_table_name = table_names_map[table_lower]
            
            # Validate schema
            schema_status = validate_schema(source_conn, target_conn, actual_table_name)
            if not schema_status['match']:
                # Add missing columns to target if any
                if schema_status['missing_in_target']:
                    add_missing_columns(target_conn, actual_table_name, schema_status['missing_in_target'])
            
            # Get columns that exist in both source and target
            source_cols = set(get_table_columns(source_conn, actual_table_name))
            target_cols = set(get_table_columns(target_conn, actual_table_name))
            common_cols = sorted(list(source_cols & target_cols))
            
            if not common_cols:
                print(f"Warning: No common columns found for table {actual_table_name}")
                continue
                
            column_list = ', '.join(common_cols)
            
            # Perform merge using UPSERT pattern
            print(f"Merging data for table: {actual_table_name}")
            cursor.execute(f"""
                INSERT OR IGNORE INTO [{actual_table_name}] ({column_list})
                SELECT {column_list} 
                FROM {safe_alias}.[{actual_table_name}]
            """)
            
            rows_added = cursor.rowcount
            print(f"Added {rows_added} new rows to {actual_table_name}")
        
        target_conn.commit()
        return True
        
    except sqlite3.Error as e:
        print(f"Error during merge: {str(e)}")
        target_conn.rollback()
        return False
    finally:
        # Cleanup: detach source database and close connections
        try:
            cursor.execute(f"DETACH DATABASE IF EXISTS {safe_alias}")
        except:
            pass
        source_conn.close() 