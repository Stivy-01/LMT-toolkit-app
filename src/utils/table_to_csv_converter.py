#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import sqlite3
import pandas as pd
from tkinter import filedialog, simpledialog, messagebox
import tkinter as tk

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.db_selector import get_db_path
from src.utils.database_utils import get_table_mapping

def get_analysis_type():
    """Get analysis type from user via GUI."""
    root = tk.Tk()
    root.withdraw()
    
    table_type = simpledialog.askinteger(
        "Analysis Type",
        "Choose analysis type (enter a number):\n\n" +
        "1: Hourly Analysis\n" +
        "   - behavior_hourly\n\n" +
        "2: Interval Analysis\n" +
        "   - behavior_stats_intervals\n\n" +
        "3: Daily Analysis\n" +
        "   - BEHAVIOR_STATS",
        minvalue=1, maxvalue=3
    )
    
    if not table_type:
        raise ValueError("No analysis type selected")
    
    # Map analysis type to table name
    table_mapping = {
        1: 'behavior_hourly',
        2: 'behavior_stats_intervals',
        3: 'BEHAVIOR_STATS'
    }
    
    return table_type, table_mapping[table_type]

def verify_table_exists(db_path, table_name):
    """Verify if the specified table exists in the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        
        exists = cursor.fetchone() is not None
        
        if not exists:
            print(f"Table '{table_name}' not found in database")
            
        return exists
    finally:
        conn.close()

def convert_table_to_csv(db_path, table_name):
    """Convert specified table to CSV and save in organized folder structure."""
    try:
        # Use existing data directory in LDA folder
        data_dir = project_root / 'data'
        if not data_dir.exists():
            print(f"\n❌ Error: Data directory not found at {data_dir}")
            print("Please ensure the 'data' directory exists in your LDA project folder.")
            return False
            
        # Create analysis subdirectory
        analysis_dir = data_dir / f"{table_name}_to_analize"
        analysis_dir.mkdir(exist_ok=True)
        
        # Generate output filename based on database name and date
        db_name = Path(db_path).stem
        csv_path = analysis_dir / f"{db_name}_{table_name}.csv"
        
        # Read data from database
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM [{table_name}]"
        df = pd.read_sql_query(query, conn)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Successfully created CSV at: {csv_path}")
        print(f"Number of rows exported: {len(df)}")
        print(f"File saved in: {analysis_dir}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error converting table to CSV: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    try:
        # Get analysis type and corresponding table name
        analysis_type, table_name = get_analysis_type()
        
        # Select source database
        print("\nSelect source database...")
        db_paths = get_db_path()
        if not db_paths:
            raise ValueError("No database selected")
        
        db_path = db_paths[0]  # Take first database if multiple selected
        
        # Verify table exists
        if not verify_table_exists(db_path, table_name):
            return
        
        # Convert table to CSV
        convert_table_to_csv(db_path, table_name)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 