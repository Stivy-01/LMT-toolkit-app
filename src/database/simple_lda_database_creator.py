import os
import sys
from pathlib import Path

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import sqlite3
import tkinter as tk
from tkinter import messagebox
from src.utils.db_selector import get_db_path

DEFAULT_MERGED_DB = "merged_analysis.sqlite"

def get_table_mapping():
    """Get mapping of analysis types to their required tables."""
    return {
        1: ['behavior_hourly', 'group_events_hourly'],
        2: ['behavior_stats_intervals', 'multi_mouse_events_intervals'],
        3: ['BEHAVIOR_STATS', 'MULTI_MOUSE_EVENTS']
    }

def find_table_in_db(conn, table_name):
    """Find a table in database regardless of case."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? COLLATE NOCASE", (table_name,))
    result = cursor.fetchone()
    return result[0] if result else None

def merge_databases(db_paths, table_names, output_db):
    """Merge specified tables from multiple databases into one."""
    print("\nStarting database merge...")
    
    # Create new database
    with sqlite3.connect(output_db) as target_conn:
        # Process each table
        for table_name in table_names:
            print(f"\nProcessing table: {table_name}")
            all_data = []
            
            # Collect data from each source database
            for db_path in db_paths:
                try:
                    with sqlite3.connect(db_path) as source_conn:
                        # Find actual table name in this database
                        actual_table = find_table_in_db(source_conn, table_name)
                        if not actual_table:
                            print(f"Table {table_name} not found in {Path(db_path).name}")
                            continue
                            
                        # Read the data
                        df = pd.read_sql_query(f'SELECT * FROM "{actual_table}"', source_conn)
                        print(f"Read {len(df)} rows from {Path(db_path).name}")
                        
                        # Print sample of data for debugging
                        if len(df) > 0:
                            print("\nSample data from this database:")
                            print(df[['mouse_id', 'interval_start']].head())
                            
                        all_data.append(df)
                        
                except Exception as e:
                    print(f"Error reading from {Path(db_path).name}: {str(e)}")
                    continue
            
            if not all_data:
                print(f"No data found for table {table_name}")
                continue
                
            # Combine all dataframes
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nBefore deduplication: {len(combined_df)} total rows")
            
            # For interval tables, we need to handle the merge carefully
            if 'interval_start' in combined_df.columns and 'mouse_id' in combined_df.columns:
                # Remove exact duplicates first
                combined_df = combined_df.drop_duplicates()
                print(f"After removing exact duplicates: {len(combined_df)} rows")
                
                # Show distribution of data
                print("\nRows per database:")
                for i, df in enumerate(all_data):
                    print(f"Database {i+1}: {len(df)} rows")
                    if len(df) > 0:
                        print("Intervals:", df['interval_start'].nunique())
                        print("Mice:", df['mouse_id'].nunique())
                
                print("\nFinal combined data:")
                print("Total rows:", len(combined_df))
                print("Unique intervals:", combined_df['interval_start'].nunique())
                print("Unique mice:", combined_df['mouse_id'].nunique())
                
                # Show sample of final data
                print("\nSample of final data:")
                print(combined_df[['mouse_id', 'interval_start']].head())
            
            # Save to new database
            combined_df.to_sql(table_name, target_conn, if_exists='replace', index=False)
            print(f"\nSaved table {table_name} to merged database")
            
            # Export to CSV
            csv_dir = project_root / 'src' / 'data'
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"merged_analysis_{table_name}.csv"
            combined_df.to_csv(csv_path, index=False)
            print(f"Exported to CSV: {csv_path}")
            
            # Verify the saved data
            verification_df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', target_conn)
            print(f"\nVerification - rows in saved table: {len(verification_df)}")

def main():
    try:
        # Create GUI window
        root = tk.Tk()
        root.title("LMT Analysis Type Selection")
        root.geometry("400x350")
        
        # Create frame
        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack(expand=True, fill='both')
        
        # Add label
        tk.Label(
            frame,
            text="Choose analysis type:",
            font=('Helvetica', 12, 'bold')
        ).pack(pady=(0, 20))
        
        # Variable to store selection
        selected_type = [0]
        
        def make_selection(choice):
            analysis_types = {
                1: "Hourly Analysis",
                2: "Interval Analysis (12-hour)",
                3: "Daily Analysis"
            }
            confirm = messagebox.askokcancel(
                "Confirm Analysis Type",
                f"You selected: {analysis_types[choice]}\n\n"
                "Click OK to proceed to database selection\n"
                "Click Cancel to change selection"
            )
            if confirm:
                selected_type[0] = choice
                root.quit()
        
        # Analysis type buttons
        tk.Button(
            frame,
            text="1: Hourly Analysis\n- behavior_hourly\n- group_events_hourly",
            command=lambda: make_selection(1),
            font=('Helvetica', 10),
            width=40,
            height=3,
            relief=tk.RAISED,
            bg='#E8E8E8'
        ).pack(pady=10)
        
        tk.Button(
            frame,
            text="2: Interval Analysis (12-hour)\n- behavior_stats_intervals\n- multi_mouse_events_intervals",
            command=lambda: make_selection(2),
            font=('Helvetica', 10),
            width=40,
            height=3,
            relief=tk.RAISED,
            bg='#E8E8E8'
        ).pack(pady=10)
        
        tk.Button(
            frame,
            text="3: Daily Analysis\n- BEHAVIOR_STATS\n- MULTI_MOUSE_EVENTS",
            command=lambda: make_selection(3),
            font=('Helvetica', 10),
            width=40,
            height=3,
            relief=tk.RAISED,
            bg='#E8E8E8'
        ).pack(pady=10)
        
        # Run window
        root.mainloop()
        
        # Get selection and cleanup
        analysis_type = selected_type[0]
        root.destroy()
        
        if not analysis_type:
            raise ValueError("No analysis type selected")
        
        # Get source databases
        print(f"\nSelected Analysis Type: {analysis_type}")
        print("Select source databases or folder containing databases...")
        db_paths = get_db_path()
        if not db_paths:
            raise ValueError("No databases selected")
        
        # Get required tables for selected analysis
        tables = get_table_mapping()[analysis_type]
        
        # Create output database path
        output_path = Path(db_paths[0]).parent / DEFAULT_MERGED_DB
        
        # Merge the databases
        merge_databases(db_paths, tables, output_path)
        
        print("\n✅ Process complete!")
        print(f"Merged database location: {output_path}")
        print("CSV files are in the src/data directory")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 