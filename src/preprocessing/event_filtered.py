# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import sqlite3
import tkinter as tk
from tkinter import messagebox, filedialog
from tkcalendar import Calendar

# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.db_selector import get_db_path, get_experiment_time

EXCLUDED_BEHAVIORS = [
    'Detection', 'Head detected', 'Look down', 'MACHINE LEARNING ASSOCIATION',
    'RFID ASSIGN ANONYMOUS TRACK', 'RFID MATCH', 'RFID MISMATCH', 'Water Stop', 'Water Zone'
]

def create_event_filtered_table(cursor):
    """Create the EVENT_FILTERED table"""
    cursor.execute("DROP TABLE IF EXISTS EVENT_FILTERED;")
    cursor.execute("""
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
    """)

def insert_merged_events(cursor):
    """Insert merged events into EVENT_FILTERED table"""
    excluded = ", ".join(f"'{b}'" for b in EXCLUDED_BEHAVIORS)
    
    cursor.execute(f"""
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
    """)

def update_timestamps(cursor, exp_start, fps=30.0):
    """Update timestamps in EVENT_FILTERED table"""
    cursor.execute(f"""
    UPDATE EVENT_FILTERED
    SET
        duration = endframe - startframe,
        duration_seconds = ROUND((endframe - startframe) / {fps}, 2),
        event_start_datetime = datetime(
            '{exp_start.isoformat()}', 
            '+' || CAST(ROUND(startframe / {fps}) AS INTEGER) || ' seconds'
        );
    """)

def main():
    conn = None
    try:
        db_path = get_db_path()[0]
        exp_start = get_experiment_time()
        
        print(f"🔬 Processing: {Path(db_path).name}")
        print(f"⏰ Experiment Start: {exp_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚫 Excluding behaviors: {', '.join(EXCLUDED_BEHAVIORS)}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        create_event_filtered_table(cursor)
        insert_merged_events(cursor)
        update_timestamps(cursor, exp_start)
        
        # Animal metadata columns
        for letter in ['A', 'B', 'C', 'D']:
            cursor.execute(f"""
            ALTER TABLE EVENT_FILTERED
            ADD COLUMN GENOTYPE_{letter} TEXT;
            """)
            cursor.execute(f"""
            UPDATE EVENT_FILTERED
            SET GENOTYPE_{letter} = (SELECT GENOTYPE FROM ANIMAL WHERE ANIMAL.ID = EVENT_FILTERED.idanimal{letter})
            WHERE idanimal{letter} IS NOT NULL;
            """)
            cursor.execute(f"""
            ALTER TABLE EVENT_FILTERED
            ADD COLUMN SETUP_{letter} TEXT;
            """)
            cursor.execute(f"""
            UPDATE EVENT_FILTERED
            SET SETUP_{letter} = (SELECT SETUP FROM ANIMAL WHERE ANIMAL.ID = EVENT_FILTERED.idanimal{letter})
            WHERE idanimal{letter} IS NOT NULL;
            """)
        
        conn.commit()
        print("✅ Processing completed successfully!")

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main() 