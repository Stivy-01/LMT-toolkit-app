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
import json
from collections import defaultdict
import pandas as pd
from src.utils.db_selector import get_db_path, get_date_from_filename
from src.utils.database_utils import get_db_connection, verify_table_structure

def print_flush(*args, **kwargs):
    """Print and flush immediately"""
    print(*args, **kwargs)
    sys.stdout.flush()

class BehaviorProcessor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.date = get_date_from_filename(db_path)
        self.conn = get_db_connection(db_path)
        
        # Get actual mouse IDs from ANIMAL table
        self.all_mice = self.get_animal_ids()
        
    def get_animal_ids(self):
        """Fetch actual mouse IDs from ANIMAL table"""
        query = "SELECT ID FROM ANIMAL"
        df = pd.read_sql(query, self.conn)
        return sorted(df['ID'].tolist())  
    
    def process_events(self):
        print(f"\n🔍 Processing: {Path(self.db_path).name}")
        # Get all events with necessary information
        query = """
        SELECT id, idanimalA, idanimalB, idanimalC, idanimalD, name, 
               startframe, endframe, event_start_datetime
        FROM EVENT_FILTERED
        """
        events_df = pd.read_sql(query, self.conn)
        print_flush(f"Loaded {len(events_df)} events")
        print(f"✅ Completed: {self.db_path}")
        # Initialize dictionaries to store behavior counts
        behavior_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        group_behavior_counts = defaultdict(lambda: defaultdict(int))

        print_flush("Processing events...")
        for _, row in events_df.iterrows():
            participants = [int(row[f'idanimal{letter}']) for letter in ['A', 'B', 'C', 'D'] 
                          if pd.notnull(row[f'idanimal{letter}'])]
            
            # Handle group events
            if len(participants) >= 3 or row['name'].startswith('Nest'):
                participants = sorted(participants)
                participants_key = json.dumps(participants)
                
                # Add to multi-mouse events table
                group_behavior_counts[participants_key][row['name']] += 1
                
                # SPECIAL HANDLING FOR GROUP3
                if len(participants) == 3:
                    # Find isolated mouse (the one not in participants)
                    isolated_mouse = [m for m in self.all_mice if m not in participants]
                    if isolated_mouse:
                        # Track isolation in individual stats
                        behavior_counts[isolated_mouse[0]]['isolated']['count'] += 1
            
            # Pairwise events
            elif len(participants) == 2:
                animal_a, animal_b = participants[0], participants[1]
                # Active count for initiator
                behavior_counts[int(animal_a)][row['name']]['active'] += 1
                # Passive count for receiver
                behavior_counts[int(animal_b)][row['name']]['passive'] += 1
            
            # Individual events
            elif len(participants) == 1:
                animal = participants[0]
                behavior_counts[int(animal)][row['name']]['count'] += 1

        # Process BEHAVIOR_STATS table
        print_flush("Creating behavior stats table...")
        
        # Get unique behaviors
        all_behaviors = set()
        pairwise_behaviors = set()
        individual_behaviors = set()
        
        for mouse_behaviors in behavior_counts.values():
            for behavior, counts in mouse_behaviors.items():
                if 'active' in counts:
                    pairwise_behaviors.add(behavior)
                else:
                    individual_behaviors.add(behavior)
                all_behaviors.add(behavior)

        # Create columns for the BEHAVIOR_STATS table
        behavior_columns = ["mouse_id INTEGER", "date TEXT"]
        for behavior in sorted(all_behaviors):
            if behavior in pairwise_behaviors:
                behavior_columns.extend([
                    f"{self.sanitize_column_name(behavior + '_active')} INTEGER DEFAULT 0",
                    f"{self.sanitize_column_name(behavior + '_passive')} INTEGER DEFAULT 0"
                ])
            else:
                behavior_columns.append(f"{self.sanitize_column_name(behavior)} INTEGER DEFAULT 0")
        behavior_columns.append("PRIMARY KEY (mouse_id, date)")

        # Get unique group behaviors
        group_behaviors = set()
        for behaviors in group_behavior_counts.values():
            group_behaviors.update(behaviors.keys())

        # Create columns for the MULTI_MOUSE_EVENTS table
        group_columns = ["participants TEXT", "date TEXT"]
        group_columns.extend([f"{self.sanitize_column_name(behavior)} INTEGER DEFAULT 0" 
                            for behavior in sorted(group_behaviors)])
        group_columns.append("PRIMARY KEY (participants, date)")

        # Create tables
        create_stats_table = f"""
            CREATE TABLE IF NOT EXISTS BEHAVIOR_STATS (
                {', '.join(behavior_columns)}
            )
        """
        create_group_table = f"""
            CREATE TABLE IF NOT EXISTS MULTI_MOUSE_EVENTS (
                {', '.join(group_columns)}
            )
        """
        
        print_flush("Creating tables with columns:")
        print_flush("BEHAVIOR_STATS columns:")
        print_flush(behavior_columns)
        print_flush("\nMULTI_MOUSE_EVENTS columns:")
        print_flush(group_columns)
        
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS BEHAVIOR_STATS")
            self.conn.execute(create_stats_table)
            
            self.conn.execute("DROP TABLE IF EXISTS MULTI_MOUSE_EVENTS")
            self.conn.execute(create_group_table)

        # Insert behavior stats with transaction
        print_flush("Inserting behavior statistics...")
        with self.conn:  # Transaction block
            for mouse_id in self.all_mice:
                values = [mouse_id, self.date]
                for behavior in sorted(all_behaviors):
                    if behavior in pairwise_behaviors:
                        values.extend([
                            behavior_counts[mouse_id][behavior].get('active', 0),
                            behavior_counts[mouse_id][behavior].get('passive', 0)
                        ])
                    else:
                        values.append(behavior_counts[mouse_id][behavior].get('count', 0))
                
                placeholders = ','.join(['?' for _ in range(len(values))])
                columns = ['mouse_id', 'date']
                for behavior in sorted(all_behaviors):
                    if behavior in pairwise_behaviors:
                        columns.extend([
                            self.sanitize_column_name(behavior + '_active'),
                            self.sanitize_column_name(behavior + '_passive')
                        ])
                    else:
                        columns.append(self.sanitize_column_name(behavior))
                
                insert_query = f"""
                    INSERT INTO BEHAVIOR_STATS 
                    ({','.join(columns)})
                    VALUES ({placeholders})
                """
                self.conn.execute(insert_query, values)

        # Insert group events with transaction
        print_flush("Inserting group events statistics...")
        with self.conn:  # Transaction block
            for participants_key, behaviors in group_behavior_counts.items():
                values = [participants_key, self.date]
                for behavior in sorted(group_behaviors):
                    values.append(behaviors.get(behavior, 0))
                
                placeholders = ','.join(['?' for _ in range(len(values))])
                columns = ['participants', 'date']
                columns.extend([self.sanitize_column_name(b) for b in sorted(group_behaviors)])
                
                insert_query = f"""
                    INSERT INTO MULTI_MOUSE_EVENTS 
                    ({','.join(columns)})
                    VALUES ({placeholders})
                """
                self.conn.execute(insert_query, values)

        print_flush("✅ Processing complete!")

    def sanitize_column_name(self, name):
        """Convert behavior name to valid SQLite column name"""
        sanitized = name.replace(' ', '_')
        sanitized = sanitized.replace('-', '_')
        sanitized = sanitized.replace(',', '')
        sanitized = sanitized.replace('(', '')
        sanitized = sanitized.replace(')', '')
        if not sanitized[0].isalpha():
            sanitized = 'behavior_' + sanitized
        return sanitized

def main():
    try:
        db_path = get_db_path()[0]
        processor = BehaviorProcessor(db_path)
        processor.process_events()

        print_flush("\n🔎 Verification:")
        verify_table_structure(processor.conn)

        print_flush("\nSample BEHAVIOR_STATS:")
        stats = pd.read_sql("SELECT * FROM BEHAVIOR_STATS LIMIT 5", processor.conn)
        print_flush(stats)
        print_flush("\nBEHAVIOR_STATS columns:")
        print_flush(stats.columns.tolist())

        print_flush("\nSample MULTI_MOUSE_EVENTS:")
        print_flush(pd.read_sql("SELECT * FROM MULTI_MOUSE_EVENTS LIMIT 5", processor.conn))

    except Exception as e:
        print_flush(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()