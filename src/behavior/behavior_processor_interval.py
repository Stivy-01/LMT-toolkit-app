# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from tkinter import messagebox
# Add the project root to the Python path
current_file = Path(os.path.abspath(__file__))
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import our modules
import json
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
from src.utils.db_selector import get_db_path
from src.utils.database_utils import get_db_connection, verify_table_structure
import numpy as np

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

class BehaviorProcessor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = get_db_connection(db_path)
        self.all_mice = self.get_animal_ids()
        
    def get_animal_ids(self):
        query = "SELECT ID FROM ANIMAL"
        df = pd.read_sql(query, self.conn)
        return sorted(df['ID'].tolist())
    
    def process_events(self):
        print(f"\n🔍 Processing: {Path(self.db_path).name}")
        query = """
        SELECT id, idanimalA, idanimalB, idanimalC, idanimalD, name, 
               startframe, endframe, event_start_datetime, duration_seconds
        FROM EVENT_FILTERED
        """
        events_df = pd.read_sql(query, self.conn)
        print_flush(f"Loaded {len(events_df)} events")

        # Initialize data structures
        behavior_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'count': 0}))))
        behavior_durations = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total_duration': 0.0, 'durations': []})))
        group_behavior_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'count': 0, 'total_duration': 0.0, 'durations': []})))
        interval_strs = set()
        all_behaviors = set()
        pairwise_behaviors = set()
        individual_behaviors = set()
        group_behaviors = set()

        # Process each event in the events dataframe
        # This code block processes behavioral events and categorizes them into night-time intervals (19:00-07:00)
        # For each event, it:
        # 1. Extracts and validates the event timestamp
        # 2. Filters for night-time events only (19:00-07:00)
        # 3. Determines the correct interval start time (always starting at 19:00)
        # 4. Identifies participating animals and the behavior type
        # 5. Stores the interval information for later processing
        #
        # Example: An event at 2023-05-20 21:30:00 will be assigned to interval starting at 2023-05-20 19:00:00
        #         An event at 2023-05-21 02:30:00 will be assigned to interval starting at 2023-05-20 19:00:00
        for _, row in events_df.iterrows():
            event_time_str = row['event_start_datetime']
            try:
                event_time = datetime.strptime(event_time_str, '%Y-%m-%d %H:%M:%S')
            except:
                continue
            
            hour = event_time.hour
            if not (hour >= 19 or hour < 7):
                continue  # Skip events outside 19:00-07:00 window

            # Determine interval start time
            if hour >= 19:
                interval_date = event_time.date()
            else:
                interval_date = (event_time - timedelta(days=1)).date()
            interval_start = datetime(interval_date.year, interval_date.month, interval_date.day, 19, 0)
            interval_str = interval_start.strftime('%Y-%m-%d %H:%M:%S')
            interval_strs.add(interval_str)

            participants = [int(row[f'idanimal{letter}']) for letter in ['A', 'B', 'C', 'D'] 
                          if pd.notnull(row[f'idanimal{letter}'])]
            behavior = row['name']
            duration = row['duration_seconds']

            # Handle group events
            if len(participants) >= 3 or behavior.startswith('Nest'):
                participants_key = json.dumps(sorted(participants))
                group_behavior_counts[interval_str][participants_key][behavior]['count'] += 1
               # group_behavior_counts[interval_str][participants_key][behavior]['total_duration'] += duration
                group_behavior_counts[interval_str][participants_key][behavior]['durations'].append(duration)
                group_behaviors.add(behavior)
                
                # Handle isolation for groups of 3
                if len(participants) == 3:
                    isolated = [m for m in self.all_mice if m not in participants]
                    if isolated:
                        behavior_counts[interval_str][isolated[0]]['isolated']['count']['count'] += 1
                       # behavior_durations[interval_str][isolated[0]]['isolated']['total_duration'] += duration
                        behavior_durations[interval_str][isolated[0]]['isolated']['durations'].append(duration)
                        all_behaviors.add('isolated')
            elif len(participants) == 2:
                animal_a, animal_b = participants[0], participants[1]
                # Active count for initiator
                behavior_counts[interval_str][animal_a][behavior]['active']['count'] += 1
                # Passive count for receiver
                behavior_counts[interval_str][animal_b][behavior]['passive']['count'] += 1
                # Store duration once per behavior instance
               # behavior_durations[interval_str][behavior][f"{animal_a}_{animal_b}"]['total_duration'] += duration
                behavior_durations[interval_str][behavior][f"{animal_a}_{animal_b}"]['durations'].append(duration)
                pairwise_behaviors.add(behavior)
                all_behaviors.add(behavior)
            elif len(participants) == 1:
                mouse = participants[0]
                behavior_counts[interval_str][mouse][behavior]['count']['count'] += 1
                #behavior_durations[interval_str][mouse][behavior]['total_duration'] += duration
                behavior_durations[interval_str][mouse][behavior]['durations'].append(duration)
                individual_behaviors.add(behavior)
                all_behaviors.add(behavior)

        # Create tables
        behavior_columns = ["mouse_id INTEGER", "interval_start TEXT"]
        for behavior in sorted(all_behaviors):
            if behavior in pairwise_behaviors:
                for suffix in ['active', 'passive']:
                    behavior_columns.append(f"{self.sanitize(behavior + '_' + suffix + '_count')} INTEGER DEFAULT 0")
                # Duration columns stored once per behavior
                behavior_columns.extend([
                 #   f"{self.sanitize(behavior + '_total_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_mean_duration')} REAL DEFAULT 0",
                  #  f"{self.sanitize(behavior + '_median_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_std_duration')} REAL DEFAULT 0"
                ])
            else:
                behavior_columns.extend([
                    f"{self.sanitize(behavior + '_count')} INTEGER DEFAULT 0",
                  #  f"{self.sanitize(behavior + '_total_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_mean_duration')} REAL DEFAULT 0",
                 #   f"{self.sanitize(behavior + '_median_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_std_duration')} REAL DEFAULT 0"
                ])
        behavior_columns.append("PRIMARY KEY (mouse_id, interval_start)")

        # Create multi_mouse_events_intervals table
        group_columns = ["participants TEXT", "interval_start TEXT"]
        for behavior in sorted(group_behaviors):
            group_columns.extend([
                f"{self.sanitize(behavior + '_count')} INTEGER DEFAULT 0",
               # f"{self.sanitize(behavior + '_total_duration')} REAL DEFAULT 0",
                f"{self.sanitize(behavior + '_mean_duration')} REAL DEFAULT 0",
               # f"{self.sanitize(behavior + '_median_duration')} REAL DEFAULT 0",
                f"{self.sanitize(behavior + '_std_duration')} REAL DEFAULT 0"
            ])
        group_columns.append("PRIMARY KEY (participants, interval_start)")

        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS behavior_stats_intervals")
            self.conn.execute(f"CREATE TABLE behavior_stats_intervals ({', '.join(behavior_columns)})")
            
            self.conn.execute("DROP TABLE IF EXISTS multi_mouse_events_intervals")
            self.conn.execute(f"CREATE TABLE multi_mouse_events_intervals ({', '.join(group_columns)})")

        # Insert behavior stats
        print_flush("Inserting behavior statistics...")
        with self.conn:
            for interval_str in sorted(interval_strs):
                for mouse_id in self.all_mice:
                    values = [
                        int(mouse_id),  # Ensure mouse_id is integer
                        str(interval_str)  # Ensure interval is string
                    ]
                    columns = ['mouse_id', 'interval_start']
                    
                    for behavior in sorted(all_behaviors):
                        if behavior in pairwise_behaviors:
                            # Add active and passive counts
                            for suffix in ['active', 'passive']:
                                count = behavior_counts[interval_str][mouse_id][behavior][suffix]['count']
                                values.append(int(count))  # Ensure count is integer
                                columns.append(self.sanitize(behavior + '_' + suffix + '_count'))
                            
                            # Add duration statistics (stored once per behavior)
                            behavior_dur = behavior_durations[interval_str][behavior]
                            # Collect all durations where this mouse was involved
                            mouse_durations = []
                            for pair, stats in behavior_dur.items():
                                if str(mouse_id) in pair:  # If mouse was part of this interaction
                                    mouse_durations.extend(stats['durations'])
                            
                            # Convert all duration stats to float
                            values.extend([
                                float(np.mean(mouse_durations)) if mouse_durations else 0.0,  # mean
                                float(np.std(mouse_durations)) if len(mouse_durations) > 1 else 0.0  # std
                            ])
                            columns.extend([
                                self.sanitize(behavior + '_mean_duration'),
                                self.sanitize(behavior + '_std_duration')
                            ])
                        else:
                            # Individual behaviors
                            count = behavior_counts[interval_str][mouse_id][behavior]['count']['count']
                            values.append(int(count))  # Ensure count is integer
                            columns.append(self.sanitize(behavior + '_count'))
                            
                            durations = behavior_durations[interval_str][mouse_id][behavior]['durations']
                            # Convert all duration stats to float
                            values.extend([
                                float(np.mean(durations)) if durations else 0.0,
                                float(np.std(durations)) if len(durations) > 1 else 0.0
                            ])
                            columns.extend([
                                self.sanitize(behavior + '_mean_duration'),
                                self.sanitize(behavior + '_std_duration')
                            ])
                    
                    placeholders = ','.join(['?' for _ in range(len(values))])
                    insert_query = f"""
                        INSERT INTO behavior_stats_intervals 
                        ({','.join(columns)})
                        VALUES ({placeholders})
                    """
                    self.conn.execute(insert_query, values)

    def sanitize(self, name):
        sanitized = name.replace(' ', '_').replace('-', '_').replace(',', '')
        sanitized = ''.join(c for c in sanitized if c not in '()')
        if not sanitized[0].isalpha():
            sanitized = 'b_' + sanitized
        return sanitized

def main():
    try:
        db_paths = get_db_path()  # Get all paths without [0]
        
        if not db_paths:
            print_flush("No files selected. Exiting...")
            return
            
        print_flush(f"\n📂 Processing {len(db_paths)} files...")
        
        for i, db_path in enumerate(db_paths, 1):
            print_flush(f"\n[{i}/{len(db_paths)}] Processing file: {Path(db_path).name}")
            processor = BehaviorProcessor(db_path)
            processor.process_events()

            print_flush("\n🔎 Verification:")
            verify_table_structure(processor.conn)

            print_flush("\nSample behavior_stats_intervals:")
            print_flush(pd.read_sql("SELECT * FROM behavior_stats_intervals LIMIT 5", processor.conn))

            print_flush("\nSample multi_mouse_events_intervals:")
            print_flush(pd.read_sql("SELECT * FROM multi_mouse_events_intervals LIMIT 5", processor.conn))
            
            # Close the connection after processing each file
            processor.conn.close()
            
        print_flush("\n✅ All files processed successfully!")

    except Exception as e:
        print_flush(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
