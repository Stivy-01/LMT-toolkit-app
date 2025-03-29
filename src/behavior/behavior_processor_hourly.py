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

        # Process each event
        for _, row in events_df.iterrows():
            event_time_str = row['event_start_datetime']
            try:
                event_time = datetime.strptime(event_time_str, '%Y-%m-%d %H:%M:%S')
            except:
                continue

            # Round event time to the start of the hour
            event_time = event_time.replace(minute=0, second=0)
            
            # Determine the first and last interval for hourly interval generation
            if 'first_interval' not in locals():
                first_interval = event_time
            last_interval = event_time
            if event_time > last_interval:
                last_interval = event_time

        # Generate hourly intervals from first to last event time
        current_interval = first_interval
        while current_interval <= last_interval:
            interval_str = current_interval.strftime('%Y-%m-%d %H:%M:%S')
            interval_strs.add(interval_str)
            current_interval += timedelta(hours=1)

        # Process each event again to count occurrences within each hourly interval
        for _, row in events_df.iterrows():
            event_time_str = row['event_start_datetime']
            try:
                event_time = datetime.strptime(event_time_str, '%Y-%m-%d %H:%M:%S')
            except:
                continue

            # Round event time to the start of the hour
            event_time = event_time.replace(minute=0, second=0)
            interval_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
            interval_strs.add(interval_str)  # Add the interval to the set

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
              #  behavior_durations[interval_str][behavior][f"{animal_a}_{animal_b}"]['total_duration'] += duration
                behavior_durations[interval_str][behavior][f"{animal_a}_{animal_b}"]['durations'].append(duration)
                pairwise_behaviors.add(behavior)
                all_behaviors.add(behavior)
            elif len(participants) == 1:
                mouse = participants[0]
                behavior_counts[interval_str][mouse][behavior]['count']['count'] += 1
              #  behavior_durations[interval_str][mouse][behavior]['total_duration'] += duration
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
                   # f"{self.sanitize(behavior + '_total_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_mean_duration')} REAL DEFAULT 0",
                   # f"{self.sanitize(behavior + '_median_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_std_duration')} REAL DEFAULT 0"
                ])
            else:
                behavior_columns.extend([
                    f"{self.sanitize(behavior + '_count')} INTEGER DEFAULT 0",
                   # f"{self.sanitize(behavior + '_total_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_mean_duration')} REAL DEFAULT 0",
                   # f"{self.sanitize(behavior + '_median_duration')} REAL DEFAULT 0",
                    f"{self.sanitize(behavior + '_std_duration')} REAL DEFAULT 0"
                ])
        behavior_columns.append("PRIMARY KEY (mouse_id, interval_start)")

        # Create group_events_hourly table
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
            self.conn.execute("DROP TABLE IF EXISTS behavior_stats_hourly")
            self.conn.execute(f"CREATE TABLE behavior_stats_hourly ({', '.join(behavior_columns)})")
            
            self.conn.execute("DROP TABLE IF EXISTS group_events_stats_hourly")
            self.conn.execute(f"CREATE TABLE group_events_stats_hourly ({', '.join(group_columns)})")

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
                        INSERT INTO behavior_stats_hourly 
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
        db_path = get_db_path()[0]
        processor = BehaviorProcessor(db_path)
        processor.process_events()

        print_flush("\n🔎 Verification:")
        verify_table_structure(processor.conn)

        print_flush("\nSample behavior_stats_hourly:")
        print_flush(pd.read_sql("SELECT * FROM behavior_stats_hourly LIMIT 5", processor.conn))

        print_flush("\nSample group_events_stats_hourly:")
        print_flush(pd.read_sql("SELECT * FROM group_events_stats_hourly LIMIT 5", processor.conn))

    except Exception as e:
        print_flush(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
