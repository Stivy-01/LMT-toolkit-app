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
               startframe, endframe, event_start_datetime
        FROM EVENT_FILTERED
        """
        events_df = pd.read_sql(query, self.conn)
        print_flush(f"Loaded {len(events_df)} events")

        # Initialize data structures
        behavior_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        group_behavior_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
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

            # Handle group events
            if len(participants) >= 3 or behavior.startswith('Nest'):
                participants_key = json.dumps(sorted(participants))
                group_behavior_counts[interval_str][participants_key][behavior] += 1
                group_behaviors.add(behavior)
                
                # Handle isolation for groups of 3
                if len(participants) == 3:
                    isolated = [m for m in self.all_mice if m not in participants]
                    if isolated:
                        behavior_counts[interval_str][isolated[0]]['isolated']['count'] += 1
                        all_behaviors.add('isolated')
            elif len(participants) == 2:
                a, b = participants
                behavior_counts[interval_str][a][behavior]['active'] += 1
                behavior_counts[interval_str][b][behavior]['passive'] += 1
                pairwise_behaviors.add(behavior)
                all_behaviors.add(behavior)
            elif len(participants) == 1:
                mouse = participants[0]
                behavior_counts[interval_str][mouse][behavior]['count'] += 1
                individual_behaviors.add(behavior)
                all_behaviors.add(behavior)

        # Determine pairwise and individual behaviors
        for interval_data in behavior_counts.values():
            for mouse_data in interval_data.values():
                for behavior in mouse_data.keys():
                    if behavior in pairwise_behaviors:
                        continue
                    # Check if this behavior has active/passive keys
                    if any(key in ('active', 'passive') for key in mouse_data[behavior].keys()):
                        pairwise_behaviors.add(behavior)
                        if behavior in individual_behaviors:
                            individual_behaviors.remove(behavior)

        all_behaviors = sorted(pairwise_behaviors.union(individual_behaviors))
        group_behaviors = sorted(group_behaviors)
        interval_strs = sorted(interval_strs)

        # Create tables
        self.create_tables(all_behaviors, pairwise_behaviors, group_behaviors)

        # Insert behavior stats
        self.insert_behavior_stats_intervals(all_behaviors, pairwise_behaviors, behavior_counts, interval_strs)

        # Insert group events
        self.insert_group_events(group_behaviors, group_behavior_counts, interval_strs)

        print_flush("✅ Processing complete!")

    def create_tables(self, all_behaviors, pairwise_behaviors, group_behaviors):
        # Create behavior_stats_intervals table
        behavior_columns = ["mouse_id INTEGER", "interval_start TEXT"]
        for behavior in all_behaviors:
            if behavior in pairwise_behaviors:
                behavior_columns.extend([
                    f"{self.sanitize(behavior + '_active')} INTEGER DEFAULT 0",
                    f"{self.sanitize(behavior + '_passive')} INTEGER DEFAULT 0"
                ])
            else:
                behavior_columns.append(f"{self.sanitize(behavior)} INTEGER DEFAULT 0")
        behavior_columns.append("PRIMARY KEY (mouse_id, interval_start)")

        # Create MULTI_MOUSE_EVENTS table
        group_columns = ["participants TEXT", "interval_start TEXT"]
        group_columns.extend([f"{self.sanitize(b)} INTEGER DEFAULT 0" for b in group_behaviors])
        group_columns.append("PRIMARY KEY (participants, interval_start)")

        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS behavior_stats_intervals")
            self.conn.execute(f"CREATE TABLE behavior_stats_intervals ({', '.join(behavior_columns)})")
            
            self.conn.execute("DROP TABLE IF EXISTS MULTI_MOUSE_EVENTS")
            self.conn.execute(f"CREATE TABLE MULTI_MOUSE_EVENTS ({', '.join(group_columns)})")

    def insert_behavior_stats_intervals(self, all_behaviors, pairwise_behaviors, behavior_counts, interval_strs):
        print_flush("Inserting behavior statistics...")
        with self.conn:
            for interval_str in interval_strs:
                for mouse in self.all_mice:
                    values = [mouse, interval_str]
                    for behavior in all_behaviors:
                        if behavior in pairwise_behaviors:
                            active = behavior_counts[interval_str][mouse][behavior].get('active', 0)
                            passive = behavior_counts[interval_str][mouse][behavior].get('passive', 0)
                            values.extend([active, passive])
                        else:
                            count = behavior_counts[interval_str][mouse][behavior].get('count', 0)
                            values.append(count)
                    
                    placeholders = ','.join(['?'] * len(values))
                    self.conn.execute(
                        f"INSERT INTO behavior_stats_intervals VALUES ({placeholders})", 
                        values
                    )

    def insert_group_events(self, group_behaviors, group_behavior_counts, interval_strs):
        print_flush("Inserting group events statistics...")
        with self.conn:
            for interval_str in interval_strs:
                for participants, behaviors in group_behavior_counts[interval_str].items():
                    values = [participants, interval_str]
                    values.extend([behaviors.get(b, 0) for b in group_behaviors])
                    placeholders = ','.join(['?'] * len(values))
                    self.conn.execute(
                        f"INSERT INTO MULTI_MOUSE_EVENTS VALUES ({placeholders})", 
                        values
                    )

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

        print_flush("\nSample behavior_stats_intervals:")
        print_flush(pd.read_sql("SELECT * FROM behavior_stats_intervals LIMIT 5", processor.conn))

        print_flush("\nSample MULTI_MOUSE_EVENTS:")
        print_flush(pd.read_sql("SELECT * FROM MULTI_MOUSE_EVENTS LIMIT 5", processor.conn))

    except Exception as e:
        print_flush(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
