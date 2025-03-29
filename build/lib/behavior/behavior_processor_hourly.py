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
import sqlite3
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
        self.failed_time_parses = 0  # Track parsing errors
        
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

        # Get comprehensive time range (full days)
        min_time, max_time = self.get_time_range(events_df)
        all_intervals = self.generate_full_coverage_intervals(min_time, max_time)

        # Initialize counting structures
        behavior_counts, group_counts = self.initialize_structures(all_intervals)

        # Process events with error tracking
        for _, row in events_df.iterrows():
            event_time = self.parse_event_time(row['event_start_datetime'])
            if not event_time:
                continue
            
            interval_start = event_time.replace(minute=0, second=0, microsecond=0)
            interval_str = interval_start.strftime('%Y-%m-%d %H:%M:%S')

            # Only count if interval exists in our comprehensive list
            if interval_str not in all_intervals:
                continue

            participants = self.get_valid_participants(row)
            behavior = row['name']

            self.classify_behavior(behavior_counts, group_counts, 
                                 participants, behavior, interval_str)

        # Create tables and insert data
        self.create_and_populate_tables(behavior_counts, group_counts, all_intervals)

        print_flush(f"\n⚠️ Failed time parses: {self.failed_time_parses}")
        print_flush("✅ Processing complete!")

    def get_time_range(self, events_df):
        """Get first and last event dates (full days)"""
        valid_times = []
        for t in events_df['event_start_datetime']:
            try:
                dt = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                valid_times.append(dt)
            except:
                self.failed_time_parses += 1
                
        if not valid_times:
            return datetime.now(), datetime.now()
            
        start = valid_times[0].replace(hour=0, minute=0, second=0)
        end = valid_times[-1].replace(hour=23, minute=59, second=59)
        return start, end

    def generate_full_coverage_intervals(self, start, end):
        """Generate all hours from start day 00:00 to end day 23:00"""
        intervals = []
        current = start.replace(minute=0, second=0, microsecond=0)
        end_limit = end.replace(hour=23, minute=0, second=0)
        
        while current <= end_limit:
            intervals.append(current.strftime('%Y-%m-%d %H:%M:%S'))
            current += timedelta(hours=1)
        return intervals

    def initialize_structures(self, intervals):
        """Create default entries for all mice in all intervals"""
        behavior = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(int)
                )
            )
        )
        
        group = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(int)
            )
        )
        
        # Pre-initialize all mice in all intervals
        for interval in intervals:
            for mouse in self.all_mice:
                behavior[interval][mouse]  # Touch to initialize
        return behavior, group

    def parse_event_time(self, time_str):
        """Parse with error handling and logging"""
        try:
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            self.failed_time_parses += 1
            return None

    def get_valid_participants(self, row):
        """Extract participants with validation"""
        participants = []
        for letter in ['A', 'B', 'C', 'D']:
            animal_id = row[f'idanimal{letter}']
            if pd.notnull(animal_id) and animal_id in self.all_mice:
                participants.append(int(animal_id))
        return participants

    def classify_behavior(self, behavior_counts, group_counts, 
                        participants, behavior, interval_str):
        """Classify behaviors with enhanced validation"""
        if len(participants) >= 3 or behavior.startswith('Nest'):
            participants_key = json.dumps(sorted(participants))
            group_counts[interval_str][participants_key][behavior] += 1
        elif len(participants) == 2:
            a, b = participants
            behavior_counts[interval_str][a][behavior]['active'] += 1
            behavior_counts[interval_str][b][behavior]['passive'] += 1
        elif len(participants) == 1:
            mouse = participants[0]
            behavior_counts[interval_str][mouse][behavior]['count'] += 1

    def create_and_populate_tables(self, behavior_counts, group_counts, intervals):
        """Handle table creation and data insertion"""
        # Discover all behaviors
        all_behaviors = set()
        pairwise = set()
        for interval in behavior_counts.values():
            for mouse in interval.values():
                for behavior, counts in mouse.items():
                    if 'active' in counts or 'passive' in counts:
                        pairwise.add(behavior)
                    all_behaviors.add(behavior)

        group_behaviors = set()
        for interval in group_counts.values():
            for participants in interval.values():
                group_behaviors.update(participants.keys())

        # Create tables
        with self.conn:
            self.create_behavior_table(sorted(all_behaviors), sorted(pairwise))
            self.create_group_table(sorted(group_behaviors))
            
            # Insert behavior data
            self.insert_behavior_data(behavior_counts, intervals, sorted(all_behaviors), sorted(pairwise))
            self.insert_group_data(group_counts, intervals, sorted(group_behaviors))

    def create_behavior_table(self, behaviors, pairwise):
        columns = ["mouse_id INTEGER", "interval_start TEXT"]
        for b in behaviors:
            if b in pairwise:
                columns += [f"{self.sanitize(b)}_active INTEGER DEFAULT 0",
                          f"{self.sanitize(b)}_passive INTEGER DEFAULT 0"]
            else:
                columns.append(f"{self.sanitize(b)} INTEGER DEFAULT 0")
        columns.append("PRIMARY KEY (mouse_id, interval_start)")
        
        self.conn.execute("DROP TABLE IF EXISTS behavior_hourly")
        self.conn.execute(f"CREATE TABLE behavior_hourly ({', '.join(columns)})")

    def create_group_table(self, behaviors):
        columns = ["participants TEXT", "interval_start TEXT"] + \
                [f"{self.sanitize(b)} INTEGER DEFAULT 0" for b in behaviors] + \
                ["PRIMARY KEY (participants, interval_start)"]
                
        self.conn.execute("DROP TABLE IF EXISTS group_events_hourly")
        self.conn.execute(f"CREATE TABLE group_events_hourly ({', '.join(columns)})")

    def insert_behavior_data(self, counts, intervals, behaviors, pairwise):
        print_flush("Inserting behavior data...")
        for interval in intervals:
            for mouse in self.all_mice:
                row = [mouse, interval]
                for b in behaviors:
                    if b in pairwise:
                        row.append(counts[interval][mouse][b].get('active', 0))
                        row.append(counts[interval][mouse][b].get('passive', 0))
                    else:
                        row.append(counts[interval][mouse][b].get('count', 0))
                self.conn.execute(
                    "INSERT INTO behavior_hourly VALUES ({})".format(','.join(['?']*len(row))),
                    row
                )

    def insert_group_data(self, counts, intervals, behaviors):
        print_flush("Inserting group data...")
        for interval in intervals:
            for participants, data in counts[interval].items():
                row = [participants, interval] + [data.get(b, 0) for b in behaviors]
                self.conn.execute(
                    "INSERT INTO group_events_hourly VALUES ({})".format(','.join(['?']*len(row))),
                    row
                )

    def sanitize(self, name):
        clean = name.replace(' ', '_').replace('-', '_').replace(',', '')
        clean = ''.join(c for c in clean if c not in '()')
        if not clean[0].isalpha():
            clean = f"b_{clean}"
        return clean.lower()

def main():
    try:
        db_path = get_db_path()[0]
        processor = BehaviorProcessor(db_path)
        processor.process_events()

        print_flush("\n✅ Verification:")
        verify_table_structure(processor.conn)

        print_flush("\nSample behavior_hourly:")
        print_flush(pd.read_sql("SELECT * FROM behavior_hourly LIMIT 5", processor.conn))

        print_flush("\nSample group_events_hourly:")
        print_flush(pd.read_sql("SELECT * FROM group_events_hourly LIMIT 5", processor.conn))

    except Exception as e:
        print_flush(f"🚨 Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()