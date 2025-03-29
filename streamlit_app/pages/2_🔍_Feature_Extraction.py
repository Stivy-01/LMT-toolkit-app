import streamlit as st
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
# Path and import setup
streamlit_app_path = Path(__file__).parent.parent
project_path = streamlit_app_path.parent
sys.path.extend([str(streamlit_app_path), str(project_path)])

from utils.db_utils import get_db_connection
from utils.analysis_utils import extract_features_from_database
from src.behavior.behavior_processor import BehaviorProcessor as FullExperimentProcessor
from src.behavior.behavior_processor_hourly import BehaviorProcessor as HourlyProcessor
from src.behavior.behavior_processor_interval import BehaviorProcessor as IntervalProcessor

def display_behavior_stats(df):
    st.session_state.current_behavior_stats = df
    st.subheader("Behavior Statistics")
    st.metric("Total Behavior Records", len(df))
    
    tab1, tab2 = st.tabs(["Data Table", "Summary Statistics"])
    
    with tab1:
        st.dataframe(df)
        st.download_button("Download as CSV", df.to_csv(), f"{st.session_state.get('current_table_name', 'behavior_stats')}.csv")
    
    with tab2:
        if {'ANIMAL_ID', 'BEHAVIOR'}.issubset(df.columns):
            behavior_counts = df.groupby('BEHAVIOR').size().reset_index(name='COUNT')
            st.dataframe(behavior_counts.sort_values('COUNT', ascending=False))
            st.bar_chart(behavior_counts.set_index('BEHAVIOR').sort_values('COUNT', ascending=False).head(15))
            
            if df['ANIMAL_ID'].nunique() > 1:
                st.dataframe(df.groupby('ANIMAL_ID').size().reset_index(name='COUNT').sort_values('COUNT', ascending=False))
                
        elif 'INTERVAL' in df.columns:
            st.dataframe(df.groupby('INTERVAL').size().reset_index(name='COUNT').sort_values('INTERVAL'))
            
        elif 'mouse_id' in df.columns:
            behavior_cols = [c for c in df.columns if c not in ['mouse_id', 'interval_start', 'id', 'index']]
            if behavior_cols:
                summary = pd.DataFrame({
                    'Behavior': behavior_cols,
                    'Total': df[behavior_cols].sum().values,
                    'Mean': df[behavior_cols].mean().values,
                    'Std Dev': df[behavior_cols].std().values
                }).sort_values('Total', ascending=False)
                
                st.dataframe(summary)
                st.bar_chart(summary.set_index('Behavior')['Total'].head(15))
                
                if df['mouse_id'].nunique() > 1:
                    mouse_totals = df.groupby('mouse_id')[behavior_cols].sum()
                    mouse_totals['Total Behaviors'] = mouse_totals.sum(axis=1)
                    st.dataframe(mouse_totals.sort_values('Total Behaviors', ascending=False))
            else:
                st.info("No behavior columns found")
        else:
            st.info("No summary statistics available")

def display_feature_results(features_df):
    # Store the features data in session state to prevent it from disappearing
    # Initialize session state parameters
    st.session_state.update({
        'current_features': st.session_state.get('current_features', features_df),
        'heatmap_transform': st.session_state.get('heatmap_transform', "None"),
        'heatmap_clip': st.session_state.get('heatmap_clip', True),
        'heatmap_percentile': st.session_state.get('heatmap_percentile', (5.0, 95.0)),
        'heatmap_colormap': st.session_state.get('heatmap_colormap', "viridis")
    })
    
    display_df = st.session_state.current_features
    behavior_cols = [c for c in display_df.columns if c.startswith(('count_', 'avg_duration_'))]
    
    # Metrics display
    st.subheader("Feature Summary")
    metrics = [
        ("Animals", len(display_df)),
        ("Total Features", len(display_df.columns)),
        ("Behavioral Features", sum(1 for c in display_df.columns if c.startswith(('count_', 'avg_duration_'))))
    ]
    for i, (label, value) in enumerate(metrics):
        st.columns(len(metrics))[i].metric(label, value)
    
    # Tab navigation
    tab_titles = ["Features Table", "Metadata Summary", "Feature Statistics", "Multi-Behavior Plot"]
    active_tab = st.session_state.get('active_feature_tab', 0)
    
    cols = st.columns(len(tab_titles))
    for i, col in enumerate(cols):
        if col.button(tab_titles[i], disabled=(i == active_tab), key=f"tab_{i}"):
            st.session_state.active_feature_tab = i
            st.rerun()
    st.markdown("---")
    
    # Tab content
    if active_tab == 0:
        st.dataframe(display_df)
        st.download_button("Download as CSV", display_df.to_csv(), "extracted_features.csv")
        
    elif active_tab == 1:
        metadata_cols = ['animal_id', 'genotype', 'sex', 'strain', 'setup']
        for col in [c for c in metadata_cols if c in display_df]:
            st.bar_chart(display_df[col].value_counts())
            
    elif active_tab == 2:
        st.session_state.selected_feature = st.selectbox(
            "Select feature", 
            behavior_cols,
            index=0 if not st.session_state.get('selected_feature') else behavior_cols.index(st.session_state.selected_feature)
        )
        
        if st.session_state.selected_feature:
            st.write(f"**Statistics for {st.session_state.selected_feature}**")
            st.write(display_df[st.session_state.selected_feature].describe())
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            display_df[st.session_state.selected_feature].plot.hist(bins=30, ax=ax1)
            display_df.boxplot(column=st.session_state.selected_feature, ax=ax2)
            st.pyplot(fig)
            
    elif active_tab == 3:
        count_cols = [c for c in behavior_cols if c.startswith('count_')]
        if count_cols:
            behavior_type = st.radio("Select behavior type", ["Counts", "Durations"], horizontal=True)
            plot_cols = count_cols if behavior_type == "Counts" else [c for c in behavior_cols if c.startswith('avg_duration_')]
            
            if plot_cols:
                st.checkbox("Show charts", value=True, key="show_charts")
                if st.session_state.show_charts:
                    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(10, 3*len(plot_cols)))
                    for ax, col in zip(axes, plot_cols):
                        display_df[col].plot.bar(ax=ax, title=col)
                    st.pyplot(fig)
                
                with st.expander("Heatmap Settings"):
                    st.selectbox("Transform", ["None", "Log", "Z-score"], 
                        key="heatmap_transform",
                        on_change=lambda: st.session_state.update(heatmap_transform=st.session_state.heatmap_transform))
                    
                    st.checkbox("Clip outliers", 
                        key="heatmap_clip",
                        on_change=lambda: st.session_state.update(heatmap_clip=st.session_state.heatmap_clip))
                    
                    st.slider("Percentile", 0.0, 100.0, 
                        key="heatmap_percentile",
                        on_change=lambda: st.session_state.update(heatmap_percentile=st.session_state.heatmap_percentile))
                    
                    st.selectbox("Colormap", ["viridis", "plasma", "inferno"], 
                        key="heatmap_colormap",
                        on_change=lambda: st.session_state.update(heatmap_colormap=st.session_state.heatmap_colormap))
                
                plot_data = display_df[plot_cols].apply(
                    lambda x: np.log1p(x) if st.session_state.heatmap_transform == "Log" else x)
                st.dataframe(plot_data.describe())
                fig = px.imshow(plot_data, color_continuous_scale=st.session_state.heatmap_colormap)
                st.plotly_chart(fig)

# Page configuration
st.set_page_config(
    page_title="Feature Extraction - LMT Toolkit",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'db_path' not in st.session_state:
    st.session_state.db_path = None
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'valid_db' not in st.session_state:
    st.session_state.valid_db = False
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'features' not in st.session_state:
    st.session_state.features = None
if 'selected_feature' not in st.session_state:
    st.session_state.selected_feature = None
if 'current_features' not in st.session_state:
    st.session_state.current_features = None
if 'current_behavior_stats' not in st.session_state:
    st.session_state.current_behavior_stats = None
if 'extraction_completed' not in st.session_state:
    st.session_state.extraction_completed = False

st.title("🔍 Feature Extraction")
st.markdown("""
Extract behavioral features from the LMT database for analysis.
This page allows you to extract and preprocess features based on various filters.
""")

# Check if database is connected
if not st.session_state.db_connection:
    st.warning("Please connect to a database first in the Database Management page")
    st.stop()

# Display database status
st.success(f"Connected to database: {os.path.basename(st.session_state.db_path)}")

# Display available behavior stats tables (always visible)
conn = st.session_state.db_connection
cursor = conn.cursor()

# Define the stat tables to check (both uppercase and lowercase variants)
stat_tables = [
    "BEHAVIOR_STATS", 
    "behavior_stats_hourly", 
    "behavior_stats_intervals"
]
available_tables = []

# Check each table
for table in stat_tables:
    # Check for the table in any case (upper, lower, mixed)
    cursor.execute(f"""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND 
        (name='{table}' OR name='{table.upper()}' OR name='{table.lower()}')
    """)
    table_info = cursor.fetchone()
    if table_info:
        # Use the actual table name found
        actual_table_name = table_info[0]
        cursor.execute(f"SELECT COUNT(*) FROM {actual_table_name}")
        row_count = cursor.fetchone()[0]
        if row_count > 0:
            available_tables.append((actual_table_name, row_count))

# If tables exist with data, give option to view them
if available_tables:
    with st.expander("View Existing Behavior Statistics", expanded=False):
        st.info("Select a table to view existing behavior statistics:")
        
        table_options = [f"{table} ({rows} rows)" for table, rows in available_tables]
        selected_table_option = st.selectbox("Select statistics to view:", table_options)
        
        if selected_table_option:
            # Extract the table name from the selection
            selected_table = selected_table_option.split(" (")[0]
            
            if st.button(f"Load {selected_table}"):
                # Load the selected table
                behavior_stats_df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
                
                # Store the table name for later use in downloads
                st.session_state.current_table_name = selected_table
                
                st.session_state.features = behavior_stats_df
                st.session_state.current_behavior_stats = behavior_stats_df
                st.session_state.extraction_completed = True
                
                st.success(f"Loaded {len(behavior_stats_df)} rows from {selected_table}")
                st.session_state.active_behavior_tab = 0  # Reset to first tab
                st.rerun()

# Create columns for the main interface sections
col1, col2 = st.columns([1, 1])

# --- Feature Extraction Parameters ---
st.header("Feature Extraction Parameters")

# Select extraction method
st.subheader("Extraction Method")
extraction_method = st.radio(
    "Select feature extraction method:",
    ["Basic Features", "Full Experiment (All Time)", "Hourly Intervals", "Night Intervals (19:00-07:00)"]
)

# Create columns layout
col1, col2 = st.columns(2)

with col1:
    # Time window filter
    st.subheader("Time Window Filter")
    use_time_window = st.checkbox("Filter by time window", value=False)
    
    time_window = None
    if use_time_window:
        # Get the minimum and maximum datetime from the EVENT_FILTERED table
        try:
            min_time_query = "SELECT MIN(event_start_datetime) FROM EVENT_FILTERED"
            max_time_query = "SELECT MAX(event_start_datetime) FROM EVENT_FILTERED"
            
            min_datetime_str = pd.read_sql(min_time_query, st.session_state.db_connection).iloc[0, 0]
            max_datetime_str = pd.read_sql(max_time_query, st.session_state.db_connection).iloc[0, 0]
            
            # Convert to datetime objects
            if min_datetime_str and max_datetime_str:
                try:
                    min_datetime = datetime.datetime.strptime(min_datetime_str, '%Y-%m-%d %H:%M:%S')
                    max_datetime = datetime.datetime.strptime(max_datetime_str, '%Y-%m-%d %H:%M:%S')
                except:
                    # Fallback if parsing fails
                    min_datetime = datetime.datetime.now() - datetime.timedelta(days=1)
                    max_datetime = datetime.datetime.now()
            else:
                # Fallback if no data
                min_datetime = datetime.datetime.now() - datetime.timedelta(days=1)
                max_datetime = datetime.datetime.now()
                
            # Split into date and time parts for the input widgets
            min_date = min_datetime.date()
            min_time = min_datetime.time()
            max_date = max_datetime.date()
            max_time = max_datetime.time()
            
            # Date inputs
            st.write("Start Time:")
            start_date = st.date_input("Date", min_date, min_value=min_date, max_value=max_date, key="start_date")
            start_time = st.time_input("Time", min_time, key="start_time")
            
            st.write("End Time:")
            end_date = st.date_input("Date", max_date, min_value=min_date, max_value=max_date, key="end_date")
            end_time = st.time_input("Time", max_time, key="end_time")
            
            # Combine date and time
            start_datetime = datetime.datetime.combine(start_date, start_time)
            end_datetime = datetime.datetime.combine(end_date, end_time)
            
            # Store as strings in the format expected by the database
            start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_str = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            # Time window will now be a tuple of datetime strings
            time_window = (start_datetime_str, end_datetime_str)
            
            st.info(f"Selected time window: {start_datetime_str} to {end_datetime_str}")
            
        except Exception as e:
            st.error(f"Error retrieving time range: {str(e)}")
            st.warning("Using default time window")
            # Default to None, which will not apply time filtering
            time_window = None

with col2:
    # Animal filter
    st.subheader("Animal Filter")
    use_animal_filter = st.checkbox("Filter by animal", value=False)
    
    animal_ids = None
    if use_animal_filter:
        # Get all animal IDs from the database
        try:
            # Try first with 'animal' column
            animals_query = "SELECT animal FROM ANIMAL"
            animals_df = pd.read_sql(animals_query, st.session_state.db_connection)
            all_animals = animals_df['animal'].tolist()
        except:
            try:
                # If that fails, try with 'ID' column
                animals_query = "SELECT ID FROM ANIMAL"
                animals_df = pd.read_sql(animals_query, st.session_state.db_connection)
                all_animals = animals_df['ID'].tolist()
            except Exception as e:
                st.error(f"Error retrieving animals from database: {str(e)}")
                all_animals = []
        
        # Multi-select for animals
        if all_animals:
            selected_animals = st.multiselect(
                "Select animals",
                options=all_animals,
                default=all_animals[:min(5, len(all_animals))],
                key="animal_select"
            )
            
            if selected_animals:
                animal_ids = selected_animals
                st.info(f"Selected {len(selected_animals)} animals")
            else:
                st.warning("No animals selected. All animals will be included.")
                animal_ids = None  # Ensure filter is disabled
        else:
            st.warning("No animals found in database")

# --- Extract Features Button ---
st.header("Extract Features")

# Use a placeholder for displaying results to avoid rerunning extraction
results_placeholder = st.empty()

extract_button = st.button("Extract Features", type="primary")

if extract_button or st.session_state.get('extraction_completed', False):
    if extract_button:
        # Clear previous results when explicitly extracting again
        st.session_state.current_features = None
        st.session_state.current_behavior_stats = None

    with st.spinner("Extracting features from database..."):
        try:
            if extraction_method == "Basic Features":
                # Use the original feature extraction method
                features_df = extract_features_from_database(
                    st.session_state.db_connection,
                    time_window=time_window,
                    animal_ids=animal_ids
                )
                
                # Store features in session state
                st.session_state.features = features_df
                st.session_state.current_features = features_df
                st.session_state.extraction_completed = True

                # Display results
                st.success(f"Successfully extracted basic features for {len(features_df)} animals")
                display_feature_results(features_df)

            else:
                # Initialize processor based on selection
                if extraction_method == "Full Experiment (All Time)":
                    processor = FullExperimentProcessor(st.session_state.db_path)
                    behavior_table = "BEHAVIOR_STATS"
                    multi_mouse_table = "MULTI_MOUSE_EVENTS"
                elif extraction_method == "Hourly Intervals":
                    processor = HourlyProcessor(st.session_state.db_path)
                    behavior_table = "behavior_stats_hourly"
                    multi_mouse_table = "group_events_stats_hourly"
                elif extraction_method == "Night Intervals (19:00-07:00)":
                    processor = IntervalProcessor(st.session_state.db_path)
                    behavior_table = "behavior_stats_intervals"
                    multi_mouse_table = "multi_mouse_events_intervals"

                # Check table existence
                conn = st.session_state.db_connection
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT name 
                    FROM sqlite_master 
                    WHERE type='table' AND 
                    (name='{behavior_table}' OR 
                     name='{behavior_table.upper()}' OR 
                     name='{behavior_table.lower()}')
                """)
                table_exists = cursor.fetchone() is not None

                # Handle table creation/replacement
                if table_exists:
                    replace_data = st.checkbox(
                        f"The {behavior_table} table exists. Check to replace, uncheck to append.",
                        value=True
                    )
                    processor.should_drop_tables = replace_data
                    st.info(f"Will {'REPLACE' if replace_data else 'APPEND TO'} {behavior_table}")
                else:
                    processor.should_drop_tables = True
                    st.info(f"Creating new table: {behavior_table}")

                # Apply time window filter
                if use_time_window and time_window:
                    start_time_str, end_time_str = time_window
                    create_temp_table_sql = f"""
                        DROP TABLE IF EXISTS EVENT_FILTERED_WINDOW;
                        CREATE TEMPORARY TABLE EVENT_FILTERED_WINDOW AS
                        SELECT * FROM EVENT_FILTERED
                        WHERE event_start_datetime BETWEEN '{start_time_str}' AND '{end_time_str}'
                    """
                    try:
                        cursor.executescript(create_temp_table_sql)
                        conn.commit()
                        event_count = pd.read_sql("SELECT COUNT(*) FROM EVENT_FILTERED_WINDOW", conn).iloc[0, 0]
                        st.info(f"Filtered to {event_count} events in time window")
                    except Exception as e:
                        st.warning(f"Time filter failed: {str(e)}. Using all events.")

                # Connection wrapper for table name handling
                original_process_events = processor.process_events

                class ConnectionWrapper:
                    def __init__(self, original_conn, behavior_table, multi_mouse_table, should_drop):
                        self.conn = original_conn
                        self.behavior_table = behavior_table
                        self.multi_mouse_table = multi_mouse_table
                        self.should_drop_tables = should_drop

                    def execute(self, sql, *args, **kwargs):
                        # Modify CREATE/DROP statements
                        if sql.strip().upper().startswith(('CREATE TABLE', 'DROP TABLE')):
                            sql_lower = sql.lower()
                            if self.behavior_table.lower() in sql_lower or self.multi_mouse_table.lower() in sql_lower:
                                if 'DROP TABLE' in sql.upper() and not self.should_drop_tables:
                                    return None
                                if 'CREATE TABLE' in sql.upper() and 'IF NOT EXISTS' not in sql:
                                    sql = sql.replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS', 1)
                        return self.conn.execute(sql, *args, **kwargs)

                    def __getattr__(self, attr):
                        return getattr(self.conn, attr)
                        
                    def __enter__(self):
                        # Delegate the enter call to the underlying connection
                        self.conn.__enter__()
                        return self
                        
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        # Delegate the exit call to the underlying connection
                        return self.conn.__exit__(exc_type, exc_val, exc_tb)

                # Monkey patch process_events
                def custom_process_events():
                    original_conn = processor.conn
                    processor.conn = ConnectionWrapper(
                        original_conn,
                        behavior_table,
                        multi_mouse_table,
                        processor.should_drop_tables
                    )
                    try:
                        return original_process_events()
                    finally:
                        processor.conn = original_conn

                processor.process_events = custom_process_events

                # Execute processing
                processor.process_events()

                # Retrieve and store results
                behavior_df = pd.read_sql(f"SELECT * FROM {behavior_table}", conn)
                st.session_state.features = behavior_df
                st.session_state.current_behavior_stats = behavior_df
                st.session_state.extraction_completed = True

                # Cleanup and display
                if use_time_window and time_window:
                    cursor.execute("DROP TABLE IF EXISTS EVENT_FILTERED_WINDOW")
                    conn.commit()
                st.success(f"Processed {len(behavior_df)} records with {extraction_method}")
                display_behavior_stats(behavior_df)

        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")
            st.exception(e)
            st.session_state.extraction_completed = False
            raise

elif 'features' in st.session_state and st.session_state.features is not None:
    # Display previous results
    st.header("Previously Extracted Features")
    st.info("Using cached results. Click Extract to refresh.")
    
    with results_placeholder.container():
        if isinstance(st.session_state.features, pd.DataFrame):
            if 'BEHAVIOR' in st.session_state.features.columns:
                display_behavior_stats(st.session_state.features)
            else:
                display_feature_results(st.session_state.features)
        else:
            st.warning("Invalid cached data. Please re-extract.")

else:
    # Show next steps
    st.markdown("---")
    st.header("Next Steps")
    st.markdown("""
    1. Configure extraction parameters
    2. Click 'Extract Features'
    3. Analyze/download results
    """)