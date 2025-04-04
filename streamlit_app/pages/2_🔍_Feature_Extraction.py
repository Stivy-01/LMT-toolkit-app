import streamlit as st
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
import time
from datetime import timedelta

# Path and import setup
streamlit_app_path = Path(__file__).parent.parent
project_path = streamlit_app_path.parent
sys.path.extend([str(streamlit_app_path), str(project_path)])

from utils.db_utils import get_db_connection
from utils.analysis_utils import extract_features_from_database
from src.behavior.behavior_processor import BehaviorProcessor as FullExperimentProcessor
from src.behavior.behavior_processor_hourly import BehaviorProcessor as HourlyProcessor
from src.behavior.behavior_processor_interval import BehaviorProcessor as IntervalProcessor
from utils.db_direct_access import get_available_databases, get_tables, get_table_data, connect_to_database
from config import DATA_DIR, validate_data_directory

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
This page allows you to extract behavioral features from your LMT data.
The app will access databases directly from your configured data directory.
""")

# Check if data directory is valid
is_valid, message = validate_data_directory()
if not is_valid:
    st.error(message)
    st.error(f"Please check your data directory configuration in config.py: {DATA_DIR}")
    st.stop()

st.success(f"Using data directory: {DATA_DIR}")

# Get available databases
available_dbs = get_available_databases()
if not available_dbs:
    st.warning(f"No database files (.db, .sqlite, .sqlite3) found in {DATA_DIR}")
    st.info("Please add your database files to this directory")
    st.stop()

# Database selection
selected_db = st.selectbox("Select Database", available_dbs, format_func=lambda x: x)

# Create a database connection
try:
    conn = connect_to_database(selected_db)
    # Check if this is an EVENT_FILTERED database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    # Check if EVENT_FILTERED table exists
    if 'EVENT_FILTERED' not in tables:
        st.warning(f"The database {selected_db} does not contain an EVENT_FILTERED table.")
        st.info("You need a database with processed event data. Please use the Database Management page first to process your events.")
        st.stop()
        
    # Get animal IDs
    cursor.execute("SELECT DISTINCT Animal FROM EVENT_FILTERED ORDER BY Animal;")
    animal_ids = [str(row[0]) for row in cursor.fetchall()]
    
    if not animal_ids:
        st.error("No animals found in the EVENT_FILTERED table.")
        st.stop()
        
    # Parameter selection
    st.header("Feature Extraction Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select animals
        selected_animals = st.multiselect(
            "Select animals to include",
            animal_ids,
            default=animal_ids
        )
        
        # Time interval selection
        interval_options = ["1 hour", "2 hours", "4 hours", "6 hours", "12 hours", "24 hours", "Custom"]
        interval_selection = st.selectbox("Time interval for feature aggregation", interval_options)
        
        if interval_selection == "Custom":
            custom_hours = st.number_input("Custom interval hours", min_value=0.5, max_value=48.0, value=2.0, step=0.5)
            interval_hours = custom_hours
        else:
            interval_hours = float(interval_selection.split()[0])
    
    with col2:
        # Select behaviors to include
        cursor.execute("SELECT DISTINCT Name FROM EVENT_FILTERED ORDER BY Name;")
        all_behaviors = [row[0] for row in cursor.fetchall()]
        
        selected_behaviors = st.multiselect(
            "Select behaviors to include",
            all_behaviors,
            default=all_behaviors
        )
        
        # Advanced options
        min_duration = st.number_input(
            "Minimum event duration (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
            help="Events shorter than this will be excluded from feature calculation"
        )
    
    # Run feature extraction
    if st.button("Extract Features", type="primary"):
        if not selected_animals:
            st.error("Please select at least one animal")
            st.stop()
            
        if not selected_behaviors:
            st.error("Please select at least one behavior")
            st.stop()
            
        # Start the extraction process
        with st.spinner(f"Extracting features for {len(selected_animals)} animals with {interval_hours} hour intervals..."):
            start_time = time.time()
            
            try:
                # Query to get the time range
                cursor.execute("""
                    SELECT MIN(StartTime), MAX(EndTime) 
                    FROM EVENT_FILTERED 
                    WHERE Animal IN ({})
                """.format(','.join(['?' for _ in selected_animals])), selected_animals)
                
                time_range = cursor.fetchone()
                start_datetime = datetime.strptime(time_range[0], '%Y-%m-%d %H:%M:%S')
                end_datetime = datetime.strptime(time_range[1], '%Y-%m-%d %H:%M:%S')
                
                # Create time intervals
                interval_td = timedelta(hours=interval_hours)
                intervals = []
                current = start_datetime
                
                while current < end_datetime:
                    next_interval = current + interval_td
                    intervals.append((current, next_interval))
                    current = next_interval
                
                # Initialize results dataframe
                results = []
                
                # Process each animal and interval
                for animal_id in selected_animals:
                    for i, (interval_start, interval_end) in enumerate(intervals):
                        # Convert datetimes to strings for SQLite query
                        start_str = interval_start.strftime('%Y-%m-%d %H:%M:%S')
                        end_str = interval_end.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Get events for this animal in this interval
                        events_query = """
                            SELECT Name, Duration 
                            FROM EVENT_FILTERED 
                            WHERE Animal = ? 
                            AND Name IN ({})
                            AND Duration >= ?
                            AND StartTime >= ? 
                            AND StartTime < ?
                        """.format(','.join(['?' for _ in selected_behaviors]))
                        
                        cursor.execute(
                            events_query, 
                            [animal_id] + selected_behaviors + [min_duration, start_str, end_str]
                        )
                        
                        events = cursor.fetchall()
                        
                        # Calculate features
                        result_row = {
                            'mouse_id': animal_id,
                            'interval_start': interval_start.strftime('%Y-%m-%d %H:%M:%S'),
                            'interval_end': interval_end.strftime('%Y-%m-%d %H:%M:%S'),
                            'interval_number': i + 1
                        }
                        
                        # Get animal metadata if available
                        try:
                            cursor.execute("""
                                SELECT SEX, GENOTYPE, STRAIN, SETUP 
                                FROM ANIMAL 
                                WHERE ANIMAL = ? OR ID = ?
                            """, [animal_id, animal_id])
                            
                            metadata = cursor.fetchone()
                            if metadata:
                                result_row['sex'] = metadata[0]
                                result_row['genotype'] = metadata[1]
                                result_row['strain'] = metadata[2]
                                result_row['setup'] = metadata[3]
                        except:
                            # Metadata may not be available - that's ok
                            pass
                        
                        # Calculate behavior counts and durations
                        behavior_counts = {}
                        behavior_durations = {}
                        
                        for event in events:
                            behavior = event[0]
                            duration = event[1]
                            
                            # Update count
                            key = f'count_{behavior}'
                            behavior_counts[key] = behavior_counts.get(key, 0) + 1
                            
                            # Add to total duration
                            duration_key = f'total_duration_{behavior}'
                            behavior_durations[duration_key] = behavior_durations.get(duration_key, 0) + duration
                        
                        # Add counts and durations to result row
                        for behavior in selected_behaviors:
                            count_key = f'count_{behavior}'
                            result_row[count_key] = behavior_counts.get(count_key, 0)
                            
                            duration_key = f'total_duration_{behavior}'
                            result_row[duration_key] = behavior_durations.get(duration_key, 0)
                            
                            # Calculate average duration if count > 0
                            avg_key = f'avg_duration_{behavior}'
                            if result_row[count_key] > 0:
                                result_row[avg_key] = result_row[duration_key] / result_row[count_key]
                            else:
                                result_row[avg_key] = 0
                        
                        results.append(result_row)
                
                # Create features dataframe
                features_df = pd.DataFrame(results)
                
                # Store in session state
                st.session_state.features = features_df
                
                # Display success message
                elapsed_time = time.time() - start_time
                st.success(f"Feature extraction completed in {elapsed_time:.2f} seconds!")
                st.write(f"Extracted {len(features_df)} feature rows across {len(selected_animals)} animals and {len(intervals)} time intervals")
                
                # Display the features dataframe
                st.subheader("Extracted Features")
                st.dataframe(features_df)
                
                # Download option
                csv = features_df.to_csv(index=False)
                st.download_button(
                    label="Download Features as CSV",
                    data=csv,
                    file_name=f"extracted_features_{interval_hours}h.csv",
                    mime="text/csv"
                )
                
                # Show feature statistics
                st.subheader("Feature Statistics")
                
                # Basic statistics on counts and durations
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                count_cols = [col for col in numeric_cols if col.startswith('count_')]
                duration_cols = [col for col in numeric_cols if col.startswith('avg_duration_')]
                
                if count_cols:
                    st.write("**Behavior Counts (per interval)**")
                    count_stats = features_df[count_cols].describe().T
                    st.dataframe(count_stats)
                
                if duration_cols:
                    st.write("**Average Behavior Durations (seconds)**")
                    duration_stats = features_df[duration_cols].describe().T
                    st.dataframe(duration_stats)
                
            except Exception as e:
                st.error(f"Error during feature extraction: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    conn.close()
    
except Exception as e:
    st.error(f"Error connecting to database {selected_db}: {str(e)}")

# Alternative upload option
st.markdown("---")
st.header("Alternative: Use Existing Feature CSV")
st.info("You can also upload a previously generated features CSV file.")

uploaded_features = st.file_uploader("Upload Features CSV", type=["csv"])

if uploaded_features is not None:
    try:
        features_df = pd.read_csv(uploaded_features)
        st.success(f"Loaded features CSV with {len(features_df)} rows")
        
        # Store in session state
        st.session_state.features = features_df
        
        # Display the features
        st.dataframe(features_df)
        
    except Exception as e:
        st.error(f"Error loading features CSV: {str(e)}")

# Display footer
st.markdown("---")
st.markdown("© 2025 LMT Feature Extraction")