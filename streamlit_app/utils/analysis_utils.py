import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def extract_features_from_database(conn, time_window=None, animal_ids=None):
    """
    Extract behavioral features from LMT database
    
    Args:
        conn (sqlite3.Connection): Database connection
        time_window (tuple): Optional (start_time, end_time) - can be seconds or datetime strings
        animal_ids (list): Optional list of animal IDs to include
        
    Returns:
        pandas.DataFrame: DataFrame containing extracted features
    """
    # Based on the user's database structure, we're constructing a query that works with their exact schema
    # The EVENT table has: ID, NAME, STARTFRAME, ENDFRAME, IDANIMALA, etc.
    # The ANIMAL table has: ID, NAME, GENOTYPE, SEX, STRAIN, etc.
    
    # Build a query specifically for this database structure
    query = """
    SELECT 
        e.IDANIMALA as animal_id,
        e.NAME as eventName,
        e.STARTFRAME as startFrame,
        e.ENDFRAME as endFrame,
        e.METADATA as metadata,
        a.GENOTYPE as genotype,
        a.SEX as sex,
        a.STRAIN as strain,
        a.SETUP as setup
    FROM EVENT e
    JOIN ANIMAL a ON e.IDANIMALA = a.ID
    """
    
    # Add filters if provided
    where_clauses = []
    
    if time_window:
        # Check if time_window contains datetime strings or seconds
        if isinstance(time_window[0], str) and isinstance(time_window[1], str):
            # For datetime strings, we need to use the event_start_datetime from EVENT_FILTERED
            # Since the original EVENT table doesn't have this column, we need a different approach
            
            # We can use a subquery to identify frames within the time window
            subquery = f"""
            SELECT startframe, endframe 
            FROM EVENT_FILTERED 
            WHERE event_start_datetime >= '{time_window[0]}' 
            AND event_start_datetime <= '{time_window[1]}'
            """
            
            # Get the min and max frames from the subquery results
            try:
                frame_range_df = pd.read_sql(subquery, conn)
                if not frame_range_df.empty:
                    min_frame = frame_range_df['startframe'].min()
                    max_frame = frame_range_df['endframe'].max()
                    where_clauses.append(f"e.STARTFRAME >= {min_frame} AND e.STARTFRAME <= {max_frame}")
            except Exception as e:
                print(f"Error determining frame range from datetime: {str(e)}")
        else:
            # Original handling for seconds (convert to frames assuming 30fps)
            start_time, end_time = time_window
            start_frame = int(start_time * 30)
            end_frame = int(end_time * 30)
            where_clauses.append(f"e.STARTFRAME >= {start_frame} AND e.STARTFRAME <= {end_frame}")
    
    if animal_ids:
        animal_list = ','.join([f"'{aid}'" for aid in animal_ids])
        where_clauses.append(f"e.IDANIMALA IN ({animal_list})")
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # For debugging
    print(f"Executing query: {query}")
    
    # Execute the query and get the raw events
    try:
        events_df = pd.read_sql_query(query, conn)
        print(f"Query returned {len(events_df)} rows with columns: {events_df.columns.tolist()}")
        
        # Check if we have results
        if len(events_df) == 0:
            # Try again with a simpler query (no filtering)
            fallback_query = """
            SELECT 
                e.IDANIMALA as animal_id,
                e.NAME as eventName,
                e.STARTFRAME as startFrame,
                e.ENDFRAME as endFrame
            FROM EVENT e
            WHERE e.IDANIMALA IS NOT NULL
            LIMIT 1000
            """
            
            print(f"No results with initial query. Trying fallback query: {fallback_query}")
            events_df = pd.read_sql_query(fallback_query, conn)
            print(f"Fallback query returned {len(events_df)} rows")
        
        # Process events into features
        features_df = process_events_to_features(events_df)
        
        return features_df
    except Exception as e:
        raise Exception(f"Database query failed: {str(e)}\nQuery: {query}")

def process_events_to_features(events_df):
    """
    Process raw events data into behavioral features
    
    Args:
        events_df (pandas.DataFrame): Raw events data from the database
        
    Returns:
        pandas.DataFrame: Processed features
    """
    print(f"Processing events DataFrame with columns: {events_df.columns.tolist()}")
    
    # Check if the DataFrame is empty
    if events_df.empty:
        print("Warning: Empty events DataFrame, returning empty features DataFrame")
        return pd.DataFrame()
    
    # Ensure we have the required columns
    required_cols = ['animal_id', 'eventName']
    missing_cols = [col for col in required_cols if col not in events_df.columns]
    
    if missing_cols:
        # More detailed error message with available columns
        available_cols = events_df.columns.tolist()
        raise Exception(f"Required columns missing in events data: {missing_cols}. Available columns: {available_cols}")
    
    # Group events by animal
    grouped = events_df.groupby('animal_id')
    
    # Dictionary to store features for each animal
    animal_features = {}
    
    # Process each animal's events
    for animal_id, animal_events in grouped:
        # Extract metadata for this animal
        metadata = {
            'animal_id': animal_id,
        }
        
        # Add any available metadata columns
        for col in ['genotype', 'sex', 'strain', 'setup']:
            if col in animal_events.columns and not animal_events[col].isnull().all():
                metadata[col] = animal_events[col].iloc[0]
            else:
                metadata[col] = 'Unknown'
        
        # Initialize feature dictionary with metadata
        features = metadata.copy()
        
        # Count occurrences of different event types
        event_counts = animal_events['eventName'].value_counts()
        for event_name, count in event_counts.items():
            # Remove spaces and special characters from event names
            if event_name is None or pd.isna(event_name):
                clean_name = "unknown"
            else:
                clean_name = str(event_name).replace(' ', '_').replace('/', '_').replace('-', '_').lower()
            features[f'count_{clean_name}'] = count
        
        # Calculate average duration of events by type
        if 'startFrame' in animal_events.columns and 'endFrame' in animal_events.columns:
            for event_name in event_counts.index:
                if event_name is None or pd.isna(event_name):
                    clean_name = "unknown"
                else:
                    clean_name = str(event_name).replace(' ', '_').replace('/', '_').replace('-', '_').lower()
                event_subset = animal_events[animal_events['eventName'] == event_name]
                durations = event_subset['endFrame'] - event_subset['startFrame']
                features[f'avg_duration_{clean_name}'] = durations.mean() / 30  # Convert frames to seconds
        
        # Add to the animal features dictionary
        animal_features[animal_id] = features
    
    # Convert to DataFrame
    features_df = pd.DataFrame.from_dict(animal_features, orient='index')
    
    # Fill NaN values with 0 for count and duration features
    for col in features_df.columns:
        if col.startswith('count_') or col.startswith('avg_duration_'):
            features_df[col] = features_df[col].fillna(0)
    
    return features_df

def preprocess_features(features_df, target_column=None, exclude_columns=None, normalize=True):
    """
    Preprocess feature data for analysis
    
    Args:
        features_df (pandas.DataFrame): Features DataFrame
        target_column (str): Optional column to use as target for supervised methods
        exclude_columns (list): Optional list of columns to exclude
        normalize (bool): Whether to normalize the features
        
    Returns:
        tuple: (X, y, feature_names, scaler) where:
               - X is the feature matrix
               - y is the target array (if target_column is provided)
               - feature_names is a list of feature names
               - scaler is the fitted StandardScaler (if normalize=True)
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Always exclude metadata columns
    metadata_columns = ['animal_id', 'genotype', 'sex', 'strain', 'setup']
    if target_column in metadata_columns:
        metadata_columns.remove(target_column)
    
    # Automatically detect and exclude datetime columns
    datetime_columns = []
    for col in features_df.columns:
        # Check the first non-null value in each column
        sample_val = features_df[col].dropna().iloc[0] if not features_df[col].dropna().empty else None
        
        # If the column contains datetime strings (check common patterns)
        if isinstance(sample_val, str) and any(pattern in sample_val for pattern in ['-', '/']):
            try:
                # Try to parse as datetime
                import datetime
                import pandas as pd
                pd.to_datetime(sample_val)
                datetime_columns.append(col)
            except:
                pass
        
        # Also check column name for common datetime indicators
        if col.lower() in ['date', 'time', 'datetime', 'timestamp'] or any(
            term in col.lower() for term in ['date', 'time', '_dt', 'interval_start', 'interval_end']):
            datetime_columns.append(col)
    
    # Show info about excluded datetime columns
    if datetime_columns:
        import streamlit as st
        st.info(f"Excluding datetime columns from dimensionality reduction: {', '.join(datetime_columns)}")
    
    # Combine all exclusions
    exclude_columns = list(set(exclude_columns + metadata_columns + datetime_columns))
    
    # Separate target if provided
    y = None
    if target_column and target_column in features_df.columns:
        y = features_df[target_column].values
    
    # Select feature columns
    feature_cols = [col for col in features_df.columns if col not in exclude_columns and col != target_column]
    
    # Ensure all data can be converted to numeric
    numeric_data = features_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    # Check for columns with NaN values after conversion and exclude them
    invalid_numeric_cols = [col for col in feature_cols if numeric_data[col].isna().any()]
    if invalid_numeric_cols:
        import streamlit as st
        st.warning(f"Excluding non-numeric columns: {', '.join(invalid_numeric_cols)}")
        feature_cols = [col for col in feature_cols if col not in invalid_numeric_cols]
    
    X = features_df[feature_cols].values
    
    # Normalize features if requested
    scaler = None
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, feature_cols, scaler

def perform_pca(X, n_components=2):
    """
    Perform Principal Component Analysis (PCA)
    
    Args:
        X (numpy.ndarray): Feature matrix
        n_components (int): Number of components to keep
        
    Returns:
        tuple: (transformed_data, pca_model, explained_variance_ratio)
    """
    # Ensure we don't try to extract more components than features
    n_components = min(n_components, X.shape[1])
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(X)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return transformed_data, pca, explained_variance_ratio

def perform_lda(X, y, n_components=2):
    """
    Perform Linear Discriminant Analysis (LDA)
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target array (can be strings or categorical)
        n_components (int): Number of components to keep
        
    Returns:
        tuple: (transformed_data, lda_model, explained_variance_ratio)
    """
    import streamlit as st
    from sklearn import preprocessing
    
    # Convert target to categorical numbers if they're strings or datetime objects
    if y is None:
        st.error("Target variable (y) is None. LDA requires a categorical target variable.")
        return None, None, None
    
    # Handle different types of targets
    if isinstance(y[0], (str, np.datetime64)) or 'datetime' in str(type(y[0])):
        # Convert to categorical codes
        le = preprocessing.LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Show mapping for better interpretability
        categories = le.classes_
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        
        st.info(f"Converted {len(categories)} categorical values to numeric codes for LDA")
        if len(categories) <= 10:  # Only show mapping if manageable number of categories
            st.write("Category mapping:", mapping)
    else:
        # If y is already numeric, just ensure it's the right type
        y_encoded = y.astype(int)
    
    # Ensure we have enough components (# classes - 1 is max for LDA)
    n_classes = len(np.unique(y_encoded))
    
    # Check if we have at least 2 classes
    if n_classes < 2:
        st.error(f"LDA requires at least 2 different classes, but only found {n_classes}.")
        return None, None, None
    
    # Check if we have enough samples per class
    for class_val in np.unique(y_encoded):
        count = np.sum(y_encoded == class_val)
        if count < 2:
            st.warning(f"Class {class_val} has only {count} sample. LDA works best with multiple samples per class.")
    
    n_components = min(n_components, n_classes - 1)
    
    # Initialize and fit LDA
    try:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        transformed_data = lda.fit_transform(X, y_encoded)
        
        # LDA doesn't have explained_variance_ratio_ attribute, so we need to compute it
        # This is an approximation based on the eigenvalues
        explained_variance_ratio = lda.explained_variance_ratio_ if hasattr(lda, 'explained_variance_ratio_') else None
        
        st.success(f"LDA completed successfully with {n_components} components for {n_classes} classes")
        return transformed_data, lda, explained_variance_ratio
        
    except Exception as e:
        st.error(f"Error performing LDA: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def get_feature_importance(model, feature_names):
    """
    Get feature importance from PCA or LDA model
    
    Args:
        model: Fitted PCA or LDA model
        feature_names (list): List of feature names
        
    Returns:
        pandas.DataFrame: DataFrame with feature importance for each component
    """
    if hasattr(model, 'components_'):
        # For PCA
        components = model.components_
        importance_df = pd.DataFrame(components.T, index=feature_names)
        
        # Rename columns to Component 1, Component 2, etc.
        importance_df.columns = [f'Component {i+1}' for i in range(components.shape[0])]
        
        return importance_df
    elif hasattr(model, 'coef_'):
        # For LDA
        components = model.coef_
        importance_df = pd.DataFrame(components.T, index=feature_names)
        
        # Rename columns to Component 1, Component 2, etc.
        importance_df.columns = [f'Component {i+1}' for i in range(components.shape[0])]
        
        return importance_df
    else:
        return pd.DataFrame() 