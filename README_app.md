# LMT Dimensionality Reduction Toolkit - Streamlit App

This Streamlit application provides a user-friendly interface for the LMT Dimensionality Reduction Toolkit, allowing researchers to analyze and visualize mouse behavior data collected using the Live Mouse Tracker.

## 🔍 Overview

The LMT Dimensionality Reduction Toolkit is designed to analyze behavioral data from the Live Mouse Tracker system. This Streamlit app provides an interactive user interface to make the analysis toolkit accessible to researchers without requiring extensive programming knowledge.

## 📚 Application Architecture

The application is structured as follows:

### Core Components

1. **Main App (`app.py`)**
   - Entry point for the Streamlit application
   - Provides a landing page and navigation interface
   - Manages session state and global settings

2. **Database Management (`pages/1_📊_Database_Management.py`)**
   - Handles database connections and exploration
   - Supports both direct file path access and file uploads
   - Displays database structure and table information
   - Provides SQL query capabilities

3. **Utility Modules**
   - **Database Utilities (`utils/db_utils.py`)**
     - Core database connectivity functions
     - Thread-safe SQLite connection management
     - Table metadata extraction and query execution
   - **Analysis Utilities (`utils/analysis_utils.py`)**
     - Feature extraction and preprocessing
     - Dimensionality reduction algorithms (PCA, LDA)
     - Data preparation for visualization
   - **Visualization Utilities (`utils/visualization_utils.py`)**
     - Interactive plotting functions
     - Data visualization helpers
     - Chart configuration and styling

## 💾 Database Management System

The app integrates specially designed components to handle SQLite databases with flexibility and robustness:

### File Path Connection System

For connecting to local database files, the app provides:

- **Direct Path Connection**: Connect to databases using file paths, bypassing the 200MB upload limit of Streamlit
- **Path Format Handling**: Intelligent handling of various path formats (Windows backslashes, forward slashes, etc.)
- **Path Conversion Tool**: Utility to convert between different path formats
- **Path Validation**: Comprehensive checks for file existence, readability, and format

### Thread-Safe Database Connections

SQLite has a limitation where connections can only be used in the thread they were created in. Our app implements:

- **Connection Pooling**: Maintains separate connections for different threads
- **Thread-Local Storage**: Thread-specific connection tracking
- **Automatic Reconnection**: Detects and handles thread errors by reconnecting in the current thread
- **Safe Query Execution**: Wrapped query methods to handle thread-safety issues gracefully

### Database Structure Flexibility

To accommodate variations in LMT database structures:

- **Flexible Schema Detection**: Identifies required tables regardless of capitalization or naming variations
- **Alternative Table Recognition**: Detects alternative table names (e.g., EVENT vs EVENTS)
- **Table Structure Diagnostics**: Detailed reporting of expected vs. actual database structure
- **Graceful Degradation**: Falls back to alternative tables when standard ones are missing

## 🧠 Analysis Capabilities

The analysis system is built to handle behavioral data with sophistication:

### Feature Processing

- **Feature Extraction**: Extracts behavioral features from raw event data
- **Feature Selection**: Identifies and selects relevant features based on variance and correlation
- **Feature Standardization**: Normalizes features for consistent analysis
- **Feature Categorization**: Separates social and individual behavior features

### Dimensionality Reduction

- **PCA Implementation**: Principal Component Analysis for unsupervised dimensionality reduction
- **LDA Implementation**: Linear Discriminant Analysis for supervised dimensionality reduction
- **Variance Analysis**: Explained variance tracking and visualization
- **Feature Importance**: Identification of key features contributing to components

## 📊 Visualization System

The visualization system provides interactive and informative views of the data:

- **PCA/LDA Plots**: Interactive scatter plots of dimensionality reduction results
- **Feature Correlation Heatmaps**: Visualize relationships between behavioral features
- **Feature Importance Charts**: Bar charts showing feature contributions to components
- **Behavior Distribution Plots**: Distribution visualizations for specific behaviors

## 🔧 Technical Implementation Details

### Thread Safety Implementation

SQLite connections in Python can only be used in the thread they were created in. Our solution includes:

```python
# Connection pool for thread-safety
_connection_pool = {}
_thread_local = threading.local()

def get_db_connection(db_path):
    # Get current thread ID
    thread_id = threading.get_ident()
    
    # Use thread-specific connection
    thread_key = f"{normalized_path}_{thread_id}"
    if thread_key in _connection_pool:
        return _connection_pool[thread_key]
    
    # Create new connection for this thread
    conn = sqlite3.connect(normalized_path, check_same_thread=False)
    _connection_pool[thread_key] = conn
    return conn
```

This pattern is applied to all database interaction functions, allowing seamless operation across Streamlit's multi-threaded environment.

### Database Structure Compatibility

The app is specifically designed to work with the user's LMT database structure:

```python
# Standard tables in the user's database
expected_structure = ['ANIMAL', 'EVENT', 'DETECTION', 'FRAME', 'LOG', 'sqlite_sequence']

# Flexible table checking
has_animal = 'ANIMAL' in tables
has_event = 'EVENT' in tables or 'EVENTS' in tables

# Can proceed with minimum required tables
return has_animal and has_event
```

This allows the app to work with different variations of LMT database structures.

### Path Handling for Windows Compatibility

Special care is taken to handle Windows paths properly:

```python
# Replace backslashes with forward slashes
path = path.replace('\\', '/')

# Handle escaped backslashes
path = path.replace('//', '/')

# Normalize path separators
path = os.path.normpath(path)
```

This ensures that paths with spaces, special characters, and different slash styles all work correctly.

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9 or higher
- SQLite database from LMT experiments

### Installation Steps

1. **Create and activate a virtual environment**
   ```bash
   python -m venv streamlit_env
   
   # Windows
   .\streamlit_env\Scripts\activate
   
   # macOS/Linux
   source streamlit_env/bin/activate
   ```

2. **Install dependencies in the correct order**
   ```bash
   pip install numpy>=1.23.5
   pip install scipy>=1.9.0
   pip install streamlit>=1.22.0
   pip install -r streamlit_app/requirements.txt
   pip install -e .
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app/app.py
   ```

## 📋 Using the Application

### Connecting to a Database

1. Navigate to the "Database Management" page
2. Use the "Connect via File Path" option for large database files (>200MB)
3. Enter the full path to your SQLite database file
4. If the file has spaces or special characters, use the Path Format Converter

### Working with Large Databases

For databases larger than 200MB:
- Always use the file path connection method instead of uploading
- Ensure your database has the required tables (`ANIMAL` and `EVENT`)
- If using a non-standard table structure, check the tables found in the connection diagnostic

### Running SQL Queries

1. Connect to a database
2. Go to the "Run SQL Queries" tab
3. Enter your SQL query
4. If you encounter a thread error, try running the query again (the app will reconnect automatically)

## 🧩 Component Descriptions

### Database Management Page
- **Connect via File Path**: Connects to databases using direct file paths
- **File Path Format Help**: Provides guidance on formatting file paths for different operating systems
- **Path Format Converter**: Converts between different path formats
- **Database Structure Display**: Shows tables found and their compatibility with LMT requirements
- **SQL Query Interface**: Allows running custom SQL queries against the database

### Utils/db_utils.py
- **Connection Management**: Thread-safe database connection handling
- **Query Execution**: Safe execution of SQL queries across threads
- **Table Information**: Extraction of table metadata and structure
- **Database Validation**: Checking databases against expected LMT structure

### Utils/analysis_utils.py
- **Feature Management**: Functions to extract and preprocess behavioral features
- **Dimensionality Reduction**: Implementations of PCA and LDA algorithms
- **Feature Importance**: Analysis of feature contributions to components

### Utils/visualization_utils.py
- **Plotting Functions**: Creation of interactive visualizations
- **Chart Configuration**: Customization of plot appearance and behavior
- **Data Formatting**: Preparation of data for visualization

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Thread-Related Errors
- **Symptom**: "SQLite objects created in a thread can only be used in that same thread"
- **Solution**: The app will automatically attempt to reconnect. If the error persists, refresh the page.

#### Database Connection Failures
- **Symptom**: "Failed to connect to the database"
- **Solution**: Check that the path is correct and the file exists. Use the Path Format Converter if needed.

#### Missing Tables
- **Symptom**: "The database does not appear to be a valid LMT database"
- **Solution**: The app now recognizes alternative table structures (EVENT instead of EVENTS). If your table structure is different, it will still try to work with it.

## 📚 References

- [Live Mouse Tracker Project](https://github.com/fdechaumont/lmt-analysis)
- [Forkosh et al., 2019](https://www.nature.com/articles/s41593-019-0516-y) - Identity domains capture individual differences from across the behavioral repertoire
- [Streamlit Documentation](https://docs.streamlit.io/) 