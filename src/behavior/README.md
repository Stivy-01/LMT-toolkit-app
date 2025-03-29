# Behavior Processing Module Documentation

## Overview
This directory contains the core behavior processing modules for the LMT analysis toolkit. These modules are responsible for processing raw behavioral data from the LMT database and converting it into structured formats suitable for analysis.

## Module Descriptions

### 1. behavior_processor.py
Base behavior processor implementing daily analysis.
- **Key Class**: `BehaviorProcessor`
- **Main Functions**:
  - `process_events()`: Processes behavioral events from database
  - `get_animal_ids()`: Retrieves animal IDs from database
  - `sanitize()`: Sanitizes column names for database
- **Features**:
  - Processes entire experimental period as one interval
  - Handles individual, pairwise, and group behaviors
  - Calculates behavior counts and durations
  - Creates BEHAVIOR_STATS and MULTI_MOUSE_EVENTS tables
- **Output Tables**:
  - `BEHAVIOR_STATS`: Individual mouse behavior statistics
  - `MULTI_MOUSE_EVENTS`: Group behavior statistics

### 2. behavior_processor_hourly.py
Processes behavioral data in hourly intervals.
- **Key Class**: `BehaviorProcessor`
- **Main Functions**: Same as base processor
- **Features**:
  - Processes data in 1-hour intervals
  - Automatically generates hourly intervals
  - Maintains temporal resolution of behaviors
  - Handles all behavior types (individual, pairwise, group)
- **Output Tables**:
  - `behavior_hourly`: Hourly individual mouse behavior statistics
  - `group_events_hourly`: Hourly group behavior statistics

### 3. behavior_processor_interval.py
Processes behavioral data in night-time intervals (19:00-07:00).
- **Key Class**: `BehaviorProcessor`
- **Main Functions**: Same as base processor
- **Features**:
  - Focuses on night-time behavior (19:00-07:00)
  - Filters out day-time events
  - Aggregates behaviors into 12-hour intervals
  - Handles all behavior types
- **Output Tables**:
  - `behavior_stats_intervals`: Night-time interval behavior statistics
  - `multi_mouse_events_intervals`: Night-time interval group behavior statistics

## Common Features Across All Processors

### Data Processing
- Event filtering and validation
- Temporal aggregation
- Behavior categorization:
  - Individual behaviors
  - Pairwise interactions (active/passive roles)
  - Group behaviors (3+ mice)
  - Nest behaviors

### Statistics Calculation
- Behavior counts
- Duration statistics:
  - Total duration
  - Mean duration
  - Median duration
  - Standard deviation
- Role-based statistics for pairwise behaviors

### Database Operations
- Table creation and schema management
- Efficient batch inserts
- Data type validation
- Column name sanitization

## Usage
The processors are typically used through the analysis pipeline, but can be used independently:

```python
from behavior.behavior_processor_interval import BehaviorProcessor

# Initialize processor with database path
processor = BehaviorProcessor("path/to/database.sqlite")

# Process events
processor.process_events()
```

## Dependencies
- pandas
- numpy
- sqlite3
- datetime

## Data Structure
### Input
- Raw LMT database with EVENT_FILTERED table containing:
  - Animal IDs (A, B, C, D)
  - Behavior names
  - Event timestamps
  - Duration information

### Output
- Structured tables with:
  - Temporal intervals
  - Individual mouse statistics
  - Group behavior statistics
  - Comprehensive behavior metrics 