# LMT Dimensionality Reduction Toolkit - Streamlit App

This Streamlit application provides a user-friendly interface for the LMT Dimensionality Reduction Toolkit, allowing researchers to analyze and visualize mouse behavior data collected using the Live Mouse Tracker.

## 🔍 Overview

The LMT Dimensionality Reduction Toolkit is designed to analyze behavioral data from the Live Mouse Tracker system. This Streamlit app provides an interactive user interface to make the analysis toolkit accessible to researchers without requiring extensive programming knowledge.

## 📚 Features

The application includes the following key features:

1. **Database Management**
   - Connect to SQLite databases from LMT
   - Support for file paths or direct uploads
   - Run custom SQL queries
   - Explore database structure

2. **Feature Extraction**
   - Extract behavioral features from event data
   - Filter by time window or specific animals
   - View feature statistics and distributions
   - Download extracted features as CSV

3. **Dimensionality Reduction**
   - PCA (Principal Component Analysis)
   - LDA (Linear Discriminant Analysis)
   - Customizable components and parameters
   - Visualize results with interactive plots

4. **Visualization**
   - Feature distribution plots
   - Correlation heatmaps
   - Feature relationship exploration
   - Animal comparisons
   - Custom visualization creation

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
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📋 Using the Application

### Connecting to a Database

1. Navigate to the "Database Management" page
2. Use the "Connect via File Path" option for large database files (>200MB)
3. Enter the full path to your SQLite database file
4. If the file has spaces or special characters, use the Path Format Converter

### Extracting Features

1. After connecting to a database, go to the "Feature Extraction" page
2. Set parameters for time window and animal filters if needed
3. Click "Extract Features" to process the data
4. Review the extracted features and metadata

### Running Dimensionality Reduction

1. After extracting features, go to the "Dimensionality Reduction" page
2. Choose between PCA or LDA methods
3. Set the parameters and feature selection options
4. Click "Run Dimensionality Reduction" to analyze
5. Explore the results with interactive visualizations

### Creating Visualizations

1. After extracting features, go to the "Visualization" page
2. Choose from different visualization types:
   - Feature Distribution: Explore distribution of individual features
   - Correlation Heatmap: View correlations between features
   - Feature Relationships: Examine relationships between feature pairs
   - Animal Comparisons: Compare individual animals or groups
   - Custom Plot: Create your own custom visualizations

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Database Connection Failures
- **Issue**: "Failed to connect to the database"
- **Solution**: Check that the path is correct and the file exists. Use the Path Format Converter if needed.

#### Missing Tables
- **Issue**: "The database does not appear to be a valid LMT database"
- **Solution**: Ensure your database contains at least ANIMAL and EVENT tables. The app will work with different naming conventions (e.g., EVENT or EVENTS).

#### Performance Issues
- **Issue**: App runs slowly with large databases
- **Solution**: Use smaller time windows when extracting features or select specific animals of interest.

## 📚 References

- [Live Mouse Tracker Project](https://github.com/fdechaumont/lmt-analysis)
- [Forkosh et al., 2019](https://www.nature.com/articles/s41593-019-0516-y) - Identity domains capture individual differences from across the behavioral repertoire
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 