import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def plot_pca_results(pca_data, labels=None, feature_df=None, color_column=None):
    """
    Plot PCA results using Plotly for interactive visualization
    
    Args:
        pca_data (numpy.ndarray): PCA transformed data
        labels (numpy.ndarray): Optional labels for coloring points
        feature_df (pandas.DataFrame): Original feature DataFrame for additional info
        color_column (str): Column name in feature_df to use for coloring
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot of PCA results
    """
    # Create a DataFrame from PCA results
    df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
    
    # Add label/color information if provided
    if labels is not None:
        df['Label'] = labels
    
    # Add additional columns from feature_df if provided
    if feature_df is not None and color_column in feature_df.columns:
        df[color_column] = feature_df[color_column].values
    
    # Determine what to use for coloring
    color_by = color_column if color_column in df.columns else 'Label' if 'Label' in df.columns else None
    
    # Create interactive plot
    if pca_data.shape[1] >= 3 and st.checkbox("Show 3D plot", value=False, key="pca_main_3d_checkbox"):
        # 3D plot (if we have at least 3 components)
        fig = px.scatter_3d(
            df, x='PC1', y='PC2', z='PC3',
            color=color_by,
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
            title='PCA Results - 3D Visualization',
            opacity=0.7,
            size_max=10,
            hover_data=df.columns
        )
    else:
        # 2D plot
        fig = px.scatter(
            df, x='PC1', y='PC2',
            color=color_by,
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            title='PCA Results - 2D Visualization',
            opacity=0.7,
            hover_data=df.columns
        )
    
    # Add personalized layout
    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )
    
    # Add a hover effect that displays all metadata
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' + 
                     'PC1: %{x:.2f}<br>' +
                     'PC2: %{y:.2f}<br>' +
                     (('PC3: %{z:.2f}<br>') if pca_data.shape[1] >= 3 and st.checkbox("Show 3D plot", value=False, key="pca_hover_3d_checkbox") else '') +
                     '<extra></extra>'
    )
    
    return fig

def plot_lda_results(lda_data, labels, feature_df=None, color_column=None):
    """
    Plot LDA results using Plotly for interactive visualization
    
    Args:
        lda_data (numpy.ndarray): LDA transformed data
        labels (numpy.ndarray): Labels for coloring points
        feature_df (pandas.DataFrame): Original feature DataFrame for additional info
        color_column (str): Column name in feature_df to use for coloring
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot of LDA results
    """
    # Create a DataFrame from LDA results
    df = pd.DataFrame(lda_data, columns=[f'LD{i+1}' for i in range(lda_data.shape[1])])
    
    # Add label information
    df['Label'] = labels
    
    # Add additional columns from feature_df if provided
    if feature_df is not None and color_column in feature_df.columns:
        df[color_column] = feature_df[color_column].values
    
    # Determine what to use for coloring
    color_by = color_column if color_column in df.columns else 'Label'
    
    # Create interactive plot
    if lda_data.shape[1] >= 3 and st.checkbox("Show 3D plot", value=False, key="lda_3d_checkbox"):
        # 3D plot (if we have at least 3 components)
        fig = px.scatter_3d(
            df, x='LD1', y='LD2', z='LD3',
            color=color_by,
            labels={'LD1': 'Linear Discriminant 1', 'LD2': 'Linear Discriminant 2', 'LD3': 'Linear Discriminant 3'},
            title='LDA Results - 3D Visualization',
            opacity=0.7,
            hover_data=df.columns
        )
    else:
        # 2D plot
        fig = px.scatter(
            df, x='LD1', y='LD2',
            color=color_by,
            labels={'LD1': 'Linear Discriminant 1', 'LD2': 'Linear Discriminant 2'},
            title='LDA Results - 2D Visualization',
            opacity=0.7,
            hover_data=df.columns
        )
    
    # Add personalized layout
    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )
    
    return fig

def plot_explained_variance(explained_variance_ratio, title="Explained Variance Ratio"):
    """
    Plot explained variance ratio for PCA or LDA components
    
    Args:
        explained_variance_ratio (numpy.ndarray): Array of explained variance ratios
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Bar chart of explained variance
    """
    # Create data
    df = pd.DataFrame({
        'Component': [f'Component {i+1}' for i in range(len(explained_variance_ratio))],
        'Explained Variance (%)': explained_variance_ratio * 100
    })
    
    # Calculate cumulative explained variance
    df['Cumulative Explained Variance (%)'] = np.cumsum(explained_variance_ratio) * 100
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add bars for individual explained variance
    fig.add_trace(go.Bar(
        x=df['Component'],
        y=df['Explained Variance (%)'],
        name='Individual Explained Variance',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Add line for cumulative explained variance
    fig.add_trace(go.Scatter(
        x=df['Component'],
        y=df['Cumulative Explained Variance (%)'],
        name='Cumulative Explained Variance',
        marker_color='rgb(26, 118, 255)',
        mode='lines+markers'
    ))
    
    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title='Component',
        yaxis_title='Explained Variance (%)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_feature_importance(importance_df, n_features=20):
    """
    Plot feature importance for each component
    
    Args:
        importance_df (pandas.DataFrame): DataFrame with feature importance for each component
        n_features (int): Number of top features to display
        
    Returns:
        dict: Dictionary mapping component names to plotly.graph_objects.Figure objects
    """
    fig_dict = {}
    
    for column in importance_df.columns:
        # Get absolute importance values and sort
        abs_importance = importance_df[column].abs()
        sorted_idx = abs_importance.sort_values(ascending=False).index
        
        # Select top N features
        top_features = sorted_idx[:n_features]
        top_values = importance_df.loc[top_features, column]
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Feature': top_features,
            'Importance': top_values
        })
        
        # Sort by absolute importance for display
        plot_df = plot_df.sort_values('Importance', key=abs, ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            plot_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title=f'Top {n_features} Features for {column}',
            color='Importance',
            color_continuous_scale='RdBu_r',  # Red for negative, Blue for positive
            range_color=[-max(abs(plot_df['Importance'])), max(abs(plot_df['Importance']))]
        )
        
        # Customize layout
        fig.update_layout(
            template='plotly_white',
            xaxis_title='Feature Importance',
            yaxis_title='Feature',
            coloraxis_showscale=False
        )
        
        fig_dict[column] = fig
    
    return fig_dict

def plot_correlation_heatmap(features_df, exclude_cols=None):
    """
    Plot correlation heatmap for features
    
    Args:
        features_df (pandas.DataFrame): DataFrame containing features
        exclude_cols (list): List of columns to exclude
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Filter non-numeric columns and excluded columns
    numeric_df = features_df.select_dtypes(include=['number'])
    filtered_df = numeric_df[[col for col in numeric_df.columns if col not in exclude_cols]]
    
    # Calculate correlation matrix
    corr_matrix = filtered_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix, 
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Heatmap',
        range_color=[-1, 1]
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        height=800,
        width=800,
    )
    
    return fig

def plot_feature_distribution(features_df, feature_name, group_by=None):
    """
    Plot distribution of a specific feature
    
    Args:
        features_df (pandas.DataFrame): DataFrame containing features
        feature_name (str): Name of the feature to plot
        group_by (str): Optional column to group by
        
    Returns:
        plotly.graph_objects.Figure: Feature distribution plot
    """
    if feature_name not in features_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"Feature '{feature_name}' not found in data",
            xaxis_title="No data",
            yaxis_title="No data"
        )
        return fig
    
    if group_by is not None and group_by in features_df.columns:
        # Create grouped histogram
        fig = px.histogram(
            features_df, 
            x=feature_name, 
            color=group_by,
            marginal="box",
            opacity=0.7,
            barmode="overlay",
            title=f'Distribution of {feature_name} by {group_by}'
        )
    else:
        # Create simple histogram
        fig = px.histogram(
            features_df, 
            x=feature_name,
            marginal="box",
            opacity=0.7,
            title=f'Distribution of {feature_name}'
        )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        xaxis_title=feature_name,
        yaxis_title="Count"
    )
    
    return fig 