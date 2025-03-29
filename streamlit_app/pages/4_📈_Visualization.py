import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add the streamlit_app directory to the path to import utils
streamlit_app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(streamlit_app_path)
from utils.visualization_utils import plot_feature_distribution, plot_correlation_heatmap

# Page configuration
st.set_page_config(
    page_title="Visualization - LMT Toolkit",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'features' not in st.session_state:
    st.session_state.features = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

st.title("📈 Interactive Visualization")
st.markdown("""
Explore your data with interactive visualizations.
This page provides advanced visualization options for your extracted features and analysis results.
""")

# Check if features are available
if 'features' not in st.session_state or st.session_state.features is None:
    st.warning("No features available. Please extract features in the Feature Extraction page first.")
    st.stop()

# Display feature information
features_df = st.session_state.features
st.success(f"Using features for {len(features_df)} animals with {len(features_df.columns)} features")

# Create tabs for different visualization options
viz_tabs = st.tabs([
    "Feature Distribution", 
    "Correlation Heatmap", 
    "Feature Relationships", 
    "Animal Comparisons",
    "Custom Plot"
])

# --- Feature Distribution Tab ---
with viz_tabs[0]:
    st.header("Feature Distribution")
    st.markdown("Explore the distribution of individual features, optionally grouped by animal metadata.")
    
    # Create columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Get behavioral features
        behavior_features = [col for col in features_df.columns if col.startswith('count_') or col.startswith('avg_duration_')]
        
        # Feature selection
        if behavior_features:
            selected_feature = st.selectbox(
                "Select feature to visualize",
                behavior_features,
                index=0
            )
        else:
            st.warning("No behavioral features found in the data")
            selected_feature = None
    
    with col2:
        # Group by selection
        metadata_columns = ['genotype', 'sex', 'strain', 'setup']
        available_columns = [col for col in metadata_columns if col in features_df.columns]
        
        if available_columns:
            group_by = st.selectbox(
                "Group by (optional)",
                ["None"] + available_columns,
                index=0
            )
            group_by = None if group_by == "None" else group_by
        else:
            group_by = None
    
    # Plot feature distribution
    if selected_feature:
        dist_fig = plot_feature_distribution(features_df, selected_feature, group_by=group_by)
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Display basic statistics
        st.subheader(f"Statistics for {selected_feature}")
        
        if group_by:
            # Statistics by group
            stats_by_group = features_df.groupby(group_by)[selected_feature].describe()
            st.dataframe(stats_by_group)
        else:
            # Overall statistics
            stats = features_df[selected_feature].describe()
            st.dataframe(pd.DataFrame(stats).transpose())

# --- Correlation Heatmap Tab ---
with viz_tabs[1]:
    st.header("Correlation Heatmap")
    st.markdown("Visualize correlations between behavioral features.")
    
    # Options for correlation calculation
    st.subheader("Correlation Options")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature type filter
        feature_type = st.multiselect(
            "Feature types to include",
            ["count_", "avg_duration_"],
            default=["count_", "avg_duration_"]
        )
    
    with col2:
        # Correlation method
        corr_method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
            index=0
        )
    
    # Filter features based on selection
    selected_features = []
    for prefix in feature_type:
        selected_features.extend([col for col in features_df.columns if col.startswith(prefix)])
    
    # Display correlation heatmap
    if selected_features:
        # Filter DataFrame to selected features
        filtered_df = features_df[selected_features]
        
        # Calculate correlation with selected method
        corr_matrix = filtered_df.corr(method=corr_method)
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix, 
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title=f'Feature Correlation Heatmap ({corr_method})',
            range_color=[-1, 1]
        )
        
        # Update layout for better display
        fig.update_layout(
            template='plotly_white',
            height=800,
            width=800,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Option to download correlation matrix
        corr_csv = corr_matrix.to_csv()
        st.download_button(
            label="Download correlation matrix as CSV",
            data=corr_csv,
            file_name=f"correlation_matrix_{corr_method}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No features selected for correlation analysis")

# --- Feature Relationships Tab ---
with viz_tabs[2]:
    st.header("Feature Relationships")
    st.markdown("Explore relationships between pairs of behavioral features.")
    
    # Get behavioral features
    behavior_features = [col for col in features_df.columns if col.startswith('count_') or col.startswith('avg_duration_')]
    
    # Feature selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if behavior_features:
            x_feature = st.selectbox(
                "X-axis feature",
                behavior_features,
                index=0
            )
        else:
            st.warning("No behavioral features found in the data")
            x_feature = None
    
    with col2:
        if behavior_features:
            y_feature = st.selectbox(
                "Y-axis feature",
                behavior_features,
                index=min(1, len(behavior_features) - 1)
            )
        else:
            y_feature = None
    
    with col3:
        # Color by selection
        metadata_columns = ['genotype', 'sex', 'strain', 'setup']
        available_columns = [col for col in metadata_columns if col in features_df.columns]
        
        if available_columns:
            color_by = st.selectbox(
                "Color by (optional)",
                ["None"] + available_columns,
                index=0
            )
            color_by = None if color_by == "None" else color_by
        else:
            color_by = None
    
    # Create scatter plot
    if x_feature and y_feature:
        # Plot options
        col1, col2 = st.columns(2)
        
        with col1:
            plot_type = st.selectbox(
                "Plot type",
                ["Scatter", "Scatter with trendline", "Hexbin"],
                index=0
            )
        
        with col2:
            add_hover_info = st.checkbox("Show animal ID on hover", value=True, key="behavior_hover_info")
        
        # Prepare data
        plot_data = features_df.copy()
        
        # Handle missing values
        if plot_data[x_feature].isna().any() or plot_data[y_feature].isna().any():
            st.warning("Some values are missing. Rows with missing values will be filtered out.")
            plot_data = plot_data.dropna(subset=[x_feature, y_feature])
        
        # Create plot based on selected type
        if plot_type == "Scatter":
            fig = px.scatter(
                plot_data,
                x=x_feature,
                y=y_feature,
                color=color_by,
                hover_name="animal_id" if "animal_id" in plot_data.columns and add_hover_info else None,
                title=f"Relationship between {x_feature} and {y_feature}",
                opacity=0.7,
                height=600
            )
        
        elif plot_type == "Scatter with trendline":
            fig = px.scatter(
                plot_data,
                x=x_feature,
                y=y_feature,
                color=color_by,
                hover_name="animal_id" if "animal_id" in plot_data.columns and add_hover_info else None,
                title=f"Relationship between {x_feature} and {y_feature}",
                opacity=0.7,
                height=600,
                trendline="ols"  # Add ordinary least squares trendline
            )
        
        else:  # Hexbin
            fig = px.density_heatmap(
                plot_data,
                x=x_feature,
                y=y_feature,
                title=f"Density Heatmap of {x_feature} vs {y_feature}",
                height=600,
                nbinsx=20,
                nbinsy=20,
                marginal_x="histogram",
                marginal_y="histogram"
            )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation information
        st.subheader("Correlation Analysis")
        
        # Calculate correlation
        pearson_corr = plot_data[x_feature].corr(plot_data[y_feature], method='pearson')
        spearman_corr = plot_data[x_feature].corr(plot_data[y_feature], method='spearman')
        
        col1, col2 = st.columns(2)
        col1.metric("Pearson Correlation", f"{pearson_corr:.3f}")
        col2.metric("Spearman Correlation", f"{spearman_corr:.3f}")
        
        # If colored by a group, show correlation by group
        if color_by:
            st.subheader(f"Correlation by {color_by}")
            
            # Calculate correlation for each group
            group_corrs = []
            for group in plot_data[color_by].unique():
                group_data = plot_data[plot_data[color_by] == group]
                if len(group_data) >= 3:  # Need at least 3 points for meaningful correlation
                    pearson = group_data[x_feature].corr(group_data[y_feature], method='pearson')
                    spearman = group_data[x_feature].corr(group_data[y_feature], method='spearman')
                    group_corrs.append({
                        color_by: group,
                        'Count': len(group_data),
                        'Pearson': pearson,
                        'Spearman': spearman
                    })
                else:
                    group_corrs.append({
                        color_by: group,
                        'Count': len(group_data),
                        'Pearson': 'N/A (not enough data)',
                        'Spearman': 'N/A (not enough data)'
                    })
            
            # Display as DataFrame
            if group_corrs:
                st.dataframe(pd.DataFrame(group_corrs))

# --- Animal Comparisons Tab ---
with viz_tabs[3]:
    st.header("Animal Comparisons")
    st.markdown("Compare behavioral profiles between animals or groups of animals.")
    
    # Get animal IDs
    if 'animal_id' in features_df.columns:
        animal_ids = features_df['animal_id'].tolist()
    else:
        animal_ids = features_df.index.tolist()
    
    # Comparison type selection
    comparison_type = st.radio(
        "Comparison type",
        ["Individual animals", "Group comparison"],
        index=0
    )
    
    if comparison_type == "Individual animals":
        # Select animals to compare
        selected_animals = st.multiselect(
            "Select animals to compare",
            animal_ids,
            default=animal_ids[:min(3, len(animal_ids))]
        )
        
        if selected_animals:
            # Filter features for selected animals
            if 'animal_id' in features_df.columns:
                animals_data = features_df[features_df['animal_id'].isin(selected_animals)]
            else:
                animals_data = features_df.loc[selected_animals]
            
            # Get behavioral features
            behavior_features = [col for col in animals_data.columns if col.startswith('count_') or col.startswith('avg_duration_')]
            
            # Visualization type
            viz_type = st.selectbox(
                "Visualization type",
                ["Bar chart", "Radar chart", "Heatmap"],
                index=0
            )
            
            # Feature selection
            n_features = min(len(behavior_features), 10)  # Limit to top 10 by default
            top_n = st.slider("Number of top features to display", 1, min(30, len(behavior_features)), n_features)
            
            # Sort features by average value
            feature_means = animals_data[behavior_features].mean()
            top_features = feature_means.sort_values(ascending=False).head(top_n).index.tolist()
            
            # Create visualization
            if viz_type == "Bar chart":
                # Reshape data for grouped bar chart
                plot_data = animals_data[top_features].transpose().reset_index()
                plot_data = plot_data.rename(columns={'index': 'Feature'})
                
                # Melt the DataFrame for Plotly
                plot_data_melted = pd.melt(
                    plot_data, 
                    id_vars=['Feature'], 
                    var_name='Animal', 
                    value_name='Value'
                )
                
                # Create grouped bar chart
                fig = px.bar(
                    plot_data_melted,
                    x='Feature',
                    y='Value',
                    color='Animal',
                    barmode='group',
                    title=f"Comparison of Top {top_n} Features Across Selected Animals",
                    height=600
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Radar chart":
                # Create radar chart (polar plot)
                fig = go.Figure()
                
                # Add trace for each animal
                for animal in animals_data.index:
                    fig.add_trace(go.Scatterpolar(
                        r=animals_data.loc[animal, top_features].values,
                        theta=top_features,
                        fill='toself',
                        name=str(animal)
                    ))
                
                # Update layout
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        )
                    ),
                    title=f"Radar Chart of Top {top_n} Features",
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Heatmap
                # Create heatmap
                fig = px.imshow(
                    animals_data[top_features].transpose(),
                    title=f"Heatmap of Top {top_n} Features Across Selected Animals",
                    labels=dict(x="Animal", y="Feature", color="Value"),
                    height=800,
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_tickangle=0,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Option to download comparison data
            comparison_csv = animals_data[top_features].to_csv()
            st.download_button(
                label="Download comparison data as CSV",
                data=comparison_csv,
                file_name="animal_comparison.csv",
                mime="text/csv"
            )
        
        else:
            st.warning("Please select at least one animal to compare")
    
    else:  # Group comparison
        # Select grouping variable
        metadata_columns = ['genotype', 'sex', 'strain', 'setup']
        available_columns = [col for col in metadata_columns if col in features_df.columns]
        
        if available_columns:
            group_by = st.selectbox(
                "Group animals by",
                available_columns,
                index=0
            )
            
            # Get groups
            groups = features_df[group_by].unique().tolist()
            
            # Select groups to compare
            selected_groups = st.multiselect(
                f"Select {group_by} groups to compare",
                groups,
                default=groups[:min(3, len(groups))]
            )
            
            if selected_groups:
                # Filter features for selected groups
                groups_data = features_df[features_df[group_by].isin(selected_groups)]
                
                # Get behavioral features
                behavior_features = [col for col in groups_data.columns if col.startswith('count_') or col.startswith('avg_duration_')]
                
                # Visualization type
                viz_type = st.selectbox(
                    "Visualization type",
                    ["Bar chart", "Radar chart", "Box plot"],
                    index=0
                )
                
                # Feature selection
                n_features = min(len(behavior_features), 10)  # Limit to top 10 by default
                top_n = st.slider("Number of top features to display", 1, min(30, len(behavior_features)), n_features)
                
                # Sort features by average value
                feature_means = groups_data[behavior_features].mean()
                top_features = feature_means.sort_values(ascending=False).head(top_n).index.tolist()
                
                # Create visualization
                if viz_type == "Bar chart":
                    # Group by the selected variable and calculate means
                    group_means = groups_data.groupby(group_by)[top_features].mean().reset_index()
                    
                    # Melt the DataFrame for Plotly
                    plot_data_melted = pd.melt(
                        group_means, 
                        id_vars=[group_by], 
                        var_name='Feature', 
                        value_name='Mean Value'
                    )
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        plot_data_melted,
                        x='Feature',
                        y='Mean Value',
                        color=group_by,
                        barmode='group',
                        title=f"Comparison of Top {top_n} Features Across {group_by} Groups",
                        height=600
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Radar chart":
                    # Group by the selected variable and calculate means
                    group_means = groups_data.groupby(group_by)[top_features].mean()
                    
                    # Create radar chart (polar plot)
                    fig = go.Figure()
                    
                    # Add trace for each group
                    for group in group_means.index:
                        fig.add_trace(go.Scatterpolar(
                            r=group_means.loc[group].values,
                            theta=top_features,
                            fill='toself',
                            name=str(group)
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                            )
                        ),
                        title=f"Radar Chart of Top {top_n} Features by {group_by}",
                        height=700
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # Box plot
                    # Create box plot
                    # Select a feature to plot
                    selected_feature = st.selectbox(
                        "Select feature for box plot",
                        top_features,
                        index=0
                    )
                    
                    # Create box plot
                    fig = px.box(
                        groups_data,
                        x=group_by,
                        y=selected_feature,
                        color=group_by,
                        points="all",
                        title=f"Distribution of {selected_feature} by {group_by}",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical comparison
                st.subheader("Statistical Comparison")
                st.markdown("""
                This section provides basic statistical comparisons between groups. For more rigorous 
                statistical analysis, download the data and use specialized statistical software.
                """)
                
                # Calculate statistics for each group
                group_stats = groups_data.groupby(group_by)[top_features].agg(['mean', 'std', 'min', 'max', 'count'])
                st.dataframe(group_stats)
                
                # Option to download group comparison data
                stats_csv = group_stats.to_csv()
                st.download_button(
                    label="Download group statistics as CSV",
                    data=stats_csv,
                    file_name=f"group_comparison_by_{group_by}.csv",
                    mime="text/csv"
                )
            
            else:
                st.warning(f"Please select at least one {group_by} group to compare")
        
        else:
            st.warning("No metadata columns available for group comparison")

# --- Custom Plot Tab ---
with viz_tabs[4]:
    st.header("Custom Plot")
    st.markdown("Create your own custom visualization by selecting axes, color, and plot type.")
    
    # Get all numeric columns
    numeric_cols = features_df.select_dtypes(include=['number']).columns.tolist()
    
    # Create columns for controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # X-axis selection
        x_axis = st.selectbox(
            "X-axis",
            numeric_cols,
            index=0
        )
    
    with col2:
        # Y-axis selection
        y_axis = st.selectbox(
            "Y-axis",
            numeric_cols,
            index=min(1, len(numeric_cols) - 1)
        )
    
    with col3:
        # Color by selection
        all_columns = features_df.columns.tolist()
        color_by = st.selectbox(
            "Color by (optional)",
            ["None"] + all_columns,
            index=0
        )
        color_by = None if color_by == "None" else color_by
    
    # Plot type selection
    plot_type = st.selectbox(
        "Plot type",
        ["Scatter", "Line", "Bar", "Box", "Violin", "Histogram", "Density Contour", "3D Scatter"],
        index=0
    )
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        # Size by selection (for scatter plots)
        if plot_type in ["Scatter", "3D Scatter"]:
            size_by = st.selectbox(
                "Size by (optional)",
                ["None"] + numeric_cols,
                index=0
            )
            size_by = None if size_by == "None" else size_by
        else:
            size_by = None
        
        # Facet by selection (for multiple plots)
        facet_by = st.selectbox(
            "Facet by (optional)",
            ["None"] + all_columns,
            index=0
        )
        facet_by = None if facet_by == "None" else facet_by
    
    with col2:
        # Z-axis selection (for 3D plots)
        if plot_type == "3D Scatter":
            z_axis = st.selectbox(
                "Z-axis",
                numeric_cols,
                index=min(2, len(numeric_cols) - 1)
            )
        else:
            z_axis = None
        
        # Log scale options
        log_x = st.checkbox("Log scale for X-axis", value=False, key="scatter_log_x")
        log_y = st.checkbox("Log scale for Y-axis", value=False, key="scatter_log_y")
    
    # Create the plot
    if plot_type == "Scatter":
        fig = px.scatter(
            features_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            size=size_by,
            facet_col=facet_by,
            log_x=log_x,
            log_y=log_y,
            title=f"Custom Scatter Plot: {y_axis} vs {x_axis}",
            height=600,
            opacity=0.7,
            hover_name="animal_id" if "animal_id" in features_df.columns else None
        )
    
    elif plot_type == "Line":
        fig = px.line(
            features_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            log_x=log_x,
            log_y=log_y,
            title=f"Custom Line Plot: {y_axis} vs {x_axis}",
            height=600,
            markers=True
        )
    
    elif plot_type == "Bar":
        fig = px.bar(
            features_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            log_x=log_x,
            log_y=log_y,
            title=f"Custom Bar Plot: {y_axis} vs {x_axis}",
            height=600,
            barmode="group" if color_by else "relative"
        )
    
    elif plot_type == "Box":
        fig = px.box(
            features_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            log_x=log_x,
            log_y=log_y,
            title=f"Custom Box Plot: {y_axis} by {x_axis}",
            height=600,
            points="all"
        )
    
    elif plot_type == "Violin":
        fig = px.violin(
            features_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            log_x=log_x,
            log_y=log_y,
            title=f"Custom Violin Plot: {y_axis} by {x_axis}",
            height=600,
            box=True,
            points="all"
        )
    
    elif plot_type == "Histogram":
        fig = px.histogram(
            features_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            log_x=log_x,
            log_y=log_y,
            title=f"Custom Histogram: {y_axis} vs {x_axis}",
            height=600,
            marginal="box"
        )
    
    elif plot_type == "Density Contour":
        fig = px.density_contour(
            features_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            log_x=log_x,
            log_y=log_y,
            title=f"Custom Density Contour: {y_axis} vs {x_axis}",
            height=600,
            marginal_x="histogram",
            marginal_y="histogram"
        )
    
    elif plot_type == "3D Scatter":
        fig = px.scatter_3d(
            features_df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color=color_by,
            size=size_by,
            title=f"Custom 3D Scatter Plot: {x_axis}, {y_axis}, {z_axis}",
            height=700,
            opacity=0.7
        )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Option to download plot data
    plot_data = features_df[[x_axis, y_axis]].copy()
    if z_axis:
        plot_data[z_axis] = features_df[z_axis]
    if color_by:
        plot_data[color_by] = features_df[color_by]
    if size_by:
        plot_data[size_by] = features_df[size_by]
    
    plot_csv = plot_data.to_csv()
    st.download_button(
        label="Download plot data as CSV",
        data=plot_csv,
        file_name="custom_plot_data.csv",
        mime="text/csv"
    )

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 LMT Dimensionality Reduction Toolkit") 