# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd
import numpy as np

def calculate_behavioral_ratios(df):
    """Calculate biologically meaningful ratios from behavioral data based on taxonomy.
    
    Args:
        df: DataFrame with behavioral counts/durations
        
    Returns:
        DataFrame with additional ratio columns
    """
    ratios_df = df.copy()
    
    # Calculate total durations and counts for normalization
    total_duration = 0
    total_count = 0
    for col in df.columns:
        if col.endswith('_total_duration'):
            total_duration += df[col].fillna(0)
        elif col.endswith('_count'):
            total_count += df[col].fillna(0)
    
    # Ensure we don't divide by zero
    total_duration = total_duration.replace(0, 1e-10)
    total_count = total_count.replace(0, 1e-10)
    
    # Helper function to add social behavior metrics
    def add_social_metrics(name, behavior_cols, weights=None):
        """Add metrics for social behaviors, separating initiative (counts) from engagement (duration)"""
        if weights is None:
            weights = {col: 1.0 for col in behavior_cols}
            
        # Initiative ratios (using counts which genuinely reflect active vs passive)
        for behavior in behavior_cols:
            active_count = weights[behavior] * df[f'{behavior}_active_count']
            passive_count = weights[behavior] * df[f'{behavior}_passive_count']
            total_count = active_count + passive_count
            
            # Initiative ratio (proportion of active initiations)
            ratios_df[f'{behavior}_initiative_ratio'] = active_count / (total_count + 1e-10)
            
            # Engagement intensity (duration per event)
            total_events = df[f'{behavior}_active_count'] + df[f'{behavior}_passive_count']
            total_duration = df[f'{behavior}_active_total_duration']  # Same as passive duration
            ratios_df[f'{behavior}_engagement_intensity'] = total_duration / (total_events + 1e-10)
        
        # Overall category metrics
        total_active_count = sum(weights[b] * df[f'{b}_active_count'] for b in behavior_cols)
        total_passive_count = sum(weights[b] * df[f'{b}_passive_count'] for b in behavior_cols)
        total_duration = sum(weights[b] * df[f'{b}_active_total_duration'] for b in behavior_cols)
        
        # Overall initiative ratio for category
        ratios_df[f'{name}_overall_initiative'] = total_active_count / (total_active_count + total_passive_count + 1e-10)
        
        # Time allocation for this category of behaviors
        ratios_df[f'{name}_time_allocation'] = total_duration / total_duration
    
    # Helper function to add behavioral allocations
    def add_role_allocation(name, behaviors, weights=None):
        """Add count and duration allocations for social behaviors"""
        if weights is None:
            weights = {b: 1.0 for b in behaviors}
            
        # Active vs Passive count allocations
        active_count = sum(weights[b] * df[f'{b}_active_count'] for b in behaviors)
        passive_count = sum(weights[b] * df[f'{b}_passive_count'] for b in behaviors)
        
        # Duration allocation (same for active/passive)
        behavior_duration = sum(weights[b] * df[f'{b}_active_total_duration'] for b in behaviors)
        
        # Add to dataframe
        ratios_df[f'{name}_active_ratio'] = active_count / (active_count + passive_count + 1e-10)
        ratios_df[f'{name}_time_allocation'] = behavior_duration / total_duration
    
    # 1. Social Investigation
    add_social_metrics('investigation',
                    ['Oral_genital_Contact', 'Oral_oral_Contact'],
                    weights={'Oral_genital_Contact': 3.0,
                            'Oral_oral_Contact': 2.8})
    
    # 2. Social Approach
    add_social_metrics('approach',
                    ['Approach', 'Social_approach'],
                    weights={'Social_approach': 2.5,
                            'Approach': 2.0})
    
    # 3. Contact Maintenance
    add_social_metrics('contact',
                    ['Contact', 'Move_in_contact', 'Stop_in_contact'],
                    weights={'Contact': 1.5,
                            'Move_in_contact': 1.2,
                            'Stop_in_contact': 1.3})
    
    # 4. Social Response
    add_social_metrics('social_response',
                    ['Social_escape', 'Get_away'],
                    weights={'Social_escape': 2.0,
                            'Get_away': 1.8})
    
    # Add role-based allocations
    # Social behaviors
    add_role_allocation('social_interaction',
                       ['Contact', 'Approach', 'Social_approach',
                        'Oral_genital_Contact', 'Oral_oral_Contact'],
                       weights={'Contact': 1.5,
                               'Approach': 2.0,
                               'Social_approach': 2.5,
                               'Oral_genital_Contact': 3.0,
                               'Oral_oral_Contact': 2.8})
    
    # Movement behaviors
    add_role_allocation('movement',
                       ['Move_in_contact', 'Stop_in_contact'],
                       weights={'Move_in_contact': 1.2,
                               'Stop_in_contact': 1.3})
    
    # Response behaviors
    add_role_allocation('avoidance',
                       ['Social_escape', 'Get_away'],
                       weights={'Social_escape': 2.0,
                               'Get_away': 1.8})
    
    # Non-social behaviors (these don't have active/passive distinction)
    def add_solo_ratio(name, numerator_cols, denominator_cols, weights=None, discrete=False):
        """Add ratios for non-social behaviors"""
        if weights is None:
            weights = {col: 1.0 for col in numerator_cols + denominator_cols}
            
        num_duration = sum(weights[col] * df[f'{col}_total_duration'] for col in numerator_cols)
        den_duration = sum(weights[col] * df[f'{col}_total_duration'] for col in denominator_cols) + 1e-10
        ratios_df[f'{name}_time_ratio'] = num_duration / den_duration
        
        if discrete:
            num_count = sum(weights[col] * df[f'{col}_count'] for col in numerator_cols)
            den_count = sum(weights[col] * df[f'{col}_count'] for col in denominator_cols) + 1e-10
            ratios_df[f'{name}_freq_ratio'] = num_count / den_count
    
    # Anxiety-related ratios
    add_solo_ratio('anxiety_behavior',
                   ['WallJump', 'SAP', 'Huddling'],
                   ['Rearing', 'Center_Zone'],
                   weights={'WallJump': 2.0,
                           'SAP': 1.5,
                           'Huddling': 2.5,
                           'Rearing': 1.5,
                           'Center_Zone': 1.5},
                   discrete=True)
    
    # Spatial preference
    add_solo_ratio('spatial_preference',
                   ['Center_Zone'],
                   ['Center_Zone', 'Periphery_Zone'],
                   weights={'Center_Zone': 1.5,
                           'Periphery_Zone': 1.0})
    
    # Exploration
    add_solo_ratio('exploration',
                   ['Rear_in_centerWindow', 'Center_Zone'],
                   ['Rear_at_periphery', 'Periphery_Zone'],
                   weights={'Rear_in_centerWindow': 1.4,
                           'Center_Zone': 1.5,
                           'Rear_at_periphery': 1.0,
                           'Periphery_Zone': 1.0})
    
    return ratios_df

def calculate_composite_scores(df):
    """Calculate composite behavioral scores.
    
    Args:
        df: DataFrame with behavioral counts/durations and ratios
        
    Returns:
        DataFrame with additional composite score columns
    """
    scores_df = df.copy()
    
    # 1. Anxiety Score (lower is more anxious)
    scores_df['anxiety_score'] = -1 * (
        0.4 * df['anxiety_spatial_time_ratio'] +
        0.4 * (1 / (df['anxiety_time_ratio'] + 1e-10)) +
        0.2 * df['rearing_location_time_ratio']
    )
    
    # 2. Sociability Score
    scores_df['sociability_score'] = (
        0.3 * df['social_initiative_time_ratio'] +
        0.2 * df['contact_maintenance_time_ratio'] +
        0.2 * df['investigation_time_ratio'] +
        0.15 * df['social_activity_time_ratio'] +
        0.15 * df['contact_stability_time_ratio']
    )
    
    # 3. Social Investigation Score
    scores_df['investigation_score'] = (
        0.4 * df['investigation_time_ratio'] +
        0.4 * df['investigation_sequence_time_ratio'] +
        0.2 * df['approach_success_time_ratio']
    )
    
    # 4. Activity Score
    scores_df['activity_score'] = (
        0.4 * (df['Move_in_contact_active_total_duration'] + df['Move_isolated_total_duration']) +
        0.3 * df['social_activity_time_ratio'] +
        0.3 * df['Rearing_total_duration']
    )
    
    # 5. Social Stability Score
    scores_df['stability_score'] = (
        0.4 * df['contact_stability_time_ratio'] +
        0.3 * (1 - df['social_escape_time_ratio'])
    )
    
    return scores_df

def prepare_pca_features(df):
    """Prepare features for PCA analysis including raw counts, ratios, and composite scores.
    
    Args:
        df: DataFrame with raw behavioral data
        
    Returns:
        DataFrame ready for PCA analysis
    """
    # 1. Calculate ratios
    df_with_ratios = calculate_behavioral_ratios(df)
    
    # 2. Add composite scores
    df_with_scores = calculate_composite_scores(df_with_ratios)
    
    # 3. Separate features by type for different normalization
    time_cols = [col for col in df_with_scores.columns if '_time_' in col]
    freq_cols = [col for col in df_with_scores.columns if '_freq_' in col]
    score_cols = [col for col in df_with_scores.columns if col.endswith('_score')]
    allocation_cols = [col for col in df_with_scores.columns if '_allocation' in col]
    
    # 4. Normalize features appropriately
    df_normalized = df_with_scores.copy()
    
    # Z-score normalization for time ratios and scores
    for col in time_cols + score_cols:
        df_normalized[col] = (df_with_scores[col] - df_with_scores[col].mean()) / df_with_scores[col].std()
    
    # Special handling for frequency ratios to reduce impact of high-frequency behaviors
    for col in freq_cols:
        # Log transform then z-score normalize
        log_vals = np.log1p(df_with_scores[col])  # log1p handles zeros
        df_normalized[col] = (log_vals - log_vals.mean()) / log_vals.std()
    
    # Allocations are already normalized (sum to 1)
    
    return df_normalized

def get_feature_descriptions():
    """Get descriptions of calculated features and their biological significance."""
    return {
        'anxiety_spatial_time_ratio': 'Ratio of center to total zone occupation (inverse anxiety measure)',
        'anxiety_time_ratio': 'Ratio of anxiety-like behaviors to exploration (time-based)',
        'social_initiative_time_ratio': 'Ratio of active to passive social interactions (time-based)',
        'contact_maintenance_time_ratio': 'Tendency to maintain vs break contact (time-based)',
        'investigation_time_ratio': 'Proportion of contacts involving detailed investigation (time-based)',
        'social_rest_time_ratio': 'Preference for resting in social context (time-based)',
        'investigation_sequence_time_ratio': 'Completion rate of investigation sequences (time-based)',
        'approach_success_time_ratio': 'Success rate of social approaches (time-based)',
        'approach_response_time_ratio': 'Tendency to respond to approaches (time-based)',
        'rearing_location_time_ratio': 'Preference for central vs peripheral rearing (time-based)',
        'social_rearing_time_ratio': 'Preference for social vs isolated rearing (time-based)',
        'contact_stability_time_ratio': 'Stability of social contacts (time-based)',
        'social_escape_time_ratio': 'Tendency to escape from social interaction (time-based)',
        'train_formation_time_ratio': 'Tendency to form organized following behavior (time-based)',
        'social_time_allocation': 'Proportion of total time spent in social interactions',
        'exploration_time_allocation': 'Proportion of total time spent in exploration',
        'anxiety_time_allocation': 'Proportion of total time spent in anxiety-like behaviors',
        'investigation_time_allocation': 'Proportion of total time spent in investigation',
        'movement_time_allocation': 'Proportion of total time spent in movement',
        'anxiety_score': 'Composite anxiety score (lower = more anxious)',
        'sociability_score': 'Overall social engagement score (time-based)',
        'investigation_score': 'Social investigation thoroughness (time-based)',
        'activity_score': 'General activity level (time-based)',
        'stability_score': 'Social interaction stability (time-based)'
    } 