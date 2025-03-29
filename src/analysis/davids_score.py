import numpy as np
import pandas as pd

def calculate_davids_score(chase_matrix):
    """
    Calculate normalized David's Score for dominance hierarchy.
    
    Args:
        chase_matrix: NxN matrix where entry (i,j) is number of times mouse i chased mouse j
        
    Returns:
        normalized_ds: Array of normalized David's Scores for each mouse
    """
    N = chase_matrix.shape[0]
    
    # Calculate Pij matrix (proportion of wins)
    total_interactions = chase_matrix + chase_matrix.T
    Pij = np.zeros_like(chase_matrix, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and total_interactions[i,j] > 0:
                Pij[i,j] = chase_matrix[i,j] / total_interactions[i,j]
    
    # Calculate w and l values
    w = np.sum(Pij, axis=1) / (N-1)  # wins
    l = np.sum(Pij.T, axis=1) / (N-1)  # losses
    
    # Calculate w2 and l2 (weighted sums)
    w2 = np.zeros(N)
    l2 = np.zeros(N)
    for i in range(N):
        w2[i] = sum(w[j] * Pij[i,j] for j in range(N) if j != i) / (N-1)
        l2[i] = sum(w[j] * Pij[j,i] for j in range(N) if j != i) / (N-1)
    
    # Calculate David's Score
    DS = w + w2 - l - l2
    
    # Normalize David's Score to be between 0 and N-1
    normalized_ds = (DS + (N * (N-1))/2) / N
    
    return normalized_ds

def process_chase_events(events_df):
    """
    Process chase events from behavior data to create chase matrix.
    
    Args:
        events_df: DataFrame with columns for initiator, receiver, and behavior type
        
    Returns:
        chase_matrix: NxN matrix of chase counts
    """
    # Get unique mouse IDs
    mouse_ids = sorted(pd.unique(events_df['mouse_id']))
    N = len(mouse_ids)
    
    # Create chase matrix
    chase_matrix = np.zeros((N, N))
    
    # Fill chase matrix from events
    for _, row in events_df.iterrows():
        if row['name'] == 'chase':  # or whatever identifies chase behavior
            initiator_idx = mouse_ids.index(row['idanimalA'])
            receiver_idx = mouse_ids.index(row['idanimalB'])
            chase_matrix[initiator_idx, receiver_idx] += 1
            
    return chase_matrix, mouse_ids

def analyze_dominance(events_df):
    """
    Analyze dominance hierarchy using David's Score.
    
    Args:
        events_df: DataFrame with behavioral events
        
    Returns:
        dict with:
            - dominance_scores: Dictionary mapping mouse IDs to their normalized David's Score
            - chase_matrix: Matrix of chase counts between mice
    """
    # Process events into chase matrix
    chase_matrix, mouse_ids = process_chase_events(events_df)
    
    # Calculate David's Scores
    scores = calculate_davids_score(chase_matrix)
    
    # Create results dictionary
    dominance_scores = {mouse_id: score for mouse_id, score in zip(mouse_ids, scores)}
    
    return {
        'dominance_scores': dominance_scores,
        'chase_matrix': chase_matrix
    } 