import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def normalize_scores(df):
    """Normalize ID scores to range [0, 1] for better visualization."""
    score_columns = [col for col in df.columns if col.startswith('ID_')]
    df_norm = df.copy()
    df_norm[score_columns] = (df[score_columns] - df[score_columns].min()) / (df[score_columns].max() - df[score_columns].min())
    return df_norm

def create_heptagram(ax, values, labels, title):
    """Create a heptagram (7-sided radar plot) for a single mouse."""
    # Number of variables
    num_vars = len(labels)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    
    # Close the plot by appending the first value
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Set title
    ax.set_title(title)
    
    # Set limits for the plot
    ax.set_ylim(0, 1)

def plot_mouse_profiles(data_path, output_dir, id_names=None):
    """Generate heptagram plots for all mice."""
    # Read the data
    df = pd.read_csv(data_path)
    
    # Normalize the scores
    df_norm = normalize_scores(df)
    
    # Get ID columns
    id_cols = [col for col in df.columns if col.startswith('ID_')]
    
    # Use provided ID names or create generic ones
    if id_names is None:
        id_names = [
            "High-Intensity Social Investigator",
            "Sustained Social Affiliator",
            "Environmental-Social Explorer",
            "Passive Social Recipient",
            "Anxiety-Prone Individual",
            "Dynamic Social Integrator",
            "Social Contact Maintainer"
        ]
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot settings
    plt.style.use('seaborn')
    
    # Create individual plots for each mouse
    for idx, row in df_norm.iterrows():
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Get the values for this mouse
        values = row[id_cols].values
        
        # Create the heptagram
        create_heptagram(ax, values, id_names, f"Mouse {row['mouse_id']} ID Profile")
        
        # Save the plot
        plt.savefig(output_dir / f"mouse_{row['mouse_id']}_profile.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # Create a summary plot with all mice
    n_mice = len(df_norm)
    n_cols = 3
    n_rows = (n_mice + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5*n_rows))
    for idx, row in df_norm.iterrows():
        ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='polar')
        values = row[id_cols].values
        create_heptagram(ax, values, id_names, f"Mouse {row['mouse_id']}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_mice_profiles.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create correlation matrix between IDs
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[id_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation between Identity Domains")
    plt.tight_layout()
    plt.savefig(output_dir / "id_correlations.png", bbox_inches='tight', dpi=300)
    plt.close()
if __name__ == "__main__":
    # Paths
    data_path = "data/identity_space_results.csv"
    output_dir = "data/visualization/id_profiles"
    
    # Generate plots
    plot_mouse_profiles(data_path, output_dir)
    print(f"Plots saved to {output_dir}")