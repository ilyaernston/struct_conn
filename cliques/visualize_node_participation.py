"""
Visualization script for node participation analysis.

This script generates visualizations from clique_measures and node_participation data:
1. Line plot of node participation averaged across subjects with SD bands
   - Supports multiple node participation datasets with different line styles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import os


# Yeo-7 network color scheme
YEO7_COLORS = {
    'Visual': '#781286',
    'Somatomotor': '#4682b4',
    'Dorsal Attention': '#00760e',
    'Ventral Attention': '#c43afb',
    'Limbic': '#dcf8a4',
    'Frontoparietal': '#e69422',
    'Default': '#cd3e4e',
    'Unknown': '#808080'
}

# Line styles for different datasets
LINE_STYLES = [
    ('solid', '-'),
    ('dashed', '--'),
    ('dashdot', '-.'),
    ('dotted', ':'),
]


def load_node_participation_files(node_participation_paths: List[str]) -> List[pd.DataFrame]:
    """Load multiple node participation data files.
    
    Args:
        node_participation_paths: List of paths to node participation files (csv or parquet).
        
    Returns:
        List of node participation DataFrames
    """
    dataframes = []
    
    for path in node_participation_paths:
        # Determine file format and load
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        dataframes.append(df)
        print(f"Loaded node participation data from {path}")
        print(f"  - {len(df['node_id'].unique())} unique nodes, {len(df['subject_id'].unique())} subjects")
    
    return dataframes


def validate_node_participation_consistency(dataframes: List[pd.DataFrame]) -> None:
    """Validate that all node participation DataFrames have the same structure and nodes.
    
    Args:
        dataframes: List of node participation DataFrames to validate.
        
    Raises:
        ValueError: If DataFrames don't have consistent structure or nodes.
    """
    if len(dataframes) < 1:
        raise ValueError("At least one node participation DataFrame is required")
    
    # Check that all DataFrames have required columns
    required_cols = {'subject_id', 'node_id', 'n_cliques'}
    for i, df in enumerate(dataframes):
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame {i} missing required columns. Expected: {required_cols}, Got: {set(df.columns)}")
    
    # Get reference set of nodes from first DataFrame
    reference_nodes = set(dataframes[0]['node_id'].unique())
    
    # Check that all DataFrames have the same nodes
    for i, df in enumerate(dataframes[1:], start=1):
        current_nodes = set(df['node_id'].unique())
        if current_nodes != reference_nodes:
            missing_in_current = reference_nodes - current_nodes
            extra_in_current = current_nodes - reference_nodes
            error_msg = f"Node mismatch in DataFrame {i}:"
            if missing_in_current:
                error_msg += f"\n  Missing nodes: {sorted(list(missing_in_current))[:10]}{'...' if len(missing_in_current) > 10 else ''}"
            if extra_in_current:
                error_msg += f"\n  Extra nodes: {sorted(list(extra_in_current))[:10]}{'...' if len(extra_in_current) > 10 else ''}"
            raise ValueError(error_msg)
    
    print(f"\n✓ Validation passed: All {len(dataframes)} DataFrames have consistent structure with {len(reference_nodes)} nodes")


def load_node_participation(node_participation_path: str) -> pd.DataFrame:
    """Load node participation data.
    
    Args:
        node_participation_path: Path to node participation file (csv or parquet).
        
    Returns:
        Node participation DataFrame
    """
    # Determine file format and load
    if node_participation_path.endswith('.parquet'):
        node_participation = pd.read_parquet(node_participation_path)
    else:
        node_participation = pd.read_csv(node_participation_path)
    
    print(f"Loaded node participation data for {len(node_participation['node_id'].unique())} unique nodes")
    
    return node_participation


def load_clique_measures(clique_measures_path: str) -> pd.DataFrame:
    """Load clique measures data.
    
    Args:
        clique_measures_path: Path to clique measures file (csv or parquet).
        
    Returns:
        Clique measures DataFrame
    """
    # Determine file format and load
    if clique_measures_path.endswith('.parquet'):
        clique_measures = pd.read_parquet(clique_measures_path)
    else:
        clique_measures = pd.read_csv(clique_measures_path)
    
    print(f"Loaded {len(clique_measures)} clique measures from {len(clique_measures['subject_id'].unique())} subjects")
    
    return clique_measures


def get_node_yeo7_mapping(clique_measures: pd.DataFrame) -> Dict[int, str]:
    """Extract node to Yeo-7 network mapping from clique measures.
    
    For each node, determine the most common Yeo-7 network it belongs to
    across all cliques it participates in.
    
    Args:
        clique_measures: DataFrame with clique measures.
        
    Returns:
        Dictionary mapping node_id to Yeo-7 network name.
    """
    node_networks = {}
    
    for _, row in clique_measures.iterrows():
        # Parse nodes from comma-separated string
        if isinstance(row['nodes'], str):
            nodes = [int(n) for n in row['nodes'].split(',')]
        else:
            nodes = row['nodes']
        
        yeo7_primary = row['yeo7_primary']
        
        # Assign this network to all nodes in the clique
        for node in nodes:
            if node not in node_networks:
                node_networks[node] = []
            node_networks[node].append(yeo7_primary)
    
    # For each node, pick the most common network
    node_yeo7_map = {}
    for node, networks in node_networks.items():
        # Get most common network
        network_counts = pd.Series(networks).value_counts()
        node_yeo7_map[node] = network_counts.index[0]
    
    return node_yeo7_map


def plot_node_participation(node_participation_dfs: List[pd.DataFrame], 
                            df_labels: List[str],
                            node_yeo7_map: Dict[int, str], 
                            output_dir: Path, 
                            show_plot: bool = False):
    """Create line plot of node participation averaged across subjects.
    
    Args:
        node_participation_dfs: List of DataFrames with columns [subject_id, node_id, n_cliques].
        df_labels: List of labels for each DataFrame (for legend).
        node_yeo7_map: Dictionary mapping node_id to Yeo-7 network.
        output_dir: Directory to save the plot.
        show_plot: Whether to display the plot interactively.
    """
    print("\nGenerating node participation plot...")
    
    if len(node_participation_dfs) > len(LINE_STYLES):
        raise ValueError(f"Too many datasets ({len(node_participation_dfs)}). Maximum supported: {len(LINE_STYLES)}")
    
    # Process first DataFrame to get node ordering and network assignment
    node_stats_list = []
    node_order = []
    network_order = []
    
    for df_idx, node_participation in enumerate(node_participation_dfs):
        # Calculate mean and std for each node across subjects
        node_stats = node_participation.groupby('node_id')['n_cliques'].agg(['mean', 'std']).reset_index()
        node_stats['std'] = node_stats['std'].fillna(0)  # Handle single subject case
        
        # Add Yeo-7 network mapping (only for first DataFrame)
        if df_idx == 0:
            node_stats['yeo7_network'] = node_stats['node_id'].map(node_yeo7_map)
            node_stats['yeo7_network'] = node_stats['yeo7_network'].fillna('Unknown')
            
            # Sort by Yeo-7 network first, then by node_id within each network
            node_stats = node_stats.sort_values(['yeo7_network', 'node_id'])
            
            # Save the ordering for other DataFrames
            node_order = node_stats['node_id'].tolist()
            network_order = node_stats['yeo7_network'].tolist()
        else:
            # Use the same ordering as the first DataFrame
            node_stats['yeo7_network'] = node_stats['node_id'].map(node_yeo7_map)
            node_stats['yeo7_network'] = node_stats['yeo7_network'].fillna('Unknown')
            
            # Reorder to match first DataFrame
            node_stats = node_stats.set_index('node_id').loc[node_order].reset_index()
            node_stats['yeo7_network'] = network_order
        
        node_stats_list.append(node_stats)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create x-axis positions
    x_positions = np.arange(len(node_stats_list[0]))
    
    # Add background color-coding by network (using first DataFrame's network assignment)
    current_network = network_order[0]
    start_idx = 0
    
    for i, network in enumerate(network_order):
        if i == len(network_order) - 1 or network_order[i + 1] != current_network:
            # End of current network group
            end_idx = i + 1
            color = YEO7_COLORS.get(current_network, '#808080')
            ax.axvspan(start_idx - 0.5, end_idx - 0.5, facecolor=color, alpha=0.15)
            
            # Add vertical line to separate networks (except at the end)
            if i < len(network_order) - 1:
                ax.axvline(x=end_idx - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            
            # Update for next network
            if i < len(network_order) - 1:
                current_network = network_order[i + 1]
                start_idx = i + 1
    
    # Plot lines for each dataset with different line styles
    for df_idx, (node_stats, label) in enumerate(zip(node_stats_list, df_labels)):
        line_name, line_style = LINE_STYLES[df_idx]
        
        # Plot main line for mean node participation
        ax.plot(x_positions, node_stats['mean'], 
               color='black', linewidth=2, linestyle=line_style,
               label=f'{label} (Mean)', alpha=0.8, zorder=3 + df_idx)
        
        # Plot SD bands
        ax.fill_between(x_positions, 
                        node_stats['mean'] - node_stats['std'],
                        node_stats['mean'] + node_stats['std'],
                        color='gray', alpha=0.15 / (df_idx + 1), 
                        label=f'{label} (±1 SD)', zorder=2 + df_idx)
    
    # Set x-axis labels (remove node IDs for cleaner look)
    ax.set_xticks([])
    
    # Create legend with network labels and dataset labels
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    unique_networks = list(dict.fromkeys(network_order))  # Preserve order
    
    # Build legend elements
    legend_elements = []
    
    # Add dataset-specific elements
    for df_idx, label in enumerate(df_labels):
        line_name, line_style = LINE_STYLES[df_idx]
        legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                     linestyle=line_style, label=f'{label} (Mean)'))
        legend_elements.append(Patch(facecolor='gray', alpha=0.15 / (df_idx + 1), 
                                    label=f'{label} (±1 SD)'))
    
    # Add separator
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Add network colors
    for network in unique_networks:
        legend_elements.append(Patch(facecolor=YEO7_COLORS.get(network, '#808080'), 
                                    alpha=0.3, label=network))
    
    ax.set_xlabel('Network Nodes', fontsize=12)
    ax.set_ylabel('Node Participation (Mean ± SD)', fontsize=12)
    
    # Update title based on number of datasets
    if len(node_participation_dfs) == 1:
        title = 'Node Participation in Maximal Cliques by Yeo-7 Network'
    else:
        title = f'Node Participation in Maximal Cliques by Yeo-7 Network ({len(node_participation_dfs)} Datasets)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'node_participation_by_network.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main(clique_measures_path: str, node_participation_paths: List[str], 
         output_dir: str, labels: Optional[List[str]] = None, show_plots: bool = False):
    """Main function to generate node participation visualization.
    
    Args:
        clique_measures_path: Path to clique measures file.
        node_participation_paths: List of paths to node participation files.
        output_dir: Directory to save plots.
        labels: List of labels for each dataset. If None, uses filenames.
        show_plots: Whether to display plots interactively.
    """
    print("="*80)
    print("Node Participation Visualization")
    print("="*80)
    
    # Load node participation data
    node_participation_dfs = load_node_participation_files(node_participation_paths)
    
    # Validate consistency across DataFrames
    validate_node_participation_consistency(node_participation_dfs)
    
    # Load clique measures (only need one for network mapping)
    clique_measures = load_clique_measures(clique_measures_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Get node to Yeo-7 network mapping
    print("\nExtracting node to Yeo-7 network mapping...")
    node_yeo7_map = get_node_yeo7_mapping(clique_measures)
    print(f"  Mapped {len(node_yeo7_map)} nodes to Yeo-7 networks")
    
    # Generate labels if not provided
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(node_participation_paths))]
    elif len(labels) != len(node_participation_paths):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of input files ({len(node_participation_paths)})")
    
    # Generate node participation plot
    plot_node_participation(node_participation_dfs, labels, node_yeo7_map, output_path, show_plots)
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == "__main__":

    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_name = f'node_participation_visualization_{current_time}'
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'clique_visualizations', default_output_name)

    parser = argparse.ArgumentParser(
        description='Visualize node participation in maximal cliques',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset using CSV files
  python visualize_node_participation.py -c clique_measures.csv -n node_participation.csv
  
  # Multiple datasets with custom labels
  python visualize_node_participation.py -c clique_measures.csv -n dataset1.csv dataset2.csv -l "Expert" "Naive"
  
  # Using Parquet files with multiple datasets
  python visualize_node_participation.py -c clique_measures.parquet -n data1.parquet data2.parquet data3.parquet
  
  # Custom output directory and show plots
  python visualize_node_participation.py -c clique_measures.csv -n node_participation.csv -o ./viz --show
        """
    )
    
    parser.add_argument('-c', '--clique_measures', type=str, required=True,
                        help='Path to clique measures file (CSV or Parquet)')
    parser.add_argument('-n', '--node_participation', type=str, nargs='+', required=True,
                        help='Path(s) to node participation file(s) (CSV or Parquet). Multiple files will be plotted with different line styles.')
    parser.add_argument('-l', '--labels', type=str, nargs='+',
                        help='Labels for each dataset (optional). Must match the number of input files.')
    parser.add_argument('-o', '--output_dir', type=str, default=default_output_dir,
                        help='Output directory for plots (default: ../output/clique_visualizations/node_participation_visualization_<timestamp>)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: False)')
    
    args = parser.parse_args()
    
    main(
        clique_measures_path=args.clique_measures,
        node_participation_paths=args.node_participation,
        output_dir=args.output_dir,
        labels=args.labels,
        show_plots=args.show
    )
