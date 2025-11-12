"""
Visualization script for clique mapping analysis results.

This script generates visualizations from clique_measures and node_participation data:
1. Line plot of node participation averaged across subjects with SD bands
2. Box plot of clique size by Yeo-7 network
3. Box plot of clique size by primary gyrus
4. Box plot of internal degree by Yeo-7 network
5. Box plot of internal degree by primary gyrus
6. Box plot of boundary degree by Yeo-7 network
7. Box plot of boundary degree by primary gyrus
8. Box plot of conductance by Yeo-7 network
9. Box plot of conductance by primary gyrus
10. Box plot of average external degree by Yeo-7 network
11. Box plot of average external degree by primary gyrus
12. Box plot of boundary ratio by Yeo-7 network
13. Box plot of boundary ratio by primary gyrus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional, List
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

# Lobe color scheme
LOBE_COLORS = {
    'Frontal Lobe': '#e74c3c',
    'Frontal Lobe ': '#e74c3c',  # With trailing space
    'Parietal Lobe': '#3498db',
    'Temporal Lobe': '#2ecc71',
    'Occipital Lobe': '#9b59b6',
    'Insular Lobe': '#f39c12',
    'Limbic Lobe': '#1abc9c',
    'Subcortical Nuclei': '#34495e',
    'Unknown': '#808080'
}


def load_data(clique_measures_path: str, node_participation_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load clique measures and node participation data.
    
    Args:
        clique_measures_path: Path to clique measures file (csv or parquet).
        node_participation_path: Path to node participation file (csv or parquet).
        
    Returns:
        Tuple of (clique_measures_df, node_participation_df)
    """
    # Determine file format and load
    if clique_measures_path.endswith('.parquet'):
        clique_measures = pd.read_parquet(clique_measures_path)
    else:
        clique_measures = pd.read_csv(clique_measures_path)
    
    if node_participation_path.endswith('.parquet'):
        node_participation = pd.read_parquet(node_participation_path)
    else:
        node_participation = pd.read_csv(node_participation_path)
    
    print(f"Loaded {len(clique_measures)} clique measures from {len(clique_measures['subject_id'].unique())} subjects")
    print(f"Loaded node participation data for {len(node_participation['node_id'].unique())} unique nodes")
    
    return clique_measures, node_participation


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


def plot_node_participation(node_participation: pd.DataFrame, node_yeo7_map: Dict[int, str], 
                            output_dir: Path, show_plot: bool = False):
    """Create line plot of node participation averaged across subjects.
    
    Args:
        node_participation: DataFrame with columns [subject_id, node_id, n_cliques].
        node_yeo7_map: Dictionary mapping node_id to Yeo-7 network.
        output_dir: Directory to save the plot.
        show_plot: Whether to display the plot interactively.
    """
    print("\nGenerating node participation plot...")
    
    # Calculate mean and std for each node across subjects
    node_stats = node_participation.groupby('node_id')['n_cliques'].agg(['mean', 'std']).reset_index()
    node_stats['std'] = node_stats['std'].fillna(0)  # Handle single subject case
    
    # Add Yeo-7 network mapping
    node_stats['yeo7_network'] = node_stats['node_id'].map(node_yeo7_map)
    node_stats['yeo7_network'] = node_stats['yeo7_network'].fillna('Unknown')
    
    # Sort by Yeo-7 network first, then by node_id within each network
    node_stats = node_stats.sort_values(['yeo7_network', 'node_id'])
    
    # Reset index for plotting
    node_stats = node_stats.reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create x-axis positions
    x_positions = np.arange(len(node_stats))
    
    # Add background color-coding by network
    current_network = node_stats['yeo7_network'].iloc[0]
    start_idx = 0
    
    for i, network in enumerate(node_stats['yeo7_network']):
        if i == len(node_stats) - 1 or node_stats['yeo7_network'].iloc[i + 1] != current_network:
            # End of current network group
            end_idx = i + 1
            color = YEO7_COLORS.get(current_network, '#808080')
            ax.axvspan(start_idx - 0.5, end_idx - 0.5, facecolor=color, alpha=0.15)
            
            # Add vertical line to separate networks (except at the end)
            if i < len(node_stats) - 1:
                ax.axvline(x=end_idx - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            
            # Update for next network
            if i < len(node_stats) - 1:
                current_network = node_stats['yeo7_network'].iloc[i + 1]
                start_idx = i + 1
    
    # Plot main line for mean node participation
    ax.plot(x_positions, node_stats['mean'], 
           color='black', linewidth=2, label='Mean', alpha=0.8, zorder=3)
    
    # Plot SD bands
    ax.fill_between(x_positions, 
                    node_stats['mean'] - node_stats['std'],
                    node_stats['mean'] + node_stats['std'],
                    color='gray', alpha=0.3, label='±1 SD', zorder=2)
    
    # Set x-axis labels (remove node IDs for cleaner look)
    ax.set_xticks([])
    
    # Create legend with network labels
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    unique_networks = list(dict.fromkeys(node_stats['yeo7_network'].tolist()))  # Preserve order
    
    # Build legend elements
    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, label='Mean'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.3, label='±1 SD'))
    
    for network in unique_networks:
        legend_elements.append(Patch(facecolor=YEO7_COLORS.get(network, '#808080'), alpha=0.3, label=network))
    
    ax.set_xlabel('Network Nodes', fontsize=12)
    ax.set_ylabel('Node Participation (Mean ± SD)', fontsize=12)
    ax.set_title('Node Participation in Maximal Cliques by Yeo-7 Network', fontsize=14, fontweight='bold')
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True, ncol=1)
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


def plot_metric_by_yeo7(clique_measures: pd.DataFrame, metric_col: str, metric_label: str,
                        output_dir: Path, show_plot: bool = False):
    """Create box plot of a clique metric by Yeo-7 network.
    
    For each subject and network, calculate the mean metric value.
    Then plot distribution across subjects.
    
    Args:
        clique_measures: DataFrame with clique measures.
        metric_col: Column name of the metric to plot.
        metric_label: Human-readable label for the metric.
        output_dir: Directory to save the plot.
        show_plot: Whether to display the plot interactively.
    """
    print(f"\nGenerating {metric_label} by Yeo-7 network plot...")
    
    # Calculate mean metric per subject and network
    subject_network_stats = clique_measures.groupby(['subject_id', 'yeo7_primary'])[metric_col].mean().reset_index()
    subject_network_stats.columns = ['subject_id', 'yeo7_primary', f'mean_{metric_col}']
    
    # Calculate overall statistics per network
    network_stats = subject_network_stats.groupby('yeo7_primary')[f'mean_{metric_col}'].agg(['mean', 'std']).reset_index()
    network_stats = network_stats.sort_values('mean', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for box plot
    networks = network_stats['yeo7_primary'].tolist()
    data_by_network = [subject_network_stats[subject_network_stats['yeo7_primary'] == net][f'mean_{metric_col}'].values 
                       for net in networks]
    
    # Create box plot
    bp = ax.boxplot(data_by_network, patch_artist=True,
                    showmeans=True, meanline=False,
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6))
    
    # Set x-axis labels
    ax.set_xticks(range(1, len(networks) + 1))
    ax.set_xticklabels(networks, rotation=45, ha='right')
    
    # Color boxes by network
    for patch, network in zip(bp['boxes'], networks):
        color = YEO7_COLORS.get(network, '#808080')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add error bars for SD
    positions = range(1, len(networks) + 1)
    for i, network in enumerate(networks):
        mean_val = network_stats[network_stats['yeo7_primary'] == network]['mean'].values[0]
        std_val = network_stats[network_stats['yeo7_primary'] == network]['std'].values[0]
        ax.errorbar(positions[i], mean_val, yerr=std_val, 
                   fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Yeo-7 Network', fontsize=12)
    ax.set_ylabel(f'Average {metric_label}', fontsize=12)
    ax.set_title(f'{metric_label} Distribution by Yeo-7 Network', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    filename = metric_col.replace('clique_', '').replace('_', '')
    # Special case for clique_size to maintain backward compatibility
    if metric_col == 'clique_size':
        output_path = output_dir / 'clique_size_by_yeo7_network.png'
    else:
        output_path = output_dir / f'{filename}_by_yeo7_network.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_metric_by_gyrus(clique_measures: pd.DataFrame, metric_col: str, metric_label: str,
                         output_dir: Path, show_plot: bool = False):
    """Create box plot of a clique metric by primary gyrus.
    
    For each subject and gyrus, calculate the mean metric value.
    Then plot distribution across subjects for all gyri, grouped by lobe.
    
    Args:
        clique_measures: DataFrame with clique measures.
        metric_col: Column name of the metric to plot.
        metric_label: Human-readable label for the metric.
        output_dir: Directory to save the plot.
        show_plot: Whether to display the plot interactively.
    """
    print(f"\nGenerating {metric_label} by primary gyrus plot...")
    
    # Step 1: For each gyrus, determine its unique lobe (most common)
    gyrus_to_lobe = clique_measures.groupby('gyrus_primary')['lobe_primary'].agg(
        lambda x: x.value_counts().index[0]
    ).to_dict()
    
    # Step 2: Calculate mean metric per subject and gyrus
    subject_gyrus_stats = clique_measures.groupby(['subject_id', 'gyrus_primary'])[metric_col].mean().reset_index()
    subject_gyrus_stats.columns = ['subject_id', 'gyrus_primary', f'mean_{metric_col}']
    
    # Add lobe mapping
    subject_gyrus_stats['lobe_primary'] = subject_gyrus_stats['gyrus_primary'].map(gyrus_to_lobe)
    
    # Step 3: Calculate mean and std across subjects for each gyrus
    gyrus_stats = subject_gyrus_stats.groupby(['gyrus_primary', 'lobe_primary'])[f'mean_{metric_col}'].agg(['mean', 'std']).reset_index()
    gyrus_stats['std'] = gyrus_stats['std'].fillna(0)  # Handle single subject case
    
    # Sort by lobe first, then by mean metric within each lobe (descending)
    gyrus_stats = gyrus_stats.sort_values(['lobe_primary', 'mean'], ascending=[True, False])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Prepare data for box plot
    gyri = gyrus_stats['gyrus_primary'].tolist()
    lobes = gyrus_stats['lobe_primary'].tolist()
    
    # For each gyrus, get all subject values
    data_by_gyrus = []
    for gyrus in gyri:
        gyrus_data = subject_gyrus_stats[subject_gyrus_stats['gyrus_primary'] == gyrus][f'mean_{metric_col}'].values
        data_by_gyrus.append(gyrus_data)
    
    # Create box plot
    bp = ax.boxplot(data_by_gyrus, patch_artist=True,
                    showmeans=True, meanline=False,
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6))
    
    # Set x-axis labels
    ax.set_xticks(range(1, len(gyri) + 1))
    ax.set_xticklabels(gyri, rotation=45, ha='right')
    
    # Color boxes by lobe
    for patch, lobe in zip(bp['boxes'], lobes):
        color = LOBE_COLORS.get(lobe, '#808080')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add vertical lines to separate lobe groups
    current_lobe = lobes[0]
    for i, lobe in enumerate(lobes[1:], start=1):
        if lobe != current_lobe:
            ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            current_lobe = lobe
    
    # Create legend for lobes
    from matplotlib.patches import Patch
    unique_lobes = list(dict.fromkeys(lobes))  # Preserve order, remove duplicates
    legend_elements = [Patch(facecolor=LOBE_COLORS.get(lobe, '#808080'), alpha=0.6, label=lobe) 
                      for lobe in unique_lobes]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    ax.set_xlabel('Primary Gyrus (Grouped by Lobe)', fontsize=12)
    ax.set_ylabel(f'Average {metric_label}', fontsize=12)
    ax.set_title(f'{metric_label} Distribution by Primary Gyrus', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    filename = metric_col.replace('clique_', '').replace('_', '')
    # Special case for clique_size to maintain backward compatibility
    if metric_col == 'clique_size':
        output_path = output_dir / 'clique_size_by_gyrus.png'
    else:
        output_path = output_dir / f'{filename}_by_gyrus.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main(clique_measures_path: str, node_participation_path: str, 
         output_dir: str, metrics_to_plot: Optional[List[str]] = None, show_plots: bool = False):
    """Main function to generate all visualizations.
    
    Args:
        clique_measures_path: Path to clique measures file.
        node_participation_path: Path to node participation file.
        output_dir: Directory to save plots.
        metrics_to_plot: List of metric names to plot. If None, plots all metrics.
        show_plots: Whether to display plots interactively.
    """
    print("="*80)
    print("Clique Mapping Results Visualization")
    print("="*80)
    
    # Load data
    clique_measures, node_participation = load_data(clique_measures_path, node_participation_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Get node to Yeo-7 network mapping
    print("\nExtracting node to Yeo-7 network mapping...")
    node_yeo7_map = get_node_yeo7_mapping(clique_measures)
    print(f"  Mapped {len(node_yeo7_map)} nodes to Yeo-7 networks")
    
    # Generate node participation plot
    plot_node_participation(node_participation, node_yeo7_map, output_path, show_plots)
    
    # Define all available metrics
    all_metrics = [
        ('clique_size', 'Clique Size'),
        ('clique_deg_in', 'Internal Degree'),
        ('clique_total_ext_deg', 'Boundary Degree'),
        ('clique_conductance', 'Conductance'),
        ('clique_avg_ext_deg', 'Average External Degree'),
        ('clique_bound_ratio', 'Boundary Ratio')
    ]
    
    # Filter metrics based on user selection
    if metrics_to_plot is None:
        metrics = all_metrics
        print(f"\nGenerating plots for all metrics...")
    else:
        # Filter to only requested metrics
        metrics = [(col, label) for col, label in all_metrics if col in metrics_to_plot]
        if not metrics:
            print(f"\nWarning: No valid metrics specified. Available metrics:")
            for col, label in all_metrics:
                print(f"  - {col}: {label}")
            return
        print(f"\nGenerating plots for selected metrics: {', '.join([label for _, label in metrics])}")
    
    # Generate plots for each metric
    for metric_col, metric_label in metrics:
        plot_metric_by_yeo7(clique_measures, metric_col, metric_label, output_path, show_plots)
        plot_metric_by_gyrus(clique_measures, metric_col, metric_label, output_path, show_plots)
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == "__main__":

    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_name = f'clique_visualization_{current_time}'
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'clique_visualizations', default_output_name)

    parser = argparse.ArgumentParser(
        description='Visualize clique mapping analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using CSV files
  python visualize_clique_mapping.py --clique_measures clique_measures.csv --node_participation node_participation.csv
  
  # Using Parquet files
  python visualize_clique_mapping.py --clique_measures clique_measures.parquet --node_participation node_participation.parquet
  
  # Custom output directory and show plots
  python visualize_clique_mapping.py -c clique_measures.csv -n node_participation.csv -o ./viz --show
  
  # Plot only specific metrics
  python visualize_clique_mapping.py -c clique_measures.csv -n node_participation.csv --metrics clique_size clique_conductance
  
  # Plot all metrics (default)
  python visualize_clique_mapping.py -c clique_measures.csv -n node_participation.csv

Available metrics:
  clique_size          - Clique Size
  clique_deg_in        - Internal Degree
  clique_total_ext_deg - Boundary Degree
  clique_conductance   - Conductance
  clique_avg_ext_deg   - Average External Degree
  clique_bound_ratio   - Boundary Ratio
        """
    )
    
    parser.add_argument('-c', '--clique_measures', type=str, required=True,
                        help='Path to clique measures file (CSV or Parquet)')
    parser.add_argument('-n', '--node_participation', type=str, required=True,
                        help='Path to node participation file (CSV or Parquet)')
    parser.add_argument('-o', '--output_dir', type=str, default=default_output_dir,
                        help='Output directory for plots (default: ../output/clique_visualizations/clique_visualization_<timestamp>)')
    parser.add_argument('--metrics', nargs='+',
                        help='Metrics to plot. Specify metric names to plot only those (default: all metrics). Available metrics: clique_size, clique_deg_in, clique_total_ext_deg, clique_conductance, clique_avg_ext_deg, clique_bound_ratio  ')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: False)')
    
    args = parser.parse_args()
    
    main(
        clique_measures_path=args.clique_measures,
        node_participation_path=args.node_participation,
        output_dir=args.output_dir,
        metrics_to_plot=args.metrics,
        show_plots=args.show
    )
