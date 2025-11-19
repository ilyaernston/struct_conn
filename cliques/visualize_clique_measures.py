"""
Visualization script for clique measures analysis.

This script generates visualizations from clique_measures data:
1. Box plot of clique size by Yeo-7 network
2. Box plot of clique size by primary gyrus
3. Box plot of volume by Yeo-7 network
4. Box plot of volume by primary gyrus
5. Box plot of average degree by Yeo-7 network
6. Box plot of average degree by primary gyrus
7. Box plot of boundary edges by Yeo-7 network
8. Box plot of boundary edges by primary gyrus
9. Box plot of boundary ratio by Yeo-7 network
10. Box plot of boundary ratio by primary gyrus
11. Box plot of average embeddedness by Yeo-7 network
12. Box plot of average embeddedness by primary gyrus
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Optional, List
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


def main(clique_measures_path: str, output_dir: str, 
         metrics_to_plot: Optional[List[str]] = None, show_plots: bool = False):
    """Main function to generate all clique measures visualizations.
    
    Args:
        clique_measures_path: Path to clique measures file.
        output_dir: Directory to save plots.
        metrics_to_plot: List of metric names to plot. If None, plots all metrics.
        show_plots: Whether to display plots interactively.
    """
    print("="*80)
    print("Clique Measures Visualization")
    print("="*80)
    
    # Load data
    clique_measures = load_clique_measures(clique_measures_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Define all available metrics
    all_metrics = [
        ('clique_size', 'Clique Size'),
        ('clique_volume', 'Volume'),
        ('clique_avg_degree', 'Average Degree'),
        ('clique_boundary_edges', 'Boundary Edges'),
        ('clique_boundary_ratio', 'Boundary Ratio'),
        ('clique_avg_embeddedness', 'Average Embeddedness')
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
    default_output_name = f'clique_measures_visualization_{current_time}'
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'clique_visualizations', default_output_name)

    parser = argparse.ArgumentParser(
        description='Visualize clique measures analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using CSV files
  python visualize_clique_measures.py --clique_measures clique_measures.csv
  
  # Using Parquet files
  python visualize_clique_measures.py --clique_measures clique_measures.parquet
  
  # Custom output directory and show plots
  python visualize_clique_measures.py -c clique_measures.csv -o ./viz --show
  
  # Plot only specific metrics
  python visualize_clique_measures.py -c clique_measures.csv --metrics clique_size clique_conductance
  
  # Plot all metrics (default)
  python visualize_clique_measures.py -c clique_measures.csv

Available metrics:
  clique_size            - Clique Size
  clique_volume          - Volume
  clique_avg_degree      - Average Degree
  clique_boundary_edges  - Boundary Edges
  clique_boundary_ratio  - Boundary Ratio
  clique_avg_embeddedness - Average Embeddedness
        """
    )
    
    parser.add_argument('-c', '--clique_measures', type=str, required=True,
                        help='Path to clique measures file (CSV or Parquet)')
    parser.add_argument('-o', '--output_dir', type=str, default=default_output_dir,
                        help='Output directory for plots (default: ../output/clique_visualizations/clique_measures_visualization_<timestamp>)')
    parser.add_argument('--metrics', nargs='+',
                        help='Metrics to plot. Specify metric names to plot only those (default: all metrics). Available metrics: clique_size, clique_volume, clique_avg_degree, clique_boundary_edges, clique_boundary_ratio, clique_avg_embeddedness')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: False)')
    
    args = parser.parse_args()
    
    main(
        clique_measures_path=args.clique_measures,
        output_dir=args.output_dir,
        metrics_to_plot=args.metrics,
        show_plots=args.show
    )
