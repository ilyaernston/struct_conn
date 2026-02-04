"""
Correlation analysis between node loop participation and clique participation.

This script computes subject-wise Spearman correlations between node participation
in loops and cliques, applies Fisher z-transformation, and tests group-level effects.

For each subject:
1. Calculate Spearman correlation between n_loops and n_cliques across all nodes
2. Apply Fisher z-transformation to correlation coefficient
3. Aggregate across subjects and test if mean z differs from 0
4. Report group-level effect size (Cohen's d)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse
from typing import Tuple, Dict, List, Optional
import os


# Yeo-7 network color scheme (matching visualize_node_participation.py)
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


def load_node_network_mapping(mapping_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Load node to Yeo-7 network and gyrus mapping from mapping CSV.
    
    Args:
        mapping_path: Path to mapping CSV file with columns [node_id, yeo7_network, gyrus] or similar.
        
    Returns:
        Tuple of (node_network_map, node_gyrus_map) dictionaries.
    """
    mapping_df = pd.read_csv(mapping_path)
    
    # Try to find the appropriate columns
    # Common column names for node ID
    node_col = None
    for col in ['Label']:
        if col in mapping_df.columns:
            node_col = col
            break
    
    # Common column names for network
    network_col = None
    for col in ['Yeo_7network_name']:
        if col in mapping_df.columns:
            network_col = col
            break
    
    # Column for gyrus
    gyrus_col = None
    for col in ['Gyrus']:
        if col in mapping_df.columns:
            gyrus_col = col
            break
    
    if node_col is None or network_col is None:
        raise ValueError(f"Could not find node ID and network columns in mapping file. Available columns: {list(mapping_df.columns)}")
    
    # Create mapping dictionaries
    # Node IDs in the mapping file start from 1, but in data they start from 0
    # So we subtract 1 from the Label column
    if node_col == 'Label':
        node_network_map = dict(zip(mapping_df[node_col] - 1, mapping_df[network_col]))
        if gyrus_col:
            # Clean up gyrus names (strip whitespace)
            gyrus_cleaned = mapping_df[gyrus_col].str.strip()
            node_gyrus_map = dict(zip(mapping_df[node_col] - 1, gyrus_cleaned))
        else:
            node_gyrus_map = {}
    else:
        node_network_map = dict(zip(mapping_df[node_col], mapping_df[network_col]))
        if gyrus_col:
            gyrus_cleaned = mapping_df[gyrus_col].str.strip()
            node_gyrus_map = dict(zip(mapping_df[node_col], gyrus_cleaned))
        else:
            node_gyrus_map = {}
    
    print(f"\nLoaded node mappings from {mapping_path}")
    print(f"  - Mapped {len(node_network_map)} nodes to Yeo-7 networks")
    if node_gyrus_map:
        print(f"  - Mapped {len(node_gyrus_map)} nodes to gyri")
        print(f"  - Found {len(set(node_gyrus_map.values()))} unique gyri")
    print(f"  - Using columns: {node_col} -> {network_col}" + (f", {gyrus_col}" if gyrus_col else ""))
    if node_col == 'Label':
        print(f"  - Adjusted node IDs (Label - 1) to match 0-indexed data")
    
    return node_network_map, node_gyrus_map


def load_participation_data(node_participation_path: str, 
                           loop_participation_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load node participation and loop participation data.
    
    Args:
        node_participation_path: Path to node (clique) participation file (CSV or Parquet).
        loop_participation_path: Path to loop participation file (CSV or Parquet).
        
    Returns:
        Tuple of (node_participation_df, loop_participation_df)
    """
    # Load node (clique) participation
    if node_participation_path.endswith('.parquet'):
        node_df = pd.read_parquet(node_participation_path)
    else:
        node_df = pd.read_csv(node_participation_path)
    
    # Load loop participation
    if loop_participation_path.endswith('.parquet'):
        loop_df = pd.read_parquet(loop_participation_path)
    else:
        loop_df = pd.read_csv(loop_participation_path)
    
    # Validate required columns
    if not {'subject_id', 'node_id', 'n_cliques'}.issubset(node_df.columns):
        raise ValueError(f"node_participation file missing required columns. Expected ['subject_id', 'node_id', 'n_cliques'], got {list(node_df.columns)}")
    
    if not {'subject_id', 'node_id', 'n_loops'}.issubset(loop_df.columns):
        raise ValueError(f"loop_participation file missing required columns. Expected ['subject_id', 'node_id', 'n_loops'], got {list(loop_df.columns)}")
    
    print(f"Loaded node participation data:")
    print(f"  - {len(node_df['subject_id'].unique())} subjects")
    print(f"  - {len(node_df['node_id'].unique())} unique nodes")
    print(f"  - Total records: {len(node_df)}")
    
    print(f"\nLoaded loop participation data:")
    print(f"  - {len(loop_df['subject_id'].unique())} subjects")
    print(f"  - {len(loop_df['node_id'].unique())} unique nodes")
    print(f"  - Total records: {len(loop_df)}")
    
    return node_df, loop_df


def validate_data_consistency(node_df: pd.DataFrame, loop_df: pd.DataFrame) -> None:
    """Validate that both datasets have the same subjects and nodes.
    
    Args:
        node_df: Node participation DataFrame.
        loop_df: Loop participation DataFrame.
        
    Raises:
        ValueError: If subjects or nodes don't match between datasets.
    """
    node_subjects = set(node_df['subject_id'].unique())
    loop_subjects = set(loop_df['subject_id'].unique())
    
    if node_subjects != loop_subjects:
        missing_in_loop = node_subjects - loop_subjects
        extra_in_loop = loop_subjects - node_subjects
        error_msg = "Subject mismatch between datasets:"
        if missing_in_loop:
            error_msg += f"\n  Subjects in node data but not loop data: {sorted(list(missing_in_loop))[:5]}{'...' if len(missing_in_loop) > 5 else ''}"
        if extra_in_loop:
            error_msg += f"\n  Subjects in loop data but not node data: {sorted(list(extra_in_loop))[:5]}{'...' if len(extra_in_loop) > 5 else ''}"
        raise ValueError(error_msg)
    
    node_nodes = set(node_df['node_id'].unique())
    loop_nodes = set(loop_df['node_id'].unique())
    
    if node_nodes != loop_nodes:
        missing_in_loop = node_nodes - loop_nodes
        extra_in_loop = loop_nodes - node_nodes
        error_msg = "Node mismatch between datasets:"
        if missing_in_loop:
            error_msg += f"\n  Nodes in node data but not loop data: {sorted(list(missing_in_loop))[:5]}{'...' if len(missing_in_loop) > 5 else ''}"
        if extra_in_loop:
            error_msg += f"\n  Nodes in loop data but not node data: {sorted(list(extra_in_loop))[:5]}{'...' if len(extra_in_loop) > 5 else ''}"
        raise ValueError(error_msg)
    
    print(f"\n✓ Validation passed: Both datasets have {len(node_subjects)} subjects and {len(node_nodes)} nodes")


def calculate_subject_correlations(node_df: pd.DataFrame, 
                                   loop_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlation for each subject between n_cliques and n_loops.
    
    Args:
        node_df: Node participation DataFrame with columns [subject_id, node_id, n_cliques].
        loop_df: Loop participation DataFrame with columns [subject_id, node_id, n_loops].
        
    Returns:
        DataFrame with columns [subject_id, r, z, n_nodes] where:
            - r: Spearman correlation coefficient
            - z: Fisher z-transformed correlation
            - n_nodes: Number of nodes used in correlation
    """
    # Merge datasets on subject_id and node_id
    merged = pd.merge(
        node_df[['subject_id', 'node_id', 'n_cliques']],
        loop_df[['subject_id', 'node_id', 'n_loops']],
        on=['subject_id', 'node_id'],
        how='inner'
    )
    
    print(f"\nMerged data: {len(merged)} records across {len(merged['subject_id'].unique())} subjects")
    
    # Calculate correlation for each subject
    results = []
    
    for subject_id in merged['subject_id'].unique():
        subject_data = merged[merged['subject_id'] == subject_id]
        
        # Get participation values
        n_cliques = subject_data['n_cliques'].values
        n_loops = subject_data['n_loops'].values
        n_nodes = len(subject_data)
        
        # Calculate Pearson correlation
        if n_nodes < 3:
            print(f"  Warning: Subject {subject_id} has only {n_nodes} nodes, skipping")
            continue
        
        # Check for variance (constant values will cause correlation to fail)
        if np.std(n_cliques) == 0 or np.std(n_loops) == 0:
            print(f"  Warning: Subject {subject_id} has zero variance in one measure, skipping")
            continue
        
        r, p_value = stats.spearmanr(n_cliques, n_loops)
        
        # Apply Fisher z-transformation
        z = np.arctanh(r)
        
        results.append({
            'subject_id': subject_id,
            'r': r,
            'z': z,
            'p_value': p_value,
            'n_nodes': n_nodes
        })
    
    results_df = pd.DataFrame(results)
    print(f"\nCalculated correlations for {len(results_df)} subjects")
    print(f"  Mean r: {results_df['r'].mean():.3f} (SD: {results_df['r'].std():.3f})")
    print(f"  Median r: {results_df['r'].median():.3f}")
    print(f"  Range r: [{results_df['r'].min():.3f}, {results_df['r'].max():.3f}]")
    
    return results_df


def calculate_node_correlations(node_df: pd.DataFrame, 
                                loop_df: pd.DataFrame,
                                node_network_map: Dict[int, str],
                                node_gyrus_map: Dict[int, str]) -> pd.DataFrame:
    """Calculate Spearman correlation for each node between n_cliques and n_loops across subjects.
    
    Args:
        node_df: Node participation DataFrame with columns [subject_id, node_id, n_cliques].
        loop_df: Loop participation DataFrame with columns [subject_id, node_id, n_loops].
        node_network_map: Dictionary mapping node_id to Yeo-7 network.
        node_gyrus_map: Dictionary mapping node_id to gyrus.
        
    Returns:
        DataFrame with columns [node_id, r, z, p_value, n_subjects, yeo7_network, gyrus] where:
            - r: Spearman correlation coefficient
            - z: Fisher z-transformed correlation
            - p_value: p-value for correlation
            - n_subjects: Number of subjects used in correlation
            - yeo7_network: Yeo-7 network assignment
            - gyrus: Gyrus assignment
    """
    # Merge datasets on subject_id and node_id
    merged = pd.merge(
        node_df[['subject_id', 'node_id', 'n_cliques']],
        loop_df[['subject_id', 'node_id', 'n_loops']],
        on=['subject_id', 'node_id'],
        how='inner'
    )
    
    print(f"\nCalculating per-node correlations...")
    
    # Calculate correlation for each node
    results = []
    
    for node_id in merged['node_id'].unique():
        node_data = merged[merged['node_id'] == node_id]
        
        # Get participation values across subjects
        n_cliques = node_data['n_cliques'].values
        n_loops = node_data['n_loops'].values
        n_subjects = len(node_data)
        
        # Calculate Pearson correlation
        if n_subjects < 3:
            print(f"  Warning: Node {node_id} has only {n_subjects} subjects, skipping")
            continue
        
        # Check for variance (constant values will cause correlation to fail)
        if np.std(n_cliques) == 0 or np.std(n_loops) == 0:
            # No variance - assign r=NaN
            r = np.nan
            p_value = np.nan
            z = np.nan
        else:
            r, p_value = stats.spearmanr(n_cliques, n_loops)
            # Apply Fisher z-transformation
            z = np.arctanh(r)
        
        # Get network and gyrus assignment
        yeo7_network = node_network_map.get(node_id, 'Unknown')
        gyrus = node_gyrus_map.get(node_id, 'Unknown') if node_gyrus_map else 'Unknown'
        
        results.append({
            'node_id': node_id,
            'r': r,
            'z': z,
            'p_value': p_value,
            'n_subjects': n_subjects,
            'yeo7_network': yeo7_network,
            'gyrus': gyrus
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by network and node_id for consistent ordering
    results_df = results_df.sort_values(['yeo7_network', 'node_id'])
    
    # Filter out NaN correlations for statistics
    valid_results = results_df.dropna(subset=['r'])
    
    print(f"\nCalculated correlations for {len(results_df)} nodes")
    print(f"  Valid correlations: {len(valid_results)}")
    print(f"  Mean r: {valid_results['r'].mean():.3f} (SD: {valid_results['r'].std():.3f})")
    print(f"  Median r: {valid_results['r'].median():.3f}")
    print(f"  Range r: [{valid_results['r'].min():.3f}, {valid_results['r'].max():.3f}]")
    
    return results_df


def test_node_correlation_variance(results_df: pd.DataFrame) -> Dict:
    """Analyze variance in per-node correlations.
    
    Args:
        results_df: DataFrame with per-node correlation results.
        
    Returns:
        Dictionary with variance statistics including:
            - n_nodes: Number of nodes analyzed
            - mean_r: Mean correlation across nodes
            - sd_r: Standard deviation of correlations
            - var_r: Variance of correlations
            - mean_z: Mean Fisher z across nodes
            - sd_z: Standard deviation of Fisher z
            - significant_nodes: Number of nodes with p < 0.05
            - positive_nodes: Number of nodes with r > 0
            - negative_nodes: Number of nodes with r < 0
    """
    # Filter out NaN values
    valid_results = results_df.dropna(subset=['r'])
    
    n_nodes = len(valid_results)
    mean_r = valid_results['r'].mean()
    sd_r = valid_results['r'].std()
    var_r = valid_results['r'].var()
    
    mean_z = valid_results['z'].mean()
    sd_z = valid_results['z'].std()
    
    significant_nodes = (valid_results['p_value'] < 0.05).sum()
    positive_nodes = (valid_results['r'] > 0).sum()
    negative_nodes = (valid_results['r'] < 0).sum()
    
    # Network-specific statistics
    network_stats = []
    for network in valid_results['yeo7_network'].unique():
        network_data = valid_results[valid_results['yeo7_network'] == network]
        network_stats.append({
            'network': network,
            'n_nodes': len(network_data),
            'mean_r': network_data['r'].mean(),
            'sd_r': network_data['r'].std()
        })
    
    results = {
        'n_nodes': n_nodes,
        'mean_r': mean_r,
        'sd_r': sd_r,
        'var_r': var_r,
        'mean_z': mean_z,
        'sd_z': sd_z,
        'significant_nodes': significant_nodes,
        'positive_nodes': positive_nodes,
        'negative_nodes': negative_nodes,
        'network_stats': pd.DataFrame(network_stats)
    }
    
    return results


def print_node_variance_report(variance_results: Dict) -> None:
    """Print formatted report for per-node correlation variance.
    
    Args:
        variance_results: Dictionary with variance results from test_node_correlation_variance.
    """
    print("\n" + "="*80)
    print("VARIANCE REPORT: Per-Node Correlation Analysis")
    print("="*80)
    
    print(f"\nNumber of Nodes: N = {variance_results['n_nodes']}")
    
    print("\nOverall Correlation Statistics:")
    print(f"  Mean r = {variance_results['mean_r']:.4f}")
    print(f"  SD r = {variance_results['sd_r']:.4f}")
    print(f"  Variance r = {variance_results['var_r']:.4f}")
    
    print("\nFisher z-transformed correlations:")
    print(f"  Mean z = {variance_results['mean_z']:.4f}")
    print(f"  SD z = {variance_results['sd_z']:.4f}")
    
    print("\nDistribution of Correlation Signs:")
    print(f"  Positive correlations (r > 0): {variance_results['positive_nodes']} nodes ({100*variance_results['positive_nodes']/variance_results['n_nodes']:.1f}%)")
    print(f"  Negative correlations (r < 0): {variance_results['negative_nodes']} nodes ({100*variance_results['negative_nodes']/variance_results['n_nodes']:.1f}%)")
    
    print(f"\nSignificant Correlations (p < 0.05): {variance_results['significant_nodes']} nodes ({100*variance_results['significant_nodes']/variance_results['n_nodes']:.1f}%)")
    
    print("\nNetwork-Specific Statistics:")
    network_df = variance_results['network_stats'].sort_values('mean_r', ascending=False)
    for _, row in network_df.iterrows():
        print(f"  {row['network']:20s}: n={row['n_nodes']:3.0f}, mean r={row['mean_r']:7.4f}, SD={row['sd_r']:6.4f}")
    
    print("="*80)


def plot_node_correlations(results_df: pd.DataFrame, 
                           variance_results: Dict,
                           output_dir: Path, 
                           show_plot: bool = False) -> None:
    """Create line plot of per-node correlations organized by Yeo-7 network.
    
    Args:
        results_df: DataFrame with per-node correlation results.
        variance_results: Dictionary with variance statistics.
        output_dir: Directory to save the plot.
        show_plot: Whether to display the plot interactively.
    """
    print("\nGenerating per-node correlation plot...")
    
    # Filter out NaN values for plotting
    plot_data = results_df.dropna(subset=['r']).copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create x-axis positions
    x_positions = np.arange(len(plot_data))
    
    # Get network order
    network_order = plot_data['yeo7_network'].tolist()
    
    # Add background color-coding by network
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
    
    # Plot correlation values as line with markers
    ax.plot(x_positions, plot_data['r'].values, 
           color='black', linewidth=2, linestyle='-',
           marker='o', markersize=3, alpha=0.8, zorder=3)
    
    # Add horizontal line at r=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)
    
    # Add horizontal line at mean r
    ax.axhline(y=variance_results['mean_r'], color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label=f"Mean r = {variance_results['mean_r']:.3f}", zorder=4)
    
    # Highlight significant correlations
    significant_mask = plot_data['p_value'] < 0.05
    if significant_mask.any():
        sig_positions = x_positions[significant_mask]
        sig_values = plot_data.loc[significant_mask, 'r'].values
        ax.scatter(sig_positions, sig_values, color='red', s=50, 
                  marker='o', alpha=0.6, zorder=5, label='p < 0.05')
    
    # Set x-axis (remove node IDs for cleaner look)
    ax.set_xticks([])
    ax.set_xlabel('Network Nodes', fontsize=12)
    
    # Set y-axis
    ax.set_ylabel('Spearman Correlation (r)', fontsize=12)
    ax.set_title('Per-Node Correlation between Clique and Loop Participation by Yeo-7 Network', 
                fontsize=14, fontweight='bold')
    
    # Create legend with network labels
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    unique_networks = list(dict.fromkeys(network_order))  # Preserve order
    
    # Build legend elements
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Correlation (r)'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', 
              label=f"Mean r = {variance_results['mean_r']:.3f}"),
    ]
    
    if significant_mask.any():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='red', markersize=8, 
                                     alpha=0.6, label='p < 0.05', linestyle='None'))
    
    # Add separator
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Add network colors
    for network in unique_networks:
        legend_elements.append(Patch(facecolor=YEO7_COLORS.get(network, '#808080'), 
                                    alpha=0.3, label=network))
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'node_correlations_by_network.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_node_correlations_by_gyrus(results_df: pd.DataFrame, 
                                     variance_results: Dict,
                                     output_dir: Path, 
                                     show_plot: bool = False) -> None:
    """Create line plot of per-node correlations organized by gyrus.
    
    Args:
        results_df: DataFrame with per-node correlation results (must include 'gyrus' column).
        variance_results: Dictionary with variance statistics.
        output_dir: Directory to save the plot.
        show_plot: Whether to display the plot interactively.
    """
    print("\nGenerating per-node correlation plot by gyrus...")
    
    # Check if gyrus column exists
    if 'gyrus' not in results_df.columns:
        print("  Warning: No gyrus information available. Skipping gyrus-based plot.")
        return
    
    # Filter out NaN values for plotting
    plot_data = results_df.dropna(subset=['r']).copy()
    
    # Sort by gyrus and node_id
    plot_data = plot_data.sort_values(['gyrus', 'node_id'])
    
    # Generate color palette for gyri
    unique_gyri = plot_data['gyrus'].unique()
    n_gyri = len(unique_gyri)
    
    # Use a colormap with good distinction
    cmap = plt.cm.get_cmap('tab20')
    if n_gyri > 20:
        cmap = plt.cm.get_cmap('hsv')
    
    gyrus_colors = {gyrus: cmap(i / n_gyri) for i, gyrus in enumerate(unique_gyri)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Create x-axis positions
    x_positions = np.arange(len(plot_data))
    
    # Get gyrus order
    gyrus_order = plot_data['gyrus'].tolist()
    
    # Add background color-coding by gyrus
    current_gyrus = gyrus_order[0]
    start_idx = 0
    
    for i, gyrus in enumerate(gyrus_order):
        if i == len(gyrus_order) - 1 or gyrus_order[i + 1] != current_gyrus:
            # End of current gyrus group
            end_idx = i + 1
            color = gyrus_colors.get(current_gyrus, '#808080')
            ax.axvspan(start_idx - 0.5, end_idx - 0.5, facecolor=color, alpha=0.15)
            
            # Add vertical line to separate gyri (except at the end)
            if i < len(gyrus_order) - 1:
                ax.axvline(x=end_idx - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            
            # Update for next gyrus
            if i < len(gyrus_order) - 1:
                current_gyrus = gyrus_order[i + 1]
                start_idx = i + 1
    
    # Plot correlation values as line with markers
    ax.plot(x_positions, plot_data['r'].values, 
           color='black', linewidth=2, linestyle='-',
           marker='o', markersize=3, alpha=0.8, zorder=3)
    
    # Add horizontal line at r=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)
    
    # Add horizontal line at mean r
    ax.axhline(y=variance_results['mean_r'], color='red', linestyle='--', 
              linewidth=2, alpha=0.7, label=f"Mean r = {variance_results['mean_r']:.3f}", zorder=4)
    
    # Highlight significant correlations
    significant_mask = plot_data['p_value'] < 0.05
    if significant_mask.any():
        sig_positions = x_positions[significant_mask]
        sig_values = plot_data.loc[significant_mask, 'r'].values
        ax.scatter(sig_positions, sig_values, color='red', s=50, 
                  marker='o', alpha=0.6, zorder=5, label='p < 0.05')
    
    # Set x-axis (remove node IDs for cleaner look)
    ax.set_xticks([])
    ax.set_xlabel('Nodes Grouped by Gyrus', fontsize=12)
    
    # Set y-axis
    ax.set_ylabel('Spearman Correlation (r)', fontsize=12)
    ax.set_title('Per-Node Correlation between Clique and Loop Participation by Gyrus', 
                fontsize=14, fontweight='bold')
    
    # Create legend - limit to most common gyri to avoid overcrowding
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Count nodes per gyrus
    gyrus_counts = plot_data['gyrus'].value_counts()
    top_gyri = gyrus_counts.head(15).index.tolist()  # Show top 15 gyri
    
    # Build legend elements
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Correlation (r)'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', 
              label=f"Mean r = {variance_results['mean_r']:.3f}"),
    ]
    
    if significant_mask.any():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='red', markersize=8, 
                                     alpha=0.6, label='p < 0.05', linestyle='None'))
    
    # Add separator
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    legend_elements.append(Line2D([0], [0], color='none', label=f'Top {len(top_gyri)} Gyri:'))
    
    # Add gyrus colors (top gyri only)
    for gyrus in top_gyri:
        legend_elements.append(Patch(facecolor=gyrus_colors.get(gyrus, '#808080'), 
                                    alpha=0.3, label=f"{gyrus} (n={gyrus_counts[gyrus]})"))
    
    if len(unique_gyri) > len(top_gyri):
        legend_elements.append(Line2D([0], [0], color='none', 
                                     label=f'... and {len(unique_gyri) - len(top_gyri)} more'))
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'node_correlations_by_gyrus.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def test_group_effect(results_df: pd.DataFrame) -> Dict:
    """Test whether mean Fisher z differs from 0 and calculate effect size.
    
    Args:
        results_df: DataFrame with subject-wise correlation results.
        
    Returns:
        Dictionary with test results including:
            - mean_z: Mean Fisher z
            - sd_z: Standard deviation of Fisher z
            - mean_r: Mean correlation (back-transformed from mean z)
            - t_stat: t-statistic
            - p_value: p-value from one-sample t-test
            - df: Degrees of freedom
            - cohens_d: Cohen's d effect size
            - ci_lower: Lower bound of 95% CI for mean z
            - ci_upper: Upper bound of 95% CI for mean z
    """
    z_values = results_df['z'].values
    n = len(z_values)
    
    # Calculate mean and SD of Fisher z
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values, ddof=1)
    
    # Back-transform mean z to correlation
    mean_r = np.tanh(mean_z)
    
    # One-sample t-test against 0
    t_stat, p_value = stats.ttest_1samp(z_values, 0)
    
    # Cohen's d effect size (for one-sample test)
    cohens_d = mean_z / sd_z
    
    # 95% confidence interval for mean z
    ci = stats.t.interval(0.95, df=n-1, loc=mean_z, scale=sd_z/np.sqrt(n))
    
    results = {
        'n_subjects': n,
        'mean_z': mean_z,
        'sd_z': sd_z,
        'mean_r': mean_r,
        't_stat': t_stat,
        'p_value': p_value,
        'df': n - 1,
        'cohens_d': cohens_d,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'ci_lower_r': np.tanh(ci[0]),
        'ci_upper_r': np.tanh(ci[1])
    }
    
    return results


def print_statistical_report(test_results: Dict) -> None:
    """Print formatted statistical report.
    
    Args:
        test_results: Dictionary with test results from test_group_effect.
    """
    print("\n" + "="*80)
    print("STATISTICAL REPORT: Group-Level Correlation Analysis")
    print("="*80)
    
    print(f"\nSample Size: N = {test_results['n_subjects']} subjects")
    
    print("\nFisher z-transformed correlations:")
    print(f"  Mean z = {test_results['mean_z']:.4f} (SD = {test_results['sd_z']:.4f})")
    print(f"  95% CI for z: [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]")
    
    print("\nBack-transformed to Spearman r:")
    print(f"  Mean r = {test_results['mean_r']:.4f}")
    print(f"  95% CI for r: [{test_results['ci_lower_r']:.4f}, {test_results['ci_upper_r']:.4f}]")
    
    print("\nOne-sample t-test (H0: mean z = 0):")
    print(f"  t({test_results['df']}) = {test_results['t_stat']:.4f}")
    print(f"  p = {test_results['p_value']:.6f}")
    
    if test_results['p_value'] < 0.001:
        sig_str = "p < 0.001 ***"
    elif test_results['p_value'] < 0.01:
        sig_str = "p < 0.01 **"
    elif test_results['p_value'] < 0.05:
        sig_str = "p < 0.05 *"
    else:
        sig_str = "p ≥ 0.05 (ns)"
    print(f"  Significance: {sig_str}")
    
    print("\nEffect Size:")
    print(f"  Cohen's d = {test_results['cohens_d']:.4f}")
    
    # Effect size interpretation
    d_abs = abs(test_results['cohens_d'])
    if d_abs < 0.2:
        effect_interp = "negligible"
    elif d_abs < 0.5:
        effect_interp = "small"
    elif d_abs < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"  Interpretation: {effect_interp} effect")
    
    print("\nConclusion:")
    if test_results['p_value'] < 0.05:
        print(f"  There is a significant correlation (r = {test_results['mean_r']:.3f}) between")
        print(f"  node participation in cliques and loops across subjects.")
    else:
        print(f"  There is no significant correlation between node participation")
        print(f"  in cliques and loops across subjects.")
    
    print("="*80)


def plot_correlation_results(results_df: pd.DataFrame, test_results: Dict, 
                             output_dir: Path, show_plot: bool = False) -> None:
    """Create visualization of correlation results.
    
    Args:
        results_df: DataFrame with subject-wise correlation results.
        test_results: Dictionary with group-level test results.
        output_dir: Directory to save plots.
        show_plot: Whether to display plots interactively.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Distribution of Fisher z values
    ax = axes[0, 0]
    ax.hist(results_df['z'], bins=20, edgecolor='black', alpha=0.6, color=plt.cm.plasma(1))
    ax.axvline(test_results['mean_z'], color='red', linestyle='--', linewidth=2, 
               label=f"Mean $z$ = {test_results['mean_z']:.3f}")
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='$z$ = 0')
    ax.set_xlabel('Fisher $z$', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Fisher z-transformed Correlations', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution of Spearman r values
    ax = axes[0, 1]
    ax.hist(results_df['r'], bins=20, edgecolor='black', alpha=0.8, color=plt.cm.plasma(4))
    ax.axvline(test_results['mean_r'], color='red', linestyle='--', linewidth=2, 
               label=f"Mean $\\rho$ = {test_results['mean_r']:.3f}")
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='$\\rho$ = 0')
    ax.set_xlabel(f'Spearman $\\rho$', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Spearman Correlations', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Q-Q plot for normality check
    ax = axes[1, 0]
    stats.probplot(results_df['z'], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Fisher z vs. Normal Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Subject-wise correlation values
    ax = axes[1, 1]
    subjects_sorted = results_df.sort_values('r')
    x_pos = np.arange(len(subjects_sorted))
    colors = ['red' if r < 0 else 'steelblue' for r in subjects_sorted['r']]
    ax.bar(x_pos, subjects_sorted['r'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(test_results['mean_r'], color='darkred', linestyle='--', linewidth=2, 
               label=f"Mean r = {test_results['mean_r']:.3f}")
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Subject (sorted by r)', fontsize=11)
    ax.set_ylabel(f'Spearman $\\rho$', fontsize=11)
    ax.set_title('Subject-wise Correlations (Sorted)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'correlation_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_results_csv(results_df: pd.DataFrame, test_results: Dict, output_dir: Path,
                    node_results_df: Optional[pd.DataFrame] = None, 
                    variance_results: Optional[Dict] = None) -> None:
    """Save detailed results to CSV files.
    
    Args:
        results_df: DataFrame with subject-wise correlation results.
        test_results: Dictionary with group-level test results.
        output_dir: Directory to save results.
        node_results_df: DataFrame with per-node correlation results (optional).
        variance_results: Dictionary with variance statistics (optional).
    """
    # Save subject-wise results
    subject_results_path = output_dir / 'subject_correlations.csv'
    results_df.to_csv(subject_results_path, index=False)
    print(f"Saved subject-wise results to {subject_results_path}")
    
    # Save group-level results
    group_results = pd.DataFrame([test_results])
    group_results_path = output_dir / 'group_statistics.csv'
    group_results.to_csv(group_results_path, index=False)
    print(f"Saved group-level statistics to {group_results_path}")
    
    # Save node-wise results if provided
    if node_results_df is not None:
        node_results_path = output_dir / 'node_correlations.csv'
        node_results_df.to_csv(node_results_path, index=False)
        print(f"Saved node-wise results to {node_results_path}")
    
    # Save variance results if provided
    if variance_results is not None:
        variance_summary = {
            'n_nodes': variance_results['n_nodes'],
            'mean_r': variance_results['mean_r'],
            'sd_r': variance_results['sd_r'],
            'var_r': variance_results['var_r'],
            'mean_z': variance_results['mean_z'],
            'sd_z': variance_results['sd_z'],
            'significant_nodes': variance_results['significant_nodes'],
            'positive_nodes': variance_results['positive_nodes'],
            'negative_nodes': variance_results['negative_nodes']
        }
        variance_df = pd.DataFrame([variance_summary])
        variance_path = output_dir / 'node_variance_statistics.csv'
        variance_df.to_csv(variance_path, index=False)
        print(f"Saved node variance statistics to {variance_path}")
        
        # Save network-specific statistics
        network_stats_path = output_dir / 'network_statistics.csv'
        variance_results['network_stats'].to_csv(network_stats_path, index=False)
        print(f"Saved network-specific statistics to {network_stats_path}")


def main(node_clique_participation_path: str, node_loop_participation_path: str, 
         output_dir: str, mapping_path: Optional[str] = None, show_plots: bool = False):
    """Main function for correlation analysis.
    
    Args:
        node_clique_participation_path: Path to node (clique) participation file.
        node_loop_participation_path: Path to node loop participation file.
        output_dir: Directory to save results and plots.
        mapping_path: Path to node-network mapping CSV (optional, for per-node analysis).
        show_plots: Whether to display plots interactively.
    """
    print("="*80)
    print("Node-Clique Participation Correlation Analysis")
    print("="*80)
    
    # Load data
    print("\nLoading participation data...")
    node_df, loop_df = load_participation_data(node_clique_participation_path, node_loop_participation_path)
    
    # Validate consistency
    print("\nValidating data consistency...")
    validate_data_consistency(node_df, loop_df)
    
    # Calculate subject-wise correlations
    print("\n" + "="*80)
    print("ANALYSIS 1: Subject-wise Correlations")
    print("="*80)
    print("\nCalculating subject-wise correlations...")
    results_df = calculate_subject_correlations(node_df, loop_df)
    
    # Test group-level effect
    print("\nTesting group-level effect...")
    test_results = test_group_effect(results_df)
    
    # Print statistical report
    print_statistical_report(test_results)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Generate subject-wise plots
    print("\nGenerating subject-wise visualizations...")
    plot_correlation_results(results_df, test_results, output_path, show_plots)
    
    # Per-node analysis (if mapping provided)
    node_results_df = None
    variance_results = None
    
    if mapping_path:
        print("\n" + "="*80)
        print("ANALYSIS 2: Per-Node Correlations")
        print("="*80)
        
        # Load node-network and node-gyrus mapping
        node_network_map, node_gyrus_map = load_node_network_mapping(mapping_path)
        
        # Calculate per-node correlations
        node_results_df = calculate_node_correlations(node_df, loop_df, node_network_map, node_gyrus_map)
        
        # Analyze variance
        print("\nAnalyzing variance in per-node correlations...")
        variance_results = test_node_correlation_variance(node_results_df)
        
        # Print variance report
        print_node_variance_report(variance_results)
        
        # Generate per-node plots
        print("\nGenerating per-node visualizations...")
        plot_node_correlations(node_results_df, variance_results, output_path, show_plots)
        plot_node_correlations_by_gyrus(node_results_df, variance_results, output_path, show_plots)
    else:
        print("\n" + "="*80)
        print("ANALYSIS 2: Per-Node Correlations - SKIPPED")
        print("="*80)
        print("\nTo run per-node analysis, provide --mapping argument with node-network mapping CSV")
    
    # Save all results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    save_results_csv(results_df, test_results, output_path, node_results_df, variance_results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_name = f'correlation_analysis_{current_time}'
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'clique_loop_correlation', default_output_name)
    default_mapping_file = os.path.join(os.path.dirname(script_dir), 'data', 'mapping.csv')

    parser = argparse.ArgumentParser(
        description='Correlation analysis between node participation in cliques and loops',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Pipeline:
1. SUBJECT-WISE ANALYSIS:
   - Load node participation (n_cliques) and loop participation (n_loops) data
   - For each subject, calculate Spearman correlation between n_cliques and n_loops across all nodes
   - Apply Fisher z-transformation to correlation coefficients
   - Test whether mean z differs from 0 using one-sample t-test
   - Calculate Cohen's d effect size
   
2. PER-NODE ANALYSIS (if --mapping provided):
   - For each node, calculate Spearman correlation between n_cliques and n_loops across all subjects
   - Apply Fisher z-transformation to correlation coefficients
   - Analyze variance in correlations across nodes
   - Generate visualizations organized by Yeo-7 network

Examples:
  # Subject-wise analysis only
  python correlate_node_clique_participation.py \\
      -n node_participation.csv \\
      -l node_loop_participation.csv
  
  # Both subject-wise and per-node analysis
  python correlate_node_clique_participation.py \\
      -n node_participation.csv \\
      -l node_loop_participation.csv \\
      -m mapping.csv
  
  # With custom output and interactive plots
  python correlate_node_clique_participation.py \\
      -n node_participation.csv \\
      -l node_loop_participation.csv \\
      -m mapping.csv \\
      -o ./correlation_results \\
      --show
        """
    )
    
    parser.add_argument('-c', '--node_clique_participation', type=str, required=True,
                        help='Path to node clique participation file with columns [subject_id, node_id, n_cliques]')
    parser.add_argument('-l', '--node_loop_participation', type=str, required=True,
                        help='Path to node loop participation file with columns [subject_id, node_id, n_loops]')
    parser.add_argument('-m', '--mapping', type=str, default=default_mapping_file,
                        help='Path to node-network mapping CSV with columns [Lable, Yeo_7network_name] (optional, for per-node analysis)')
    parser.add_argument('-o', '--output_dir', type=str, default=default_output_dir,
                        help='Output directory for results and plots (default: ../output/clique_loop_correlation/correlation_analysis_<timestamp>)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: False)')
    
    args = parser.parse_args()
    
    main(
        node_clique_participation_path=args.node_clique_participation,
        node_loop_participation_path=args.node_loop_participation,
        output_dir=args.output_dir,
        mapping_path=args.mapping,
        show_plots=args.show
    )
