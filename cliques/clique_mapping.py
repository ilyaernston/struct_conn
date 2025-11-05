import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional
import argparse
import os
import glob

from preprocessing import drop_cerebellum, connect_components, normalize


def detect_cliques(matrix: np.ndarray) -> pd.DataFrame:
    """Detect all maximal cliques and compute their properties.
    
    Args:
        matrix (np.ndarray): Adjacency matrix representing the graph.
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - clique_index: Index of the clique
            - nodes: List of nodes in the clique
            - clique_size: Number of nodes in the clique
            - clique_degree: Average degree of nodes in the clique
            - clique_betweenness: Average betweenness centrality of nodes in the clique
            - clique_group_betweenness: Group betweenness centrality of the clique
            - clique_group_closeness: Group closeness centrality of the clique
            - clique_group_degree: Group degree centrality of the clique
    """
    # Create networkx graph
    graph = nx.from_numpy_array(matrix)
    
    # Find maximal cliques
    maximal_cliques = list(nx.find_cliques(graph))
    
    # Compute graph-level metrics
    degree_dict = dict(graph.degree)  # type: ignore
    betweenness_dict = nx.betweenness_centrality(graph)

    
    # Prepare data for DataFrame
    clique_data = []
    
    for idx, clique in enumerate(maximal_cliques):
        clique_list = list(clique)
        
        # Clique degree: average degree of nodes in the clique
        clique_degree = float(np.mean([degree_dict[node] for node in clique_list]))  # type: ignore
        
        # Clique betweenness: average betweenness centrality
        clique_betweenness = float(np.mean([betweenness_dict[node] for node in clique_list]))  # type: ignore

        # Clique group centrality
        subgraph = graph.subgraph(clique_list)
        group_betweenness = nx.group_betweenness_centrality(graph, subgraph)
        group_closeness = nx.group_closeness_centrality(graph, subgraph)
        group_degree = nx.group_degree_centrality(graph, subgraph)
        
        
        clique_data.append({
            'clique_index': idx,
            'nodes': clique_list,
            'clique_size': len(clique_list),
            'clique_degree': clique_degree,
            'clique_betweenness': clique_betweenness,
            'clique_group_betweenness': group_betweenness,
            'clique_group_closeness': group_closeness,
            'clique_group_degree': group_degree
        })
    
    return pd.DataFrame(clique_data)


def map_cliques_to_regions(cliques_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Map cliques to Yeo-7/17 networks and anatomical regions.
    
    Args:
        cliques_df (pd.DataFrame): DataFrame from detect_cliques function.
        mapping_df (pd.DataFrame): Brain region mapping DataFrame with columns:
            ROIname, Lobe, Gyrus, Yeo_7network_name, Yeo_17network_name, Hemi.
            
    Returns:
        pd.DataFrame: Input DataFrame with additional columns:
            - yeo7_networks: List of Yeo-7 networks for nodes in clique
            - yeo7_primary: Most common Yeo-7 network in clique
            - yeo17_networks: List of Yeo-17 networks for nodes in clique
            - yeo17_primary: Most common Yeo-17 network in clique
            - gyrus_regions: List of gyrus regions for nodes in clique
            - gyrus_primary: Most common gyrus in clique
            - lobes: List of lobes for nodes in clique
            - lobe_primary: Most common lobe in clique
            - hemispheres: List of hemispheres for nodes in clique
    """
    yeo7_networks_list = []
    yeo7_primary_list = []
    yeo17_networks_list = []
    yeo17_primary_list = []
    gyrus_list = []
    gyrus_primary_list = []
    lobes_list = []
    lobe_primary_list = []
    hemispheres_list = []
    
    for _, row in cliques_df.iterrows():
        nodes = row['nodes']
        
        # Extract region information for each node
        yeo7_nets = []
        yeo17_nets = []
        gyrus_regs = []
        lobes = []
        hemis = []
        
        for node in nodes:
            if node < len(mapping_df):
                yeo7_nets.append(mapping_df.iloc[node]['Yeo_7network_name'])
                yeo17_nets.append(mapping_df.iloc[node]['Yeo_17network_name'])
                gyrus_value = str(mapping_df.iloc[node]['Gyrus'])
                gyrus_regs.append(gyrus_value.split(',')[0].strip())
                lobes.append(mapping_df.iloc[node]['Lobe'])
                hemis.append(mapping_df.iloc[node]['Hemi'])
        
        # Find most common (primary) regions
        yeo7_primary = Counter(yeo7_nets).most_common(1)[0][0] if yeo7_nets else 'Unknown'
        yeo17_primary = Counter(yeo17_nets).most_common(1)[0][0] if yeo17_nets else 'Unknown'
        gyrus_primary = Counter(gyrus_regs).most_common(1)[0][0] if gyrus_regs else 'Unknown'
        lobe_primary = Counter(lobes).most_common(1)[0][0] if lobes else 'Unknown'
        
        yeo7_networks_list.append(yeo7_nets)
        yeo7_primary_list.append(yeo7_primary)
        yeo17_networks_list.append(yeo17_nets)
        yeo17_primary_list.append(yeo17_primary)
        gyrus_list.append(gyrus_regs)
        gyrus_primary_list.append(gyrus_primary)
        lobes_list.append(lobes)
        lobe_primary_list.append(lobe_primary)
        hemispheres_list.append(hemis)
    
    # Add new columns to DataFrame
    result_df = cliques_df.copy()
    result_df['yeo7_networks'] = yeo7_networks_list
    result_df['yeo7_primary'] = yeo7_primary_list
    result_df['yeo17_networks'] = yeo17_networks_list
    result_df['yeo17_primary'] = yeo17_primary_list
    result_df['gyrus_regions'] = gyrus_list
    result_df['gyrus_primary'] = gyrus_primary_list
    result_df['lobes'] = lobes_list
    result_df['lobe_primary'] = lobe_primary_list
    result_df['hemispheres'] = hemispheres_list
    
    return result_df


def visualize(cliques_df: pd.DataFrame, output_dir: str, subject_id: str = '', 
              save_plots: bool = True, show_plots: bool = False):
    """Create visualizations for clique analysis.
    
    Generates 4 plots:
    1. Histogram of cliques per Yeo-7 network
    2. Histogram of cliques per gyrus (top 20)
    3. Average clique properties per network (6 metrics: size, degree, betweenness,
       group betweenness, group closeness, group degree)
    4. Average clique properties per gyrus (top 15, same 6 metrics)
    
    Args:
        cliques_df (pd.DataFrame): DataFrame from map_cliques_to_regions.
        output_dir (str): Directory to save plots.
        subject_id (str): Identifier for the subject (used in plot titles).
        save_plots (bool): Whether to save plots to files. Default True.
        show_plots (bool): Whether to display plots interactively. Default False.
    """
    output_path = Path(output_dir)
    if save_plots:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Histogram of number of cliques per Yeo-7 network
    fig, ax = plt.subplots(figsize=(12, 6))
    yeo7_counts = Counter(cliques_df['yeo7_primary'])
    yeo7_sorted = sorted(yeo7_counts.items(), key=lambda x: x[1], reverse=True)
    networks, counts = zip(*yeo7_sorted) if yeo7_sorted else ([], [])
    
    ax.bar(range(len(networks)), counts, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(networks)))
    ax.set_xticklabels(networks, rotation=45, ha='right')
    ax.set_xlabel('Yeo-7 Network', fontsize=12)
    ax.set_ylabel('Number of Cliques', fontsize=12)
    ax.set_title(f'Clique Distribution across Yeo-7 Networks{" - " + subject_id if subject_id else ""}', fontsize=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_path / f'cliques_per_yeo7_network{("_" + subject_id) if subject_id else ""}.png', dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Histogram of number of cliques per gyrus
    fig, ax = plt.subplots(figsize=(14, 6))
    gyrus_counts = Counter(cliques_df['gyrus_primary'])
    gyrus_sorted = sorted(gyrus_counts.items(), key=lambda x: x[1], reverse=True)[:20]  # Top 20
    gyri, counts = zip(*gyrus_sorted) if gyrus_sorted else ([], [])
    
    ax.bar(range(len(gyri)), counts, color='coral', alpha=0.7)
    ax.set_xticks(range(len(gyri)))
    ax.set_xticklabels(gyri, rotation=45, ha='right')
    ax.set_xlabel('Gyrus (Top 20)', fontsize=12)
    ax.set_ylabel('Number of Cliques', fontsize=12)
    ax.set_title(f'Clique Distribution across Gyri{" - " + subject_id if subject_id else ""}', fontsize=14)
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_path / f'cliques_per_gyrus{("_" + subject_id) if subject_id else ""}.png', dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 3. Average clique properties per Yeo-7 network
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # Group by Yeo-7 network and compute averages
    network_stats = cliques_df.groupby('yeo7_primary').agg({
        'clique_size': 'mean',
        'clique_degree': 'mean',
        'clique_betweenness': 'mean',
        'clique_group_betweenness': 'mean',
        'clique_group_closeness': 'mean',
        'clique_group_degree': 'mean'
    }).reset_index()
    
    network_stats = network_stats.sort_values('clique_size', ascending=False)
    
    # Plot 1: Average clique size
    axes[0, 0].bar(range(len(network_stats)), network_stats['clique_size'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(network_stats)))
    axes[0, 0].set_xticklabels(network_stats['yeo7_primary'], rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Average Clique Size', fontsize=10)
    axes[0, 0].set_title('Average Clique Size per Network', fontsize=11)
    
    # Plot 2: Average clique degree
    axes[0, 1].bar(range(len(network_stats)), network_stats['clique_degree'], color='forestgreen', alpha=0.7)
    axes[0, 1].set_xticks(range(len(network_stats)))
    axes[0, 1].set_xticklabels(network_stats['yeo7_primary'], rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylabel('Average Clique Degree', fontsize=10)
    axes[0, 1].set_title('Average Clique Degree per Network', fontsize=11)
    
    # Plot 3: Average clique betweenness
    axes[1, 0].bar(range(len(network_stats)), network_stats['clique_betweenness'], color='darkorange', alpha=0.7)
    axes[1, 0].set_xticks(range(len(network_stats)))
    axes[1, 0].set_xticklabels(network_stats['yeo7_primary'], rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_ylabel('Average Betweenness', fontsize=10)
    axes[1, 0].set_title('Average Betweenness Centrality per Network', fontsize=11)
    
    # Plot 4: Average group betweenness
    axes[1, 1].bar(range(len(network_stats)), network_stats['clique_group_betweenness'], color='purple', alpha=0.7)
    axes[1, 1].set_xticks(range(len(network_stats)))
    axes[1, 1].set_xticklabels(network_stats['yeo7_primary'], rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Average Group Betweenness', fontsize=10)
    axes[1, 1].set_title('Average Group Betweenness per Network', fontsize=11)
    
    # Plot 5: Average group closeness
    axes[2, 0].bar(range(len(network_stats)), network_stats['clique_group_closeness'], color='teal', alpha=0.7)
    axes[2, 0].set_xticks(range(len(network_stats)))
    axes[2, 0].set_xticklabels(network_stats['yeo7_primary'], rotation=45, ha='right', fontsize=9)
    axes[2, 0].set_ylabel('Average Group Closeness', fontsize=10)
    axes[2, 0].set_title('Average Group Closeness per Network', fontsize=11)
    
    # Plot 6: Average group degree
    axes[2, 1].bar(range(len(network_stats)), network_stats['clique_group_degree'], color='crimson', alpha=0.7)
    axes[2, 1].set_xticks(range(len(network_stats)))
    axes[2, 1].set_xticklabels(network_stats['yeo7_primary'], rotation=45, ha='right', fontsize=9)
    axes[2, 1].set_ylabel('Average Group Degree', fontsize=10)
    axes[2, 1].set_title('Average Group Degree per Network', fontsize=11)
    
    plt.suptitle(f'Average Clique Properties per Yeo-7 Network{" - " + subject_id if subject_id else ""}', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_path / f'avg_clique_properties_per_network{("_" + subject_id) if subject_id else ""}.png', dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 4. Average clique properties per gyrus (top 15)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    gyrus_stats = cliques_df.groupby('gyrus_primary').agg({
        'clique_size': 'mean',
        'clique_degree': 'mean',
        'clique_betweenness': 'mean',
        'clique_group_betweenness': 'mean',
        'clique_group_closeness': 'mean',
        'clique_group_degree': 'mean'
    }).reset_index()
    
    gyrus_stats = gyrus_stats.nlargest(15, 'clique_size')
    
    # Plot 1: Average clique size
    axes[0, 0].barh(range(len(gyrus_stats)), gyrus_stats['clique_size'], color='steelblue', alpha=0.7)
    axes[0, 0].set_yticks(range(len(gyrus_stats)))
    axes[0, 0].set_yticklabels(gyrus_stats['gyrus_primary'], fontsize=8)
    axes[0, 0].set_xlabel('Average Clique Size', fontsize=10)
    axes[0, 0].set_title('Average Clique Size per Gyrus (Top 15)', fontsize=11)
    axes[0, 0].invert_yaxis()
    
    # Plot 2: Average clique degree
    axes[0, 1].barh(range(len(gyrus_stats)), gyrus_stats['clique_degree'], color='forestgreen', alpha=0.7)
    axes[0, 1].set_yticks(range(len(gyrus_stats)))
    axes[0, 1].set_yticklabels(gyrus_stats['gyrus_primary'], fontsize=8)
    axes[0, 1].set_xlabel('Average Clique Degree', fontsize=10)
    axes[0, 1].set_title('Average Clique Degree per Gyrus (Top 15)', fontsize=11)
    axes[0, 1].invert_yaxis()
    
    # Plot 3: Average clique betweenness
    axes[1, 0].barh(range(len(gyrus_stats)), gyrus_stats['clique_betweenness'], color='darkorange', alpha=0.7)
    axes[1, 0].set_yticks(range(len(gyrus_stats)))
    axes[1, 0].set_yticklabels(gyrus_stats['gyrus_primary'], fontsize=8)
    axes[1, 0].set_xlabel('Average Betweenness', fontsize=10)
    axes[1, 0].set_title('Average Betweenness Centrality per Gyrus (Top 15)', fontsize=11)
    axes[1, 0].invert_yaxis()
    
    # Plot 4: Average group betweenness
    axes[1, 1].barh(range(len(gyrus_stats)), gyrus_stats['clique_group_betweenness'], color='purple', alpha=0.7)
    axes[1, 1].set_yticks(range(len(gyrus_stats)))
    axes[1, 1].set_yticklabels(gyrus_stats['gyrus_primary'], fontsize=8)
    axes[1, 1].set_xlabel('Average Group Betweenness', fontsize=10)
    axes[1, 1].set_title('Average Group Betweenness per Gyrus (Top 15)', fontsize=11)
    axes[1, 1].invert_yaxis()
    
    # Plot 5: Average group closeness
    axes[2, 0].barh(range(len(gyrus_stats)), gyrus_stats['clique_group_closeness'], color='teal', alpha=0.7)
    axes[2, 0].set_yticks(range(len(gyrus_stats)))
    axes[2, 0].set_yticklabels(gyrus_stats['gyrus_primary'], fontsize=8)
    axes[2, 0].set_xlabel('Average Group Closeness', fontsize=10)
    axes[2, 0].set_title('Average Group Closeness per Gyrus (Top 15)', fontsize=11)
    axes[2, 0].invert_yaxis()
    
    # Plot 6: Average group degree
    axes[2, 1].barh(range(len(gyrus_stats)), gyrus_stats['clique_group_degree'], color='crimson', alpha=0.7)
    axes[2, 1].set_yticks(range(len(gyrus_stats)))
    axes[2, 1].set_yticklabels(gyrus_stats['gyrus_primary'], fontsize=8)
    axes[2, 1].set_xlabel('Average Group Degree', fontsize=10)
    axes[2, 1].set_title('Average Group Degree per Gyrus (Top 15)', fontsize=11)
    axes[2, 1].invert_yaxis()
    
    plt.suptitle(f'Average Clique Properties per Gyrus{" - " + subject_id if subject_id else ""}', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    if save_plots:
        plt.savefig(output_path / f'avg_clique_properties_per_gyrus{("_" + subject_id) if subject_id else ""}.png', dpi=300)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    if save_plots:
        print(f"Visualizations saved to {output_path}")
    if show_plots:
        print(f"Visualizations displayed")


def analyze_single_matrix(matrix: np.ndarray, mapping_df: pd.DataFrame, 
                          output_dir: str, subject_id: str = '', 
                          save_plots: bool = True, show_plots: bool = False) -> pd.DataFrame:
    """Perform complete clique analysis on a single connectivity matrix.
    
    Args:
        matrix (np.ndarray): Connectivity matrix.
        mapping_df (pd.DataFrame): Brain region mapping DataFrame.
        output_dir (str): Directory to save outputs.
        subject_id (str): Identifier for the subject.
        save_plots (bool): Whether to save plots to files. Default True.
        show_plots (bool): Whether to display plots interactively. Default False.
        
    Returns:
        pd.DataFrame: Complete clique analysis results.
    """
    print(f"Analyzing matrix{' for ' + subject_id if subject_id else ''}...")
    
    # Detect cliques and compute properties
    cliques_df = detect_cliques(matrix)
    print(f"  Found {len(cliques_df)} maximal cliques")
    print(f"  Average clique size: {cliques_df['clique_size'].mean():.4f}")
    print(f"  Average clique degree: {cliques_df['clique_degree'].mean():.4f}")
    print(f"  Average clique betweenness: {cliques_df['clique_betweenness'].mean():.4f}")
    print(f"  Average clique group betweenness: {cliques_df['clique_group_betweenness'].mean():.4f}")
    
    # Map cliques to regions
    cliques_with_regions = map_cliques_to_regions(cliques_df, mapping_df)
    print(f"  Mapped cliques to brain regions")
    
    # Create visualizations
    visualize(cliques_with_regions, output_dir, subject_id, save_plots, show_plots)
    
    # Save results to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f'clique_analysis_results{("_" + subject_id) if subject_id else ""}.csv'
    
    # Flatten lists for CSV export
    export_df = cliques_with_regions.copy()
    export_df['nodes'] = export_df['nodes'].apply(lambda x: ','.join(map(str, x)))
    export_df['yeo7_networks'] = export_df['yeo7_networks'].apply(lambda x: ','.join(x))
    export_df['yeo17_networks'] = export_df['yeo17_networks'].apply(lambda x: ','.join(x))
    export_df['gyrus_regions'] = export_df['gyrus_regions'].apply(lambda x: ','.join(x))
    export_df['lobes'] = export_df['lobes'].apply(lambda x: ','.join(x))
    export_df['hemispheres'] = export_df['hemispheres'].apply(lambda x: ','.join(x))
    
    export_df.to_csv(csv_path, index=False)
    print(f"  Results saved to {csv_path}")
    
    return cliques_with_regions


def main(connectivity_files: List[str], mapping_file: str, output_base_dir: str,
         save_plots: bool = True, show_plots: bool = False):
    """Main function to run clique analysis on multiple connectivity matrices.
    
    Args:
        connectivity_files (List[str]): List of paths to connectivity matrix files.
        mapping_file (str): Path to mapping CSV file.
        output_base_dir (str): Base directory for output files.
        save_plots (bool): Whether to save plots to files. Default True.
        show_plots (bool): Whether to display plots interactively. Default False.
    """
    # Load brain region mapping
    print(f"Loading brain region mapping from {mapping_file}...")
    mapping_df = pd.read_csv(mapping_file)
    print(f"  Loaded mapping for {len(mapping_df)} regions")
    
    # Analyze each connectivity matrix
    all_results = []
    
    for conn_file in connectivity_files:
        # Extract subject ID from filename
        subject_id = Path(conn_file).stem
        
        # Load connectivity matrix
        print(f"\nLoading connectivity matrix: {conn_file}")
        matrix = np.loadtxt(conn_file, delimiter=',')
        print(f"  Matrix shape: {matrix.shape}")

        # Preprocess matrix
        print(f"Preprocessing matrix...")
        matrix = drop_cerebellum(matrix, mapping_df)
        matrix = connect_components(matrix, mapping_df)
        matrix = normalize(matrix) #type: ignore
        print(f"  Preprocessed matrix shape: {matrix.shape}")
        g = nx.from_numpy_array(matrix)
        components = nx.number_connected_components(g)
        print(f"  Connected components: {components}")
        print(f"  Min value: {matrix.min()}, Max value: {matrix.max()}")
        del g, components

        # Create subject-specific output directory
        output_dir = Path(output_base_dir) / subject_id
        
        # Run analysis
        results = analyze_single_matrix(matrix, mapping_df, str(output_dir), subject_id, 
                                       save_plots, show_plots)
        
        # Add subject ID to results
        results['subject_id'] = subject_id
        all_results.append(results)
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_path = Path(output_base_dir) / 'combined_clique_analysis_results.csv'
        
        # Flatten lists for CSV export
        export_combined = combined_results.copy()
        export_combined['nodes'] = export_combined['nodes'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
        export_combined['yeo7_networks'] = export_combined['yeo7_networks'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_combined['yeo17_networks'] = export_combined['yeo17_networks'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_combined['gyrus_regions'] = export_combined['gyrus_regions'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_combined['lobes'] = export_combined['lobes'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_combined['hemispheres'] = export_combined['hemispheres'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        
        export_combined.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")
        print(f"Total cliques analyzed: {len(combined_results)}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Determine default data directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'dev_connectomes', 'expert')
    default_mapping_file = os.path.join(os.path.dirname(script_dir), 'data', 'mapping.csv')

    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    data_folder_name = os.path.basename(default_data_dir)
    default_output_name = f'clique_analysis_{data_folder_name}_{current_time}'
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', default_output_name)

    parser = argparse.ArgumentParser(description='Clique Analysis for Structural Connectivity Networks')
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='Directory containing connectivity matrix files (.csv or .npy)')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Output directory for results and visualizations')
    parser.add_argument('--mapping_file', type=str, default=default_mapping_file,
                        help='Path to brain region mapping CSV file')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='File pattern to match (e.g., "*.csv", "*.npy")')
    parser.add_argument('--show_plots', action='store_true',
                        help='Whether to display plots interactively')
    parser.add_argument('--save_plots', action='store_true',
                        help='Whether to save plots to files')
    
    args = parser.parse_args()
    
    # Find all connectivity files matching pattern
    connectivity_files = glob.glob(os.path.join(args.data_dir, args.pattern))
    
    if connectivity_files:
        print(f"Found {len(connectivity_files)} connectivity files in {args.data_dir}")
        print(f"Output will be saved to: {args.output_dir}")
        print(f"Save plots: {args.save_plots}, Show plots: {args.show_plots}")
        main(connectivity_files, args.mapping_file, args.output_dir, 
             args.save_plots, args.show_plots)
    else:
        print(f"No connectivity files found matching pattern '{args.pattern}' in {args.data_dir}")
        print("Please check the data directory and pattern.")
