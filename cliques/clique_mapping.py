import numpy as np
import pandas as pd
from igraph import Graph

from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional
import argparse
import os
import glob
import time

from preprocessing import drop_cerebellum, connect_components, normalize


def compute_node_participation(cliques_df: pd.DataFrame, num_nodes: int) -> Dict[int, int]:
    """Compute the number of maximal cliques each node participates in.
    
    Optimized version using vectorized operations where possible.
    
    Args:
        cliques_df (pd.DataFrame): DataFrame with 'nodes' column containing lists of node indices.
        num_nodes (int): Total number of nodes in the network.
        
    Returns:
        Dict[int, int]: Dictionary mapping node index to participation count.
    """
    # Initialize participation count for all nodes
    participation = {node: 0 for node in range(num_nodes)}
    
    # Optimized: flatten all nodes and count in one pass
    all_nodes = [node for nodes_list in cliques_df['nodes'] for node in nodes_list]
    for node in all_nodes:
        participation[node] += 1
    
    return participation

def compute_clique_metrics(clique: list, g: Graph) -> List[float]:
    """Compute various metrics for a clique.
    
    Args:
        clique: List of node indices in the clique.
        g: igraph Graph object.
        
    Returns:
        List of metrics: [size, volume, avg_degree, boundary_edges, boundary_ratio, avg_embeddedness]
    """
    s = len(clique) # size of clique
    vol = sum(g.degree(v) for v in clique) # volume: sum of degrees of all nodes in the clique
    avg_deg = vol / s if s else 0.0 # average degree of nodes in the clique
    n_bound_edges = sum(g.degree(v) - (s - 1) for v in clique) # number of boundary edges: sum of external degrees of all nodes in the clique
    bound_ratio = n_bound_edges / vol if vol else 0.0 # boundary ratio (aka conductance): ratio of boundary edges to volume
    avg_embed = sum((s - 1) / g.degree(v) if g.degree(v) > 0 else 0.0 for v in clique) / s if s else 0.0 # average node embeddedness in the clique

    return [float(s), float(vol), float(avg_deg), float(n_bound_edges), float(bound_ratio), float(avg_embed)]


def detect_cliques(matrix: np.ndarray, min_size: int = 4) -> pd.DataFrame:
    """Detect all maximal cliques and compute their properties.
    
    Args:
        matrix (np.ndarray): Adjacency matrix representing the graph.
        min_size (int): Minimum clique size to detect. Default is 4.
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - clique_index: Index of the clique
            - nodes: List of nodes in the clique
            - clique_size: Number of nodes in the clique
            - clique_volume: Volume (sum of degrees) of the clique
            - clique_avg_degree: Average degree of nodes in the clique
            - clique_boundary_edges: Number of boundary edges
            - clique_boundary_ratio: Boundary ratio (conductance)
            - clique_avg_embeddedness: Average node embeddedness
    """
    # Create igraph graph
    graph = Graph.Weighted_Adjacency(matrix.tolist(), mode='undirected', attr='weight', loops='ignore')
    
    # Find maximal cliques (min size specified by parameter)
    cliques = graph.maximal_cliques(min=min_size)
    
    # Prepare data for DataFrame
    clique_data = []

    for idx, clique in enumerate(cliques):
        # Compute metrics using the integrated function
        metrics = compute_clique_metrics(clique, graph)
        # metrics = [size, volume, avg_degree, boundary_edges, boundary_ratio, avg_embeddedness]
        
        clique_data.append({
            'clique_index': idx,
            'nodes': list(clique),  # Convert to list for DataFrame storage
            'clique_size': int(metrics[0]),
            'clique_volume': int(metrics[1]),
            'clique_avg_degree': metrics[2],
            'clique_boundary_edges': int(metrics[3]),
            'clique_boundary_ratio': metrics[4],
            'clique_avg_embeddedness': metrics[5]
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


def analyze_single_matrix(matrix: np.ndarray, mapping_df: pd.DataFrame, 
                          subject_id: str = '', min_clique_size: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform complete clique analysis on a single connectivity matrix.
    
    Args:
        matrix (np.ndarray): Connectivity matrix.
        mapping_df (pd.DataFrame): Brain region mapping DataFrame.
        subject_id (str): Identifier for the subject.
        min_clique_size (int): Minimum clique size to detect. Default is 4.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame with clique measures (with subject_id column)
            - DataFrame with node participation (columns: node_id, n_cliques)
    """
    print(f"Analyzing matrix{' for ' + subject_id if subject_id else ''}...")
    
    # Detect cliques and compute properties
    cliques_df = detect_cliques(matrix, min_size=min_clique_size)
    print(f"  Found {len(cliques_df)} maximal cliques")
    print(f"  Max clique size: {cliques_df['clique_size'].max()}, minimum clique size: {cliques_df['clique_size'].min()}")
    print(f"  Average clique size: {cliques_df['clique_size'].mean():.4f}")
    print(f"  Average volume: {cliques_df['clique_volume'].mean():.4f}")
    print(f"  Average degree: {cliques_df['clique_avg_degree'].mean():.4f}")
    print(f"  Average boundary ratio: {cliques_df['clique_boundary_ratio'].mean():.4f}")
    print(f"  Average embeddedness: {cliques_df['clique_avg_embeddedness'].mean():.4f}")
    
    # Map cliques to regions
    cliques_with_regions = map_cliques_to_regions(cliques_df, mapping_df)
    print(f"  Mapped cliques to brain regions")

    # Add subject_id as first column to clique DataFrame
    cliques_with_regions.insert(0, 'subject_id', subject_id)
    
    # Compute node participation
    num_nodes = matrix.shape[0]
    node_participation = compute_node_participation(cliques_df, num_nodes)
    print(f"  Computed node participation for {num_nodes} nodes")
    
    # Create node participation DataFrame
    node_participation_df = pd.DataFrame([
        {'node_id': node, 'n_cliques': count}
        for node, count in sorted(node_participation.items())
    ])

    # Add subject_id to node participation DataFrame
    node_participation_df.insert(0, 'subject_id', subject_id)
    
    return cliques_with_regions, node_participation_df


def main(connectivity_files: List[str], mapping_file: str, output_base_dir: str, min_clique_size: int = 4, export_mode: str = 'csv') -> None:
    """Main function to run clique analysis on multiple connectivity matrices.
    
    Args:
        connectivity_files (List[str]): List of paths to connectivity matrix files.
        mapping_file (str): Path to mapping CSV file.
        output_base_dir (str): Base directory for output files.
        min_clique_size (int): Minimum clique size to detect. Default is 4.
    """
    # Load brain region mapping
    print(f"Loading brain region mapping from {mapping_file}...")
    mapping_df = pd.read_csv(mapping_file)
    print(f"  Loaded mapping for {len(mapping_df)} regions")
    
    # Collect results from all subjects
    all_clique_measures = []
    all_node_participation = []
    
    for conn_file in connectivity_files:

        time_start = time.time()
        
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
        
        # Check connectivity with igraph
        g = Graph.Weighted_Adjacency(matrix.tolist(), mode='undirected', attr='weight', loops='ignore')
        components = len(g.clusters())
        print(f"  Connected components: {components}")
        print(f"  Min value: {matrix.min()}, Max value: {matrix.max()}")
        del g, components

        # Run analysis
        clique_measures, node_participation = analyze_single_matrix(matrix, mapping_df, subject_id, min_clique_size)
        
        # Collect results
        all_clique_measures.append(clique_measures)
        all_node_participation.append(node_participation)

        time_end = time.time()
        elapsed = time_end - time_start
        print(f"Completed analysis for {subject_id} in {elapsed:.2f} seconds")
        print("\n")

    
    # Combine all results
    if all_clique_measures:
        # Create consolidated DataFrames
        clique_measures_combined = pd.concat(all_clique_measures, ignore_index=True)
        node_participation_combined = pd.concat(all_node_participation, ignore_index=True)
        
        # Create output directory
        output_path = Path(output_base_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Flatten list columns for clique_measures export
        export_clique = clique_measures_combined.copy()
        export_clique['nodes'] = export_clique['nodes'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
        export_clique['yeo7_networks'] = export_clique['yeo7_networks'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_clique['yeo17_networks'] = export_clique['yeo17_networks'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_clique['gyrus_regions'] = export_clique['gyrus_regions'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_clique['lobes'] = export_clique['lobes'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        export_clique['hemispheres'] = export_clique['hemispheres'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        
        # Save both DataFrames
        print(f"  Total cliques analyzed: {len(clique_measures_combined)}")
        print(f"Saving consolidated results to {output_path}...")

        if export_mode in ['csv', 'both']:
            clique_measures_path = output_path / 'clique_measures.csv'
            node_participation_path = output_path / 'node_participation.csv'
            export_clique.to_csv(clique_measures_path, index=False)
            node_participation_combined.to_csv(node_participation_path, index=False)

        if export_mode in ['parquet', 'both']:
            clique_measures_parquet_path = output_path / 'clique_measures.parquet'
            node_participation_parquet_path = output_path / 'node_participation.parquet'
            export_clique.to_parquet(clique_measures_parquet_path, index=False)
            node_participation_combined.to_parquet(node_participation_parquet_path, index=False)
            
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Determine default data directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'test_sample')
    default_mapping_file = os.path.join(os.path.dirname(script_dir), 'data', 'mapping.csv')

    parser = argparse.ArgumentParser(description='Clique Analysis for Structural Connectivity Networks')
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='Directory containing connectivity matrix files (.csv or .npy)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results and visualizations')
    parser.add_argument('--mapping_file', type=str, default=default_mapping_file,
                        help='Path to brain region mapping CSV file')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='File pattern to match (e.g., "*.csv", "*.npy")')
    parser.add_argument('--min_clique', type=int, default=4,
                        help='Minimum clique size to detect (default: 4)')
    parser.add_argument('--export_mode', type=str, choices=['csv', 'parquet', 'both'], default='csv',
                        help='Export format for results (default: csv)')

    args = parser.parse_args()

    # Generate default output directory based on actually used data_dir
    if args.output_dir is None:
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        data_folder_name = os.path.basename(args.data_dir)
        default_output_name = f'clique_mapping_{data_folder_name}_{current_time}'
        args.output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'clique_mapping', default_output_name)
    
    print(f"Starting clique analysis...")

    # Find all connectivity files matching pattern
    connectivity_files = glob.glob(os.path.join(args.data_dir, args.pattern))
    
    if connectivity_files:
        print(f"Found {len(connectivity_files)} connectivity files in {args.data_dir}")
        print(f"Output will be saved to: {args.output_dir}")
        print(f"Minimum clique size: {args.min_clique}")
        print(f"Export mode: {args.export_mode}")
        main(connectivity_files, args.mapping_file, args.output_dir, args.min_clique, args.export_mode)
    else:
        print(f"No connectivity files found matching pattern '{args.pattern}' in {args.data_dir}")
        print("Please check the data directory and pattern.")
