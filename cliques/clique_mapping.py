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
        List of metrics: [size, deg_in, total_ext_deg, cond, avg_ext_deg, bound_ratio]
    """
    s = len(clique) # size of clique
    deg_in = s*(s-1)//2 # clique inner degree: number of internal edges in the clique
    vol = sum(g.degree(v) for v in clique) # volume: sum of degrees of all nodes in the clique
    avg_deg = vol / s if s else 0.0 # average degree of nodes in the clique
    total_ext_deg = sum(g.degree(v) - (s - 1) for v in clique) # boundary degree: total external degree (of all nodes in the clique)
    avg_ext_deg = total_ext_deg / s if s else 0.0 # average external degree of nodes in the clique
    cond = total_ext_deg / vol if vol else 0.0 # conductance: ratio of boundary edges to volume
    bound_ratio = total_ext_deg / deg_in if deg_in else 0.0 # boundary ratio: ratio of boundary edges to internal edges

    return [float(s), float(deg_in), float(total_ext_deg), float(cond), float(avg_ext_deg), float(bound_ratio)]



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
            - clique_deg_in: Internal degree of the clique
            - clique_total_ext_deg: Total external degree
            - clique_conductance: Conductance metric
            - clique_avg_ext_deg: Average external degree
            - clique_bound_ratio: Boundary ratio
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
        # metrics = [size, deg_in, total_ext_deg, cond, avg_ext_deg, bound_ratio]
        
        clique_data.append({
            'clique_index': idx,
            'nodes': list(clique),  # Convert to list for DataFrame storage
            'clique_size': int(metrics[0]),
            'clique_deg_in': int(metrics[1]),
            'clique_total_ext_deg': int(metrics[2]),
            'clique_conductance': metrics[3],
            'clique_avg_ext_deg': metrics[4],
            'clique_bound_ratio': metrics[5]
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
                          output_dir: str, subject_id: str = '', min_clique_size: int = 4) -> pd.DataFrame:
    """Perform complete clique analysis on a single connectivity matrix.
    
    Args:
        matrix (np.ndarray): Connectivity matrix.
        mapping_df (pd.DataFrame): Brain region mapping DataFrame.
        output_dir (str): Directory to save outputs.
        subject_id (str): Identifier for the subject.
        min_clique_size (int): Minimum clique size to detect. Default is 4.
        
    Returns:
        pd.DataFrame: Complete clique analysis results.
    """
    print(f"Analyzing matrix{' for ' + subject_id if subject_id else ''}...")
    
    # Detect cliques and compute properties
    cliques_df = detect_cliques(matrix, min_size=min_clique_size)
    print(f"  Found {len(cliques_df)} maximal cliques")
    print(f"  Max clique size: {cliques_df['clique_size'].max()}, minimum clique size: {cliques_df['clique_size'].min()}")
    print(f"  Average clique size: {cliques_df['clique_size'].mean():.4f}")
    print(f"  Average conductance: {cliques_df['clique_conductance'].mean():.4f}")
    print(f"  Average boundary ratio: {cliques_df['clique_bound_ratio'].mean():.4f}")
    
    # Map cliques to regions
    cliques_with_regions = map_cliques_to_regions(cliques_df, mapping_df)
    print(f"  Mapped cliques to brain regions")
    
    # Compute node participation
    num_nodes = matrix.shape[0]
    node_participation = compute_node_participation(cliques_df, num_nodes)
    print(f"  Computed node participation for {num_nodes} nodes")
    
    # Add node participation as a column (same value for all rows)
    cliques_with_regions['node_participation'] = [node_participation] * len(cliques_with_regions)
    
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
    # Convert node_participation dict to string
    export_df['node_participation'] = export_df['node_participation'].apply(
        lambda x: ','.join(f'{k}:{v}' for k, v in sorted(x.items()))
    )
    
    export_df.to_csv(csv_path, index=False)
    print(f"  Results saved to {csv_path}")
    
    return cliques_with_regions


def main(connectivity_files: List[str], mapping_file: str, output_base_dir: str, min_clique_size: int = 4):
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
    
    # Analyze each connectivity matrix
    all_results = []
    
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

        # Create subject-specific output directory
        output_dir = Path(output_base_dir) / subject_id
        
        # Run analysis
        results = analyze_single_matrix(matrix, mapping_df, str(output_dir), subject_id, min_clique_size)
        
        # Add subject ID to results
        results['subject_id'] = subject_id
        all_results.append(results)

        time_end = time.time()
        elapsed = time_end - time_start
        print(f"Completed analysis for {subject_id} in {elapsed:.2f} seconds")
        print("\n")

    
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
        # Convert node_participation dict to string
        export_combined['node_participation'] = export_combined['node_participation'].apply(
            lambda x: ','.join(f'{k}:{v}' for k, v in sorted(x.items())) if isinstance(x, dict) else x
        )
        
        export_combined.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")
        print(f"Total cliques analyzed: {len(combined_results)}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Determine default data directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'test_sample')
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
    parser.add_argument('--min_clique', type=int, default=4,
                        help='Minimum clique size to detect (default: 4)')
    
    args = parser.parse_args()
    
    # Find all connectivity files matching pattern
    connectivity_files = glob.glob(os.path.join(args.data_dir, args.pattern))
    
    if connectivity_files:
        print(f"Found {len(connectivity_files)} connectivity files in {args.data_dir}")
        print(f"Output will be saved to: {args.output_dir}")
        print(f"Minimum clique size: {args.min_clique}")
        main(connectivity_files, args.mapping_file, args.output_dir, args.min_clique)
    else:
        print(f"No connectivity files found matching pattern '{args.pattern}' in {args.data_dir}")
        print("Please check the data directory and pattern.")
