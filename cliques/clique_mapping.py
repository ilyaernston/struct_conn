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

def graph_richclub_statistics(g: Graph, 
                              n_random: int = 100,
                              C_null_min: int = 2,
                              C_null_max: int = 20) -> Dict:
    """Compute graph-level statistics for rich-clubness analysis.
    
    This function computes:
    1. Ranked edge weights (descending order)
    2. Cumulative sums of ranked weights
    3. Null distribution statistics from random rewired networks
    
    Args:
        g: igraph Graph object.
        n_random: Number of random networks to generate for null model (default: 100).
        C_null_min: Minimum clique size for null model (default: 2).
        C_null_max: Maximum clique size for null model (default: 20).
        
    Returns:
        Dict containing:
            - 'ranked_weights': Array of edge weights sorted in descending order
            - 'cumsum_weights': Cumulative sum S(t) of ranked weights
            - 'null_mean': Dict mapping clique size to mean null φ_w^random
            - 'null_std': Dict mapping clique size to std null φ_w^random
    """
    # Extract all edge weights
    weights = np.array(g.es['weight'])
    
    # Check for None or NaN values
    if np.any(weights == None) or np.any(np.isnan(weights)):
        raise ValueError("Graph contains edges with None or NaN weights. Ensure all edges have valid weights.")
    
    # Sort edges by weight in descending order
    ranked_weights = np.sort(weights)[::-1]
    
    # Compute cumulative sums S(t)
    cumsum_weights = np.cumsum(ranked_weights)
    
    # Initialize null distribution storage
    null_distributions = {size: [] for size in range(C_null_min, C_null_max + 1)}  # Clique sizes C_null_min to C_null_max
    
    # Generate random networks and compute null rich-clubness
    print(f"    Generating {n_random} random networks for null model...")
    for i in range(n_random):
        # Create random network by rewiring topology and shuffling weights
        g_random = g.copy()
        
        # Store original weights before rewiring (rewiring doesn't preserve edge attributes)
        original_weights = np.array(g_random.es['weight'])
        
        # Rewire graph topology
        g_random.rewire(n=len(g.es) * 10, mode='simple')  # 10x edges for thorough rewiring
        
        # Randomly shuffle weights and reassign to rewired edges
        shuffled_weights = np.random.permutation(original_weights)
        g_random.es['weight'] = shuffled_weights.tolist()
        
        # Get ranked weights for random network
        random_weights = np.array(g_random.es['weight'])
        
        # Validate random weights
        if np.any(random_weights == None) or np.any(np.isnan(random_weights)):
            raise ValueError(f"Random network {i} contains None or NaN weights after shuffling.")
        
        random_ranked = np.sort(random_weights)[::-1]
        random_cumsum = np.cumsum(random_ranked)
        
        # For each clique size, compute φ_w^random
        for clique_size in range(C_null_min, C_null_max + 1):
            # Number of edges in a clique of given size
            n_edges = clique_size * (clique_size - 1) // 2
            
            if n_edges <= len(random_cumsum):
                # Sample random cliques and compute their rich-clubness
                # Approximate by sampling random sets of n_edges edges
                n_samples = 10  # Sample multiple random "cliques" per random network
                for _ in range(n_samples):
                    # Select random edges
                    random_edge_indices = np.random.choice(len(random_weights), size=min(n_edges, len(random_weights)), replace=False)
                    w_c_random = np.sum(random_weights[random_edge_indices])
                    
                    # Compute φ_w^random for this sample
                    phi_random = w_c_random / random_cumsum[n_edges - 1] if random_cumsum[n_edges - 1] > 0 else 0.0
                    null_distributions[clique_size].append(phi_random)
    
    # Compute mean and std for each clique size
    null_mean = {}
    null_std = {}
    for size in range(2, 21):
        if null_distributions[size]:
            null_mean[size] = np.mean(null_distributions[size])
            null_std[size] = np.std(null_distributions[size])
        else:
            null_mean[size] = 0.0
            null_std[size] = 1.0  # Avoid division by zero
    
    print(f"    Completed null model computation")
    
    return {
        'ranked_weights': ranked_weights,
        'cumsum_weights': cumsum_weights,
        'null_mean': null_mean,
        'null_std': null_std
    }

def clique_rich_clubness(g: Graph, 
                         clique: list, 
                         graph_richclub_stats: Dict) -> Tuple[float, float]:
    """Compute the weighted rich-clubness of a clique.
    
    Args:
        g: igraph Graph object.
        clique: List of node indices in the clique.
        graph_richclub_stats: Dictionary with pre-computed graph statistics from graph_richclub_statistics.
        
    Returns:
        Tuple of (raw_rich_clubness, normalized_rich_clubness).
    """
    clique_size = len(clique)
    
    # Handle edge cases
    if clique_size < 2:
        return 0.0, 0.0
    
    # Compute number of edges in clique
    n_edges_clique = clique_size * (clique_size - 1) // 2
    
    # Compute sum of weights inside the clique (W_C)
    w_c = 0.0
    for i, node_i in enumerate(clique):
        for node_j in clique[i+1:]:
            eid = g.get_eid(node_i, node_j, error=False)
            if eid != -1:
                w_c += g.es[eid]['weight']
    
    # Get cumulative sum for top E_C edges
    cumsum_weights = graph_richclub_stats['cumsum_weights']
    if n_edges_clique > len(cumsum_weights):
        # Clique has more edges than graph (shouldn't happen)
        return 0.0, 0.0
    
    # Compute raw weighted rich-clubness φ_w(C)
    s_ec = cumsum_weights[n_edges_clique - 1]
    phi_w = w_c / s_ec if s_ec > 0 else 0.0
    
    # Compute normalized rich-clubness φ_w^norm(C)
    null_mean = graph_richclub_stats['null_mean']
    null_std = graph_richclub_stats['null_std']
    
    # Use clique_size for null statistics, capped at 20
    size_key = min(clique_size, 20)
    mu = null_mean.get(size_key, 0.0)
    sigma = null_std.get(size_key, 1.0)
    
    phi_w_norm = (phi_w - mu) / sigma if sigma > 0 else 0.0
    
    return phi_w, phi_w_norm

def compute_clique_metrics(clique: list, g: Graph, graph_richclub_stats: Optional[Dict] = None) -> List[float]:
    """Compute various metrics for a clique.
    
    Args:
        clique: List of node indices in the clique.
        g: igraph Graph object.
        graph_richclub_stats: Optional pre-computed graph statistics for rich-clubness computation.
        
    Returns:
        List of metrics: [size, volume, avg_degree, boundary_edges, boundary_ratio, avg_embeddedness, 
                         rich_clubness_raw, rich_clubness_norm]
    """
    s = len(clique) # size of clique
    vol = sum(g.degree(v) for v in clique) # volume: sum of degrees of all nodes in the clique
    avg_deg = vol / s if s else 0.0 # average degree of nodes in the clique
    n_bound_edges = sum(g.degree(v) - (s - 1) for v in clique) # number of boundary edges: sum of external degrees of all nodes in the clique
    bound_ratio = n_bound_edges / vol if vol else 0.0 # boundary ratio (aka conductance): ratio of boundary edges to volume
    avg_embed = sum((s - 1) / g.degree(v) if g.degree(v) > 0 else 0.0 for v in clique) / s if s else 0.0 # average node embeddedness in the clique
    
    # Compute rich-clubness if graph_richclub_stats provided
    if graph_richclub_stats is not None:
        rc_raw, rc_norm = clique_rich_clubness(g, clique, graph_richclub_stats)
    else:
        rc_raw, rc_norm = 0.0, 0.0

    return [float(s), float(vol), float(avg_deg), float(n_bound_edges), float(bound_ratio), float(avg_embed), 
            float(rc_raw), float(rc_norm)]


def detect_cliques(matrix: np.ndarray, 
                   min_size: Optional[int] = None, 
                   max_size: Optional[int] = None,
                   compute_richclub: bool = True,
                   n_random: int = 100) -> pd.DataFrame:
    """Detect all maximal cliques and compute their properties.
    
    Args:
        matrix (np.ndarray): Adjacency matrix representing the graph.
        min_size (Optional[int]): Minimum clique size to detect. Default is 4.
        max_size (Optional[int]): Maximum clique size to detect. Default is None (detects [min_size; +∞]).
        compute_richclub (bool): Whether to compute rich-clubness metrics (default: True).
        n_random (int): Number of random networks for null model (default: 100).
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
            - clique_richclub_raw: Raw weighted rich-clubness φ_w(C)
            - clique_richclub_norm: Normalized weighted rich-clubness φ_w^norm(C)
    """
    # Create igraph graph
    graph = Graph.Weighted_Adjacency(matrix.tolist(), mode='undirected', attr='weight', loops='ignore')
    
    # Find maximal cliques (min size specified by parameter)
    if max_size is None:
        cliques = graph.maximal_cliques(min=min_size)
    elif min_size is None:
        cliques = graph.maximal_cliques(max=max_size)
    else:
        cliques = graph.maximal_cliques(min=min_size, max=max_size)
    
    # Compute graph-level rich-clubness statistics if requested
    graph_richclub_stats = None
    if compute_richclub and len(cliques) > 0:
        print(f"  Computing graph-level rich-clubness statistics...")
        graph_richclub_stats = graph_richclub_statistics(graph, n_random=n_random)
    
    # Prepare data for DataFrame
    clique_data = []

    for idx, clique in enumerate(cliques):
        # Compute metrics using the integrated function
        metrics = compute_clique_metrics(clique, graph, graph_richclub_stats)
        # metrics = [size, volume, avg_degree, boundary_edges, boundary_ratio, avg_embeddedness, 
        #            richclub_raw, richclub_norm]
        
        clique_data.append({
            'clique_index': idx,
            'nodes': list(clique),  # Convert to list for DataFrame storage
            'clique_size': int(metrics[0]),
            'clique_volume': int(metrics[1]),
            'clique_avg_degree': metrics[2],
            'clique_boundary_edges': int(metrics[3]),
            'clique_boundary_ratio': metrics[4],
            'clique_avg_embeddedness': metrics[5],
            'clique_richclub_raw': metrics[6],
            'clique_richclub_norm': metrics[7]
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


def analyze_single_matrix(matrix: np.ndarray, 
                          mapping_df: pd.DataFrame, 
                          subject_id: str = '', 
                          min_clique_size: Optional[int] = None,
                          max_clique_size: Optional[int] = None,
                          compute_richclub: bool = True,
                          n_random: int = 100
                          ) -> pd.DataFrame:
    """Perform complete clique analysis on a single connectivity matrix.
    
    Args:
        matrix (np.ndarray): Connectivity matrix.
        mapping_df (pd.DataFrame): Brain region mapping DataFrame.
        subject_id (str): Identifier for the subject.
        min_clique_size (Optional[int]): Minimum clique size to detect. Default is 4.
        max_clique_size (Optional[int]): Maximum clique size to detect. Default is None (detects [min_size; +∞]).
        compute_richclub (bool): Whether to compute rich-clubness metrics (default: True).
        n_random (int): Number of random networks for null model (default: 100).
    Returns:
        pd.DataFrame:
            - DataFrame with clique measures (with subject_id column)
    """
    print(f"Analyzing matrix...")
    
    # Detect cliques and compute properties
    cliques_df = detect_cliques(matrix, min_size=min_clique_size, max_size=max_clique_size, 
                                compute_richclub=compute_richclub, n_random=n_random)
    print(f"  Found {len(cliques_df)} maximal cliques")
    
    # Check if any cliques were found
    if len(cliques_df) == 0:
        print(f"  WARNING: No cliques of size >= {min_clique_size} found for {subject_id}")
        print(f"  Skipping this matrix and returning empty DataFrames")
        
        # Create empty DataFrame with correct structure for cliques
        empty_cliques = pd.DataFrame(columns=[
            'subject_id', 'clique_index', 'nodes', 'clique_size', 'clique_volume',
            'clique_avg_degree', 'clique_boundary_edges', 'clique_boundary_ratio',
            'clique_avg_embeddedness', 'clique_richclub_raw', 'clique_richclub_norm',
            'yeo7_networks', 'yeo7_primary', 'yeo17_networks', 'yeo17_primary', 
            'gyrus_regions', 'gyrus_primary', 'lobes', 'lobe_primary', 'hemispheres'
        ])
        
        return empty_cliques
    
    print(f"  Max clique size: {cliques_df['clique_size'].max()}, minimum clique size: {cliques_df['clique_size'].min()}")
    print(f"  Average clique size: {cliques_df['clique_size'].mean():.4f}")
    print(f"  Average volume: {cliques_df['clique_volume'].mean():.4f}")
    print(f"  Average degree: {cliques_df['clique_avg_degree'].mean():.4f}")
    print(f"  Average boundary ratio: {cliques_df['clique_boundary_ratio'].mean():.4f}")
    print(f"  Average embeddedness: {cliques_df['clique_avg_embeddedness'].mean():.4f}")
    if compute_richclub and 'clique_richclub_raw' in cliques_df.columns:
        print(f"  Average rich-clubness (raw): {cliques_df['clique_richclub_raw'].mean():.4f}")
        print(f"  Average rich-clubness (norm): {cliques_df['clique_richclub_norm'].mean():.4f}")
    
    # Map cliques to regions
    cliques_with_regions = map_cliques_to_regions(cliques_df, mapping_df)
    print(f"  Mapped cliques to brain regions")

    # Add subject_id as first column to clique DataFrame
    cliques_with_regions.insert(0, 'subject_id', subject_id)
    
    return cliques_with_regions


def main(connectivity_files: List[str], 
         mapping_file: str, 
         output_base_dir: str, 
         min_clique_size: Optional[int] = None, 
         max_clique_size: Optional[int] = None, 
         export_mode: str = 'csv',
         check_components: bool = False,
         compute_richclub: bool = True,
         n_random: int = 100
         ):
    """Main function to run clique analysis on multiple connectivity matrices.
    
    Args:
        connectivity_files (List[str]): List of paths to connectivity matrix files.
        mapping_file (str): Path to mapping CSV file.
        output_base_dir (str): Base directory for output files.
        min_clique_size (Optional[int]): Minimum clique size to detect. Default is 4.
        max_clique_size (Optional[int]): Maximum clique size to detect. Default is None (detects [min_size; +∞]).
    """

    # Load brain region mapping
    print(f"Loading brain region mapping from {mapping_file}...")
    mapping_df = pd.read_csv(mapping_file)
    print(f"  Loaded mapping for {len(mapping_df)} regions")
    
    # Collect results from all subjects
    all_clique_measures = []
    
    for conn_file in connectivity_files:

        time_start = time.time()
        
        # Extract subject ID from filename
        subject_id = Path(conn_file).stem
        subject_id_short = subject_id.split('_')[0]  # e.g., 'sub-01'
        
        # Load connectivity matrix
        print(f"\nLoading connectivity matrix for subject: {subject_id_short}")
        matrix = np.loadtxt(conn_file, delimiter=',')
        print(f"  Matrix shape: {matrix.shape}")

        # Preprocess matrix
        print(f"Preprocessing matrix...")
        matrix = drop_cerebellum(matrix, mapping_df)
        matrix = connect_components(matrix, mapping_df, return_altered_nodes=False)
        matrix = normalize(matrix) #type: ignore
        print(f"  Preprocessed matrix shape: {matrix.shape}")
        
        if check_components:
            # Check connectivity with igraph
            g = Graph.Weighted_Adjacency(matrix.tolist(), mode='undirected', attr='weight', loops='ignore')
            components = len(g.connected_components(mode='weak'))
            print(f"  Connected components: {components}")
            print(f"  Min value: {matrix.min()}, Max value: {matrix.max()}")
            del g, components

        # Run analysis
        clique_measures = analyze_single_matrix(matrix, mapping_df, subject_id, min_clique_size, max_clique_size,
                                               compute_richclub=compute_richclub, n_random=n_random)
        
        # Collect results
        all_clique_measures.append(clique_measures)

        time_end = time.time()
        elapsed = time_end - time_start
        print(f"Completed analysis for {subject_id_short} in {elapsed:.2f} seconds")
        print("\n")

    
    # Combine all results
    if all_clique_measures:
        # Create consolidated DataFrames
        clique_measures_combined = pd.concat(all_clique_measures, ignore_index=True)
        
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
            export_clique.to_csv(clique_measures_path, index=False)

        if export_mode in ['parquet', 'both']:
            clique_measures_parquet_path = output_path / 'clique_measures.parquet'
            export_clique.to_parquet(clique_measures_parquet_path, index=False)

            
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Determine default data directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'test_sample')
    default_mapping_file = os.path.join(os.path.dirname(script_dir), 'data', 'mapping.csv')

    parser = argparse.ArgumentParser(description='Clique Analysis for Structural Connectivity Networks')
    parser.add_argument('-d', '--data_dir', type=str, default=default_data_dir,
                        help='Directory containing connectivity matrix files (.csv or .npy)')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory for results and visualizations')
    parser.add_argument('-m', '--mapping_file', type=str, default=default_mapping_file,
                        help='Path to brain region mapping CSV file')
    parser.add_argument('--input_format', type=str, default='*.csv',
                        help='File pattern to match (e.g., "*.csv", "*.npy")')
    parser.add_argument('--min_size', type=int, default=4,
                        help='Minimum clique size to detect (default: 4)')
    parser.add_argument('--max_size', type=int, default=None,
                        help='Maximum clique size to detect (default: None, meaning no upper limit)')
    parser.add_argument('--export_format', type=str, choices=['csv', 'parquet', 'both'], default='csv',
                        help='Export format for results (default: csv)')
    parser.add_argument('--no_richclub', action='store_true',
                        help='Skip rich-clubness computation (faster but less complete)')
    parser.add_argument('--n_random', type=int, default=100,
                        help='Number of random networks for null model (default: 100)')

    args = parser.parse_args()

    # Generate default output directory based on actually used data_dir
    if args.output_dir is None:
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        data_folder_name = os.path.basename(args.data_dir)
        default_output_name = f'clique_mapping_{data_folder_name}_{current_time}'
        args.output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'clique_mapping', default_output_name)
    
    print(f"Starting clique analysis...")

    # Find all connectivity files matching pattern
    connectivity_files = glob.glob(os.path.join(args.data_dir, args.input_format))
    
    if connectivity_files:
        print(f"Found {len(connectivity_files)} connectivity files in {args.data_dir}")
        print(f"Output will be saved to: {args.output_dir}")
        print(f"Minimum clique size: {args.min_size}")
        print(f"Maximum clique size: {args.max_size}")
        print(f"Export format: {args.export_format}")
        print(f"Compute rich-clubness: {not args.no_richclub}")
        if not args.no_richclub:
            print(f"Random networks for null model: {args.n_random}")
        main(connectivity_files, args.mapping_file, args.output_dir, args.min_size, args.max_size, 
             args.export_format, compute_richclub=not args.no_richclub, n_random=args.n_random)
    else:
        print(f"No connectivity files found matching pattern '{args.input_format}' in {args.data_dir}")
        print("Please check the data directory and input format.")
