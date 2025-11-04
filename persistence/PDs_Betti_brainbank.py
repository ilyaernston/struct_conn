"""
Developer Connectomes Persistence Analysis: Betti curves

Created on Thu May  1 17:18:00 2025
@author: elijah

This script performs persistence analysis on expert vs naive developer connectivity data
using Betti curves to characterize topological features across H0, H1, and H2 homology groups.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import gudhi as gd
import networkx as nx

import argparse
import time
from typing import Tuple, List, Dict, Optional
import sys
import os
from tqdm import tqdm
# Import helper functions from submodules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import drop_cerebellum, connect_components, normalize_matrix

# Set modern scientific style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (20, 10),
    "figure.dpi": 150,
    "lines.linewidth": 2.5,
    "axes.titlepad": 14,
    "image.cmap": "viridis"
})

def compute_persistence_full(distance_matrix: np.ndarray, max_edge_length: float = 1.0, max_dimension: int = 2) -> Dict[int, np.ndarray]:
    """
    Compute persistence diagrams for H0, H1, and H2 homology groups.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix for Rips complex construction
    max_edge_length : float
        Maximum edge length for Rips complex
    max_dimension : int
        Maximum homology dimension to compute
    
    Returns:
    --------
    Dict[int, np.ndarray] : Dictionary with homology dimensions as keys and persistence intervals as values
    """
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    persistence = simplex_tree.persistence()
    
    # Initialize dictionary for different homology groups
    homology_groups: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(max_dimension + 1)}
    
    for interval in persistence:
        dim = interval[0]
        birth, death = interval[1]
        
        # Handle infinite death times
        if death == float('inf'):
            death = max_edge_length
            
        if dim <= max_dimension:
            homology_groups[dim].append((birth, death))
    
    # Convert to numpy arrays
    result: Dict[int, np.ndarray] = {}
    for dim in homology_groups:
        if homology_groups[dim]:
            result[dim] = np.array(homology_groups[dim])
        else:
            result[dim] = np.empty((0, 2))
    
    return result

def compute_betti_curve(persistence_intervals: np.ndarray, filtration_values: np.ndarray) -> np.ndarray:
    """
    Compute Betti curve from persistence intervals.
    
    Parameters:
    -----------
    persistence_intervals : np.ndarray
        Array of (birth, death) pairs
    filtration_values : np.ndarray
        Array of filtration parameter values
    
    Returns:
    --------
    np.ndarray : Betti numbers at each filtration value
    """
    if len(persistence_intervals) == 0:
        return np.zeros_like(filtration_values)
    
    betti_numbers = np.zeros_like(filtration_values)
    
    for i, t in enumerate(filtration_values):
        # Count intervals that are alive at filtration value t
        alive = np.sum((persistence_intervals[:, 0] <= t) & (persistence_intervals[:, 1] > t))
        betti_numbers[i] = alive
    
    return betti_numbers

def preprocess_connectome(matrix: np.ndarray, mapping_df: pd.DataFrame, 
                         drop_cerebellum_flag: bool = True, 
                         connect_components_flag: bool = True) -> np.ndarray:
    """
    Preprocess connectivity matrix for persistence analysis.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Connectivity matrix
    mapping_df : pd.DataFrame
        Region mapping information
    drop_cerebellum_flag : bool
        Whether to remove cerebellum regions
    connect_components_flag : bool
        Whether to connect disconnected components
    
    Returns:
    --------
    np.ndarray : Preprocessed distance matrix
    """
    # Remove cerebellum if requested
    if drop_cerebellum_flag:
        matrix = drop_cerebellum(matrix, mapping_df)
    
    # Convert to distance matrix (1 - normalized weights)
    normalized_matrix = normalize_matrix(matrix)
    distance_matrix = 1.0 - normalized_matrix
    
    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0.0)
    
    # Connect components if requested
    if connect_components_flag:
        # Convert to graph and reconnect components
        graph = nx.from_numpy_array(1.0 - distance_matrix)
        connected_graph, _ = connect_components(graph, mapping_df)
        adj_matrix = nx.to_numpy_array(connected_graph)
        distance_matrix = 1.0 - adj_matrix
        np.fill_diagonal(distance_matrix, 0.0)
    
    return distance_matrix

def analyze_group_persistence(matrices: np.ndarray, names: np.ndarray, 
                            mapping_df: pd.DataFrame, 
                            homology_dims: List[int] = [0, 1, 2],
                            max_edge_length: float = 1.0,
                            n_filtration_points: int = 100) -> Dict:
    """
    Analyze persistence for a group of subjects.
    
    Parameters:
    -----------
    matrices : np.ndarray
        3D array of connectivity matrices (n_subjects, n_regions, n_regions)
    names : np.ndarray
        Subject names
    mapping_df : pd.DataFrame
        Region mapping information
    homology_dims : List[int]
        Homology dimensions to compute
    max_edge_length : float
        Maximum edge length for Rips complex
    n_filtration_points : int
        Number of filtration parameter values
    
    Returns:
    --------
    Dict : Analysis results containing Betti curves and statistics
    """
    n_subjects = matrices.shape[0]
    filtration_values = np.linspace(0, max_edge_length, n_filtration_points)
    
    # Initialize storage for Betti curves
    betti_curves = {dim: np.zeros((n_subjects, n_filtration_points)) for dim in homology_dims}
    
    print(f"Computing persistence for {n_subjects} subjects...")
    
    for i, matrix in enumerate(matrices):
        if i == 1:
            print(f"  Processed subject {i}/{n_subjects}")
        elif (i-1) % 10 == 0:
            print(f"  Processed subject {i-1}/{n_subjects}")
        
        try:
            # Preprocess connectivity matrix
            distance_matrix = preprocess_connectome(matrix, mapping_df)
            
            # Compute persistence diagrams
            # For safety and correctness, we always compute up to max(homology_dims) + 1, with minimum of 2
            max_dim = max(max(homology_dims) + 1, 2)
            persistence_dict = compute_persistence_full(distance_matrix, max_edge_length, max_dim)
            
            # Compute Betti curves for each homology dimension
            for dim in homology_dims:
                if dim in persistence_dict:
                    betti_curve = compute_betti_curve(persistence_dict[dim], filtration_values)
                    betti_curves[dim][i, :] = betti_curve
                    
        except Exception as e:
            print(f"  Warning: Failed to process subject {names[i]}: {str(e)}")
            # Fill with zeros for failed subjects
            for dim in homology_dims:
                betti_curves[dim][i, :] = np.zeros(n_filtration_points)
    
    # Compute statistics
    results = {
        'filtration_values': filtration_values,
        'subject_names': names,
        'betti_curves': betti_curves,
        'mean_curves': {},
        'std_curves': {},
        'sem_curves': {}
    }
    
    for dim in homology_dims:
        results['mean_curves'][dim] = np.mean(betti_curves[dim], axis=0)
        results['std_curves'][dim] = np.std(betti_curves[dim], axis=0)
        results['sem_curves'][dim] = np.std(betti_curves[dim], axis=0) / np.sqrt(n_subjects)
    
    return results

def generate_random_networks(reference_matrix: np.ndarray, 
                           n_random_networks: int = 20,
                           preserve_degree_sequence: bool = True) -> List[np.ndarray]:
    """
    Generate random networks with similar properties to the reference matrix.
    
    Parameters:
    -----------
    reference_matrix : np.ndarray
        Reference connectivity matrix to match properties
    n_random_networks : int
        Number of random networks to generate
    preserve_degree_sequence : bool
        Whether to preserve the degree sequence (default: True)
    
    Returns:
    --------
    List[np.ndarray] : List of random connectivity matrices
    """
    # Convert to binary for topology analysis
    binary_matrix = (reference_matrix > 0).astype(int)
    np.fill_diagonal(binary_matrix, 0)
    
    # Create networkx graph
    G = nx.from_numpy_array(binary_matrix)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    random_matrices = []
    
    print(f"Generating {n_random_networks} random networks with {n_nodes} nodes and {n_edges} edges...")
    
    for i in tqdm(range(n_random_networks)):
        if preserve_degree_sequence:
            # Generate random graph with same degree sequence
            try:
                # Use configuration model to preserve degree sequence
                degree_sequence = [G.degree(n) for n in G.nodes()]
                random_G = nx.configuration_model(degree_sequence)
                # Remove self-loops and parallel edges
                random_G = nx.Graph(random_G)
                random_G.remove_edges_from(nx.selfloop_edges(random_G))
            except:
                # Fallback to Erdős-Rényi if configuration model fails
                p = 2 * n_edges / (n_nodes * (n_nodes - 1))
                random_G = nx.erdos_renyi_graph(n_nodes, p)
        else:
            # Simple Erdős-Rényi random graph
            p = 2 * n_edges / (n_nodes * (n_nodes - 1))
            random_G = nx.erdos_renyi_graph(n_nodes, p)
        
        # Convert back to adjacency matrix
        random_adj = nx.to_numpy_array(random_G)
        
        # Assign random weights from the original matrix's weight distribution
        nonzero_weights = reference_matrix[reference_matrix > 0]
        if len(nonzero_weights) > 0:
            # Sample weights from the original distribution
            random_weights = np.random.choice(nonzero_weights, size=np.sum(random_adj > 0))
            random_weighted = random_adj.copy()
            random_weighted[random_adj > 0] = random_weights
        else:
            random_weighted = random_adj
            
        random_matrices.append(random_weighted)
    
    return random_matrices

def analyze_random_networks(matrices: List[np.ndarray], 
                          mapping_df: pd.DataFrame,
                          homology_dims: List[int] = [0, 1, 2],
                          max_edge_length: float = 1.0,
                          n_filtration_points: int = 100) -> Dict:
    """
    Analyze persistence for random networks.
    
    Parameters:
    -----------
    matrices : List[np.ndarray]
        List of random connectivity matrices
    mapping_df : pd.DataFrame
        Region mapping information
    homology_dims : List[int]
        Homology dimensions to compute
    max_edge_length : float
        Maximum edge length for Rips complex
    n_filtration_points : int
        Number of filtration parameter values
    
    Returns:
    --------
    Dict : Analysis results containing Betti curves and statistics
    """
    n_networks = len(matrices)
    filtration_values = np.linspace(0, max_edge_length, n_filtration_points)
    
    # Initialize storage for Betti curves
    betti_curves = {dim: np.zeros((n_networks, n_filtration_points)) for dim in homology_dims}
    
    print(f"Computing persistence for {n_networks} random networks...")
    
    for i, matrix in enumerate(matrices):
        if i == 0:
            print(f"  Processed random network {i+1}/{n_networks}")
        elif i % 5 == 0:
            print(f"  Processed random network {i}/{n_networks}")
        
        try:
            # Preprocess connectivity matrix
            distance_matrix = preprocess_connectome(matrix, mapping_df)
            
            # Compute persistence diagrams
            max_dim = max(max(homology_dims) + 1, 2)
            persistence_dict = compute_persistence_full(distance_matrix, max_edge_length, max_dim)
            
            # Compute Betti curves for each homology dimension
            for dim in homology_dims:
                if dim in persistence_dict:
                    betti_curve = compute_betti_curve(persistence_dict[dim], filtration_values)
                    betti_curves[dim][i, :] = betti_curve
                    
        except Exception as e:
            print(f"  Warning: Failed to process random network {i}: {str(e)}")
            # Fill with zeros for failed networks
            for dim in homology_dims:
                betti_curves[dim][i, :] = np.zeros(n_filtration_points)
    
    # Compute statistics
    results = {
        'filtration_values': filtration_values,
        'betti_curves': betti_curves,
        'mean_curves': {},
        'std_curves': {},
        'sem_curves': {}
    }
    
    for dim in homology_dims:
        results['mean_curves'][dim] = np.mean(betti_curves[dim], axis=0)
        results['std_curves'][dim] = np.std(betti_curves[dim], axis=0)
        results['sem_curves'][dim] = np.std(betti_curves[dim], axis=0) / np.sqrt(n_networks)
    
    return results

def plot_betti_curves_comparison(expert_results: Dict, naive_results: Dict, 
                                random_results: Optional[Dict] = None,
                                homology_dims: List[int] = [0, 1, 2],
                                output_dir: str = ".", 
                                show_plots: bool = True) -> None:
    """
    Plot Betti curves comparing expert, naive, and optionally random networks.
    
    Parameters:
    -----------
    expert_results : Dict
        Analysis results for expert group
    naive_results : Dict
        Analysis results for naive group
    random_results : Dict, optional
        Analysis results for random networks (default: None)
    homology_dims : List[int]
        Homology dimensions to plot
    output_dir : str
        Directory to save plots
    show_plots : bool
        Whether to display plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for different groups
    expert_color = '#1f77b4'  # Blue
    naive_color = '#ff7f0e'   # Orange
    random_color = '#808080'  # Gray
    
    # Create subplots for each homology dimension
    # Make plots bigger when only one homology dimension is processed
    if len(homology_dims) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, len(homology_dims), figsize=(5 * len(homology_dims), 6))
    
    for i, dim in enumerate(homology_dims):
        ax = axes[i]
        
        # Get data
        filtration = expert_results['filtration_values']
        expert_mean = expert_results['mean_curves'][dim]
        expert_std = expert_results['std_curves'][dim]
        naive_mean = naive_results['mean_curves'][dim]
        naive_std = naive_results['std_curves'][dim]
        
        # Adjust line width and font sizes based on number of dimensions
        linewidth = 3.5 if len(homology_dims) == 1 else 2.5
        fontsize_labels = 16 if len(homology_dims) == 1 else 14
        fontsize_title = 20 if len(homology_dims) == 1 else 16
        fontsize_legend = 14 if len(homology_dims) == 1 else 12
        
        # Plot random networks first (as reference/background)
        if random_results is not None:
            random_mean = random_results['mean_curves'][dim]
            random_std = random_results['std_curves'][dim]
            
            ax.plot(filtration, random_mean, color=random_color, linewidth=linewidth, 
                    linestyle='--', label=f'Random Networks (n={len(random_results["betti_curves"][dim])})')
            ax.fill_between(filtration, random_mean - random_std, random_mean + random_std, 
                           color=random_color, alpha=0.15)
        
        # Plot expert and naive curves on top
        ax.plot(filtration, expert_mean, color=expert_color, linewidth=linewidth, 
                label=f'Expert (n={len(expert_results["subject_names"])})')
        ax.fill_between(filtration, expert_mean - expert_std, expert_mean + expert_std, 
                       color=expert_color, alpha=0.2)
        
        ax.plot(filtration, naive_mean, color=naive_color, linewidth=linewidth, 
                label=f'Naive (n={len(naive_results["subject_names"])})')
        ax.fill_between(filtration, naive_mean - naive_std, naive_mean + naive_std, 
                       color=naive_color, alpha=0.2)
        
        # Formatting
        ax.set_xlabel('Filtration Parameter', fontsize=fontsize_labels)
        ax.set_ylabel(f'Betti Number $\\beta_{dim}$', fontsize=fontsize_labels)
        ax.set_title(f'$H_{dim}$ Betti Curves', fontsize=fontsize_title)
        ax.legend(fontsize=fontsize_legend)
        ax.grid(True, alpha=0.3)
        
        # Increase tick label sizes for single dimension plots
        if len(homology_dims) == 1:
            ax.tick_params(labelsize=14)
        
        # Set reasonable y-limits
        max_y = max(np.max(expert_mean + expert_std), np.max(naive_mean + naive_std))
        if random_results is not None:
            random_mean = random_results['mean_curves'][dim]
            random_std = random_results['std_curves'][dim]
            max_y = max(max_y, np.max(random_mean + random_std))
        ax.set_ylim(0, max_y * 1.1)
    
    #plt.tight_layout()
    
    # Save plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_with_random" if random_results is not None else ""
    filename = f"betti_curves_comparison_H{''.join(map(str, homology_dims))}{suffix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved Betti curves comparison to: {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compute persistence diagrams and plot Betti curves for expert vs naive developer connectomes"
    )

    # Determine default data directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'dev_connectomes')
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'dev_connectomes_betti')
    default_mapping_file = os.path.join(os.path.dirname(script_dir), 'data', 'mapping.csv')

    # Data parameters
    parser.add_argument(
        "--expert_mats", 
        type=str, 
        default=os.path.join(default_data_dir, "expert_mats.npy"),
        help="Path to expert connectivity matrices (.npy file, shape: n_subjects x n_regions x n_regions)"
    )
    parser.add_argument(
        "--expert_names", 
        type=str, 
        default=os.path.join(default_data_dir, "expert_names.npy"),
        help="Path to expert subject names (.npy file)"
    )
    parser.add_argument(
        "--naive_mats", 
        type=str, 
        default=os.path.join(default_data_dir, "naive_mats.npy"),
        help="Path to naive connectivity matrices (.npy file, shape: n_subjects x n_regions x n_regions)"
    )
    parser.add_argument(
        "--naive_names", 
        type=str, 
        default=os.path.join(default_data_dir, "naive_names.npy"),
        help="Path to naive subject names (.npy file)"
    )
    parser.add_argument(
        "--mapping", 
        type=str,
        default=default_mapping_file,
        help="Path to region mapping CSV file (default: search for mapping.csv in common locations)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.join(default_output_dir, "betti_analysis_output"),
        help="Output directory for plots and results (default: ./output/dev_connectomes_betti)"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--homology_dims", 
        nargs="+", 
        type=int, 
        default=[0, 1, 2],
        choices=[0, 1, 2],
        help="Homology dimensions to compute and plot (choices: 0, 1, 2; default: 0 1 2)"
    )
    
    parser.add_argument(
        "--max_edge_length", 
        type=float, 
        default=1.0,
        help="Maximum edge length for Rips complex (default: 1.0)"
    )
    
    parser.add_argument(
        "--n_filtration_points", 
        type=int, 
        default=100,
        help="Number of filtration parameter values (default: 100)"
    )
    
    parser.add_argument(
        "--no_show_plots", 
        action="store_true",
        help="Do not display plots (only save to files)"
    )
    
    # Random network comparison parameters
    parser.add_argument(
        "--include_random", 
        action="store_true",
        help="Generate random networks for comparison baseline"
    )
    
    parser.add_argument(
        "--n_random_networks", 
        type=int, 
        default=20,
        help="Number of random networks to generate (default: 20)"
    )
    
    parser.add_argument(
        "--preserve_degree_sequence", 
        action="store_true",
        help="Preserve degree sequence in random networks (default: False, uses Erdős-Rényi)"
    )
    
    args = parser.parse_args()
    
    print("=== Developer Connectomes Betti Curves Analysis ===")
    print(f"Expert data: {args.expert_mats}")
    print(f"Naive data: {args.naive_mats}")
    print(f"Homology dimensions: {args.homology_dims}")
    print(f"Output directory: {args.output_dir}")
    if args.include_random:
        print(f"Random networks: {args.n_random_networks} networks, preserve degree: {args.preserve_degree_sequence}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading connectivity data...")
    expert_mats = np.load(args.expert_mats, allow_pickle=True)
    expert_names = np.load(args.expert_names, allow_pickle=True)
    naive_mats = np.load(args.naive_mats, allow_pickle=True)
    naive_names = np.load(args.naive_names, allow_pickle=True)
    
    print(f"Expert group: {expert_mats.shape[0]} subjects, {expert_mats.shape[1]}x{expert_mats.shape[2]} connectivity matrices")
    print(f"Naive group: {naive_mats.shape[0]} subjects, {naive_mats.shape[1]}x{naive_mats.shape[2]} connectivity matrices")
    
    # Load mapping file
    if args.mapping is None:
        # Search for mapping file in common locations
        possible_paths = [
            "mapping.csv",
            "data/mapping.csv",
            "../data/mapping.csv",
            os.path.join(os.path.dirname(args.expert_mats), "mapping.csv"),
            os.path.join(os.path.dirname(os.path.dirname(args.expert_mats)), "mapping.csv")
        ]
        
        mapping_path = None
        for path in possible_paths:
            if os.path.exists(path):
                mapping_path = path
                break
        
        if mapping_path is None:
            raise FileNotFoundError("Could not find mapping.csv file. Please specify --mapping path.")
        
        args.mapping = mapping_path
    
    print(f"Loading region mapping from: {args.mapping}")
    mapping_df = pd.read_csv(args.mapping)
    print(f"Loaded mapping for {len(mapping_df)} regions")
    print()
    
    # Analyze expert group
    print("Analyzing expert group...")
    expert_results = analyze_group_persistence(
        expert_mats, expert_names, mapping_df,
        homology_dims=args.homology_dims,
        max_edge_length=args.max_edge_length,
        n_filtration_points=args.n_filtration_points
    )
    
    # Analyze naive group
    print("Analyzing naive group...")
    naive_results = analyze_group_persistence(
        naive_mats, naive_names, mapping_df,
        homology_dims=args.homology_dims,
        max_edge_length=args.max_edge_length,
        n_filtration_points=args.n_filtration_points
    )
    
    # Generate and analyze random networks if requested
    random_results = None
    if args.include_random:
        print("\nGenerating random networks for comparison...")
        # Use the combined matrices to get representative network properties
        all_mats = np.concatenate([expert_mats, naive_mats], axis=0)
        # Use the mean connectivity matrix as reference
        reference_matrix = np.mean(all_mats, axis=0)
        
        random_matrices = generate_random_networks(
            reference_matrix, 
            n_random_networks=args.n_random_networks,
            preserve_degree_sequence=args.preserve_degree_sequence
        )
        
        print("Analyzing random networks...")
        random_results = analyze_random_networks(
            random_matrices, mapping_df,
            homology_dims=args.homology_dims,
            max_edge_length=args.max_edge_length,
            n_filtration_points=args.n_filtration_points
        )
    
    # Plot comparison
    print("Creating Betti curves comparison plot...")
    plot_betti_curves_comparison(
        expert_results, naive_results,
        random_results=random_results,
        homology_dims=args.homology_dims,
        output_dir=args.output_dir,
        show_plots=not args.no_show_plots
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, "betti_analysis_results.npz")
    save_dict = {
        'expert_filtration': expert_results['filtration_values'],
        'expert_names': expert_results['subject_names'],
        'naive_filtration': naive_results['filtration_values'],
        'naive_names': naive_results['subject_names'],
        **{f'expert_mean_H{dim}': expert_results['mean_curves'][dim] for dim in args.homology_dims},
        **{f'expert_std_H{dim}': expert_results['std_curves'][dim] for dim in args.homology_dims},
        **{f'naive_mean_H{dim}': naive_results['mean_curves'][dim] for dim in args.homology_dims},
        **{f'naive_std_H{dim}': naive_results['std_curves'][dim] for dim in args.homology_dims}
    }
    
    # Add random network results if available
    if random_results is not None:
        save_dict.update({
            'random_filtration': random_results['filtration_values'],
            **{f'random_mean_H{dim}': random_results['mean_curves'][dim] for dim in args.homology_dims},
            **{f'random_std_H{dim}': random_results['std_curves'][dim] for dim in args.homology_dims}
        })
    
    np.savez(results_file, **save_dict)
    print(f"Saved analysis results to: {results_file}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for dim in args.homology_dims:
        expert_mean = expert_results['mean_curves'][dim]
        naive_mean = naive_results['mean_curves'][dim]
        expert_max = np.max(expert_mean)
        naive_max = np.max(naive_mean)
        
        if random_results is not None:
            random_mean = random_results['mean_curves'][dim]
            random_max = np.max(random_mean)
            print(f"H{dim} - Expert max Betti: {expert_max:.2f}, Naive max Betti: {naive_max:.2f}, Random max Betti: {random_max:.2f}")
        else:
            print(f"H{dim} - Expert max Betti: {expert_max:.2f}, Naive max Betti: {naive_max:.2f}")
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
