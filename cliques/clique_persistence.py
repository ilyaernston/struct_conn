"""
Persistence analysis for structural connectivity networks.

This script performs topological data analysis on connectivity matrices to detect
loops (1-dimensional homology) and voids (2-dimensional homology) using persistent homology.

Analysis steps:
1. Load and preprocess connectomes
2. Perform Weight Rank Clique Filtration for each tractogram
3. Construct flag complex and detect persistent homologies in dimensions 0-2
4. Compute persistence measures (lifetime, death-birth ratio)
5. Track minimal cycles (nodes participating in each homology)
6. Generate persistence diagrams and Betti curves for each subject
7. Export consolidated results across subjects
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import os
import glob
import time

import numpy as np
import pandas as pd
import igraph as ig

import matplotlib.pyplot as plt
import seaborn as sns

import gudhi as gd
from preprocessing import drop_cerebellum, connect_components, normalize


def preprocess_matrix(matrix: np.ndarray, 
                      mapping_df: pd.DataFrame,
                      drop_cerebellum_flag: bool = True, 
                      connect_components_flag: bool = True) -> np.ndarray:
            # Preprocess matrix
    print(f"Preprocessing matrix...")
    if drop_cerebellum_flag:
        matrix = drop_cerebellum(matrix, mapping_df)
    if connect_components_flag:
        matrix = connect_components(matrix, mapping_df) # type: ignore
    matrix = normalize(matrix)  # type: ignore
    # Convert to distance: higher connectivity = smaller distance
    distance_matrix = 1.0 - matrix
    np.fill_diagonal(distance_matrix, 0.0)
    print(f"  Preprocessed matrix shape: {matrix.shape}")
    print(f"  Min value: {matrix.min():.4f}, Max value: {matrix.max():.4f}")

    return distance_matrix

def compute_persistence(matrix: np.ndarray, max_dimension: int = 2) -> Dict:
    """
    Compute persistent homology using Gudhi's Rips complex.
    
    Args:
        matrix: Distance adjacency matrix (lower values = stronger connections)
        max_dimension: Maximum homology dimension to compute (default: 2)
        
    Returns:
        Dictionary containing persistence diagrams and the Rips complex object
    """
    
    try:
        # Create Rips complex from distance matrix
        rips_complex = gd.RipsComplex(  # type: ignore
            distance_matrix=matrix,
            max_edge_length=1.0
        ) 
        
        # Create simplex tree with specified max dimension
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract persistence diagrams by dimension
        persistence_pairs = simplex_tree.persistence()
        
        # Organize by dimension
        dgms = {}
        for dim in range(max_dimension + 1):
            pairs = [(birth, death) for (d, (birth, death)) in persistence_pairs if d == dim]
            dgms[dim] = np.array(pairs) if pairs else np.array([]).reshape(0, 2)
        
        results = {
            'simplex_tree': simplex_tree,
            'persistence_pairs': persistence_pairs,
            'dgms': dgms,
            'h0_persistence': dgms.get(0, np.array([])),
            'h1_persistence': dgms.get(1, np.array([])),
            'h2_persistence': dgms.get(2, np.array([])),
        }
        
        return results
        
    except Exception as e:
        print(f"Error computing persistence: {e}")
        return {'error': str(e)}

def extract_representative_cycles(
    distance_matrix: np.ndarray,
    simplex_tree: gd.SimplexTree, # type: ignore
    persistence_pairs: List,
    percentile: float = 25.0
) -> List[Dict]:
    """
    Extract representative cycles for the most persistent H1 features using Gudhi.
    
    Args:
        distance_matrix: The distance matrix used for persistence computation
        simplex_tree: The Gudhi SimplexTree object
        persistence_pairs: List of (dimension, (birth, death)) tuples
        percentile: Percentile threshold for selecting persistent loops (default: 25.0)
                   Only loops with persistence >= this percentile will be extracted
        
    Returns:
        List of dictionaries containing cycle information
    """
    
    # Filter for H1 features only
    h1_features = [(idx, birth, death) for idx, (dim, (birth, death)) in enumerate(persistence_pairs) if dim == 1]
    
    if not h1_features:
        print("No H1 features detected.")
        return []
    
    # Sort by persistence (death - birth)
    h1_features_sorted = sorted(
        h1_features, 
        key=lambda f: f[2] - f[1],  # death - birth
        reverse=True
    )
    
    # Calculate persistence threshold based on percentile
    persistences = [f[2] - f[1] for f in h1_features_sorted]
    threshold = np.percentile(persistences, 100 - percentile)
    
    # Filter features above threshold
    top_features = [f for f in h1_features_sorted if (f[2] - f[1]) >= threshold]
    
    results = []
    
    for feature_idx, (orig_idx, birth, death) in enumerate(top_features):
        persistence_value = death - birth
        
        # Get the persistence pair (simplex that creates and destroys the cycle)
        dim, pair = persistence_pairs[orig_idx]
        
        # In Gudhi, we can extract the edges that form the cycle by looking at
        # the 1-skeleton (edges) that exist at the birth time
        edges_at_birth = []
        
        # Get all edges (1-simplices) with filtration value <= birth + small epsilon
        epsilon = 1e-10
        for simplex, filtration in simplex_tree.get_filtration():
            if len(simplex) == 2 and filtration <= birth + epsilon:
                edges_at_birth.append((simplex[0], simplex[1], filtration))
        
        if not edges_at_birth:
            continue
        
        # Build a graph from edges at birth time to find the cycle
        # The cycle is created when a new edge closes a loop
        G = ig.Graph()
        
        # Add all unique vertices first
        vertices = set()
        for u, v, filt in edges_at_birth:
            vertices.add(u)
            vertices.add(v)
        
        # Map original node IDs to igraph vertex indices (igraph uses 0-based sequential IDs)
        node_to_idx = {node: idx for idx, node in enumerate(sorted(vertices))}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        G.add_vertices(len(vertices))
        
        # Add edges with weights
        edge_list = []
        weights = []
        filtrations = []
        for u, v, filt in edges_at_birth:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_list.append((u_idx, v_idx))
            weights.append(distance_matrix[u, v])
            filtrations.append(filt)
        
        G.add_edges(edge_list)
        G.es['weight'] = weights
        G.es['filtration'] = filtrations
        
        # Find the edge that creates the cycle (closest to birth time)
        creator_edge = None
        min_diff = float('inf')
        for u, v, filt in edges_at_birth:
            diff = abs(filt - birth)
            if diff < min_diff:
                min_diff = diff
                creator_edge = (u, v)
        
        if creator_edge is None:
            continue
        
        # Find shortest cycle containing the creator edge
        u, v = creator_edge
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # Temporarily remove the creator edge and find shortest path between u and v
        # Find the edge ID for the creator edge
        edge_id = G.get_eid(u_idx, v_idx, error=False)
        
        if edge_id == -1:
            continue
        
        # Delete the edge temporarily
        G.delete_edges([edge_id])
        
        try:
            # Find shortest path (by weight) between u and v
            path_indices = G.get_shortest_paths(u_idx, v_idx, weights='weight', output='vpath')
            
            if not path_indices or not path_indices[0]:
                # No path found
                continue
            
            path_indices = path_indices[0]
            
            # Convert back to original node IDs
            cycle_nodes = [idx_to_node[idx] for idx in path_indices]
            cycle_edges = [(cycle_nodes[i], cycle_nodes[i+1]) for i in range(len(cycle_nodes)-1)]
            cycle_edges.append((cycle_nodes[-1], cycle_nodes[0]))
            
            # Calculate total cost
            total_cost = sum(distance_matrix[u, v] for u, v in cycle_edges)
            
            results.append({
                'persistence_id': feature_idx,
                'lifetime': float(persistence_value),
                'death_birth_ratio': float(death / birth) if birth > 0 else float('inf'),
                'birth': float(birth),
                'death': float(death),
                'nodes': cycle_nodes,
                'edges': cycle_edges,
                'total_cost': float(total_cost),
                'num_nodes': len(cycle_nodes)
            })
        except Exception as e:
            print(f"Could not find path for cycle {feature_idx}: {e}")
            continue
    
    return results

def compute_betti_curve(dgm: np.ndarray, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Betti curve from persistence diagram.
    
    Args:
        dgm: Persistence diagram
        num_points: Number of points to sample
        
    Returns:
        Tuple of (filtration_values, betti_numbers)
    """
    if len(dgm) == 0:
        return np.array([]), np.array([])
    
    # Determine filtration range (excluding infinite death times)
    finite_dgm = dgm[~np.isinf(dgm[:, 1])]
    if len(finite_dgm) == 0:
        return np.array([]), np.array([])
    
    min_filt = finite_dgm.min()
    max_filt = finite_dgm.max()
    
    # Sample filtration values
    filtration_values = np.linspace(min_filt, max_filt, num_points)
    betti_numbers = np.zeros(num_points)
    
    # Count features alive at each filtration value
    for i, filt_val in enumerate(filtration_values):
        betti_numbers[i] = np.sum((dgm[:, 0] <= filt_val) & (dgm[:, 1] > filt_val))
    
    return filtration_values, betti_numbers


def plot_persistence_diagram(dgms: Dict, subject_id: str, output_dir: Path):
    """Plot persistence diagram showing all dimensions on a single plot.
    
    Args:
        dgms: Dictionary mapping dimension to persistence diagrams
        subject_id: Subject identifier
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Use seaborn color palette
    colors = sns.color_palette('tab10', 3)
    dimension_names = ['H₀ (Components)', 'H₁ (Loops)', 'H₂ (Voids)']
    markers = ['o', 's', '^']  # circle, square, triangle
    
    max_val = 0
    total_features = 0
    
    for dim in range(3):
        dgm = dgms.get(dim, np.array([]))
        if len(dgm) == 0:
            continue
        
        # Filter out infinite persistence
        finite_dgm = dgm[~np.isinf(dgm[:, 1])]
        
        if len(finite_dgm) > 0:
            # Plot points
            ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], 
                      alpha=0.7, c=[colors[dim]], s=50, 
                      marker=markers[dim], 
                      label=f'{dimension_names[dim]} ({len(finite_dgm)})',
                      edgecolors='white', linewidth=0.5)
            
            max_val = max(max_val, finite_dgm.max())
            total_features += len(finite_dgm)
    
    if total_features == 0:
        ax.text(0.5, 0.5, 'No finite features detected', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        # Plot diagonal
        max_val = max(max_val, 1.0)
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1.5, label='y = x')
        
        ax.set_xlabel(f'Birth ($\\rho_b$)', fontsize=12)
        ax.set_ylabel(f'Death ($\\rho_d$)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        ax.legend(loc='lower right', framealpha=0.9)
    
    ax.set_title(f'Persistence Diagram', fontsize=13, pad=15)
    plt.tight_layout()
    
    output_path = output_dir / f'persistence_diagram_{subject_id}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved persistence diagram to {output_path}")


def plot_betti_curves(dgms: Dict, subject_id: str, output_dir: Path):
    """Plot Betti curves, creating subplots only for dimensions with features.
    
    Args:
        dgms: Dictionary mapping dimension to persistence diagrams
        subject_id: Subject identifier
        output_dir: Directory to save the plot
    """
    # Use seaborn color palette
    colors = sns.color_palette('tab10', 3)
    dimension_names = {0: 'H₀ (Components)', 1: 'H₁ (Loops)', 2: 'H₂ (Voids)'}
    
    # Determine which dimensions have features
    dims_with_features = []
    for dim in range(3):
        dgm = dgms.get(dim, np.array([]))
        if len(dgm) > 0:
            filt_vals, betti_nums = compute_betti_curve(dgm)
            if len(filt_vals) > 0:
                dims_with_features.append(dim)
    
    if not dims_with_features:
        print("  No features to plot Betti curves")
        return
    
    # Create subplots based on number of dimensions with features
    n_plots = len(dims_with_features)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    
    # Handle case of single subplot
    if n_plots == 1:
        axes = [axes]
    
    for ax, dim in zip(axes, dims_with_features):
        dgm = dgms.get(dim, np.array([]))
        filt_vals, betti_nums = compute_betti_curve(dgm)
        
        # Plot curve
        ax.plot(filt_vals, betti_nums, color=colors[dim], linewidth=2.5)
        ax.fill_between(filt_vals, betti_nums, alpha=0.3, color=colors[dim])
        
        ax.set_xlabel('Filtration value', fontsize=11)
        ax.set_ylabel(f'$\\beta_{{{dim}}}$', fontsize=11)
        ax.set_title(dimension_names[dim], fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylim(bottom=-0.5)
        
        # Add max Betti number annotation
        max_betti = int(betti_nums.max())
        if max_betti > 0:
            ax.text(0.98, 0.98, f'max $\\beta_{{{dim}}}$ = {max_betti}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
    
    plt.suptitle(f'Betti Curves', fontsize=13, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f'betti_curves_{subject_id}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Betti curves to {output_path}")


def analyze_single_matrix(matrix: np.ndarray,
                          mapping_df: pd.DataFrame,
                          filtered_mapping: pd.DataFrame,
                          subject_id: str = '', 
                          output_dir: Optional[Path] = None,
                          percentile: float = 25.0
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform persistence analysis on a single connectivity matrix.
    
    Args:
        matrix: Connectivity matrix
        subject_id: Subject identifier
        output_dir: Directory to save plots (if None, plots are not saved)
        mapping_df: DataFrame for region mapping (used in preprocessing)
        percentile: Percentile threshold for selecting persistent loops (default: 25.0)
        
    Returns:
        Tuple of (h1_features_df, h2_features_df)
    """

    start_time = time.time()

    # Preprocess matrix
    preprocessed_matrix = preprocess_matrix(matrix, 
                                            mapping_df=mapping_df, 
                                            drop_cerebellum_flag=True, 
                                            connect_components_flag=True)
    # Important: preprocessing includes transforming to distance matrix!

    preprocess_time = time.time()
    print(f"  Time elapsed: {preprocess_time - start_time:.3f}s")

    print(f"Analyzing persistence{' for ' + subject_id if subject_id else ''}...")
    
    # Compute persistent homology
    persistence_result = compute_persistence(preprocessed_matrix, max_dimension=2)
    
    persistence_time = time.time()
    print(f"  Persistence computed in {persistence_time - preprocess_time:.3f}s")

    if 'error' in persistence_result:
        print(f"  Error in persistence computation: {persistence_result['error']}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Print summary statistics
    print("\nPersistence Summary:")
    for dim in [0, 1, 2]:
        dgm = persistence_result['dgms'].get(dim, np.array([]))
        print(f"H{dim}: {len(dgm)} features detected")
    
    # Extract representative cycles for H1
    h1_dgm = persistence_result['dgms'].get(1, np.array([]))
    
    cycles = []
    if len(h1_dgm) > 0:
        print("\nExtracting Representative Cycles:")
        cycles = extract_representative_cycles(
            distance_matrix=preprocessed_matrix,
            simplex_tree=persistence_result['simplex_tree'],
            persistence_pairs=persistence_result['persistence_pairs'],
            percentile=percentile
        )
        
        extraction_time = time.time()
        print(f"  Cycles extracted in {extraction_time - persistence_time:.3f}s")
        
        if cycles:
            print(f"\nFound {len(cycles)} representative cycles:\n")
            
            for i, cycle in enumerate(cycles, 1):
                print(f" Cycle {i}")
                print(f"  Persistence: {cycle['lifetime']:.6f}")
                print(f"  Death/Birth Ratio: {cycle['death_birth_ratio']:.6f}")
                print(f"  Birth: {cycle['birth']:.6f}")
                print(f"  Death: {cycle['death']:.6f}")
                print(f"  Number of nodes: {cycle['num_nodes']}")
                print(f"  Nodes: {cycle['nodes']}")
                print(f"  Total cost: {cycle['total_cost']:.6f}")
                
                # Map to region labels
                labels = [filtered_mapping.iloc[n]['ROIname'] if n < len(filtered_mapping) else f"Node_{n}" 
                         for n in cycle['nodes']]
                print(f"  Regions: {labels}")
                print()
                
                # Add subject_id to cycle
                cycle['subject_id'] = subject_id
        else:
            print("No cycles could be extracted.")
    else:
        print("\nNo H1 loops detected in the persistence diagram.")
    
    # Generate and save plots if output directory is provided
    if output_dir is not None:
        plot_persistence_diagram(persistence_result['dgms'], subject_id, output_dir)
        plot_betti_curves(persistence_result['dgms'], subject_id, output_dir)
    
    # Convert cycles to DataFrame
    h1_features_df = pd.DataFrame(cycles) if cycles else pd.DataFrame()
    h2_features_df = pd.DataFrame()  # Placeholder for H2 features
    
    return h1_features_df, h2_features_df

def main(connectivity_files: List[str], 
         mapping_file: str, 
         output_base_dir: str, 
         export_mode: str = 'csv', 
         save_plots: bool = True,
         percentile: float = 25.0) -> None:
    """Main function to run persistence analysis on multiple connectivity matrices.
    
    Args:
        connectivity_files: List of paths to connectivity matrix files
        mapping_file: Path to mapping CSV file
        output_base_dir: Base directory for output files
        export_mode: Export format ('csv', 'parquet', or 'both')
        save_plots: Whether to save individual plots for each subject
        percentile: Percentile threshold for selecting persistent loops (default: 25.0)
    """
    # Load brain region mapping
    print(f"Loading brain region mapping from {mapping_file}...")
    mapping_df = pd.read_csv(mapping_file)
    print(f"  Loaded mapping for {len(mapping_df)} regions")
    filtered_mapping = mapping_df[mapping_df['Lobe'] != 'Cerebellum'].reset_index(drop=True)
    
    # Create output directory
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create plots subdirectory if needed
    if save_plots:
        plots_dir = output_path / 'plots'
        plots_dir.mkdir(exist_ok=True)
    else:
        plots_dir = None
    
    # Collect results from all subjects
    all_h1_features = []
    all_h2_features = []
    
    for conn_file in connectivity_files:
        subject_time_start = time.time()
        
        # Extract subject ID from filename
        subject_id = Path(conn_file).stem
        
        # Load connectivity matrix
        print(f"\nLoading connectivity matrix: {conn_file}")
        matrix = np.loadtxt(conn_file, delimiter=',')
        print(f"  Matrix shape: {matrix.shape}")
        
        # Run persistence analysis
        h1_features, h2_features = analyze_single_matrix(matrix, mapping_df, filtered_mapping, subject_id, plots_dir, percentile)
        
        # Collect results
        if not h1_features.empty:
            all_h1_features.append(h1_features)
        if not h2_features.empty:
            all_h2_features.append(h2_features)
        
        subject_time_end = time.time()
        elapsed = subject_time_end - subject_time_start
        print(f"Completed persistence analysis for {subject_id} in {elapsed:.2f} seconds")
        print()
    
    # Combine all results
    print("\n" + "="*80)
    print("Consolidating results...")
    
    if all_h1_features:
        h1_combined = pd.concat(all_h1_features, ignore_index=True)
        
        # Reorder columns
        column_order = ['subject_id', 'persistence_id', 'lifetime', 'death_birth_ratio', 
                       'birth', 'death', 'num_nodes', 'total_cost', 'nodes', 'edges']
        h1_combined = h1_combined[column_order]
        
        print(f"  Total H1 features (loops): {len(h1_combined)}")
        print(f"  Subjects with H1 features: {h1_combined['subject_id'].nunique()}")
        
        # Save H1 results
        if export_mode in ['csv', 'both']:
            h1_path = output_path / 'h1_loops.csv'
            h1_combined.to_csv(h1_path, index=False)
            print(f"  Saved H1 features to {h1_path}")
        
        if export_mode in ['parquet', 'both']:
            h1_parquet_path = output_path / 'h1_loops.parquet'
            h1_combined.to_parquet(h1_parquet_path, index=False)
            print(f"  Saved H1 features to {h1_parquet_path}")
    else:
        print("  No H1 features found across all subjects")
    
    if all_h2_features:
        h2_combined = pd.concat(all_h2_features, ignore_index=True)
        print(f"  Total H2 features (voids): {len(h2_combined)}")
        print(f"  Subjects with H2 features: {h2_combined['subject_id'].nunique()}")
        
        # Save H2 results
        if export_mode in ['csv', 'both']:
            h2_path = output_path / 'h2_voids.csv'
            h2_combined.to_csv(h2_path, index=False)
            print(f"  Saved H2 features to {h2_path}")
        
        if export_mode in ['parquet', 'both']:
            h2_parquet_path = output_path / 'h2_voids.parquet'
            h2_combined.to_parquet(h2_parquet_path, index=False)
            print(f"  Saved H2 features to {h2_parquet_path}")
    else:
        print("  No H2 features found across all subjects")
    
    print("\n" + "="*80)
    print("Persistence analysis complete!")
    print("="*80)


if __name__ == "__main__":
    # Determine default data directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'test_sample')
    default_mapping_file = os.path.join(os.path.dirname(script_dir), 'data', 'mapping.csv')
    
    parser = argparse.ArgumentParser(
        description='Persistence Analysis for Structural Connectivity Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs topological data analysis using persistent homology to detect
loops (1-dimensional) and voids (2-dimensional) in brain connectivity networks.

Examples:
  # Basic usage
  python clique_persistence.py --data_dir /path/to/connectomes
  
  # Custom output directory
  python clique_persistence.py --data_dir /path/to/connectomes --output_dir /path/to/output
  
  # Without saving individual plots
  python clique_persistence.py --data_dir /path/to/connectomes --no_plots
  
  # Export as parquet
  python clique_persistence.py --data_dir /path/to/connectomes --export_mode parquet
        """
    )
    
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='Directory containing connectivity matrix files (.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results and visualizations')
    parser.add_argument('--mapping_file', type=str, default=default_mapping_file,
                        help='Path to brain region mapping CSV file')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='File pattern to match (default: *.csv)')
    parser.add_argument('--export_mode', type=str, choices=['csv', 'parquet', 'both'], 
                        default='csv', help='Export format for results (default: csv)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip saving individual plots for each subject')
    parser.add_argument('--percentile', type=float, default=25.0,
                        help='Percentile threshold for selecting persistent loops (default: 25.0). Only loops with persistence >= this percentile will be extracted.')
    
    args = parser.parse_args()
    
    # Generate default output directory if not specified
    if args.output_dir is None:
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        data_folder_name = os.path.basename(args.data_dir)
        default_output_name = f'clique_persistence_{data_folder_name}_{current_time}'
        args.output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'clique_persistence', default_output_name)
    
    print("="*80)
    print("Starting clique persistence analysis...")
    print("="*80)
    
    # Find all connectivity files matching pattern
    connectivity_files = glob.glob(os.path.join(args.data_dir, args.pattern))
    
    if connectivity_files:
        print(f"Found {len(connectivity_files)} connectivity files in {args.data_dir}")
        print(f"Output will be saved to: {args.output_dir}")
        print(f"Export mode: {args.export_mode}")
        print(f"Save individual plots: {not args.no_plots}")
        print()
        
        main(
            connectivity_files=connectivity_files,
            mapping_file=args.mapping_file,
            output_base_dir=args.output_dir,
            export_mode=args.export_mode,
            save_plots=not args.no_plots,
            percentile=args.percentile
        )
    else:
        print(f"No connectivity files found matching pattern '{args.pattern}' in {args.data_dir}")
        print("Please check the data directory and pattern.")