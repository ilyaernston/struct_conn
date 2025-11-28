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
from typing import Dict, List, Tuple, Optional, Set
import argparse
import os
import glob
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ripser import ripser
from preprocessing import drop_cerebellum, connect_components, normalize


def compute_persistence(matrix: np.ndarray, max_dimension: int = 2) -> Dict:
    """Compute persistent homology using Ripser with cocycle representatives.
    
    Weight Rank Clique Filtration: edges with higher weights appear earlier
    in the filtration, so we use (1 - normalized_weight) as distance.
    
    Args:
        matrix: Connectivity matrix
        max_dimension: Maximum homology dimension to compute (default: 2)
        
    Returns:
        Dictionary containing persistence diagrams, cocycles, and related information
    """
    # Ensure matrix is normalized to [0, 1]
    if matrix.max() > 1.0 or matrix.min() < 0.0:
        matrix = normalize(matrix)
    
    # Convert to distance: higher connectivity = smaller distance
    distance_matrix = 1.0 - matrix
    
    # Set diagonal to 0
    np.fill_diagonal(distance_matrix, 0.0)
    
    try:
        # Compute persistent homology using ripser with cocycles
        persistence_result = ripser(
            distance_matrix,
            maxdim=max_dimension,
            thresh=np.inf,
            coeff=2,  # Use Z/2Z coefficients
            do_cocycles=True,
            distance_matrix=True
        )
        
        # Extract persistence diagrams
        dgms = persistence_result['dgms']
        cocycles = persistence_result.get('cocycles', [])
        
        results = {
            'dgms': dgms,
            'cocycles': cocycles,
            'h0_persistence': dgms[0] if len(dgms) > 0 else np.array([]),
            'h1_persistence': dgms[1] if len(dgms) > 1 else np.array([]),
            'h2_persistence': dgms[2] if len(dgms) > 2 else np.array([]),
            'num_edges': persistence_result.get('num_edges', 0)
        }
        
        return results
        
    except Exception as e:
        print(f"Error computing persistence: {e}")
        return {'error': str(e)}


def extract_nodes_from_cocycle(cocycle: np.ndarray, dimension: int, n_points: int) -> Set[int]:
    """Extract unique node IDs from a cocycle representative.
    
    Decodes ripser's simplex indices to get actual vertex IDs.
    
    Args:
        cocycle: Array of [simplex_idx, coefficient] pairs from ripser
        dimension: Dimension of the homology (1 for edges, 2 for triangles)
        n_points: Number of points in the original distance matrix
        
    Returns:
        Set of node IDs matching matrix row/column indices
    """
    nodes = set()
    
    for entry in cocycle:
        if len(entry) >= 1:
            simplex_idx = int(entry[0])
            vertices = decode_simplex_index(simplex_idx, dimension + 1, n_points)
            nodes.update(vertices)
    
    return nodes


def decode_simplex_index(idx: int, simplex_dim: int, n_points: int) -> List[int]:
    """Decode simplex index to vertices using combinatorial number system.
    
    Ripser uses binomial coefficient encoding. This reverses it via greedy algorithm.
    
    Args:
        idx: Simplex index from ripser
        simplex_dim: Number of vertices (2 for edges, 3 for triangles)
        n_points: Total number of points
        
    Returns:
        Sorted list of vertex indices
    """
    from math import comb
    
    vertices = []
    remaining_idx = idx
    
    for i in range(simplex_dim, 0, -1):
        v = i - 1
        while v < n_points:
            if comb(v + 1, i) > remaining_idx:
                break
            v += 1
        
        vertices.append(v)
        if v >= i:
            remaining_idx -= comb(v, i)
    
    return sorted(vertices)


def compute_homology_metrics(birth: float, death: float) -> Dict[str, float]:
    """Compute metrics for a single homology class.
    
    Args:
        birth: Birth filtration value (ρ_b)
        death: Death filtration value (ρ_d)
        
    Returns:
        Dictionary with lifetime and death-birth ratio
    """
    lifetime = death - birth
    death_birth_ratio = death / birth if birth > 0 else np.inf
    
    return {
        'lifetime': lifetime,
        'death_birth_ratio': death_birth_ratio
    }


def extract_homology_features(dgm: np.ndarray, dimension: int, subject_id: str, 
                            cocycles: Optional[List] = None, n_points: int = 0) -> pd.DataFrame:
    """Extract features from a persistence diagram for a specific dimension.
    
    Args:
        dgm: Persistence diagram (N x 2 array of [birth, death] pairs)
        dimension: Homology dimension (1 for loops, 2 for voids)
        subject_id: Subject identifier
        cocycles: Optional list of cocycle representatives from ripser
        n_points: Number of nodes in the connectivity matrix
        
    Returns:
        DataFrame with homology features including node participation
    """
    if len(dgm) == 0:
        return pd.DataFrame()
    
    features = []
    for idx, (birth, death) in enumerate(dgm):
        # Skip infinite persistence (connected components for H0)
        if np.isinf(death):
            continue
            
        metrics = compute_homology_metrics(birth, death)
        
        # Extract participating nodes from cocycle if available
        participating_nodes = []
        num_nodes = 0
        if cocycles is not None and idx < len(cocycles) and n_points > 0:
            try:
                cocycle = cocycles[idx]
                nodes = extract_nodes_from_cocycle(cocycle, dimension, n_points)
                participating_nodes = sorted(list(nodes))
                num_nodes = len(participating_nodes)
            except Exception as e:
                # If extraction fails, continue without node info
                pass
        
        feature = {
            'subject_id': subject_id,
            'homology_id': idx,
            'dimension': dimension,
            'birth_filtration': float(birth),
            'death_filtration': float(death),
            'lifetime': metrics['lifetime'],
            'death_birth_ratio': metrics['death_birth_ratio'],
            'num_nodes': num_nodes,
            'participating_nodes': str(participating_nodes) if participating_nodes else '[]'
        }
        
        features.append(feature)
    
    return pd.DataFrame(features)


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


def plot_persistence_diagram(dgms: List[np.ndarray], subject_id: str, output_dir: Path):
    """Plot persistence diagram for all dimensions.
    
    Args:
        dgms: List of persistence diagrams for each dimension
        subject_id: Subject identifier
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dimension_names = ['H₀ (Components)', 'H₁ (Loops)', 'H₂ (Voids)']
    colors = ['blue', 'red', 'green']
    
    for dim, (ax, dgm, name, color) in enumerate(zip(axes, dgms[:3], dimension_names, colors)):
        if len(dgm) == 0:
            ax.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
            continue
        
        # Filter out infinite persistence
        finite_dgm = dgm[~np.isinf(dgm[:, 1])]
        
        if len(finite_dgm) > 0:
            # Plot points
            ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], alpha=0.6, c=color, s=30)
            
            # Plot diagonal
            max_val = max(finite_dgm.max(), 1.0)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('Birth (ρ_b)', fontsize=10)
            ax.set_ylabel('Death (ρ_d)', fontsize=10)
            ax.set_title(f'{name} ({len(finite_dgm)} features)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            ax.text(0.5, 0.5, 'All infinite', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
    
    plt.suptitle(f'Persistence Diagram - {subject_id}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'persistence_diagram_{subject_id}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved persistence diagram to {output_path}")


def plot_betti_curves(dgms: List[np.ndarray], subject_id: str, output_dir: Path):
    """Plot Betti curves for dimensions 1 and 2.
    
    Args:
        dgms: List of persistence diagrams
        subject_id: Subject identifier
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dimension_names = ['H₁ (Loops)', 'H₂ (Voids)']
    colors = ['red', 'green']
    
    for dim_idx, (ax, name, color) in enumerate(zip(axes, dimension_names, colors)):
        dgm_idx = dim_idx + 1  # Skip H0
        if dgm_idx >= len(dgms) or len(dgms[dgm_idx]) == 0:
            ax.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
            continue
        
        filt_vals, betti_nums = compute_betti_curve(dgms[dgm_idx])
        
        if len(filt_vals) > 0:
            ax.plot(filt_vals, betti_nums, color=color, linewidth=2)
            ax.fill_between(filt_vals, betti_nums, alpha=0.3, color=color)
            ax.set_xlabel('Filtration value', fontsize=10)
            ax.set_ylabel(f'β_{dim_idx}', fontsize=10)
            ax.set_title(name, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=-0.5)
        else:
            ax.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
    
    plt.suptitle(f'Betti Curves - {subject_id}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'betti_curves_{subject_id}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Betti curves to {output_path}")


def analyze_single_matrix(matrix: np.ndarray, subject_id: str = '', 
                          output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform persistence analysis on a single connectivity matrix.
    
    Args:
        matrix: Connectivity matrix
        subject_id: Subject identifier
        output_dir: Directory to save plots (if None, plots are not saved)
        
    Returns:
        Tuple of (h1_features_df, h2_features_df)
    """
    print(f"Analyzing persistence{' for ' + subject_id if subject_id else ''}...")
    
    # Compute persistent homology
    persistence_result = compute_persistence(matrix, max_dimension=2)
    
    if 'error' in persistence_result:
        print(f"  Error in persistence computation: {persistence_result['error']}")
        return pd.DataFrame(), pd.DataFrame()
    
    dgms = persistence_result['dgms']
    cocycles = persistence_result.get('cocycles', [])
    
    # Report statistics
    print(f"  H0 features: {len(persistence_result['h0_persistence'])}")
    print(f"  H1 features (loops): {len(persistence_result['h1_persistence'])}")
    print(f"  H2 features (voids): {len(persistence_result['h2_persistence'])}")
    
    if len(persistence_result['h1_persistence']) > 0:
        h1_finite = persistence_result['h1_persistence'][~np.isinf(persistence_result['h1_persistence'][:, 1])]
        if len(h1_finite) > 0:
            avg_lifetime_h1 = np.mean(h1_finite[:, 1] - h1_finite[:, 0])
            print(f"  Average H1 lifetime: {avg_lifetime_h1:.4f}")
    
    if len(persistence_result['h2_persistence']) > 0:
        h2_finite = persistence_result['h2_persistence'][~np.isinf(persistence_result['h2_persistence'][:, 1])]
        if len(h2_finite) > 0:
            avg_lifetime_h2 = np.mean(h2_finite[:, 1] - h2_finite[:, 0])
            print(f"  Average H2 lifetime: {avg_lifetime_h2:.4f}")
    
    # Extract features for H1 (loops) and H2 (voids) with cocycle information
    h1_cocycles = cocycles[1] if len(cocycles) > 1 else None
    h2_cocycles = cocycles[2] if len(cocycles) > 2 else None
    n_points = matrix.shape[0]
    
    h1_features = extract_homology_features(persistence_result['h1_persistence'], 1, subject_id, h1_cocycles, n_points)
    h2_features = extract_homology_features(persistence_result['h2_persistence'], 2, subject_id, h2_cocycles, n_points)
    
    # Generate and save plots if output directory is provided
    if output_dir is not None:
        plot_persistence_diagram(dgms, subject_id, output_dir)
        plot_betti_curves(dgms, subject_id, output_dir)
    
    return h1_features, h2_features


def main(connectivity_files: List[str], mapping_file: str, output_base_dir: str, 
         export_mode: str = 'csv', save_plots: bool = True) -> None:
    """Main function to run persistence analysis on multiple connectivity matrices.
    
    Args:
        connectivity_files: List of paths to connectivity matrix files
        mapping_file: Path to mapping CSV file
        output_base_dir: Base directory for output files
        export_mode: Export format ('csv', 'parquet', or 'both')
        save_plots: Whether to save individual plots for each subject
    """
    # Load brain region mapping
    print(f"Loading brain region mapping from {mapping_file}...")
    mapping_df = pd.read_csv(mapping_file)
    print(f"  Loaded mapping for {len(mapping_df)} regions")
    
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
        matrix = normalize(matrix)  # type: ignore
        print(f"  Preprocessed matrix shape: {matrix.shape}")
        print(f"  Min value: {matrix.min():.4f}, Max value: {matrix.max():.4f}")
        
        # Run persistence analysis
        h1_features, h2_features = analyze_single_matrix(matrix, subject_id, plots_dir)
        
        # Collect results
        if not h1_features.empty:
            all_h1_features.append(h1_features)
        if not h2_features.empty:
            all_h2_features.append(h2_features)
        
        time_end = time.time()
        elapsed = time_end - time_start
        print(f"Completed persistence analysis for {subject_id} in {elapsed:.2f} seconds")
        print()
    
    # Combine all results
    print("\n" + "="*80)
    print("Consolidating results...")
    
    if all_h1_features:
        h1_combined = pd.concat(all_h1_features, ignore_index=True)
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
            save_plots=not args.no_plots
        )
    else:
        print(f"No connectivity files found matching pattern '{args.pattern}' in {args.data_dir}")
        print("Please check the data directory and pattern.")