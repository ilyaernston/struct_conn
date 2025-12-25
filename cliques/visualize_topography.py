"""
Visualize maximal cliques and persistence loops on 3D brain using nilearn.

This script creates 3D brain visualizations for each subject using nilearn:
1. Maximal cliques - visualized using plot_markers with marker size = clique size
2. Persistence loops (H1 features) - visualized using plot_connectome showing cycle edges
3. Brain nodes positioned based on anatomical coordinates (MNI space)

Each subject gets two separate plots (one for cliques, one for loops).

Usage:
    python visualize_brain_topology.py -c <cliques.csv> -p <h1_loops.csv> -m <mapping.csv>
    python visualize_brain_topology.py -c <cliques.csv> -p <h1_loops.csv> -m <mapping.csv> --subject sub-0002
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import ast
import os

# Import nilearn plotting functions
from nilearn import plotting


def load_coordinates_from_mapping(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load actual anatomical 3D coordinates from mapping file (MNI space).
    
    Args:
        mapping_df: DataFrame with columns including 'x', 'y', 'z'
        
    Returns:
        DataFrame with renamed columns: 'coord_x', 'coord_y', 'coord_z'
    """
    df = mapping_df.copy()
    
    # Use actual coordinates from mapping file (should be in MNI space)
    if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        raise ValueError("Mapping file must contain 'x', 'y', 'z' coordinate columns")
    
    df['coord_x'] = df['x']
    df['coord_y'] = df['y']
    df['coord_z'] = df['z']
    
    return df


def parse_node_list(node_string: str) -> List[int]:
    """Parse node list from string representation.
    
    Handles formats: '[1, 2, 3]' or '1,2,3'
    """
    if pd.isna(node_string):
        return []
    
    node_string = str(node_string).strip()
    
    if not node_string or node_string == '':
        return []
    
    # Try parsing as Python literal (list) first
    try:
        parsed = ast.literal_eval(node_string)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        elif isinstance(parsed, (int, float)):
            return [int(parsed)]
        elif isinstance(parsed, tuple):
            return [int(x) for x in parsed]
        else:
            # Fall through to comma-separated parsing
            pass
    except (ValueError, SyntaxError):
        # Fall through to comma-separated parsing
        pass
    
    # Parse as comma-separated string
    try:
        return [int(x.strip()) for x in node_string.split(',') if x.strip()]
    except ValueError:
        print(f"Warning: Could not parse node string: {node_string}")
        return []


def parse_edge_list(edge_string: str) -> List[Tuple[int, int]]:
    """Parse edge list from string representation.
    
    Handles formats: '[(1, 2), (2, 3)]' or list of tuples
    """
    if pd.isna(edge_string):
        return []
    
    edge_string = str(edge_string).strip()
    
    # Try parsing as Python literal
    try:
        parsed = ast.literal_eval(edge_string)
        if isinstance(parsed, list):
            # Convert to list of tuples
            edges = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    edges.append((int(item[0]), int(item[1])))
            return edges
        return []
    except (ValueError, SyntaxError):
        return []


def get_clique_centroid(nodes: List[int], coords_df: pd.DataFrame) -> Tuple[float, float, float]:
    """Calculate centroid of clique nodes in 3D space.
    
    Args:
        nodes: List of node indices
        coords_df: DataFrame with coordinate columns
        
    Returns:
        Tuple of (x, y, z) centroid coordinates
    """
    valid_nodes = [n for n in nodes if n < len(coords_df)]
    if not valid_nodes:
        return (0.0, 0.0, 0.0)
    
    x_coords = coords_df.iloc[valid_nodes]['coord_x'].values
    y_coords = coords_df.iloc[valid_nodes]['coord_y'].values
    z_coords = coords_df.iloc[valid_nodes]['coord_z'].values
    
    return (float(np.mean(x_coords)), float(np.mean(y_coords)), float(np.mean(z_coords)))


def plot_cliques(cliques_df: pd.DataFrame, coords_df: pd.DataFrame, 
                    subject_id: Optional[str] = None, output_dir: Optional[Path] = None,
                    top_k: Optional[int] = None, show_plot: bool = True, display_mode: str = 'ortho',
                    show_title: bool = True):
    """
    Create 3D brain visualization of maximal cliques using nilearn's plot_markers.
    
    Args:
        cliques_df: DataFrame with clique information (must have 'nodes' and 'clique_size')
        coords_df: DataFrame with node coordinates
        subject_id: Optional subject identifier for filtering
        output_dir: Directory to save plots
        top_k: Number of largest cliques to display (default: None, show all)
        show_plot: Whether to display the plot (default: True)
        display_mode: Nilearn display mode (default: 'ortho')
        show_title: Whether to show title (default: True)
    """
    # Filter by subject if specified
    if subject_id and 'subject_id' in cliques_df.columns:
        df = cliques_df[cliques_df['subject_id'] == subject_id].copy()
        # Crop subject_id to 8 characters for display
        subject_id_short = subject_id[:8] if len(subject_id) > 8 else subject_id
        title_suffix = f'{subject_id_short}'
    else:
        df = cliques_df.copy()
        title_suffix = ''
    
    if len(df) == 0:
        print(f"No cliques found{' for ' + subject_id if subject_id else ''}")
        return
    
    # Sort by clique size and take top k if specified
    if top_k is not None:
        df = df.sort_values('clique_size', ascending=False).head(top_k)
    
    # Parse node lists and compute centroids
    df['nodes_parsed'] = df['nodes'].apply(parse_node_list)
    centroids = df['nodes_parsed'].apply(lambda nodes: get_clique_centroid(nodes, coords_df))
    df['centroid_x'] = centroids.apply(lambda c: c[0])
    df['centroid_y'] = centroids.apply(lambda c: c[1])
    df['centroid_z'] = centroids.apply(lambda c: c[2])
    
    # Prepare coordinates array for nilearn (shape: n_cliques x 3)
    node_coords = df[['centroid_x', 'centroid_y', 'centroid_z']].values
    
    # Prepare marker sizes (scale clique size to reasonable marker sizes)
    sizes = df['clique_size'].values
    # Normalize to range suitable for nilearn markers (e.g., 20-200)
    if len(sizes) > 0 and np.max(sizes) > np.min(sizes):
        marker_sizes = ((sizes - np.min(sizes)) / (np.max(sizes) - np.min(sizes))) * 180 + 20
    else:
        marker_sizes = np.full(len(sizes), 100)
    
    # Plot with multiple views - without nilearn title
    display = plotting.plot_markers(
        node_values=sizes,  # Color by clique size
        node_coords=node_coords,
        node_size=marker_sizes,
        node_cmap='YlOrRd',
        colorbar=True,
        title=None,  # Don't use nilearn's title
        display_mode=display_mode,
        annotate=False
    )
    
    # Get the figure from the display object
    fig = display.frame_axes.figure if hasattr(display, 'frame_axes') else plt.gcf()
    
    # Add matplotlib title if requested
    if show_title:
        main_title = f'Maximal Cliques Topography for {title_suffix}'
        fig.suptitle(main_title, fontsize=12, fontweight='bold', y=1.05)
    
    # Get actual clique sizes for legend
    min_clique_size = int(np.min(sizes))
    max_clique_size = int(np.max(sizes))
    
    # Add custom clique size legend
    from matplotlib.lines import Line2D
    
    # Create legend handles with actual marker sizes
    min_marker_pts = 6  # Small marker
    max_marker_pts = 14  # Large marker
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               label=f'Max clique: {max_clique_size}',
               markerfacecolor='orange', markersize=max_marker_pts, 
               markeredgewidth=1, linewidth=0),
        Line2D([0], [0], marker='o', color='w', 
               label=f'Min clique: {min_clique_size}',
               markerfacecolor='orange', markersize=min_marker_pts, 
               markeredgewidth=1, linewidth=0)
    ]
    
    # Add legend on the left side
    leg = fig.legend(handles=legend_elements, loc='center left', 
                     bbox_to_anchor=(-0.22, 0.5), 
                     frameon=True, fancybox=True, shadow=False, 
                     fontsize=9, title='Clique Sizes', title_fontsize=10)
    
    # Label the colorbar
    fig.text(1.05, 1.05, 'Clique Size', 
            ha='right', va='top', fontsize=10)
    
    # Adjust layout to prevent overlap
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.15, right=0.92)
    
    # Save if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f'_{subject_id}' if subject_id else ''
        filepath = output_dir / f'cliques_3d{suffix}.png'
        display.savefig(str(filepath), dpi=300, bbox_inches='tight')
        print(f"Saved clique visualization to {filepath}")
    
    if show_plot:
        plotting.show()
    else:
        display.close()


def plot_loops(loops_df: pd.DataFrame, coords_df: pd.DataFrame,
                               subject_id: Optional[str] = None, output_dir: Optional[Path] = None,
                               top_k: int = 10, show_plot: bool = True, display_mode: str = 'ortho',
                               show_title: bool = True):
    """
    Create 3D brain visualization of persistence loops using nilearn's plot_connectome.
    
    Args:
        loops_df: DataFrame with loop information (must have 'nodes' and 'lifetime')
        coords_df: DataFrame with node coordinates
        subject_id: Optional subject identifier for filtering
        output_dir: Directory to save plots
        top_k: Number of most persistent loops to display
        show_plot: Whether to display the plot (default: True)
        display_mode: Nilearn display mode (default: 'ortho')
        show_title: Whether to show title (default: True)
    """
    # Filter by subject if specified
    if subject_id and 'subject_id' in loops_df.columns:
        df = loops_df[loops_df['subject_id'] == subject_id].copy()
        # Crop subject_id to 8 characters for display
        subject_id_short = subject_id[:8] if len(subject_id) > 8 else subject_id
        title_suffix = f'{subject_id_short}'
    else:
        df = loops_df.copy()
        title_suffix = ''
    
    if len(df) == 0:
        print(f"No persistence loops found{' for ' + subject_id if subject_id else ''}")
        return
    
    # Sort by persistence and take top k
    df = df.sort_values('lifetime', ascending=False).head(top_k)
    
    # Parse node lists
    df['nodes_parsed'] = df['nodes'].apply(parse_node_list)
    
    # Parse edge lists if available
    if 'edges' in df.columns:
        df['edges_parsed'] = df['edges'].apply(parse_edge_list)
    
    # Create adjacency matrix for all loops combined
    n_nodes = len(coords_df)
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    
    # Build adjacency matrix with edge weights based on loop persistence
    # Use maximum persistence value for edges that appear in multiple loops
    for idx, row in df.iterrows():
        persistence = row['lifetime']
        
        # Add edges to adjacency matrix - prioritize 'edges' column
        if 'edges_parsed' in df.columns and len(row.get('edges_parsed', [])) > 0:
            edges = row['edges_parsed']
            # Plot every edge in the loop
            for u, v in edges:
                if u < n_nodes and v < n_nodes:
                    # Use max persistence for overlapping edges to ensure they're visible
                    adjacency_matrix[u, v] = max(adjacency_matrix[u, v], persistence)
                    adjacency_matrix[v, u] = max(adjacency_matrix[v, u], persistence)
        else:
            # Fallback: draw cycle connecting consecutive nodes from 'nodes' column
            nodes = row['nodes_parsed']
            valid_nodes = [n for n in nodes if n < n_nodes]
            
            if valid_nodes:
                for i in range(len(valid_nodes)):
                    u = valid_nodes[i]
                    v = valid_nodes[(i + 1) % len(valid_nodes)]
                    adjacency_matrix[u, v] = max(adjacency_matrix[u, v], persistence)
                    adjacency_matrix[v, u] = max(adjacency_matrix[v, u], persistence)
    
    # Prepare node coordinates (shape: n_nodes x 3)
    node_coords = coords_df[['coord_x', 'coord_y', 'coord_z']].values
    
    # Plot with multiple views - without nilearn title
    # Set edge threshold to show all loop edges (no filtering)
    display = plotting.plot_connectome(
        adjacency_matrix,
        node_coords,
        edge_threshold=0,  # Show all edges from loops (no filtering)
        edge_cmap='Reds',
        edge_vmin=0,
        edge_vmax=adjacency_matrix.max() if adjacency_matrix.max() > 0 else 1,
        node_size=1,
        node_color='steelblue',
        title=None,  # Don't use nilearn's title
        display_mode=display_mode,
        colorbar=True,
        annotate=False
    )
    
    # Get the figure from the display object
    fig = display.frame_axes.figure if hasattr(display, 'frame_axes') else plt.gcf()
    
    # Add matplotlib title if requested
    if show_title:
        main_title = f'Persistence Loops Topography for {title_suffix}'
        fig.suptitle(main_title, fontsize=12, fontweight='bold', y=1.05)
    
    # Label the colorbar
    fig.text(1.05, 1.05, 'Loop Persistence', 
            ha='right', va='top', fontsize=10)
    
    # Adjust layout to prevent overlap
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.15, right=0.92)
    
    # Save if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f'_{subject_id}' if subject_id else ''
        filepath = output_dir / f'persistence_loops_3d{suffix}.png'
        display.savefig(str(filepath), dpi=300, bbox_inches='tight')
        print(f"Saved persistence loop visualization to {filepath}")
    
    if show_plot:
        plotting.show()
    else:
        display.close()


def plot_combined(cliques_df: pd.DataFrame, loops_df: pd.DataFrame, coords_df: pd.DataFrame,
                  subject_id: Optional[str] = None, output_dir: Optional[Path] = None,
                  top_k_cliques: Optional[int] = None, top_k_loops: int = 10,
                  show_plot: bool = True, display_mode: str = 'ortho', 
                  show_title: bool = True):
    """
    Create 3D brain visualization with both cliques and loops on the same plot.
    Uses nilearn's plot_connectome with node markers.
    
    Args:
        cliques_df: DataFrame with clique information
        loops_df: DataFrame with loop information
        coords_df: DataFrame with node coordinates
        subject_id: Optional subject identifier for filtering
        output_dir: Directory to save plots
        top_k_cliques: Number of largest cliques to display
        top_k_loops: Number of most persistent loops to display
        show_plot: Whether to display the plot (default: True)
        display_mode: Nilearn display mode (default: 'ortho')
    """
    # Filter by subject if specified
    if subject_id:
        if 'subject_id' in cliques_df.columns:
            cliques_filtered = cliques_df[cliques_df['subject_id'] == subject_id].copy()
        else:
            cliques_filtered = cliques_df.copy()
        
        if 'subject_id' in loops_df.columns:
            loops_filtered = loops_df[loops_df['subject_id'] == subject_id].copy()
        else:
            loops_filtered = loops_df.copy()
        
        # Crop subject_id to 8 characters for display
        subject_id_short = subject_id[:8] if len(subject_id) > 8 else subject_id
        title_suffix = f'{subject_id_short}'
    else:
        cliques_filtered = cliques_df.copy()
        loops_filtered = loops_df.copy()
        title_suffix = ''
    
    # Sort and filter cliques
    if top_k_cliques is not None:
        cliques_filtered = cliques_filtered.sort_values('clique_size', ascending=False).head(top_k_cliques)
    
    # Sort and filter loops
    loops_filtered = loops_filtered.sort_values('lifetime', ascending=False).head(top_k_loops)
    
    # Parse node lists for cliques and compute centroids
    cliques_filtered['nodes_parsed'] = cliques_filtered['nodes'].apply(parse_node_list)
    centroids = cliques_filtered['nodes_parsed'].apply(lambda nodes: get_clique_centroid(nodes, coords_df))
    cliques_filtered['centroid_x'] = centroids.apply(lambda c: c[0])
    cliques_filtered['centroid_y'] = centroids.apply(lambda c: c[1])
    cliques_filtered['centroid_z'] = centroids.apply(lambda c: c[2])
    
    # Parse node and edge lists for loops
    loops_filtered['nodes_parsed'] = loops_filtered['nodes'].apply(parse_node_list)
    if 'edges' in loops_filtered.columns:
        loops_filtered['edges_parsed'] = loops_filtered['edges'].apply(parse_edge_list)
    
    # Create adjacency matrix for loops
    n_nodes = len(coords_df)
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    
    # Build adjacency matrix with edge weights based on loop persistence
    # Use maximum persistence value for edges that appear in multiple loops
    for idx, row in loops_filtered.iterrows():
        persistence = row['lifetime']
        
        # Add edges to adjacency matrix - prioritize 'edges' column
        if 'edges_parsed' in loops_filtered.columns and len(row.get('edges_parsed', [])) > 0:
            edges = row['edges_parsed']
            # Plot every edge in the loop
            for u, v in edges:
                if u < n_nodes and v < n_nodes:
                    # Use max persistence for overlapping edges to ensure they're visible
                    adjacency_matrix[u, v] = max(adjacency_matrix[u, v], persistence)
                    adjacency_matrix[v, u] = max(adjacency_matrix[v, u], persistence)
        else:
            # Fallback: draw cycle connecting consecutive nodes from 'nodes' column
            nodes = row['nodes_parsed']
            valid_nodes = [n for n in nodes if n < n_nodes]
            
            if valid_nodes:
                for i in range(len(valid_nodes)):
                    u = valid_nodes[i]
                    v = valid_nodes[(i + 1) % len(valid_nodes)]
                    adjacency_matrix[u, v] = max(adjacency_matrix[u, v], persistence)
                    adjacency_matrix[v, u] = max(adjacency_matrix[v, u], persistence)
    
    # Prepare node coordinates for all brain regions
    node_coords = coords_df[['coord_x', 'coord_y', 'coord_z']].values
    
    # Prepare clique marker information
    # Create a node_size array where most nodes are 0 (invisible) and clique centroids are sized
    node_sizes = np.zeros(n_nodes)
    node_values = np.zeros(n_nodes)
    
    # For each clique, find the nearest actual node to its centroid and mark it
    for idx, row in cliques_filtered.iterrows():
        centroid = np.array([row['centroid_x'], row['centroid_y'], row['centroid_z']])
        # Find nearest node to centroid
        distances = np.sqrt(np.sum((node_coords - centroid)**2, axis=1))
        nearest_node = np.argmin(distances)
        # Accumulate size and value (in case multiple cliques map to same node)
        node_sizes[nearest_node] += row['clique_size']
        node_values[nearest_node] = max(node_values[nearest_node], row['clique_size'])
    
    # Normalize node sizes
    nonzero_sizes = node_sizes[node_sizes > 0]
    if len(nonzero_sizes) > 0 and np.max(nonzero_sizes) > np.min(nonzero_sizes):
        node_sizes_normalized = np.where(
            node_sizes > 0,
            ((node_sizes - np.min(nonzero_sizes)) / (np.max(nonzero_sizes) - np.min(nonzero_sizes))) * 180 + 20,
            0
        )
    else:
        node_sizes_normalized = np.where(node_sizes > 0, 100, 0)
    
    # Create title with proper wrapping
    clique_str = f'{len(cliques_filtered)} cliques' if top_k_cliques is None else f'Top {len(cliques_filtered)} cliques'
    loop_str = f'Top {len(loops_filtered)} loops'
    
    # Plot using plot_connectome with node markers - without nilearn title
    # Note: plot_connectome doesn't support node_cmap, so we use a single color for nodes
    # Set edge threshold to show all loop edges (any edge with weight > 0)
    display = plotting.plot_connectome(
        adjacency_matrix,
        node_coords,
        edge_threshold=0,  # Show all edges from loops (no filtering)
        edge_cmap='Reds',
        edge_vmin=0,
        edge_vmax=adjacency_matrix.max() if adjacency_matrix.max() > 0 else 1,
        node_size=node_sizes_normalized,
        node_color='orange',  # Cliques shown as orange nodes
        node_kwargs={'alpha': 0.7},  # Make clique markers 70% transparent
        display_mode=display_mode,
        colorbar=True,
        annotate=False,
        title=None  # Don't use nilearn's title
    )

    # Get the figure from the display object
    fig = display.frame_axes.figure if hasattr(display, 'frame_axes') else plt.gcf()

    # Add matplotlib title if requested (pad down slightly to avoid overlap)
    if show_title:
        main_title = f'Cliques and Loops Topography for {title_suffix}'
        fig.suptitle(main_title, fontsize=12, fontweight='bold', y=1.05)
    
    # Get actual clique sizes for legend
    actual_clique_sizes = cliques_filtered['clique_size'].values
    min_clique_size = int(np.min(actual_clique_sizes))
    max_clique_size = int(np.max(actual_clique_sizes))
    
    # Add custom clique size legend using matplotlib legend instead of axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    
    # Create legend handles with actual marker sizes
    # Use marker size in points that roughly corresponds to the plot
    min_marker_pts = 6  # Small marker
    max_marker_pts = 14  # Large marker
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               label=f'Max clique: {max_clique_size}',
               markerfacecolor='orange', markersize=max_marker_pts, 
               markeredgewidth=1, linewidth=0),
        Line2D([0], [0], marker='o', color='w', 
               label=f'Min clique: {min_clique_size}',
               markerfacecolor='orange', markersize=min_marker_pts, 
               markeredgewidth=1, linewidth=0)
    ]
    
    # Add legend on the left side, vertically centered
    # Position further left to avoid overlap with plot
    leg = fig.legend(handles=legend_elements, loc='center left', 
                     bbox_to_anchor=(-0.22, 0.5), 
                     frameon=True, fancybox=True, shadow=False, 
                     fontsize=9, title='Clique Sizes', title_fontsize=10)
    
    # Use subtitle position to label the colorbar
    # Position it above the colorbar on the right side
    fig.text(1.05, 1.05, 'Loop Persistence', 
            ha='right', va='top', fontsize=10)

    # Adjust layout to prevent overlap - give more space at top and left
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.15, right=0.92)
    
    # Save if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f'_{subject_id}' if subject_id else ''
        filepath = output_dir / f'combined_cliques_loops{suffix}.png'
        display.savefig(str(filepath), dpi=300, bbox_inches='tight')
        print(f"Saved combined visualization to {filepath}")
    
    if show_plot:
        plotting.show()
    else:
        display.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize brain topology: maximal cliques and persistence loops in 3D using nilearn',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects - save only (no display)
  python visualize_brain_topology.py -c clique_measures.csv -p h1_loops.csv -m mapping.csv
  
  # Process all subjects and display plots interactively
  python visualize_brain_topology.py -c clique_measures.csv -p h1_loops.csv -m mapping.csv --show
  
  # Show only top 20 largest cliques and top 10 loops
  python visualize_brain_topology.py -c cliques.csv -p loops.csv -m mapping.csv --top_k_cliques 20 --top_k_loops 10
  
  # Process specific subject only
  python visualize_brain_topology.py -c cliques.csv -p loops.csv -m mapping.csv --subject sub-0002 --show
  
  # Specify custom output directory
  python visualize_brain_topology.py -c cliques.csv -p loops.csv -m mapping.csv -o ./output
  
  # Change display mode (ortho, x, y, z, lyrz, etc.)
  python visualize_brain_topology.py -c cliques.csv -p loops.csv -m mapping.csv --display_mode lyrz
  
  # Combined view with both cliques and loops on same plot
  python visualize_brain_topology.py -c cliques.csv -p loops.csv -m mapping.csv --combined --show
        """
    )
    
    parser.add_argument('-c', '--clique_file', type=str,
                       help='Path to clique_measures.csv from clique_mapping.py')
    parser.add_argument('-p', '--persistence_file', type=str,
                       help='Path to h1_loops.csv from clique_persistence.py')
    parser.add_argument('-m', '--mapping_file', type=str,
                       help='Path to brain region mapping CSV file (default: ../data/mapping.csv)')
    parser.add_argument('-o', '--output_dir', type=str,
                       help='Output directory for saved plots')
    parser.add_argument('--subject', type=str,
                       help='Subject ID to filter and visualize (if not provided, processes all subjects separately)')
    parser.add_argument('--top_k_loops', type=int, default=10,
                       help='Number of top persistent loops to display (default: 10)')
    parser.add_argument('--top_k_cliques', type=int, default=50,
                       help='Number of largest cliques to display (default: 50)')
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively (default: only save to file)')
    parser.add_argument('--display_mode', type=str, default='ortho',
                       choices=['ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r', 'lr', 'lyrz', 'lzr', 'lzry', 'lyr'],
                       help='Nilearn display mode for brain slices (default: ortho)')
    parser.add_argument('--combined', action='store_true',
                       help='Create combined visualization with both cliques and loops on same plot (requires both -c and -p)')
    
    args = parser.parse_args()
    
    if not args.clique_file and not args.persistence_file:
        parser.error("At least one of --clique_file or --persistence_file must be provided")
    
    if args.combined and (not args.clique_file or not args.persistence_file):
        parser.error("--combined mode requires both --clique_file and --persistence_file")
    
    # Determine default mapping file if not provided
    if not args.mapping_file:
        script_dir = Path(__file__).parent
        args.mapping_file = str(script_dir.parent / 'data' / 'mapping.csv')
        print(f"Using default mapping file: {args.mapping_file}")
    
    # Load mapping and coordinates
    print(f"Loading brain region mapping from {args.mapping_file}...")
    mapping_df = pd.read_csv(args.mapping_file)
    print(f"  Loaded {len(mapping_df)} regions")
    
    print("Loading anatomical coordinates from mapping file...")
    coords_df = load_coordinates_from_mapping(mapping_df)
    print(f"  Loaded coordinates for {len(coords_df)} regions")
    print(f"  X range: [{coords_df['coord_x'].min():.2f}, {coords_df['coord_x'].max():.2f}]")
    print(f"  Y range: [{coords_df['coord_y'].min():.2f}, {coords_df['coord_y'].max():.2f}]")
    print(f"  Z range: [{coords_df['coord_z'].min():.2f}, {coords_df['coord_z'].max():.2f}]")
    
    # Setup output directory
    if args.output_dir is None:
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_output_name = f'topographic_visualization_{current_time}'
        default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'topographic_visualizations', default_output_name)
        output_dir = Path(default_output_dir)
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which subjects to process
    subjects_to_process = []
    
    # Load and process cliques
    cliques_df = None
    if args.clique_file:
        print(f"\nLoading clique data from {args.clique_file}...")
        cliques_df = pd.read_csv(args.clique_file)
        print(f"  Loaded {len(cliques_df)} cliques")
        
        if 'subject_id' in cliques_df.columns:
            clique_subjects = cliques_df['subject_id'].unique()
            subjects_to_process.extend(clique_subjects)
            print(f"  Found {len(clique_subjects)} subjects in clique data")
    
    # Load and process persistence loops
    loops_df = None
    if args.persistence_file:
        print(f"\nLoading persistence loop data from {args.persistence_file}...")
        loops_df = pd.read_csv(args.persistence_file)
        print(f"  Loaded {len(loops_df)} loops")
        
        if 'subject_id' in loops_df.columns:
            loop_subjects = loops_df['subject_id'].unique()
            subjects_to_process.extend(loop_subjects)
            print(f"  Found {len(loop_subjects)} subjects in loop data")
    
    # Get unique subjects
    subjects_to_process = sorted(set(subjects_to_process))
    
    # If --subject flag is provided, filter to that subject only
    if args.subject:
        if args.subject in subjects_to_process:
            subjects_to_process = [args.subject]
            print(f"\nProcessing only subject: {args.subject}")
        else:
            print(f"\nWarning: Subject {args.subject} not found in data. Available subjects: {subjects_to_process}")
            return
    
    if not subjects_to_process:
        print("\nNo subjects found with 'subject_id' column. Creating single plot for all data.")
        subjects_to_process = [None]
    else:
        print(f"\nProcessing {len(subjects_to_process)} subjects: {subjects_to_process}")
    
    # Process each subject separately
    for subject in subjects_to_process:
        if subject:
            print(f"\n{'='*60}")
            print(f"Processing subject: {subject}")
            print(f"{'='*60}")
        
        # Combined mode - plot both on same graph
        if args.combined:
            if cliques_df is not None and loops_df is not None:
                if subject:
                    subject_cliques = cliques_df[cliques_df['subject_id'] == subject]
                    subject_loops = loops_df[loops_df['subject_id'] == subject]
                    print(f"  Cliques: {len(subject_cliques)}")
                    print(f"  Loops: {len(subject_loops)}")
                else:
                    subject_cliques = cliques_df
                    subject_loops = loops_df
                    print(f"  Cliques: {len(subject_cliques)}")
                    print(f"  Loops: {len(subject_loops)}")
                
                if len(subject_cliques) > 0 and len(subject_loops) > 0:
                    print("  Creating combined visualization...")
                    plot_combined(cliques_df, loops_df, coords_df, subject, output_dir,
                                args.top_k_cliques, args.top_k_loops, args.show, args.display_mode)
                else:
                    print("  Insufficient data for combined plot")
        else:
            # Separate mode - plot cliques and loops separately
            # Plot cliques for this subject
            if cliques_df is not None:
                if subject:
                    subject_cliques = cliques_df[cliques_df['subject_id'] == subject]
                    print(f"  Cliques: {len(subject_cliques)}")
                else:
                    subject_cliques = cliques_df
                    print(f"  Cliques: {len(subject_cliques)}")
                
                if len(subject_cliques) > 0:
                    print("  Plotting cliques...")
                    plot_cliques(cliques_df, coords_df, subject, output_dir, 
                                   args.top_k_cliques, args.show, args.display_mode)
                else:
                    print("  No cliques found for this subject")
            
            # Plot persistence loops for this subject
            if loops_df is not None:
                if subject:
                    subject_loops = loops_df[loops_df['subject_id'] == subject]
                    print(f"  Loops: {len(subject_loops)}")
                else:
                    subject_loops = loops_df
                    print(f"  Loops: {len(subject_loops)}")
                
                if len(subject_loops) > 0:
                    print("  Plotting loops...")
                    plot_loops(loops_df, coords_df, subject, output_dir, 
                                              args.top_k_loops, args.show, args.display_mode)
                else:
                    print("  No loops found for this subject")
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
