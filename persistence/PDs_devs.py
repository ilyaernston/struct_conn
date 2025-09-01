"""
Developer Connectomes Persistence Analysis with Interactive UI

Created on Thu May  1 17:18:00 2025
@author: elijah

This script performs persistence analysis on expert vs naive developer connectivity data
with an interactive command-line interface for parameter selection.
"""

import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import persim
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import os
import re
import networkx as nx
import time
import pandas as pd
import random
import pickle
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import squareform
import argparse
from typing import Tuple, List, Dict, Optional, Union

# Import helper functions from separate modules
from .preprocessing import drop_cerebellum, connect_components, normalize_matrix
from .persistence_diagrams import compute_persistence, plot_persistence_diagram

# Set modern scientific style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "lines.linewidth": 2,
    "axes.titlepad": 14,
    "image.cmap": "viridis"
})

def compute_pds_distances(
    PDs_H0: List[np.ndarray], 
    PDs_H1: List[np.ndarray],
    subject_ids: List[str],
    group_name: str,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """Compute pairwise Wasserstein distances between persistence diagrams"""
    save_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    
    num_graphs = len(PDs_H0)
    
    print(f'Computing distances for {group_name} group ({num_graphs} subjects)')
    distance_matrix = np.zeros((num_graphs, num_graphs))
    
    for i in range(num_graphs):
        print(f'Processing subject {i+1} of {num_graphs}')
        for j in range(i, num_graphs):
            d0 = persim.sliced_wasserstein(PDs_H0[i], PDs_H0[j])
            d1 = persim.sliced_wasserstein(PDs_H1[i], PDs_H1[j])
            distance_matrix[i, j] = distance_matrix[j, i] = d0 + d1
            
        if (i + 1) % 50 == 0 or (i + 1) == num_graphs:
            dist_df = pd.DataFrame(distance_matrix, index=subject_ids, columns=subject_ids)
            fname = f'distance_matrix_{group_name}_{i+1}_subjects.csv'
            csv_path = os.path.join(save_dir, fname)
            dist_df.to_csv(csv_path)
            print(f'→ saved distance matrix to {csv_path}')
    
    dist_df = pd.DataFrame(distance_matrix, index=subject_ids, columns=subject_ids)
    return dist_df

def create_metadata_plots(emb, subject_ids, metadata_df, metadata_type, group_name, embedding_method, 
                         perp=None, plot_mode='save', save_dir='.'):
    """Helper function to create metadata-colored plots"""
    if metadata_df is None:
        return
    
    grouping_cols = [c for c in metadata_df.columns if c != 'subject_id']
    # Filter metadata to match embedding subjects
    available_subjects = metadata_df['subject_id'].astype(str)
    matching_subjects = [sid for sid in subject_ids if sid in available_subjects.values]
    
    if not matching_subjects:
        print(f"Warning: No matching subjects found for {metadata_type} metadata")
        return
    
    # Get indices of matching subjects in embedding
    subject_indices = [i for i, sid in enumerate(subject_ids) if sid in matching_subjects]
    
    # Filter metadata and embedding to matching subjects only
    filtered_metadata = metadata_df.set_index('subject_id').loc[matching_subjects, grouping_cols]
    filtered_emb = emb[subject_indices]
    
    for col in grouping_cols:
        values = filtered_metadata[col]
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Create scatter plot for each unique value
        for val in values.unique():
            mask = (values == val)
            ax.scatter(filtered_emb[mask, 0], filtered_emb[mask, 1], 
                      label=str(val), alpha=0.8, s=60)
        
        # Set title based on embedding method
        method_str = embedding_method.upper()
        if perp is not None:
            method_str += f" (pp={perp})"
        
        ax.set_title(f"{method_str} {group_name} embedding colored by {col} ({metadata_type})")
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.grid(alpha=0.3)
        ax.legend(title=col, bbox_to_anchor=(1, 1))
        
        if plot_mode in ('save', 'both'):
            filename_parts = [embedding_method.lower(), group_name, 'by', col, metadata_type]
            if perp is not None:
                filename_parts.insert(2, f'pp{perp}')
            fn = os.path.join(save_dir, '_'.join(filename_parts) + '.png')
            fig.savefig(fn, dpi=300, bbox_inches='tight')
            print(f"Saved {metadata_type} metadata plot → {fn}")
        if plot_mode in ('show', 'both'): 
            plt.show()
        plt.close(fig)

def cluster_and_visualize_distances(
    distance_matrix: pd.DataFrame,
    group_name: str,
    embedding_method: str = 'MDS',
    plot_mode: str = 'save',
    output_dir: Optional[str] = None,
    group_metadata_df: Optional[pd.DataFrame] = None,
    external_metadata_df: Optional[pd.DataFrame] = None,
    color_scheme: Optional[int] = None,
    k_number: Optional[int] = None,
    export_params: bool = True
):
    """
    Enhanced embedding and clustering analysis with flexible color-coding schemes
    
    Args:
        distance_matrix     : square pd.DataFrame (index=subject_id)
        group_name          : name of the group being analyzed
        embedding_method    : 'MDS' or 'TSNE'
        plot_mode           : 'save', 'show', or 'both'
        output_dir          : directory to write outputs
        group_metadata_df   : DataFrame with group info (expert/naive)
        external_metadata_df: DataFrame with external metadata columns
        color_scheme        : 1=group only, 2=external only, 3=both
        k_number            : k number to cluster into, computed automatically if None
        export_params       : if True, save CSV of embedding coords and cluster labels
    
    Returns:
        labels_df           : DataFrame with subject_id, embedding dims, and cluster labels
        meta_df             : DataFrame with metadata on analysis
    """
    save_dir = output_dir or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    subject_ids = distance_matrix.index.astype(str).tolist()
    D = distance_matrix.values
    n = D.shape[0]
    
    # Handle metadata based on color scheme
    group_meta = None
    external_meta = None
    group_cols = []
    external_cols = []
    
    if group_metadata_df is not None:
        group_cols = [c for c in group_metadata_df.columns if c != 'subject_id']
        group_meta = group_metadata_df.set_index('subject_id').loc[subject_ids, group_cols]
    
    if external_metadata_df is not None:
        external_cols = [c for c in external_metadata_df.columns if c != 'subject_id']
        # Filter external metadata to match subject_ids in distance matrix
        available_subjects = external_metadata_df['subject_id'].astype(str)
        matching_subjects = [sid for sid in subject_ids if sid in available_subjects.values]
        if matching_subjects:
            external_meta = external_metadata_df.set_index('subject_id').loc[matching_subjects, external_cols]
        else:
            print("Warning: No matching subjects found in external metadata")

    # Create subject-to-group mapping for export
    def get_subject_group(subject_id):
        """Determine if subject is expert or naive based on metadata"""
        if group_metadata_df is not None:
            # Look up in group metadata
            subject_row = group_metadata_df[group_metadata_df['subject_id'] == subject_id]
            if not subject_row.empty:
                return subject_row.iloc[0]['group']
        # Default fallback based on group_name
        return group_name if group_name in ['expert', 'naive'] else 'unknown'

    # Initialize results
    all_labels_dfs = []
    all_meta_dfs = []

    if embedding_method.upper() == 'TSNE':
        perps_to_test = [5, 15, 30, min(50, n-1)]
        for perp in perps_to_test:
            if perp >= n:
                continue
                
            tsne = TSNE(
                perplexity=perp,
                metric='precomputed',
                n_components=2,
                learning_rate='auto',
                init='random',
                random_state=42
            )
            emb = tsne.fit_transform(D)
            quality_text = f"KL={tsne.kl_divergence_:.3f}"
            
            # Create labels dataframe for this embedding
            labels_df = pd.DataFrame({'subject_id': subject_ids})
            labels_df['dim1'] = emb[:, 0]
            labels_df['dim2'] = emb[:, 1]

            # Determine k
            if k_number is not None:
                if isinstance(k_number, int):
                    best_k = k_number
                    best_bic = 'number of k set manually'
                else:
                    raise TypeError("Only integers are allowed as k_number")
            else:
                ks = np.arange(2, min(10, n-1) + 1)
                bic_scores = [GaussianMixture(n_components=k, random_state=42).fit(emb).bic(emb) for k in ks]
                bic_array = np.array(bic_scores)
                best_bic_idx = np.argmin(bic_array)
                best_k = int(ks[best_bic_idx])
                best_bic = bic_array[best_bic_idx]

            # Fit clusters
            gm = GaussianMixture(n_components=best_k, random_state=42).fit(emb)
            labels = gm.predict(emb)
            labels_df['cluster'] = labels

            # Add expert/naive group assignment for each subject
            labels_df['group'] = labels_df['subject_id'].apply(get_subject_group)
            
            # Create clean run_id without group name
            run_id = f"tsne_pp{perp}_k{best_k}"
            labels_df['run_id'] = run_id

            meta_df = pd.DataFrame([{
                'run_id': run_id,
                'group_analyzed': group_name,
                'embedding_method': f'tsne_pp{perp}',
                'quality': quality_text,
                'n_clusters': best_k,
                'bic_score': best_bic,
                'perplexity': perp
            }])

            # Plot clustering
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(
                emb[:, 0], emb[:, 1],
                c=labels, cmap='tab10', s=60,
                edgecolor='k', alpha=0.8
            )
            if isinstance(best_bic, str):
                ax.set_title(f"t-SNE {group_name} (pp={perp}) + GMM (k={best_k})\n{quality_text}")
            else:
                ax.set_title(f"t-SNE {group_name} (pp={perp}) + GMM (k={best_k})\n{quality_text}\nBIC={best_bic:.3f}")
            ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.grid(alpha=0.3)
            handles, _ = sc.legend_elements()
            ax.legend(handles, [str(i) for i in range(best_k)], title='cluster')
            
            if plot_mode in ('save', 'both'):
                fn = os.path.join(save_dir, f"tsne_{group_name}_pp{perp}_k{best_k}_clusters.png")
                fig.savefig(fn, dpi=300, bbox_inches='tight')
                print(f"Saved clustering plot → {fn}")
            if plot_mode in ('show', 'both'): 
                plt.show()
            plt.close(fig)

            # Create metadata-colored plots based on color scheme
            if color_scheme in [1, 3] and group_meta is not None:
                # Group-based coloring (expert vs naive)
                create_metadata_plots(emb, subject_ids, group_metadata_df, 'group', 
                                     group_name, 'tsne', perp, plot_mode, save_dir)
            
            if color_scheme in [2, 3] and external_meta is not None:
                # External metadata coloring
                create_metadata_plots(emb, subject_ids, external_metadata_df, 'external', 
                                     group_name, 'tsne', perp, plot_mode, save_dir)

            # Export parameters for this specific t-SNE run
            if export_params:
                export_path = os.path.join(save_dir, f'embedding_and_clustering_{group_name}_tsne_pp{perp}.csv')
                labels_df.to_csv(export_path, index=False)
                print(f"Exported t-SNE (pp={perp}) embedding and clustering → {export_path}")

                export_path_meta = os.path.join(save_dir, f'embedding_and_clustering_meta_{group_name}_tsne_pp{perp}.csv')
                meta_df.to_csv(export_path_meta, index=False)
                print(f"Exported t-SNE (pp={perp}) metadata → {export_path_meta}")

            all_labels_dfs.append(labels_df)
            all_meta_dfs.append(meta_df)

    else:  # MDS
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            normalized_stress='auto',
            random_state=42
        )
        emb = mds.fit_transform(D)
        i, j = np.triu_indices(n, k=1)
        d_orig = D[i, j]
        D_embed = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
        d_hat = D_embed[i, j]
        RSS, TSS = np.sum((d_orig - d_hat)**2), np.sum(d_orig**2)
        R2 = 1 - RSS/TSS
        stress = np.sqrt(RSS/TSS)
        quality_text = f"R²={R2:.3f}, Stress={stress:.3f}"

        # Create labels dataframe
        labels_df = pd.DataFrame({'subject_id': subject_ids})
        labels_df['dim1'] = emb[:, 0]
        labels_df['dim2'] = emb[:, 1]

        # Determine k
        if k_number is not None:
            if isinstance(k_number, int):
                best_k = k_number
                best_bic = 'number of k set manually'
            else:
                raise TypeError("Only integers are allowed as k_number")
        else:
            ks = np.arange(2, min(10, n-1) + 1)
            bic_scores = [GaussianMixture(n_components=k, random_state=42).fit(emb).bic(emb) for k in ks]
            bic_array = np.array(bic_scores)
            best_bic_idx = np.argmin(bic_array)
            best_k = int(ks[best_bic_idx])
            best_bic = bic_array[best_bic_idx]

        # Fit clusters
        gm = GaussianMixture(n_components=best_k, random_state=42).fit(emb)
        labels = gm.predict(emb)
        labels_df['cluster'] = labels

        # Add expert/naive group assignment for each subject
        labels_df['group'] = labels_df['subject_id'].apply(get_subject_group)
        
        # Create clean run_id without group name
        run_id = f"mds_k{best_k}"
        labels_df['run_id'] = run_id

        meta_df = pd.DataFrame([{
            'run_id': run_id,
            'group_analyzed': group_name,
            'embedding_method': 'mds',
            'quality': quality_text,
            'n_clusters': best_k,
            'bic_score': best_bic,
            'R2': R2,
            'stress': stress
        }])

        # Plot clustering
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=labels, cmap='tab10', s=60,
            edgecolor='k', alpha=0.8
        )
        if isinstance(best_bic, str):
            ax.set_title(f"MDS {group_name} + GMM (k={best_k})\n{quality_text}")
        else:
            ax.set_title(f"MDS {group_name} + GMM (k={best_k})\n{quality_text}\nBIC={best_bic:.3f}")
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.grid(alpha=0.3)
        handles, _ = sc.legend_elements()
        ax.legend(handles, [str(i) for i in range(best_k)], title='cluster')
        
        if plot_mode in ('save', 'both'):
            fn = os.path.join(save_dir, f"mds_{group_name}_k{best_k}_clusters.png")
            fig.savefig(fn, dpi=300, bbox_inches='tight')
            print(f"Saved clustering plot → {fn}")
        if plot_mode in ('show', 'both'): 
            plt.show()
        plt.close(fig)

        # Create metadata-colored plots based on color scheme
        if color_scheme in [1, 3] and group_meta is not None:
            # Group-based coloring (expert vs naive)
            create_metadata_plots(emb, subject_ids, group_metadata_df, 'group', 
                                 group_name, 'mds', None, plot_mode, save_dir)
        
        if color_scheme in [2, 3] and external_meta is not None:
            # External metadata coloring
            create_metadata_plots(emb, subject_ids, external_metadata_df, 'external', 
                                 group_name, 'mds', None, plot_mode, save_dir)

        # Export parameters for MDS
        if export_params:
            export_path = os.path.join(save_dir, f'embedding_and_clustering_{group_name}_mds.csv')
            labels_df.to_csv(export_path, index=False)
            print(f"Exported MDS embedding and clustering → {export_path}")

            export_path_meta = os.path.join(save_dir, f'embedding_and_clustering_meta_{group_name}_mds.csv')
            meta_df.to_csv(export_path_meta, index=False)
            print(f"Exported MDS metadata → {export_path_meta}")

        all_labels_dfs.append(labels_df)
        all_meta_dfs.append(meta_df)

    # Combine all results
    combined_labels_df = pd.concat(all_labels_dfs, ignore_index=True) if all_labels_dfs else pd.DataFrame()
    combined_meta_df = pd.concat(all_meta_dfs, ignore_index=True) if all_meta_dfs else pd.DataFrame()

    # Export combined parameters (in addition to individual files)
    if export_params and not combined_labels_df.empty:
        export_path = os.path.join(save_dir, f'embedding_and_clustering_{group_name}_combined.csv')
        combined_labels_df.to_csv(export_path, index=False)
        print(f"Exported combined embedding parameters and clusters → {export_path}")

        export_path_meta = os.path.join(save_dir, f'embedding_and_clustering_meta_{group_name}_combined.csv')
        combined_meta_df.to_csv(export_path_meta, index=False)
        print(f"Exported combined embedding metadata → {export_path_meta}")

    return combined_labels_df, combined_meta_df

### MAIN PROCESSING FUNCTION ###

def process_group_data(
    matrices: np.ndarray,
    names: np.ndarray,
    group_name: str,
    mapping: pd.DataFrame,
    output_dir: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
    """Process connectivity matrices for a single group (sampling handled earlier)"""
    
    print(f"\n=== Processing {group_name} group ===")
    print(f"Total subjects: {len(matrices)}")
    
    PDs_H0, PDs_H1, subject_ids, dropped_subjects = [], [], [], []
    processed = 0

    for idx, (matrix, subject_name) in enumerate(zip(matrices, names)):
        start_time = time.time()
        
        subject_id = str(subject_name)
        subject_ids.append(subject_id)
        
        # Zero diagonal
        np.fill_diagonal(matrix, 0)
        
        # Drop cerebellum
        matrix = drop_cerebellum(matrix, mapping)
        
        # Connect components
        graph = nx.from_numpy_array(matrix)
        graph, nodes = connect_components(graph, mapping)
        
        # Compute distance matrix
        distance_matrix = nx.floyd_warshall_numpy(graph)
        distance_matrix = normalize_matrix(distance_matrix)

        if np.max(distance_matrix) != 1:
            print(f'{subject_id} matrix invalid - dropping')
            dropped_subjects.append(subject_id)
            continue
        
        # Compute persistence
        h0, h1 = compute_persistence(distance_matrix, max_edge_length=1.0)
        PDs_H0.append(h0)
        PDs_H1.append(h1)
        processed += 1
        
        # Save intermediate results
        if processed % 50 == 0:
            fname = os.path.join(output_dir, f'PDs_{group_name}_up_to_{processed}.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({
                    'PDs_H0': PDs_H0,
                    'PDs_H1': PDs_H1,
                    'subject_ids': subject_ids,
                    'dropped': dropped_subjects,
                    'group': group_name
                }, f)
            print(f'→ Saved intermediate PDs to {fname}')
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f'Processed {subject_id} in {processing_time:.2f} sec (total: {processed}/{len(matrices)})')
    
    # Save final results
    final_fname = os.path.join(output_dir, f'PDs_{group_name}_final_{processed}.pkl')
    with open(final_fname, 'wb') as f:
        pickle.dump({
            'PDs_H0': PDs_H0,
            'PDs_H1': PDs_H1,
            'subject_ids': subject_ids,
            'dropped': dropped_subjects,
            'group': group_name
        }, f)
    print(f'→ Saved final PDs to {final_fname}')
    
    return PDs_H0, PDs_H1, subject_ids, dropped_subjects

def create_metadata_df(expert_names: np.ndarray, naive_names: np.ndarray) -> pd.DataFrame:
    """Create metadata DataFrame for expert and naive developer groups"""
    metadata_rows = []
    
    # Add expert developer subjects
    for name in expert_names:
        metadata_rows.append({
            'subject_id': str(name),
            'group': 'expert'
        })
    
    # Add naive developer subjects  
    for name in naive_names:
        metadata_rows.append({
            'subject_id': str(name),
            'group': 'naive'
        })
    
    return pd.DataFrame(metadata_rows)

### INTERACTIVE UI FUNCTIONS ###

def get_user_parameters():
    """Interactive parameter selection"""
    print("\n" + "="*60)
    print("  DEVELOPER CONNECTOMES PERSISTENCE ANALYSIS")
    print("="*60)
    
    # Group selection
    print("\nGroup Selection:")
    print("1. Expert developers only")
    print("2. Naive developers only") 
    print("3. Comparative analysis (expert vs naive developers)")
    
    while True:
        try:
            group_choice = int(input("\nSelect analysis type (1-3): "))
            if group_choice in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Sample size limitation
    print("\nSample Size:")
    print("Large datasets can take significant time to process.")
    limit_samples = input("Limit number of subjects per group? (y/n): ").lower().startswith('y')
    max_subjects = None
    
    if limit_samples:
        while True:
            try:
                max_subjects = int(input("Maximum subjects per group: "))
                if max_subjects > 0:
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
    
    # Embedding method
    print("\nEmbedding Method:")
    print("1. MDS (faster, deterministic)")
    print("2. t-SNE (slower, can reveal different patterns)")
    print("3. Both")
    
    while True:
        try:
            embed_choice = int(input("\nSelect embedding method (1-3): "))
            if embed_choice in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Output visualization
    print("\nVisualization:")
    print("1. Save plots only")
    print("2. Show plots interactively")
    print("3. Both save and show")
    
    while True:
        try:
            plot_choice = int(input("\nSelect visualization option (1-3): "))
            if plot_choice in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Manual cluster number
    print("\nClustering:")
    manual_k = input("Specify number of clusters manually? (y/n): ").lower().startswith('y')
    k_number = None
    
    if manual_k:
        while True:
            try:
                k_number = int(input("Number of clusters (k): "))
                if k_number > 1:
                    break
                else:
                    print("Please enter a number greater than 1")
            except ValueError:
                print("Please enter a valid number")
    
    # Export parameters
    print("\nData Export:")
    export_params = input("Export embedding coordinates and cluster labels? (y/n): ").lower().startswith('y')
    
    # Color-coding options (only for combined analysis)
    color_scheme = None
    external_metadata_path = None
    if group_choice == 3:
        print("\nColor-coding Options for Combined Analysis:")
        print("1. Group-based coloring (expert vs naive developers)")
        print("2. External metadata coloring (requires metadata file)")
        print("3. Both group and metadata coloring")
        
        while True:
            try:
                color_choice = int(input("\nSelect color-coding scheme (1-3): "))
                if color_choice in [1, 2, 3]:
                    color_scheme = color_choice
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Please enter a valid number")
        
        # If external metadata is needed, get the file path
        if color_scheme in [2, 3]:
            external_metadata_path = input("Enter path to external metadata CSV file: ").strip()
            if not external_metadata_path or not os.path.exists(external_metadata_path):
                print(f"Warning: Metadata file not found. Will use group-based coloring only.")
                external_metadata_path = None
                if color_scheme == 2:
                    color_scheme = 1  # Fall back to group coloring
                elif color_scheme == 3:
                    color_scheme = 1  # Fall back to group coloring only
    
    # Output directory
    default_output = "/Users/elijah/Desktop/thesis/struct_conn_developer_output"
    output_dir = input(f"\nOutput directory (press Enter for default: {default_output}): ").strip()
    if not output_dir:
        output_dir = default_output
    
    return {
        'group_choice': group_choice,
        'max_subjects': max_subjects,
        'embed_choice': embed_choice,
        'plot_choice': plot_choice,
        'k_number': k_number,
        'export_params': export_params,
        'color_scheme': color_scheme,
        'external_metadata_path': external_metadata_path,
        'output_dir': output_dir
    }

def run_analysis_with_ui():
    """Main analysis function with interactive UI"""
    
    # Get user parameters
    params = get_user_parameters()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'dev_connectomes')
    mapping_path = os.path.join(os.path.dirname(script_dir), 'graph_measures', 'mapping.csv')
    
    # Create output directory
    os.makedirs(params['output_dir'], exist_ok=True)
    
    # Load mapping
    mapping = pd.read_csv(mapping_path)
    
    # Load data
    print(f"\nLoading developer connectome data from {data_dir}...")
    expert_mats = np.load(os.path.join(data_dir, 'expert_mats.npy'))
    expert_names = np.load(os.path.join(data_dir, 'expert_names.npy'), allow_pickle=True)
    naive_mats = np.load(os.path.join(data_dir, 'naive_mats.npy'))
    naive_names = np.load(os.path.join(data_dir, 'naive_names.npy'), allow_pickle=True)
    
    print(f"Loaded expert developers: {len(expert_mats)} subjects")
    print(f"Loaded naive developers: {len(naive_mats)} subjects")
    
    # Process groups based on selection with proper sampling
    groups_to_process = []
    external_metadata = None
    
    # Apply sampling early if requested
    if params['max_subjects']:
        if len(expert_mats) > params['max_subjects']:
            expert_indices = np.random.choice(len(expert_mats), params['max_subjects'], replace=False)
            expert_mats = expert_mats[expert_indices]
            expert_names = expert_names[expert_indices]
            print(f"Sampled {params['max_subjects']} expert developers from {len(expert_indices)} total")
            
        if len(naive_mats) > params['max_subjects']:
            naive_indices = np.random.choice(len(naive_mats), params['max_subjects'], replace=False)
            naive_mats = naive_mats[naive_indices]
            naive_names = naive_names[naive_indices]
            print(f"Sampled {params['max_subjects']} naive developers from {len(naive_indices)} total")
    
    if params['group_choice'] == 1:  # Expert developers only
        groups_to_process.append(('expert', expert_mats, expert_names))
    elif params['group_choice'] == 2:  # Naive developers only
        groups_to_process.append(('naive', naive_mats, naive_names))
    elif params['group_choice'] == 3:  # Combined analysis
        # Now combine the already-sampled matrices and names
        combined_mats = np.concatenate([expert_mats, naive_mats], axis=0)
        combined_names = np.concatenate([expert_names, naive_names], axis=0)
        groups_to_process.append(('combined', combined_mats, combined_names))
        print(f"Combined analysis will use {len(expert_mats)} expert + {len(naive_mats)} naive = {len(combined_mats)} total subjects")
        
        # Load external metadata if provided
        if params['external_metadata_path']:
            try:
                external_metadata = pd.read_csv(params['external_metadata_path'])
                print(f"Loaded external metadata for {len(external_metadata)} subjects")
                # Ensure subject_id column exists
                if 'subject_id' not in external_metadata.columns:
                    print("Warning: External metadata must contain 'subject_id' column. Ignoring external metadata.")
                    external_metadata = None
            except Exception as e:
                print(f"Error loading external metadata: {e}. Using group-based coloring only.")
                external_metadata = None
    
    # Analysis results storage
    results = {}
    
    # Plot mode mapping
    plot_modes = {1: 'save', 2: 'show', 3: 'both'}
    plot_mode = plot_modes[params['plot_choice']]
    
    # Embedding method mapping
    embed_methods = {1: ['MDS'], 2: ['TSNE'], 3: ['MDS', 'TSNE']}
    methods = embed_methods[params['embed_choice']]
    
    # Process each group
    for group_name, matrices, names in groups_to_process:
        print(f"\n{'='*60}")
        print(f"PROCESSING {group_name.upper()} GROUP")
        print(f"{'='*60}")
        
        # Process connectivity data
        PDs_H0, PDs_H1, subject_ids, dropped = process_group_data(
            matrices, names, group_name, mapping, 
            params['output_dir']
        )
        
        if len(PDs_H0) == 0:
            print(f"No valid subjects found for {group_name} group!")
            continue
        
        # Plot sample persistence diagram
        print(f'\nPlotting sample persistence diagram for {group_name}')
        plot_persistence_diagram(
            PDs_H0[0], PDs_H1[0], 
            title=f"{group_name.title()} Group Sample PD",
            plot_mode=plot_mode,
            save_dir=params['output_dir']
        )
        
        # Compute distance matrix
        print(f'\nComputing distance matrix for {group_name}')
        distance_matrix = compute_pds_distances(
            PDs_H0, PDs_H1, subject_ids, group_name, params['output_dir']
        )
        
        # Clustering and visualization for each embedding method
        for method in methods:
            print(f'\nClustering and visualization using {method} for {group_name}')
            
            # Prepare metadata for color coding based on group
            group_metadata_df = None
            external_metadata_df = None
            
            if group_name == 'expert':
                # Create metadata for the actual processed expert subjects
                expert_subject_names = np.array(subject_ids)  # Use the sampled subject IDs
                group_metadata_df = create_metadata_df(expert_subject_names, np.array([]))
            elif group_name == 'naive':
                # Create metadata for the actual processed naive subjects  
                naive_subject_names = np.array(subject_ids)  # Use the sampled subject IDs
                group_metadata_df = create_metadata_df(np.array([]), naive_subject_names)
            elif group_name == 'combined':
                # Create metadata for the actual processed combined subjects
                # Determine which subjects are expert vs naive based on the original naming
                combined_expert_names = []
                combined_naive_names = []
                
                for subj_id in subject_ids:
                    # Check if this subject was originally from expert or naive group
                    if subj_id in [str(name) for name in expert_names]:
                        combined_expert_names.append(subj_id)
                    elif subj_id in [str(name) for name in naive_names]:
                        combined_naive_names.append(subj_id)
                
                group_metadata_df = create_metadata_df(
                    np.array(combined_expert_names), 
                    np.array(combined_naive_names)
                )
                print(f"Created developer group metadata for {len(group_metadata_df)} processed subjects")
                
                if params['external_metadata_path'] and 'external_metadata' in locals():
                    external_metadata_df = external_metadata
            
            labels_df, meta_df = cluster_and_visualize_distances(
                distance_matrix, 
                group_name, 
                method, 
                plot_mode, 
                params['output_dir'],
                group_metadata_df=group_metadata_df,
                external_metadata_df=external_metadata_df,
                color_scheme=params['color_scheme'],
                k_number=params['k_number'],
                export_params=params['export_params']
            )
            
            results[f'{group_name}_{method}'] = {
                'distance_matrix': distance_matrix,
                'labels': labels_df,
                'meta': meta_df,
                'PDs_H0': PDs_H0,
                'PDs_H1': PDs_H1,
                'subject_ids': subject_ids,
                'dropped_subjects': dropped
            }
    
    # Save combined results
    combined_results_path = os.path.join(params['output_dir'], 'combined_analysis_results.pkl')
    with open(combined_results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n→ Saved combined results to {combined_results_path}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE - SUMMARY")
    print(f"{'='*60}")
    
    for key, result in results.items():
        group_name = key.split('_')[0]
        method = '_'.join(key.split('_')[1:])
        n_subjects = len(result['subject_ids'])
        n_dropped = len(result['dropped_subjects'])
        print(f"{group_name.title()} group ({method}): {n_subjects} subjects analyzed, {n_dropped} dropped")
    
    print(f"\nAll outputs saved to: {params['output_dir']}")
    
    return results

### COMMAND LINE INTERFACE ###

def main():
    """Main function - can be run interactively or via command line"""
    parser = argparse.ArgumentParser(description='Developer Connectomes Persistence Analysis')
    parser.add_argument('--non-interactive', action='store_true', 
                        help='Run with default parameters (both developer groups, MDS, save only)')
    parser.add_argument('--output-dir', type=str, 
                        default='/Users/elijah/Desktop/thesis/struct_conn_developer_output',
                        help='Output directory')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Maximum subjects per group')
    
    args = parser.parse_args()
    
    if args.non_interactive:
        # Run with default parameters
        print("Running in non-interactive mode with default parameters...")
        
        # Setup paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), 'dev_connectomes')
        mapping_path = os.path.join(os.path.dirname(script_dir), 'graph_measures', 'mapping.csv')
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load data
        mapping = pd.read_csv(mapping_path)
        expert_mats = np.load(os.path.join(data_dir, 'expert_mats.npy'))
        expert_names = np.load(os.path.join(data_dir, 'expert_names.npy'), allow_pickle=True)
        naive_mats = np.load(os.path.join(data_dir, 'naive_mats.npy'))
        naive_names = np.load(os.path.join(data_dir, 'naive_names.npy'), allow_pickle=True)
        
        # Process both developer groups
        results = {}
        for group_name, matrices, names in [('expert', expert_mats, expert_names), 
                                           ('naive', naive_mats, naive_names)]:
            PDs_H0, PDs_H1, subject_ids, dropped = process_group_data(
                matrices, names, group_name, mapping, args.output_dir
            )
            
            if len(PDs_H0) > 0:
                distance_matrix = compute_pds_distances(
                    PDs_H0, PDs_H1, subject_ids, group_name, args.output_dir
                )
                labels_df, meta_df = cluster_and_visualize_distances(
                    distance_matrix, group_name, 'MDS', 'save', args.output_dir
                )
                results[group_name] = {
                    'distance_matrix': distance_matrix,
                    'labels': labels_df,
                    'meta': meta_df,
                    'subject_ids': subject_ids,
                    'dropped_subjects': dropped
                }
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
    else:
        # Run interactive mode
        results = run_analysis_with_ui()
    
    return results

if __name__ == "__main__":
    main()
