"""
Developer Connectomes Persistence Analysis with Interactive UI

Created on Thu May  1 17:18:00 2025
@author: elijah

This script performs persistence analysis on expert vs naive developer connectivity data
with an interactive command-line interface for parameter selection.
"""

import numpy as np
import matplotlib.pyplot as plt
import gudhi
import persim
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import networkx as nx
import time
import pandas as pd
import random
import pickle
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import squareform
import argparse
from typing import Tuple, List, Dict, Optional, Union
import sys
import os
# Import helper functions from submodules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import drop_cerebellum, connect_components, normalize_matrix
from persistence_diagrams import compute_persistence, plot_persistence_diagram

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
    output_dir: Optional[str] = None,
    dist_metric: str = 'wasserstein'
) -> pd.DataFrame:
    """Compute pairwise distances between persistence diagrams
    Args:
        PDs_H0       : list of H0 persistence diagrams (np.ndarray)
        PDs_H1       : list of H1 persistence diagrams (np.ndarray)
        subject_ids  : list of subject IDs corresponding to PDs
        group_name   : name of the group being analyzed
        output_dir   : directory to save intermediate distance matrices
        dist_metric  : distance metric to use ('wasserstein', 'bottleneck' or 'heat')
    """
    save_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    
    num_graphs = len(PDs_H0)
    
    print(f'Computing distances for {group_name} group ({num_graphs} subjects)')
    distance_matrix = np.zeros((num_graphs, num_graphs))
    
    if dist_metric == 'wasserstein':
        print('Using sliced Wasserstein distance metric')
        for i in range(num_graphs):
            print(f'Processing subject {i+1} of {num_graphs}')
            for j in range(i, num_graphs):
                d0 = gudhi.wasserstein.wasserstein_distance(PDs_H0[i], PDs_H0[j])
                d1 = gudhi.wasserstein.wasserstein_distance(PDs_H1[i], PDs_H1[j])
                distance_matrix[i, j] = distance_matrix[j, i] = d0 + d1
                
            if (i + 1) % 50 == 0 or (i + 1) == num_graphs:
                dist_df = pd.DataFrame(distance_matrix, index=subject_ids, columns=subject_ids)
                fname = f'distance_matrix_{group_name}_{i+1}_subjects.csv'
                csv_path = os.path.join(save_dir, fname)
                dist_df.to_csv(csv_path)
                print(f'Saved Wasserstein distance matrix to {csv_path}')
    elif dist_metric == 'bottleneck':
        print('Using bottleneck distance metric')
        for i in range(num_graphs):
            print(f'Processing subject {i+1} of {num_graphs}')
            for j in range(i, num_graphs):
                d0 = gudhi.bottleneck_distance(PDs_H0[i], PDs_H0[j])
                d1 = gudhi.bottleneck_distance(PDs_H1[i], PDs_H1[j])
                distance_matrix[i, j] = distance_matrix[j, i] = d0 + d1
                
            if (i + 1) % 50 == 0 or (i + 1) == num_graphs:
                dist_df = pd.DataFrame(distance_matrix, index=subject_ids, columns=subject_ids)
                fname = f'distance_matrix_{group_name}_{i+1}_subjects.csv'
                csv_path = os.path.join(save_dir, fname)
                dist_df.to_csv(csv_path)
                print(f'Saved Bottleneck distance matrix to {csv_path}')
    elif dist_metric == 'heat':
        print('Using heat kernel distance metric')
        for i in range(num_graphs):
            print(f'Processing subject {i+1} of {num_graphs}')
            for j in range(i, num_graphs):
                d0 = persim.heat(PDs_H0[i], PDs_H0[j])
                d1 = persim.heat(PDs_H1[i], PDs_H1[j])
                distance_matrix[i, j] = distance_matrix[j, i] = d0 + d1
                
            if (i + 1) % 50 == 0 or (i + 1) == num_graphs:
                dist_df = pd.DataFrame(distance_matrix, index=subject_ids, columns=subject_ids)
                fname = f'distance_matrix_{group_name}_{i+1}_subjects.csv'
                csv_path = os.path.join(save_dir, fname)
                dist_df.to_csv(csv_path)
                print(f'Saved Heat Kernel distance matrix to {csv_path}')

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
        ax.set_aspect('equal', adjustable='box')
        
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

            # Plot clustering with special handling for both groups
            if group_name == 'both_groups' and group_metadata_df is not None:
                # Special combined plot with group=color, cluster=shape
                plot_combined_embedding_with_groups_and_clusters(
                    emb, labels_df, f"t-SNE (pp={perp})", quality_text, best_k, best_bic, 
                    plot_mode, save_dir, perp
                )
            else:
                # Standard clustering plot
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
                ax.set_aspect('equal', adjustable='box')
                
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

        # Plot clustering with special handling for both groups
        if group_name == 'both_groups' and group_metadata_df is not None:
            # Special combined plot with group=color, cluster=shape
            plot_combined_embedding_with_groups_and_clusters(
                emb, labels_df, "MDS", quality_text, best_k, best_bic, 
                plot_mode, save_dir, None
            )
        else:
            # Standard clustering plot
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
            ax.set_aspect('equal', adjustable='box')
            
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

def plot_combined_persistence_diagrams(
    expert_PDs_H0: List[np.ndarray], 
    expert_PDs_H1: List[np.ndarray],
    naive_PDs_H0: List[np.ndarray], 
    naive_PDs_H1: List[np.ndarray],
    plot_mode: str = 'save',
    save_dir: str = '.'
):
    """Plot combined persistence diagrams for both expert and naive groups with color coding"""
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Track whether labels have been added
    expert_h0_labeled = False
    naive_h0_labeled = False
    expert_h1_labeled = False
    naive_h1_labeled = False
    
    # Plot H0 diagrams
    for pd_h0 in expert_PDs_H0:
        if len(pd_h0) > 0:
            label = 'Expert' if not expert_h0_labeled else None
            ax0.scatter(pd_h0[:, 0], pd_h0[:, 1], c='tab:blue', alpha=0.5, s=10, label=label)
            expert_h0_labeled = True
    
    for pd_h0 in naive_PDs_H0:
        if len(pd_h0) > 0:
            label = 'Naive' if not naive_h0_labeled else None
            ax0.scatter(pd_h0[:, 0], pd_h0[:, 1], c='tab:orange', alpha=0.5, s=10, label=label)
            naive_h0_labeled = True
    
    # Plot H1 diagrams
    for pd_h1 in expert_PDs_H1:
        if len(pd_h1) > 0:
            label = 'Expert' if not expert_h1_labeled else None
            ax1.scatter(pd_h1[:, 0], pd_h1[:, 1], c='tab:blue', alpha=0.5, s=10, label=label)
            expert_h1_labeled = True
    
    for pd_h1 in naive_PDs_H1:
        if len(pd_h1) > 0:
            label = 'Naive' if not naive_h1_labeled else None
            ax1.scatter(pd_h1[:, 0], pd_h1[:, 1], c='tab:orange', alpha=0.5, s=10, label=label)
            naive_h1_labeled = True
    
    # Add diagonal lines and set axis properties
    max_val = 1.0
    min_val = -0.1
    for ax in [ax0, ax1]:
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    ax0.set_title('H0 Persistence Diagrams (Expert vs Naive)')
    ax1.set_title('H1 Persistence Diagrams (Expert vs Naive)')
    
    plt.tight_layout()
    
    if plot_mode in ('save', 'both'):
        filename = os.path.join(save_dir, 'combined_persistence_diagrams_both_groups.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"→ Saved combined persistence diagram to {filename}")
    
    if plot_mode in ('show', 'both'):
        plt.show()
    
    plt.close(fig)

def plot_combined_embedding_with_groups_and_clusters(
    emb: np.ndarray,
    labels_df: pd.DataFrame,
    method_name: str,
    quality_text: str,
    n_clusters: int,
    bic_score: Union[float, str],
    plot_mode: str,
    save_dir: str,
    perp: Optional[int] = None
):
    """Plot embedding with group=color and cluster=shape for combined analysis"""
    
    # Define marker styles for clusters
    cluster_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', '|', '_']
    
    # Define colors for groups
    group_colors = {'expert': 'tab:blue', 'naive': 'tab:orange'}
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each group-cluster combination
    for group in labels_df['group'].unique():
        group_mask = labels_df['group'] == group
        group_data = labels_df[group_mask]
        
        for cluster in sorted(group_data['cluster'].unique()):
            cluster_mask = group_data['cluster'] == cluster
            cluster_data = group_data[cluster_mask]
            
            if len(cluster_data) > 0:
                marker = cluster_markers[cluster % len(cluster_markers)]
                color = group_colors.get(group, 'tab:gray')
                
                # Get corresponding embedding coordinates
                indices = cluster_data.index
                ax.scatter(
                    emb[indices, 0], emb[indices, 1],
                    c=color, marker=marker, s=80, alpha=0.7,
                    edgecolor='black', linewidth=0.5,
                    label=f'{group.title()} Cluster {cluster}'
                )
    
    # Set title based on method
    title_parts = [method_name, "Both Groups + GMM", f"(k={n_clusters})"]
    if perp is not None:
        title_parts.insert(1, f"(pp={perp})")
    
    title = " ".join(title_parts) + f"\n{quality_text}"
    if not isinstance(bic_score, str):
        title += f"\nBIC={bic_score:.3f}"
    
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Create legend with two parts: groups and clusters
    import matplotlib.lines as mlines
    
    # Group legend elements
    expert_line = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None',
                               markersize=8, label='Expert')
    naive_line = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='None',
                              markersize=8, label='Naive')
    
    # Cluster legend elements  
    cluster_lines = []
    for i in range(n_clusters):
        marker = cluster_markers[i % len(cluster_markers)]
        cluster_lines.append(
            mlines.Line2D([], [], color='tab:gray', marker=marker, linestyle='None',
                         markersize=8, label=f'Cluster {i}')
        )
    
    # Create two legends
    legend1 = ax.legend(handles=[expert_line, naive_line], 
                       title='Groups', loc='upper left', bbox_to_anchor=(1.02, 1))
    legend2 = ax.legend(handles=cluster_lines,
                       title='Clusters', loc='upper left', bbox_to_anchor=(1.02, 0.7))
    
    # Add both legends
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if plot_mode in ('save', 'both'):
        method_lower = method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        if perp is not None:
            fn = os.path.join(save_dir, f"{method_lower}_pp{perp}_both_groups_k{n_clusters}_combined.png")
        else:
            fn = os.path.join(save_dir, f"{method_lower}_both_groups_k{n_clusters}_combined.png")
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        print(f"Saved combined embedding plot → {fn}")
    
    if plot_mode in ('show', 'both'):
        plt.show()
    
    plt.close(fig)

### COMMAND LINE INTERFACE ###

def main():
    """Main function to parse arguments and run analysis"""

    # Determine default data directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'dev_connectomes')
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output', 'dev_connectomes_analysis')
    default_mapping_file = os.path.join(os.path.dirname(script_dir), 'data', 'mapping.csv')

    parser = argparse.ArgumentParser(description='Developer Connectomes Persistence Analysis')
    
    # Analysis type
    parser.add_argument('--group', type=str, required=True,
                        choices=['expert', 'naive', 'both'],
                        help='Group to analyze: expert, naive, or both')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Maximum subjects per group')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, 
                        default=default_data_dir,
                        help='Directory containing connectivity data: expert_mats.npy, naive_mats.npy, expert_names.npy, naive_names.npy')
    parser.add_argument('--output-dir', type=str, 
                        default=default_output_dir,
                        help='Output directory')
    parser.add_argument('--mapping-file', type=str, 
                        default=default_mapping_file,
                        help='Path to mapping CSV file (if not specified, uses default)')
    
    # Analysis parameters
    parser.add_argument('--distance-metric', type=str, default='bottleneck',
                        choices=['bottleneck', 'wasserstein', 'heat'],
                        help='Metric to use for inter-PDs distance matrix')
    parser.add_argument('--embedding', type=str, default='mds',
                        choices=['mds', 'tsne', 'both'],
                        help='Embedding method: mds, tsne, or both')
    parser.add_argument('--k-clusters', type=int, default=None,
                        help='Number of clusters (if not specified, will be determined automatically)')
    
    # Output parameters
    parser.add_argument('--plot-mode', type=str, default='save',
                        choices=['save', 'show', 'both'],
                        help='How to handle plots: save, show, or both')
    parser.add_argument('--export-params', action='store_true',
                        help='Export embedding coordinates and cluster labels to CSV')
    
    args = parser.parse_args()
    
    print(f"Running Developer Connectomes Persistence Analysis")
    print(f"Group: {args.group}")
    print(f"Distance metric: {args.distance_metric}")
    print(f"Embedding: {args.embedding}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup paths
    data_dir = args.data_dir
    output_dir = args.output_dir
    mapping_path = args.mapping_file

    # Insure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_dir}...")
    mapping = pd.read_csv(mapping_path)
    expert_mats = np.load(os.path.join(data_dir, 'expert_mats.npy'))
    expert_names = np.load(os.path.join(data_dir, 'expert_names.npy'), allow_pickle=True)
    naive_mats = np.load(os.path.join(data_dir, 'naive_mats.npy'))
    naive_names = np.load(os.path.join(data_dir, 'naive_names.npy'), allow_pickle=True)
    
    print(f"Loaded {len(expert_mats)} expert and {len(naive_mats)} naive developers")
    
    # Apply sampling if requested
    if args.max_subjects:
        if len(expert_mats) > args.max_subjects:
            expert_indices = np.random.choice(len(expert_mats), args.max_subjects, replace=False)
            expert_mats = expert_mats[expert_indices]
            expert_names = expert_names[expert_indices]
            print(f"Sampled {args.max_subjects} expert developers")
            
        if len(naive_mats) > args.max_subjects:
            naive_indices = np.random.choice(len(naive_mats), args.max_subjects, replace=False)
            naive_mats = naive_mats[naive_indices]
            naive_names = naive_names[naive_indices]
            print(f"Sampled {args.max_subjects} naive developers")
    
    # Determine embedding methods
    if args.embedding == 'mds':
        methods = ['MDS']
    elif args.embedding == 'tsne':
        methods = ['TSNE'] 
    elif args.embedding == 'both':
        methods = ['MDS', 'TSNE']
    else:
        methods = ['MDS']  # Default fallback
    
    # Process groups based on selection
    results = {}
    
    if args.group == 'both':
        # Combined processing for both groups
        print(f"\n{'='*60}")
        print("PROCESSING BOTH GROUPS TOGETHER")
        print(f"{'='*60}")
        
        # Process expert group
        print("\nProcessing expert group...")
        expert_PDs_H0, expert_PDs_H1, expert_subject_ids, expert_dropped = process_group_data(
            expert_mats, expert_names, 'expert', mapping, args.output_dir
        )
        
        # Process naive group
        print("\nProcessing naive group...")
        naive_PDs_H0, naive_PDs_H1, naive_subject_ids, naive_dropped = process_group_data(
            naive_mats, naive_names, 'naive', mapping, args.output_dir
        )
        
        if len(expert_PDs_H0) == 0 or len(naive_PDs_H0) == 0:
            print("Error: Need valid subjects from both groups!")
            return {}
        
        # Combine all persistence diagrams
        all_PDs_H0 = expert_PDs_H0 + naive_PDs_H0
        all_PDs_H1 = expert_PDs_H1 + naive_PDs_H1
        all_subject_ids = expert_subject_ids + naive_subject_ids
        all_groups = ['expert'] * len(expert_subject_ids) + ['naive'] * len(naive_subject_ids)
        
        # Save combined PDs
        combined_fname = os.path.join(args.output_dir, f'PDs_both_groups_combined.pkl')
        with open(combined_fname, 'wb') as f:
            pickle.dump({
                'PDs_H0': all_PDs_H0,
                'PDs_H1': all_PDs_H1,
                'subject_ids': all_subject_ids,
                'groups': all_groups,
                'expert_dropped': expert_dropped,
                'naive_dropped': naive_dropped
            }, f)
        print(f'→ Saved combined PDs to {combined_fname}')
        
        # Plot combined persistence diagrams
        print('\nPlotting combined persistence diagrams...')
        plot_combined_persistence_diagrams(
            expert_PDs_H0, expert_PDs_H1, naive_PDs_H0, naive_PDs_H1,
            plot_mode=args.plot_mode, save_dir=args.output_dir
        )
        
        # Compute combined distance matrix
        print('\nComputing combined distance matrix...')
        combined_distance_matrix = compute_pds_distances(
            PDs_H0=all_PDs_H0, 
            PDs_H1=all_PDs_H1, 
            subject_ids=all_subject_ids, 
            group_name='both_groups', 
            output_dir=args.output_dir,
            dist_metric=args.distance_metric
        )
        
        # Create metadata DataFrame for group information
        group_metadata_df = pd.DataFrame({
            'subject_id': all_subject_ids,
            'group': all_groups
        })
        
        # Clustering and visualization for each embedding method
        for method in methods:
            print(f'\nClustering and visualization using {method} for combined groups')
            
            labels_df, meta_df = cluster_and_visualize_distances(
                combined_distance_matrix, 
                'both_groups', 
                method, 
                args.plot_mode, 
                args.output_dir,
                group_metadata_df=group_metadata_df,
                external_metadata_df=None,
                color_scheme=1,  # Use group-based coloring
                k_number=args.k_clusters,
                export_params=args.export_params
            )
            
            results[f'both_groups_{method}'] = {
                'distance_matrix': combined_distance_matrix,
                'labels': labels_df,
                'meta': meta_df,
                'PDs_H0': all_PDs_H0,
                'PDs_H1': all_PDs_H1,
                'subject_ids': all_subject_ids,
                'groups': all_groups,
                'expert_dropped': expert_dropped,
                'naive_dropped': naive_dropped
            }
    
    else:
        # Single group processing
        groups_to_process = []
        if args.group == 'expert':
            groups_to_process.append(('expert', expert_mats, expert_names))
        elif args.group == 'naive':
            groups_to_process.append(('naive', naive_mats, naive_names))
        
        for group_name, matrices, names in groups_to_process:
            print(f"\n{'='*60}")
            print(f"PROCESSING {group_name.upper()} GROUP")
            print(f"{'='*60}")
            
            # Process connectivity data
            PDs_H0, PDs_H1, subject_ids, dropped = process_group_data(
                matrices, names, group_name, mapping, args.output_dir
            )
            
            if len(PDs_H0) == 0:
                print(f"No valid subjects found for {group_name} group!")
                continue
            
            # Plot sample persistence diagram
            print(f'\nPlotting sample persistence diagram for {group_name}')
            plot_persistence_diagram(
                PDs_H0[0], PDs_H1[0], 
                title=f"{group_name.title()} Group Sample PD",
                plot_mode=args.plot_mode,
                save_dir=args.output_dir
            )
            
            # Compute distance matrix
            print(f'\nComputing distance matrix for {group_name}')
            distance_matrix = compute_pds_distances(
                PDs_H0=PDs_H0, 
                PDs_H1=PDs_H1, 
                subject_ids=subject_ids, 
                group_name=group_name, 
                output_dir=args.output_dir,
                dist_metric=args.distance_metric
            )
            
            # Clustering and visualization for each embedding method
            for method in methods: # type: ignore
                print(f'\nClustering and visualization using {method} for {group_name}')
                
                labels_df, meta_df = cluster_and_visualize_distances(
                    distance_matrix, 
                    group_name, 
                    method, 
                    args.plot_mode, 
                    args.output_dir,
                    group_metadata_df=None,
                    external_metadata_df=None,
                    color_scheme=None,
                    k_number=args.k_clusters,
                    export_params=args.export_params
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
    combined_results_path = os.path.join(args.output_dir, 'analysis_results.pkl')
    with open(combined_results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n→ Saved results to {combined_results_path}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE - SUMMARY")
    print(f"{'='*60}")
    
    for key, result in results.items():
        group_name = key.split('_')[0]
        method = '_'.join(key.split('_')[1:])
        n_subjects = len(result['subject_ids'])
        
        # Handle different key names for dropped subjects
        if 'dropped_subjects' in result:
            n_dropped = len(result['dropped_subjects'])
        elif 'expert_dropped' in result and 'naive_dropped' in result:
            n_dropped = len(result['expert_dropped']) + len(result['naive_dropped'])
        else:
            n_dropped = 0
            
        print(f"{group_name.title()} group ({method}): {n_subjects} subjects analyzed, {n_dropped} dropped")
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    
    return results

if __name__ == "__main__":
    main()
