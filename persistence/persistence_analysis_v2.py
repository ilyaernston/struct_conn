#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:18:00 2025

@author: elijah
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

def normalize_matrix(matrix):
    """Normalize MRI matrix to [0, 1] range"""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

def largest_connected_component(matrix):
    """Extract largest connected component from adjacency matrix"""
    # Binarize matrix for connectivity check
    binarized = (matrix > 0).astype(int)
    n_components, labels = connected_components(csr_matrix(binarized), directed=False)
    
    if n_components == 1:
        return matrix
    
    # Find largest component
    _, counts = np.unique(labels, return_counts=True)
    largest_idx = np.argmax(counts)
    component_mask = (labels == largest_idx)
    return matrix[component_mask][:, component_mask]

def compute_persistence(distance_matrix, max_edge_length=1.0):
    """Compute H0 and H1 persistence diagrams"""
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()
    
    H0, H1 = [], []
    max_death = max_edge_length
    
    for interval in persistence:
        dim = interval[0]
        birth, death = interval[1]
        
        if death == float('inf'):
            death = max_death
            
        if dim == 0:
            H0.append((birth, death))
        elif dim == 1:
            H1.append((birth, death))
    
    return np.array(H0), np.array(H1)

import os
import matplotlib.pyplot as plt

def plot_persistence_diagram(
    H0: np.ndarray,
    H1: np.ndarray,
    plot_mode: str = "save",
    save_dir: str = None
):
    """
    Visualize H0 and H1 persistence diagrams, either saving to file or displaying.

    Args:
        H0, H1       : Arrays of shape (n_points, 2), birth/death coords.
        title (str)  : Figure title.
        plot_mode    : 'save' (default), 'show', or 'both'.
        save_dir     : Directory to save the figure (defaults to script dir).
    """
    # Determine save directory
    save_dir = save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(H0) > 0:
        ax.scatter(
            H0[:, 0], H0[:, 1],
            c='#1f77b4', marker='^', s=80,
            label='H0', alpha=0.7, edgecolor='w'
        )
    if len(H1) > 0:
        ax.scatter(
            H1[:, 0], H1[:, 1],
            c='#ff7f0e', marker='o', s=80,
            label='H1', alpha=0.7, edgecolor='w'
        )

    # Diagonal line
    all_deaths = []
    if len(H0) > 0: all_deaths.extend(H0[:, 1])
    if len(H1) > 0: all_deaths.extend(H1[:, 1])
    max_val = max([1.0] + all_deaths)
    ax.plot([0, max_val], [0, max_val], '--', color='#2ca02c', alpha=0.7)

    ax.set(xlabel='Birth', ylabel='Death', title="Persistence Diagram")
    ax.legend()
    plt.tight_layout()

    # Save and/or show
    fname = os.path.join(
        save_dir,
        "PDs.png"
    )
    if plot_mode in ('save', 'both'):
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved persistence diagram to {fname}")
    if plot_mode in ('show', 'both'):
        plt.show()

    plt.close(fig)

from sklearn.mixture import GaussianMixture
from matplotlib.cm import tab20
from scipy.spatial.distance import squareform

def compute_pds_distences(
        PDs_H0 : list, 
        PDs_H1 : list,
        output_dir : str = None):
    '''
    Computes pairwise intersubject Wasserstien distances based on H0 and H1 PDs' birth-death lists.
    Intermediate matrices saved as .npy in defined directory, or script directory if none provided.

    Args:
        PDs_H0 (list): list of H0 PDs features (connected components) in for every subject.
        Features are stored in np.array of shape (n0_i, 2) (n0_i = number of H0‐features for subject i), 
        and rows are (birth, death) pairs
        PDs_H1 (list): list of H1 PDs features (loops) in for every subject.
        Features are stored in np.array of shape (n1_i, 2) (n1_i = number of H1‐features for subject i), and rows are (birth, death) pairs

    Returns:
        distance_matrix (np.array): 2d-array of pairwise intersubject Wasserstein distances
    '''
    
    save_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    
    num_graphs = len(PDs_H0)
    
    # Compute distance matrix
    print('Computing distances')
    distance_matrix = np.zeros((num_graphs, num_graphs))
    for i in range(num_graphs):
        print(f'Processing data for subject {i} of {num_graphs}')
        for j in range(i, num_graphs):
            d0 = persim.sliced_wasserstein(PDs_H0[i], PDs_H0[j])
            d1 = persim.sliced_wasserstein(PDs_H1[i], PDs_H1[j])
            distance_matrix[i, j] = distance_matrix[j, i] = d0 + d1
            
        # save after every 50 subjects, and once at the end
        if (i + 1) % 50 == 0 or (i + 1) == num_graphs:
            fname = f'distance_matrix_{i+1}_subjects.npy'
            fdir = os.path.join(save_dir, fname)
            np.save(fdir, distance_matrix)
            print(f'Saved intermediate matrix to "{fname}"')
    
    return distance_matrix

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import squareform

def cluster_and_visualize_distances(
    distance_matrix: np.ndarray,
    embedding_method: str = 'MDS',
    plot_mode: str = 'save',
    output_dir: str = None
):
    """
    1) Embed distances into 2D (MDS or t-SNE)
    2) Compute embedding quality (R² & stress for MDS; KL-divergence for t-SNE)
    3) Choose top-3 cluster counts via BIC
    4) For each of those 3 k’s, plot the embedding colored by cluster and save/show
    5) Save an (N,3) array of labelings for each top-k as .csv
    """
    n = distance_matrix.shape[0]
    save_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # perform embedding
    if embedding_method.upper() == 'TSNE':
        tsne = TSNE(
            n_components=2,
            metric='precomputed',
            learning_rate='auto',
            init='random',
            random_state=42
        )
        embeddings = tsne.fit_transform(distance_matrix)
        quality_text = f"KL={tsne.kl_divergence_:.3f}"
    else:
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            normalized_stress='auto',
            random_state=42
        )
        embeddings = mds.fit_transform(distance_matrix)

        # compute R² & stress
        i, j = np.triu_indices(n, k=1)
        d_orig = distance_matrix[i, j]
        diffs = embeddings[:, None, :] - embeddings[None, :, :]
        D_embed = np.linalg.norm(diffs, axis=2)
        d_hat = D_embed[i, j]
        RSS = np.sum((d_orig - d_hat) ** 2)
        TSS = np.sum(d_orig ** 2)
        R2 = 1 - RSS / TSS
        stress = np.sqrt(RSS / TSS)
        quality_text = f"R²={R2:.3f}, Stress={stress:.3f}"

    # find top-3 ks via BIC
    max_k = min(10, n - 1)
    ks = np.arange(2, max_k + 1)
    bic = []
    for k in ks:
        g = GaussianMixture(n_components=k, random_state=42).fit(embeddings)
        bic.append(g.bic(embeddings))
    bic = np.array(bic)
    top3_idx = np.argsort(bic)[:3]
    top3_ks = ks[top3_idx]

    # prepare output array
    cluster_labels = np.zeros((n, 3), dtype=int)

    # plot for each k
    for col, k in enumerate(top3_ks):
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(embeddings)
        cluster_labels[:, col] = labels

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap='tab10',
            s=60,
            edgecolor='k',
            alpha=0.8
        )
        ax.set_title(f"{embedding_method.upper()} + GMM (k={k})\n{quality_text}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(alpha=0.3)

        fname = os.path.join(
            save_dir,
            f"embedding_{embedding_method.lower()}_k{k}_clusters.png"
        )
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved plot for k={k} → {fname}")

        if plot_mode in ('show', 'both'):
            plt.show()
        plt.close(fig)
    
    # save clusterings as .csv
    csv_name = os.path.join(save_dir, 'PDs_labels.csv')
    np.savetxt(csv_name, cluster_labels, delimiter=',')
    print(f'→ Saved cluterings to {csv_name}')


    
import os
import re
import networkx as nx
import time
import pandas as pd
import random
import pickle

def compute_distance_matrix(adj_matrix, mode):
    
    if mode == 'scipy':
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
    
        # W is your (n×n) numpy array of weights in (0,1), zeros = no edge
        # transform a sparse graph
        G = csr_matrix(adj_matrix)
        # compute all-pairs shortest paths
        dist_matrix = shortest_path(csgraph=G, directed=False, unweighted=False)
        
    elif mode == 'nx':
        
        # build the graph ( skip zero entries)
        G = nx.Graph()
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                w = adj_matrix[i,j]
                if w > 0:
                    G.add_edge(i, j, weight=w)
        
        # (a) Floyd–Warshall → full distance matrix
        D_fw = nx.floyd_warshall_numpy(G, weight='weight')
        # (b) Dijkstra per node (returns dict of dicts)
        D_dij = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        # to get a numpy array from D_dij:
        D = np.full((n,n), np.inf)
        for i, distdict in D_dij.items():
            for j, dij in distdict.items():
                D[i,j] = dij
        dist_matrix = D

    return dist_matrix

def drop_cerebellum(matrix, mapping):
    
    to_drop = mapping.loc[mapping['Lobe'] == 'Cerebellum', 'index'].values
     
    # build a boolean mask of size N
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[to_drop] = False
     
    # filter the matrix: keep only rows & cols where mask==True
    filtered_matrix = matrix[np.ix_(mask, mask)]  
    
    return filtered_matrix

def connect_components(graph, mapping):
    '''
    Checks if graph has multiple connected components.
    If so, connects smaller components to the anatomacallly closest node in the largest component .
    Anatomical data is obtained from the mapping file. File should be stored in the script's directory and called 'mapping.csv'
     
    
    Parameters
    ----------
    graph : nx.Graph
        networkx graph to process
    mapping : str
        path to mapping file in .csv
        
    Returns
    ----------    
    connected_graph : nx.Graph
        processed graph
    conn_nodes : list
        list of dicts with pairs of nodes, between which edges were constructed
    
    '''
    
    conn_nodes_list = []
    
    # check for unconnected components
    comps = list(nx.connected_components(graph)) # list connected components
    n_components = nx.number_connected_components(graph)
    
    if n_components > 1:
        main_comp = max(comps, key=len)
        main_nodes = set(main_comp)
    
        for comp in comps:
            if comp is main_comp:
                continue
            else:
                # pick a random node in the smaller component and define it'd position
                u = random.choice(list(comp))
                attr_u = mapping.loc[u]
                hemi_u  = attr_u['Hemi']
                gyrus_u = attr_u['Gyrus']
                lobe_u  = attr_u['Lobe']
        
                # restrict main‐component nodes to same hemisphere
                main_attrs = mapping.loc[list(main_nodes)]
                same_hemi = main_attrs[main_attrs['Hemi'] == hemi_u]
        
                # try to connect within same gyrus
                candidates = same_hemi[same_hemi['Gyrus'] == gyrus_u].index.tolist()
                # if no nodes in same gyrus, try same lobe
                if not candidates:
                    candidates = same_hemi[same_hemi['Lobe'] == lobe_u].index.tolist()
                # if no nodes in the same lobe, connect to any same‐hemi node
                if not candidates:
                    candidates = same_hemi.index.tolist()
                # last option: connect to any main node
                if not candidates:
                    candidates = list(main_nodes)
        
                v = random.choice(candidates)
        
                # compute average edge weight among candidate region in the main comp
                matrix = nx.to_numpy_array(graph)
                sub_idx = candidates
                submat  = matrix[np.ix_(sub_idx, sub_idx)]
                weights = submat[submat > 0]
                avg_w   = float(weights.mean()) if weights.size else float(matrix.mean())
        
                # add the edge
                graph.add_edge(u, v, weight=avg_w)
                
                '''
                # print data on added edge
                attr_v = mapping_df.loc[v]
                gyrus_v = attr_v['Gyrus']
                lobe_v = attr_v['Lobe']
                if gyrus_u == gyrus_v:
                    print(f'Added the edge of weight {avg_w} between nodes {u} from and {v} in {gyrus_u} of {hemi_u} hemisphere')
                elif lobe_u == lobe_v:
                    print(f'Added the edge of weight {avg_w} between nodes {u} from and {v} in {lobe_u} of {hemi_u} hemisphere')
                else:
                    print('Added a shitty edge between nodes {u} from and {v}')
                '''
                
                conn_nodes_list.append({u:v})
        
        # rebuild graph with addaed connections
        connected_graph = graph
    else:
        conn_nodes_list.append(np.nan)
        connected_graph = graph
        
    return connected_graph, conn_nodes_list

def preprocess_and_compute_persistence(
        directory : str,
        mapping_path : str = None,
        output_dir : str = None):
    
    save_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    if mapping_path == None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.join(script_dir, 'mapping.csv')
        mapping = pd.read_csv(mapping_path)
    else:
        mapping_path = mapping_path
        mapping = pd.read_csv(mapping_path)

    PDs_H0, PDs_H1, dropped_subjects = [], [], []
    processed = 0

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            match = re.search(r'sub-([A-Z0-9]+)', filename)
            if match:
                
                start_time = time.time()            
                
                # Get the subject ID
                subject_id = match.group(0)
                filepath = os.path.join(directory, filename)
                matrix = np.loadtxt(filepath, delimiter=',', dtype=float)
                
                np.fill_diagonal(matrix, 0) # zero diagonal
                
                matrix = drop_cerebellum(matrix, mapping) # exclude cerebellum from connectivity
                
                # check for unconnected components and connect
                graph = nx.from_numpy_array(matrix)
                graph, nodes = connect_components(graph, mapping)
                
                distance_matrix = nx.floyd_warshall_numpy(graph) # construct distance matrix with Floyd-Warshall algothm
                distance_matrix = normalize_matrix(distance_matrix) # normalize (min-max)

                if np.max(distance_matrix) != 1:
                    print(f'{subject_id} matrix invalid')
                    dropped_subjects.append(subject_id)
                    continue
                    
                '''
                # plot adjecency
                graph = nx.from_numpy_array(matrix)
                nx.draw_circular(graph)
                im = plt.matshow(distance_matrix)
                cbar = plt.colorbar(im)
                cbar.set_label('Distance', rotation=270, labelpad=15)
                plt.title(f'Disatnce matrix for {subject_id}')
                plt.show()
                '''
                
                # Compute persistence
                h0, h1 = compute_persistence(distance_matrix, max_edge_length=1.0)
                PDs_H0.append(h0)
                PDs_H1.append(h1)
                processed += 1
                
                # Save intermediate lists (every 100 iterations)

                if processed % 100 == 0:
                    fname = os.path.join(save_dir, f'PDs_up_to_{processed}.pkl')
                    with open(fname, 'wb') as f:
                        pickle.dump({
                            'PDs_H0': PDs_H0,
                            'PDs_H1': PDs_H1,
                            'dropped': dropped_subjects
                        }, f)
                    print(f'→ Saved intermediate PDs to {fname}')
                
                end_time = time.time()
                processing_time = end_time - start_time
                print(f'Processed {subject_id} in {processing_time:.2f} sec (total processed: {processed} / {len(os.listdir(directory))})')
                
    # Save final lists
    final_fname = os.path.join(save_dir, f'PDs_final_{processed}.pkl')
    with open(final_fname, 'wb') as f:
        pickle.dump({
            'PDs_H0': PDs_H0,
            'PDs_H1': PDs_H1,
            'dropped': dropped_subjects
        }, f)
    print(f'→ Saved final PDs to {final_fname}')
    return  PDs_H0, PDs_H1, dropped_subjects


# Directory containing the connectivity matrices
directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder'
#directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'
#directory = os.path.join(script_dir, 'rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum')
mapping = '/Users/elijah/Desktop/thesis/Connectomes/mapping.csv'

output_directory = '/Users/elijah/Desktop/thesis/struct_conn_output'

PDs_H0, PDs_H1, dropped_subjects = preprocess_and_compute_persistence(directory, output_dir=output_directory, mapping_path=mapping)

# Visualize sample PD
print('Plotting persistence diagram')
plot_persistence_diagram(PDs_H0[0], PDs_H1[0], save_dir=output_directory)

# Process and visualize all diagrams
print('Analyzing PDs')
PDs_distance_matrix = compute_pds_distences(PDs_H0, PDs_H1, output_dir=output_directory)

#%%

cluster_and_visualize_distances(PDs_distance_matrix, embedding_method = 'tsne', plot_mode = 'save', output_dir=output_directory)