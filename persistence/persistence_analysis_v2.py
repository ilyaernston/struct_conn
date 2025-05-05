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
#from sklearn.cluster import KMeans
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

def plot_persistence_diagram(H0, H1, title="Persistence Diagram"):
    """Visualize H0 and H1 persistence diagrams with modern style"""
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    if len(H0) > 0:
        ax.scatter(H0[:, 0], H0[:, 1], c='#1f77b4', marker='^', 
                  s=80, label='H0', alpha=0.7, edgecolor='w')
    if len(H1) > 0:
        ax.scatter(H1[:, 0], H1[:, 1], c='#ff7f0e', marker='o', 
                  s=80, label='H1', alpha=0.7, edgecolor='w')
    
    max_val = max(1.0, *H0[:, 1], *H1[:, 1]) if len(H1) > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], '--', color='#2ca02c', alpha=0.7)
    
    ax.set(xlabel='Birth', ylabel='Death', title=title)
    ax.legend()
    plt.tight_layout()
    plt.show()

from sklearn.mixture import GaussianMixture
from matplotlib.cm import tab20
from scipy.spatial.distance import squareform

def process_and_visualize_pds(PDs_H0, PDs_H1):
    """Enhanced version with GMM clustering and model selection"""
    
    # script direcory
    script_path = os.path.abspath(__file__) # full path to this script
    script_dir = os.path.dirname(script_path) # directory containing the script
    
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
            fdir = os.path.join(script_dir, fname)
            np.save(fdir, distance_matrix)
            print(f'Saved intermediate matrix to "{fname}"')

    # Convert to condensed form for MDS
    #condensed_dist = squareform(distance_matrix, checks=False)
    
    # Metric MDS embedding
    print('MDS emnedding')
    mds = MDS(n_components=2, dissimilarity='precomputed', 
             random_state=42, normalized_stress='auto')
    embeddings = mds.fit_transform(distance_matrix)
    
    # Cluster selection using GMM and BIC
    print('Cluster selection')
    max_clusters = min(10, num_graphs-1)  # Practical upper limit
    cluster_range = range(2, max_clusters+1)
    bic_scores = []
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: BIC scores for different numbers of clusters
    plt.subplot(1, 2, 1)
    for k in cluster_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(embeddings)
        bic_scores.append(gmm.bic(embeddings))
    
    optimal_k = cluster_range[np.argmin(bic_scores)]
    plt.plot(cluster_range, bic_scores, 'o-', color='#2ca02c')
    plt.axvline(optimal_k, color='#d62728', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC Score')
    plt.title(f'Optimal Clusters: {optimal_k}')
    plt.grid(alpha=0.3)
    
    # Plot 2: MDS embedding with optimal clusters
    plt.subplot(1, 2, 2)
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    #clusters = gmm.fit_predict(embeddings)
    
    # Create cyclical colormap for clusters
    #colors = [tab20(i % 20) for i in clusters]
    
    #scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=150, edgecolor='w', linewidth=1, alpha=0.9)
    
    # Create legend for clusters
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=tab20(i), markersize=10)
                for i in range(optimal_k)]
    plt.legend(handles, [f'Cluster {i+1}' for i in range(optimal_k)], 
              loc='best', title='Clusters')
    
    plt.title(f'MDS Embedding with {optimal_k} Clusters')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
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

def drop_cerebellum(matrix):
    
    # load mapping file
    script_path = os.path.abspath(__file__) # full path to this script
    script_dir = os.path.dirname(script_path) # directory containing the script
    mapping_path = os.path.join(script_dir, 'mapping.csv')
    mapping = pd.read_csv(mapping_path)
    
    to_drop = mapping.loc[mapping['Lobe'] == 'Cerebellum', 'index'].values
     
    # build a boolean mask of size N
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[to_drop] = False
     
    # filter the matrix: keep only rows & cols where mask==True
    filtered_matrix = matrix[np.ix_(mask, mask)]  
    
    return filtered_matrix

def connect_components(graph):
    '''
    Checks if graph has multiple connected components.
    If so, connects smaller components to the anatomacallly closest node in the largest component .
    Anatomical data is obtained from the mapping file. File should be stored in the script's directory and called 'mapping.csv'
     
    
    Parameters
    ----------
    graph : nx.Graph
        networkx graph to process
        
    Returns
    ----------    
    connected_graph : nx.Graph
        processed graph
    conn_nodes : list
        list of dicts with pairs of nodes, between which edges were constructed
    
    '''
    
    # load mapping file
    script_path = os.path.abspath(__file__) # full path to this script
    script_dir = os.path.dirname(script_path) # directory containing the script
    mapping_path = os.path.join(script_dir, 'mapping.csv')
    mapping = pd.read_csv(mapping_path)
    
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
    
    #%%
  
# script direcory
script_path = os.path.abspath(__file__) # full path to this script
script_dir = os.path.dirname(script_path) # directory containing the script
    
# Directory containing the connectivity matrices
#directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder'
#directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'
directory = os.path.join(script_dir, 'rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum')

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
            
            matrix = drop_cerebellum(matrix) # exclude cerebellum from connectivity
            
            # check for unconnected components and connect
            graph = nx.from_numpy_array(matrix)
            graph, nodes = connect_components(graph)
            
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
                fname = os.path.join(script_dir, f'PDs_up_to_{processed}.pkl')
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
final_fname = os.path.join(script_dir, f'PDs_final_{processed}.pkl')
with open(final_fname, 'wb') as f:
    pickle.dump({
        'PDs_H0': PDs_H0,
        'PDs_H1': PDs_H1,
        'dropped': dropped_subjects
    }, f)
print(f'→ Saved final PDs to {final_fname}')
                

# Visualize sample PD
print('Plotting persistence diagram')
plot_persistence_diagram(PDs_H0[0], PDs_H1[0], "Persistence Diagram")

# Process and visualize all diagrams
print('Analyzing PDs')
process_and_visualize_pds(PDs_H0, PDs_H1)


    