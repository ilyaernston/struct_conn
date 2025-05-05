#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:45:15 2025

@author: elijah
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score, adjusted_mutual_info_score
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity
import community.community_louvain
import os
import re
import pandas as pd
import random


def spectral_clustering(adj_matrix, k, normalized=False):
    degree = np.diag(adj_matrix.sum(axis=1))
    laplacian = degree - adj_matrix
    if normalized:
        sqrt_degree = np.sqrt(degree)
        sqrt_degree_inv = np.linalg.inv(sqrt_degree)
        laplacian = sqrt_degree_inv @ laplacian @ sqrt_degree_inv
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    features = eigenvectors[:, :k]
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

def modularity_clustering(adj_matrix, k):
    degrees = adj_matrix.sum(axis=1)
    m = adj_matrix.sum() / 2
    B = adj_matrix - np.outer(degrees, degrees) / (2 * m)
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_indices[:k]]
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(selected_eigenvectors)
    return labels

def louvain_clustering(adj_matrix, k):
    G = nx.from_numpy_array(adj_matrix)
    partition = community.community_louvain.best_partition(G)
    labels = list(partition.values())
    labels = np.array(labels)
    return labels

'''
def compute_metrics(adj_matrix, labels):
    distance_matrix = 1 - adj_matrix.copy()
    np.fill_diagonal(distance_matrix, 0)
    try:
        silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
    except ValueError:
        silhouette = 0
    G = nx.from_numpy_array(adj_matrix)
    communities = [set(np.where(labels == c)[0]) for c in np.unique(labels)]
    modularity = nx_modularity(G, communities)
    return silhouette, modularity
'''

def compute_metrics(adj_matrix, labels):
    
    A = adj_matrix.copy() # normalize adjecency matrix to [0 ; 1]
    max_w = A.max()
    if max_w > 0:
        A /= max_w
    distance_matrix = 1 - A # build a distance matrix
    np.fill_diagonal(distance_matrix, 0)
    # check: how many clusters, and how big is each
    uniques, counts = np.unique(labels, return_counts=True)
    # need at least 2 clusters, and each cluster must have at least 2 members
    if len(uniques) > 1 and np.min(counts) > 1:
        silhouette = silhouette_score(distance_matrix,
                                      labels,
                                      metric='precomputed')
    else:
        silhouette = np.nan   # or 0, or skip this k entirely

    # modularity as before
    G = nx.from_numpy_array(adj_matrix)
    communities = [set(np.where(labels == c)[0]) for c in uniques]
    modularity = nx_modularity(G, communities)

    return silhouette, modularity

def drop_cerebellum(matrix):
    
    # load mapping file
    script_path = os.path.abspath(__file__) # full path to this script
    script_dir = os.path.dirname(script_path) # directory containing the script
    mapping_path = script_dir + "/mapping.csv"
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
    mapping_path = script_dir + "/mapping.csv"
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

'''
# Generate SBM with 10 clusters
np.random.seed(42)
n_blocks = 10
block_size = 20  
sizes = [block_size] * n_blocks
probs = [[0.7 if i == j else 0.1 for j in range(n_blocks)] for i in range(n_blocks)]
G = nx.stochastic_block_model(sizes, probs, seed=42, directed=False)
adj_matrix = nx.to_numpy_array(G)
'''

# Directory containing the connectivity matrices
directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder'
directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'

res_clutserings = {}
res_metrics = []

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        match = re.search(r'sub-([A-Z0-9]+)', filename)
        if match:
            # Get the subject ID
            subject_id = match.group(0)
            filepath = os.path.join(directory, filename)
            adj_matrix = np.loadtxt(filepath, delimiter=',', dtype=float)
            
            np.fill_diagonal(adj_matrix, 0)
            
            adj_matrix = drop_cerebellum(adj_matrix)
            
            # min-max normalization
            adj_matrix = (adj_matrix - np.min(adj_matrix)) / (np.max(adj_matrix) - np.min(adj_matrix))

            graph = nx.from_numpy_array(adj_matrix)
            graph, nodes = connect_components(graph)
            
            adj_matrix = nx.to_numpy_array(graph)

            max_k = 8
            k_values = range(2, max_k+1)
            
            methods = {
                'Laplacian': lambda adj, k: spectral_clustering(adj, k, normalized=False),
                'Normalized Laplacian': lambda adj, k: spectral_clustering(adj, k, normalized=True),
                'Modularity Matrix': modularity_clustering,
                'Louvain': louvain_clustering
            }
            
            metrics = {'Silhouette': {}, 'Modularity': {}}
            for method_name in methods:
                metrics['Silhouette'][method_name] = []
                metrics['Modularity'][method_name] = []
            
            clusterings = {}
            
            for k in k_values:
                for method_name, method_func in methods.items():
                    labels = method_func(adj_matrix, k)
                    silhouette, modularity = compute_metrics(adj_matrix, labels)
                    metrics['Silhouette'][method_name].append(silhouette)
                    metrics['Modularity'][method_name].append(modularity)
                    if k == max_k:
                        clusterings[method_name] = labels
                        
            
            mods = np.array([metrics['Modularity'][method_name][-1] for method_name in methods])
            sils = np.array([metrics['Silhouette'][method_name][-1] for method_name in methods])
            res = np.concatenate([mods, sils])
            
            res_clutserings[subject_id] = clusterings
            res_metrics.append([subject_id] + list(res))
            
            '''
            # Plot metrics
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            for method_name in methods:
                plt.plot(k_values, metrics['Silhouette'][method_name], label=method_name)
            plt.xlabel('k')
            plt.ylabel('Silhouette Score')
            plt.legend()
            plt.title('Silhouette Score vs k')
            
            plt.subplot(1, 2, 2)
            for method_name in methods:
                plt.plot(k_values, metrics['Modularity'][method_name], label=method_name)
            plt.xlabel('k')
            plt.ylabel('Modularity')
            plt.legend(loc = 'lower right')
            plt.title('Modularity vs k')
            plt.tight_layout()
            plt.show()
            
            # Plot sorted matrices for true k=10
            k = 10
            plt.figure(figsize=(16, 4))
            for i, (method_name, method_func) in enumerate(methods.items()):
                plt.subplot(1, 4, i+1)
                labels = method_func(adj_matrix, k)
                sorted_matrix = adj_matrix[np.argsort(labels)][:, np.argsort(labels)]
                plt.imshow(sorted_matrix, cmap='viridis', interpolation='none')
                plt.title(method_name)
            plt.tight_layout()
            plt.show()

            '''
# store metrics in pd.DataFrame
columns = (
    'subject_id', 
    'laplacian_modularity',
    'norm_laplacian_modularity', 
    'matrix_modularity', 
    'louvain_modularity',
    'laplacian_silhouette',
    'norm_laplacian_silhouette', 
    'matrix_silhouette', 
    'louvain_silhouette',
    )
res_clust_qual_metrics_df = pd.DataFrame(res_metrics, columns=columns)
            
#%%



def hypergeometric_test(cluster_labels, alpha=0.01, correction='bonferroni'):
    """
    Test enrichment of GMM clusters in network components using the hypergeometric test.
    
    Parameters:
    - cluster_labels: List, array, or pd.Series of cluster assignments (e.g., 'white', 'gray', 'mixed')
    - cluster_labels2: List, array, or pd.Series of cluster2 assignments (e.g., 0, 1, 2)
    - alpha: Significance threshold (default: 0.01)
    - correction: Multiple testing correction method (default: 'bonferroni')
    
    Returns:
    - results: pd.DataFrame with columns ['component', 'cluster', 'overlap', 'pval', 'significant']
    """

    import pandas as pd
    from scipy.stats import hypergeom
    
    # Load reference parcellation from mapping file
    mapping = pd.read_csv('/Users/elijah/Desktop/thesis/Connectomes/fan2016/Copy of space-MNI152_atlas-fan2016_res-1mm_dseg.csv')
    mapping = mapping[mapping['Lobe'] != 'Cerebellum']
    component_labels = np.array(mapping['Yeo_7network'])
    
    # Convert inputs to pandas Series
    if not isinstance(cluster_labels, pd.Series):
        cluster_labels = pd.Series(cluster_labels)
    if not isinstance(component_labels, pd.Series):
        component_labels = pd.Series(component_labels)
    
    # Get unique clusters and components
    clusters = cluster_labels.unique()
    components = component_labels.unique()
    
    # Total molecules
    N = len(cluster_labels)
    
    results = []
    for component in components:
        component_mask = (component_labels == component)
        n = component_mask.sum()  # Size of network component
        
        for cluster in clusters:
            cluster_mask = (cluster_labels == cluster)
            D = cluster_mask.sum()  # Size of GMM cluster
            k = (component_mask & cluster_mask).sum()  # Overlap
            
            # Hypergeometric test (right-tailed: P(X >= k))
            pval = hypergeom.sf(k-1, N, D, n)
            
            results.append({
                'yeo7_cluster': component,
                'pred_cluster': cluster,
                'overlap': k,
                'pval': pval
            })
    
    results_df = pd.DataFrame(results)
    
    # Apply Bonferroni correction
    n_tests = len(results_df)
    if correction == 'bonferroni':
        results_df['significant'] = results_df['pval'] < (alpha / n_tests)
    else:
        results_df['significant'] = results_df['pval'] < alpha
    
    return results_df


def compare_to_parcellations(clustering):
    '''
    Function to compare provided clustering to reference mapping (Yeo-17) via Rand Index, 
    Adjusted Rand Index, Adjusted Mutual Information Index, V-measure and 
    Hypergeometric test. Reqiures mapping .csv file as a benchmark

    Parameters
    ----------
    clustering : np.array
        Clusteing to assess

    Returns
    -------
    rand_index : float
        Rand Index value
    adj_rand_index : float
        Adjusted Rand Index value
    adj_mi_index : float
        Adjusted Mutual Information Index value
    v_measure_index : float
        V-measure value

    '''
    
    from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
    
    mapping = pd.read_csv('/Users/elijah/Desktop/thesis/Connectomes/fan2016/Copy of space-MNI152_atlas-fan2016_res-1mm_dseg.csv')
    mapping = mapping[mapping['Lobe'] != 'Cerebellum']
    labels_true_yeo_7 = mapping['Yeo_7network'].to_numpy()
    
    labels_pred = np.array(clustering)

    rand_index = rand_score(labels_true_yeo_7, labels_pred)
    adj_rand_index = adjusted_rand_score(labels_true_yeo_7, labels_pred)
    adj_mi_index = adjusted_mutual_info_score(labels_true_yeo_7, labels_pred, average_method='arithmetic')
    v_measure_index = v_measure_score(labels_true_yeo_7, labels_pred)
        
    return rand_index, adj_rand_index, adj_mi_index, v_measure_index





    
indices_records, hypergeom_records = [], []

for subject_id, clusterings in res_clutserings.items():
    for method, clustering in clusterings.items():
        
        # Compute clustering similarity indices
        rand_index, adj_rand_index, adj_mi_index, v_measure_index = compare_to_parcellations(clustering)
        indices_records.append({
                'subject_id': subject_id,
                'method': method,
                'rand_index': rand_index,
                'adjusted_rand_index': adj_rand_index,
                'adjusted_mutual_info_index': adj_mi_index,
                'v-measure': v_measure_index
            })
        
        # Perform hypergeometric test
        labels_pred = np.array(clustering)
        hg = hypergeometric_test(labels_pred)
        hg['subject_id'] = subject_id
        hg['method'] = method
        # reorder columns if you like:
        hg = hg[['subject_id','method','yeo7_cluster','pred_cluster','overlap','pval','significant']]
        hypergeom_records.append(hg)
        
        
        
res_clust_indices_df    = pd.DataFrame(indices_records)
res_hypergeom_df  = pd.concat(hypergeom_records, ignore_index=True)

res_hypergeom_significant = res_hypergeom_df[res_hypergeom_df['significant'] == True]

#%%

dfs_to_save = [res_clust_indices_df, res_hypergeom_df, res_hypergeom_significant, res_clust_qual_metrics_df]  

# save .csv into the same directory
script_path = os.path.abspath(__file__) # full path to this script
script_dir = os.path.dirname(script_path) # directory containing the script
base_name = os.path.splitext(os.path.basename(script_path))[0] # script filename without extension

def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name

for df in dfs_to_save:
    # define output path: same folder + same name as script file + same name as df
    df_name = get_var_name(df)
    output_path = os.path.join(script_dir, base_name + f"_{df_name}.csv") 
    # save .csv
    df.to_csv(output_path, index=False)



  
