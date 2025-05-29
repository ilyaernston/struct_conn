#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:35:19 2025

@author: elijah
"""


import os
import re
import numpy as np

import networkx as nx
from networkx.algorithms import community as nx_comm

import igraph as ig
#import leidenalg as la

from cdlib import algorithms
#import multiprocessing as mp
import time
import pandas as pd 
import random

### HEPLER FUNCTIONS ###

def drop_cerebellum(
        matrix : np.ndarray, 
        mapping_path : str):
    '''
    Args:
        matrix (np.ndarray): adjecency matrix as np.array
        mapping_path (str): path to mapping file in .csv

    Returns:
        filtered_matrix (np.ndarray): adjecency matrix as np.array with cerebellum nodes dropped 
    '''
    
    mapping = pd.read_csv(mapping_path)
    to_drop = mapping.loc[mapping['Lobe'] == 'Cerebellum', 'index'].values
     
    # build a boolean mask of size N
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[to_drop] = False
     
    # filter the matrix: keep only rows & cols where mask==True
    filtered_matrix = matrix[np.ix_(mask, mask)]  
    
    return filtered_matrix

def connect_components(
        graph : nx.Graph, 
        mapping_path : str):
    '''
    Checks if graph has multiple connected components.
    If so, connects smaller components to the anatomacallly closest node in the largest component .
    Anatomical data is obtained from the mapping file. File should be stored in the script's directory and called 'mapping.csv'
     
    
    Parameters
    ----------
    graph : nx.Graph
        networkx graph to process
    mapping_path : str
        path to mapping file in .csv
        
    Returns
    ----------    
    connected_graph : nx.Graph
        processed graph
    conn_nodes : list
        list of dicts with pairs of nodes, between which edges were constructed
    
    '''
    
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

### GRAPH METRICS ESTIMATION FUNCTIONS ###

def create_reversed_matrix(adj_matrix):
    adj_matrix_d = adj_matrix.flatten()
    adj_matrix_d = adj_matrix_d[adj_matrix_d != 0]
    ref_value = adj_matrix_d.max() + adj_matrix_d.min()
    rev_matrix = ref_value - adj_matrix
    rev_matrix[rev_matrix == ref_value] = 0
    return rev_matrix

def compute_small_world_indices(matrix):
    # Create an igraph graph from the adjacency matrix
    g = ig.Graph.Adjacency((matrix > 0).tolist())

    # Calculate average shortest path length
    avg_path_length = g.average_path_length()

    # Calculate clustering coefficient
    clustering_coeff = g.transitivity_undirected()

    # Calculate random graph for comparison
    n = g.vcount()
    m = g.ecount()
    p = 2 * m / (n * (n - 1))  # Probability for Erdős-Rényi graph
    rand_graph = ig.Graph.Erdos_Renyi(n=n, p=p)
    rand_avg_path_length = rand_graph.average_path_length()
    rand_clustering_coeff = rand_graph.transitivity_undirected()

    # Calculate small-world indices
    sigma = (clustering_coeff / rand_clustering_coeff) / (avg_path_length / rand_avg_path_length)
    omega = (rand_avg_path_length / avg_path_length) - (clustering_coeff / rand_clustering_coeff)

    return sigma, omega, avg_path_length, clustering_coeff 

### PROCESSING FUNCTION ###

def compute_metrics(
        matrix : np.ndarray, 
        mapping_path : str):
    
    # zero out diagonal
    np.fill_diagonal(matrix, 0)
    
    # drop cerebellum from ajecency matrix
    matrix = drop_cerebellum(matrix, mapping_path=mapping_path)
    
    # min-max normalization
    #matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    
    # nuild nx graph
    graph = nx.from_numpy_array(matrix)
    #n_components = nx.number_connected_components(graph) # count components in inital graph
    
    # check for unconnected components and construct edges (if needed)
    graph, conn_nodes = connect_components(graph, mapping_path=mapping_path)
    
    # COMPUTE CONNECTIVITY MEASURES #
    n_components = nx.number_connected_components(graph) # double-check: cound components in resulting graph
    density = nx.density(graph)

    sw_sigma, sw_omega, avg_path, clust = compute_small_world_indices(matrix)

    # compute mean rich-clubness
    phi = list(nx.rich_club_coefficient(graph, normalized=False).values())
    phi_mean = np.mean(phi)
    
    # compute global efficiency
    ge = nx.algorithms.efficiency_measures.global_efficiency(graph)

    # define communities and compute modelarity
    coms = algorithms.leiden(graph).communities
    modularity = nx_comm.modularity(graph, coms)

    k = [val for (node, val) in graph.degree()]  # compute degrees for every node
    k_mean = np.mean(k)  # average node degree

    return (
        sw_sigma,
        sw_omega,
        phi_mean,
        ge,
        clust,
        modularity,
        avg_path,
        k_mean,
        density,
        n_components,
        conn_nodes
    )

def process_files(
        directory : str, 
        mapping_dir : str = None):
    '''_summary_

    Args:
        directory (str) :   path to directory, where connectivity matrices are stored is .csv
        mapping_dir (str, optional) :   path to directory with mapping table, named 'mapping.csv'. 
                                        Defaults to None, then searches for mapping file in script's directory.

    Returns:
        _type_: _description_
    '''
    # Determine the directory of the current script if mapping_dir not provided
    if mapping_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.join(script_dir, 'mapping.csv')
    else:
        mapping_path = os.path.join(mapping_dir, 'mapping.csv')

    results = []
    # Loop through CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            match = re.search(r'sub-([A-Z0-9]+)', filename)
            if match:
                # Get the subject ID (e.g., "sub-BB00006")
                subject_id = match.group(0)
                filepath = os.path.join(directory, filename)
                matrix = np.loadtxt(filepath, delimiter=',', dtype=float)

                start_time = time.time()
                metrics = compute_metrics(matrix, mapping_path=mapping_path)
                end_time = time.time()
                processing_time = end_time - start_time

                # Append subject id and metrics as a new row
                results.append([subject_id] + list(metrics))
                print(f"Processed {filename} in {processing_time:.2f} seconds")

    # Define column names for the DataFrame
    columns = ['subject_id', 'sw_sigma', 'sw_omega', 'avg_rich_club',
               'global_efficiency', 'avg_clustering', 'modularity',
               'avg_path_length', 'avg_degree', 'density', 'n_comp', 'constructed_edges']

    # Create a pandas DataFrame with the results
    df = pd.DataFrame(results, columns=columns)
    return df

# APPLY ANALYSIS #

# Directory containing the connectivity matrices
#directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder'
directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'

# Process files and compute metrics
results_df = process_files(directory)

def append_metadata_and_save(
        metrics_df : pd.DataFrame,
        output_dir : str = None,
        mode : str = 'save'
        ):
    # APPEND DEMOGRAPHIC DATA AND SAVE #

    # set directory to save or make one in current folder, if none provided
    save_dir = output_dir or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    # Load subjects metadata
    labels_df = pd.read_csv('/Users/elijah/Desktop/thesis/tests_2/all_labels.csv')
    labels_df['subject_id'] = 'sub-' + labels_df['subject_id']

    # Merge with metrics df 
    output_df = metrics_df.merge(labels_df, how = 'inner', on = 'subject_id')

    if mode == 'save':
        # save measures with appended metadata as .csv
        out_csv = os.path.join(save_dir, "measures&pd&metadata.csv")
        output_df.to_csv(out_csv, index=False)
        print(f"→ Saved labels to {out_csv}")

output_dir = '/Users/elijah/Desktop/thesis/tests_21'
append_metadata_and_save(results_df, output_dir=output_dir, mode='save')
