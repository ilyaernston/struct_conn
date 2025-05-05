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

#%%


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
        

def compute_metrics(matrix):
    
    # zero out diagonal
    np.fill_diagonal(matrix, 0)
    
    #mapping = pd.read_csv('/Users/elijah/Desktop/thesis/Connectomes/fan2016/Copy of space-MNI152_atlas-fan2016_res-1mm_dseg.csv')

    # drop cerebellum from ajecency matrix
    matrix = drop_cerebellum(matrix)
    
    # min-max normalization
    #matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    
    # nuild nx graph
    graph = nx.from_numpy_array(matrix)
    #n_components = nx.number_connected_components(graph) # count components in inital graph
    
    # check for unconnected components and construct edges (if needed)
    graph, conn_nodes = connect_components(graph)
    
    # COMPUTE CONNECTIVITY MEASURES #
    n_components = nx.number_connected_components(graph)
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
        n_components, # n of components in initial graph
        conn_nodes
    )

def process_files(directory):
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
                metrics = compute_metrics(matrix)
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

'''
def parallel_processing(directory):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_df = process_files(directory)
    return results_df
'''

#%%

# APPLY ANALYSIS #

# Directory containing the connectivity matrices
directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder'
directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'

# Process files and compute metrics
results_df = process_files(directory)

#%%

# APPEND DEMOGRAPHIC DATA AND SAVE #

# Load subjects metadata
subjects_info = pd.read_csv(('/Users/elijah/Desktop/thesis/Connectomes/subjects.csv'))
subjects_info = subjects_info.drop(columns = ['Trimmed SubjectCode'])

# Merge with metrics df 
output_df = results_df.merge(subjects_info, how = 'inner', on = 'subject_id')

# save .csv into the same directory
script_path = os.path.abspath(__file__) # full path to this script
script_dir = os.path.dirname(script_path) # directory containing the script
base_name = os.path.splitext(os.path.basename(script_path))[0] # script filename without extension

# define output path: same folder + same name as script file
output_path = os.path.join(script_dir, base_name + "_output.csv") 
# sace .csv
output_df.to_csv(output_path, index=False)



