'''
Functions to preprocess DTI connectomes with faan-2016 parcellation
'''

import numpy as np
import pandas as pd
import igraph as ig
import random
from typing import Tuple, List

### HELPER FUNCTIONS ###

def drop_cerebellum(matrix: np.ndarray, mapping: pd.DataFrame) -> np.ndarray:
    """Remove cerebellum regions from connectivity matrix"""
    to_drop = mapping.loc[mapping['Lobe'] == 'Cerebellum', 'index'].values
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[to_drop.astype(int)] = False
    filtered_matrix = matrix[np.ix_(mask, mask)]
    return filtered_matrix

def connect_components(matrix: np.ndarray, 
                       mapping: pd.DataFrame, 
                       return_altered_nodes: bool = False) -> np.ndarray | Tuple[np.ndarray, List]:
    """
    Connect disconnected components using anatomical proximity.

    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix.
    mapping : pd.DataFrame
        DataFrame containing region attributes.
    return_altered_nodes : bool, optional
        If True, also returns a list of nodes that were altered/connected.

    Returns
    -------
    connected_matrix : np.ndarray
        The connectivity matrix after connecting components.
    conn_nodes_list : list, optional
        List of altered/connected nodes (only if return_altered_nodes is True).
    """

    # Create weighted graph from adjacency matrix
    graph = ig.Graph.Weighted_Adjacency(matrix.tolist(), mode="undirected")

    conn_nodes_list = []
    components = graph.connected_components(mode="weak")
    n_components = len(components)
    
    if n_components > 1:
        # Find largest component
        main_comp_idx = max(range(n_components), key=lambda i: len(components[i]))
        main_comp = components[main_comp_idx]
        main_nodes = set(main_comp)
    
        for i, comp in enumerate(components):
            if i == main_comp_idx:
                continue
            else:
                u = random.choice(comp)
                attr_u = mapping.iloc[u]
                hemi_u = attr_u['Hemi']
                gyrus_u = attr_u['Gyrus']
                lobe_u = attr_u['Lobe']

                main_attrs = mapping.loc[list(main_nodes)]
                same_hemi = main_attrs[main_attrs['Hemi'] == hemi_u]

                candidates = same_hemi[same_hemi['Gyrus'] == gyrus_u].index.tolist()
                if not candidates:
                    candidates = same_hemi[same_hemi['Lobe'] == lobe_u].index.tolist()
                if not candidates:
                    candidates = same_hemi.index.tolist()
                if not candidates:
                    candidates = list(main_nodes)

                v = random.choice(candidates)

                sub_idx = candidates
                submat = matrix[np.ix_(sub_idx, sub_idx)]
                weights = submat[submat > 0]
                avg_w = float(weights.mean()) if weights.size else float(matrix.mean())

                graph.add_edge(u, v, weight=avg_w)
                conn_nodes_list.append({u: v})
        
        # Convert back to numpy array
        connected_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    else:
        conn_nodes_list = []
        connected_matrix = np.array(graph.get_adjacency(attribute='weight').data)

    if return_altered_nodes:
        return connected_matrix, conn_nodes_list
    else:
        return connected_matrix

def normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize matrix to [0, 1] range"""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

def invert(matrix: np.ndarray) -> np.ndarray:
    """Convert weights to distances"""
    safe_weights = np.where(matrix > 0, matrix, np.inf)
    inv_matrix = np.where(safe_weights != np.inf, 1 / safe_weights, 0)
    return inv_matrix