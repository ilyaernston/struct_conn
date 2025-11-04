'''
Functions to preprocess DTI connectomes with faan-2016 parcellation
'''

import numpy as np
import pandas as pd
import networkx as nx
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

def connect_components(graph: nx.Graph, mapping: pd.DataFrame) -> Tuple[nx.Graph, List]:
    """Connect disconnected components using anatomical proximity"""
    conn_nodes_list = []
    comps = list(nx.connected_components(graph))
    n_components = nx.number_connected_components(graph)
    
    if n_components > 1:
        main_comp = max(comps, key=len)
        main_nodes = set(main_comp)
    
        for comp in comps:
            if comp is main_comp:
                continue
            else:
                u = random.choice(list(comp))
                attr_u = mapping.loc[u]
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
        
                matrix = nx.to_numpy_array(graph)
                sub_idx = candidates
                submat = matrix[np.ix_(sub_idx, sub_idx)]
                weights = submat[submat > 0]
                avg_w = float(weights.mean()) if weights.size else float(matrix.mean())
        
                graph.add_edge(u, v, weight=avg_w)
                conn_nodes_list.append({u: v})
        
        connected_graph = graph
    else:
        conn_nodes_list.append(np.nan)
        connected_graph = graph
        
    return connected_graph, conn_nodes_list

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize matrix to [0, 1] range"""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

def invert_weights(matrix: np.ndarray) -> np.ndarray:
    """Convert weights to distances"""
    safe_weights = np.where(matrix > 0, matrix, np.inf)
    inv_matrix = np.where(safe_weights != np.inf, 1 / safe_weights, 0)
    return inv_matrix