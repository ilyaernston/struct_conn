import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from ripser import Rips, ripser
from persim import PersImage
from persim import PersistenceImager
from persim import plot_diagrams

import networkx as nx
import os
import re
import time
import random

from ripser import Rips
from persim import PersistenceImager
import matplotlib.pyplot as plt

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

def normalize_matrix(matrix):
    """Normalize MRI matrix to [0, 1] range"""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

def prepocess(matrix, mapping):
    
    np.fill_diagonal(matrix, 0) # zero diagonal
            
    matrix = drop_cerebellum(matrix, mapping) # exclude cerebellum from connectivity
    
    # check for unconnected components and connect
    graph = nx.from_numpy_array(matrix)
    graph, nodes = connect_components(graph, mapping)
    
    distance_matrix = nx.floyd_warshall_numpy(graph) # construct distance matrix with Floyd-Warshall algothm
    distance_matrix = normalize_matrix(distance_matrix) # normalize (min-max)

    return distance_matrix     

from ripser import ripser
from persim import PersistenceImager
import numpy as np
import matplotlib.pyplot as plt

def compute_persistance_images(
        distance_matrix: np.ndarray,
        h_dimension: str = 'both',
        plot: bool = False,
        max_death: float = 1.0
    ) -> np.ndarray:
    """
    Computes persistence-image feature(s) for H0 and/or H1, fitting each imager separately.

    Args:
        distance_matrix (np.ndarray):  
            Pairwise distance matrix.
        h_dimension (str):  
            One of 'h0', 'h1', or 'both'.
        plot (bool):  
            If True, displays the computed image(s).
        max_death (float):  
            Replacement for any infinite death times.

    Returns:
        feature_vector (np.ndarray):  
            1D array of concatenated PI pixels (H0 then H1 if both).
    """
    # get the persistence diagrams
    result = ripser(distance_matrix, maxdim=1, distance_matrix=True)
    diag_h0, diag_h1 = result['dgms'][0], result['dgms'][1]

    # clamp infinite deaths
    for diag in (diag_h0, diag_h1):
        infs = np.isinf(diag[:, 1])
        if infs.any():
            diag[infs, 1] = max_death
    
    # fit & transform separately
    max_val = np.max(distance_matrix) # get upper threshold for homologies' birth
    img_list  = []
    pimgrs     = {}

    if h_dimension in ('h0', 'both'):
        pimgr0 = PersistenceImager(pixel_size=0.02, birth_range=(0, max_val), pers_range=(0, max_val))
        img0 = pimgr0.transform(diag_h0)
        img_list.append(img0)
        pimgrs['h0'] = pimgr0

    if h_dimension in ('h1', 'both'):
        pimgr1 = PersistenceImager(pixel_size=0.02, birth_range=(0, max_val), pers_range=(0, max_val))
        img1 = pimgr1.transform(diag_h1)
        img_list.append(img1)
        pimgrs['h1'] = pimgr1

    # 4) stack flattened images into a 2D array
    #    each row corresponds to one homology dimension
    imgs_array    = np.array([img.flatten() for img in img_list])
    # 5) concatenate rows into one long feature vector
    feature_vector = imgs_array.flatten()

    # 6) optional plotting
    if plot and img_list:
        dims = list(pimgrs.keys())
        fig, axes = plt.subplots(1, len(dims), figsize=(5*len(dims), 4))
        if len(dims) == 1:
            axes = [axes]
        for ax, dim, img in zip(axes, dims, img_list):
            pimgrs[dim].plot_image(img, ax=ax)
            ax.set_title(f"PI of ${dim.upper()}$")
        plt.tight_layout()
        plt.show()

    return feature_vector


directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder_1'
#directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'
mapping = pd.read_csv('/Users/elijah/Desktop/thesis/Connectomes/mapping.csv')

subject_ids = []
features    = []

for fname in os.listdir(directory):
    if not fname.endswith('.csv'): 
        continue
    m = re.search(r'sub-([A-Z0-9]+)', fname)
    if not m:
        continue

    sid = m.group(0)
    mat = np.loadtxt(os.path.join(directory, fname), delimiter=',')
    dist = prepocess(mat, mapping)

    vec = compute_persistance_images(distance_matrix=dist, h_dimension='both', plot=True)

    subject_ids.append(sid)
    features.append(vec)

