'''
Script for topological data analysis of structural connectivity via Persistance Images methodology
'''

# Import dependencies

import os
import re
import time
import random
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ripser import Rips, ripser
from persim import PersImage
from persim import PersistenceImager
from persim import plot_diagrams

# Hepler functions (preprocessing)

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

# PI analysis function

def compute_persistence_images(
        distance_matrix: np.ndarray,
        h_dimension: str = 'both',
        max_death: float = 1.0,
        display_plot: bool = False,
        save_plot: bool = False,
        subject_id: str = None,
        output_dir: str = None
    ) -> np.ndarray:
    """
    Computes persistence-image feature(s) for H0 and/or H1.

    Args:
        distance_matrix (np.ndarray):  
            Pairwise distance matrix.
        h_dimension (str):  
            One of 'h0', 'h1', or 'both'.
        max_death (float):  
            Replacement for any infinite death times.
        display_plot (bool):  
            If True, displays the computed image(s).
        save_plot (bool):
            If True, saves PI plots into output_dir/plots.
        subject_id (str):
            Identifier used for naming files (e.g. 'sub-ABC123').
        output_dir (str):
            Base folder for ‘plots/’. Defaults to this script’s directory.
    Returns:
        feature_vector (np.ndarray):  
            1D array of concatenated PI pixels (H0 then H1 if both).
    """
    # determine base folder
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(output_dir, 'plots')
    if save_plot:
        os.makedirs(plots_dir, exist_ok=True)

    # 1) get the persistence diagrams
    result  = ripser(distance_matrix, maxdim=1, distance_matrix=True)
    diag_h0 = result['dgms'][0]
    diag_h1 = result['dgms'][1]

    # 2) clamp infinite deaths
    for diag in (diag_h0, diag_h1):
        infs = np.isinf(diag[:, 1])
        if infs.any():
            diag[infs, 1] = max_death

    # 3) fit & transform separately
    max_val = np.max(distance_matrix)
    img_list = []
    pimgrs   = {}

    if h_dimension in ('h0', 'both'):
        pimgr0 = PersistenceImager(pixel_size=0.02,
                                   birth_range=(0, max_val),
                                   pers_range=(0, max_val))
        img0 = pimgr0.transform(diag_h0)
        img_list.append(img0)
        pimgrs['h0'] = pimgr0

    if h_dimension in ('h1', 'both'):
        pimgr1 = PersistenceImager(pixel_size=0.02,
                                   birth_range=(0, max_val),
                                   pers_range=(0, max_val))
        img1 = pimgr1.transform(diag_h1)
        img_list.append(img1)
        pimgrs['h1'] = pimgr1

    # 4) flatten & concatenate into feature vector
    imgs_array     = np.array([img.flatten() for img in img_list])
    feature_vector = imgs_array.flatten()

    # 5) plot
    dims = list(pimgrs.keys())
    fig, axes = plt.subplots(1, len(dims), figsize=(5*len(dims), 4))
    if len(dims) == 1:
        axes = [axes]
    for ax, dim, img in zip(axes, dims, img_list):
        pimgrs[dim].plot_image(img, ax=ax)
        ax.set_title(f"PI of {dim.upper()}")
    plt.tight_layout()

    if save_plot and subject_id is not None:
        fname = f"{subject_id}.png"
        fig.savefig(os.path.join(plots_dir, fname))
    if display_plot:
        plt.show()
    plt.close(fig)

    return feature_vector

# Main function

if __name__ == '__main__':
    directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder'
    #directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'
    mapping = pd.read_csv('/Users/elijah/Desktop/thesis/Connectomes/mapping.csv')

    output_directory = '/Users/elijah/Desktop/thesis/test_PI'

    subject_ids = []
    features    = []

    for fname in tqdm(os.listdir(directory)):
        if not fname.endswith('.csv'):
            continue
        m = re.search(r'sub-([A-Z0-9]+)', fname)
        if not m:
            continue

        sid  = m.group(0)
        mat  = np.loadtxt(os.path.join(directory, fname), delimiter=',')
        dist = prepocess(mat, mapping)

        vec = compute_persistence_images(
            distance_matrix=dist,
            h_dimension='both',
            max_death=1.0,
            display_plot=False,
            save_plot=True,
            subject_id=sid,
            output_dir=output_directory
        )

        subject_ids.append(sid)
        features.append(vec)

    # save all features in one CSV

    # Create DataFrame: one row per subject, columns f0, f1, f2, …
    df = pd.DataFrame(features)
    df.insert(0, 'subject_id', subject_ids)

    out_csv = os.path.join(output_directory, 'PI_features.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved combined features to {out_csv}")




 