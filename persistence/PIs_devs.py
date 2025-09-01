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

from .preprocessing import drop_cerebellum, connect_components, normalize_matrix

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

    '''plot intermidiate PDs
    plt.figure(figsize=(12,6))
    plt.subplot(121)

    Rips().plot(diag_h0, show=False)
    plt.title("PD of $H_0$")

    plt.subplot(122)
    Rips().plot(diag_h1, show=False)
    plt.title("PD of $H_1$")

    plt.show()
    '''

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
    #directory = '/Users/elijah/Desktop/thesis/Connectomes/test_folder'
    directory = '/Users/elijah/Desktop/thesis/Connectomes/rec-SDStream_atlas-fan2016_desc-SIFT2_scale-None_meas-sum'
    mapping = pd.read_csv('/Users/elijah/Desktop/thesis/Connectomes/mapping.csv')

    output_directory = '/Users/elijah/Desktop/thesis/test_PI_1_allsub'

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



### COMMAND LINE INTERFACE ###

def main():
    """Main function - can be run interactively or via command line"""
    parser = argparse.ArgumentParser(description='Developer Connectomes Persistence Analysis')
    parser.add_argument('--non-interactive', action='store_true', 
                        help='Run with default parameters (both developer groups, MDS, save only)')
    parser.add_argument('--output-dir', type=str, 
                        default='/Users/elijah/Desktop/thesis/struct_conn_developer_output',
                        help='Output directory')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Maximum subjects per group')
    
    args = parser.parse_args()
    
    if args.non_interactive:
        # Run with default parameters
        print("Running in non-interactive mode with default parameters...")
        
        # Setup paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), 'dev_connectomes')
        mapping_path = os.path.join(os.path.dirname(script_dir), 'graph_measures', 'mapping.csv')
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load data
        mapping = pd.read_csv(mapping_path)
        expert_mats = np.load(os.path.join(data_dir, 'expert_mats.npy'))
        expert_names = np.load(os.path.join(data_dir, 'expert_names.npy'), allow_pickle=True)
        naive_mats = np.load(os.path.join(data_dir, 'naive_mats.npy'))
        naive_names = np.load(os.path.join(data_dir, 'naive_names.npy'), allow_pickle=True)
        
        # Process both developer groups
        results = {}
        for group_name, matrices, names in [('expert', expert_mats, expert_names), 
                                           ('naive', naive_mats, naive_names)]:
            PDs_H0, PDs_H1, subject_ids, dropped = process_group_data(
                matrices, names, group_name, mapping, args.output_dir
            )
            
            if len(PDs_H0) > 0:
                distance_matrix = compute_pds_distances(
                    PDs_H0, PDs_H1, subject_ids, group_name, args.output_dir
                )
                labels_df, meta_df = cluster_and_visualize_distances(
                    distance_matrix, group_name, 'MDS', 'save', args.output_dir
                )
                results[group_name] = {
                    'distance_matrix': distance_matrix,
                    'labels': labels_df,
                    'meta': meta_df,
                    'subject_ids': subject_ids,
                    'dropped_subjects': dropped
                }
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
    else:
        # Run interactive mode
        results = run_analysis_with_ui()
    
    return results

if __name__ == "__main__":
    main()
 