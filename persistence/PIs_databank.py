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

# Import helper functions from separate modules
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
            Base folder for 'plots/'. Defaults to this script's directory.
    Returns:
        feature_vector (np.ndarray):  
            The persistence image feature vector(s). If h_dimension='both',
            returns concatenated H0+H1 features.
    """
    
    # Default output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Compute persistence diagrams using ripser
    diagrams = ripser(distance_matrix, maxdim=1, distance_matrix=True)['dgms']
    
    # Extract diagrams (replacing inf with max_death)
    H0 = diagrams[0]
    H1 = diagrams[1]
    
    # Replace infinite deaths
    H0[H0 == np.inf] = max_death
    H1[H1 == np.inf] = max_death
    
    # Initialize PersistenceImager
    pimgr = PersistenceImager(pixel_size=1.0)
    
    # Compute persistence images
    if h_dimension.lower() == 'h0':
        pimg = pimgr.transform([H0])
        feature_vector = pimg[0].flatten()
        
        if display_plot or save_plot:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(pimg[0], origin='lower')
            ax.set_title(f'H0 Persistence Image')
            if save_plot:
                plots_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                fname = os.path.join(plots_dir, f'PI_H0_{subject_id}.png')
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Saved H0 PI plot → {fname}")
            if display_plot:
                plt.show()
            plt.close()
            
    elif h_dimension.lower() == 'h1':
        pimg = pimgr.transform([H1])
        feature_vector = pimg[0].flatten()
        
        if display_plot or save_plot:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(pimg[0], origin='lower')
            ax.set_title(f'H1 Persistence Image')
            if save_plot:
                plots_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                fname = os.path.join(plots_dir, f'PI_H1_{subject_id}.png')
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Saved H1 PI plot → {fname}")
            if display_plot:
                plt.show()
            plt.close()
            
    elif h_dimension.lower() == 'both':
        pimg_h0 = pimgr.transform([H0])
        pimg_h1 = pimgr.transform([H1])
        
        # Concatenate H0 and H1 features
        feature_vector = np.concatenate([pimg_h0[0].flatten(), pimg_h1[0].flatten()])
        
        if display_plot or save_plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(pimg_h0[0], origin='lower')
            axes[0].set_title('H0 Persistence Image')
            axes[1].imshow(pimg_h1[0], origin='lower')
            axes[1].set_title('H1 Persistence Image')
            if save_plot:
                plots_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                fname = os.path.join(plots_dir, f'PI_both_{subject_id}.png')
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Saved combined PI plot → {fname}")
            if display_plot:
                plt.show()
            plt.close()
    
    return feature_vector


# Main processing function for databank analysis

def main_analysis(
    expert_matrices,
    naive_matrices, 
    expert_names,
    naive_names,
    mapping,
    h_dimension='both',
    output_dir=None,
    save_features=True,
    save_plots=False
):
    """
    Main function to process expert and naive developer groups and extract persistence image features.
    
    Args:
        expert_matrices: array of expert connectivity matrices
        naive_matrices: array of naive connectivity matrices
        expert_names: array of expert subject names
        naive_names: array of naive subject names
        mapping: pandas DataFrame with anatomical mapping
        h_dimension: which homology to compute ('h0', 'h1', or 'both')
        output_dir: directory to save results
        save_features: whether to save feature vectors to CSV
        save_plots: whether to save persistence image plots
    
    Returns:
        expert_features: array of expert feature vectors
        naive_features: array of naive feature vectors
    """
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing expert developers...")
    expert_features = []
    for i, (matrix, name) in enumerate(zip(expert_matrices, expert_names)):
        print(f"Processing expert {i+1}/{len(expert_matrices)}: {name}")
        
        # Preprocess matrix
        distance_matrix = prepocess(matrix, mapping)
        
        # Compute persistence images
        features = compute_persistence_images(
            distance_matrix, 
            h_dimension=h_dimension,
            subject_id=name,
            output_dir=output_dir,
            save_plot=save_plots
        )
        expert_features.append(features)
    
    print("Processing naive developers...")
    naive_features = []
    for i, (matrix, name) in enumerate(zip(naive_matrices, naive_names)):
        print(f"Processing naive {i+1}/{len(naive_matrices)}: {name}")
        
        # Preprocess matrix
        distance_matrix = prepocess(matrix, mapping)
        
        # Compute persistence images
        features = compute_persistence_images(
            distance_matrix,
            h_dimension=h_dimension, 
            subject_id=name,
            output_dir=output_dir,
            save_plot=save_plots
        )
        naive_features.append(features)
    
    # Convert to arrays
    expert_features = np.array(expert_features)
    naive_features = np.array(naive_features)
    
    # Save features if requested
    if save_features:
        expert_df = pd.DataFrame(expert_features)
        expert_df['subject_id'] = expert_names
        expert_df['group'] = 'expert'
        
        naive_df = pd.DataFrame(naive_features)
        naive_df['subject_id'] = naive_names  
        naive_df['group'] = 'naive'
        
        # Combine and save
        combined_df = pd.concat([expert_df, naive_df], ignore_index=True)
        features_path = os.path.join(output_dir, f'persistence_image_features_{h_dimension}.csv')
        combined_df.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")
    
    return expert_features, naive_features


if __name__ == "__main__":
    print("This module contains functions for persistence image analysis.")
    print("Import and use main_analysis() function for processing connectivity data.")
