'''
Core functions to compute persistence diagrams
'''

import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Tuple, Optional

def compute_persistence(distance_matrix: np.ndarray, max_edge_length: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
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

def plot_persistence_diagram(
    H0: np.ndarray,
    H1: np.ndarray,
    title: str = "Persistence Diagram",
    plot_mode: str = "save",
    save_dir: Optional[str] = None
):
    """Visualize H0 and H1 persistence diagrams"""
    save_dir = save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(H0) > 0:
        ax.scatter(
            H0[:, 0], H0[:, 1],
            c='#1f77b4', marker='^', s=80,
            label='H0', alpha=0.7, edgecolors='w'
        )
    if len(H1) > 0:
        ax.scatter(
            H1[:, 0], H1[:, 1],
            c='#ff7f0e', marker='o', s=80,
            label='H1', alpha=0.7, edgecolors='w'
        )

    all_deaths = []
    if len(H0) > 0: all_deaths.extend(H0[:, 1])
    if len(H1) > 0: all_deaths.extend(H1[:, 1])
    max_val = max([1.0] + all_deaths) if all_deaths else 1.0
    ax.plot([0, max_val], [0, max_val], '--', color='#2ca02c', alpha=0.7)

    ax.set(xlabel='Birth', ylabel='Death', title=title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.tight_layout()

    safe_title = re.sub(r'[^\w\-_\.]', '_', title)
    fname = os.path.join(save_dir, f"PD_{safe_title}.png")
    if plot_mode in ('save', 'both'):
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved persistence diagram to {fname}")
    if plot_mode in ('show', 'both'):
        plt.show()

    plt.close(fig)