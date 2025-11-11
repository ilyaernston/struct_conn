"""
Clique analysis module for structural connectivity networks.

This module provides tools for detecting and analyzing maximal cliques in brain
connectivity networks using igraph, mapping them to anatomical regions, and 
computing clique metrics.

Features:
- Detects maximal cliques using igraph (min size 4)
- Computes clique metrics: internal degree, external degree, conductance, boundary ratio
- Maps cliques to Yeo-7/17 networks and anatomical regions
- Computes node participation in maximal cliques (optimized)
"""

from .clique_mapping import (
    compute_node_participation,
    compute_clique_metrics,
    detect_cliques,
    map_cliques_to_regions,
    analyze_single_matrix,
    main
)

__all__ = [
    'compute_node_participation',
    'compute_clique_metrics',
    'detect_cliques',
    'map_cliques_to_regions',
    'analyze_single_matrix',
    'main'
]

__version__ = '2.0.0'
__author__ = 'elijah' 
