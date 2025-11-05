"""
Clique analysis module for structural connectivity networks.

This module provides tools for detecting and analyzing maximal cliques in brain
connectivity networks, mapping them to anatomical regions, and visualizing results.

Features:
- Detects maximal cliques and computes network properties (degree, betweenness)
- Computes group centrality metrics (group betweenness, closeness, degree)
- Maps cliques to Yeo-7/17 networks and anatomical regions
- Creates comprehensive visualizations with save/show options
"""

from .clique_mapping import (
    detect_cliques,
    map_cliques_to_regions,
    visualize,
    analyze_single_matrix,
    main
)

__all__ = [
    'detect_cliques',
    'map_cliques_to_regions',
    'visualize',
    'analyze_single_matrix',
    'main'
]

__version__ = '1.1.0'
__author__ = 'elijah' 
