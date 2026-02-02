import pandas as pd
import ast
from pathlib import Path
from typing import Dict, Optional, Tuple
import os
from collections import defaultdict
import argparse


def compute_loop_participation(h1_loops_df: pd.DataFrame, num_nodes: int = 246) -> pd.DataFrame:
    """Compute the number of persistent H1 loops each node participates in for each subject.
    
    Args:
        h1_loops_df (pd.DataFrame): DataFrame with 'subject_id' and 'nodes' columns.
        num_nodes (int): Total number of nodes in the network (default: 246 for Fan2016 atlas with dropped cerebellum).
        
    Returns:
        pd.DataFrame: DataFrame with columns [subject_id, node_id, n_loops].
    """
    # Dictionary to store participation counts per subject
    # Structure: {subject_id: {node_id: count}}
    subject_participation = defaultdict(lambda: defaultdict(int))
    
    # Process each loop
    for _, row in h1_loops_df.iterrows():
        subject_id = row['subject_id']
        try:
            # Convert string representation to actual list
            nodes_list = ast.literal_eval(row['nodes'])
            
            # Count each node's participation in this loop
            for node in nodes_list:
                subject_participation[subject_id][node] += 1
                
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse nodes string: {row['nodes']}")
            continue
    
    # Convert to DataFrame format
    loop_results = []
    for subject_id in subject_participation:
        for node_id in range(num_nodes):
            n_loops = subject_participation[subject_id].get(node_id, 0)
            loop_results.append({
                'subject_id': subject_id,
                'node_id': node_id,
                'n_loops': n_loops
            })
    
    return pd.DataFrame(loop_results)

def compute_clique_participation(cliques_df: pd.DataFrame, num_nodes: int) -> pd.DataFrame:
    """Compute the number of max cliques each node participates in for each subject.
    
    Args:
        cliques_df (pd.DataFrame): DataFrame with 'subject_id' and 'nodes' columns.
        num_nodes (int): Total number of nodes in the network (default: 246 for Fan2016 atlas with dropped cerebellum).
        
    Returns:
        pd.DataFrame: DataFrame with columns [subject_id, node_id, n_cliques].
    """
    # Dictionary to store participation counts per subject
    # Structure: {subject_id: {node_id: count}}
    subject_participation = defaultdict(lambda: defaultdict(int))
    
    # Process each clique
    for _, row in cliques_df.iterrows():
        subject_id = row['subject_id']
        try:
            # Convert string representation to actual list
            nodes_list = ast.literal_eval(row['nodes'])
            
            # Count each node's participation in this clique
            for node in nodes_list:
                subject_participation[subject_id][node] += 1
                
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse nodes string: {row['nodes']}")
            continue
    
    # Convert to DataFrame format
    clique_results = []
    for subject_id in subject_participation:
        for node_id in range(num_nodes):
            n_cliques = subject_participation[subject_id].get(node_id, 0)
            clique_results.append({
                'subject_id': subject_id,
                'node_id': node_id,
                'n_cliques': n_cliques
            })

    return pd.DataFrame(clique_results)


def main(loop_path: str,
         clique_path: str,
         num_nodes: int = 246, 
         output_dir: Optional[str] = None):
    """Extract node participation in H1 loops from CSV file, preserving subject-node correspondence.
    
    Args:
        loop_path (str): Path to the h1_loops.csv file.
        num_nodes (int): Total number of nodes in the network (default: 246 for Fan2016 atlas).
        output_dir (str, optional): Directory to save the participation results as CSV.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames with loop and clique participation results
    """

    # Compute clique participation
    print(f"\nComputing loop participation...")
    # Load H1 loops data
    print(f"Loading H1 loops data from {loop_path}...")
    h1_loops_df = pd.read_csv(loop_path)
    print(f"Loaded {len(h1_loops_df)} H1 loops from {h1_loops_df['subject_id'].nunique()} subjects")
    
    # Compute node participation by subject
    print("Computing node participation in H1 loops for each subject...")
    loop_participation_df = compute_loop_participation(h1_loops_df, num_nodes)
    
    # Display summary statistics
    print("\n=== Node Participation Summary ===")
    print(f"Total subjects: {loop_participation_df['subject_id'].nunique()}")
    print(f"Total nodes: {num_nodes}")
    print(f"Average loops per node: {loop_participation_df['n_loops'].mean():.2f}")
    
    # Top participating nodes across all subjects
    print("\nTop 10 node-subject pairs by loop participation:")
    top_pairs = loop_participation_df[loop_participation_df['n_loops'] > 0].nlargest(10, 'n_loops')
    print(top_pairs.to_string(index=False))


    # Compute clique participation
    print(f"\nComputing clique participation...")
    # Load cliques data
    print(f"Loading cliques data from {clique_path}...")
    cliques_df = pd.read_csv(clique_path)
    print(f"Loaded {len(cliques_df)} cliques from {cliques_df['subject_id'].nunique()} subjects")

    # Compute node participation by subject
    print("Computing node participation in cliques for each subject...")
    clique_participation_df = compute_clique_participation(cliques_df, num_nodes)
    
    
    # Display summary statistics
    print("\n=== Node Participation Summary ===")
    print(f"Total subjects: {clique_participation_df['subject_id'].nunique()}")
    print(f"Total nodes: {num_nodes}")
    print(f"Average cliques per node: {clique_participation_df['n_cliques'].mean():.2f}")
    
    # Top participating nodes across all subjects
    print("\nTop 10 node-subject pairs by clique participation:")
    top_pairs = clique_participation_df[clique_participation_df['n_cliques'] > 0].nlargest(10, 'n_cliques')
    print(top_pairs.to_string(index=False))
    
    # Save to CSV if output directory provided
    if output_dir:
        loop_output_path = os.path.join(output_dir, 'node_loop_participation.csv')
        loop_participation_df.to_csv(loop_output_path, index=False)
        print(f"\nSaved loop participation results to {loop_output_path}")

        clique_output_path = os.path.join(output_dir, 'node_clique_participation.csv')
        clique_participation_df.to_csv(clique_output_path, index=False)
        print(f"Saved clique participation results to {clique_output_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute node participation in H1 loops")
    parser.add_argument(
        "-l", "--loop_data_dir",
        type=str,
        help="Path to the h1_loops.csv file"
    )
    parser.add_argument(
        "-c", "--clique_data_dir",
        type=str,
        help="Path to the clique_measures.csv file"
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=246,
        help="Total number of nodes in the network (default: 246; for Fan2016 atlas with dropped cerebellum)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Output directory for saving participation results as CSV"
    )
    
    args = parser.parse_args()

     # Generate default output directory based on actually used data_dir
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to struct_conn directory (two levels up from _legacy)
        pack_dir = os.path.dirname(os.path.dirname(script_dir))
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        data_folder_name = os.path.basename(args.loop_data_dir)
        default_output_name = f'node_loop_participation_{current_time}'
        args.output_dir = os.path.join(pack_dir, 'output', 'node_participation', default_output_name)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.loop_data_dir, args.clique_data_dir, args.n_nodes, args.output_dir)
