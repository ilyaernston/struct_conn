"""
Clique Loop Incidence Calculation

Compute loop incidence measure for each clique:
- Loop incidence = number of loops that touch the clique (share at least one node)
- Combines clique_measures with h1_loops
- Adds 'loop_incidence' column to clique measures
- Saves augmented file for downstream correlation analysis

Usage: python clique_loop_incidence.py -l h1_loops.csv -c clique_measures.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os


def load_dataframe(file_path, usecols=None):
    """Load DataFrame from CSV or Parquet file based on extension.
    
    Args:
        file_path: Path to the file (CSV or Parquet)
        usecols: List of columns to load (for memory efficiency), or None to load all
        
    Returns:
        pandas DataFrame
    """
    file_path_str = str(file_path)
    if file_path_str.endswith('.parquet'):
        return pd.read_parquet(file_path, columns=usecols)
    elif file_path_str.endswith('.csv'):
        return pd.read_csv(file_path, usecols=usecols, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Expected .csv or .parquet")


def parse_nodes_column(nodes_str):
    """Parse the nodes column which may be a comma-separated string or list-like.
    
    Args:
        nodes_str: String of comma-separated node indices, a list, or string representation of list
        
    Returns:
        Set of node indices (as integers)
    """
    if isinstance(nodes_str, str):
        # Handle string representation of list like "[1, 2, 3]"
        if nodes_str.startswith('[') and nodes_str.endswith(']'):
            import ast
            return set(ast.literal_eval(nodes_str))
        # Handle comma-separated string like "1,2,3"
        return set(int(x) for x in nodes_str.split(',') if x.strip())
    elif isinstance(nodes_str, list):
        return set(nodes_str)
    else:
        return set()


def calculate_loop_incidence(clique_measures_df, h1_loops_df):
    """Calculate loop incidence for each clique.
    
    Loop incidence is defined as the number of unique loops that touch the clique
    (i.e., share at least one node with the clique), within the same subject.
    
    Args:
        clique_measures_df: DataFrame with columns: subject_id, clique_index, nodes
        h1_loops_df: DataFrame with columns: subject_id, persistence_id, nodes
        
    Returns:
        DataFrame with added 'loop_incidence' column
    """
    print("Calculating loop incidence for each clique...")
    
    # Group loops by subject_id for efficient lookup
    loops_by_subject = {}
    for subject_id, group in h1_loops_df.groupby('subject_id'):
        loops_by_subject[subject_id] = [
            parse_nodes_column(row['nodes']) for _, row in group.iterrows()
        ]
    
    print(f"  Organized loops for {len(loops_by_subject)} subjects")
    
    loop_incidence_list = []
    
    for idx, row in clique_measures_df.iterrows():
        subject_id = row['subject_id']
        clique_nodes = parse_nodes_column(row['nodes'])
        
        # Get all loops for this subject
        subject_loops = loops_by_subject.get(subject_id, [])
        
        # Count how many loops touch this clique (share at least one node)
        incidence = 0
        for loop_nodes in subject_loops:
            if clique_nodes & loop_nodes:  # Set intersection - any common nodes
                incidence += 1
        
        loop_incidence_list.append(incidence)
        
        # Print progress every 10%
        if (idx + 1) % max(1, len(clique_measures_df) // 10) == 0:  
            pc = (idx + 1) / len(clique_measures_df) * 100
            print(f"  Processed {idx + 1}/{len(clique_measures_df)} cliques ({pc:.1f}%)")
    
    # Add loop_incidence column to DataFrame
    result_df = clique_measures_df.copy()
    result_df['loop_incidence'] = loop_incidence_list
    
    print(f"Completed loop incidence calculation")
    print(f"  Mean loop incidence: {np.mean(loop_incidence_list):.2f}")
    print(f"  Median loop incidence: {np.median(loop_incidence_list):.2f}")
    print(f"  Max loop incidence: {max(loop_incidence_list)}")
    print(f"  Min loop incidence: {min(loop_incidence_list)}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate loop incidence for cliques, i.e. number of loops that share at least one node with given clique'
    )
    parser.add_argument('-l', '--h1_loops', type=str, required=True,
                        help='Path to h1_loops file (CSV or Parquet)')
    parser.add_argument('-c', '--clique_measures', type=str, required=True,
                        help='Path to clique_measures file (CSV or Parquet)')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory for results (default: auto-generated)')
    parser.add_argument('--export_format', type=str, choices=['csv', 'parquet', 'both'], default='csv',
                        help='Export format for results (default: csv)')
    
    args = parser.parse_args()
    
    # Generate default output directory if not provided
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            os.path.dirname(script_dir), 
            'output', 
            'clique_loop_incidence',
            f'clique_loop_incidence_{current_time}'
        )
    
    print("="*80)
    print("Clique Loop Incidence Calculation")
    print("="*80)
    print(f"H1 loops file: {args.h1_loops}")
    print(f"Clique measures file: {args.clique_measures}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    # Load only necessary columns from h1_loops for memory efficiency
    h1_loops_df = load_dataframe(args.h1_loops, 
                                 usecols=['subject_id', 'persistence_id', 'nodes'])
    print(f"  Loaded {len(h1_loops_df)} loops from h1_loops")
    print(f"  Columns: {list(h1_loops_df.columns)}")
    
    # Load all columns from clique_measures to preserve them in output
    clique_measures_df = load_dataframe(args.clique_measures)
    print(f"  Loaded {len(clique_measures_df)} cliques from clique_measures")
    print(f"  Columns: {list(clique_measures_df.columns)}")
    
    # Calculate loop incidence
    clique_with_incidence = calculate_loop_incidence(clique_measures_df, h1_loops_df)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.export_format in ['csv', 'both']:
        csv_path = output_path / 'clique_measures_with_loop_incidence.csv'
        clique_with_incidence.to_csv(csv_path, index=False)
        print(f"\nSaved clique measures with loop incidence to {csv_path}")

    if args.export_format in ['parquet', 'both']:
        parquet_path = output_path / 'clique_measures_with_loop_incidence.parquet'
        clique_with_incidence.to_parquet(parquet_path, index=False)
        print(f"\nSaved clique measures with loop incidence to {parquet_path}")
    
    print("\n" + "="*80)
    print("Calculation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
