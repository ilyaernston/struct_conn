"""
Example script to load and visualize Betti curves analysis results.

This script demonstrates how to:
1. Load saved analysis results from the main script
2. Create custom visualizations
3. Perform additional statistical analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (12, 8),
    "figure.dpi": 150,
    "lines.linewidth": 2.5,
})

def load_betti_results(results_file: str) -> dict:
    """Load analysis results from NPZ file."""
    print(f"Loading results from: {results_file}")
    data = np.load(results_file, allow_pickle=True)
    
    # Extract available homology dimensions
    homology_dims = []
    for key in data.keys():
        if key.startswith('expert_mean_H'):
            dim = int(key.split('H')[1])
            homology_dims.append(dim)
    
    results = {
        'filtration_values': data['expert_filtration'],
        'expert_names': data['expert_names'],
        'naive_names': data['naive_names'],
        'homology_dims': sorted(homology_dims),
        'expert_mean': {},
        'expert_std': {},
        'naive_mean': {},
        'naive_std': {}
    }
    
    for dim in homology_dims:
        results['expert_mean'][dim] = data[f'expert_mean_H{dim}']
        results['expert_std'][dim] = data[f'expert_std_H{dim}']
        results['naive_mean'][dim] = data[f'naive_mean_H{dim}']
        results['naive_std'][dim] = data[f'naive_std_H{dim}']
    
    print(f"Loaded data for homology dimensions: {results['homology_dims']}")
    print(f"Expert subjects: {len(results['expert_names'])}")
    print(f"Naive subjects: {len(results['naive_names'])}")
    print(f"Filtration points: {len(results['filtration_values'])}")
    
    return results

def plot_individual_homology(results: dict, dim: int, output_dir: str = "."):
    """Plot individual homology dimension with enhanced statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    filtration = results['filtration_values']
    expert_mean = results['expert_mean'][dim]
    expert_std = results['expert_std'][dim]
    naive_mean = results['naive_mean'][dim]
    naive_std = results['naive_std'][dim]
    
    # Colors
    expert_color = '#1f77b4'
    naive_color = '#ff7f0e'
    
    # Plot 1: Mean curves with error bands
    ax1.plot(filtration, expert_mean, color=expert_color, linewidth=3, 
             label=f'Expert (n={len(results["expert_names"])})')
    ax1.fill_between(filtration, expert_mean - expert_std, expert_mean + expert_std, 
                     color=expert_color, alpha=0.2)
    
    ax1.plot(filtration, naive_mean, color=naive_color, linewidth=3, 
             label=f'Naive (n={len(results["naive_names"])})')
    ax1.fill_between(filtration, naive_mean - naive_std, naive_mean + naive_std, 
                     color=naive_color, alpha=0.2)
    
    ax1.set_xlabel('Filtration Parameter')
    ax1.set_ylabel(f'Betti Number $\\beta_{dim}$')
    ax1.set_title(f'$H_{dim}$ Betti Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference between groups
    difference = expert_mean - naive_mean
    ax2.plot(filtration, difference, color='purple', linewidth=3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(filtration, 0, difference, 
                     where=(difference > 0), color='blue', alpha=0.3, label='Expert > Naive')
    ax2.fill_between(filtration, 0, difference, 
                     where=(difference < 0), color='red', alpha=0.3, label='Naive > Expert')
    
    ax2.set_xlabel('Filtration Parameter')
    ax2.set_ylabel(f'Difference in $\\beta_{dim}$ (Expert - Naive)')
    ax2.set_title(f'$H_{dim}$ Group Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"H{dim}_detailed_analysis.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved detailed H{dim} analysis to: {filepath}")
    plt.show()

def compute_summary_statistics(results: dict):
    """Compute and display summary statistics."""
    print("\n=== DETAILED SUMMARY STATISTICS ===")
    
    for dim in results['homology_dims']:
        expert_mean = results['expert_mean'][dim]
        expert_std = results['expert_std'][dim]
        naive_mean = results['naive_mean'][dim]
        naive_std = results['naive_std'][dim]
        
        print(f"\nH{dim} Statistics:")
        print(f"  Expert - Max: {np.max(expert_mean):.3f}, Mean: {np.mean(expert_mean):.3f}, Std: {np.mean(expert_std):.3f}")
        print(f"  Naive  - Max: {np.max(naive_mean):.3f}, Mean: {np.mean(naive_mean):.3f}, Std: {np.mean(naive_std):.3f}")
        
        # Compute area under curve (AUC) as a summary measure
        expert_auc = float(np.trapz(expert_mean, results['filtration_values']))
        naive_auc = float(np.trapz(naive_mean, results['filtration_values']))
        print(f"  Area Under Curve - Expert: {expert_auc:.3f}, Naive: {naive_auc:.3f}")
        print(f"  AUC Difference (Expert - Naive): {expert_auc - naive_auc:.3f}")

def plot_summary_grid(results: dict, output_dir: str = "."):
    """Create a comprehensive summary grid plot."""
    n_dims = len(results['homology_dims'])
    fig, axes = plt.subplots(2, n_dims, figsize=(5 * n_dims, 10))
    if n_dims == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['#1f77b4', '#ff7f0e']
    
    for i, dim in enumerate(results['homology_dims']):
        filtration = results['filtration_values']
        
        # Top row: Betti curves
        ax_top = axes[0, i] if n_dims > 1 else axes[0]
        expert_mean = results['expert_mean'][dim]
        expert_std = results['expert_std'][dim]
        naive_mean = results['naive_mean'][dim]
        naive_std = results['naive_std'][dim]
        
        ax_top.plot(filtration, expert_mean, color=colors[0], linewidth=2.5, label='Expert')
        ax_top.fill_between(filtration, expert_mean - expert_std, expert_mean + expert_std, 
                           color=colors[0], alpha=0.2)
        
        ax_top.plot(filtration, naive_mean, color=colors[1], linewidth=2.5, label='Naive')
        ax_top.fill_between(filtration, naive_mean - naive_std, naive_mean + naive_std, 
                           color=colors[1], alpha=0.2)
        
        ax_top.set_title(f'$H_{dim}$ Betti Curves')
        ax_top.set_ylabel(f'$\\beta_{dim}$')
        ax_top.legend()
        ax_top.grid(True, alpha=0.3)
        
        # Bottom row: Statistics
        ax_bottom = axes[1, i] if n_dims > 1 else axes[1]
        
        # Compute statistics for visualization
        max_vals = [np.max(expert_mean), np.max(naive_mean)]
        auc_vals = [np.trapz(expert_mean, filtration), np.trapz(naive_mean, filtration)]
        
        x_pos = np.arange(2)
        width = 0.35
        
        bars1 = ax_bottom.bar(x_pos - width/2, max_vals, width, label='Max Betti', color=colors)
        bars2 = ax_bottom.bar(x_pos + width/2, auc_vals, width, label='AUC', color=colors, alpha=0.7)
        
        ax_bottom.set_title(f'$H_{dim}$ Summary Statistics')
        ax_bottom.set_ylabel('Value')
        ax_bottom.set_xticks(x_pos)
        ax_bottom.set_xticklabels(['Expert', 'Naive'])
        ax_bottom.legend()
        ax_bottom.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax_bottom.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax_bottom.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    filename = "betti_summary_grid.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved summary grid to: {filepath}")
    plt.show()

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Load and visualize Betti curves analysis results"
    )
    
    parser.add_argument(
        "--results_file", 
        type=str, 
        required=True,
        help="Path to NPZ results file from main analysis"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./visualization_output",
        help="Output directory for additional plots (default: ./visualization_output)"
    )
    
    parser.add_argument(
        "--individual_plots", 
        action="store_true",
        help="Create individual detailed plots for each homology dimension"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_betti_results(args.results_file)
    
    # Compute and display summary statistics
    compute_summary_statistics(results)
    
    # Create summary grid plot
    plot_summary_grid(results, args.output_dir)
    
    # Create individual plots if requested
    if args.individual_plots:
        print("\nCreating individual homology dimension plots...")
        for dim in results['homology_dims']:
            plot_individual_homology(results, dim, args.output_dir)
    
    print(f"\nVisualization completed! Check outputs in: {args.output_dir}")

if __name__ == "__main__":
    main()
