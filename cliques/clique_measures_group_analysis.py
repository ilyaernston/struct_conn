"""
Group-level statistical analysis of clique-loop incidence correlations.

This script applies Fisher z-transformation methodology to analyze correlations
between loop incidence and clique metrics across subjects:

1. For each subject: Calculate Spearman correlation between loop_incidence and each clique metric
2. Apply Fisher z-transformation to correlation coefficients
3. Test if mean z differs from 0 using one-sample t-test
4. Calculate Cohen's d effect size
5. Generate visualizations of correlation distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List
import argparse
import os

from utils import load_dataframe

def calculate_subject_correlations(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Calculate Spearman correlations for each subject between loop_incidence and clique metrics.
    
    Args:
        df: DataFrame with columns [subject_id, loop_incidence, <metrics>]
        metrics: List of metric column names to correlate with loop_incidence
        
    Returns:
        DataFrame with columns [subject_id, metric, r, z, n_cliques] where:
            - r: Spearman correlation coefficient
            - z: Fisher z-transformed correlation
            - n_cliques: Number of cliques used in correlation
    """
    print("\nCalculating subject-wise correlations...")
    
    results = []
    
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id]
        
        # Filter out any rows with missing loop_incidence
        subject_data = subject_data.dropna(subset=['loop_incidence'])
        
        n_cliques = len(subject_data)
        
        if n_cliques < 3:
            print(f"  Warning: Subject {subject_id} has only {n_cliques} cliques, skipping")
            continue
        
        for metric in metrics:
            # Filter out NaN values for this specific metric
            valid_data = subject_data.dropna(subset=[metric, 'loop_incidence'])
            
            if len(valid_data) < 3:
                print(f"  Warning: Subject {subject_id}, metric {metric} has insufficient data")
                results.append({
                    'subject_id': subject_id,
                    'metric': metric,
                    'r': np.nan,
                    'z': np.nan,
                    'n_cliques': len(valid_data)
                })
                continue
            
            x = valid_data[metric].values
            y = valid_data['loop_incidence'].values
            
            # Check for zero variance
            if np.std(x) == 0 or np.std(y) == 0:
                print(f"  Warning: Zero variance for subject {subject_id}, metric {metric}")
                results.append({
                    'subject_id': subject_id,
                    'metric': metric,
                    'r': np.nan,
                    'z': np.nan,
                    'n_cliques': len(valid_data)
                })
                continue
            
            # Calculate Spearman correlation
            r, _ = stats.spearmanr(x, y)
            
            # Ensure r is a scalar (not a tuple or array)
            r = float(r) # type: ignore
            
            # Fisher z-transformation
            # Handle edge cases where r is very close to ±1
            if r >= 0.9999:
                r = 0.9999
            elif r <= -0.9999:
                r = -0.9999
            
            z = np.arctanh(r)
            
            results.append({
                'subject_id': subject_id,
                'metric': metric,
                'r': r,
                'z': z,
                'n_cliques': len(valid_data)
            })
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics per metric
    print(f"\nCalculated correlations for {len(results_df['subject_id'].unique())} subjects")
    for metric in metrics:
        metric_data = results_df[results_df['metric'] == metric].dropna(subset=['r'])
        if len(metric_data) > 0:
            print(f"\n  {metric}:")
            print(f"    N subjects: {len(metric_data)}")
            print(f"    Mean r: {metric_data['r'].mean():.3f} (SD: {metric_data['r'].std():.3f})")
            print(f"    Median r: {metric_data['r'].median():.3f}")
            print(f"    Range r: [{metric_data['r'].min():.3f}, {metric_data['r'].max():.3f}]")
    
    return results_df


def test_group_effect(results_df: pd.DataFrame, metric: str) -> Dict:
    """Test whether mean Fisher z differs from 0 for a specific metric.
    
    Args:
        results_df: DataFrame with subject-wise correlation results
        metric: Name of the metric to test
        
    Returns:
        Dictionary with test results including:
            - metric: Name of the metric
            - n_subjects: Number of subjects
            - mean_z: Mean Fisher z
            - sd_z: Standard deviation of Fisher z
            - mean_r: Mean correlation (back-transformed from mean z)
            - t_stat: t-statistic
            - p_value: p-value from one-sample t-test
            - df: Degrees of freedom
            - cohens_d: Cohen's d effect size
            - ci_lower: Lower bound of 95% CI for mean z
            - ci_upper: Upper bound of 95% CI for mean z
    """
    metric_data = results_df[results_df['metric'] == metric].dropna(subset=['z'])
    
    if len(metric_data) < 2:
        return {
            'metric': metric,
            'n_subjects': len(metric_data),
            'mean_z': np.nan,
            'sd_z': np.nan,
            'mean_r': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'df': np.nan,
            'cohens_d': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_lower_r': np.nan,
            'ci_upper_r': np.nan
        }
    
    z_values = list(metric_data['z'].values)
    n = len(z_values)
    
    # Calculate mean and SD of Fisher z
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values, ddof=1)
    
    # Back-transform mean z to correlation
    mean_r = np.tanh(mean_z)
    
    # One-sample t-test against 0
    t_stat, p_value = stats.ttest_1samp(z_values, 0)
    
    # Cohen's d effect size (for one-sample test)
    cohens_d = mean_z / sd_z if sd_z > 0 else np.nan
    
    # 95% confidence interval for mean z
    ci = stats.t.interval(0.95, df=n-1, loc=mean_z, scale=sd_z/np.sqrt(n))
    
    results = {
        'metric': metric,
        'n_subjects': n,
        'mean_z': mean_z,
        'sd_z': sd_z,
        'mean_r': mean_r,
        't_stat': t_stat,
        'p_value': p_value,
        'df': n - 1,
        'cohens_d': cohens_d,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'ci_lower_r': np.tanh(ci[0]),
        'ci_upper_r': np.tanh(ci[1])
    }
    
    return results


def print_statistical_report(test_results: List[Dict]) -> None:
    """Print formatted statistical report for all metrics.
    
    Args:
        test_results: List of dictionaries with test results from test_group_effect
    """
    print("\n" + "="*80)
    print("STATISTICAL REPORT: Group-Level Correlation Analysis")
    print("Loop Incidence vs Clique Metrics")
    print("="*80)
    
    for result in test_results:
        metric = result['metric']
        
        print(f"\n{'='*80}")
        print(f"METRIC: {metric.replace('_', ' ').title()}")
        print(f"{'='*80}")
        
        print(f"\nSample Size: N = {result['n_subjects']} subjects")
        
        print("\nFisher z-transformed correlations:")
        print(f"  Mean z = {result['mean_z']:.4f} (SD = {result['sd_z']:.4f})")
        print(f"  95% CI for z: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        
        print("\nBack-transformed to Spearman r:")
        print(f"  Mean r = {result['mean_r']:.4f}")
        print(f"  95% CI for r: [{result['ci_lower_r']:.4f}, {result['ci_upper_r']:.4f}]")
        
        print("\nOne-sample t-test (H0: mean z = 0):")
        print(f"  t({result['df']}) = {result['t_stat']:.4f}")
        print(f"  p = {result['p_value']:.6f}")
        
        if result['p_value'] < 0.001:
            sig_str = "p < 0.001 ***"
        elif result['p_value'] < 0.01:
            sig_str = "p < 0.01 **"
        elif result['p_value'] < 0.05:
            sig_str = "p < 0.05 *"
        else:
            sig_str = "p ≥ 0.05 (not significant)"
        print(f"  Significance: {sig_str}")
        
        print("\nEffect Size:")
        print(f"  Cohen's d = {result['cohens_d']:.4f}")
        
        # Effect size interpretation
        d_abs = abs(result['cohens_d'])
        if np.isnan(d_abs):
            effect_interp = "undefined"
        elif d_abs < 0.2:
            effect_interp = "negligible"
        elif d_abs < 0.5:
            effect_interp = "small"
        elif d_abs < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        print(f"  Interpretation: {effect_interp} effect")
        
        print("\nConclusion:")
        if result['p_value'] < 0.05:
            direction = "positive" if result['mean_r'] > 0 else "negative"
            print(f"  Significant {direction} correlation between loop incidence and {metric}")
        else:
            print(f"  No significant correlation between loop incidence and {metric}")
    
    print("\n" + "="*80)


def plot_correlation_distributions(results_df: pd.DataFrame, metrics: List[str], 
                                   output_dir: Path, show_plot: bool = False,
                                   colors = None) -> None:
    """Create visualization of correlation distributions for all metrics.
    
    Args:
        results_df: DataFrame with subject-wise correlation results
        metrics: List of metric names
        output_dir: Directory to save plots
        show_plot: Whether to display plots interactively
        colors: List of colors for the plots
    """
    print("\nGenerating correlation distribution plots...")
    
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure for Spearman r distributions
    fig_r, axes_r = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes_r = axes_r.flatten() if n_metrics > 1 else [axes_r]
    
    # Create figure for Fisher z distributions
    fig_z, axes_z = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes_z = axes_z.flatten() if n_metrics > 1 else [axes_z]
    
    for idx, metric in enumerate(metrics):
        metric_data = results_df[results_df['metric'] == metric].dropna(subset=['r', 'z'])
        
        if len(metric_data) == 0:
            axes_r[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes_r[idx].set_title(metric.replace('_', ' ').title())
            axes_z[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes_z[idx].set_title(metric.replace('_', ' ').title())
            continue
        
        mean_r = metric_data['r'].mean()
        mean_z = metric_data['z'].mean()
        
        # Plot Spearman r distribution
        ax_r = axes_r[idx]
        ax_r.hist(metric_data['r'], bins=20, edgecolor='black', alpha=0.7, color=colors[0])
        ax_r.axvline(mean_r, color='red', linestyle='--', linewidth=2, 
                    label=f"Mean ρ = {mean_r:.3f}")
        ax_r.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_r.set_xlabel('Spearman ρ', fontsize=10)
        ax_r.set_ylabel('Frequency', fontsize=10)
        ax_r.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax_r.legend(fontsize=9)
        ax_r.grid(True, alpha=0.3)
        
        # Plot Fisher z distribution
        ax_z = axes_z[idx]
        ax_z.hist(metric_data['z'], bins=20, edgecolor='black', alpha=0.7, color=colors[1])
        ax_z.axvline(mean_z, color='red', linestyle='--', linewidth=2, 
                    label=f"Mean z = {mean_z:.3f}")
        ax_z.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_z.set_xlabel('Fisher z', fontsize=10)
        ax_z.set_ylabel('Frequency', fontsize=10)
        ax_z.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax_z.legend(fontsize=9)
        ax_z.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes_r)):
        axes_r[idx].axis('off')
        axes_z[idx].axis('off')
    
    # Adjust layout and save
    fig_r.suptitle('Distribution of Spearman Correlations (ρ) Across Subjects', 
                   fontsize=14, fontweight='bold', y=0.995)
    fig_r.tight_layout()
    
    fig_z.suptitle('Distribution of Fisher z-Transformed Correlations Across Subjects', 
                   fontsize=14, fontweight='bold', y=0.995)
    fig_z.tight_layout()
    
    # Save plots
    r_plot_path = output_dir / 'spearman_r_distributions.png'
    fig_r.savefig(r_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved Spearman r distributions to {r_plot_path}")
    
    z_plot_path = output_dir / 'fisher_z_distributions.png'
    fig_z.savefig(z_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved Fisher z distributions to {z_plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig_r)
        plt.close(fig_z)


def plot_effect_size_summary(test_results: List[Dict], output_dir: Path, 
                             show_plot: bool = False, colors = None) -> None:
    """Create summary visualization of effect sizes and significance.
    
    Args:
        test_results: List of test result dictionaries
        output_dir: Directory to save plot
        show_plot: Whether to display plot interactively
        colors: List of colors for the plots
    """
    print("\nGenerating effect size summary plot...")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(test_results)
    results_df = results_df.sort_values('mean_r', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean r with confidence intervals
    x_pos = np.arange(len(results_df))
    colors = ['green' if p < 0.05 else 'gray' for p in results_df['p_value']]
    
    ax1.bar(x_pos, results_df['mean_r'], color=colors, alpha=0.7, edgecolor='black')
    
    # Add error bars (95% CI)
    ci_errors = np.array([
        results_df['mean_r'] - results_df['ci_lower_r'],
        results_df['ci_upper_r'] - results_df['mean_r']
    ])
    ax1.errorbar(x_pos, results_df['mean_r'], yerr=ci_errors, 
                 fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Add significance stars
    for i, (r, p) in enumerate(zip(results_df['mean_r'], results_df['p_value'])):
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        y_pos = r + (0.05 if r > 0 else -0.08)
        ax1.text(i, y_pos, stars, ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Clique Metric', fontsize=12)
    ax1.set_ylabel('Mean Spearman Correlation (ρ)', fontsize=12)
    ax1.set_title('Mean Correlations with Loop Incidence\n(with 95% CI)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.replace('clique_', '').replace('_', ' ').title() 
                         for m in results_df['metric']], rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[2], alpha=0.7, edgecolor='black', label='p < 0.05'),
        Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='p ≥ 0.05')
    ]
    ax1.legend(handles=legend_elements, loc='best')
    
    # Plot 2: Cohen's d effect sizes
    colors_d = [colors[2] if p < 0.05 else 'gray' for p in results_df['p_value']]
    
    ax2.bar(x_pos, results_df['cohens_d'], color=colors_d, alpha=0.7, edgecolor='black')
    
    # Add horizontal lines for effect size thresholds
    ax2.axhline(y=0.2, color='lightgray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.8, color='darkgray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-0.2, color='lightgray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-0.8, color='darkgray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add text labels for thresholds
    ax2.text(len(results_df) - 0.5, 0.2, 'small', ha='right', va='bottom', 
             fontsize=8, color='gray', alpha=0.7)
    ax2.text(len(results_df) - 0.5, 0.5, 'medium', ha='right', va='bottom', 
             fontsize=8, color='gray', alpha=0.7)
    ax2.text(len(results_df) - 0.5, 0.8, 'large', ha='right', va='bottom', 
             fontsize=8, color='gray', alpha=0.7)
    
    ax2.set_xlabel('Clique Metric', fontsize=12)
    ax2.set_ylabel("Cohen's d", fontsize=12)
    ax2.set_title('Effect Sizes', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.replace('clique_', '').replace('_', ' ').title() 
                         for m in results_df['metric']], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    summary_plot_path = output_dir / 'group_effect_summary.png'
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved effect size summary to {summary_plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_results_csv(results_df: pd.DataFrame, test_results: List[Dict], 
                     output_dir: Path) -> None:
    """Save detailed results to CSV files.
    
    Args:
        results_df: DataFrame with subject-wise correlation results
        test_results: List of test result dictionaries
        output_dir: Directory to save results
    """
    # Save subject-wise results
    subject_results_path = output_dir / 'subject_correlations.csv'
    results_df.to_csv(subject_results_path, index=False)
    print(f"\nSaved subject-wise results to {subject_results_path}")
    
    # Save group-level results
    group_results = pd.DataFrame(test_results)
    group_results_path = output_dir / 'group_statistics.csv'
    group_results.to_csv(group_results_path, index=False)
    print(f"Saved group-level statistics to {group_results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Group-level analysis of clique-loop incidence correlations using Fisher z-transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Pipeline:
1. For each subject: Calculate Spearman correlation between loop_incidence and each clique metric
2. Apply Fisher z-transformation to correlation coefficients
3. Test whether mean z differs from 0 using one-sample t-test
4. Calculate Cohen's d effect size
5. Generate visualizations of correlation distributions

Required CSV columns:
  - subject_id: Subject identifier
  - clique_index: Clique identifier (optional, not used in analysis)
  - loop_incidence: Number of loops touching the clique
  - clique_size: Number of nodes in clique
  - clique_volume: Number of edges in clique
  - clique_avg_degree: Average degree of nodes in clique
  - clique_boundary_edges: Number of edges connecting clique to rest of network
  - clique_boundary_ratio: Ratio of boundary edges to internal edges
  - clique_avg_embeddedness: Average embeddedness of nodes in clique

Examples:
  # Basic analysis
  python clique_measures_group_analysis.py -m clique_measures_with_loop_incidence.csv
  
  # With custom output and interactive plots
  python clique_measures_group_analysis.py -m clique_measures.csv -o ./results --show
        """
    )
    
    parser.add_argument('-m', '--measures_data', type=str, required=True,
                        help='Path to CSV file with clique measures and loop incidence, e.g. clique_measures_with_loop_incidence.csv from clique_loop_incidence.py')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory for results (default: auto-generated)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')
    parser.add_argument('--measures', type=str, nargs='+', default='all',   
                        help='List of clique metrics to show on plots (default: all). ' \
                        'Available metrics: clique_size, clique_volume, clique_avg_degree, clique_boundary_edges, clique_boundary_ratio, clique_avg_embeddedness')
    parser.add_argument('--min_size', type=int, default=4,
                        help='Minimum clique size to include in analysis (default: 4)')
    parser.add_argument('--max_size', type=int, default=None,
                        help='Maximum clique size to include in analysis (default: no limit)')
    
    args = parser.parse_args()
    
    # Generate default output directory if not provided
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            os.path.dirname(script_dir), 
            'output', 
            'clique_measures_group_analysis',
            f'group_analysis_{current_time}'
        )
    
    print("="*80)
    print("Group-Level Clique-Loop Incidence Correlation Analysis")
    print("="*80)
    print(f"Input file: {args.measures_data}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_dataframe(args.measures_data)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Subjects: {df['subject_id'].nunique()}")

    # Apply clique size filtering if specified
    if args.min_size:
        original_count = len(df)
        df = df[df['clique_size'] >= args.min_size]
        filtered_count = len(df)
        print(f"  Applied min_size={args.min_size}: {original_count} -> {filtered_count} rows")
    
    # Apply max_size filtering if specified
    if args.max_size:
        original_count = len(df)
        df = df[df['clique_size'] <= args.max_size]
        filtered_count = len(df)
        print(f"  Applied max_size={args.max_size}: {original_count} -> {filtered_count} rows")
    
    # Validate required columns
    required_cols = ['subject_id', 'loop_incidence']
    metric_cols = [
        'clique_size', 'clique_volume', 'clique_avg_degree',
        'clique_boundary_edges', 'clique_boundary_ratio', 'clique_avg_embeddedness'
    ]
    
    missing_cols = [col for col in required_cols + metric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter metrics based on --measures parameter
    if args.measures == 'all':
        metrics_to_analyze = metric_cols
        print(f"\nAnalyzing correlations for all {len(metrics_to_analyze)} metrics:")
    else:
        # Convert to list if single string
        measures_requested = args.measures if isinstance(args.measures, list) else [args.measures]
        
        # Validate requested metrics exist
        invalid_metrics = [m for m in measures_requested if m not in metric_cols]
        if invalid_metrics:
            print(f"\nWarning: Invalid metrics specified: {invalid_metrics}")
            print(f"Available metrics: {metric_cols}")
            raise ValueError(f"Invalid metrics: {invalid_metrics}")
        
        metrics_to_analyze = measures_requested
        print(f"\nAnalyzing correlations for {len(metrics_to_analyze)} selected metric(s):")
    
    for metric in metrics_to_analyze:
        print(f"  - {metric}")
    
    # Calculate subject-wise correlations
    results_df = calculate_subject_correlations(df, metrics_to_analyze)
    
    # Test group-level effects for each metric
    print("\n" + "="*80)
    print("Testing group-level effects...")
    print("="*80)
    
    test_results = []
    for metric in metric_cols:
        result = test_group_effect(results_df, metric)
        test_results.append(result)
    
    # Print statistical report
    print_statistical_report(test_results)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)

    # Get colors from plasma colormap
    plasma = cm.get_cmap('plasma')
    colors = [plasma(0.0), plasma(0.5), plasma(1.0)]  # Sample three colors from plasma
    
    plot_correlation_distributions(results_df, metrics_to_analyze, output_path, args.show, colors=colors)
    plot_effect_size_summary(test_results, output_path, args.show, colors=colors)
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    save_results_csv(results_df, test_results, output_path)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
