"""
Mixed Effects Linear Model for Clique-Loop Incidence Analysis

This script fits a mixed effects linear model to predict loop incidence from clique metrics:
- Model: loop_incidence ~ clique_metrics + (1 | subject_id)
- Random intercepts for subjects to account for within-subject correlation
- Standardized predictors for interpretability
- Comprehensive model diagnostics and visualizations

Usage: python clique_measures_mixed_model.py -m clique_measures_with_loop_incidence.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import List
import argparse
import os
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper

from utils import load_dataframe

def fit_mixed_effects_model(df: pd.DataFrame, metrics: List[str]) -> MixedLMResultsWrapper:
    """Fit mixed effects linear model: loop_incidence ~ metrics with random intercepts by subject.
    
    Args:
        df: DataFrame with columns [subject_id, loop_incidence, <metrics>]
        metrics: List of metric column names to use as predictors
        
    Returns:
        Fitted mixed linear model results
    """
    print("\nFitting mixed effects linear model...")
    print(f"  Model: loop_incidence ~ {' + '.join(metrics)} + (1 | subject_id)")
    
    # Prepare data: remove rows with missing values
    model_data = df[['subject_id', 'loop_incidence'] + metrics].dropna()
    
    print(f"  Sample size: {len(model_data)} observations from {model_data['subject_id'].nunique()} subjects")
    
    # Standardize predictors for interpretability and numerical stability
    print("  Standardizing predictors (z-scoring)...")
    model_data_std = model_data.copy()
    for metric in metrics:
        model_data_std[metric] = (model_data[metric] - model_data[metric].mean()) / model_data[metric].std()
    
    # Build formula
    formula = f"loop_incidence ~ {' + '.join(metrics)}"
    
    # Fit mixed effects model with random intercepts for subjects
    try:
        model = smf.mixedlm(formula, model_data_std, groups=model_data_std["subject_id"])
        result = model.fit(method='lbfgs')
        print("  Model fit successful!")
        return result
    except Exception as e:
        print(f"  Error fitting model: {e}")
        raise


def plot_mixed_model_results(result: MixedLMResultsWrapper, output_dir: Path, 
                             show_plot: bool = False) -> None:
    """Create visualizations of mixed effects model results.
    
    Args:
        result: Fitted mixed linear model results
        output_dir: Directory to save plots
        show_plot: Whether to display plots interactively
    """
    print("\nGenerating mixed effects model visualizations...")
    
    # Extract data from model
    model_data = result.model.data.frame
    fe_params = result.fe_params
    fe_pvalues = result.pvalues
    
    # Get metric names (exclude intercept)
    metrics = [p for p in fe_params.index if p != 'Intercept']
    
    # Define color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(metrics))) # type: ignore
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        # Get data for this metric
        x_data = model_data['loop_incidence'].values
        y_data = model_data[metric].values
        
        # Sample 10% of points randomly for visualization
        n_points = len(x_data)
        sample_size = max(int(n_points * 0.1), 10)  # At least 10 points
        sample_indices = np.random.choice(n_points, size=sample_size, replace=False)
        x_sample = x_data[sample_indices]
        y_sample = y_data[sample_indices]
        
        # Plot point cloud with low alpha (sampled points)
        ax.scatter(x_sample, y_sample, alpha=0.1, s=10, color=colors[i], label=None)
        
        # Calculate bivariate linear regression (metric ~ loop_incidence)
        # Each metric gets its own intercept based on the bivariate relationship
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        
        # Generate fit line
        x_fit = np.array([x_data.min(), x_data.max()])
        y_fit = slope * x_fit + intercept
        
        # Determine line style based on significance in mixed model
        linestyle = '-' if fe_pvalues[metric] < 0.05 else '--'
        linewidth = 2.5 if fe_pvalues[metric] < 0.05 else 1.5
        
        # Plot fit line with metric name
        metric_label = metric.replace('clique_', '').replace('_', ' ').title()
        p_str = f"p={fe_pvalues[metric]:.3f}" if fe_pvalues[metric] >= 0.001 else "p<0.001"
        label = f"{metric_label} ({p_str})"
        
        ax.plot(x_fit, y_fit, color=colors[i], linestyle=linestyle, 
               linewidth=linewidth, label=label, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('Loop Incidence', fontsize=13, fontweight='bold')
    ax.set_ylabel('Clique Metric Value (Standardized)', fontsize=13, fontweight='bold')
    ax.set_title('Mixed Effects Model: Linear Relationships with Loop Incidence', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9, 
             title='Clique Metrics', title_fontsize=11)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # Add note about line styles
    note_text = "Solid lines: p < 0.05 | Dashed lines: p ≥ 0.05"
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    model_plot_path = output_dir / 'mixed_effects_linear_fits.png'
    plt.savefig(model_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved mixed effects linear fits plot to {model_plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_mixed_model_report(result: MixedLMResultsWrapper) -> None:
    """Print formatted report of mixed effects model results.
    
    Args:
        result: Fitted mixed linear model results
    """
    print("\n" + "="*80)
    print("MIXED EFFECTS LINEAR MODEL RESULTS")
    print("Model: loop_incidence ~ clique_measures + (1 | subject_id)")
    print("="*80)
    
    print("\nModel Summary:")
    print(result.summary())
    
    print("\n" + "="*80)
    print("FIXED EFFECTS (Standardized Predictors):")
    print("="*80)
    
    # Extract fixed effects
    fe_params = result.fe_params
    fe_bse = result.bse
    fe_pvalues = result.pvalues
    fe_conf_int = result.conf_int()
    
    for param in fe_params.index:
        coef = fe_params[param]
        se = fe_bse[param]
        pval = fe_pvalues[param]
        ci_lower = fe_conf_int.loc[param, 0]
        ci_upper = fe_conf_int.loc[param, 1]
        
        # Significance stars
        if pval < 0.001:
            sig_str = "***"
        elif pval < 0.01:
            sig_str = "**"
        elif pval < 0.05:
            sig_str = "*"
        else:
            sig_str = ""
        
        print(f"\n{param}:")
        print(f"  β = {coef:.4f} (SE = {se:.4f})")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  p = {pval:.6f} {sig_str}")
    
    print("\n" + "="*80)
    print("RANDOM EFFECTS:")
    print("="*80)
    print(f"Subject (Intercept) SD: {result.cov_re.iloc[0, 0]**0.5:.4f}")
    print(f"Residual SD: {result.scale**0.5:.4f}")
    
    print("\n" + "="*80)
    print("MODEL FIT:")
    print("="*80)
    print(f"Log-Likelihood: {result.llf:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    
    # Intraclass correlation (ICC)
    var_random = result.cov_re.iloc[0, 0]
    var_residual = result.scale
    icc = var_random / (var_random + var_residual)
    print(f"\nIntraclass Correlation (ICC): {icc:.4f}")
    print(f"  (Proportion of variance explained by subject grouping)")
    
    print("\n" + "="*80)


def save_mixed_model_results(result: MixedLMResultsWrapper, output_dir: Path) -> None:
    """Save mixed effects model results to CSV files.
    
    Args:
        result: Fitted mixed effects model results
        output_dir: Directory to save results
    """
    print("\nSaving mixed effects model results...")
    
    # Extract fixed effects
    fe_params = result.fe_params
    fe_conf_int = result.conf_int()
    
    fe_summary = pd.DataFrame({
        'parameter': list(fe_params.index),
        'coefficient': list(fe_params.values),
        'std_error': list(result.bse[fe_params.index].values),
        'z_value': list(result.tvalues[fe_params.index].values),
        'p_value': list(result.pvalues[fe_params.index].values),
        'ci_lower': list(fe_conf_int.loc[fe_params.index, 0].values),
        'ci_upper': list(fe_conf_int.loc[fe_params.index, 1].values)
    })
    
    mixed_model_path = output_dir / 'mixed_model_fixed_effects.csv'
    fe_summary.to_csv(mixed_model_path, index=False)
    print(f"  Saved fixed effects to {mixed_model_path}")
    
    # Save model fit statistics
    var_random = result.cov_re.iloc[0, 0]
    var_residual = result.scale
    icc = var_random / (var_random + var_residual)
    
    fit_stats = pd.DataFrame({
        'statistic': ['log_likelihood', 'AIC', 'BIC', 'random_intercept_variance', 
                     'residual_variance', 'random_intercept_sd', 'residual_sd', 'ICC'],
        'value': [result.llf, result.aic, result.bic,
                 var_random, var_residual, np.sqrt(var_random), np.sqrt(var_residual), icc]
    })
    
    fit_stats_path = output_dir / 'mixed_model_fit_statistics.csv'
    fit_stats.to_csv(fit_stats_path, index=False)
    print(f"  Saved fit statistics to {fit_stats_path}")
    
    # Save full model summary as text
    summary_path = output_dir / 'mixed_model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    print(f"  Saved model summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Mixed Effects Linear Model for Clique-Loop Incidence Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Specification:
  loop_incidence ~ clique_metrics + (1 | subject_id)
  
  - Fixed effects: clique metrics (standardized)
  - Random effects: subject-specific intercepts
  - Accounts for within-subject correlation

Required columns:
  - subject_id: Subject identifier
  - loop_incidence: Number of loops touching the clique
  - clique_size: Number of nodes in clique
  - clique_volume: Number of edges in clique
  - clique_avg_degree: Average degree of nodes in clique
  - clique_boundary_edges: Number of edges connecting clique to rest of network
  - clique_boundary_ratio: Ratio of boundary edges to internal edges
  - clique_avg_embeddedness: Average embeddedness of nodes in clique

Examples:
  # Basic analysis with all metrics
  python clique_measures_mixed_model.py -m clique_measures_with_loop_incidence.csv
  
  # With specific metrics and custom output
  python clique_measures_mixed_model.py -m data.csv -o ./results --measures clique_size clique_volume --show
        """
    )
    
    parser.add_argument('-m', '--measures_data', type=str, required=True,
                        help='Path to file with clique measures and loop incidence (CSV or Parquet)')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory for results (default: auto-generated in output/clique_measures_group_analysis)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')
    parser.add_argument('--measures', type=str, nargs='+', default='all',
                        help='List of clique metrics to include in model (default: all). ' \
                        'Available: clique_size, clique_volume, clique_avg_degree, ' \
                        'clique_boundary_edges, clique_boundary_ratio, clique_avg_embeddedness')
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
            f'mixed_model_{current_time}'
        )
    
    print("="*80)
    print("Mixed Effects Linear Model: Clique-Loop Incidence Analysis")
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
    
    # Select metrics for model
    if args.measures == 'all':
        metrics_to_use = metric_cols
        print(f"\nUsing all {len(metrics_to_use)} metrics as predictors:")
    else:
        measures_requested = args.measures if isinstance(args.measures, list) else [args.measures]
        invalid_metrics = [m for m in measures_requested if m not in metric_cols]
        if invalid_metrics:
            print(f"\nError: Invalid metrics specified: {invalid_metrics}")
            print(f"Available metrics: {metric_cols}")
            raise ValueError(f"Invalid metrics: {invalid_metrics}")
        
        metrics_to_use = measures_requested
        print(f"\nUsing {len(metrics_to_use)} selected metric(s) as predictors:")
    
    for metric in metrics_to_use:
        print(f"  - {metric}")
    
    # Fit mixed effects model
    print("\n" + "="*80)
    print("Fitting Mixed Effects Linear Model...")
    print("="*80)
    
    try:
        model_result = fit_mixed_effects_model(df, metrics_to_use)
        print_mixed_model_report(model_result)
    except Exception as e:
        print(f"\nError: Could not fit mixed effects model: {e}")
        raise
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    plot_mixed_model_results(model_result, output_path, show_plot=args.show)
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    save_mixed_model_results(model_result, output_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
