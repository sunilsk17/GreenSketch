"""
Visualization for real-world dataset results.

Generates plots comparing synthetic vs real-world performance.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set publication-quality style
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 12


def load_csv(filepath):
    """Load CSV results."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            results.append(row)
    return results


def plot_real_world_comparison(results, output_path='results/plots/real_world_comparison.png'):
    """
    Plot real-world dataset comparison.
    
    Shows energy-accuracy for Enron vs Wikipedia.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter by dataset
    enron_results = [r for r in results if r['dataset'] == 'enron']
    wiki_results = [r for r in results if r['dataset'] == 'wikipedia']
    
    # Group by variant
    variants = ['std_hll', 'lhll', 'thll', 'thll_adaptive', 'aphll', 'aphll_adaptive']
    variant_labels = {
        'std_hll': 'Std HLL',
        'lhll': 'LHLL',
        'thll': 'THLL',
        'thll_adaptive': 'THLL+Ctrl',
        'aphll': 'AP-HLL',
        'aphll_adaptive': 'AP-HLL+Ctrl'
    }
    colors = {
        'std_hll': '#2E86AB',
        'lhll': '#A23B72',
        'thll': '#F18F01',
        'thll_adaptive': '#C73E1D',
        'aphll': '#06A77D',
        'aphll_adaptive': '#D4A574'
    }
    
    # Plot Enron
    for variant in variants:
        variant_data = [r for r in enron_results if r['variant'] == variant]
        if not variant_data:
            continue
        
        errors = [r['relative_error_pct'] for r in variant_data]
        energies = [r['energy_proxy'] for r in variant_data]
        
        ax1.scatter(errors, energies, color=colors[variant], s=100,
                   alpha=0.7, label=variant_labels[variant], edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Relative Error (%)', fontweight='bold')
    ax1.set_ylabel('Energy Proxy', fontweight='bold')
    ax1.set_title('Enron Email Corpus (High Skew)', fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot Wikipedia (filter out 100% errors for visibility)
    for variant in variants:
        variant_data = [r for r in wiki_results if r['variant'] == variant and r['relative_error_pct'] < 50]
        if not variant_data:
            continue
        
        errors = [r['relative_error_pct'] for r in variant_data]
        energies = [r['energy_proxy'] for r in variant_data]
        
        if errors:  # Only plot if we have data
            ax2.scatter(errors, energies, color=colors[variant], s=100,
                       alpha=0.7, label=variant_labels[variant], edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Relative Error (%)', fontweight='bold')
    ax2.set_ylabel('Energy Proxy', fontweight='bold')
    ax2.set_title('Wikipedia Pageviews (Low Skew)', fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_dataset_characteristics(results, output_path='results/plots/dataset_characteristics.png'):
    """
    Plot dataset characteristics comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract unique stream info
    stream_info = {}
    for r in results:
        key = r['stream_name']
        if key not in stream_info:
            stream_info[key] = {
                'ground_truth': r['ground_truth'],
                'stream_size': r['stream_size'],
                'dup_ratio': 1.0 - (r['ground_truth'] / r['stream_size'])
            }
    
    streams = sorted(stream_info.keys())
    unique_counts = [stream_info[s]['ground_truth'] for s in streams]
    dup_ratios = [stream_info[s]['dup_ratio'] * 100 for s in streams]
    
    x = np.arange(len(streams))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, unique_counts, width, label='Unique Items',
                   color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, dup_ratios, width, label='Duplicate Ratio (%)',
                    color='#F18F01', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dataset Stream', fontweight='bold')
    ax.set_ylabel('Unique Items', fontweight='bold', color='#2E86AB')
    ax2.set_ylabel('Duplicate Ratio (%)', fontweight='bold', color='#F18F01')
    ax.set_title('Real-World Dataset Characteristics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in streams], rotation=0, ha='center')
    
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
             frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_real_world_results(results_dir='results'):
    """Generate all real-world visualizations."""
    print("="*80)
    print("GENERATING REAL-WORLD VISUALIZATIONS")
    print("="*80)
    
    # Load real-world results
    real_world_path = os.path.join(results_dir, 'real_world_benchmark_results.csv')
    if os.path.exists(real_world_path):
        print("\nGenerating real-world plots...")
        results = load_csv(real_world_path)
        plot_real_world_comparison(results)
        plot_dataset_characteristics(results)
    else:
        print(f"❌ Real-world results not found: {real_world_path}")
    
    print("\n" + "="*80)
    print("✓ Real-world visualizations generated!")
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GreenSketch Real-World Visualization')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory containing CSV files')
    
    args = parser.parse_args()
    
    visualize_real_world_results(args.results_dir)


if __name__ == "__main__":
    main()
