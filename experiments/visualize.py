"""
Visualization for GreenSketch results.

Generates publication-quality plots for:
- Energy vs Accuracy scatter
- Error vs Stream Order bars
- Energy breakdown stacked bars
- Sensitivity curves
- Ablation comparisons
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


def plot_energy_vs_accuracy(results, output_path='results/plots/energy_vs_accuracy.png'):
    """
    Plot Energy vs Accuracy scatter.
    
    Shows Pareto frontier of energy-accuracy tradeoff.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Group by variant
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r['variant']].append(r)
    
    # Colors and markers
    colors = {'std_hll': '#2E86AB', 'lhll': '#A23B72', 'thll': '#F18F01', 'aphll': '#C73E1D'}
    markers = {'std_hll': 'o', 'lhll': 's', 'thll': '^', 'aphll': 'd'}
    labels = {'std_hll': 'Std HLL', 'lhll': 'LHLL', 'thll': 'THLL', 'aphll': 'AP-HLL'}
    
    for variant in ['std_hll', 'lhll', 'thll', 'aphll']:
        if variant not in by_variant:
            continue
        
        variant_results = by_variant[variant]
        errors = [r['relative_error_pct'] for r in variant_results]
        energies = [r['energy_proxy'] for r in variant_results]
        
        ax.scatter(errors, energies, 
                  color=colors[variant], marker=markers[variant],
                  s=100, alpha=0.7, label=labels[variant], edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Relative Error (%)', fontweight='bold')
    ax.set_ylabel('Energy Proxy', fontweight='bold')
    ax.set_title('Energy-Accuracy Tradeoff', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_error_vs_stream(results, output_path='results/plots/error_vs_stream.png'):
    """
    Plot Error vs Stream Order (bar chart).
    
    Shows robustness to different stream orderings.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique streams and variants
    streams = sorted(list(set(r['stream'] for r in results)))
    variants = ['std_hll', 'lhll', 'thll', 'aphll']
    variant_labels = {'std_hll': 'Std HLL', 'lhll': 'LHLL', 'thll': 'THLL', 'aphll': 'AP-HLL'}
    
    # Organize data
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r['stream']][r['variant']].append(r['relative_error_pct'])
    
    # Average errors
    avg_errors = {}
    for stream in streams:
        avg_errors[stream] = {}
        for variant in variants:
            if variant in data[stream]:
                avg_errors[stream][variant] = np.mean(data[stream][variant])
            else:
                avg_errors[stream][variant] = 0
    
    # Plot
    x = np.arange(len(streams))
    width = 0.2
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, variant in enumerate(variants):
        errors = [avg_errors[stream][variant] for stream in streams]
        offset = (i - 1.5) * width
        ax.bar(x + offset, errors, width, label=variant_labels[variant],
               color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Stream Type', fontweight= 'bold')
    ax.set_ylabel('Relative Error (%)', fontweight='bold')
    ax.set_title('Accuracy Across Stream Orderings', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in streams], rotation=0, ha='center')
    ax.legend(frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_energy_breakdown(results, output_path='results/plots/energy_breakdown.png'):
    """
    Plot stacked bar chart of energy components.
    
    Shows contribution of hash calls, memory updates, and CPU time.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Average by variant
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r['variant']].append(r)
    
    variants = ['std_hll', 'lhll', 'thll', 'aphll']
    variant_labels = {'std_hll': 'Std HLL', 'lhll': 'LHLL', 'thll': 'THLL', 'aphll': 'AP-HLL'}
    
    # Compute averages for each component (per million)
    full_hash_avg = []
    cheap_hash_avg = []
    updates_avg = []
    
    for variant in variants:
        if variant not in by_variant:
            full_hash_avg.append(0)
            cheap_hash_avg.append(0)
            updates_avg.append(0)
            continue
        
        variant_results = by_variant[variant]
        full_hash_avg.append(np.mean([r.get('full_calls_per_M', 0) for r in variant_results]) / 1000)  # /1000 for scale
        cheap_hash_avg.append(np.mean([r.get('cheap_calls_per_M', 0) for r in variant_results]) / 1000)
        updates_avg.append(np.mean([r.get('register_updates_per_M', 0) for r in variant_results]) / 1000)
    
    x = np.arange(len(variants))
    width = 0.5
    
    p1 = ax.bar(x, full_hash_avg, width, label='Full Hash Calls (K)',
                color='#E63946', edgecolor='black', linewidth=0.5)
    p2 = ax.bar(x, cheap_hash_avg, width, bottom=full_hash_avg,
                label='Cheap Hash Calls (K)', color='#F1FAEE', edgecolor='black', linewidth=0.5)
    p3 = ax.bar(x, updates_avg, width,
                bottom=np.array(full_hash_avg) + np.array(cheap_hash_avg),
                label='Register Updates (K)', color='#457B9D', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Operations per Million Items (K)', fontweight='bold')
    ax.set_xlabel('HLL Variant', fontweight='bold')
    ax.set_title('Energy Operation Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([variant_labels[v] for v in variants])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_sensitivity_precision(results, output_path='results/plots/sensitivity_precision.png'):
    """Plot precision sensitivity (p vs error and energy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Organize by variant and p
    by_variant = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_variant[r['variant']][r['p']].append(r)
    
    colors = {'Std HLL': '#2E86AB', 'THLL': '#F18F01'}
    markers = {'Std HLL': 'o', 'THLL': '^'}
    
    for variant in ['Std HLL', 'THLL']:
        if variant not in by_variant:
            continue
        
        p_values = sorted(by_variant[variant].keys())
        errors = [np.mean([r['relative_error_pct'] for r in by_variant[variant][p]]) for p in p_values]
        energies = [np.mean([r['energy_proxy'] for r in by_variant[variant][p]]) for p in p_values]
        
        ax1.plot(p_values, errors, marker=markers[variant], color=colors[variant],
                label=variant, linewidth=2, markersize=8)
        ax2.plot(p_values, energies, marker=markers[variant], color=colors[variant],
                label=variant, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Precision (p)', fontweight='bold')
    ax1.set_ylabel('Relative Error (%)', fontweight='bold')
    ax1.set_title('Accuracy vs Precision', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2.set_xlabel('Precision (p)', fontweight='bold')
    ax2.set_ylabel('Energy Proxy', fontweight='bold')
    ax2.set_title('Energy vs Precision', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_sensitivity_delta(results, output_path='results/plots/sensitivity_delta.png'):
    """Plot THLL delta sensitivity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    delta_values = sorted(list(set(r['delta'] for r in results)))
    errors = [np.mean([r['relative_error_pct'] for r in results if r['delta'] == d]) for d in delta_values]
    energies = [np.mean([r['energy_proxy'] for r in results if r['delta'] == d]) for d in delta_values]
    updates = [np.mean([r['register_updates_per_M'] for r in results if r['delta'] == d]) for d in delta_values]
    
    ax1.plot(delta_values, errors, marker='o', color='#F18F01', linewidth=2, markersize=10)
    ax1.set_xlabel('Delta Threshold', fontweight='bold')
    ax1.set_ylabel('Relative Error (%)', fontweight='bold')
    ax1.set_title('THLL: Error vs Delta', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(delta_values, energies, marker='s', color='#2E86AB',
                     linewidth=2, markersize=8, label='Energy Proxy')
    line2 = ax2_twin.plot(delta_values, [u/1000 for u in updates], marker='^', color='#C73E1D',
                          linewidth=2, markersize=8, label='Reg Updates (K/M)')
    
    ax2.set_xlabel('Delta Threshold', fontweight='bold')
    ax2.set_ylabel('Energy Proxy', fontweight='bold', color='#2E86AB')
    ax2_twin.set_ylabel('Register Updates (K/M)', fontweight='bold', color='#C73E1D')
    ax2.set_title('THLL: Energy vs Delta', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_all_results(results_dir='results'):
    """Generate all visualizations."""
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Load benchmark results
    benchmark_path = os.path.join(results_dir, 'benchmark_results.csv')
    if os.path.exists(benchmark_path):
        print("\nGenerating benchmark plots...")
        benchmark_results = load_csv(benchmark_path)
        plot_energy_vs_accuracy(benchmark_results)
        plot_error_vs_stream(benchmark_results)
        plot_energy_breakdown(benchmark_results)
    
    # Load sensitivity results
    precision_path = os.path.join(results_dir, 'sensitivity_precision.csv')
    if os.path.exists(precision_path):
        print("\nGenerating precision sensitivity plot...")
        precision_results = load_csv(precision_path)
        plot_sensitivity_precision(precision_results)
    
    delta_path = os.path.join(results_dir, 'sensitivity_delta.csv')
    if os.path.exists(delta_path):
        print("\nGenerating delta sensitivity plot...")
        delta_results = load_csv(delta_path)
        plot_sensitivity_delta(delta_results)
    
    print("\n" + "="*80)
    print("✓ All visualizations generated!")
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GreenSketch Visualization')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory containing CSV files')
    
    args = parser.parse_args()
    
    visualize_all_results(args.results_dir)


if __name__ == "__main__":
    main()
