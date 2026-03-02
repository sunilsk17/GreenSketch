"""
Main benchmark evaluation for GreenSketch.

Evaluates all HLL variants across all stream types and computes:
- Cardinality accuracy (error, bias, variance)
- Energy metrics (CPU time, hash calls, updates)
- Energy-accuracy tradeoffs
"""

import sys
import os
import numpy as np
import time
import csv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.hll import HyperLogLog
from core.lhll import LazyHashingHLL
from core.thll import ThresholdedHLL
from core.aphll import AdaptivePrecisionHLL
from streams.generators import get_all_generators
from energy.metrics import EnergyMetrics, compute_energy_proxy
from utils.hash_funcs import reset_hash_counter, get_hash_counter


def run_single_experiment(hll_class, stream, ground_truth, p=10, **hll_kwargs):
    """
    Run a single experiment: one variant on one stream.
    
    Args:
        hll_class: HLL variant class
        stream: List of items
        ground_truth: True cardinality
        p: Precision parameter
        **hll_kwargs: Additional arguments for HLL constructor
    
    Returns:
        Dictionary with results
    """
    # Reset hash counter
    reset_hash_counter()
    
    # Create HLL instance
    hll = hll_class(p=p, **hll_kwargs)
    
    # Start energy measurement
    energy_metrics = EnergyMetrics()
    energy_metrics.start()
    
    # Process stream
    start_time = time.time()
    for item in stream:
        hll.add(item)
    end_time = time.time()
    
    # Stop energy measurement
    energy_metrics.stop()
    
    # Get estimate
    estimate = hll.estimate()
    
    # Get metrics
    hll_metrics = hll.get_metrics()
    measurements = energy_metrics.get_measurements(hll_metrics)
    hash_stats = get_hash_counter().get_stats()
    
    # Add hash stats to measurements
    measurements['cheap_hash_calls'] = hash_stats['cheap_hash_calls']
    measurements['full_hash_calls'] = hash_stats['full_hash_calls']
    measurements['cheap_calls_per_M'] = (hash_stats['cheap_hash_calls'] / len(stream) * 1e6) if len(stream) > 0 else 0
    measurements['full_calls_per_M'] = (hash_stats['full_hash_calls'] / len(stream) * 1e6) if len(stream) > 0 else 0
    
    # Compute accuracy metrics
    error = abs(estimate - ground_truth) / ground_truth if ground_truth > 0 else 0
    relative_error_pct = error * 100
    
    # Compute energy proxy
    energy_proxy = compute_energy_proxy(measurements)
    
    results = {
        'estimate': estimate,
        'ground_truth': ground_truth,
        'relative_error_pct': relative_error_pct,
        'energy_proxy': energy_proxy,
        **measurements
    }
    
    return results


def run_benchmark(n_unique=10000, n_total=50000, p=10, output_dir='results'):
    """
    Run full benchmark across all variants and streams.
    
    Args:
        n_unique: Number of unique items
        n_total: Total items (including duplicates)
        p: Precision parameter
        output_dir: Output directory for results
    """
    print("="*80)
    print("GREENSKETCH BENCHMARK EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Unique items: {n_unique}")
    print(f"  Total items: {n_total}")
    print(f"  Precision p: {p}")
    print(f"  Registers: {1 << p}")
    
    # Define variants
    variants = {
        'std_hll': (HyperLogLog, {}),
        'lhll': (LazyHashingHLL, {}),
        'thll': (ThresholdedHLL, {'delta': 1, 'adaptive_delta': True}),
        'aphll': (AdaptivePrecisionHLL, {'rho_cap': 48, 'adaptive': True}),
    }
    
    # Define streams
    stream_generators = get_all_generators()
    
    # Select key streams for benchmark
    selected_streams = ['random', 'sorted_hash', 'adversarial', 'zipfian',
                       'iot_devices', 'network_flows', 'traffic_sensors']
    
    # Results storage
    all_results = []
    
    # Run experiments
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    for stream_name in selected_streams:
        if stream_name not in stream_generators:
            continue
        
        print(f"\n--- Stream: {stream_name.upper().replace('_', ' ')} ---")
        
        # Generate stream
        GeneratorClass = stream_generators[stream_name]
        generator = GeneratorClass(n_unique=n_unique, n_total=n_total, seed=42)
        stream = generator.generate()
        ground_truth = len(set(stream))
        
        print(f"  Generated {len(stream)} items ({ground_truth} unique)")
        
        for variant_name, (VariantClass, kwargs) in variants.items():
            print(f"    Running {variant_name.upper()}...", end=' ')
            
            try:
                results = run_single_experiment(
                    VariantClass, stream, ground_truth, p=p, **kwargs
                )
                
                results['variant'] = variant_name
                results['stream'] = stream_name
                results['n_unique'] = n_unique
                results['n_total'] = n_total
                results['p'] = p
                
                all_results.append(results)
                
                print(f"✓ Error: {results['relative_error_pct']:.2f}%, "
                      f"Energy: {results['energy_proxy']:.0f}")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
    
    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    
    if all_results:
        fieldnames = list(all_results[0].keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Results saved to {csv_path}")
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def print_summary(results):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Group by variant
    from collections import defaultdict
    by_variant = defaultdict(list)
    
    for result in results:
        by_variant[result['variant']].append(result)
    
    # Print table
    print(f"\n{'Variant':<12} {'Avg Error':<12} {'Avg CPU(s/M)':<15} "
          f"{'Avg Energy':<12} {'Hash Savings':<12}")
    print("-"*80)
    
    baseline_energy = None
    baseline_hash = None
    
    for variant_name in ['std_hll', 'lhll', 'thll', 'aphll']:
        if variant_name not in by_variant:
            continue
        
        variant_results = by_variant[variant_name]
        
        avg_error = np.mean([r['relative_error_pct'] for r in variant_results])
        avg_cpu = np.mean([r['cpu_time_per_M'] for r in variant_results])
        avg_energy = np.mean([r['energy_proxy'] for r in variant_results])
        avg_full_hash = np.mean([r['full_calls_per_M'] for r in variant_results])
        
        if variant_name == 'std_hll':
            baseline_energy = avg_energy
            baseline_hash = avg_full_hash
        
        hash_savings = ""
        if baseline_hash and baseline_hash > 0:
            savings_pct = (1 - avg_full_hash / baseline_hash) * 100
            hash_savings = f"{savings_pct:+.1f}%"
        
        energy_str = f"{avg_energy:.0f}"
        if baseline_energy and baseline_energy > 0:
            energy_rel = avg_energy / baseline_energy
            energy_str += f" ({energy_rel:.2%})"
        
        print(f"{variant_name.upper():<12} {avg_error:<12.2f} {avg_cpu:<15.4f} "
              f"{energy_str:<12} {hash_savings:<12}")
    
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GreenSketch Benchmark Evaluation')
    parser.add_argument('--n-unique', type=int, default=10000,
                       help='Number of unique items (default: 10000)')
    parser.add_argument('--n-total', type=int, default=50000,
                       help='Total items including duplicates (default: 50000)')
    parser.add_argument('--precision', '-p', type=int, default=10,
                       help='HLL precision parameter (default: 10)')
    parser.add_argument('--output', '-o', default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    results = run_benchmark(
        n_unique=args.n_unique,
        n_total=args.n_total,
        p=args.precision,
        output_dir=args.output
    )
    
    print(f"\n✓ Benchmark complete! {len(results)} experiments run.")


if __name__ == "__main__":
    main()
