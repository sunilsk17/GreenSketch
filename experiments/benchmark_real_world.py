"""
Real-world dataset benchmark for GreenSketch.

Validates energy-aware HLL variants on real traces:
- Enron email corpus (high skew, 97% duplicates)
- Wikipedia pageviews (low skew, 9% duplicates)

Runs same evaluation as synthetic benchmarks for comparison.
"""

import sys
import os
import numpy as np
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.hll import HyperLogLog
from core.lhll import LazyHashingHLL
from core.thll import ThresholdedHLL
from core.aphll import AdaptivePrecisionHLL
from streams.real_traces import EnronEmailTrace, WikipediaPageviewTrace, get_all_real_traces
from energy.metrics import EnergyMetrics, compute_energy_proxy
from utils.hash_funcs import reset_hash_counter, get_hash_counter


def run_real_trace_experiment(hll_class, stream, ground_truth, stream_name, p=10, **hll_kwargs):
    """
    Run single experiment on real trace.
    
    Args:
        hll_class: HLL variant class
        stream: List of items
        ground_truth: True cardinality
        stream_name: Name for logging
        p: Precision parameter
        **hll_kwargs: Additional HLL arguments
    
    Returns:
        Results dictionary
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
    
    # Add hash stats
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
        'stream_name': stream_name,
        'variant': hll_class.__name__,  # Will be overridden by caller
        'estimate': estimate,
        'ground_truth': ground_truth,
        'relative_error_pct': relative_error_pct,
        'energy_proxy': energy_proxy,
        **measurements
    }
    
    return results


def run_real_world_benchmark(max_items=None, p=10, output_dir='results'):
    """
    Run benchmark on real-world datasets.
    
    Args:
        max_items: Maximum items to load from each dataset (None = all)
        p: Precision parameter
        output_dir: Output directory
    """
    print("="*80)
    print("GREENSKETCH REAL-WORLD DATASET BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Max items per dataset: {max_items if max_items else 'ALL (100K)'}")
    print(f"  Precision p: {p}")
    print(f"  Registers: {1 << p}")
    
    # Define variants (focus on key ones for real data)
    variants = {
        'std_hll': (HyperLogLog, {}),
        'lhll': (LazyHashingHLL, {}),
        'thll': (ThresholdedHLL, {'delta': 1, 'adaptive_delta': False}),
        'thll_adaptive': (ThresholdedHLL, {'delta': 1, 'adaptive_delta': True}),
        'aphll': (AdaptivePrecisionHLL, {'rho_cap': 48, 'adaptive': False}),
        'aphll_adaptive': (AdaptivePrecisionHLL, {'rho_cap': 48, 'adaptive': True}),
    }
    
    # Results storage
    all_results = []
    
    print(f"\n{'='*80}")
    print("RUNNING REAL-WORLD EXPERIMENTS")
    print("="*80)
    
    # Test Enron traces
    print(f"\n{'='*80}")
    print("ENRON EMAIL CORPUS")
    print("="*80)
    print("Characteristics: High skew (97% duplicates), 2,995 unique senders")
    print("Use case: Edge gateway communication monitoring")
    
    for trace_type in ['chrono', 'random', 'grouped']:
        stream_name = f'enron_{trace_type}'
        print(f"\n--- {stream_name.upper().replace('_', ' ')} ---")
        
        # Load trace
        loader = EnronEmailTrace(trace_type=trace_type, max_items=max_items)
        stream = loader.load()
        ground_truth = len(set(stream))
        stats = loader.get_stats()
        
        print(f"  Loaded: {len(stream):,} emails")
        print(f"  Unique senders: {ground_truth:,}")
        print(f"  Duplicate ratio: {stats['loaded_dup_ratio']:.1%}")
        
        for variant_name, (VariantClass, kwargs) in variants.items():
            print(f"    Running {variant_name.upper()}...", end=' ')
            
            try:
                # Pass only valid kwargs to constructor
                results = run_real_trace_experiment(
                    VariantClass, stream, ground_truth, stream_name, p=p, **kwargs
                )
                
                # Add variant name to results after creation
                results['variant'] = variant_name
                results['dataset'] = 'enron'
                results['trace_type'] = trace_type
                results['stream_size'] = len(stream)
                results['p'] = p
                
                all_results.append(results)
                
                print(f"✓ Error: {results['relative_error_pct']:.2f}%, "
                      f"Energy: {results['energy_proxy']:.0f}")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
    
    # Test Wikipedia traces
    print(f"\n{'='*80}")
    print("WIKIPEDIA PAGEVIEWS")
    print("="*80)
    print("Characteristics: Low skew (9% duplicates), 90K unique pages")
    print("Use case: Regional data center content analytics")
    
    for trace_type in ['chrono', 'random', 'grouped']:
        stream_name = f'wiki_{trace_type}'
        print(f"\n--- {stream_name.upper().replace('_', ' ')} ---")
        
        # Load trace
        loader = WikipediaPageviewTrace(trace_type=trace_type, max_items=max_items)
        stream = loader.load()
        ground_truth = len(set(stream))
        stats = loader.get_stats()
        
        print(f"  Loaded: {len(stream):,} pageviews")
        print(f"  Unique pages: {ground_truth:,}")
        print(f"  Duplicate ratio: {stats['loaded_dup_ratio']:.1%}")
        
        for variant_name, (VariantClass, kwargs) in variants.items():
            print(f"    Running {variant_name.upper()}...", end=' ')
            
            try:
                # Pass only valid kwargs to constructor
                results = run_real_trace_experiment(
                    VariantClass, stream, ground_truth, stream_name, p=p, **kwargs
                )
                
                # Add variant name to results after creation
                results['variant'] = variant_name
                results['dataset'] = 'wikipedia'
                results['trace_type'] = trace_type
                results['stream_size'] = len(stream)
                results['p'] = p
                
                all_results.append(results)
                
                print(f"✓ Error: {results['relative_error_pct']:.2f}%, "
                      f"Energy: {results['energy_proxy']:.0f}")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
    
    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'real_world_benchmark_results.csv')
    
    if all_results:
        fieldnames = list(all_results[0].keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Results saved to {csv_path}")
    
    # Print summary
    print_real_world_summary(all_results)
    
    return all_results


def print_real_world_summary(results):
    """Print summary table for real-world results."""
    print("\n" + "="*80)
    print("REAL-WORLD BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by dataset and variant
    from collections import defaultdict
    by_dataset_variant = defaultdict(list)
    
    for result in results:
        key = (result['dataset'], result['variant'])
        by_dataset_variant[key].append(result)
    
    # Print Enron results
    print("\n--- ENRON EMAIL CORPUS ---")
    print(f"{'Variant':<20} {'Avg Error':<12} {'Avg Energy':<15} {'Avg CPU(s/M)':<15}")
    print("-"*80)
    
    baseline_enron = None
    for variant in ['std_hll', 'lhll', 'thll', 'thll_adaptive', 'aphll', 'aphll_adaptive']:
        key = ('enron', variant)
        if key not in by_dataset_variant:
            continue
        
        variant_results = by_dataset_variant[key]
        avg_error = np.mean([r['relative_error_pct'] for r in variant_results])
        avg_energy = np.mean([r['energy_proxy'] for r in variant_results])
        avg_cpu = np.mean([r['cpu_time_per_M'] for r in variant_results])
        
        if variant == 'std_hll':
            baseline_enron = avg_energy
        
        energy_str = f"{avg_energy:.0f}"
        if baseline_enron and baseline_enron > 0:
            energy_rel = avg_energy / baseline_enron
            energy_str += f" ({energy_rel:.2%})"
        
        print(f"{variant.upper():<20} {avg_error:<12.2f} {energy_str:<15} {avg_cpu:<15.3f}")
    
    # Print Wikipedia results
    print("\n--- WIKIPEDIA PAGEVIEWS ---")
    print(f"{'Variant':<20} {'Avg Error':<12} {'Avg Energy':<15} {'Avg CPU(s/M)':<15}")
    print("-"*80)
    
    baseline_wiki = None
    for variant in ['std_hll', 'lhll', 'thll', 'thll_adaptive', 'aphll', 'aphll_adaptive']:
        key = ('wikipedia', variant)
        if key not in by_dataset_variant:
            continue
        
        variant_results = by_dataset_variant[key]
        avg_error = np.mean([r['relative_error_pct'] for r in variant_results])
        avg_energy = np.mean([r['energy_proxy'] for r in variant_results])
        avg_cpu = np.mean([r['cpu_time_per_M'] for r in variant_results])
        
        if variant == 'std_hll':
            baseline_wiki = avg_energy
        
        energy_str = f"{avg_energy:.0f}"
        if baseline_wiki and baseline_wiki > 0:
            energy_rel = avg_energy / baseline_wiki
            energy_str += f" ({energy_rel:.2%})"
        
        print(f"{variant.upper():<20} {avg_error:<12.2f} {energy_str:<15} {avg_cpu:<15.3f}")
    
    # Controller impact analysis
    print("\n--- CONTROLLER IMPACT ---")
    print(f"{'Variant':<15} {'Dataset':<12} {'No Ctrl':<12} {'With Ctrl':<12} {'Improvement':<12}")
    print("-"*80)
    
    for variant_base in ['thll', 'aphll']:
        for dataset in ['enron', 'wikipedia']:
            key_no = (dataset, variant_base)
            key_yes = (dataset, f'{variant_base}_adaptive')
            
            if key_no in by_dataset_variant and key_yes in by_dataset_variant:
                energy_no = np.mean([r['energy_proxy'] for r in by_dataset_variant[key_no]])
                energy_yes = np.mean([r['energy_proxy'] for r in by_dataset_variant[key_yes]])
                
                improvement = (energy_no - energy_yes) / energy_no * 100 if energy_no > 0 else 0
                
                print(f"{variant_base.upper():<15} {dataset:<12} {energy_no:<12.0f} "
                      f"{energy_yes:<12.0f} {improvement:+.1f}%")
    
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GreenSketch Real-World Benchmark')
    parser.add_argument('--max-items', type=int, default=None,
                       help='Max items to load (default: all 100K)')
    parser.add_argument('--precision', '-p', type=int, default=10,
                       help='HLL precision parameter (default: 10)')
    parser.add_argument('--output', '-o', default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    results = run_real_world_benchmark(
        max_items=args.max_items,
        p=args.precision,
        output_dir=args.output
    )
    
    print(f"\n✓ Real-world benchmark complete! {len(results)} experiments run.")


if __name__ == "__main__":
    main()
