"""
Ablation study: Evaluate impact of adaptive controller.

Compares each variant with and without the adaptive controller
to demonstrate the controller's contribution to energy savings.
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.hll import HyperLogLog
from core.lhll import LazyHashingHLL
from core.thll import ThresholdedHLL
from core.aphll import AdaptivePrecisionHLL
from streams.generators import RandomOrderStream, AdversarialStream, BurstyZipfianStream
from energy.metrics import EnergyMetrics, compute_energy_proxy
from utils.hash_funcs import reset_hash_counter, get_hash_counter


def run_ablation_experiment(hll_class, stream, ground_truth, p=10, with_controller=True, **hll_kwargs):
    """
    Run ablation experiment for one configuration.
    
    Args:
        hll_class: HLL variant class
        stream: Stream items
        ground_truth: True cardinality
        p: Precision
        with_controller: Whether to use adaptive features
        **hll_kwargs: Additional HLL arguments
    
    Returns:
        Results dictionary
    """
    reset_hash_counter()
    
    # Create HLL with or without adaptive features
    if hll_class == ThresholdedHLL:
        # THLL: toggle adaptive_delta
        hll = hll_class(p=p, delta=hll_kwargs.get('delta', 1),
                       adaptive_delta=with_controller)
    elif hll_class == AdaptivePrecisionHLL:
        # AP-HLL: toggle adaptive
        hll = hll_class(p=p, rho_cap=hll_kwargs.get('rho_cap', 48),
                       adaptive=with_controller)
    else:
        # Standard HLL or LHLL (no controller)
        hll = hll_class(p=p, **hll_kwargs)
    
    # Measure
    energy_metrics = EnergyMetrics()
    energy_metrics.start()
    
    for item in stream:
        hll.add(item)
    
    energy_metrics.stop()
    
    # Results
    estimate = hll.estimate()
    hll_metrics = hll.get_metrics()
    measurements = energy_metrics.get_measurements(hll_metrics)
    hash_stats = get_hash_counter().get_stats()
    
    measurements['cheap_hash_calls'] = hash_stats['cheap_hash_calls']
    measurements['full_hash_calls'] = hash_stats['full_hash_calls']
    measurements['cheap_calls_per_M'] = (hash_stats['cheap_hash_calls'] / len(stream) * 1e6) if len(stream) > 0 else 0
    measurements['full_calls_per_M'] = (hash_stats['full_hash_calls'] / len(stream) * 1e6) if len(stream) > 0 else 0
    
    error = abs(estimate - ground_truth) / ground_truth if ground_truth > 0 else 0
    energy_proxy = compute_energy_proxy(measurements)
    
    return {
        'estimate': estimate,
        'ground_truth': ground_truth,
        'relative_error_pct': error * 100,
        'energy_proxy': energy_proxy,
        **measurements
    }


def run_ablation_study(n_unique=10000, n_total=50000, p=10, output_dir='results'):
    """
    Run ablation study across variants and streams.
    
    Args:
        n_unique: Unique items
        n_total: Total items
        p: Precision
        output_dir: Output directory
    """
    print("="*80)
    print("ABLATION STUDY: Controller Impact")
    print("="*80)
    print(f"\nConfiguration: n_unique={n_unique}, n_total={n_total}, p={p}\n")
    
    # Variants that support controller
    variants = {
        'THLL': (ThresholdedHLL, {'delta': 1}),
        'AP-HLL': (AdaptivePrecisionHLL, {'rho_cap': 48}),
    }
    
    # Streams
    streams = {
        'Random': RandomOrderStream(n_unique, n_total, seed=42),
        'Adversarial': AdversarialStream(n_unique, n_total, seed=42),
        'Bursty': BurstyZipfianStream(n_unique, n_total, alpha=1.5, seed=42),
    }
    
    all_results = []
    
    for stream_name, generator in streams.items():
        print(f"\n--- {stream_name} Stream ---")
        stream = generator.generate()
        ground_truth = len(set(stream))
        print(f"  {len(stream)} items, {ground_truth} unique")
        
        # Baseline (always without controller)
        print(f"    Std HLL... ", end='')
        baseline_result = run_ablation_experiment(
            HyperLogLog, stream, ground_truth, p=p, with_controller=False
        )
        baseline_result['variant'] = 'Std HLL'
        baseline_result['controller'] = 'N/A'
        baseline_result['stream'] = stream_name
        all_results.append(baseline_result)
        print(f"✓ Error: {baseline_result['relative_error_pct']:.2f}%, "
              f"Energy: {baseline_result['energy_proxy']:.0f}")
        
        # Test each variant with/without controller
        for variant_name, (VariantClass, kwargs) in variants.items():
            # Without controller
            print(f"    {variant_name} (no controller)... ", end='')
            result_no_ctrl = run_ablation_experiment(
                VariantClass, stream, ground_truth, p=p,
                with_controller=False, **kwargs
            )
            result_no_ctrl['variant'] = variant_name
            result_no_ctrl['controller'] = False
            result_no_ctrl['stream'] = stream_name
            all_results.append(result_no_ctrl)
            print(f"✓ Error: {result_no_ctrl['relative_error_pct']:.2f}%, "
                  f"Energy: {result_no_ctrl['energy_proxy']:.0f}")
            
            # With controller
            print(f"    {variant_name} (with controller)... ", end='')
            result_with_ctrl = run_ablation_experiment(
                VariantClass, stream, ground_truth, p=p,
                with_controller=True, **kwargs
            )
            result_with_ctrl['variant'] = variant_name
            result_with_ctrl['controller'] = True
            result_with_ctrl['stream'] = stream_name
            all_results.append(result_with_ctrl)
            print(f"✓ Error: {result_with_ctrl['relative_error_pct']:.2f}%, "
                  f"Energy: {result_with_ctrl['energy_proxy']:.0f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    
    if all_results:
        with open(csv_path, 'w', newline='') as f:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Results saved to {csv_path}")
    
    # Print summary table
    print_ablation_summary(all_results)
    
    return all_results


def print_ablation_summary(results):
    """Print ablation summary table."""
    print("\n" + "="*80)
    print("ABLATION SUMMARY: Controller Impact")
    print("="*80)
    
    # Organize by variant and controller
    from collections import defaultdict
    by_variant_ctrl = defaultdict(list)
    
    for r in results:
        key = (r['variant'], r['controller'])
        by_variant_ctrl[key].append(r)
    
    print(f"\n{'Variant':<20} {'Controller':<12} {'Avg Error':<12} "
          f"{'Avg Energy':<15} {'Energy Δ':<12}")
    print("-"*80)
    
    baseline_energy = np.mean([r['energy_proxy'] for r in by_variant_ctrl[('Std HLL', 'N/A')]])
    
    for variant in ['THLL', 'AP-HLL']:
        results_no = by_variant_ctrl.get((variant, False), [])
        results_yes = by_variant_ctrl.get((variant, True), [])
        
        if not results_no or not results_yes:
            continue
        
        # Without controller
        error_no = np.mean([r['relative_error_pct'] for r in results_no])
        energy_no = np.mean([r['energy_proxy'] for r in results_no])
        
        print(f"{variant:<20} {'No':<12} {error_no:<12.2f} "
              f"{energy_no:<15.0f} {energy_no/baseline_energy:<12.2%}")
        
        # With controller
        error_yes = np.mean([r['relative_error_pct'] for r in results_yes])
        energy_yes = np.mean([r['energy_proxy'] for r in results_yes])
        
        delta_pct = ((energy_yes - energy_no) / energy_no * 100) if energy_no > 0 else 0
        
        print(f"{variant:<20} {'Yes':<12} {error_yes:<12.2f} "
              f"{energy_yes:<15.0f} {energy_yes/baseline_energy:<12.2%} "
              f"({delta_pct:+.1f}%)")
        
        print()
    
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GreenSketch Ablation Study')
    parser.add_argument('--n-unique', type=int, default=10000)
    parser.add_argument('--n-total', type=int, default=50000)
    parser.add_argument('--precision', '-p', type=int, default=10)
    parser.add_argument('--output', '-o', default='results')
    
    args = parser.parse_args()
    
    results = run_ablation_study(
        n_unique=args.n_unique,
        n_total=args.n_total,
        p=args.precision,
        output_dir=args.output
    )
    
    print(f"\n✓ Ablation study complete! {len(results)} experiments run.")


if __name__ == "__main__":
    main()
