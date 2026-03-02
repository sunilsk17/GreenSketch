"""
Sensitivity analysis: Parameter sweeps.

Evaluates how energy and accuracy vary with key parameters:
- Precision p
- THLL delta threshold
- AP-HLL rho cap
"""

import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.hll import HyperLogLog
from core.thll import ThresholdedHLL
from core.aphll import AdaptivePrecisionHLL
from streams.generators import RandomOrderStream, BurstyZipfianStream
from energy.metrics import EnergyMetrics, compute_energy_proxy
from utils.hash_funcs import reset_hash_counter, get_hash_counter


def run_sensitivity_experiment(hll_class, stream, ground_truth, **hll_kwargs):
    """Run single sensitivity experiment."""
    reset_hash_counter()
    
    hll = hll_class(**hll_kwargs)
    
    energy_metrics = EnergyMetrics()
    energy_metrics.start()
    
    for item in stream:
        hll.add(item)
    
    energy_metrics.stop()
    
    estimate = hll.estimate()
    hll_metrics = hll.get_metrics()
    measurements = energy_metrics.get_measurements(hll_metrics)
    hash_stats = get_hash_counter().get_stats()
    
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


def precision_sweep(n_unique=10000, n_total=50000, output_dir='results'):
    """
    Sweep precision parameter p ∈ {8, 10, 12, 14}.
    
    Shows how memory-accuracy-energy tradeoff changes with p.
    """
    print("="*80)
    print("SENSITIVITY ANALYSIS: Precision (p) Sweep")
    print("="*80)
    
    p_values = [8, 10, 12, 14]
    results = []
    
    # Use random stream
    for p in p_values:
        print(f"\n--- p = {p} (m = {1 << p} registers) ---")
        
        generator = RandomOrderStream(n_unique, n_total, seed=42)
        stream = generator.generate()
        ground_truth = len(set(stream))
        
        for variant_name, variant_class in [('Std HLL', HyperLogLog),
                                            ('THLL', ThresholdedHLL)]:
            print(f"  {variant_name}... ", end='')
            
            kwargs = {'p': p}
            if variant_class == ThresholdedHLL:
                kwargs['delta'] = 1
                kwargs['adaptive_delta'] = False
            
            result = run_sensitivity_experiment(variant_class, stream, ground_truth, **kwargs)
            result['variant'] = variant_name
            result['p'] = p
            result['m'] = 1 << p
            
            results.append(result)
            
            print(f"✓ Error: {result['relative_error_pct']:.2f}%, Energy: {result['energy_proxy']:.0f}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'sensitivity_precision.csv')
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Precision sweep results saved to {csv_path}")
    
    return results


def delta_sweep(n_unique=10000, n_total=50000, p=10, output_dir='results'):
    """
    Sweep THLL delta ∈ {0, 1, 2, 3}.
    
    Shows energy-accuracy tradeoff as threshold increases.
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: THLL Delta Sweep")
    print("="*80)
    
    delta_values = [0, 1, 2, 3]
    results = []
    
    generator = RandomOrderStream(n_unique, n_total, seed=42)
    stream = generator.generate()
    ground_truth = len(set(stream))
    
    for delta in delta_values:
        print(f"\n--- delta = {delta} ---")
        print(f"  THLL... ", end='')
        
        result = run_sensitivity_experiment(
            ThresholdedHLL, stream, ground_truth,
            p=p, delta=delta, adaptive_delta=False
        )
        result['variant'] = 'THLL'
        result['delta'] = delta
        result['p'] = p
        
        results.append(result)
        
        print(f"✓ Error: {result['relative_error_pct']:.2f}%, "
              f"Energy: {result['energy_proxy']:.0f}, "
              f"Updates/M: {result['register_updates_per_M']:.0f}")
    
    # Save
    csv_path = os.path.join(output_dir, 'sensitivity_delta.csv')
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Delta sweep results saved to {csv_path}")
    
    return results


def rho_cap_sweep(n_unique=10000, n_total=50000, p=10, output_dir='results'):
    """
    Sweep AP-HLL rho_cap ∈ {16, 32, 48, 64}.
    
    Shows impact of precision capping on energy and accuracy.
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: AP-HLL Rho Cap Sweep")
    print("="*80)
    
    rho_cap_values = [16, 32, 48, 64]
    results = []
    
    generator = BurstyZipfianStream(n_unique, n_total, alpha=1.5, seed=42)
    stream = generator.generate()
    ground_truth = len(set(stream))
    
    for rho_cap in rho_cap_values:
        print(f"\n--- rho_cap = {rho_cap} ---")
        print(f"  AP-HLL... ", end='')
        
        result = run_sensitivity_experiment(
            AdaptivePrecisionHLL, stream, ground_truth,
            p=p, rho_cap=rho_cap, adaptive=False
        )
        result['variant'] = 'AP-HLL'
        result['rho_cap'] = rho_cap
        result['p'] = p
        
        results.append(result)
        
        print(f"✓ Error: {result['relative_error_pct']:.2f}%, "
              f"Energy: {result['energy_proxy']:.0f}")
    
    # Save
    csv_path = os.path.join(output_dir, 'sensitivity_rho_cap.csv')
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Rho cap sweep results saved to {csv_path}")
    
    return results


def run_all_sensitivity_analyses(n_unique=10000, n_total=50000, p=10, output_dir='results'):
    """Run all sensitivity analyses."""
    print("="*80)
    print("RUNNING ALL SENSITIVITY ANALYSES")
    print("="*80)
    
    precision_results = precision_sweep(n_unique, n_total, output_dir)
    delta_results = delta_sweep(n_unique, n_total, p, output_dir)
    rho_cap_results = rho_cap_sweep(n_unique, n_total, p, output_dir)
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSES COMPLETE")
    print("="*80)
    print(f"\nTotal experiments: {len(precision_results) + len(delta_results) + len(rho_cap_results)}")
    
    return {
        'precision': precision_results,
        'delta': delta_results,
        'rho_cap': rho_cap_results
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GreenSketch Sensitivity Analysis')
    parser.add_argument('--n-unique', type=int, default=10000)
    parser.add_argument('--n-total', type=int, default=50000)
    parser.add_argument('--precision', '-p', type=int, default=10)
    parser.add_argument('--output', '-o', default='results')
    parser.add_argument('--sweep', choices=['precision', 'delta', 'rho_cap', 'all'],
                       default='all', help='Which sweep to run')
    
    args = parser.parse_args()
    
    if args.sweep == 'precision':
        precision_sweep(args.n_unique, args.n_total, args.output)
    elif args.sweep == 'delta':
        delta_sweep(args.n_unique, args.n_total, args.precision, args.output)
    elif args.sweep == 'rho_cap':
        rho_cap_sweep(args.n_unique, args.n_total, args.precision, args.output)
    else:
        run_all_sensitivity_analyses(args.n_unique, args.n_total, args.precision, args.output)
    
    print("\n✓ Sensitivity analysis complete!")


if __name__ == "__main__":
    main()
