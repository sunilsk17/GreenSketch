"""
Energy measurement and cost modeling for GreenSketch.

Provides comprehensive energy proxy measurement using:
- CPU time tracking
- Hash call counting
- Memory operation tracking
- Formal energy cost model
"""

import time
import numpy as np
from utils.hash_funcs import get_hash_counter


class EnergyMetrics:
    """
    Comprehensive energy proxy measurement.
    
    Tracks all energy-relevant operations and computes
    weighted energy proxy using formal cost model.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time_cpu = None
        self.start_time_wall = None
        self.end_time_cpu = None
        self.end_time_wall = None
        
        self.hash_calls_start = {'cheap': 0, 'full': 0}
        self.hash_calls_end = {'cheap': 0, 'full': 0}
    
    def start(self):
        """Start measurement."""
        self.start_time_cpu = time.process_time()
        self.start_time_wall = time.time()
        
        # Snapshot hash counters
        hash_stats = get_hash_counter().get_stats()
        self.hash_calls_start = {
            'cheap': hash_stats['cheap_hash_calls'],
            'full': hash_stats['full_hash_calls']
        }
    
    def stop(self):
        """Stop measurement."""
        self.end_time_cpu = time.process_time()
        self.end_time_wall = time.time()
        
        # Snapshot hash counters
        hash_stats = get_hash_counter().get_stats()
        self.hash_calls_end = {
            'cheap': hash_stats['cheap_hash_calls'],
            'full': hash_stats['full_hash_calls']
        }
    
    def get_measurements(self, hll_metrics):
        """
        Get all measurements.
        
        Args:
            hll_metrics: Metrics dictionary from HLL variant
        
        Returns:
            Dictionary with all measurements
        """
        if self.start_time_cpu is None or self.end_time_cpu is None:
            raise ValueError("Must call start() and stop() before getting measurements")
        
        # Time measurements
        cpu_time = self.end_time_cpu - self.start_time_cpu
        wall_time = self.end_time_wall - self.start_time_wall
        
        # Hash calls (delta)
        cheap_calls = self.hash_calls_end['cheap'] - self.hash_calls_start['cheap']
        full_calls = self.hash_calls_end['full'] - self.hash_calls_start['full']
        
        # Items processed
        total_items = hll_metrics.get('total_items', 0)
        
        # Register operations
        register_updates = hll_metrics.get('register_updates', 0)
        skipped_updates = hll_metrics.get('skipped_updates', 0)
        
        measurements = {
            # Raw metrics
            'cpu_time': cpu_time,
            'wall_time': wall_time,
            'cheap_hash_calls': cheap_calls,
            'full_hash_calls': full_calls,
            'register_updates': register_updates,
            'skipped_updates': skipped_updates,
            'total_items': total_items,
            
            # Per million items
            'cpu_time_per_M': (cpu_time / total_items * 1e6) if total_items > 0 else 0,
            'wall_time_per_M': (wall_time / total_items * 1e6) if total_items > 0 else 0,
            'cheap_calls_per_M': (cheap_calls / total_items * 1e6) if total_items > 0 else 0,
            'full_calls_per_M': (full_calls / total_items * 1e6) if total_items > 0 else 0,
            'register_updates_per_M': (register_updates / total_items * 1e6) if total_items > 0 else 0,
            'skipped_updates_per_M': (skipped_updates / total_items * 1e6) if total_items > 0 else 0,
            
            # Rates
            'update_rate': (register_updates / total_items) if total_items > 0 else 0,
            'skip_rate': (skipped_updates / total_items) if total_items > 0 else 0,
        }
        
        return measurements
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


def compute_energy_proxy(measurements, weights=None):
    """
    Compute energy proxy using weighted cost model.
    
    Energy Model:
        E = α × full_hash_calls
          + β × cheap_hash_calls  
          + γ × register_updates
          + δ × cpu_time
    
    Args:
        measurements: Dictionary from EnergyMetrics.get_measurements()
        weights: Dictionary with keys {alpha, beta, gamma, delta}
                 If None, uses default weights
    
    Returns:
        Energy proxy value (higher = more energy)
    """
    if weights is None:
        # Default weights (relative energy costs)
        # These are approximate relative costs based on:
        # - Full hash ~10x more expensive than cheap hash
        # - Register update ~2x more expensive than cheap hash
        # - CPU time normalized
        weights = {
            'alpha': 10.0,   # full_hash_calls coefficient
            'beta': 1.0,     # cheap_hash_calls coefficient
            'gamma': 2.0,    # register_updates coefficient
            'delta': 1e6,    # cpu_time coefficient (normalized)
        }
    
    # Extract per-million metrics for fair comparison
    full_calls_M = measurements.get('full_calls_per_M', 0)
    cheap_calls_M = measurements.get('cheap_calls_per_M', 0)
    updates_M = measurements.get('register_updates_per_M', 0)
    cpu_time_M = measurements.get('cpu_time_per_M', 0)
    
    # Compute weighted sum
    energy = (
        weights['alpha'] * full_calls_M +
        weights['beta'] * cheap_calls_M +
        weights['gamma'] * updates_M +
        weights['delta'] * cpu_time_M
    )
    
    return energy


def normalize_metrics(metrics_dict, baseline_key='std_hll'):
    """
    Normalize metrics relative to baseline.
    
    Args:
        metrics_dict: Dictionary of {variant_name: measurements}
        baseline_key: Which variant to use as baseline (default: 'std_hll')
    
    Returns:
        Dictionary of {variant_name: normalized_measurements}
    """
    if baseline_key not in metrics_dict:
        raise ValueError(f"Baseline key '{baseline_key}' not found in metrics")
    
    baseline = metrics_dict[baseline_key]
    normalized = {}
    
    for variant_name, measurements in metrics_dict.items():
        norm = {}
        
        for key, value in measurements.items():
            baseline_val = baseline.get(key, 1.0)
            
            # Avoid division by zero
            if baseline_val == 0:
                norm[f'{key}_rel'] = 1.0
            else:
                norm[f'{key}_rel'] = value / baseline_val
        
        # Keep absolute values too
        norm.update(measurements)
        normalized[variant_name] = norm
    
    return normalized


def print_energy_comparison(metrics_dict, baseline_key='std_hll'):
    """
    Print formatted energy comparison table.
    
    Args:
        metrics_dict: Dictionary of {variant_name: measurements}
        baseline_key: Baseline variant for normalization
    """
    print("\n" + "="*80)
    print("ENERGY COMPARISON TABLE")
    print("="*80)
    
    # Headers
    print(f"\n{'Variant':<15} {'CPU(s/M)':<12} {'Full Hash':<12} {'Reg Updates':<12} {'Energy Proxy':<12}")
    print("-"*80)
    
    # Compute energy proxies
    for variant_name in metrics_dict.keys():
        measurements = metrics_dict[variant_name]
        
        cpu_time_M = measurements.get('cpu_time_per_M', 0)
        full_calls_M = measurements.get('full_calls_per_M', 0)
        updates_M = measurements.get('register_updates_per_M', 0)
        
        energy_proxy = compute_energy_proxy(measurements)
        
        print(f"{variant_name:<15} {cpu_time_M:<12.4f} {full_calls_M:<12.0f} "
              f"{updates_M:<12.0f} {energy_proxy:<12.2f}")
    
    # Normalized table
    print("\n" + "-"*80)
    print("RELATIVE TO BASELINE (lower is better)")
    print("-"*80)
    
    baseline_energy = compute_energy_proxy(metrics_dict[baseline_key])
    
    print(f"\n{'Variant':<15} {'CPU':<12} {'Full Hash':<12} {'Reg Updates':<12} {'Energy':<12}")
    print("-"*80)
    
    baseline = metrics_dict[baseline_key]
    
    for variant_name in metrics_dict.keys():
        measurements = metrics_dict[variant_name]
        
        cpu_rel = measurements['cpu_time_per_M'] / baseline['cpu_time_per_M'] if baseline['cpu_time_per_M'] > 0 else 1.0
        hash_rel = measurements['full_calls_per_M'] / baseline['full_calls_per_M'] if baseline['full_calls_per_M'] > 0 else 1.0
        update_rel = measurements['register_updates_per_M'] / baseline['register_updates_per_M'] if baseline['register_updates_per_M'] > 0 else 1.0
        
        energy = compute_energy_proxy(measurements)
        energy_rel = energy / baseline_energy if baseline_energy > 0 else 1.0
        
        print(f"{variant_name:<15} {cpu_rel:<12.2%} {hash_rel:<12.2%} "
              f"{update_rel:<12.2%} {energy_rel:<12.2%}")
    
    print("="*80)


def test_energy_metrics():
    """Test energy metrics."""
    print("Testing Energy Metrics Framework...\n")
    
    # Simulate measurements
    mock_metrics = {
        'std_hll': {
            'cpu_time_per_M': 1.0,
            'full_calls_per_M': 1000000,
            'cheap_calls_per_M': 0,
            'register_updates_per_M': 600000,
            'total_items': 1000000
        },
        'lhll': {
            'cpu_time_per_M': 0.8,
            'full_calls_per_M': 500000,
            'cheap_calls_per_M': 1000000,
            'register_updates_per_M': 600000,
            'total_items': 1000000
        },
        'thll': {
            'cpu_time_per_M': 0.85,
            'full_calls_per_M': 1000000,
            'cheap_calls_per_M': 0,
            'register_updates_per_M': 400000,
            'total_items': 1000000
        }
    }
    
    print_energy_comparison(mock_metrics, baseline_key='std_hll')
    
    print("\n✓ Energy metrics framework working!")


if __name__ == "__main__":
    test_energy_metrics()
