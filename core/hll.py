"""
Standard HyperLogLog implementation with comprehensive energy measurement.

This serves as the baseline for comparing energy-aware variants.
"""

import numpy as np
import time
from utils.hash_funcs import full_hash_64, leading_zeros


class HyperLogLog:
    """
    Standard HyperLogLog cardinality estimator.
    
    Implements the classic HLL algorithm with comprehensive instrumentation
    for energy proxy measurement.
    
    Attributes:
        p: Precision parameter (number of bits for register indexing)
        m: Number of registers (2^p)
        registers: Register array storing maximum rho values
        alpha: Bias correction constant
        
    Energy Metrics:
        total_items: Total items processed
        register_updates: Number of register modifications
        cpu_time: Total CPU processing time
        wall_time: Total wall-clock time
    """
    
    def __init__(self, p=10):
        """
        Initialize HyperLogLog.
        
        Args:
            p: Precision parameter (typically 4-16, default 10)
               Higher p = more accuracy but more memory
        """
        if p < 4 or p > 16:
            raise ValueError("Precision p must be between 4 and 16")
        
        self.p = p
        self.m = 1 << p  # 2^p registers
        self.registers = np.zeros(self.m, dtype=np.uint8)
        
        # Bias correction constant (from HLL paper)
        self.alpha = self._get_alpha_mm(self.m)
        
        # Energy & performance metrics
        self.total_items = 0
        self.register_updates = 0
        self.cpu_time = 0.0
        self.wall_time = 0.0
        
        # Internal timing
        self._start_cpu = None
        self._start_wall = None
    
    def _get_alpha_mm(self, m):
        """
        Get alpha constant for bias correction.
        
        Args:
            m: Number of registers
        
        Returns:
            Alpha constant
        """
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            # For m >= 128
            return 0.7213 / (1.0 + 1.079 / m)
    
    def add(self, item):
        """
        Add an item to the sketch.
        
        Args:
            item: Any hashable item (string, int, tuple, etc.)
        """
        start_cpu = time.process_time()
        start_wall = time.time()
        
        # Hash the item (full 64-bit hash)
        x = full_hash_64(item)
        
        # Extract index from first p bits
        idx = x & (self.m - 1)
        
        # Get remaining bits and compute rho (leading zeros + 1)
        w = x >> self.p
        rho = leading_zeros(w) + 1
        
        # Cap rho at 64 - p to prevent overflow
        rho = min(rho, 64 - self.p)
        
        # Update register if new rho is larger
        if rho > self.registers[idx]:
            self.registers[idx] = rho
            self.register_updates += 1
        
        # Update metrics
        self.total_items += 1
        self.cpu_time += time.process_time() - start_cpu
        self.wall_time += time.time() - start_wall
    
    def estimate(self):
        """
        Estimate cardinality using harmonic mean of registers.
        
        Returns:
            Estimated number of distinct items
        """
        # Harmonic mean of 2^register_values
        raw_estimate = self.alpha * (self.m ** 2) / np.sum(2.0 ** (-self.registers))
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            # Count zero registers
            zeros = np.count_nonzero(self.registers == 0)
            if zeros != 0:
                return self.m * np.log(self.m / float(zeros))
        
        # Large range correction
        if raw_estimate > (1.0/30.0) * (1 << 32):
            return -1 * (1 << 32) * np.log(1.0 - raw_estimate / (1 << 32))
        
        return raw_estimate
    
    def merge(self, other):
        """
        Merge another HLL sketch into this one.
        
        Args:
            other: Another HyperLogLog instance with same p
        """
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precision")
        
        # Take maximum register values
        self.registers = np.maximum(self.registers, other.registers)
        
        # Note: We don't merge metrics as they're instance-specific
    
    def get_metrics(self):
        """
        Get comprehensive energy and performance metrics.
        
        Returns:
            Dictionary with all tracked metrics
        """
        metrics = {
            # Core metrics
            'total_items': self.total_items,
            'register_updates': self.register_updates,
            'cpu_time': self.cpu_time,
            'wall_time': self.wall_time,
            
            # Derived metrics (per million items)
            'cpu_time_per_M': (self.cpu_time / self.total_items * 1e6) if self.total_items > 0 else 0,
            'wall_time_per_M': (self.wall_time / self.total_items * 1e6) if self.total_items > 0 else 0,
            'register_updates_per_M': (self.register_updates / self.total_items * 1e6) if self.total_items > 0 else 0,
            
            # Update rate (important for energy analysis)
            'update_rate': (self.register_updates / self.total_items) if self.total_items > 0 else 0,
            
            # Configuration
            'precision_p': self.p,
            'num_registers': self.m,
        }
        
        return metrics
    
    def reset_metrics(self):
        """Reset all energy and performance counters."""
        self.total_items = 0
        self.register_updates = 0
        self.cpu_time = 0.0
        self.wall_time = 0.0
    
    def reset(self):
        """Reset the entire sketch (registers + metrics)."""
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.reset_metrics()
    
    def __len__(self):
        """Return estimated cardinality."""
        return int(self.estimate())
    
    def __repr__(self):
        """String representation."""
        return f"HyperLogLog(p={self.p}, m={self.m}, estimate={self.estimate():.0f})"


def test_hll():
    """Test standard HLL functionality."""
    print("Testing Standard HyperLogLog...")
    
    # Create HLL with p=10 (1024 registers)
    hll = HyperLogLog(p=10)
    
    # Add known number of unique items
    n_unique = 10000
    print(f"\nAdding {n_unique} unique items...")
    
    for i in range(n_unique):
        hll.add(f"item_{i}")
    
    # Estimate cardinality
    estimate = hll.estimate()
    error = abs(estimate - n_unique) / n_unique * 100
    
    print(f"\nResults:")
    print(f"  True cardinality: {n_unique}")
    print(f"  HLL estimate: {estimate:.0f}")
    print(f"  Relative error: {error:.2f}%")
    
    # Show metrics
    metrics = hll.get_metrics()
    print(f"\nEnergy Metrics:")
    print(f"  Total items: {metrics['total_items']}")
    print(f"  Register updates: {metrics['register_updates']}")
    print(f"  Update rate: {metrics['update_rate']:.3f}")
    print(f"  CPU time/M items: {metrics['cpu_time_per_M']:.3f} seconds")
    
    # Test with duplicates
    print(f"\n\nAdding {n_unique} duplicate items...")
    hll.reset_metrics()
    
    for i in range(n_unique):
        hll.add(f"item_{i}")  # Same items again
    
    estimate2 = hll.estimate()
    print(f"  Estimate after duplicates: {estimate2:.0f}")
    print(f"  (Should be ~same)")
    
    metrics2 = hll.get_metrics()
    print(f"  Update rate with duplicates: {metrics2['update_rate']:.3f}")
    print(f"  (Should be much lower)")
    
    print("\n✓ Standard HLL working correctly!")


if __name__ == "__main__":
    test_hll()
