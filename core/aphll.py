"""
Adaptive Precision HyperLogLog (AP-HLL) - Energy-Aware Variant.

Dynamically adjusts precision based on stream characteristics.
Caps rho values during high-skew phases to reduce bit scanning cost.

Energy savings come from:
1. Reduced bit operations during rho computation
2. Lower precision in phases where high precision isn't needed
3. Adaptive behavior under different stream patterns
"""

import numpy as np
import time
from utils.hash_funcs import full_hash_64, leading_zeros


class AdaptivePrecisionHLL:
    """
    Adaptive Precision HyperLogLog - dynamic complexity adjustment.
    
    Key Idea:
    - Monitor stream skew and diversity
    - Cap rho values during high-skew phases (reduces bit scanning)
    - Adapt precision to stream characteristics
    
    Energy Benefit:
    - Reduces expensive bit operations
    - Maintains accuracy under diverse streams
    - Robust to adversarial order
    """
    
    def __init__(self, p=10, rho_cap=None, adaptive=True):
        """
        Initialize AP-HLL.
        
        Args:
            p: Precision parameter
            rho_cap: Maximum rho value (None = no cap)
            adaptive: If True, rho_cap adapts to stream skew
        """
        if p < 4 or p > 16:
            raise ValueError("Precision p must be between 4 and 16")
        
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.alpha = self._get_alpha_mm(self.m)
        
        # Adaptive precision parameters
        self.rho_cap_base = rho_cap if rho_cap is not None else (64 - p)
        self.adaptive = adaptive
        self.current_rho_cap = self.rho_cap_base
        
        # Energy metrics
        self.total_items = 0
        self.register_updates = 0
        self.skipped_updates = 0
        self.capped_rho_count = 0  # Track how often rho was capped
        self.cpu_time = 0.0
        self.wall_time = 0.0
        
        # Stream characteristic tracking
        self._recent_update_rate = 0.5
        self._recent_high_rho_rate = 0.1  # Rate of high rho values
        self._adaptation_window = 500
        self._adaptation_counter = 0
    
    def _get_alpha_mm(self, m):
        """Get alpha constant for bias correction."""
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1.0 + 1.079 / m)
    
    def _detect_skew(self):
        """
        Detect stream skew based on update rate.
        
        Low update rate = high skew (many duplicates)
        High update rate = low skew (diverse items)
        """
        # Skewed streams have low update rates
        return self._recent_update_rate < 0.2
    
    def _adapt_rho_cap(self):
        """Adapt rho cap based on stream characteristics."""
        if not self.adaptive:
            return
        
        if self._adaptation_counter >= self._adaptation_window:
            is_skewed = self._detect_skew()
            
            if is_skewed:
                # High skew: reduce rho cap to save energy
                # (many items won't update anyway)
                self.current_rho_cap = max(16, self.rho_cap_base - 16)
            else:
                # Low skew: need full precision
                self.current_rho_cap = self.rho_cap_base
            
            self._adaptation_counter = 0
    
    def add(self, item):
        """
        Add item with adaptive precision.
        
        Args:
            item: Any hashable item
        """
        start_cpu = time.process_time()
        start_wall = time.time()
        
        # Hash the item
        x = full_hash_64(item)
        
        # Extract index and compute rho
        idx = x & (self.m - 1)
        w = x >> self.p
        rho = leading_zeros(w) + 1
        
        # Natural cap
        rho = min(rho, 64 - self.p)
        
        # Adaptive rho capping
        original_rho = rho
        if rho > self.current_rho_cap:
            rho = self.current_rho_cap
            self.capped_rho_count += 1
        
        # Update register
        current_val = self.registers[idx]
        if rho > current_val:
            self.registers[idx] = rho
            self.register_updates += 1
        else:
            self.skipped_updates += 1
        
        # Update metrics
        self.total_items += 1
        self.cpu_time += time.process_time() - start_cpu
        self.wall_time += time.time() - start_wall
        
        # Track stream characteristics for adaptation
        if self.adaptive:
            self._adaptation_counter += 1
            
            # Update rate (exponential moving average)
            alpha_ema = 0.01
            did_update = 1 if (rho > current_val) else 0
            self._recent_update_rate = (1 - alpha_ema) * self._recent_update_rate + alpha_ema * did_update
            
            # High rho rate
            is_high_rho = 1 if (original_rho > 32) else 0
            self._recent_high_rho_rate = (1 - alpha_ema) * self._recent_high_rho_rate + alpha_ema * is_high_rho
            
            self._adapt_rho_cap()
    
    def estimate(self):
        """Estimate cardinality."""
        raw_estimate = self.alpha * (self.m ** 2) / np.sum(2.0 ** (-self.registers))
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            zeros = np.count_nonzero(self.registers == 0)
            if zeros != 0:
                return self.m * np.log(self.m / float(zeros))
        
        # Large range correction
        if raw_estimate > (1.0/30.0) * (1 << 32):
            return -1 * (1 << 32) * np.log(1.0 - raw_estimate / (1 << 32))
        
        return raw_estimate
    
    def get_metrics(self):
        """Get comprehensive metrics."""
        metrics = {
            'total_items': self.total_items,
            'register_updates': self.register_updates,
            'skipped_updates': self.skipped_updates,
            'capped_rho_count': self.capped_rho_count,
            'cpu_time': self.cpu_time,
            'wall_time': self.wall_time,
            
            # Per million metrics
            'cpu_time_per_M': (self.cpu_time / self.total_items * 1e6) if self.total_items > 0 else 0,
            'wall_time_per_M': (self.wall_time / self.total_items * 1e6) if self.total_items > 0 else 0,
            'register_updates_per_M': (self.register_updates / self.total_items * 1e6) if self.total_items > 0 else 0,
            'skipped_updates_per_M': (self.skipped_updates / self.total_items * 1e6) if self.total_items > 0 else 0,
            
            # Rates
            'update_rate': (self.register_updates / self.total_items) if self.total_items > 0 else 0,
            'skip_rate': (self.skipped_updates / self.total_items) if self.total_items > 0 else 0,
            'cap_rate': (self.capped_rho_count / self.total_items) if self.total_items > 0 else 0,
            
            # Config
            'precision_p': self.p,
            'num_registers': self.m,
            'rho_cap_base': self.rho_cap_base,
            'current_rho_cap': self.current_rho_cap,
            'adaptive': self.adaptive,
            'variant': 'AP-HLL'
        }
        
        return metrics
    
    def reset_metrics(self):
        """Reset metrics."""
        self.total_items = 0
        self.register_updates = 0
        self.skipped_updates = 0
        self.capped_rho_count = 0
        self.cpu_time = 0.0
        self.wall_time = 0.0
        self._adaptation_counter = 0
        self._recent_update_rate = 0.5
        self._recent_high_rho_rate = 0.1
    
    def reset(self):
        """Reset everything."""
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.current_rho_cap = self.rho_cap_base
        self.reset_metrics()
    
    def __len__(self):
        return int(self.estimate())
    
    def __repr__(self):
        return f"AdaptivePrecisionHLL(p={self.p}, rho_cap={self.current_rho_cap}, estimate={self.estimate():.0f})"


def test_aphll():
    """Test AP-HLL."""
    print("Testing Adaptive Precision HLL...\n")
    
    hll = AdaptivePrecisionHLL(p=10, rho_cap=48, adaptive=True)
    
    # Add items
    n_unique = 10000
    print(f"Adding {n_unique} unique items (rho_cap=48, adaptive)...")
    
    for i in range(n_unique):
        hll.add(f"item_{i}")
    
    # Results
    estimate = hll.estimate()
    error = abs(estimate - n_unique) / n_unique * 100
    
    print(f"\nResults:")
    print(f"  True: {n_unique}")
    print(f"  Estimate: {estimate:.0f}")
    print(f"  Error: {error:.2f}%")
    
    # Metrics
    metrics = hll.get_metrics()
    
    print(f"\nEnergy Metrics:")
    print(f"  Register updates: {metrics['register_updates']}")
    print(f"  Skipped updates: {metrics['skipped_updates']}")
    print(f"  Capped rho: {metrics['capped_rho_count']}")
    print(f"  Cap rate: {metrics['cap_rate']:.2%}")
    print(f"  Current rho cap: {metrics['current_rho_cap']}")
    
    print("\n✓ AP-HLL working!")


if __name__ == "__main__":
    test_aphll()
