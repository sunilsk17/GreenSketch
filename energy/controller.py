"""
Adaptive energy controller for order-robust HLL variants.

Monitors stream characteristics and adaptively tunes parameters
to optimize energy-accuracy tradeoff across different stream patterns.
"""

import numpy as np


class StreamMonitor:
    """
    Monitors stream characteristics in real-time.
    
    Tracks:
    - Update rate (diversity indicator)
    - Entropy estimate (distribution uniformity) 
    - Skew detection (repetition patterns)
    """
    
    def __init__(self, window_size=1000):
        """
        Initialize stream monitor.
        
        Args:
            window_size: Number of items to consider for statistics
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset monitoring state."""
        self.update_count = 0
        self.total_count = 0
        self.update_rate = 0.5  # Exponential moving average
        self.recent_updates = []
        self.ema_alpha = 0.01
    
    def observe_update(self, did_update):
        """
        Observe whether an item caused a register update.
        
        Args:
            did_update: Boolean, True if register was updated
        """
        self.total_count += 1
        
        if did_update:
            self.update_count += 1
        
        # Update exponential moving average
        self.update_rate = (1 - self.ema_alpha) * self.update_rate + self.ema_alpha * (1 if did_update else 0)
        
        # Track recent updates for entropy
        self.recent_updates.append(1 if did_update else 0)
        if len(self.recent_updates) > self.window_size:
            self.recent_updates.pop(0)
    
    def get_update_rate(self):
        """Get current update rate (0-1)."""
        return self.update_rate
    
    def get_entropy_estimate(self):
        """
        Estimate stream entropy from update patterns.
        
        Returns:
            Entropy estimate (0-1, higher = more diverse)
        """
        if len(self.recent_updates) < 10:
            return 0.5  # Default
        
        # Simple entropy: if update rate is ~0.5, high entropy
        # If very high or very low, lower entropy
        rate = sum(self.recent_updates) / len(self.recent_updates)
        
        # Entropy maximized at p=0.5
        if rate == 0 or rate == 1:
            return 0.0
        
        # Binary entropy
        entropy = -rate * np.log2(rate) - (1-rate) * np.log2(1-rate)
        
        return entropy
    
    def detect_skew(self):
        """
        Detect if stream is skewed (many duplicates).
        
        Returns:
            'high', 'medium', or 'low'
        """
        rate = self.get_update_rate()
        
        if rate < 0.2:
            return 'high'  # Very few updates = high skew
        elif rate < 0.5:
            return 'medium'
        else:
            return 'low'  # Many updates = diverse stream
    
    def get_stats(self):
        """Get all monitoring statistics."""
        return {
            'update_rate': self.get_update_rate(),
            'entropy': self.get_entropy_estimate(),
            'skew': self.detect_skew(),
            'total_observed': self.total_count
        }


class AdaptiveController:
    """
    Adaptive energy controller for HLL variants.
    
    Adjusts parameters based on stream characteristics:
    - THLL delta threshold
    - AP-HLL rho cap
    - LHLL hashing mode
    """
    
    def __init__(self, variant_type='auto', adaptation_rate='medium'):
        """
        Initialize adaptive controller.
        
        Args:
            variant_type: 'lhll', 'thll', 'aphll', or 'auto'
            adaptation_rate: 'slow', 'medium', or 'fast'
        """
        self.variant_type = variant_type
        self.monitor = StreamMonitor(window_size=1000)
        
        # Adaptation rates
        rates = {
            'slow': 2000,
            'medium': 1000,
            'fast': 500
        }
        self.adaptation_interval = rates.get(adaptation_rate, 1000)
        self.steps_since_adaptation = 0
        
        # Current parameters
        self.current_delta = 1  # For THLL
        self.current_rho_cap = 48  # For AP-HLL
        self.lazy_hash_enabled = True  # For LHLL
        
        # Parameter bounds
        self.delta_min = 0
        self.delta_max = 3
        self.rho_cap_min = 16
        self.rho_cap_max = 64
    
    def observe(self, did_update):
        """
        Observe a stream event.
        
        Args:
            did_update: Boolean, whether item updated a register
        """
        self.monitor.observe_update(did_update)
        self.steps_since_adaptation += 1
        
        # Periodically adapt
        if self.steps_since_adaptation >= self.adaptation_interval:
            self.adapt_parameters()
            self.steps_since_adaptation = 0
    
    def adapt_parameters(self):
        """Adapt parameters based on current stream characteristics."""
        stats = self.monitor.get_stats()
        update_rate = stats['update_rate']
        skew = stats['skew']
        
        # Adapt THLL delta
        if skew == 'high':
            # High skew: can use larger delta (less frequent updates needed)
            self.current_delta = min(self.delta_max, self.current_delta + 1)
        elif skew == 'low':
            # Low skew: need smaller delta to capture diversity
            self.current_delta = max(self.delta_min, self.current_delta - 1)
        
        # Adapt AP-HLL rho cap
        if skew == 'high':
            # High skew: can reduce precision
            self.current_rho_cap = max(self.rho_cap_min, self.current_rho_cap - 8)
        elif skew == 'low':
            # Low skew: need full precision
            self.current_rho_cap = min(self.rho_cap_max, self.current_rho_cap + 8)
        
        # Adapt LHLL mode
        if update_rate < 0.15:
            # Very low update rate: lazy hashing very effective
            self.lazy_hash_enabled = True
        elif update_rate > 0.6:
            # High update rate: lazy hashing less effective, but still useful
            self.lazy_hash_enabled = True
    
    def get_delta(self):
        """Get current delta for THLL."""
        return self.current_delta
    
    def get_rho_cap(self):
        """Get current rho cap for AP-HLL."""
        return self.current_rho_cap
    
    def is_lazy_hash_enabled(self):
        """Check if lazy hashing should be used."""
        return self.lazy_hash_enabled
    
    def get_parameters(self):
        """Get all current parameters."""
        return {
            'delta': self.current_delta,
            'rho_cap': self.current_rho_cap,
            'lazy_hash_enabled': self.lazy_hash_enabled,
            'stream_stats': self.monitor.get_stats()
        }
    
    def reset(self):
        """Reset controller state."""
        self.monitor.reset()
        self.steps_since_adaptation = 0
        self.current_delta = 1
        self.current_rho_cap = 48
        self.lazy_hash_enabled = True


def test_controller():
    """Test adaptive controller."""
    print("Testing Adaptive Controller...\n")
    
    controller = AdaptiveController(variant_type='auto', adaptation_rate='fast')
    
    # Simulate high-skew stream (many duplicates)
    print("Simulating high-skew stream (many duplicates)...")
    for i in range(2000):
        did_update = (i % 10 == 0)  # Only 10% update rate
        controller.observe(did_update)
    
    params = controller.get_parameters()
    print(f"\nAfter high-skew stream:")
    print(f"  Update rate: {params['stream_stats']['update_rate']:.2%}")
    print(f"  Skew: {params['stream_stats']['skew']}")
    print(f"  Delta: {params['delta']}")
    print(f"  Rho cap: {params['rho_cap']}")
    
    # Simulate low-skew stream (diverse)
    print("\n\nSimulating low-skew stream (diverse items)...")
    controller.reset()
    
    for i in range(2000):
        did_update = (i % 2 == 0)  # 50% update rate
        controller.observe(did_update)
    
    params = controller.get_parameters()
    print(f"\nAfter low-skew stream:")
    print(f"  Update rate: {params['stream_stats']['update_rate']:.2%}")
    print(f"  Skew: {params['stream_stats']['skew']}")
    print(f"  Delta: {params['delta']}")
    print(f"  Rho cap: {params['rho_cap']}")
    
    print("\n✓ Adaptive controller working!")


if __name__ == "__main__":
    test_controller()
