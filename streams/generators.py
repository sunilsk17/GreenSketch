"""
Stream generators for testing order-robustness and energy-awareness.

Provides diverse stream orderings to evaluate HLL variants under different
conditions, including adversarial patterns and realistic edge workloads.
"""

import numpy as np
from utils.hash_funcs import full_hash_64


class StreamGenerator:
    """Base class for stream generators."""
    
    def __init__(self, n_unique, n_total=None, seed=42):
        """
        Initialize stream generator.
        
        Args:
            n_unique: Number of unique items
            n_total: Total items to generate (including duplicates)
                     If None, equals n_unique
            seed: Random seed for reproducibility
        """
        self.n_unique = n_unique
        self.n_total = n_total if n_total is not None else n_unique
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self):
        """Generate stream. Must be implemented by subclasses."""
        raise NotImplementedError


class RandomOrderStream(StreamGenerator):
    """
    Random order stream (baseline).
    
    Items appear in uniformly random order.
    This is the standard assumption for HLL analysis.
    """
    
    def generate(self):
        """Generate randomly ordered stream."""
        # Generate unique items
        items = [f"item_{i}" for i in range(self.n_unique)]
        
        # If we need duplicates
        if self.n_total > self.n_unique:
            # Sample with replacement
            indices = self.rng.choice(self.n_unique, size=self.n_total, replace=True)
            items = [f"item_{i}" for i in indices]
        
        # Shuffle
        self.rng.shuffle(items)
        
        return items


class SortedByHashStream(StreamGenerator):
    """
    Sorted by hash value (adversarial best-case).
    
    Items arrive in sorted order by hash value, causing registers
    to be filled in a structured pattern rather than randomly.
    """
    
    def generate(self):
        """Generate hash-sorted stream."""
        # Generate unique items with their hashes
        items_with_hash = []
        for i in range(self.n_unique):
            item = f"item_{i}"
            h = full_hash_64(item)
            items_with_hash.append((h, item))
        
        # Sort by hash value
        items_with_hash.sort(key=lambda x: x[0])
        
        # Extract items
        items = [item for _, item in items_with_hash]
        
        # Add duplicates if needed
        if self.n_total > self.n_unique:
            # Repeat the sorted sequence
            repeats = self.n_total // self.n_unique + 1
            items = (items * repeats)[:self.n_total]
        
        return items


class AdversarialStream(StreamGenerator):
    """
    Adversarial stream (worst-case for energy).
    
    Sends items that cause high collision rates first, then
    gradually introduce more diverse items. This maximizes
    early register updates.
    """
    
    def generate(self):
        """Generate adversarial stream."""
        # Generate unique items
        items = [f"item_{i}" for i in range(self.n_unique)]
        
        # Compute hash for each and group by register index (first p bits)
        # Assume p=10 for grouping (can be made configurable)
        p = 10
        m = 1 << p
        
        buckets = [[] for _ in range(m)]
        for item in items:
            h = full_hash_64(item)
            idx = h & (m - 1)
            buckets[idx].append(item)
        
        # Build adversarial order:
        # 1. First, send items that fill each register (one per bucket)
        # 2. Then send remaining items (causes fewer updates)
        adversarial_order = []
        
        # First pass: one item per bucket (causes register fills)
        for bucket in buckets:
            if bucket:
                adversarial_order.append(bucket[0])
        
        # Second pass: remaining items
        for bucket in buckets:
            if len(bucket) > 1:
                adversarial_order.extend(bucket[1:])
        
        # Add duplicates if needed
        if self.n_total > self.n_unique:
            repeats = self.n_total // self.n_unique + 1
            adversarial_order = (adversarial_order * repeats)[:self.n_total]
        
        return adversarial_order


class BurstyZipfianStream(StreamGenerator):
    """
    Bursty stream with Zipfian distribution.
    
    Models skewed workloads common in edge analytics where
    certain items (e.g., popular IPs, frequent sensors) appear
    much more often than others.
    """
    
    def __init__(self, n_unique, n_total=None, alpha=1.5, seed=42):
        """
        Initialize bursty Zipfian stream.
        
        Args:
            n_unique: Number of unique items
            n_total: Total items (including repeats)
            alpha: Zipf parameter (larger = more skewed)
            seed: Random seed
        """
        super().__init__(n_unique, n_total, seed)
        self.alpha = alpha
    
    def generate(self):
        """Generate Zipfian-distributed stream."""
        # Generate Zipfian probabilities
        ranks = np.arange(1, self.n_unique + 1)
        probabilities = 1.0 / (ranks ** self.alpha)
        probabilities /= probabilities.sum()
        
        # Sample items according to Zipfian distribution
        indices = self.rng.choice(
            self.n_unique,
            size=self.n_total,
            replace=True,
            p=probabilities
        )
        
        items = [f"item_{i}" for i in indices]
        
        return items


# ============================================================================
# Named Sustainability Workloads
# ============================================================================

class IoTDeviceIDStream(RandomOrderStream):
    """
    IoT device identifier stream (MAC addresses, sensor IDs).
    
    Models smart city edge deployments where devices continuously
    transmit data with unique identifiers.
    """
    
    def generate(self):
        """Generate IoT device ID stream."""
        # Generate MAC-like addresses
        items = []
        for i in range(self.n_unique):
            # Format as MAC address
            mac = f"{(i >> 40) & 0xFF:02x}:{(i >> 32) & 0xFF:02x}:" \
                  f"{(i >> 24) & 0xFF:02x}:{(i >> 16) & 0xFF:02x}:" \
                  f"{(i >> 8) & 0xFF:02x}:{i & 0xFF:02x}"
            items.append(f"mac_{mac}")
        
        # Add duplicates if needed (devices send multiple packets)
        if self.n_total > self.n_unique:
            indices = self.rng.choice(self.n_unique, size=self.n_total, replace=True)
            items = [items[i] for i in indices]
        
        # Shuffle
        self.rng.shuffle(items)
        
        return items


class NetworkFlowIDStream(BurstyZipfianStream):
    """
    Network flow identifier stream (IP:port tuples).
    
    Models network telemetry where certain flows (popular services)
    appear much more frequently than others.
    """
    
    def __init__(self, n_unique, n_total=None, alpha=1.3, seed=42):
        """Initialize with Zipfian distribution (models real traffic)."""
        super().__init__(n_unique, n_total, alpha, seed)
    
    def generate(self):
        """Generate network flow ID stream."""
        # Generate Zipfian indices
        ranks = np.arange(1, self.n_unique + 1)
        probabilities = 1.0 / (ranks ** self.alpha)
        probabilities /= probabilities.sum()
        
        indices = self.rng.choice(
            self.n_unique,
            size=self.n_total,
            replace=True,
            p=probabilities
        )
        
        # Format as IP:port
        items = []
        for idx in indices:
            ip = f"{(idx >> 24) & 0xFF}.{(idx >> 16) & 0xFF}." \
                 f"{(idx >> 8) & 0xFF}.{idx & 0xFF}"
            port = 1024 + (idx % 64512)
            items.append(f"flow_{ip}:{port}")
        
        return items


class TrafficSensorIDStream(BurstyZipfianStream):
    """
    Traffic sensor stream (vehicle/pedestrian IDs).
    
    Models smart traffic systems where certain routes/sensors
    see much more activity than others.
    """
    
    def __init__(self, n_unique, n_total=None, alpha=1.2, seed=42):
        """Initialize with moderate Zipfian skew."""
        super().__init__(n_unique, n_total, alpha, seed)
    
    def generate(self):
        """Generate traffic sensor stream."""
        # Generate Zipfian indices
        ranks = np.arange(1, self.n_unique + 1)
        probabilities = 1.0 / (ranks ** self.alpha)
        probabilities /= probabilities.sum()
        
        indices = self.rng.choice(
            self.n_unique,
            size=self.n_total,
            replace=True,
            p=probabilities
        )
        
        # Format as sensor readings
        items = []
        for idx in indices:
            sensor_id = f"S{idx // 1000:03d}"
            vehicle_id = f"V{idx:06d}"
            items.append(f"sensor_{sensor_id}_{vehicle_id}")
        
        return items


# ============================================================================
# Utility Functions
# ============================================================================

def get_all_generators():
    """
    Get all available stream generator classes.
    
    Returns:
        Dictionary mapping names to generator classes
    """
    return {
        # Order patterns
        'random': RandomOrderStream,
        'sorted_hash': SortedByHashStream,
        'adversarial': AdversarialStream,
        'zipfian': BurstyZipfianStream,
        
        # Named workloads
        'iot_devices': IoTDeviceIDStream,
        'network_flows': NetworkFlowIDStream,
        'traffic_sensors': TrafficSensorIDStream,
    }


def test_generators():
    """Test stream generators."""
    print("Testing Stream Generators...\n")
    
    n_unique = 1000
    n_total = 5000
    
    generators = get_all_generators()
    
    for name, GeneratorClass in generators.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        
        gen = GeneratorClass(n_unique=n_unique, n_total=n_total, seed=42)
        stream = gen.generate()
        
        # Analyze stream properties
        unique_items = len(set(stream))
        
        print(f"  Generated: {len(stream)} items")
        print(f"  Unique: {unique_items}")
        print(f"  Repetition rate: {(1 - unique_items/len(stream)):.2%}")
        print(f"  Sample items: {stream[:3]}")
    
    print("\n✓ All stream generators working!")


if __name__ == "__main__":
    test_generators()
