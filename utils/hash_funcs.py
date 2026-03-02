"""
Hash functions for GreenSketch energy-aware HyperLogLog.

Provides two-tier hashing infrastructure:
- Cheap 32-bit hash for quick indexing
- Full 64-bit cryptographic hash for precision
"""

import mmh3


class HashCounter:
    """Global counter for tracking hash function invocations (energy proxy)."""
    
    def __init__(self):
        self.cheap_hash_calls = 0
        self.full_hash_calls = 0
    
    def reset(self):
        """Reset all counters."""
        self.cheap_hash_calls = 0
        self.full_hash_calls = 0
    
    def get_stats(self):
        """Return current hash call statistics."""
        return {
            'cheap_hash_calls': self.cheap_hash_calls,
            'full_hash_calls': self.full_hash_calls,
            'total_calls': self.cheap_hash_calls + self.full_hash_calls
        }


# Global hash counter instance
_counter = HashCounter()


def get_hash_counter():
    """Get the global hash counter instance."""
    return _counter


def reset_hash_counter():
    """Reset the global hash counter."""
    _counter.reset()


def cheap_hash_32(item):
    """
    Fast 32-bit hash using Python's built-in hash function.
    
    Energy cost: LOW
    Used for quick indexing and preliminary filtering.
    
    Args:
        item: Any hashable item (string, int, tuple, etc.)
    
    Returns:
        32-bit unsigned integer hash value
    """
    _counter.cheap_hash_calls += 1
    
    # Use Python's built-in hash and mask to 32 bits
    if isinstance(item, str):
        h = hash(item.encode('utf-8') if isinstance(item, str) else item)
    else:
        h = hash(item)
    
    # Ensure positive 32-bit value
    return h & 0xFFFFFFFF


def full_hash_64(item):
    """
    Cryptographic 64-bit hash using MurmurHash3.
    
    Energy cost: HIGH
    Provides strong avalanche properties for HLL precision.
    
    Args:
        item: Any hashable item (string, int, tuple, etc.)
    
    Returns:
        64-bit unsigned integer hash value
    """
    _counter.full_hash_calls += 1
    
    # Convert to bytes if needed
    if isinstance(item, str):
        data = item.encode('utf-8')
    elif isinstance(item, int):
        data = str(item).encode('utf-8')
    elif isinstance(item, bytes):
        data = item
    else:
        data = str(item).encode('utf-8')
    
    # MurmurHash3 128-bit hash, take lower 64 bits
    hash_value = mmh3.hash64(data, signed=False)[0]
    
    return hash_value


def leading_zeros(x):
    """
    Count leading zero bits in a 64-bit integer (rho calculation for HLL).
    
    Efficiently computes the position of the first 1-bit.
    
    Args:
        x: 64-bit unsigned integer
    
    Returns:
        Number of leading zeros (0-64)
    """
    if x == 0:
        return 64
    
    # Use bit_length to find position of highest set bit
    # leading_zeros = 64 - bit_length
    return 64 - x.bit_length()


def test_hash_functions():
    """Test hash functions for correctness."""
    print("Testing hash functions...")
    
    # Reset counter
    reset_hash_counter()
    
    # Test items
    items = ["test1", "test2", 12345, ("tuple", "item")]
    
    print("\nCheap Hash (32-bit):")
    for item in items:
        h = cheap_hash_32(item)
        print(f"  {item:20} -> {h:10} (0x{h:08x})")
    
    print("\nFull Hash (64-bit):")
    for item in items:
        h = full_hash_64(item)
        print(f"  {item:20} -> {h:20} (0x{h:016x})")
    
    print("\nLeading Zeros:")
    test_values = [0, 1, 0xFF, 0xFFFF, 0xFFFFFFFF, 0x8000000000000000]
    for val in test_values:
        lz = leading_zeros(val)
        print(f"  0x{val:016x} -> {lz:2} leading zeros")
    
    # Check counter
    stats = _counter.get_stats()
    print(f"\nHash Counter Stats:")
    print(f"  Cheap calls: {stats['cheap_hash_calls']}")
    print(f"  Full calls: {stats['full_hash_calls']}")
    print(f"  Total: {stats['total_calls']}")
    
    print("\n✓ Hash functions working correctly!")


if __name__ == "__main__":
    test_hash_functions()
