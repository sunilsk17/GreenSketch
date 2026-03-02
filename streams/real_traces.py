"""
Real-world trace generators for GreenSketch validation.

Loads preprocessed streams from Enron email and Wikipedia datasets
to validate energy-aware HLL variants on realistic workloads.
"""

import os
import gzip
import json
from typing import List, Dict


# Paths to preprocessed data
ENRON_BASE_PATH = "/Users/sunilkumars/Desktop/distinct-order-study/data"
WIKI_BASE_PATH = "/Users/sunilkumars/Desktop/distinct-order-study/data"


class RealTraceLoader:
    """Base class for real trace loaders."""
    
    def __init__(self, trace_type='chrono', max_items=None):
        """
        Initialize trace loader.
        
        Args:
            trace_type: 'chrono' (chronological), 'random', or 'grouped'
            max_items: Maximum items to load (None = all)
        """
        self.trace_type = trace_type
        self.max_items = max_items
        self.stats = {}
    
    def load(self) -> List[str]:
        """Load trace. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded trace."""
        return self.stats


class EnronEmailTrace(RealTraceLoader):
    """
    Enron email dataset trace.
    
    Real-world communication graph from Enron corporation.
    Public data from PACER investigation.
    
    Characteristics:
    - 100,000 emails
    - 2,995 unique senders
    - 97% duplicate ratio (very high skew)
    - Realistic burst patterns
    """
    
    def __init__(self, trace_type='chrono', max_items=None):
        super().__init__(trace_type, max_items)
        self.dataset_name = "Enron Email Corpus"
    
    def load(self) -> List[str]:
        """Load Enron email sender stream."""
        # Map trace type to file
        file_map = {
            'chrono': 'enron_items_chrono.txt',
            'random': 'enron_items_random.txt',
            'grouped': 'enron_items_grouped.txt',
        }
        
        filepath = os.path.join(ENRON_BASE_PATH, file_map.get(self.trace_type, 'enron_items_chrono.txt'))
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Enron trace file not found: {filepath}")
        
        items = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_items and i >= self.max_items:
                    break
                
                item = line.strip()
                if item:
                    items.append(item)
        
        # Load stats
        stats_file = os.path.join(ENRON_BASE_PATH, 'enron_stream_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
        
        # Compute actual stats for loaded subset
        self.stats['loaded_items'] = len(items)
        self.stats['loaded_unique'] = len(set(items))
        self.stats['loaded_dup_ratio'] = 1.0 - (len(set(items)) / len(items)) if items else 0
        self.stats['trace_type'] = self.trace_type
        
        return items
    
    def __repr__(self):
        return f"EnronEmailTrace(type={self.trace_type}, items={self.stats.get('loaded_items', 'N/A')})"


class WikipediaPageviewTrace(RealTraceLoader):
    """
    Wikipedia pageview dataset trace.
    
    Real-world content access patterns from Wikipedia.
    
    Characteristics:
    - 100,000 pageviews
    - 90,867 unique pages
    - 9% duplicate ratio (low skew)
    - High entropy stream
    """
    
    def __init__(self, trace_type='chrono', max_items=None):
        super().__init__(trace_type, max_items)
        self.dataset_name = "Wikipedia Pageviews"
    
    def load(self) -> List[str]:
        """Load Wikipedia pageview stream."""
        # Map trace type to file
        file_map = {
            'chrono': 'wikipedia_items_chrono.txt',
            'random': 'wikipedia_items_random.txt',
            'grouped': 'wikipedia_items_grouped.txt',
        }
        
        filepath = os.path.join(WIKI_BASE_PATH, file_map.get(self.trace_type, 'wikipedia_items_chrono.txt'))
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Wikipedia trace file not found: {filepath}")
        
        items = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_items and i >= self.max_items:
                    break
                
                item = line.strip()
                if item:
                    items.append(item)
        
        # Load stats
        stats_file = os.path.join(WIKI_BASE_PATH, 'wikipedia_stream_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
        
        # Compute actual stats for loaded subset
        self.stats['loaded_items'] = len(items)
        self.stats['loaded_unique'] = len(set(items))
        self.stats['loaded_dup_ratio'] = 1.0 - (len(set(items)) / len(items)) if items else 0
        self.stats['trace_type'] = self.trace_type
        
        return items
    
    def __repr__(self):
        return f"WikipediaPageviewTrace(type={self.trace_type}, items={self.stats.get('loaded_items', 'N/A')})"


def get_all_real_traces():
    """
    Get all available real trace configurations.
    
    Returns:
        Dictionary mapping trace names to (LoaderClass, kwargs)
    """
    traces = {}
    
    # Enron traces (all orderings)
    for trace_type in ['chrono', 'random', 'grouped']:
        name = f'enron_{trace_type}'
        traces[name] = (EnronEmailTrace, {'trace_type': trace_type})
    
    # Wikipedia traces (all orderings)
    for trace_type in ['chrono', 'random', 'grouped']:
        name = f'wiki_{trace_type}'
        traces[name] = (WikipediaPageviewTrace, {'trace_type': trace_type})
    
    return traces


def test_real_traces():
    """Test real trace loaders."""
    print("="*80)
    print("TESTING REAL TRACE LOADERS")
    print("="*80)
    
    # Test Enron
    print("\n--- Enron Email Trace ---")
    enron = EnronEmailTrace(trace_type='chrono', max_items=10000)
    items = enron.load()
    
    print(f"Loaded: {len(items)} items")
    print(f"Unique: {len(set(items))} senders")
    print(f"Duplicate ratio: {enron.stats['loaded_dup_ratio']:.2%}")
    print(f"Sample items: {items[:5]}")
    print(f"Stats: {enron.stats['dataset']}")
    
    # Test Wikipedia
    print("\n--- Wikipedia Pageview Trace ---")
    wiki = WikipediaPageviewTrace(trace_type='chrono', max_items=10000)
    items = wiki.load()
    
    print(f"Loaded: {len(items)} items")
    print(f"Unique: {len(set(items))} pages")
    print(f"Duplicate ratio: {wiki.stats['loaded_dup_ratio']:.2%}")
    print(f"Sample items: {items[:5]}")
    
    # Test all variants
    print("\n--- All Available Traces ---")
    all_traces = get_all_real_traces()
    for name, (LoaderClass, kwargs) in all_traces.items():
        print(f"  {name}: {LoaderClass.__name__}")
    
    print(f"\n✓ Total real trace configurations: {len(all_traces)}")
    print("\n✓ Real trace loaders working!")


if __name__ == "__main__":
    test_real_traces()
