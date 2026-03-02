# GreenSketch

**Energy-Aware Order-Robust Cardinality Estimation for Sustainable Edge Streams**

GreenSketch introduces three energy-optimized HyperLogLog variants for distinct counting at the edge. Our algorithms reduce computational energy while maintaining accuracy across adversarial stream orderings, validated on both synthetic and real-world datasets.

**Main Result**: 18.6% energy reduction on Wikipedia pageviews with < 1% accuracy loss.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run synthetic benchmark (28 experiments across 7 stream types)
python experiments/benchmark.py

# Run real-world experiments
python experiments/fix_validation_issues.py   # Wikipedia p=14, Enron scale comparison

# Generate all visualizations
python experiments/visualize.py
python experiments/visualize_real_world.py
```

---

## Key Results

### Wikipedia Pageviews (47K unique items, p=14)

| Variant | Error | Energy Reduction |
|---------|-------|------------------|
| Std HLL | 0.51% | — (baseline) |
| **THLL** | **0.51%** | **−18.6%** ✅ |
| **AP-HLL** | **0.51%** | **−18.3%** ✅ |

> See [`RESULTS.md`](RESULTS.md) for complete results across all datasets and configurations.

---

## Algorithms

**1. Lazy Hashing HLL (LHLL)**  
Two-tier hashing: cheap 32-bit filter + full 64-bit hash only when needed. Reduces hash operations by 30–50% on synthetic streams. Trade-off: accuracy loss on high-cardinality streams.

**2. Thresholded Update HLL (THLL)**  
Skips register updates below threshold δ, reducing memory writes. Achieves **18.6% energy reduction** on Wikipedia with < 1% accuracy impact.

**3. Adaptive Precision HLL (AP-HLL)**  
Dynamically caps ρ values during high-skew phases. Achieves **18.3% energy reduction** on Wikipedia. Minimal accuracy impact.

**4. Adaptive Controller**  
Monitors stream statistics (update rate, entropy, skew) and tunes parameters at runtime. Achieves **+10.6% additional savings** on diverse heterogeneous streams.

---

## Repository Structure

```
GreenSketch/
├── core/           # HLL implementations (Std, LHLL, THLL, AP-HLL)
├── streams/        # Stream generators (synthetic + real-world loaders)
├── energy/         # Energy cost model and adaptive controller
├── experiments/    # Benchmark, ablation, sensitivity, and visualization scripts
├── utils/          # Hash functions and utilities
├── results/
│   ├── *.csv       # 117 experiment result files
│   └── plots/      # 7 publication-quality visualizations (300 DPI)
├── Data/           # Real-world datasets (Enron, Wikipedia)
├── RESULTS.md      # Complete experimental results (all tables)
├── README.md       # This file
└── requirements.txt
```

---

## Datasets

| Dataset | Source | Size | Unique | Dup Ratio | Use Case |
|---------|--------|------|--------|-----------|----------|
| **Enron Email** | [PACER](https://www.cs.cmu.edu/~enron/) | 100K emails | 2,995 senders | 97% | Edge gateway monitoring |
| **Wikipedia Pageviews** | [Wikimedia](https://dumps.wikimedia.org/other/pageviews/) | 100K views | 90,867 pages | 9% | Regional CDN analytics |

Synthetic streams: 7 types (random, adversarial, sorted, Zipfian, IoT, network, traffic) — validates order-robustness (< 2% variance across orderings).

---

## Experiments

- **117 total experiments** — 59 synthetic + 58 real-world
- **3,500 lines** of code across 18 modules
- **7 visualizations** at 300 DPI
- All experiments deterministic with fixed seed (seed=42)
- Results reproducible within ±2% due to system load

---

## Citation

```bibtex
@software{greensketch2026,
  title   = {GreenSketch: Energy-Aware Order-Robust Cardinality Estimation for Sustainable Edge Streams},
  author  = {Sunil Kumar S},
  year    = {2026},
  url     = {https://github.com/sunilsk17/GreenSketch}
}
```

---

## Author

**Sunil Kumar S**

## License

MIT License
