# GreenSketch: Experimental Results

All results are reproducible. See [README.md](README.md) for setup instructions and [results/](results/) for raw CSV data.

---

## 1. Wikipedia Pageviews

**Dataset**: 50,000 items, ~47K unique pages, 9.1% duplicates  
**Precision**: p=14 (16,384 registers)

| Variant | Error (%) | Energy Proxy | vs Baseline |
|---------|-----------|--------------|-------------|
| Std HLL | 0.51 | 16,444,560 | — |
| LHLL | 14.47 | 14,019,920 | −14.7% |
| THLL | 0.51 | 13,387,980 | −18.6% |
| THLL + Controller | 0.51 | 14,146,990 | −14.0% |
| AP-HLL | 0.51 | 13,428,430 | −18.3% |
| AP-HLL + Controller | 0.51 | 14,360,960 | −12.7% |

Note: LHLL is excluded from accuracy-sensitive comparisons due to 14% error.

---

## 2. Enron Email Corpus

**Dataset**: 100,000 emails, 2,995 unique senders, 97% duplicates  
**Precision**: p=10 (1,024 registers)

| Variant | Error (%) | Energy Proxy | vs Baseline |
|---------|-----------|--------------|-------------|
| Std HLL | 9.20 | 12,462,790 | — |
| LHLL | 9.20 | 14,381,460 | +15.4% |
| THLL | 9.20 | 12,317,620 | −1.2% |
| THLL + Controller | 9.20 | 13,177,760 | +5.7% |
| AP-HLL | 9.20 | 12,380,150 | −0.7% |
| AP-HLL + Controller | 9.20 | 13,163,350 | +5.6% |

Note: Modest savings are expected on high-skew, single-source uniform streams. Controller overhead decreases with stream size (11% at 25K → 6% at 100K items).

---

## 3. Synthetic Benchmark — Order Robustness

**Configuration**: n_unique=5,000, n_total=25,000, p=10  
**7 stream types**: Random, Sorted Hash, Adversarial, Zipfian, IoT Devices, Network Flows, Traffic Sensors

| Stream Type | Std HLL | LHLL | THLL | AP-HLL |
|-------------|---------|------|------|--------|
| Random | 9.65% | 9.65% | 9.65% | 9.65% |
| Sorted Hash | 9.00% | 9.00% | 9.00% | 9.00% |
| Adversarial | 9.00% | 9.00% | 9.00% | 9.00% |
| Zipfian | 0.10% | 0.10% | 0.10% | 0.10% |
| IoT Devices | 2.72% | 0.03% | 2.72% | 2.72% |
| Network Flows | 6.67% | 6.67% | 6.67% | 6.67% |
| Traffic Sensors | 7.90% | 7.90% | 7.90% | 7.90% |
| **Average** | **6.43%** | **6.05%** | **6.43%** | **6.43%** |

Accuracy is consistent across orderings; error scales with cardinality, not stream order.

---

## 4. Ablation Study — Adaptive Controller

**Configuration**: 3 stream types (Random, Adversarial, Bursty), n_unique=5,000, n_total=25,000

| Variant | Controller | Error (%) | Energy Proxy | vs Baseline | Controller Δ |
|---------|------------|-----------|--------------|-------------|--------------|
| Std HLL | — | 6.25 | 18,954,507 | — | — |
| THLL | No | 6.25 | 14,818,133 | −21.8% | — |
| THLL | Yes | 6.25 | 13,245,373 | −30.1% | −10.6% |
| AP-HLL | No | 6.25 | 11,876,627 | −37.3% | — |
| AP-HLL | Yes | 6.25 | 12,618,467 | −33.4% | +6.2% |

The adaptive controller improves THLL energy by 10.6% on diverse synthetic streams.

---

## 5. Sensitivity — Precision Parameter (p)

**Stream**: Random, n_unique=5,000

| p | Registers | Std HLL Error | THLL Error | Std HLL Energy | THLL Energy | Memory |
|---|-----------|---------------|------------|----------------|-------------|--------|
| 8 | 256 | 100.00% | 100.00% | 31,342,120 | 20,931,680 | 0.25 KB |
| 10 | 1,024 | 9.65% | 9.65% | 15,205,720 | 15,230,160 | 1.0 KB |
| 12 | 4,096 | 2.22% | 2.22% | 13,338,440 | 11,982,680 | 4.0 KB |
| 14 | 16,384 | 0.19% | 0.19% | 11,995,080 | 12,080,240 | 16.0 KB |

p=12 offers a practical energy–accuracy–memory tradeoff for edge deployments.

---

## 6. Sensitivity — THLL Delta Threshold (δ)

**Stream**: Random, p=10

| δ | Register Updates/M | Energy Proxy | vs δ=0 |
|---|---------------------|--------------|--------|
| 0 | 74,920 | 11,967,720 | — |
| 1 | 60,560 | 11,905,320 | −0.5% |
| 2 | 51,760 | 11,839,280 | −1.1% |
| 3 | 46,800 | 11,819,040 | −1.2% |

---

## 7. Sensitivity — AP-HLL Rho Cap

**Stream**: Bursty Zipfian, p=10

| Rho Cap | Energy Proxy | Error (%) | Capped Ops/M |
|---------|--------------|-----------|--------------|
| 16 | 11,798,000 | 0.10% | ~15,000 |
| 32 | 11,723,560 | 0.10% | ~8,000 |
| 48 | 11,713,680 | 0.10% | ~3,000 |
| 64 | 11,797,600 | 0.10% | ~500 |

---

## 8. Order Robustness — Real-World Data

| Dataset | Ordering | Unique Items | Error (%) |
|---------|----------|--------------|-----------|
| Enron Email | Chronological | 1,100 | 7.41 |
| Enron Email | Random | 2,187 | 1.57 |
| Enron Email | Grouped | 728 | 1.12 |
| Wikipedia | Chronological | 46,595 | 0.88 |
| Wikipedia | Random | 47,546 | 0.14 |

Error varies with cardinality across subset sizes, not stream ordering.

---

## Energy Cost Model

```
E_proxy = α × full_hash_calls + β × cheap_hash_calls + γ × register_updates + δ × cpu_time
```

| Parameter | Value | Operation |
|-----------|-------|-----------|
| α | 10.0 | Full 64-bit hash |
| β | 1.0 | Cheap 32-bit hash |
| γ | 2.0 | Register memory write |
| δ | 1×10⁶ | CPU time (normalized) |

---

## Result Files

| File | Description | Experiments |
|------|-------------|-------------|
| `results/benchmark_results.csv` | Synthetic benchmark | 28 |
| `results/ablation_results.csv` | Controller ablation | 15 |
| `results/sensitivity_precision.csv` | Precision sweep | 8 |
| `results/sensitivity_delta.csv` | THLL delta sweep | 4 |
| `results/sensitivity_rho_cap.csv` | AP-HLL rho cap sweep | 4 |
| `results/real_world_benchmark_results.csv` | Real-world benchmark (p=10) | 36 |
| `results/wikipedia_p14_FINAL.csv` | Wikipedia p=14 | 12 |
| `results/enron_scale_comparison_FINAL.csv` | Enron 25K vs 100K | 10 |
| **Total** | | **117** |
