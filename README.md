# E*3: Energy-Efficient Embedding Optimization
### Optimizing Embedding Models for Edge-Class Hardware

<p align="center">
  <img src="logo.png" alt="E*3 Logo" width="250" style="border-radius: 10px;"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square" alt="Python Version" />
  <img src="https://img.shields.io/badge/hardware-Edge%20%7C%20CPU%20%7C%20GPU-orange?style=flat-square" alt="Hardware Support" />
  <img src="https://img.shields.io/badge/search-NSGA--II-purple?style=flat-square" alt="Algorithm" />
</p>

---

## 📖 Overview

**E*3** provides a practical framework to benchmark and optimize embedding models specifically for hardware with limited compute and power budgets. 

The rationale behind measuring actual inference latency and estimating energy usage is to apply a **multi-objective evolutionary search** to identify the strongest trade-offs between speed, efficiency, and embedding quality.

---

## 🔍 Goals

Most embedding benchmarks rank models by **accuracy alone**, ignoring the latency and energy costs that dominate real-world deployments. This project reframes embedding selection as a **multi-objective optimization problem** over three axes:

| Objective | Direction | Why It Matters |
|-----------|-----------|----------------|
| **Task Performance** | ↑ Maximize | Similarity, classification, or clustering quality |
| **Inference Latency** | ↓ Minimize | Interactive services have strict time budgets |
| **Energy Consumption** | ↓ Minimize | Operational cost and sustainability on edge devices |

Using **Pareto dominance**, we identify which configurations are admissible and which are strictly dominated. A weight-free **knee-point rule** then recommends a single operating point without requiring subjective preferences.

> Many popular embedding setups are wasteful. Near-optimal accuracy is often achievable at a fraction of the resource cost.

---

## 🏗️ Configuration Space

Each embedding pipeline is defined by a configuration tuple:

```

θ = (m, d, b, n, p, c, q, q_b, q_bits, q_g, q_m, q_E)

```

| Parameter | Description | Examples |
|-----------|-------------|---------|
| `m` | Embedding model | `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, … |
| `d` | Effective dimensionality | 64 – 768 |
| `b` | Batch size | Variable |
| `n` | L2 normalization | on / off |
| `p` | PCA projection | on / off |
| `c` | Downstream module | Logistic Regression, Linear SVM, k-Means, Agglomerative |
| `q` | Quantization | 4-bit / 8-bit, symmetric / affine |

---

## 🎯 Key Results

### Semantic Similarity (STSb)

The Pareto frontier reveals three distinct operating regimes:

```

Low-Cost         Knee-Point        High-Performance
─────────────    ──────────────    ──────────────────
P05: r=0.870     P10: r=0.878      P02: r=0.883
~16s / ~2095 J   ~31s / ~4079 J    ~49s / ~6512 J

````

Moving from the knee point to the top performer doubles latency and energy for a marginal +0.005 gain in Pearson correlation.

### Text Classification (AG News)

| Config | Accuracy | Latency | Energy |
|--------|----------|---------|--------|
| P02 (budget) | 0.878 | 65 ms | 7.2 J |
| P01 (balanced) | 0.888 | 218 ms | 28.8 J |
| P04 (max perf.) | 0.894 | 18,168 ms | 2,397.6 J |

P04 gains +1.6 pp accuracy over P02 but costs 280× more latency and 333× more energy.

### Text Clustering (AG News — NMI)

| Config | NMI | Latency | Energy |
|--------|-----|---------|--------|
| P01 | 0.459 | 33.2 s | 4,384.8 J |
| P02 | 0.557 | 109.2 s | 14,414.4 J |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/jorge-martinez-gil/efficient-edge-embeddings.git
cd efficient-edge-embeddings

# Install dependencies
pip install -r requirements.txt
````

---

## 📐 The Knee-Point Rule

When stakeholder preferences are unavailable, a weight-free recommendation is derived:

1. Min–max normalize latency and energy across all Pareto-optimal points
2. Compute combined cost: `C(θ) = 0.5 · L̃(θ) + 0.5 · Ẽ(θ)`
3. Sort by increasing `C(θ)` and compute discrete gain
4. Select the configuration with the largest gain

This identifies the transition between efficient and diminishing-return regimes without subjective weighting.

---

## 🏛️ Design Implications

* Dominated configurations should be excluded
* Normalization stabilizes downstream behavior at negligible overhead
* Dimensionality is a deployment knob, not a fixed model property
* Batch size affects the latency–energy trade-off
* Large encoders are justified only at the top accuracy range

---

## ⚖️ Ethics & Responsible AI

> This project follows EU Trustworthy AI principles, with explicit attention to **bias, energy efficiency, and responsible deployment**.
> See the detailed ethics statement below.

<details>
<summary><strong>📄 Full Ethics Statement (click to expand)</strong></summary>

### Ethical Scope

E*3 optimizes embedding models for efficiency. These models may be used in downstream applications that influence decisions, so their behavior must be assessed beyond performance metrics.

---

### 🔍 Risk Assessment

* **Bias propagation**
  Embedding models may reflect biases present in training data

* **Misuse in downstream tasks**
  Optimized models may be applied in sensitive domains without proper validation

* **Over-optimization trade-offs**
  Efficiency-focused configurations may reduce representation quality

* **Environmental trade-offs**
  Benchmarking and optimization still consume computational resources

---

### 🛡️ Mitigation Measures

* Multi-objective evaluation across performance, latency, and energy
* Pareto-based filtering to remove inefficient configurations
* Transparent reporting of all configurations and results
* Recommendation to validate fairness in downstream applications

---

### 🌱 Environmental Responsibility

* Focus on reducing inference-time energy consumption
* Promotion of resource-aware deployment strategies
* Evidence that strong performance can be achieved with lower energy usage

---

### 🔐 Data & Privacy

* Uses standard benchmark datasets
* No personal or sensitive data is required
* No user data is collected or stored

---

### 📊 Reproducibility & Transparency

* Explicit configuration tuples `θ`
* Fully documented evaluation pipeline
* Open-source implementation

Supports verification, comparison, and reuse.

---

### ⚠️ Responsible Use

Users should:

* Avoid deployment in high-stakes decision-making without safeguards
* Evaluate bias and fairness in their domain
* Include human oversight where needed

---

### 🇪🇺 Alignment with EU Guidelines

Aligned with key principles of Trustworthy AI:

* Human agency
* Technical robustness
* Transparency
* Societal and environmental well-being

</details>

---

## 🙏 Acknowledgments

Research supported via the **Efficient Edge Embeddings (E*3)** project, subgrant 2dAI2OC07 under EU Horizon Europe grant 101120726 ([dAIEDGE](https://daiedge.eu)). Validated using the [vLab](https://vlab.daiedge.eu) environment. Special thanks to Giulio Gambardella for his contributions.

---

## 📜 License

This project is licensed under the **MIT License**.
