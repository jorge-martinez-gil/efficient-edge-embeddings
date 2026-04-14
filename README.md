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

**E*3** is a framework for benchmarking and optimizing embedding models under strict computational and energy constraints. It is designed for scenarios where latency and power consumption are not secondary concerns but primary deployment limits.

Instead of treating model selection as a single-objective problem, E*3 measures real inference latency and estimates energy consumption, combining them with task performance in a multi-objective search. This makes it possible to identify configurations that offer strong accuracy without incurring unnecessary computational cost.

---

## 🔍 Goals

Most embedding benchmarks still focus on accuracy in isolation. This leads to choices that are difficult to justify in real systems, especially on edge hardware.

E*3 treats embedding selection as a trade-off between three competing factors: task performance, inference latency, and energy consumption. The objective is not to maximize a single metric, but to identify configurations that are efficient and defensible from a systems perspective.

Pareto dominance is used to separate viable configurations from those that are strictly worse. A knee-point rule then identifies a single recommendation when no external preference is available.

---

## 🏗️ Configuration Space

Each embedding pipeline is described by a configuration tuple:

```

θ = (m, d, b, n, p, c, q, q_b, q_bits, q_g, q_m, q_E)

```

This representation captures model choice, dimensionality, batching, normalization, projection, downstream task, and quantization settings. The goal is to expose all relevant deployment levers instead of treating the embedding model as a fixed artifact.

---

## 🎯 Key Results

### Semantic Similarity (STSb)

```

Low-Cost         Knee-Point        High-Performance
─────────────    ──────────────    ──────────────────
P05: r=0.870     P10: r=0.878      P02: r=0.883
~16s / ~2095 J   ~31s / ~4079 J    ~49s / ~6512 J

````

The results show a clear transition point. Moving beyond the knee point roughly doubles latency and energy, while the gain in correlation remains marginal.

### Text Classification (AG News)

A similar pattern appears in classification. A modest improvement in accuracy can require orders of magnitude more latency and energy, making some configurations difficult to justify outside controlled environments.

### Text Clustering (AG News — NMI)

Clustering results reinforce the same observation. Higher scores come at a steep cost, and the trade-off is not always favorable in practical deployments.

---

## 🚀 Quick Start

```bash
git clone https://github.com/jorge-martinez-gil/efficient-edge-embeddings.git
cd efficient-edge-embeddings
pip install -r requirements.txt
````

---

## 📐 The Knee-Point Rule

When no explicit preference is available, E*3 derives a recommendation directly from the Pareto frontier. Latency and energy are normalized, combined into a single cost, and evaluated against performance gains between neighboring configurations. The selected point corresponds to the largest improvement per unit of cost, marking the transition between efficient and diminishing-return regimes.

---

## 🏛️ Design Implications

The results suggest that many commonly used configurations are unnecessarily expensive. In practice, normalization tends to stabilize downstream behavior with minimal overhead, and dimensionality should be treated as a controllable parameter rather than a fixed property. Batch size also plays a central role, shifting the balance between latency and energy depending on hardware utilization. Larger encoders only become justified at the very top end of the performance spectrum.

---

## ⚖️ Ethics & Responsible AI

> This project aligns with EU Trustworthy AI principles, with attention to bias, energy efficiency, transparency, and responsible deployment.
> A detailed statement is provided below.

<details>
<summary><strong>📄 Full Ethics & Governance Statement (click to expand)</strong></summary>

### Ethical Scope

E*3 operates at the level of representation learning, yet its outputs may influence downstream decisions. For this reason, evaluation should not stop at accuracy or efficiency metrics. The behavior of embeddings must be considered in relation to the context in which they are applied.

### Risks and Limitations

Embedding models may encode biases inherited from their training data, and optimization for efficiency does not remove this issue. There is also a risk that optimized configurations are reused in sensitive applications without appropriate validation. In addition, aggressive efficiency settings can reduce representational fidelity, which may lead to misleading results. While the framework reduces inference-time energy consumption, the optimization process itself still requires computational resources.

### Mitigation Approach

E*3 addresses these concerns through its design. Performance is evaluated together with latency and energy, discouraging wasteful configurations. Pareto filtering removes options that are strictly inefficient, and all configurations are reported transparently. At the same time, the framework assumes that fairness and domain-specific validation are the responsibility of the user.

### Bias and Fairness Evaluation

Users are encouraged to assess embeddings explicitly before deployment. This can include similarity-based bias tests, comparisons across demographic or linguistic groups, and sensitivity analysis. The framework can be extended to incorporate fairness metrics as additional optimization objectives when required.

### Environmental Responsibility

A central objective of E*3 is to reduce energy consumption at inference time. The results show that strong performance can often be maintained while significantly lowering resource usage, which supports more sustainable deployment practices, especially on edge devices.

### Data and Privacy

The framework operates on standard benchmark datasets and does not require personal or sensitive data. It does not collect or store user information.

### Reproducibility and Auditability

All experiments are defined through explicit configuration tuples and a documented evaluation pipeline. The implementation is open, allowing independent verification and external auditing of results.

### Responsible Use and Boundaries

E*3 is not intended for use in high-stakes decision-making scenarios without additional safeguards. Applications in areas such as hiring, credit scoring, legal assessment, or surveillance require domain-specific validation and oversight. The framework provides recommendations, not decisions, and responsibility for deployment remains with the user.

### Governance and Lifecycle

Responsible use extends beyond initial selection. Models should be evaluated before deployment, monitored during operation, and reassessed over time to account for performance drift or changing requirements. Compliance with applicable regulations, including GDPR and the AI Act, must be ensured by the deploying party.

### Alignment with EU Principles

The project supports key elements of Trustworthy AI, including human oversight, technical reliability, transparency, and attention to environmental impact.

</details>

---

## 🙏 Acknowledgments

Research supported via the **Efficient Edge Embeddings (E*3)** project, subgrant 2dAI2OC07 under EU Horizon Europe grant 101120726 ([dAIEDGE](https://daiedge.eu)). Validated using the [vLab](https://vlab.daiedge.eu) environment. Special thanks to Giulio Gambardella for contributions.

---

## 📜 License

This project is licensed under the **MIT License**.


