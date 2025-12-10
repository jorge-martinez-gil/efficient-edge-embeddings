# E*3: Energy-Efficient Embedding Optimization  
Optimizing Embedding Models for Edge-Class Hardware

<p align="center">
  <img src="logo.png" alt="E³ Logo" width="220"/>
</p>

---

## Overview

E³ provides a practical way to benchmark and optimize embedding models on hardware with limited compute and power budgets.  
The system measures actual inference latency, estimates energy usage, applies quantization settings, and performs a multi-objective evolutionary search to identify strong trade-offs between speed, efficiency, and embedding quality.

Key capabilities include:

- Direct execution of SentenceTransformer models  
- Latency benchmarking through repeated encoding  
- Energy estimation linked to measured latency and device power  
- Quantization-aware accuracy adjustments  
- NSGA-II search over model and quantization choices  
- Extraction and visualization of Pareto-optimal configurations  

---

## Features

- Real inference timing on CPU or GPU  
- Support for several common embedding models  
- Quantization modes: `fp32`, `int8`, `int4`  
- Three optimization goals:  
  - Low energy usage  
  - Low latency  
  - Minimal accuracy impact  
- Caching to avoid repeated timing runs  
- Pareto frontier visualization for fast comparison  

---

## How It Works

### Model loading  
Models are loaded dynamically on first use.

### Latency measurement  
A collection of sentences is encoded multiple times.  
The average latency per encoding pass is computed.

### Energy estimation  
Energy usage is estimated from measured latency and the declared power profile of the device.

### Accuracy modification  
Each quantization option applies a small accuracy shift.  
This can be replaced by a custom evaluation routine.

### Multi-objective search  
NSGA-II evaluates model and quantization combinations.  
Cached timing results are reused to reduce overhead.

### Pareto frontier  
All optimal configurations are listed and visualized.  
The energy–accuracy scatter plot makes trade-offs easy to analyze.

---

## License

MIT License

