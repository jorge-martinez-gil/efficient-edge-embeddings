# E*3 Governance and Reproducibility Record

Last updated: 2026-04-30

This document records the model, dataset, experiment, metric, reproducibility, privacy, energy, fairness, and risk information for E*3 benchmark-based experimentation. It is intended to accompany each benchmark run and should be updated whenever models, datasets, hardware, trial budgets, or selected configurations change.

## 1. Model Inventory

The benchmark uses Sentence Transformers models loaded from the Hugging Face Hub through `sentence-transformers`. The scripts do not currently pin model revisions, so each run should record the resolved Hugging Face commit SHA from the local cache or model card API. The "Version / revision" field below therefore distinguishes between the candidate source and the revision evidence that must be captured for an auditable run.

| Model | Source | Version / revision to record | License status | Intended use in E*3 |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face model card | Record resolved model SHA; current card API reports SHA `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` | Apache-2.0 on current model card | Compact general-purpose sentence embeddings for similarity, classification, clustering, and quantization baselines |
| `sentence-transformers/all-MiniLM-L12-v2` | Hugging Face model card | Record resolved model SHA at run time | Apache-2.0 on current model card | Compact general-purpose sentence embeddings with larger MiniLM encoder |
| `sentence-transformers/all-mpnet-base-v2` | Hugging Face model card | Record resolved model SHA; current card API reports SHA `e8c3b32edf5434bc2275fc9bab85f82640a19130` | Apache-2.0 on current model card | Higher-quality general-purpose baseline for trade-off comparison |
| `sentence-transformers/all-distilroberta-v1` | Hugging Face model card | Record resolved model SHA at run time | Apache-2.0 on current model card | General-purpose embedding baseline with RoBERTa-family encoder |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | Hugging Face model card | Record resolved model SHA at run time | Apache-2.0 on current model card | Compact paraphrase/similarity-oriented embedding baseline |
| `sentence-transformers/paraphrase-mpnet-base-v2` | Hugging Face model card | Record resolved model SHA at run time | Apache-2.0 on current model card | Larger paraphrase/similarity-oriented embedding baseline |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Hugging Face model card | Record resolved model SHA; current card API reports SHA `b207367332321f8e44f96e224ef15bc607f4dbf0` | License not declared in current card metadata; verify before redistribution or production use | Retrieval/QA-oriented compact baseline |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | Hugging Face model card | Record resolved model SHA at run time | License not declared in current card metadata; verify before redistribution or production use | Retrieval/QA-oriented MPNet baseline |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face model card | Record resolved model SHA at run time | Apache-2.0 on current model card | Multilingual coverage probe and cross-lingual robustness baseline |
| `sentence-transformers/msmarco-distilbert-base-v4` | Hugging Face model card | Record resolved model SHA at run time | Apache-2.0 on current model card | MS MARCO retrieval-oriented diversity candidate |
| `sentence-transformers/sentence-t5-base` | Hugging Face model card; listed in `configs/default.yaml` | Record resolved model SHA at run time if used | Apache-2.0 on current model card | Optional sentence similarity baseline; not currently in the embedded script candidate lists |

## 2. Dataset Inventory

| Dataset | Source | Task | Size used by E*3 | Language coverage | License status | Personal-data status |
|---|---|---|---|---|---|---|
| GLUE STS-B (`glue`, `stsb`, validation) | Hugging Face dataset `nyu-mll/glue` | Semantic textual similarity; Pearson correlation against human similarity scores | Up to `MAX_PAIRS = 1500`; validation split is approximately 1.5k pairs | English | Hugging Face card marks GLUE license as `other`; validate task-specific terms before redistribution | Public benchmark text; no sensitive data intentionally processed, but sentence examples may contain ordinary natural-language references |
| GLUE SST-2 (`glue`, `sst2`) | Hugging Face dataset `nyu-mll/glue` | Binary sentiment classification | Script default supports train and validation; `MAX_TRAIN = 1000`, `MAX_EVAL = 500` | English | Hugging Face card marks GLUE license as `other`; validate task-specific terms before redistribution | Public benchmark text; no sensitive data intentionally processed |
| AG News (`ag_news`) | Hugging Face dataset `fancyzhx/ag_news` through `datasets.load_dataset("ag_news")` | Topic classification and unsupervised clustering | Full source split is 120,000 train and 7,600 test rows; E*3 defaults use `MAX_TRAIN = 1000`, `MAX_EVAL = 500`, or `MAX_SAMPLES = 2000` | English | Hugging Face card marks license as `unknown`; original corpus is described for research/non-commercial activity, so confirm terms before deployment or redistribution | Public news text; may contain names of public persons or organizations, but no sensitive personal data is intentionally processed |

## 3. Experiment Configuration Logs

Every run should export or retain a configuration row for each evaluated trial, not only the final Pareto front. At minimum, log the following fields:

| Field | Current values / source |
|---|---|
| Model | `model_name` sampled from each script's `model_candidates` list |
| Batch size | Similarity/classification/clustering/MLX: `[8, 16, 32, 64]`; real-time mode: fixed `1` |
| Normalization flag | `normalize` sampled from `[True, False]` |
| PCA dimension | `target_dim` sampled from `64` to `768` in steps of `32`; actual PCA components are clamped by sample count and native embedding dimension |
| PCA energy flag | `track_pca_energy` sampled from `[False, True]` |
| Quantization settings | MLX script only: `quantize_embeddings`, `quant_backend` in `["raw", "mlx"]`, `quant_bits` in `[4, 8]`, `quant_group` in `[16, 32, 64, 128]` when valid, `quant_mode` in `["symmetric", "affine"]`, `track_quant_energy` |
| Backend | Sentence Transformers/PyTorch by default; raw NumPy groupwise quantization; optional Apple MLX backend when `mlx` is installed |
| Random seed | Classification uses `RANDOM_SEED = 0`; clustering uses deterministic first `MAX_SAMPLES` slice plus `MiniBatchKMeans` configuration; Optuna samplers are not explicitly seeded in the current scripts |
| Trial budget | Similarity: `N_TRIALS = 40`; real-time: `N_TRIALS = 40`; MLX quantization: `N_TRIALS = 40`; classification: `N_TRIALS = 20`; clustering: `N_TRIALS = 3` |

Recommended run log command:

```bash
python pareto-similarity.py
python pareto-real-time.py
python pareto-classification.py
python pareto-clustering.py
python pareto-mlx.py
```

For publication-quality fronts, increase the relevant `N_TRIALS` constant to 200-400 and record the exact modified script or configuration diff.

## 4. Metric Report

Each relevant configuration is evaluated on task quality, latency, energy consumption, and Pareto-front membership. The checked-in run artifacts currently include Pareto-front CSVs for MLX quantization and clustering.

### MLX Quantization Run

Source: `runs/mlx-20260414-150749/pareto_solutions.csv`

| Pareto member | Model | Batch | Normalize | PCA dim | Quantization | Latency ms | Energy kWh | Quality |
|---|---|---:|---|---:|---|---:|---:|---:|
| Yes (`P01`) | `sentence-transformers/paraphrase-MiniLM-L6-v2` | 64 | False | 64 | raw, 8-bit, group 32, affine | 19323.3110 | 0.000708373 | Pearson `0.845173` |
| Yes (`P02`) | `sentence-transformers/paraphrase-MiniLM-L6-v2` | 8 | True | 96 | MLX, 4-bit, group 64, symmetric | 18611.1404 | 0.000682298 | Pearson `0.844431` |
| Yes (`P03`) | `sentence-transformers/all-distilroberta-v1` | 16 | True | 96 | not quantized; raw fields retained | 55484.5869 | 0.002033885 | Pearson `0.853163` |

### Clustering Run

Source: `runs/clustering-20260415-125846/pareto_solutions.csv`

| Pareto member | Model | Batch | Normalize | PCA dim | KMeans batch | Dataset | Latency ms | Energy kWh | Quality |
|---|---|---:|---|---:|---:|---|---:|---:|---:|
| Yes (`P01`) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 64 | False | 736 | 256 | AG News test, 2000 samples | 66560.3903 | 0.002439857 | NMI `0.440373` |
| Yes (`P02`) | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 32 | False | 128 | 512 | AG News test, 2000 samples | 215524.7463 | 0.007900667 | NMI `0.486744` |

### Missing Reports To Generate For Complete Coverage

The repository does not currently contain checked-in Pareto CSVs for semantic similarity without quantization, real-time mode, or classification. Generate and archive those reports before using this record as a complete audit package.

## 5. Reproducibility Package

Each reproducibility package should contain:

| Item | Current repository evidence | Required action per run |
|---|---|---|
| Code version | Git repository working tree | Record `git rev-parse HEAD`; include branch and dirty-state status |
| Dependency versions | `requirements.txt` lists top-level packages: `streamlit`, `optuna`, `sentence-transformers`, `datasets`, `codecarbon`, `plotly`, `matplotlib`, `pandas`, `numpy` | Export `pip freeze` or `conda env export` |
| Hardware details | Not checked into the repository | Record CPU model, GPU/NPU if any, memory, OS, Python version, power mode, and whether Apple MLX was available |
| Commands | Script commands listed above | Save exact command line, working directory, environment variables, and any offline cache settings |
| Seeds | Classification seed is `0`; Optuna and several other paths are not fully seeded | Record script seed constants and any added sampler seeds |
| Exported artifacts | `pareto_solutions.csv`, `pareto_3d_interactive.html`, `fig_pareto3d.pdf`, projection PDFs, CodeCarbon logs | Archive the complete timestamped run folder and `.cc_logs` files |
| Model and dataset revisions | Not pinned in scripts | Record Hugging Face model SHAs and dataset revision/cache fingerprints |

Suggested capture commands:

```bash
git rev-parse HEAD
git status --short
python --version
pip freeze > requirements-lock.txt
python -c "import platform; print(platform.platform()); print(platform.processor())"
```

## 6. Limitations Statement

Known limits:

- Dataset coverage is narrow. Current tasks are English benchmark datasets, with multilingual coverage only indirectly probed through one multilingual model candidate unless additional multilingual datasets are added.
- Hardware coverage is run-specific. Results from a laptop, workstation, Apple Silicon device, cloud VM, or edge device are not interchangeable.
- Energy estimates rely on CodeCarbon software estimation unless validated externally. These values are suitable for relative comparison within a controlled run, not as calibrated metrology by default.
- Optuna samplers are not explicitly seeded in the current scripts, so trial order and discovered fronts may vary between runs.
- Trial budgets are small for some scripts, especially clustering with `N_TRIALS = 3`; use 200-400 trials for stable publication-quality Pareto fronts.
- Benchmarks use subsampling limits (`MAX_TRAIN`, `MAX_EVAL`, `MAX_SAMPLES`, `MAX_PAIRS`) that improve runtime but may understate variance and dataset heterogeneity.
- Pareto membership is conditional on the evaluated candidate set, hardware, backend, and sampled search budget; it is not a universal property of a model.

## 7. Fairness Report

Current status: no dedicated subgroup fairness evaluation is implemented in the checked-in scripts.

Recommended checks before deployment:

- Subgroup evaluation: measure task quality across demographic, dialectal, or named-entity groups where labels and policy permit.
- Multilingual evaluation: add multilingual sentence similarity or retrieval datasets when using `paraphrase-multilingual-MiniLM-L12-v2` or deploying outside English-only settings.
- Domain-specific evaluation: evaluate separately for news, conversational text, legal, biomedical, educational, or other target domains.
- Robustness checks: compare selected configurations across random seeds, sample slices, and hardware targets.

Fairness conclusion for the current benchmark-only package: fairness risk is not fully characterized. The package should not be used to support claims of subgroup parity without additional evaluation.

## 8. Privacy Assessment

The current E*3 experiments process public benchmark text from Hugging Face datasets. They do not collect user input, store user accounts, or intentionally process sensitive personal data. For the current benchmark-based experimentation, privacy risk is low.

Residual privacy considerations:

- Public datasets may include names or references to public figures, organizations, or events.
- Local model and dataset caches may store benchmark text and model files.
- If E*3 is adapted to proprietary or user-provided corpora, a new privacy assessment is required, including data minimization, retention, access control, and lawful-basis review.

## 9. Energy Assessment

Energy is estimated with CodeCarbon through `EmissionsTracker`, using `.cc_logs` as the default log directory. The reported `energy_kwh` column is used as one of the three optimization objectives.

Assessment status:

- Current checked-in metric CSVs include CodeCarbon-derived `energy_kwh` values.
- External power-meter validation is not present in the repository.
- External-meter validation is recommended when reporting absolute energy values, comparing different hardware classes, or making procurement/deployment claims.

When external validation is required, record:

- Meter model and calibration status.
- Measurement boundary, including whether display, charger, idle baseline, and peripheral loads are included.
- Idle baseline subtraction method.
- Number of repeated runs and confidence interval.
- Difference between CodeCarbon estimate and external-meter measurement.

## 10. Human Review Record

| Field | Record |
|---|---|
| Reviewer name or role | TBD |
| Approval date | TBD |
| Selected configuration | TBD |
| Reason for selection | TBD; should reference quality, latency, energy, Pareto membership, hardware target, and task requirements |
| Conditions or safeguards | TBD |

No configuration should be presented as finally approved until this section is completed.

## 11. Risk Classification

Intended use: low-risk benchmark and research tooling for selecting efficient embedding configurations under quality, latency, and energy constraints.

Classification for current use: low-risk, because the repository performs benchmark evaluation and does not make decisions about people, allocate resources, rank individuals, perform biometric identification, or operate in a safety-critical context.

Escalation rule: if E*3 outputs are used in a downstream system for employment, education access, credit, insurance, healthcare, law enforcement, migration, legal assessment, surveillance, biometric processing, or other regulated high-stakes settings, the downstream use may become limited-risk, high-risk, or prohibited under applicable regulation. A separate legal, safety, fairness, privacy, and human-oversight review is required before such use.

## 12. Misuse and Restricted-Use Statement

E*3 is not intended for high-stakes use without further safeguards. It must not be used as the sole basis for decisions that affect a person's rights, opportunities, eligibility, safety, liberty, health, financial status, or access to essential services.

Restricted uses include, without limitation:

- Automated hiring, promotion, disciplinary, or termination decisions.
- Credit, insurance, housing, healthcare, education, or public-benefit eligibility decisions.
- Legal, law-enforcement, surveillance, biometric, or migration-control decisions.
- Any application where degraded embedding quality, biased representations, or unvalidated energy/latency measurements could cause material harm.

Permitted current use is benchmark experimentation, model-selection research, and engineering analysis, subject to dataset licenses and reproducibility requirements.

## 13. Sources Checked

- E*3 repository files: `README.md`, `configs/default.yaml`, `pareto-similarity.py`, `pareto-real-time.py`, `pareto-classification.py`, `pareto-clustering.py`, `pareto-mlx.py`, checked-in `runs/*/pareto_solutions.csv`.
- Hugging Face model cards/API for Sentence Transformers candidates, including [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [`all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2), [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [`all-distilroberta-v1`](https://huggingface.co/sentence-transformers/all-distilroberta-v1), [`paraphrase-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2), [`paraphrase-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2), [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1), [`multi-qa-mpnet-base-dot-v1`](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1), [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [`msmarco-distilbert-base-v4`](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v4), and [`sentence-t5-base`](https://huggingface.co/sentence-transformers/sentence-t5-base).
- Hugging Face dataset cards for [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) and [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news).
- [CodeCarbon project documentation](https://codecarbon.io/) for energy-estimation context.
