# E*3 EU AI Act and GDPR Compliance Evidence Statement

Last updated: 2026-04-30

This statement records the repository-level evidence supporting E*3's compliance posture under the EU AI Act and the GDPR for its current intended use: benchmark-based research and engineering analysis of sentence embedding configurations under latency, energy, and task-quality constraints.

This document is not a legal opinion, official certification, CE marking, or substitute for a deployment-specific data protection assessment, conformity assessment, or contract review. It is an auditable evidence pack for the current repository state.

## 1. Compliance Conclusion

Current repository status: conditionally compliant for low-risk benchmark and research use.

The current E*3 repository:

- does not train or place on the market a general-purpose AI model;
- uses third-party Sentence Transformers models as benchmark candidates;
- runs evaluations on public benchmark datasets loaded through Hugging Face `datasets`;
- does not intentionally collect user accounts, contact details, uploaded files, or sensitive personal data;
- does not make decisions about natural persons, allocate resources, rank individuals, identify people, perform biometric processing, or operate in a safety-critical domain;
- records technical artifacts such as Pareto-front CSV files, HTML plots, PDFs, run folders, and CodeCarbon energy estimates.

Based on that scope, E*3 is assessed as a low-risk/minimal-risk AI research and benchmarking tool. A downstream deployment can change this classification. Any use in employment, education, credit, insurance, healthcare, public benefits, law enforcement, migration, biometric identification, critical infrastructure, legal decision support, or other high-stakes contexts requires a separate AI Act and GDPR assessment before use.

## 2. Repository Evidence Reviewed

| Evidence item | Repository location | Compliance relevance |
|---|---|---|
| Project scope and intended use | `README.md` | Describes E*3 as an optimization framework for benchmarking embedding configurations under energy/latency/quality constraints |
| Responsible AI statement | `README.md`, Ethics & Responsible AI section | States benchmark scope, responsible deployment limits, privacy posture, auditability, and lifecycle considerations |
| Governance and reproducibility record | `docs/governance-reproducibility-record.md` | Inventories models, datasets, metrics, limitations, privacy assessment, energy assessment, fairness gaps, and risk classification |
| Dataset loading paths | `pareto-similarity.py`, `pareto-real-time.py`, `pareto-mlx.py`, `pareto-classification.py`, `pareto-clustering.py` | Loads GLUE STS-B, GLUE SST-2, or AG News benchmark datasets; no user-provided corpus ingestion path in the benchmark scripts |
| Model loading paths | same benchmark scripts | Loads third-party Sentence Transformers models; does not train foundation/GPAI models |
| Energy measurement | same benchmark scripts plus `.cc_logs`/run artifacts | Uses CodeCarbon `EmissionsTracker` and reports `energy_kwh` as an optimization objective |
| Output artifacts | `results/`, `runs/` | Stores technical benchmark outputs rather than profiles or decisions about individuals |
| Streamlit dashboard | `master_streamlit_app.py` | Provides local/hosted UI for executing scripts and downloading artifacts; run labels are user-controlled text and should not contain personal data |

## 3. EU AI Act Assessment

Official reference basis: the European Commission states that the AI Act uses a risk-based approach; it entered into force on 2024-08-01 and is generally fully applicable from 2026-08-02, with earlier application for prohibited practices and AI literacy from 2025-02-02 and GPAI obligations from 2025-08-02.

### 3.1 Role Assessment

| Question | Repository answer | Status |
|---|---|---|
| Does E*3 provide or train a GPAI model? | No. E*3 evaluates third-party Sentence Transformers models and optimization configurations. | Not a GPAI model provider for the current repository scope |
| Does E*3 place a high-risk AI system on the EU market? | No evidence in the repository. Current intended use is benchmarking and research, not a regulated high-risk application. | High-risk obligations not triggered by current repository use |
| Does E*3 deploy AI to make decisions about people? | No. It evaluates model/configuration tradeoffs and emits technical benchmark artifacts. | No current automated decision-making about natural persons |
| Does E*3 perform prohibited practices? | No evidence of subliminal manipulation, exploitation of vulnerabilities, social scoring, biometric identification/categorisation, emotion recognition, or predictive policing. | Prohibited-practice risk not identified |
| Does E*3 require transparency disclosures for chatbot/deepfake/emotion/biometric use? | No. It is not a chatbot, deepfake generator, emotion-recognition system, or biometric system. | Limited-risk transparency obligations not triggered by current use |

### 3.2 AI Act Controls Present

| Control area | Current evidence | Status |
|---|---|---|
| Intended-purpose documentation | `README.md`; `docs/governance-reproducibility-record.md` | Present |
| Technical documentation | Benchmark scripts, model/dataset inventory, configuration logs, metric reports | Partially present; run-specific revisions and hardware records must be captured per final audit run |
| Risk management | Risk classification, limitations, misuse statement, escalation rule in governance record | Present for research scope; not sufficient for high-risk deployment |
| Data governance | Dataset inventory and privacy assessment in governance record | Present for benchmark scope |
| Logging and traceability | Pareto CSVs, run folders, CodeCarbon logs, recommended capture commands | Present; complete logs should be archived for each audited run |
| Human oversight | Governance record includes a human review section | Template present |
| Accuracy/robustness/cybersecurity | Accuracy/quality metrics are measured; robustness and cybersecurity are not fully assessed | Partial; expand before production deployment |
| AI literacy | Documentation explains assumptions, limitations, and responsible use | Partial; maintainers/deployers must ensure personnel using the tool understand its limits |

### 3.3 AI Act Residual Gaps

The current repository should not be represented as fully certified under the AI Act. The following items remain deployment-dependent:

- final legal role mapping: provider, deployer, importer, distributor, or product manufacturer;
- final intended-use statement for each deployment;
- high-risk screening against the exact deployment domain and Annex III/product-safety context;
- user-facing instructions and human-oversight procedure for production use;
- post-market monitoring or incident-reporting process if the system is placed on the market;
- security review for hosted deployments;
- reviewer sign-off in `docs/governance-reproducibility-record.md`.

## 4. GDPR Assessment

Official reference basis: GDPR Article 5 sets principles for personal data processing, including lawfulness, fairness and transparency; purpose limitation; data minimisation; accuracy; storage limitation; integrity and confidentiality; and accountability. GDPR Article 6 requires a lawful basis where personal data is processed. The European Commission also explains data protection by design and by default as implementing safeguards early and processing only what is necessary by default.

### 4.1 Personal Data Processing Assessment

| Processing activity | Personal data involved? | Repository evidence | GDPR status |
|---|---|---|---|
| Loading public benchmark text | Possibly. Public text can contain names or references to identifiable persons, especially in news data. | GLUE and AG News dataset loading in benchmark scripts; dataset inventory in governance record | Low risk for benchmark research, but still requires dataset-license and data-protection awareness |
| Model benchmarking | No intentional personal-data collection. Embeddings are computed from benchmark text. | Benchmark scripts process text locally/in execution environment | Compatible with current research purpose if datasets are lawfully accessed |
| Output artifacts | Usually no direct personal data; outputs are model/configuration metrics, plots, and CSVs. | `pareto_solutions.csv`, PDF/HTML plots, CodeCarbon logs | Low risk; verify artifacts before publication |
| Streamlit run label | Potential personal data if a user types personal information into the run label. | `st.text_input("Run label", ...)` in `master_streamlit_app.py` | Users/deployers should avoid personal data in run labels and maintain a privacy notice for hosted deployments |
| Hosted Streamlit deployment | Deployment platform may process IP addresses, logs, cookies, or telemetry outside the repository code. | README mentions hosted app URL; repository code does not define platform controls | Requires deployment-specific privacy notice and processor/subprocessor review |

### 4.2 GDPR Principles Mapping

| GDPR principle | Current control | Evidence | Status |
|---|---|---|---|
| Lawfulness, fairness, transparency | Current use is research/benchmarking on public datasets; documentation identifies sources and limits | README and governance record | Partial; controller must document lawful basis for any deployment that processes personal data |
| Purpose limitation | Intended use limited to benchmark experimentation and model-selection research | README, governance record, this statement | Present for repository scope |
| Data minimisation | Scripts use public benchmark subsets and do not request accounts/uploads | `MAX_*` limits in benchmark scripts; no file uploader path found | Present for repository scope |
| Accuracy | Quality metrics are measured against benchmark labels | Pareto CSVs and metric reports | Present for benchmark metrics, not a claim about downstream correctness |
| Storage limitation | Run artifacts are local technical outputs; no retention schedule is enforced by code | `runs/`, `results/`, `.cc_logs` | Partial; deployments should define retention and deletion policy |
| Integrity and confidentiality | No secrets are committed in the reviewed files; outputs are local artifacts | repository review | Partial; production deployment needs access control, secrets management, dependency scanning, and platform hardening |
| Accountability | Governance and compliance records exist | `docs/governance-reproducibility-record.md`; this document | Present, with human sign-off still needed |
| Data protection by design/default | No default collection of user accounts, uploaded files, or sensitive categories | benchmark scripts and dashboard review | Present for repository scope |

### 4.3 GDPR Operational Requirements Before Deployment

Before using E*3 with proprietary, user-provided, or otherwise personal data, the controller/deployer must complete:

- controller and processor role identification;
- lawful-basis record under GDPR Article 6;
- special-category screening under GDPR Article 9, if relevant;
- Data Protection Impact Assessment screening, and DPIA if processing is likely to create high risk;
- privacy notice covering purposes, data categories, retention, recipients, international transfers, rights, and contact details;
- retention/deletion schedule for run folders, logs, caches, and model/dataset artifacts;
- access-control and security measures for hosted runs;
- data-processing agreements with hosting, analytics, model, dataset, or cloud providers where applicable;
- transfer assessment and safeguards for third-country transfers where applicable;
- procedure for data-subject access, erasure, rectification, objection, restriction, and portability requests where personal data is processed.

## 5. Compliance Commitments for This Repository

The repository should be maintained under the following commitments:

1. Keep E*3's current intended use limited to benchmark experimentation, model-selection research, and engineering analysis unless a new compliance review is completed.
2. Do not add user corpus upload, account management, tracking, analytics, or external telemetry without updating this compliance statement and the privacy assessment.
3. Do not market E*3 as suitable for high-risk or rights-affecting decisions without a deployment-specific AI Act assessment, GDPR assessment, fairness evaluation, human-oversight plan, and security review.
4. Record model revisions, dataset revisions, dependency lock files, hardware details, random seeds, and complete run artifacts for any audited result.
5. Verify third-party model and dataset licenses before redistribution, commercial use, or production deployment.
6. Keep generated run artifacts out of public releases unless reviewed for personal data, secrets, and license-sensitive content.

## 6. Proof Package Checklist

For an audit or grant deliverable, include:

- this document;
- `docs/governance-reproducibility-record.md`;
- the relevant benchmark scripts;
- `README.md` responsible-use and installation sections;
- exact Git commit SHA and dirty-state status;
- dependency lock file from the audited environment;
- timestamped run folder with Pareto CSV, plots, and CodeCarbon logs;
- model and dataset revision identifiers;
- hardware and OS record;
- completed reviewer sign-off table.

## 7. Sign-Off Record

| Field | Value |
|---|---|
| Repository | `efficient-edge-embeddings` |
| Reviewed scope | Benchmark/research use of E*3 only |
| Compliance conclusion | Compliant for low-risk repository scope |
| Conditions | No high-risk or personal-data deployment without reassessment |

## 8. Official Sources

- EUR-Lex, Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- European Commission, AI Act policy page: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- European Commission, Navigating the AI Act FAQ: https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act
- European Commission, General-purpose AI obligations under the AI Act: https://digital-strategy.ec.europa.eu/en/factpages/general-purpose-ai-obligations-under-ai-act
- EUR-Lex, Regulation (EU) 2016/679 (GDPR): https://eur-lex.europa.eu/eli/reg/2016/679/oj
- European Commission, data protection by design and by default: https://commission.europa.eu/law/law-topic/data-protection/rules-business-and-organisations/obligations/what-does-data-protection-design-and-default-mean_en
