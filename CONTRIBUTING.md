# Contributing to E*3: Efficient Edge Embeddings

Thank you for your interest in contributing! This guide explains how to report issues, propose features, set up a development environment, and submit pull requests.

---

## Table of Contents

1. [Reporting Issues](#reporting-issues)
2. [Proposing New Features](#proposing-new-features)
3. [Development Setup](#development-setup)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Adding a New Optimization Module](#adding-a-new-optimization-module)
6. [Running Experiments and Validating Outputs](#running-experiments-and-validating-outputs)
7. [Pull Request Guidelines](#pull-request-guidelines)

---

## Reporting Issues

Before opening an issue, please:

1. Search the [existing issues](https://github.com/jorge-martinez-gil/efficient-edge-embeddings/issues) to avoid duplicates.
2. Check the [FAQ / Troubleshooting section](README.md#-faq--troubleshooting) in the README for known problems.

When filing a bug report, include:

- **Python version** (`python --version`)
- **Operating system and hardware** (e.g., Ubuntu 22.04, Apple M2)
- **Dependency versions** (`pip freeze`)
- **Minimal reproduction script** — the smallest code or command that triggers the bug
- **Full traceback** — paste the complete error output

Use the issue template on GitHub when available.

---

## Proposing New Features

Open a GitHub issue with the label `enhancement` and describe:

- **Motivation** — what problem does this feature solve?
- **Proposed interface** — how would a user interact with it?
- **Relationship to existing modules** — does it extend a current `pareto-*.py` script or add a new one?
- **Scope** — is it a small addition (new model, new task) or a larger structural change?

For significant changes, please open an issue for discussion before submitting a pull request.

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/efficient-edge-embeddings.git
cd efficient-edge-embeddings

# 2. Create a feature branch
git checkout -b feature/my-new-feature

# 3. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 4. Install all dependencies
pip install -r requirements.txt

# 5. (Optional) Apple Silicon quantization backend
pip install mlx

# 6. Verify the environment is working
python -c "import optuna, sentence_transformers, datasets, codecarbon; print('OK')"
```

---

## Code Style Guidelines

This project follows **PEP 8** with a few practical notes:

- **Line length**: 100 characters maximum (relaxed from the 79-character PEP 8 default, to accommodate readable data structures).
- **Type hints**: use them on all new functions and methods.
- **Docstrings**: every module, class, and public function must have a docstring (see [Adding a New Optimization Module](#adding-a-new-optimization-module) for the expected format).
- **Imports**: standard library first, then third-party, then local — separated by blank lines.
- **Constants**: ALL_CAPS at module level; group them under a `# Config` section at the top of the script.
- **No silent failures**: prefer raising descriptive exceptions over returning sentinel values.

You can check for style issues using `flake8` (install separately with `pip install flake8`):

```bash
flake8 pareto-similarity.py --max-line-length=100
```

---

## Adding a New Optimization Module

New optimization modules should follow the pattern established by the existing `pareto-*.py` scripts. Here is the step-by-step checklist:

### 1. Module-level docstring

Start the file with a docstring that describes the task, objectives, outputs, and installation requirements:

```python
"""
Short description of the task and the three objectives.

Usage:
    python pareto-<task>.py

Outputs:
    - pareto_solutions.csv
    - pareto_3d_interactive.html
    - fig_pareto3d.pdf
    - fig_proj_<x>_<y>.pdf

Install:
    pip install optuna sentence-transformers datasets scikit-learn codecarbon plotly matplotlib
"""
```

### 2. Configuration constants

Define all tunable constants at the top of the script under a `# Config` comment block:

```python
# -----------------------------
# Config
# -----------------------------
N_TRIALS = 40
MAX_SAMPLES = 2000
WARMUP_SAMPLES = 128
CC_DIR = ".cc_logs"
```

### 3. Evaluator dataclass

Create a `@dataclass` named `<Task>BenchmarkEvaluator` with:

- Fields for all dataset/configuration parameters, with type hints and default values
- A `__post_init__` method that creates the CodeCarbon log directory and loads the dataset
- A private `_get_model()` method for cached model loading
- An `evaluate()` method with full `Args` / `Returns` docstring

```python
@dataclass
class MyTaskEvaluator:
    """Evaluator for <task> benchmarking with CodeCarbon energy tracking.

    Attributes:
        split: Dataset split to use.
        max_samples: Maximum number of samples per trial.
        warmup_samples: Samples used for warm-up (not measured).
        track_dir: Directory for CodeCarbon log files.
    """
    split: str = "test"
    max_samples: int = 2000
    warmup_samples: int = 128
    track_dir: str = ".cc_logs"

    def __post_init__(self):
        os.makedirs(self.track_dir, exist_ok=True)
        # ... load dataset ...

    def evaluate(self, model_name: str, ...) -> Tuple[float, float, float]:
        """Evaluate a configuration and return (latency_ms, energy_kwh, metric).

        Args:
            model_name: HuggingFace sentence-transformers model identifier.
            ...

        Returns:
            A tuple of (latency_ms, energy_kwh, task_metric).
        """
```

### 4. Optuna study

Use `directions=["minimize", "minimize", "maximize"]` (latency, energy, task metric):

```python
study = optuna.create_study(directions=["minimize", "minimize", "maximize"])
study.optimize(objective, n_trials=N_TRIALS)
pareto = study.best_trials
```

### 5. Export functions

Implement `export_pareto_csv()` and `plot_pareto_interactive_html()` following the conventions in existing scripts. Reuse `configure_matplotlib_for_paper()` and `plot_paper_figures_matplotlib()` as-is or adapt for task-specific axis labels.

### 6. `main()` entrypoint

Wrap everything in a `main()` function and guard execution with `if __name__ == "__main__": main()`.

### 7. Register in `master_streamlit_app.py`

Add an entry to the `MODULES` dictionary in `master_streamlit_app.py`:

```python
"My New Task · Pareto MOO": {
    "key": "my_task",
    "path": BASE_DIR / "pareto-my-task.py",
    "artifacts": [
        "pareto_solutions.csv",
        "pareto_3d_interactive.html",
        "fig_pareto3d.pdf",
        "fig_proj_latency_metric.pdf",
        "fig_proj_energy_metric.pdf",
    ],
    "about": "Short description of the module.",
},
```

---

## Running Experiments and Validating Outputs

After making changes, validate your module by running it with a small trial count:

```bash
# Quick smoke test (should complete in a few minutes)
python pareto-<task>.py  # with N_TRIALS set to 3–5

# Check outputs
ls pareto_solutions.csv pareto_3d_interactive.html fig_pareto3d.pdf
```

Verify that:
- `pareto_solutions.csv` is a well-formed CSV with the expected columns
- `pareto_3d_interactive.html` opens in a browser and displays a 3D scatter plot
- The PDF figures can be opened and are not blank or corrupted

For the Streamlit app:

```bash
streamlit run master_streamlit_app.py
# Navigate to the new module in the sidebar and run it with N_TRIALS=3
```

---

## Pull Request Guidelines

1. **One pull request per feature or fix** — keep PRs focused and reviewable.
2. **Branch naming**: `feature/<short-description>`, `fix/<short-description>`, or `docs/<short-description>`.
3. **Commit messages**: use the imperative mood and be descriptive (e.g., `Add NMI-based clustering evaluator` rather than `update stuff`).
4. **Update documentation**: if your change affects how users run the scripts or interpret outputs, update the relevant sections in `README.md`.
5. **No breaking changes without discussion**: if your change alters the CSV schema, the Pareto output format, or the Streamlit app interface, open an issue for discussion first.
6. **Respond to review comments** within a reasonable time. If a discussion stalls, the PR may be closed and re-opened once ready.

Thank you for contributing to E\*3!
