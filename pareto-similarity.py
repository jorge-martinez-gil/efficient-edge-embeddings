"""
STS-B 3-objective MOO (latency, energy, accuracy) + paper-ready plots.

Outputs:
- pareto_solutions.csv
- pareto_3d_interactive.html (offline, no CDN, linear axes)
- fig_pareto3d.pdf           (vector PDF for LaTeX)
- fig_proj_latency_accuracy.pdf
- fig_proj_energy_accuracy.pdf

Install:
  pip install optuna sentence-transformers datasets scikit-learn codecarbon plotly matplotlib
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import optuna
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from codecarbon import EmissionsTracker

import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Config
# -----------------------------
N_TRIALS = 40              # 200-400 gives smoother Pareto sets
MAX_PAIRS = 1500            # None -> full STS-B validation
WARMUP_PAIRS = 128
CC_DIR = ".cc_logs"

# Paper figure sizing (LNCS ~ 12.2cm text width; one-column ~ 6cm-ish)
# Here we use ~6.8in wide for a nice full-width figure in LNCS.
PAPER_FIGSIZE_WIDE = (6.8, 4.2)   # inches
PAPER_FIGSIZE_3D = (6.8, 4.8)     # inches

# -----------------------------
# Helpers
# -----------------------------
def short_model(name: str) -> str:
    return (
        name.replace("sentence-transformers/", "")
            .replace("all-", "")
            .replace("-v2", "")
    )


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + 1e-12
    return float((x * y).sum() / denom)


def norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)

def load_dataset_or_raise(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except Exception:
        raise RuntimeError(
            "Failed to load dataset from Hugging Face. "
            "Check internet/proxy access (for example HTTP(S)_PROXY settings) "
            "or run in offline mode with a pre-populated HF cache."
        ) from None


# -----------------------------
# Evaluator
# -----------------------------
@dataclass
class STSBenchmarkEvaluator:
    split: str = "validation"
    max_pairs: int = 1500
    warmup_pairs: int = 128
    track_dir: str = ".cc_logs"

    def __post_init__(self):
        os.makedirs(self.track_dir, exist_ok=True)

        ds = load_dataset_or_raise("glue", "stsb", split=self.split)
        if self.max_pairs is not None:
            ds = ds.select(range(min(self.max_pairs, len(ds))))

        self.s1 = ds["sentence1"]
        self.s2 = ds["sentence2"]
        self.labels = np.array(ds["label"], dtype=float)  # [0,5]
        self._model_cache: Dict[str, SentenceTransformer] = {}

    def _get_model(self, model_name: str) -> SentenceTransformer:
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
        return self._model_cache[model_name]

    @staticmethod
    def _maybe_pca(e1: np.ndarray, e2: np.ndarray, target_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        native_dim = e1.shape[1]
        if target_dim >= native_dim:
            return e1, e2

        all_embs = np.vstack([e1, e2])
        max_components = min(all_embs.shape[0], all_embs.shape[1])
        n_components = min(target_dim, max_components)

        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(all_embs)
        return pca.transform(e1), pca.transform(e2)

    def evaluate(
        self,
        model_name: str,
        target_dim: int,
        batch_size: int,
        normalize: bool,
        track_pca_energy: bool,
    ) -> Tuple[float, float, float]:
        """
        Returns: latency_ms, energy_kwh, pearson_r
        Measured region: embedding inference (+ optional PCA if track_pca_energy=True)
        Excludes: model loading, dataset loading, warm-up, metric computation
        """
        model = self._get_model(model_name)

        # Warmup (not tracked)
        if self.warmup_pairs and self.warmup_pairs > 0:
            w = min(self.warmup_pairs, len(self.s1))
            _ = model.encode(self.s1[:w], convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=normalize)
            _ = model.encode(self.s2[:w], convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=normalize)

        tracker = EmissionsTracker(
            project_name="sts_moo",
            output_dir=self.track_dir,
            measure_power_secs=1,
            save_to_file=False,
            log_level="error",
        )

        tracker.start()
        t0 = time.perf_counter()

        e1 = model.encode(self.s1, convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=normalize)
        e2 = model.encode(self.s2, convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=normalize)

        if track_pca_energy:
            e1, e2 = self._maybe_pca(e1, e2, target_dim)

        t1 = time.perf_counter()
        tracker.stop()

        latency_ms = (t1 - t0) * 1000.0
        energy_kwh = float(getattr(tracker.final_emissions_data, "energy_consumed", np.nan))

        if not track_pca_energy:
            e1, e2 = self._maybe_pca(e1, e2, target_dim)

        cos = 1.0 - pairwise_distances(e1, e2, metric="cosine")
        pred = np.diag(cos)
        acc = pearsonr(self.labels, pred)

        if not np.isfinite(acc):
            acc = -1.0
        if not np.isfinite(energy_kwh):
            energy_kwh = 1e9

        return float(latency_ms), float(energy_kwh), float(acc)


# -----------------------------
# Export + plotting
# -----------------------------
def export_pareto_csv(pareto_trials: List[optuna.trial.FrozenTrial], path_csv: str = "pareto_solutions.csv"):
    header = [
        "point_id",
        "latency_ms",
        "energy_kwh",
        "accuracy_pearson",
        "model_name",
        "target_dim",
        "batch_size",
        "normalize",
        "track_pca_energy",
    ]
    lines = [",".join(header)]
    for i, t in enumerate(pareto_trials, start=1):
        pid = f"P{i:02d}"
        lines.append(",".join([
            pid,
            f"{t.values[0]:.6f}",
            f"{t.values[1]:.9f}",
            f"{t.values[2]:.6f}",
            f"\"{t.params['model_name']}\"",
            str(t.params["target_dim"]),
            str(t.params["batch_size"]),
            str(t.params["normalize"]),
            str(t.params["track_pca_energy"]),
        ]))
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved Pareto mapping CSV: {path_csv}")


def plot_pareto_interactive_html(
    pareto_trials: List[optuna.trial.FrozenTrial],
    out_html: str = "pareto_3d_interactive.html",
):
    ids = [f"P{i:02d}" for i in range(1, len(pareto_trials) + 1)]

    latency = [t.values[0] for t in pareto_trials]   # ms
    energy = [t.values[1] for t in pareto_trials]    # kWh
    accuracy = [t.values[2] for t in pareto_trials]  # Pearson r

    hover_text = []
    for pid, t in zip(ids, pareto_trials):
        hover_text.append(
            f"<b>{pid}</b><br>"
            f"Model: {short_model(t.params['model_name'])}<br>"
            f"Dim: {t.params['target_dim']}<br>"
            f"Batch: {t.params['batch_size']}<br>"
            f"Normalize: {t.params['normalize']}<br>"
            f"PCA energy tracked: {t.params['track_pca_energy']}<br>"
            f"<br>"
            f"<b>Latency</b>: {t.values[0]:.2f} ms<br>"
            f"<b>Energy</b>: {t.values[1]:.6f} kWh<br>"
            f"<b>Pearson</b>: {t.values[2]:.4f}"
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=latency,
                y=energy,
                z=accuracy,
                mode="markers",
                marker=dict(
                    size=6,
                    color=accuracy,
                    colorscale="Viridis",
                    opacity=0.95,
                    colorbar=dict(title="Pearson r"),
                ),
                text=hover_text,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="Interactive 3D Pareto Front (STS-B, Pareto-optimal solutions)",
        template="plotly_white",
        scene=dict(
            xaxis_title="Latency (ms) ↓",
            yaxis_title="Energy (kWh) ↓",
            zaxis_title="Accuracy (Pearson r) ↑",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.write_html(out_html, include_plotlyjs=True)
    print(f"Saved interactive plot: {out_html}")


def configure_matplotlib_for_paper():
    # Vector-friendly PDF output and consistent typography.
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,      # TrueType fonts in PDF
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def plot_paper_figures_matplotlib(
    pareto_trials: List[optuna.trial.FrozenTrial],
    pdf_3d: str = "fig_pareto3d.pdf",
    pdf_la: str = "fig_proj_latency_accuracy.pdf",
    pdf_ea: str = "fig_proj_energy_accuracy.pdf",
    label_points: bool = False,
):
    """
    Static paper-ready figures (vector PDFs).
    - 3D scatter (linear axes, Pareto-only)
    - 2D projections for readability in print
    """
    configure_matplotlib_for_paper()

    ids = [f"P{i:02d}" for i in range(1, len(pareto_trials) + 1)]
    lat = np.array([t.values[0] for t in pareto_trials], dtype=float)
    eng = np.array([t.values[1] for t in pareto_trials], dtype=float)
    acc = np.array([t.values[2] for t in pareto_trials], dtype=float)

    # --- 3D figure (static) ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    # Orthographic projection improves readability (no perspective distortion).
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    # Marker sizing: emphasize better trade-offs
    score = norm01(acc) - 0.5 * norm01(lat) - 0.5 * norm01(eng)
    score = norm01(score)
    s = 22 + 70 * score

    sc = ax.scatter(lat, eng, acc, s=s, c=acc, depthshade=False, linewidth=0.3)

    ax.set_xlabel("Latency (ms) ↓")
    ax.set_ylabel("Energy (kWh) ↓")
    ax.set_zlabel("Pearson r ↑")
    ax.set_title("Pareto-optimal STS-B configurations (3 objectives)")

    # A viewing angle that usually separates ridge-like fronts
    ax.view_init(elev=18, azim=230)

    # Optional labels (usually off for paper figure; use table/CSV instead)
    if label_points:
        max_labels = 12
        idx = np.argsort(-score)[:max_labels].tolist()
        for i in idx:
            ax.text(lat[i], eng[i], acc[i], f" {ids[i]}", fontsize=8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.08)
    cbar.set_label("Pearson r")

    fig.savefig(pdf_3d)
    plt.close(fig)

    # --- 2D projection: Latency vs Accuracy ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    ax.scatter(lat, acc, s=26, linewidth=0.3)
    ax.set_xlabel("Latency (ms) ↓")
    ax.set_ylabel("Pearson r ↑")
    ax.set_title("Pareto projection: latency vs accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(pdf_la)
    plt.close(fig)

    # --- 2D projection: Energy vs Accuracy ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    ax.scatter(eng, acc, s=26, linewidth=0.3)
    ax.set_xlabel("Energy (kWh) ↓")
    ax.set_ylabel("Pearson r ↑")
    ax.set_title("Pareto projection: energy vs accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(pdf_ea)
    plt.close(fig)

    print(f"Saved paper PDFs: {pdf_3d}, {pdf_la}, {pdf_ea}")


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(CC_DIR, exist_ok=True)

    evaluator = STSBenchmarkEvaluator(split="validation", max_pairs=MAX_PAIRS, warmup_pairs=WARMUP_PAIRS, track_dir=CC_DIR)

    # More models (curated): common SentenceTransformers, mix of sizes and training objectives.
    # Note: some larger ones may increase runtime and memory requirements.
    model_candidates = [
        # compact / widely used
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",

        # strong general-purpose
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-mpnet-base-v2",

        # QA-oriented / retrieval oriented (good diversity)
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",

        # multilingual (optional but useful)
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

        # MS MARCO style (often used in retrieval; diversity)
        "sentence-transformers/msmarco-distilbert-base-v4",

    ]

    def objective(trial: optuna.trial.Trial):
        model_name = trial.suggest_categorical("model_name", model_candidates)

        # Conservative upper bound; PCA will clamp based on native dim and sample count.
        target_dim = trial.suggest_int("target_dim", 64, 768, step=32)

        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        normalize = trial.suggest_categorical("normalize", [True, False])

        # Recommended for papers: fix to True in camera-ready if you want one clean definition
        track_pca_energy = trial.suggest_categorical("track_pca_energy", [False, True])

        return evaluator.evaluate(
            model_name=model_name,
            target_dim=target_dim,
            batch_size=batch_size,
            normalize=normalize,
            track_pca_energy=track_pca_energy,
        )

    print("Running 3-objective MOO on GLUE STS-B (validation split)")
    print("Objectives: minimize latency, minimize energy, maximize Pearson correlation")
    study = optuna.create_study(directions=["minimize", "minimize", "maximize"])
    study.optimize(objective, n_trials=N_TRIALS)

    pareto = study.best_trials
    print(f"\nPareto-optimal solutions: {len(pareto)}")

    # Console mapping
    print(f"{'Point':<6} | {'Latency(ms)':<12} | {'Energy(kWh)':<12} | {'Pearson':<8} | {'Model':<32} | {'Dim':<5} | {'BS':<4} | {'Norm':<5} | {'PCA_E':<5}")
    print("-" * 140)
    for i, t in enumerate(pareto, start=1):
        pid = f"P{i:02d}"
        print(
            f"{pid:<6} | {t.values[0]:<12.2f} | {t.values[1]:<12.6f} | {t.values[2]:<8.4f} | "
            f"{short_model(t.params['model_name']):<32} | {t.params['target_dim']:<5} | {t.params['batch_size']:<4} | "
            f"{str(t.params['normalize']):<5} | {str(t.params['track_pca_energy']):<5}"
        )

    export_pareto_csv(pareto, "pareto_solutions.csv")

    # Interactive artifact (offline)
    plot_pareto_interactive_html(pareto, out_html="pareto_3d_interactive.html")

    # Paper-ready PDFs (vector)
    plot_paper_figures_matplotlib(
        pareto,
        pdf_3d="fig_pareto3d.pdf",
        pdf_la="fig_proj_latency_accuracy.pdf",
        pdf_ea="fig_proj_energy_accuracy.pdf",
        label_points=False,  # prefer ID mapping table/CSV
    )


if __name__ == "__main__":
    main()
