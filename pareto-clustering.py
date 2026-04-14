"""
Text Clustering 3-objective MOO (latency, energy, clustering quality) + paper-ready plots.

Outputs:
- pareto_solutions.csv
- pareto_3d_interactive.html (offline, no CDN, linear axes)
- fig_pareto3d.pdf           (vector PDF for LaTeX)
- fig_proj_latency_quality.pdf
- fig_proj_energy_quality.pdf

Install:
  pip install optuna sentence-transformers datasets scikit-learn codecarbon plotly matplotlib
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import optuna
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from codecarbon import EmissionsTracker

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Config
# -----------------------------
N_TRIALS = 3               # 200-400 gives smoother Pareto sets
MAX_SAMPLES = 2000         # None -> full split (may be slow)
WARMUP_SAMPLES = 128
CC_DIR = ".cc_logs"

# Dataset config for clustering
DATASET_NAME = "ag_news"
DATASET_CONFIG = None      # keep None for datasets without configs; AG News works with None
DATASET_SPLIT = "test"     # "train" or "test"
TEXT_FIELD = "text"
LABEL_FIELD = "label"

# Paper figure sizing
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
class ClusteringBenchmarkEvaluator:
    dataset_name: str = "ag_news"
    dataset_config: Optional[str] = None
    split: str = "test"
    text_field: str = "text"
    label_field: str = "label"
    max_samples: int = 2000
    warmup_samples: int = 128
    track_dir: str = ".cc_logs"

    def __post_init__(self):
        os.makedirs(self.track_dir, exist_ok=True)

        if self.dataset_config:
            ds = load_dataset_or_raise(self.dataset_name, self.dataset_config, split=self.split)
        else:
            ds = load_dataset_or_raise(self.dataset_name, split=self.split)

        if self.max_samples is not None:
            ds = ds.select(range(min(self.max_samples, len(ds))))

        self.texts = ds[self.text_field]
        self.labels = np.array(ds[self.label_field], dtype=int)
        self.n_classes = int(len(set(self.labels.tolist())))
        self._model_cache: Dict[str, SentenceTransformer] = {}

    def _get_model(self, model_name: str) -> SentenceTransformer:
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
        return self._model_cache[model_name]

    @staticmethod
    def _maybe_pca(e: np.ndarray, target_dim: int) -> np.ndarray:
        native_dim = e.shape[1]
        if target_dim >= native_dim:
            return e

        max_components = min(e.shape[0], e.shape[1])
        n_components = min(target_dim, max_components)

        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(e)
        return pca.transform(e)

    def evaluate(
        self,
        model_name: str,
        target_dim: int,
        batch_size: int,
        normalize: bool,
        track_pca_energy: bool,
        kmeans_batch: int,
    ) -> Tuple[float, float, float]:
        """
        Returns: latency_ms, energy_kwh, quality_nmi
        Measured region: embedding inference (+ optional PCA if track_pca_energy=True)
        Excludes: model loading, dataset loading, warm-up, clustering + metric computation
        """
        model = self._get_model(model_name)

        # Warmup (not tracked)
        if self.warmup_samples and self.warmup_samples > 0:
            w = min(self.warmup_samples, len(self.texts))
            _ = model.encode(
                self.texts[:w],
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )

        tracker = EmissionsTracker(
            project_name="clustering_moo",
            output_dir=self.track_dir,
            measure_power_secs=1,
            save_to_file=False,
            log_level="error",
        )

        tracker.start()
        t0 = time.perf_counter()

        embs = model.encode(
            self.texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=normalize,
        )

        if track_pca_energy:
            embs = self._maybe_pca(embs, target_dim)

        t1 = time.perf_counter()
        tracker.stop()

        latency_ms = (t1 - t0) * 1000.0
        energy_kwh = float(getattr(tracker.final_emissions_data, "energy_consumed", np.nan))

        if not track_pca_energy:
            embs = self._maybe_pca(embs, target_dim)

        # Clustering (not energy-tracked, consistent with your STS setup)
        km = MiniBatchKMeans(
            n_clusters=self.n_classes,
            batch_size=kmeans_batch,
            random_state=0,
            n_init="auto",
        )
        pred = km.fit_predict(embs)

        quality = float(normalized_mutual_info_score(self.labels, pred))

        if not np.isfinite(quality):
            quality = -1.0
        if not np.isfinite(energy_kwh):
            energy_kwh = 1e9

        return float(latency_ms), float(energy_kwh), float(quality)


# -----------------------------
# Export + plotting
# -----------------------------
def export_pareto_csv(pareto_trials: List[optuna.trial.FrozenTrial], path_csv: str = "pareto_solutions.csv"):
    header = [
        "point_id",
        "latency_ms",
        "energy_kwh",
        "quality_nmi",
        "model_name",
        "target_dim",
        "batch_size",
        "normalize",
        "track_pca_energy",
        "kmeans_batch",
        "dataset_name",
        "split",
        "max_samples",
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
            str(t.params["kmeans_batch"]),
            f"\"{DATASET_NAME}\"",
            f"\"{DATASET_SPLIT}\"",
            str(MAX_SAMPLES),
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
    quality = [t.values[2] for t in pareto_trials]   # NMI

    hover_text = []
    for pid, t in zip(ids, pareto_trials):
        hover_text.append(
            f"<b>{pid}</b><br>"
            f"Model: {short_model(t.params['model_name'])}<br>"
            f"Dim: {t.params['target_dim']}<br>"
            f"Batch: {t.params['batch_size']}<br>"
            f"Normalize: {t.params['normalize']}<br>"
            f"PCA energy tracked: {t.params['track_pca_energy']}<br>"
            f"KMeans batch: {t.params['kmeans_batch']}<br>"
            f"Dataset: {DATASET_NAME} ({DATASET_SPLIT})<br>"
            f"<br>"
            f"<b>Latency</b>: {t.values[0]:.2f} ms<br>"
            f"<b>Energy</b>: {t.values[1]:.6f} kWh<br>"
            f"<b>NMI</b>: {t.values[2]:.4f}"
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=latency,
                y=energy,
                z=quality,
                mode="markers",
                marker=dict(
                    size=6,
                    color=quality,
                    colorscale="Viridis",
                    opacity=0.95,
                    colorbar=dict(title="NMI"),
                ),
                text=hover_text,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="Interactive 3D Pareto Front (Text clustering, Pareto-optimal solutions)",
        template="plotly_white",
        scene=dict(
            xaxis_title="Latency (ms) ↓",
            yaxis_title="Energy (kWh) ↓",
            zaxis_title="Quality (NMI) ↑",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.write_html(out_html, include_plotlyjs=True)
    print(f"Saved interactive plot: {out_html}")


def configure_matplotlib_for_paper():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def plot_paper_figures_matplotlib(
    pareto_trials: List[optuna.trial.FrozenTrial],
    pdf_3d: str = "fig_pareto3d.pdf",
    pdf_lq: str = "fig_proj_latency_quality.pdf",
    pdf_eq: str = "fig_proj_energy_quality.pdf",
    label_points: bool = False,
):
    configure_matplotlib_for_paper()

    ids = [f"P{i:02d}" for i in range(1, len(pareto_trials) + 1)]
    lat = np.array([t.values[0] for t in pareto_trials], dtype=float)
    eng = np.array([t.values[1] for t in pareto_trials], dtype=float)
    q = np.array([t.values[2] for t in pareto_trials], dtype=float)

    # --- 3D figure (static) ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    score = norm01(q) - 0.5 * norm01(lat) - 0.5 * norm01(eng)
    score = norm01(score)
    s = 22 + 70 * score

    sc = ax.scatter(lat, eng, q, s=s, c=q, depthshade=False, linewidth=0.3)

    ax.set_xlabel("Latency (ms) ↓")
    ax.set_ylabel("Energy (kWh) ↓")
    ax.set_zlabel("NMI ↑")
    ax.set_title(f"Pareto-optimal clustering configs (dataset: {DATASET_NAME})")

    ax.view_init(elev=18, azim=230)

    if label_points:
        max_labels = 12
        idx = np.argsort(-score)[:max_labels].tolist()
        for i in idx:
            ax.text(lat[i], eng[i], q[i], f" {ids[i]}", fontsize=8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.08)
    cbar.set_label("NMI")

    fig.savefig(pdf_3d)
    plt.close(fig)

    # --- 2D projection: Latency vs Quality ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    ax.scatter(lat, q, s=26, linewidth=0.3)
    ax.set_xlabel("Latency (ms) ↓")
    ax.set_ylabel("NMI ↑")
    ax.set_title("Pareto projection: latency vs clustering quality")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(pdf_lq)
    plt.close(fig)

    # --- 2D projection: Energy vs Quality ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    ax.scatter(eng, q, s=26, linewidth=0.3)
    ax.set_xlabel("Energy (kWh) ↓")
    ax.set_ylabel("NMI ↑")
    ax.set_title("Pareto projection: energy vs clustering quality")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(pdf_eq)
    plt.close(fig)

    print(f"Saved paper PDFs: {pdf_3d}, {pdf_lq}, {pdf_eq}")


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(CC_DIR, exist_ok=True)

    evaluator = ClusteringBenchmarkEvaluator(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        split=DATASET_SPLIT,
        text_field=TEXT_FIELD,
        label_field=LABEL_FIELD,
        max_samples=MAX_SAMPLES,
        warmup_samples=WARMUP_SAMPLES,
        track_dir=CC_DIR,
    )

    model_candidates = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]

    def objective(trial: optuna.trial.Trial):
        model_name = trial.suggest_categorical("model_name", model_candidates)
        target_dim = trial.suggest_int("target_dim", 64, 768, step=32)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        normalize = trial.suggest_categorical("normalize", [True, False])
        track_pca_energy = trial.suggest_categorical("track_pca_energy", [False, True])

        # MiniBatchKMeans speed knob
        kmeans_batch = trial.suggest_categorical("kmeans_batch", [256, 512, 1024, 2048])

        return evaluator.evaluate(
            model_name=model_name,
            target_dim=target_dim,
            batch_size=batch_size,
            normalize=normalize,
            track_pca_energy=track_pca_energy,
            kmeans_batch=kmeans_batch,
        )

    print(f"Running 3-objective MOO on {DATASET_NAME} ({DATASET_SPLIT})")
    print("Objectives: minimize latency, minimize energy, maximize NMI clustering quality")
    study = optuna.create_study(directions=["minimize", "minimize", "maximize"])
    study.optimize(objective, n_trials=N_TRIALS)

    pareto = study.best_trials
    print(f"\nPareto-optimal solutions: {len(pareto)}")

    print(f"{'Point':<6} | {'Latency(ms)':<12} | {'Energy(kWh)':<12} | {'NMI':<8} | {'Model':<32} | {'Dim':<5} | {'BS':<4} | {'Norm':<5} | {'PCA_E':<5} | {'KM_B':<6}")
    print("-" * 150)
    for i, t in enumerate(pareto, start=1):
        pid = f"P{i:02d}"
        print(
            f"{pid:<6} | {t.values[0]:<12.2f} | {t.values[1]:<12.6f} | {t.values[2]:<8.4f} | "
            f"{short_model(t.params['model_name']):<32} | {t.params['target_dim']:<5} | {t.params['batch_size']:<4} | "
            f"{str(t.params['normalize']):<5} | {str(t.params['track_pca_energy']):<5} | {t.params['kmeans_batch']:<6}"
        )

    export_pareto_csv(pareto, "pareto_solutions.csv")
    plot_pareto_interactive_html(pareto, out_html="pareto_3d_interactive.html")

    plot_paper_figures_matplotlib(
        pareto,
        pdf_3d="fig_pareto3d.pdf",
        pdf_lq="fig_proj_latency_quality.pdf",
        pdf_eq="fig_proj_energy_quality.pdf",
        label_points=False,
    )


if __name__ == "__main__":
    main()
