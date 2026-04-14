"""
Energy-aware multi-objective optimization of TEXT EMBEDDINGS for CLASSIFICATION accuracy.

Benchmarks:
- SST-2 (GLUE)  : binary sentiment classification (metric: accuracy)
- AG News       : 4-class topic classification (metric: accuracy)

Pipeline per trial:
Text -> SentenceTransformer embeddings -> (optional PCA to target_dim) -> classifier -> accuracy

Objectives (multi-objective Optuna):
1) Latency (ms)  [minimize]
2) Energy (kWh)  [minimize]  (CodeCarbon estimate)
3) Accuracy      [maximize]

Outputs:
- pareto_solutions.csv
- pareto_3d_interactive.html (offline, no CDN, linear axes)
- fig_pareto3d.pdf (vector PDF)
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
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from codecarbon import EmissionsTracker

import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Config
# -----------------------------
TASK = "ag_news"          # "sst2" or "ag_news"
N_TRIALS = 20         # 200-400 gives smoother Pareto sets
RANDOM_SEED = 0

# To keep trials fast and stable, subsample. Increase for final experiments.
MAX_TRAIN = 1000       # SST-2 train is large; 4k is ok for per-trial fitting
MAX_EVAL = 500        # SST-2 validation ~872; AG News test is larger
WARMUP_SAMPLES = 128

CC_DIR = ".cc_logs"

# Plot sizes (paper-ready)
PAPER_FIGSIZE_WIDE = (6.8, 4.2)
PAPER_FIGSIZE_3D = (6.8, 4.8)


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


def subsample_lists(x_list: List[str], y_list: List[int], k: int, rng: np.random.Generator):
    n = len(x_list)
    if k is None or k >= n:
        return x_list, np.array(y_list, dtype=int)
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    x = [x_list[i] for i in idx]
    y = np.array([y_list[i] for i in idx], dtype=int)
    return x, y


# -----------------------------
# Dataset loading
# -----------------------------
def load_classification_task(task: str, max_train: int, max_eval: int, seed: int):
    rng = np.random.default_rng(seed)

    if task == "sst2":
        ds_train = load_dataset("glue", "sst2", split="train")
        ds_eval = load_dataset("glue", "sst2", split="validation")
        x_train, y_train = ds_train["sentence"], ds_train["label"]
        x_eval, y_eval = ds_eval["sentence"], ds_eval["label"]

    elif task == "ag_news":
        ds_train = load_dataset("ag_news", split="train")
        ds_eval = load_dataset("ag_news", split="test")
        x_train, y_train = ds_train["text"], ds_train["label"]
        x_eval, y_eval = ds_eval["text"], ds_eval["label"]

    else:
        raise ValueError(f"Unsupported TASK={task}. Use 'sst2' or 'ag_news'.")

    x_train, y_train = subsample_lists(list(x_train), list(y_train), max_train, rng)
    x_eval, y_eval = subsample_lists(list(x_eval), list(y_eval), max_eval, rng)

    return x_train, y_train, x_eval, y_eval


# -----------------------------
# Evaluator: embeddings -> PCA -> classifier
# -----------------------------
@dataclass
class EmbeddingClassificationEvaluator:
    task: str
    max_train: int
    max_eval: int
    seed: int
    warmup_samples: int = 128
    cc_dir: str = ".cc_logs"

    def __post_init__(self):
        os.makedirs(self.cc_dir, exist_ok=True)
        self.rng = np.random.default_rng(self.seed)

        self.x_train, self.y_train, self.x_eval, self.y_eval = load_classification_task(
            self.task, self.max_train, self.max_eval, self.seed
        )

        # Cache models
        self._model_cache: Dict[str, SentenceTransformer] = {}
        # Cache embeddings per (model, normalize, batch_size)
        self._emb_cache: Dict[Tuple[str, bool, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _get_model(self, model_name: str) -> SentenceTransformer:
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
        return self._model_cache[model_name]

    @staticmethod
    def _maybe_pca(e_tr: np.ndarray, e_ev: np.ndarray, target_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        native_dim = e_tr.shape[1]
        if target_dim >= native_dim:
            return e_tr, e_ev

        # PCA constraint: n_components <= min(n_samples, n_features)
        max_components = min(e_tr.shape[0], e_tr.shape[1])
        n_components = min(target_dim, max_components)

        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(e_tr)
        return pca.transform(e_tr), pca.transform(e_ev)

    def _encode_cached(self, model_name: str, batch_size: int, normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
        key = (model_name, normalize, batch_size)
        if key in self._emb_cache:
            return self._emb_cache[key]

        model = self._get_model(model_name)

        # Warmup (not measured)
        if self.warmup_samples and self.warmup_samples > 0:
            w = min(self.warmup_samples, len(self.x_train))
            _ = model.encode(
                self.x_train[:w],
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )

        e_tr = model.encode(self.x_train, convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=normalize)
        e_ev = model.encode(self.x_eval, convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=normalize)
        self._emb_cache[key] = (e_tr, e_ev)
        return e_tr, e_ev

    def evaluate(
        self,
        model_name: str,
        target_dim: int,
        batch_size: int,
        normalize: bool,
        classifier: str,
        track_pca_energy: bool,
    ) -> Tuple[float, float, float]:
        """
        Returns:
          latency_ms  (min)
          energy_kwh  (min)
          accuracy    (max)

        Measured region includes:
          - embeddings (train+eval)
          - optional PCA (if track_pca_energy=True)
          - classifier training + prediction

        Excludes:
          - dataset loading
          - model loading
          - warmup
        """
        tracker = EmissionsTracker(
            project_name=f"emb_cls_{self.task}",
            output_dir=self.cc_dir,
            measure_power_secs=1,
            save_to_file=False,
            log_level="error",
        )

        tracker.start()
        t0 = time.perf_counter()

        # 1) Embeddings (train+eval)
        e_tr, e_ev = self._encode_cached(model_name=model_name, batch_size=batch_size, normalize=normalize)

        # 2) PCA (optionally inside energy window)
        if track_pca_energy:
            e_tr2, e_ev2 = self._maybe_pca(e_tr, e_ev, target_dim)
        else:
            e_tr2, e_ev2 = e_tr, e_ev

        # 3) Classifier
        if classifier == "logreg":
            clf = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                n_jobs=1,
                multi_class="auto",
                random_state=RANDOM_SEED,
            )
        elif classifier == "linear_svm":
            clf = LinearSVC(random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unknown classifier={classifier}")

        clf.fit(e_tr2, self.y_train)
        y_pred = clf.predict(e_ev2)

        t1 = time.perf_counter()
        tracker.stop()

        latency_ms = (t1 - t0) * 1000.0
        energy_kwh = float(getattr(tracker.final_emissions_data, "energy_consumed", np.nan))

        # If PCA wasn't tracked for energy, still apply it for accuracy (fair comparison)
        if not track_pca_energy:
            e_tr2, e_ev2 = self._maybe_pca(e_tr, e_ev, target_dim)
            clf.fit(e_tr2, self.y_train)
            y_pred = clf.predict(e_ev2)

        acc = float(accuracy_score(self.y_eval, y_pred))

        if not np.isfinite(energy_kwh):
            energy_kwh = 1e9

        return float(latency_ms), float(energy_kwh), float(acc)


# -----------------------------
# Output: CSV + plots
# -----------------------------
def export_pareto_csv(pareto_trials: List[optuna.trial.FrozenTrial], path_csv: str = "pareto_solutions.csv", task: str = ""):
    header = [
        "point_id",
        "task",
        "latency_ms",
        "energy_kwh",
        "accuracy",
        "model_name",
        "target_dim",
        "batch_size",
        "normalize",
        "classifier",
        "track_pca_energy",
    ]
    lines = [",".join(header)]
    for i, t in enumerate(pareto_trials, start=1):
        pid = f"P{i:02d}"
        lines.append(",".join([
            pid,
            task,
            f"{t.values[0]:.6f}",
            f"{t.values[1]:.9f}",
            f"{t.values[2]:.6f}",
            f"\"{t.params['model_name']}\"",
            str(t.params["target_dim"]),
            str(t.params["batch_size"]),
            str(t.params["normalize"]),
            str(t.params["classifier"]),
            str(t.params["track_pca_energy"]),
        ]))
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved Pareto mapping CSV: {path_csv}")


def plot_pareto_interactive_html(pareto_trials: List[optuna.trial.FrozenTrial], out_html: str, task: str):
    ids = [f"P{i:02d}" for i in range(1, len(pareto_trials) + 1)]
    lat = [t.values[0] for t in pareto_trials]
    eng = [t.values[1] for t in pareto_trials]
    acc = [t.values[2] for t in pareto_trials]

    hover = []
    for pid, t in zip(ids, pareto_trials):
        hover.append(
            f"<b>{pid}</b><br>"
            f"Task: {task}<br>"
            f"Model: {short_model(t.params['model_name'])}<br>"
            f"Dim: {t.params['target_dim']}<br>"
            f"Batch: {t.params['batch_size']}<br>"
            f"Normalize: {t.params['normalize']}<br>"
            f"Classifier: {t.params['classifier']}<br>"
            f"PCA energy tracked: {t.params['track_pca_energy']}<br><br>"
            f"<b>Latency</b>: {t.values[0]:.2f} ms<br>"
            f"<b>Energy</b>: {t.values[1]:.6f} kWh<br>"
            f"<b>Accuracy</b>: {t.values[2]:.4f}"
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=lat, y=eng, z=acc,
                mode="markers",
                marker=dict(
                    size=6,
                    color=acc,
                    colorscale="Viridis",
                    opacity=0.95,
                    colorbar=dict(title="Accuracy"),
                ),
                text=hover,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=f"Interactive 3D Pareto Front ({task}, Pareto-optimal solutions)",
        template="plotly_white",
        scene=dict(
            xaxis_title="Latency (ms) ↓",
            yaxis_title="Energy (kWh) ↓",
            zaxis_title="Accuracy ↑",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.write_html(out_html, include_plotlyjs=True)  # offline, no CDN
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
    task: str,
    pdf_3d: str,
    pdf_la: str,
    pdf_ea: str,
):
    configure_matplotlib_for_paper()

    lat = np.array([t.values[0] for t in pareto_trials], dtype=float)
    eng = np.array([t.values[1] for t in pareto_trials], dtype=float)
    acc = np.array([t.values[2] for t in pareto_trials], dtype=float)

    # --- 3D Pareto (static) ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    score = norm01(acc) - 0.5 * norm01(lat) - 0.5 * norm01(eng)
    score = norm01(score)
    s = 22 + 70 * score

    sc = ax.scatter(lat, eng, acc, s=s, c=acc, depthshade=False, linewidth=0.3)

    ax.set_xlabel("Latency (ms) ↓")
    ax.set_ylabel("Energy (kWh) ↓")
    ax.set_zlabel("Accuracy ↑")
    ax.set_title(f"Pareto-optimal configurations ({task})")
    ax.view_init(elev=18, azim=230)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.08)
    cbar.set_label("Accuracy")

    fig.savefig(pdf_3d)
    plt.close(fig)

    # --- 2D: latency vs accuracy ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    ax.scatter(lat, acc, s=26, linewidth=0.3)
    ax.set_xlabel("Latency (ms) ↓")
    ax.set_ylabel("Accuracy ↑")
    ax.set_title(f"Pareto projection: latency vs accuracy ({task})")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(pdf_la)
    plt.close(fig)

    # --- 2D: energy vs accuracy ---
    fig = plt.figure(figsize=PAPER_FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    ax.scatter(eng, acc, s=26, linewidth=0.3)
    ax.set_xlabel("Energy (kWh) ↓")
    ax.set_ylabel("Accuracy ↑")
    ax.set_title(f"Pareto projection: energy vs accuracy ({task})")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(pdf_ea)
    plt.close(fig)

    print(f"Saved paper PDFs: {pdf_3d}, {pdf_la}, {pdf_ea}")


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(CC_DIR, exist_ok=True)

    evaluator = EmbeddingClassificationEvaluator(
        task=TASK,
        max_train=MAX_TRAIN,
        max_eval=MAX_EVAL,
        seed=RANDOM_SEED,
        warmup_samples=WARMUP_SAMPLES,
        cc_dir=CC_DIR,
    )

    # Curated model list (mix of sizes/objectives). Remove the largest if runtime is high.
    model_candidates = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-mpnet-base-v2",
    ]

    def objective(trial: optuna.trial.Trial):
        model_name = trial.suggest_categorical("model_name", model_candidates)
        target_dim = trial.suggest_int("target_dim", 64, 768, step=32)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        normalize = trial.suggest_categorical("normalize", [True, False])
        classifier = trial.suggest_categorical("classifier", ["logreg", "linear_svm"])
        track_pca_energy = trial.suggest_categorical("track_pca_energy", [False, True])

        return evaluator.evaluate(
            model_name=model_name,
            target_dim=target_dim,
            batch_size=batch_size,
            normalize=normalize,
            classifier=classifier,
            track_pca_energy=track_pca_energy,
        )

    print(f"Task: {TASK} | Train: {len(evaluator.x_train)} | Eval: {len(evaluator.x_eval)}")
    print("Objectives: minimize latency, minimize energy, maximize accuracy.")
    study = optuna.create_study(directions=["minimize", "minimize", "maximize"])
    study.optimize(objective, n_trials=N_TRIALS)

    pareto = study.best_trials
    print(f"\nPareto-optimal solutions: {len(pareto)}")

    print(f"{'Point':<6} | {'Latency(ms)':<12} | {'Energy(kWh)':<12} | {'Acc':<6} | {'Model':<30} | {'Dim':<5} | {'BS':<4} | {'Norm':<5} | {'Clf':<10} | {'PCA_E':<5}")
    print("-" * 145)
    for i, t in enumerate(pareto, start=1):
        pid = f"P{i:02d}"
        print(
            f"{pid:<6} | {t.values[0]:<12.2f} | {t.values[1]:<12.6f} | {t.values[2]:<6.4f} | "
            f"{short_model(t.params['model_name']):<30} | {t.params['target_dim']:<5} | {t.params['batch_size']:<4} | "
            f"{str(t.params['normalize']):<5} | {t.params['classifier']:<10} | {str(t.params['track_pca_energy']):<5}"
        )

    export_pareto_csv(pareto, path_csv="pareto_solutions.csv", task=TASK)

    plot_pareto_interactive_html(
        pareto,
        out_html="pareto_3d_interactive.html",
        task=TASK,
    )

    plot_paper_figures_matplotlib(
        pareto_trials=pareto,
        task=TASK,
        pdf_3d="fig_pareto3d.pdf",
        pdf_la="fig_proj_latency_accuracy.pdf",
        pdf_ea="fig_proj_energy_accuracy.pdf",
    )


if __name__ == "__main__":
    main()
