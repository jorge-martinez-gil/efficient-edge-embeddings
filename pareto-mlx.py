"""
STS-B 3-objective multi-objective optimization with embedding quantization.

Extends ``pareto-similarity.py`` by adding embedding quantization as an additional
optimization knob.  Supports raw NumPy groupwise quantization (symmetric or affine)
and Apple MLX quantization on Apple Silicon.  When ``mlx`` is not installed the
framework automatically falls back to the NumPy quantizer.

Simultaneously minimizes inference latency (ms) and energy consumption (kWh, via CodeCarbon)
while maximizing Pearson correlation on the GLUE STS-B validation split.

Usage:
    python pareto-mlx.py

Configuration constants (edit at the top of this file):
    N_TRIALS      -- number of Optuna trials (use 200-400 for paper-quality results)
    MAX_PAIRS     -- max sentence pairs evaluated per trial (None = full split)
    WARMUP_PAIRS  -- warm-up pairs not counted in measurements
    CC_DIR        -- CodeCarbon log directory (created automatically)

Outputs:
    pareto_solutions.csv              -- Pareto-optimal configurations (CSV)
    pareto_3d_interactive.html        -- Offline interactive 3-D Pareto plot
    fig_pareto3d.pdf                  -- Vector PDF for LaTeX (3-D view)
    fig_proj_latency_accuracy.pdf     -- 2-D projection: latency vs Pearson r
    fig_proj_energy_accuracy.pdf      -- 2-D projection: energy vs Pearson r

Install:
    pip install optuna sentence-transformers datasets scikit-learn codecarbon plotly matplotlib numpy

Optional (Apple Silicon):
    pip install mlx
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
MAX_PAIRS = 1500           # None -> full STS-B validation
WARMUP_PAIRS = 128
CC_DIR = ".cc_logs"

# Paper figure sizing (LNCS ~ 12.2cm text width; one-column ~ 6cm-ish)
PAPER_FIGSIZE_WIDE = (6.8, 4.2)   # inches
PAPER_FIGSIZE_3D = (6.8, 4.8)     # inches


# -----------------------------
# Helpers
# -----------------------------
def short_model(name: str) -> str:
    """Return a shortened display name for a sentence-transformers model identifier."""
    return (
        name.replace("sentence-transformers/", "")
            .replace("all-", "")
            .replace("-v2", "")
    )


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two 1-D arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + 1e-12
    return float((x * y).sum() / denom)


def norm01(a: np.ndarray) -> np.ndarray:
    """Normalize array values to the [0, 1] range."""
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


def ensure_dim_divisible(e: np.ndarray, group_size: int) -> np.ndarray:
    if group_size <= 1:
        return e

    d = e.shape[1]

    # If the requested group is larger than the embedding dimension,
    # keep the tensor unchanged. The quantizer will clamp later.
    if group_size > d:
        return e

    d2 = (d // group_size) * group_size
    if d2 <= 0 or d2 == d:
        return e

    return e[:, :d2]


# -----------------------------
# Quantization (raw + MLX)
# -----------------------------
def try_import_mlx():
    """Attempt to import the ``mlx.core`` module; return it or ``None`` if unavailable."""
    try:
        import mlx.core as mx  # type: ignore
        return mx
    except Exception:
        return None


def raw_groupwise_quantize(x: np.ndarray, bits: int, group_size: int, mode: str):
    """Group-wise quantization for float32 embedding matrices.

    Splits each row of ``x`` into groups of ``group_size`` elements and quantizes
    each group independently using either a symmetric (signed int) or affine (unsigned int)
    scheme.  Values are stored in int8/uint8 (not bitpacked) for simplicity.

    Args:
        x: 2-D float32 embedding matrix of shape ``(n_samples, embedding_dim)``.
        bits: Quantization bit-width (4 or 8).
        group_size: Number of elements per quantization group.
        mode: Quantization scheme — ``"symmetric"`` or ``"affine"``.

    Returns:
        A dict with keys ``backend``, ``q``, ``scale``, ``bias``, ``shape``,
        ``bits``, ``group_size``, and ``mode`` sufficient to reconstruct the
        approximated float32 matrix via :func:`raw_dequantize`.
    """
    x = np.asarray(x, dtype=np.float32)
    assert bits in (4, 8)
    assert mode in ("symmetric", "affine")

    n, d = x.shape

    # Clamp invalid or oversized groups to the current embedding dimension
    if group_size <= 1 or group_size > d:
        group_size = d

    if d % group_size != 0:
        x = ensure_dim_divisible(x, group_size)
        n, d = x.shape

        # Re-check after trimming
        if group_size > d:
            group_size = d

    g = group_size

    # Final guard against zero-group reshapes
    if g <= 0 or d <= 0 or d // g <= 0:
        raise ValueError(
            f"Invalid quantization shape: x.shape={x.shape}, group_size={group_size}"
        )

    xg = x.reshape(n, d // g, g)

    if mode == "symmetric":
        qmax = (2 ** (bits - 1)) - 1
        qmin = -qmax - 1
        scale = np.max(np.abs(xg), axis=2, keepdims=True) / max(qmax, 1)
        scale = np.maximum(scale, 1e-8).astype(np.float32)
        bias = None
        q = np.round(xg / scale).clip(qmin, qmax).astype(np.int8)
    else:
        qmax = (2 ** bits) - 1
        xmin = np.min(xg, axis=2, keepdims=True).astype(np.float32)
        xmax = np.max(xg, axis=2, keepdims=True).astype(np.float32)
        scale = (xmax - xmin) / max(qmax, 1)
        scale = np.maximum(scale, 1e-8).astype(np.float32)
        bias = xmin
        q = np.round((xg - bias) / scale).clip(0, qmax).astype(np.uint8)

    return {
        "backend": "raw",
        "q": q,
        "scale": scale,
        "bias": bias,
        "shape": (n, d),
        "bits": bits,
        "group_size": g,
        "mode": mode,
    }


def raw_dequantize(qpack) -> np.ndarray:
    """Reconstruct a float32 embedding matrix from a raw quantization pack."""
    q = qpack["q"]
    scale = qpack["scale"]
    bias = qpack["bias"]
    n, d = qpack["shape"]
    mode = qpack["mode"]

    if mode == "symmetric":
        xg = q.astype(np.float32) * scale
    else:
        xg = q.astype(np.float32) * scale + bias
    return xg.reshape(n, d).astype(np.float32)


def mlx_quantize(x: np.ndarray, bits: int, group_size: int):
    """Quantize a float32 embedding matrix using Apple MLX.

    Falls back to ``None`` if the ``mlx`` package is not available.

    Args:
        x: 2-D float32 embedding matrix.
        bits: Quantization bit-width passed to ``mx.quantized``.
        group_size: Group size passed to ``mx.quantized``.

    Returns:
        A dict with MLX quantization data, or ``None`` if MLX is unavailable.
    """
    mx = try_import_mlx()
    if mx is None:
        return None

    x = np.asarray(x, dtype=np.float32)
    d = x.shape[1]

    if group_size <= 1 or group_size > d:
        group_size = d

    if x.shape[1] % group_size != 0:
        x = ensure_dim_divisible(x, group_size)

    d = x.shape[1]
    if group_size > d:
        group_size = d

    mx_x = mx.array(x)
    q = mx.quantized(mx_x, group_size=group_size, bits=bits)
    return {
        "backend": "mlx",
        "q": q,
        "shape": x.shape,
        "bits": bits,
        "group_size": group_size,
    }


def mlx_dequantize(qpack) -> Optional[np.ndarray]:
    """Reconstruct a float32 embedding matrix from an MLX quantization pack."""
    mx = try_import_mlx()
    if mx is None:
        return None
    x = mx.dequantize(qpack["q"])
    return np.array(x, dtype=np.float32)


def quantize_then_dequantize(
    e: np.ndarray,
    backend: str,
    bits: int,
    group_size: int,
    mode: str,
) -> np.ndarray:
    if backend == "raw":
        qp = raw_groupwise_quantize(e, bits=bits, group_size=group_size, mode=mode)
        return raw_dequantize(qp)
    if backend == "mlx":
        qp = mlx_quantize(e, bits=bits, group_size=group_size)
        if qp is None:
            qp = raw_groupwise_quantize(e, bits=bits, group_size=group_size, mode=mode)
            return raw_dequantize(qp)
        out = mlx_dequantize(qp)
        if out is None:
            qp = raw_groupwise_quantize(e, bits=bits, group_size=group_size, mode=mode)
            return raw_dequantize(qp)
        return out
    raise ValueError("Unknown backend")


# -----------------------------
# Evaluator
# -----------------------------
@dataclass
class STSBenchmarkEvaluator:
    """Evaluator for GLUE STS-B benchmarking with quantization and energy tracking.

    Extends the base STS-B evaluator with groupwise embedding quantization
    (raw NumPy or Apple MLX) as an additional search parameter.

    Attributes:
        split: HuggingFace dataset split to use (e.g. ``"validation"``).
        max_pairs: Maximum number of sentence pairs to evaluate. ``None`` uses the full split.
        warmup_pairs: Number of sentence pairs used for a warm-up pass (not measured).
        track_dir: Directory for CodeCarbon log files; created automatically if absent.
    """
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
        quant_backend: str,
        quant_bits: int,
        quant_group: int,
        quant_mode: str,
        quantize_embeddings: bool,
        track_quant_energy: bool,
    ) -> Tuple[float, float, float]:
        """Evaluate a single embedding + quantization configuration on STS-B.

        Args:
            model_name: HuggingFace model identifier (``sentence-transformers/...``).
            target_dim: Target dimensionality after optional PCA reduction.
            batch_size: Encoding batch size passed to ``SentenceTransformer.encode``.
            normalize: Whether to L2-normalize embeddings after encoding.
            track_pca_energy: If ``True``, include PCA inside the energy measurement window.
            quant_backend: Quantization backend to use (``"raw"`` or ``"mlx"``).
            quant_bits: Bit-width for quantization (4 or 8).
            quant_group: Group size for groupwise quantization.
            quant_mode: Quantization scheme (``"symmetric"`` or ``"affine"``).
            quantize_embeddings: Whether to apply quantize-then-dequantize to embeddings.
            track_quant_energy: If ``True``, include quantization inside the energy window.

        Returns:
            A tuple ``(latency_ms, energy_kwh, pearson_r)`` where:
            - ``latency_ms``  -- wall-clock time for the measured region in ms
            - ``energy_kwh``  -- CodeCarbon energy estimate in kWh
            - ``pearson_r``   -- Pearson correlation with gold STS-B scores

        Note:
            Model loading, dataset loading, and warm-up passes are excluded from
            the measured region.  The MLX backend falls back to raw NumPy quantization
            automatically if ``mlx`` is not installed.
        """
        model = self._get_model(model_name)

        # Warmup (not tracked)
        if self.warmup_pairs and self.warmup_pairs > 0:
            w = min(self.warmup_pairs, len(self.s1))
            _ = model.encode(
                self.s1[:w],
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )
            _ = model.encode(
                self.s2[:w],
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )

        log_dir = os.path.abspath(self.track_dir)
        os.makedirs(log_dir, exist_ok=True)

        tracker = None
        energy_kwh = np.nan

        try:
            tracker = EmissionsTracker(
                project_name="sts_moo_quant",
                output_dir=log_dir,
                measure_power_secs=1,
                save_to_file=False,
                log_level="error",
            )
            tracker.start()
        except Exception as e:
            print(f"[WARN] CodeCarbon disabled: {e}")
            tracker = None

        t0 = time.perf_counter()

        e1 = model.encode(
            self.s1,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=normalize,
        )
        e2 = model.encode(
            self.s2,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=normalize,
        )

        if track_pca_energy:
            e1, e2 = self._maybe_pca(e1, e2, target_dim)

        if quantize_embeddings and track_quant_energy:
            e1 = ensure_dim_divisible(e1, quant_group)
            e2 = ensure_dim_divisible(e2, quant_group)
            e1 = quantize_then_dequantize(e1, quant_backend, quant_bits, quant_group, quant_mode)
            e2 = quantize_then_dequantize(e2, quant_backend, quant_bits, quant_group, quant_mode)

        t1 = time.perf_counter()

        if tracker is not None:
            try:
                tracker.stop()
                fed = getattr(tracker, "final_emissions_data", None)
                if fed is not None:
                    energy_kwh = float(getattr(fed, "energy_consumed", np.nan))
            except Exception:
                energy_kwh = np.nan

        latency_ms = (t1 - t0) * 1000.0

        # If PCA not tracked, do it outside tracked region
        if not track_pca_energy:
            e1, e2 = self._maybe_pca(e1, e2, target_dim)

        # If quant not tracked, do it outside tracked region
        if quantize_embeddings and (not track_quant_energy):
            e1 = ensure_dim_divisible(e1, quant_group)
            e2 = ensure_dim_divisible(e2, quant_group)
            e1 = quantize_then_dequantize(e1, quant_backend, quant_bits, quant_group, quant_mode)
            e2 = quantize_then_dequantize(e2, quant_backend, quant_bits, quant_group, quant_mode)

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
        "quantize_embeddings",
        "quant_backend",
        "quant_bits",
        "quant_group",
        "quant_mode",
        "track_quant_energy",
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
            str(t.params["quantize_embeddings"]),
            str(t.params["quant_backend"]),
            str(t.params["quant_bits"]),
            str(t.params["quant_group"]),
            str(t.params["quant_mode"]),
            str(t.params["track_quant_energy"]),
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
        qflag = t.params["quantize_embeddings"]
        qdesc = "off"
        if qflag:
            qdesc = f"{t.params['quant_backend']}, {t.params['quant_bits']}-bit, g={t.params['quant_group']}, {t.params['quant_mode']}"
        hover_text.append(
            f"<b>{pid}</b><br>"
            f"Model: {short_model(t.params['model_name'])}<br>"
            f"Dim: {t.params['target_dim']}<br>"
            f"Batch: {t.params['batch_size']}<br>"
            f"Normalize: {t.params['normalize']}<br>"
            f"PCA energy tracked: {t.params['track_pca_energy']}<br>"
            f"Quant: {qdesc}<br>"
            f"Quant energy tracked: {t.params['track_quant_energy']}<br>"
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
        title="Interactive 3D Pareto Front (STS-B, embedding quantization knobs)",
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
    pdf_la: str = "fig_proj_latency_accuracy.pdf",
    pdf_ea: str = "fig_proj_energy_accuracy.pdf",
    label_points: bool = False,
):
    configure_matplotlib_for_paper()

    ids = [f"P{i:02d}" for i in range(1, len(pareto_trials) + 1)]
    lat = np.array([t.values[0] for t in pareto_trials], dtype=float)
    eng = np.array([t.values[1] for t in pareto_trials], dtype=float)
    acc = np.array([t.values[2] for t in pareto_trials], dtype=float)

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
    ax.set_zlabel("Pearson r ↑")
    ax.set_title("Pareto-optimal STS-B configurations (quantization included)")

    ax.view_init(elev=18, azim=230)

    if label_points:
        max_labels = 12
        idx = np.argsort(-score)[:max_labels].tolist()
        for i in idx:
            ax.text(lat[i], eng[i], acc[i], f" {ids[i]}", fontsize=8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.08)
    cbar.set_label("Pearson r")
    fig.savefig(pdf_3d)
    plt.close(fig)

    fig = plt.figure(figsize=PAPER_FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    ax.scatter(lat, acc, s=26, linewidth=0.3)
    ax.set_xlabel("Latency (ms) ↓")
    ax.set_ylabel("Pearson r ↑")
    ax.set_title("Pareto projection: latency vs accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(pdf_la)
    plt.close(fig)

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

    mlx_ok = try_import_mlx() is not None
    if mlx_ok:
        print("MLX detected: MLX backend enabled.")
    else:
        print("MLX not detected: MLX backend will fall back to raw quantization.")

    evaluator = STSBenchmarkEvaluator(
        split="validation",
        max_pairs=MAX_PAIRS,
        warmup_pairs=WARMUP_PAIRS,
        track_dir=CC_DIR,
    )

    model_candidates = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ]

    quant_backends = ["raw", "mlx"]
    quant_bits = [4, 8]
    quant_groups = [16, 32, 64, 128]
    quant_modes = ["symmetric", "affine"]

    def objective(trial: optuna.trial.Trial):
        model_name = trial.suggest_categorical("model_name", model_candidates)

        target_dim = trial.suggest_int("target_dim", 64, 768, step=32)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        normalize = trial.suggest_categorical("normalize", [True, False])
        track_pca_energy = trial.suggest_categorical("track_pca_energy", [False, True])

        quantize_embeddings = trial.suggest_categorical("quantize_embeddings", [True, False])

        # If quantization is off, still store params for reporting consistency
        quant_backend = trial.suggest_categorical("quant_backend", quant_backends)
        qb = trial.suggest_categorical("quant_bits", quant_bits)

        valid_quant_groups = [g for g in quant_groups if g <= target_dim]
        if not valid_quant_groups:
            valid_quant_groups = [target_dim]
        qg = trial.suggest_categorical("quant_group", valid_quant_groups)

        qm = trial.suggest_categorical("quant_mode", quant_modes)

        # Track quant energy only matters if quantization is on
        track_quant_energy = trial.suggest_categorical("track_quant_energy", [False, True])

        return evaluator.evaluate(
            model_name=model_name,
            target_dim=target_dim,
            batch_size=batch_size,
            normalize=normalize,
            track_pca_energy=track_pca_energy,
            quant_backend=quant_backend,
            quant_bits=qb,
            quant_group=qg,
            quant_mode=qm,
            quantize_embeddings=quantize_embeddings,
            track_quant_energy=track_quant_energy,
        )

    print("Running 3-objective MOO on GLUE STS-B (validation split)")
    print("Objectives: minimize latency, minimize energy, maximize Pearson correlation")
    study = optuna.create_study(directions=["minimize", "minimize", "maximize"])
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    pareto = study.best_trials
    print(f"\nPareto-optimal solutions: {len(pareto)}")

    print(
        f"{'Point':<6} | {'Latency(ms)':<12} | {'Energy(kWh)':<12} | {'Pearson':<8} | "
        f"{'Model':<32} | {'Dim':<5} | {'BS':<4} | {'Norm':<5} | {'PCA_E':<5} | "
        f"{'Q':<1} | {'QBackend':<7} | {'Bits':<4} | {'Group':<5} | {'Mode':<9} | {'Q_E':<3}"
    )
    print("-" * 180)
    for i, t in enumerate(pareto, start=1):
        pid = f"P{i:02d}"
        qflag = t.params["quantize_embeddings"]
        print(
            f"{pid:<6} | {t.values[0]:<12.2f} | {t.values[1]:<12.6f} | {t.values[2]:<8.4f} | "
            f"{short_model(t.params['model_name']):<32} | {t.params['target_dim']:<5} | {t.params['batch_size']:<4} | "
            f"{str(t.params['normalize']):<5} | {str(t.params['track_pca_energy']):<5} | "
            f"{('1' if qflag else '0'):<1} | {str(t.params['quant_backend']):<7} | {t.params['quant_bits']:<4} | "
            f"{t.params['quant_group']:<5} | {str(t.params['quant_mode']):<9} | {str(t.params['track_quant_energy']):<3}"
        )

    export_pareto_csv(pareto, "pareto_solutions.csv")
    plot_pareto_interactive_html(pareto, out_html="pareto_3d_interactive.html")

    plot_paper_figures_matplotlib(
        pareto,
        pdf_3d="fig_pareto3d.pdf",
        pdf_la="fig_proj_latency_accuracy.pdf",
        pdf_ea="fig_proj_energy_accuracy.pdf",
        label_points=False,
    )


if __name__ == "__main__":
    main()
