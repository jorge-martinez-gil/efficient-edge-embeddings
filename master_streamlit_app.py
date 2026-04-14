import io
import os
import sys
import time
import json
import shutil
import traceback
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import streamlit as st

# Optional deps used by your scripts; the app will still load without them,
# but running a module will require them.
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import streamlit.components.v1 as components
except Exception:
    components = None


# -----------------------------
# App config + styling
# -----------------------------
st.set_page_config(
    page_title="Pareto Lab",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* remove top padding a bit */
.block-container { padding-top: 1.0rem; padding-bottom: 3rem; }

/* nicer sidebar */
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
.sidebar-title {
  font-weight: 800;
  font-size: 1.05rem;
  letter-spacing: 0.2px;
  margin-bottom: 0.35rem;
}
.badge {
  display: inline-block;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid rgba(49, 51, 63, 0.15);
  background: rgba(49, 51, 63, 0.04);
}

/* hero card */
.hero {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 18px;
  padding: 18px 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(245,245,255,0.65));
}
.hero h1 { margin: 0; font-size: 2.0rem; }
.hero p  { margin: 0.25rem 0 0 0; opacity: 0.85; }

/* small cards */
.card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 16px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.65);
}
.card h3 { margin-top: 0.1rem; }
hr.soft {
  border: 0;
  height: 1px;
  background: rgba(49, 51, 63, 0.12);
  margin: 0.8rem 0;
}
.small { font-size: 0.9rem; opacity: 0.85; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Paths to your existing software
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

MODULES = {
    "Text Similarity (STS-B) · Pareto MOO": {
        "key": "similarity",
        "path": BASE_DIR / "pareto-similarity.py",
        "artifacts": [
            "pareto_solutions.csv",
            "pareto_3d_interactive.html",
            "fig_pareto3d.pdf",
            "fig_proj_latency_accuracy.pdf",
            "fig_proj_energy_accuracy.pdf",
        ],
        "about": "3-objective optimization on GLUE STS-B: latency, energy, Pearson correlation.",
    },
    "Text Similarity + Quantization (MLX/raw) · Pareto MOO": {
        "key": "mlx",
        "path": BASE_DIR / "pareto-mlx.py",
        "artifacts": [
            "pareto_solutions.csv",
            "pareto_3d_interactive.html",
            "fig_pareto3d.pdf",
            "fig_proj_latency_accuracy.pdf",
            "fig_proj_energy_accuracy.pdf",
        ],
        "about": "STS-B with embedding quantization knobs (raw or MLX when available).",
    },
    "Text Classification (SST-2 / AG News) · Pareto MOO": {
        "key": "classification",
        "path": BASE_DIR / "pareto-classification.py",
        "artifacts": [
            "pareto_solutions.csv",
            "pareto_3d_interactive.html",
            "fig_pareto3d.pdf",
            "fig_proj_latency_accuracy.pdf",
            "fig_proj_energy_accuracy.pdf",
        ],
        "about": "Embeddings + PCA + classifier; 3-objective optimization on SST-2 or AG News.",
    },
    "Text Clustering (AG News) · Pareto MOO": {
        "key": "clustering",
        "path": BASE_DIR / "pareto-clustering.py",
        "artifacts": [
            "pareto_solutions.csv",
            "pareto_3d_interactive.html",
            "fig_pareto3d.pdf",
            "fig_proj_latency_quality.pdf",
            "fig_proj_energy_quality.pdf",
        ],
        "about": "Embeddings + PCA + MiniBatchKMeans; 3-objective optimization on AG News with NMI quality.",
    },
}

DEFAULT_OUTDIR = BASE_DIR / "runs"


# -----------------------------
# Utilities
# -----------------------------
def _safe_read_bytes(p: Path) -> Optional[bytes]:
    try:
        return p.read_bytes()
    except Exception:
        return None

def _safe_read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")

def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def load_module_from_path(name: str, path: Path):
    """Load a Python file as a module, even if filename contains hyphens."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    outdir: Path

def run_module_main(mod_label: str, mod_path: Path, outdir: Path) -> RunResult:
    """Execute module.main() in an isolated working directory and capture output."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Some scripts write into relative paths; run them with cwd=outdir via os.chdir
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    ok = False
    try:
        os.chdir(outdir)

        # Ensure folders expected by hosted runs exist.
        # This fixes codecarbon crashes like:
        # OSError: Folder '.cc_logs' doesn't exist !
        Path(".cc_logs").mkdir(parents=True, exist_ok=True)

        # Ensure module can import relative deps from app folder if needed
        sys.path.insert(0, str(BASE_DIR))

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            module_name = f"pareto_lab_{_slug(mod_label)}_{int(time.time())}"
            mod = load_module_from_path(module_name, mod_path)

            if not hasattr(mod, "main"):
                raise AttributeError(f"{mod_path.name} has no main() function.")
            mod.main()  # type: ignore
        ok = True
    except Exception:
        traceback.print_exc(file=stderr_buf)
        ok = False
    finally:
        os.chdir(old_cwd)
        sys.path = old_sys_path

    return RunResult(ok=ok, stdout=stdout_buf.getvalue(), stderr=stderr_buf.getvalue(), outdir=outdir)


def show_artifacts(artifacts: list[str], outdir: Path):
    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.subheader("Artifacts")

    cols = st.columns([1.2, 1.2, 2.0])
    with cols[0]:
        st.markdown("**Run folder**")
        st.code(str(outdir), language="text")

    with cols[1]:
        st.markdown("**Quick actions**")
        if st.button("Open run folder in file browser (local)", disabled=True, help="Streamlit cannot open OS file browser in hosted mode."):
            pass
        if st.button("Zip this run", key=f"zip_{outdir.name}"):
            zip_path = outdir.with_suffix(".zip")
            if zip_path.exists():
                zip_path.unlink()
            shutil.make_archive(str(outdir), "zip", root_dir=str(outdir))
            st.success("Created zip archive.")
            zbytes = _safe_read_bytes(zip_path)
            if zbytes:
                st.download_button(
                    "Download run.zip",
                    data=zbytes,
                    file_name=f"{outdir.name}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    with cols[2]:
        st.markdown("**Available files**")
        existing = []
        for a in artifacts:
            p = outdir / a
            if p.exists():
                existing.append(a)
        if not existing:
            st.info("No artifacts found yet in this run folder.")
        else:
            st.write(", ".join(existing))

    # Show the CSV (if present)
    csv_path = outdir / "pareto_solutions.csv"
    if csv_path.exists() and pd is not None:
        try:
            df = pd.read_csv(csv_path)
            st.markdown("#### Pareto solutions (table)")
            st.dataframe(df, use_container_width=True, height=320)
            st.download_button(
                "Download pareto_solutions.csv",
                data=csv_path.read_bytes(),
                file_name="pareto_solutions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            st.warning("Could not render pareto_solutions.csv as a table.")

    # Show interactive HTML
    html_path = outdir / "pareto_3d_interactive.html"
    if html_path.exists() and components is not None:
        st.markdown("#### Interactive 3D Pareto plot")
        html = _safe_read_text(html_path)
        if html:
            components.html(html, height=650, scrolling=True)

        st.download_button(
            "Download pareto_3d_interactive.html",
            data=html_path.read_bytes(),
            file_name="pareto_3d_interactive.html",
            mime="text/html",
            use_container_width=True,
        )

    # Offer PDFs as downloads
    pdfs = [a for a in artifacts if a.lower().endswith(".pdf")]
    if pdfs:
        st.markdown("#### PDFs")
        pdf_cols = st.columns(min(3, len(pdfs)))
        j = 0
        for pdf_name in pdfs:
            p = outdir / pdf_name
            if not p.exists():
                continue
            with pdf_cols[j % len(pdf_cols)]:
                st.download_button(
                    f"Download {pdf_name}",
                    data=p.read_bytes(),
                    file_name=pdf_name,
                    mime="application/pdf",
                    use_container_width=True,
                )
            j += 1


# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🧭 Pareto Lab</div>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Master Streamlit app</span>", unsafe_allow_html=True)
    st.markdown("")
    page = st.selectbox(
        "Choose a module",
        ["Home"] + list(MODULES.keys()),
        index=0,
    )
    st.markdown("")
    st.caption("Tip: start with smaller trials to validate everything, then scale up.")


# -----------------------------
# Home
# -----------------------------
if page == "Home":
    st.markdown(
        """
<div class="hero">
  <h1>🧪 Pareto Lab</h1>
  <p>One place to run your multi-objective experiments, compare trade-offs, and export paper-ready figures.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    with c1:
        st.markdown("<div class='card'><h3>What is inside</h3>"
                    "<p class='small'>A unified UI for your four scripts: similarity, similarity+quantization, classification, clustering.</p>"
                    "<hr class='soft'/>"
                    "<p class='small mono'>Artifacts: CSV, offline Plotly HTML, vector PDFs.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>How it runs</h3>"
                    "<p class='small'>Each module runs in its own timestamped folder under <span class='mono'>runs/</span>.</p>"
                    "<hr class='soft'/>"
                    "<p class='small'>Stdout and errors are captured so debugging is painless.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>Setup</h3>"
                    "<p class='small mono'>pip install streamlit optuna sentence-transformers datasets scikit-learn codecarbon plotly matplotlib pandas</p>"
                    "<hr class='soft'/>"
                    "<p class='small'>Optional (Apple Silicon): <span class='mono'>pip install mlx</span></p></div>", unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.subheader("Modules")
    for label, meta in MODULES.items():
        with st.expander(label, expanded=False):
            st.write(meta["about"])
            st.code(str(meta["path"]), language="text")
            st.write("Expected artifacts:")
            st.write(", ".join(meta["artifacts"]))

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.subheader("Recommended workflow")
    st.write(
        "- Run one module with low trials first.\n"
        "- Verify outputs render inside the app.\n"
        "- Increase trials and sample sizes for final experiments.\n"
        "- Zip a run to archive results with the paper."
    )

else:
    meta = MODULES[page]
    st.markdown(
        f"""
<div class="hero">
  <h1>📌 {page}</h1>
  <p>{meta["about"]}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Controls
    left, right = st.columns([1.1, 1.9], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Run settings")
        run_name = st.text_input("Run label", value=f"{meta['key']}-{_now_tag()}")
        outdir = DEFAULT_OUTDIR / run_name

        st.markdown("**Execution**")
        st.write("Runs the script's `main()` in a dedicated folder.")
        go = st.button("▶ Run now", use_container_width=True)

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        st.markdown("### Existing runs")
        DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
        existing = sorted([p.name for p in DEFAULT_OUTDIR.iterdir() if p.is_dir()], reverse=True)
        pick = st.selectbox("Open an existing run folder", ["(none)"] + existing, index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Console")
        if go:
            with st.spinner("Running…"):
                res = run_module_main(page, meta["path"], outdir=outdir)
            if res.ok:
                st.success("Run finished.")
            else:
                st.error("Run failed. See errors below.")
            st.session_state[f"last_run_{meta['key']}"] = str(res.outdir)

            if res.stdout.strip():
                st.code(res.stdout, language="text")
            if res.stderr.strip():
                st.markdown("**Errors**")
                st.code(res.stderr, language="text")

        st.markdown("</div>", unsafe_allow_html=True)

        # Show artifacts from last run or selected run
        show_dir: Optional[Path] = None
        if pick != "(none)":
            show_dir = DEFAULT_OUTDIR / pick
        else:
            last = st.session_state.get(f"last_run_{meta['key']}")
            if last:
                show_dir = Path(last)

        if show_dir and show_dir.exists():
            show_artifacts(meta["artifacts"], outdir=show_dir)
