"""Gradio app: pick a sweep hour-0 model (or real hour-0 data), roll it forward
through the autoregressive model, view the trajectories, and run TSTR.

Run:
    uv run python app.py
"""

from __future__ import annotations

import pickle
import re
import sys
import types
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

import lexisflow
import lexisflow.data
import lexisflow.data.transformers as _transformers
import lexisflow.models
import lexisflow.models.forest_flow as _forest_flow
import lexisflow.models.hs3f as _hs3f
from lexisflow.evaluation import MortalityTask, evaluate_tstr_multi_task
from lexisflow.models import sample_trajectory

# --- Legacy import shims -----------------------------------------------------
# Older pickles reference `packages.models.hs3f`, `src.forest_flow.*`, and
# `synth_gen.*` (pre-rename). Redirect all to current module locations.
for legacy, target in [
    ("packages", types.ModuleType("packages")),
    ("packages.models", types.ModuleType("packages.models")),
    ("packages.models.hs3f", _hs3f),
    ("packages.models.forest_flow", _forest_flow),
    ("src", types.ModuleType("src")),
    ("src.forest_flow", types.ModuleType("src.forest_flow")),
    ("src.forest_flow.preprocessing", _transformers),
    ("src.forest_flow.model", _forest_flow),
    ("src.forest_flow.hs3f", _hs3f),
    ("synth_gen", lexisflow),
    ("synth_gen.data", lexisflow.data),
    ("synth_gen.data.transformers", _transformers),
    ("synth_gen.models", lexisflow.models),
    ("synth_gen.models.forest_flow", _forest_flow),
    ("synth_gen.models.hs3f", _hs3f),
]:
    sys.modules.setdefault(legacy, target)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
SWEEP_DIR = ROOT / "artifacts" / "sweep"
AR_MODEL_PATH = ROOT / "artifacts" / "autoregressive_forest_flow.pkl"
AR_PREP_PATH = ROOT / "artifacts" / "preprocessor_full.pkl"
HOUR0_PREP_PATH = ROOT / "artifacts" / "hour0_preprocessor.pkl"
REAL_TEST_PATH = ROOT / "data" / "processed" / "real_test.csv"


# --- Loaders (cached via module-level singletons) ----------------------------
def _pickle_load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


_AR_CACHE: dict | None = None
_HOUR0_PREP_CACHE: dict | None = None


def load_ar_bundle() -> dict:
    global _AR_CACHE
    if _AR_CACHE is None:
        ar_model = _pickle_load(AR_MODEL_PATH)
        if isinstance(ar_model, dict) and "model" in ar_model:
            ar_model = ar_model["model"]
        prep = _pickle_load(AR_PREP_PATH)
        _AR_CACHE = {"model": ar_model, **prep}
    return _AR_CACHE


def load_hour0_prep() -> dict:
    global _HOUR0_PREP_CACHE
    if _HOUR0_PREP_CACHE is None:
        _HOUR0_PREP_CACHE = _pickle_load(HOUR0_PREP_PATH)
    return _HOUR0_PREP_CACHE


def list_sweep_models() -> list[tuple[str, str]]:
    """Return list of (label, absolute_path) for sweep hour-0 models."""
    rows: list[tuple[str, str, int, int]] = []
    pat = re.compile(r"hour0_nt(\d+)_noise(\d+)\.pkl$")
    for p in sorted(SWEEP_DIR.glob("*.pkl")):
        m = pat.search(p.name)
        if not m:
            continue
        nt, noise = int(m.group(1)), int(m.group(2))
        label = f"nt={nt:<3d} noise={noise}   ({p.name})"
        rows.append((label, str(p), nt, noise))
    rows.sort(key=lambda r: (r[2], r[3]))
    return [(label, path) for label, path, _, _ in rows]


# --- Hour-0 sourcing ---------------------------------------------------------
def sample_hour0_from_model(model_path: str, n: int, seed: int) -> pd.DataFrame:
    """Generate n initial patient states from a sweep hour-0 model."""
    artifact = _pickle_load(Path(model_path))
    model = artifact["model"]
    hour0_prep = load_hour0_prep()
    preprocessor = hour0_prep["preprocessor"]
    X = model.sample(n_samples=n, X_condition=None)
    df = preprocessor.inverse_transform(np.asarray(X))
    df = df.reset_index(drop=True)
    return df


def sample_hour0_from_real(n: int, seed: int) -> pd.DataFrame:
    """Sample n real hour-0 rows from the held-out test set."""
    df = pd.read_csv(REAL_TEST_PATH)
    if "hours_in" in df.columns:
        df = df[df["hours_in"] == 0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


# --- Trajectory generation ---------------------------------------------------
def _build_ar_init_dataframe(df_hour0: pd.DataFrame, ar: dict) -> pd.DataFrame:
    """Build a DataFrame with all AR columns (target + condition) from hour-0 data.

    Lag1 features are seeded from the corresponding hour-0 target values; missing
    columns are filled with 0.
    """
    target_cols: list[str] = ar["target_cols"]
    condition_cols: list[str] = ar["condition_cols"]
    all_cols: list[str] = ar["all_cols"]

    out = pd.DataFrame(index=range(len(df_hour0)))
    for col in target_cols:
        out[col] = df_hour0[col] if col in df_hour0.columns else 0.0

    for col in condition_cols:
        if col.endswith("_lag1"):
            base = col[: -len("_lag1")]
            if base in df_hour0.columns:
                out[col] = df_hour0[base].values
            else:
                out[col] = 0.0
        else:
            out[col] = df_hour0[col] if col in df_hour0.columns else 0.0

    return out[all_cols]


def generate_trajectories(
    df_hour0: pd.DataFrame,
    n_timesteps: int,
    seed: int,
) -> pd.DataFrame:
    ar = load_ar_bundle()
    preprocessor = ar["preprocessor"]
    target_cols: list[str] = ar["target_cols"]
    condition_cols: list[str] = ar["condition_cols"]
    target_indices = ar["target_indices"]

    static_cols = [c for c in condition_cols if not c.endswith("_lag1")]

    df_init = _build_ar_init_dataframe(df_hour0, ar)
    X_full = preprocessor.transform(df_init)

    initial_state = X_full[:, target_indices]

    if static_cols:
        all_cols_idx = {c: i for i, c in enumerate(ar["all_cols"])}
        static_positions = [all_cols_idx[c] for c in static_cols]
        static_features = X_full[:, static_positions]
    else:
        static_features = None

    trajectories = sample_trajectory(
        model=ar["model"],
        preprocessor=preprocessor,
        static_features=static_features,
        n_trajectories=len(df_hour0),
        max_hours=n_timesteps,
        target_cols=target_cols,
        condition_cols=condition_cols,
        static_cols=static_cols if static_cols else None,
        initial_state=initial_state,
        random_state=seed,
        verbose=False,
    )

    frames = []
    for i, t_df in enumerate(trajectories):
        t_df = t_df.copy()
        t_df.insert(0, "subject_id", f"synth_{i}")
        if "timestep" in t_df.columns and "hours_in" not in t_df.columns:
            t_df = t_df.rename(columns={"timestep": "hours_in"})
        frames.append(t_df)
    flat = pd.concat(frames, ignore_index=True)
    return flat


# --- UI callbacks ------------------------------------------------------------
SWEEP_MODELS = list_sweep_models()
SWEEP_LABELS = [label for label, _ in SWEEP_MODELS]
LABEL_TO_PATH = dict(SWEEP_MODELS)


def on_generate(
    source: str, model_label: str, n_patients: int, n_timesteps: int, seed: int
):
    try:
        n_patients = int(n_patients)
        n_timesteps = int(n_timesteps)
        seed = int(seed)
        if source == "Sweep hour-0 model":
            if not model_label or model_label not in LABEL_TO_PATH:
                return None, "Select a sweep model.", None, None
            df_h0 = sample_hour0_from_model(
                LABEL_TO_PATH[model_label], n_patients, seed
            )
        else:
            df_h0 = sample_hour0_from_real(n_patients, seed)

        synth_df = generate_trajectories(df_h0, n_timesteps, seed)
        status = (
            f"Generated {synth_df['subject_id'].nunique()} patients "
            f"× {n_timesteps} hours = {len(synth_df):,} rows, "
            f"{synth_df.shape[1]} columns."
        )
        preview = synth_df.head(200)
        subj_choices = sorted(synth_df["subject_id"].unique().tolist())
        return (
            synth_df,
            status,
            preview,
            gr.Dropdown(
                choices=subj_choices, value=subj_choices[0] if subj_choices else None
            ),
        )
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}", None, None


def on_view_patient(synth_df: pd.DataFrame | None, subject_id: str | None):
    if synth_df is None or subject_id is None:
        return None
    sub = synth_df[synth_df["subject_id"] == subject_id]
    return sub.reset_index(drop=True)


def on_run_tstr(synth_df: pd.DataFrame | None):
    if synth_df is None or len(synth_df) == 0:
        return pd.DataFrame([{"error": "Generate data first."}])
    try:
        real_df = pd.read_csv(REAL_TEST_PATH)
        task = MortalityTask(use_sequence_model=True)
        results = evaluate_tstr_multi_task(
            synth_df=synth_df,
            real_df=real_df,
            tasks=[task],
            test_size=0.3,
            verbose=False,
        )
        rows = []
        for task_name, metrics in results.items():
            for k, v in metrics.items():
                rows.append({"task": task_name, "metric": k, "value": _fmt(v)})
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame([{"error": f"{type(e).__name__}: {e}"}])


def _fmt(v) -> str:
    try:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


# --- UI ----------------------------------------------------------------------
with gr.Blocks(title="LexisFlow Explorer") as demo:
    gr.Markdown(
        "# LexisFlow Explorer\n"
        "Pick a hour-0 source, roll trajectories forward through the autoregressive "
        "model, inspect patients, and run TSTR against the real held-out test set."
    )

    synth_state = gr.State(value=None)

    with gr.Tab("1. Generate"):
        source = gr.Radio(
            choices=["Sweep hour-0 model", "Real hour-0 data"],
            value="Sweep hour-0 model",
            label="Hour-0 source",
        )
        model_dropdown = gr.Dropdown(
            choices=SWEEP_LABELS,
            value=SWEEP_LABELS[0] if SWEEP_LABELS else None,
            label="Sweep model",
            interactive=True,
        )
        with gr.Row():
            n_patients = gr.Slider(5, 500, value=25, step=5, label="# patients")
            n_timesteps = gr.Slider(
                2, 48, value=10, step=1, label="# timesteps (hours)"
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
        gen_btn = gr.Button("Generate trajectories", variant="primary")
        status = gr.Markdown()
        preview_table = gr.Dataframe(
            label="Preview (first 200 rows)", interactive=False, wrap=True
        )

        def _toggle_model(choice):
            return gr.Dropdown(interactive=(choice == "Sweep hour-0 model"))

        source.change(_toggle_model, inputs=source, outputs=model_dropdown)

    with gr.Tab("2. View patient"):
        gr.Markdown("Select a patient to see their full trajectory.")
        subject_picker = gr.Dropdown(choices=[], label="Subject")
        patient_table = gr.Dataframe(label="Trajectory", interactive=False, wrap=True)
        subject_picker.change(
            on_view_patient, inputs=[synth_state, subject_picker], outputs=patient_table
        )

    with gr.Tab("3. TSTR evaluation"):
        gr.Markdown(
            "Train a mortality classifier on the synthetic trajectories, test on "
            f"`{REAL_TEST_PATH.relative_to(ROOT)}`. Compares against a classifier "
            "trained on real data (reported as `real_*` metrics)."
        )
        tstr_btn = gr.Button("Run TSTR", variant="primary")
        tstr_table = gr.Dataframe(label="Metrics", interactive=False)
        tstr_btn.click(on_run_tstr, inputs=synth_state, outputs=tstr_table)

    gen_btn.click(
        on_generate,
        inputs=[source, model_dropdown, n_patients, n_timesteps, seed],
        outputs=[synth_state, status, preview_table, subject_picker],
    )


if __name__ == "__main__":
    demo.launch()
