#!/usr/bin/env python3
"""
wandb_to_matplotlib.py
----------------------
Download run histories from Weights & Biases and recreate charts with matplotlib.

Quick start:
1) pip install wandb pandas matplotlib numpy
2) Edit the CONFIG section below (PROJECT, RUN_IDS or FILTERS, METRICS).
3) Run: python wandb_to_matplotlib.py

Outputs:
- data/combined_history.csv               (all runs concatenated)
- data/<run_id>.csv                       (per-run history)
- figs/<metric>.png                       (mean curve with std band across runs)
- figs/<metric>__per_run.png              (overlay of each run's curve)

Notes:
- We DO NOT set any matplotlib style or colors to keep it journal-neutral.
- If your runs log 'epoch', we'll use that. Otherwise we fallback to '_step' or 'step'.
"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb


# ---------------------- CONFIG ----------------------

CUSTOM_NAMES_BY_NAME = {
    "upbeat-surf-315": "With ATC",
    "winter-monkey-317": "Scone",
    "elated-blaze-314": "With AC",
}
# Pretty axis titles per metric (clean, no "Job" text)
METRIC_LABELS = {
    "fpr95_test":        "FPR95 on Test Data",
    "test_accuracy":     "Test Accuracy",
    "test_accuracy_cor": "Test Accuracy on Corrupted Data",
}

# Toggle the right-side panel
USE_SIDE_PANEL = False

# Right-side panel heading + text (edit these lines to match your setup)
PANEL_HEADING = "Dataset Mapping Using Transformer"
PANEL_TEXT = (
    "ID: Clear\n"
    "Covariate: Clear-C (Gaussian)\n"
    "Semantic OOD: Places365"
)

@dataclass
class Config:
    PROJECT: str = "anaiknaware7153-san-diego-state-university/oodproject"
    RUN_IDS: List[str] = ()
    FILTERS = {"group": "ViT-Updated(Clear and Places365)", "state": "finished"}

    # Which metric keys to pull/plot
    METRICS: List[str] = (
        "fpr95_test",
        "test_accuracy",
        "test_accuracy_cor",
    )

    # X-axis by timesteps (1..N)
    X_AXIS_MODE: str = "timestep"   # "timestep" or "epoch"
    TIMESTEPS_N: int = 10

    # Smoothing / resampling
    EMA_ALPHA: float = 0.0
    RESAMPLE: bool = True
    RESAMPLE_POINTS: int = 200       # general-purpose; we'll force 10 in timestep mode

    # Output dirs
    OUT_DIR_DATA: str = "ViT-updated3(OOD-Place365)/data"
    OUT_DIR_FIGS: str = "ViT-updated3(OOD-Place365)/figs"

    FIG_W: float = 4.0               # <-- wider charts
    FIG_H: float = 3.0
    DPI: int = 200
    LEGEND_FONTSIZE: int = 8
CFG = Config()


# ---------------------- Helpers ----------------------
def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def pick_x_axis(df: pd.DataFrame) -> str:
    for cand in ("epoch", "_step", "step"):
        if cand in df.columns:
            return cand
    # last resort: use index
    df["index_step"] = np.arange(len(df), dtype=float)
    return "index_step"


def ema_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or alpha >= 1 or len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * y[i]
    return out


def resample_xy(x: np.ndarray, y: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 2 or n_points <= 0:
        return x, y
    x_new = np.linspace(np.nanmin(x), np.nanmax(x), n_points)
    # numeric interpolate with NaN handling
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return x, y
    y_new = np.interp(x_new, x[mask], y[mask])
    return x_new, y_new

def force_timesteps_axis(x: np.ndarray, n: int) -> np.ndarray:
    """Map any x-array to 1..n (inclusive) for discrete timestep display."""
    if n <= 0:
        return x
    return np.linspace(1, n, n)
    
def get_runs(project: str, run_ids: Optional[List[str]] = None, flt: Optional[Dict] = None):
    api = wandb.Api()
    if run_ids:
        runs = [api.run(f"{project}/{rid}") for rid in run_ids]
    else:
        runs = api.runs(project, filters=flt)
    # Keep only finished/active runs
    runs = [r for r in runs if r.state in ("finished", "running", "crashed")]
    return runs


def load_history(run, metrics: List[str]) -> pd.DataFrame:
    # Pull all columns to ensure we find step/epoch; then subset later
    hist = run.history(samples=None, pandas=True)
    # Some columns are nested dicts; keep only numeric/str/bool
    safe_cols = []
    for c in hist.columns:
        if pd.api.types.is_numeric_dtype(hist[c]) or pd.api.types.is_bool_dtype(hist[c]) or pd.api.types.is_string_dtype(hist[c]):
            safe_cols.append(c)
    hist = hist[safe_cols]
    # Subset to metrics + x-axis candidates
    cols = set(metrics) | {"epoch", "_step", "step"}
    cols = [c for c in hist.columns if c in cols]
    return hist[cols].copy()


def save_csvs(per_run: Dict[str, pd.DataFrame], out_dir: str) -> pd.DataFrame:
    ensure_dirs(out_dir)
    combo = []
    for run_id, df in per_run.items():
        df_out = df.copy()
        df_out["run_id"] = run_id
        combo.append(df_out)
        df_out.to_csv(os.path.join(out_dir, f"{run_id}.csv"), index=False)
    combined = pd.concat(combo, ignore_index=True) if combo else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(os.path.join(out_dir, "combined_history.csv"), index=False)
    return combined


from matplotlib.gridspec import GridSpec



def plot_metric(per_run: Dict[str, pd.DataFrame], metric: str, out_dir: str,
                ema_alpha: float = 0.0, resample: bool = True, resample_points: int = 200) -> None:
    ensure_dirs(out_dir)

    y_label = METRIC_LABELS.get(metric, metric)

    # -------------------- PER-RUN OVERLAY --------------------
    fig = plt.figure(figsize=(CFG.FIG_W, CFG.FIG_H))
    ax = plt.gca()

    y_all = []  # collect all y for optional padding
    for run_label, df in per_run.items():
        if metric not in df.columns:
            continue
        x_name = pick_x_axis(df)
        x = df[x_name].to_numpy(dtype=float)
        y = df[metric].to_numpy(dtype=float)

        if ema_alpha > 0:
            y = ema_smooth(y, ema_alpha)
        if resample:
            x, y = resample_xy(x, y, resample_points)

        # If in timestep mode, coerce to 1..N
        if CFG.X_AXIS_MODE.lower() == "timestep":
            x, y = resample_xy(x, y, CFG.TIMESTEPS_N)
            x = np.arange(1, CFG.TIMESTEPS_N + 1, dtype=float)

        ax.plot(x, y, label=run_label, linewidth=1.8, marker="o", markersize=3)
        y_all.append(y)

    # Labels only (no headline/title)
    ax.set_xlabel("Timesteps" if CFG.X_AXIS_MODE.lower() == "timestep" else "Epochs", fontweight="bold")
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.margins(x=0.02)

    if CFG.X_AXIS_MODE.lower() == "timestep":
        ax.set_xlim(1, CFG.TIMESTEPS_N)
        ax.set_xticks(np.arange(1, CFG.TIMESTEPS_N + 1))

    # ---- Conditional padding + legend placement ----
    if metric in ("test_accuracy", "test_accuracy_cor"):
        # Add vertical headroom so legend sits in white space
        if y_all:
            Y = np.concatenate([np.asarray(v, dtype=float) for v in y_all])
            Y = Y[~np.isnan(Y)]
            if Y.size > 0:
                y_min, y_max = float(np.min(Y)), float(np.max(Y))
                dy = max(1e-9, y_max - y_min)
                pad = 0.08 * dy
                ax.set_ylim(y_min, y_max + pad)

        # Legend INSIDE top-left
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            fontsize=CFG.LEGEND_FONTSIZE,
            frameon=True,
            fancybox=True,
            framealpha=0.85,
            borderpad=0.6,
            labelspacing=0.4,
            handlelength=1.8,
            handletextpad=0.6,
        )
    else:
        # FPR95 and others: tight bounds, legend INSIDE top-right
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=CFG.LEGEND_FONTSIZE,
            frameon=True,
            fancybox=True,
            framealpha=0.85,
            borderpad=0.6,
            labelspacing=0.4,
            handlelength=1.8,
            handletextpad=0.6,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{metric}__per_run.png"),
                dpi=CFG.DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # -------------------- MEAN ± STD --------------------
    series, x_ref = [], None
    for _, df in per_run.items():
        if metric not in df.columns:
            continue
        x_name = pick_x_axis(df)
        x = df[x_name].to_numpy(dtype=float)
        y = df[metric].to_numpy(dtype=float)

        if ema_alpha > 0:
            y = ema_smooth(y, ema_alpha)
        if resample:
            x, y = resample_xy(x, y, resample_points)

        if CFG.X_AXIS_MODE.lower() == "timestep":
            x, y = resample_xy(x, y, CFG.TIMESTEPS_N)
            x = np.arange(1, CFG.TIMESTEPS_N + 1, dtype=float)

        if x_ref is None:
            x_ref = x
        else:
            if len(x) != len(x_ref) or not np.allclose([x[0], x[-1]], [x_ref[0], x_ref[-1]]):
                y = np.interp(x_ref, x, y)
        series.append(y)

    if series and x_ref is not None:
        Y = np.vstack(series)
        mean, std = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)

        fig = plt.figure(figsize=(CFG.FIG_W, CFG.FIG_H))
        axm = plt.gca()
        axm.plot(x_ref, mean, label="mean", linewidth=1.8)
        axm.fill_between(x_ref, mean - std, mean + std, alpha=0.2, label="±1 std")

        axm.set_xlabel("Timesteps" if CFG.X_AXIS_MODE.lower() == "timestep" else "Epochs", fontweight="bold")
        axm.set_ylabel(y_label)
        axm.grid(True, which="both", alpha=0.25, linewidth=0.6)

        if CFG.X_AXIS_MODE.lower() == "timestep":
            axm.set_xlim(1, CFG.TIMESTEPS_N)
            axm.set_xticks(np.arange(1, CFG.TIMESTEPS_N + 1))

        # Same legend/padding rule for the mean±std figure
        if metric in ("test_accuracy", "test_accuracy_cor"):
            y_min, y_max = float(np.nanmin(Y)), float(np.nanmax(Y))
            dy = max(1e-9, y_max - y_min)
            pad = 0.08 * dy
            axm.set_ylim(y_min, y_max + pad)
            axm.legend(
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                fontsize=CFG.LEGEND_FONTSIZE,
                frameon=True,
                fancybox=True,
                framealpha=0.85,
                borderpad=0.6,
                labelspacing=0.4,
                handlelength=1.8,
                handletextpad=0.6,
            )
        else:
            axm.legend(
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                fontsize=CFG.LEGEND_FONTSIZE,
                frameon=True,
                fancybox=True,
                framealpha=0.85,
                borderpad=0.6,
                labelspacing=0.4,
                handlelength=1.8,
                handletextpad=0.6,
            )

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{metric}.png"),
                    dpi=CFG.DPI, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    
def main(cfg: Config, metrics_cli: Optional[List[str]] = None):
    project = cfg.PROJECT
    run_ids = list(cfg.RUN_IDS) if cfg.RUN_IDS else []
    filters = cfg.FILTERS

    runs = get_runs(project, run_ids if run_ids else None, filters if not run_ids else None)
    if not runs:
        print("No runs found. Check PROJECT / RUN_IDS / FILTERS.")
        return

    # Load histories
    per_run = {}
    for r in runs:
        df = load_history(r, list(metrics_cli) if metrics_cli else list(cfg.METRICS))
        # Attach a useful run label (id or name)
        run_id = r.id
        label_base = CUSTOM_NAMES_BY_NAME.get(r.name, getattr(r, "display_name", None) or r.name or run_id)
        run_label = f"{label_base}"
        per_run[run_label] = df

    # Save CSVs
    combined = save_csvs(per_run, cfg.OUT_DIR_DATA)
    print(f"Saved CSVs under: {cfg.OUT_DIR_DATA}")
    if not combined.empty:
        print("Columns available:", sorted(set(combined.columns) - {"run_id"}))

    # Determine which metrics to actually plot (only those present in any run)
    metrics_all = list(metrics_cli) if metrics_cli else list(cfg.METRICS)
    present_metrics = []
    for m in metrics_all:
        if any(m in df.columns for df in per_run.values()):
            present_metrics.append(m)
        else:
            print(f"Warning: metric '{m}' not found in any run history. Skipping.")

    # Plot
    ensure_dirs(cfg.OUT_DIR_FIGS)
    for m in present_metrics:
        plot_metric(per_run, m, cfg.OUT_DIR_FIGS, ema_alpha=cfg.EMA_ALPHA,
                    resample=cfg.RESAMPLE, resample_points=cfg.RESAMPLE_POINTS)
    print(f"Saved figures under: {cfg.OUT_DIR_FIGS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=None, help='entity/project path')
    parser.add_argument("--runs", type=str, default=None, help='Comma-separated list of run IDs')
    parser.add_argument("--metrics", type=str, default=None, help='Comma-separated metric keys to plot')
    parser.add_argument("--ema", type=float, default=None, help='EMA smoothing alpha (0-1, 0 = no smoothing)')
    parser.add_argument("--resample", action="store_true", help='Resample curves to a common axis')
    parser.add_argument("--no-resample", action="store_true", help='Do not resample curves')
    parser.add_argument("--points", type=int, default=None, help='Number of resample points')
    args = parser.parse_args()

    if args.project:
        CFG.PROJECT = args.project
    if args.runs:
        CFG.RUN_IDS = tuple([s.strip() for s in args.runs.split(",") if s.strip()])
    if args.metrics:
        CFG.METRICS = tuple([s.strip() for s in args.metrics.split(",") if s.strip()])
    if args.ema is not None:
        CFG.EMA_ALPHA = float(args.ema)
    if args.resample and not args.no_resample:
        CFG.RESAMPLE = True
    if args.no_resample:
        CFG.RESAMPLE = False
    if args.points is not None:
        CFG.RESAMPLE_POINTS = int(args.points)

    main(CFG, metrics_cli=list(CFG.METRICS))
