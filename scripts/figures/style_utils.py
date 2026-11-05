"""
Location: // fgns_engine/scripts/figures/style_utils.py
Description: Shared Publication-Grade Styling, ARC Lattice Rendering Primitives, and Real
             PyTorch Tensor Extraction Utilities for Nature Physics / PRX / Quanta Magazine
             Visualizations of FGNS-TRM v5.0.0.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import torch

from fgns.config import FGNSConfig
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCTask, load_arc_dataset
from fgns.kernels.spectral_laplacian import PoissonDirichletCavitySolver
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.types import BatchData, ModelOutput

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ARC_PALETTE_HEX: Tuple[str, ...] = (
    "#0F172A",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#94A3B8",
    "#F012BE",
    "#FF851B",
    "#38BDF8",
    "#870C25",
)

ARC_COLOR_NAMES: Tuple[str, ...] = (
    "0: Black (BG)",
    "1: Blue",
    "2: Red",
    "3: Green",
    "4: Yellow",
    "5: Grey",
    "6: Magenta",
    "7: Orange",
    "8: Cyan",
    "9: Maroon",
)

ARC_TEXT_CONTRAST: Tuple[str, ...] = (
    "#94A3B8",
    "#FFFFFF",
    "#FFFFFF",
    "#0F172A",
    "#0F172A",
    "#0F172A",
    "#FFFFFF",
    "#0F172A",
    "#0F172A",
    "#FFFFFF",
)

_MODEL_SINGLETON: Optional[Tuple[FinslerGaugeSheafNetwork, torch.device]] = None
_TASK_MAP_CACHE: Optional[Dict[str, ARCTask]] = None


def apply_nature_prx_rcparams() -> None:
    """
    Configures Matplotlib rcParams to match Nature Physics / Physical Review X
    typography, crisp vector line weights, and clean high-contrast aesthetics.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#0F172A",
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.5,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.dpi": 350,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def get_arc_cmap() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """
    Returns the discrete 10-color ARC colormap and exact integer boundary normalizer.
    """
    cmap = mcolors.ListedColormap(list(ARC_PALETTE_HEX), name="arc_official")
    bounds = np.arange(-0.5, 10.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def render_arc_grid(
    ax: Any,
    grid: np.ndarray,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    show_values: bool = False,
    highlight_cells: Optional[Sequence[Tuple[int, int, str]]] = None,
    highlight_box: Optional[Tuple[int, int, int, int, str, str]] = None,
    show_ticks: bool = False,
    cell_lw: Optional[float] = None,
    title_color: str = "#0F172A",
    title_fontsize: float = 9.8,
) -> None:
    """
    Renders a 2D integer ARC grid onto `ax` with crisp cell borders, optional cell digits,
    optional bounding box overlays, and optional cell highlights.
    """
    h, w = grid.shape
    cmap, norm = get_arc_cmap()
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest", origin="upper", aspect="equal")

    lw = cell_lw if cell_lw is not None else (0.85 if max(h, w) <= 12 else (0.55 if max(h, w) <= 22 else 0.38))
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="#1E293B", linewidth=lw, alpha=0.65)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="#1E293B", linewidth=lw, alpha=0.65)

    if show_values:
        val_fs = 8.5 if max(h, w) <= 10 else (6.8 if max(h, w) <= 16 else 5.2)
        for r in range(h):
            for c in range(w):
                val = int(grid[r, c])
                t_col = ARC_TEXT_CONTRAST[max(0, min(9, val))]
                ax.text(
                    c,
                    r,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=val_fs,
                    fontweight="bold",
                    color=t_col,
                )

    if highlight_cells is not None:
        for r_idx, c_idx, h_col in highlight_cells:
            rect = Rectangle(
                (c_idx - 0.48, r_idx - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor=h_col,
                linewidth=1.8,
                zorder=8,
            )
            ax.add_patch(rect)

    if highlight_box is not None:
        y0, x0, bh, bw, b_col, b_ls = highlight_box
        rect_box = Rectangle(
            (x0 - 0.48, y0 - 0.48),
            bw - 0.04,
            bh - 0.04,
            fill=False,
            edgecolor=b_col,
            linewidth=2.3,
            linestyle=b_ls,
            zorder=9,
        )
        ax.add_patch(rect_box)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    if show_ticks:
        step_x = 1 if w <= 10 else (5 if w <= 25 else 5)
        step_y = 1 if h <= 10 else (5 if h <= 25 else 5)
        ax.set_xticks(np.arange(0, w, step_x))
        ax.set_yticks(np.arange(0, h, step_y))
        ax.tick_params(axis="both", which="major", labelsize=7.2, length=2.5, pad=2.0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("#0F172A")
        spine.set_linewidth(1.1)

    if title is not None:
        if subtitle is not None:
            ax.set_title(
                f"{title}\n{subtitle}",
                fontsize=title_fontsize,
                fontweight="bold",
                color=title_color,
                pad=6.0,
            )
        else:
            ax.set_title(
                title,
                fontsize=title_fontsize,
                fontweight="bold",
                color=title_color,
                pad=5.5,
            )


def draw_panel_badge(
    ax: Any,
    badge_letter: str,
    title_text: str,
    subtitle_text: Optional[str] = None,
    badge_bg: str = "#0F172A",
    x_pos: float = 0.0,
    y_pos: float = 1.04,
    y_offset_pt: float = 26.0,
    use_ax_top: bool = False,
) -> None:
    """
    Draws a Nature Physics / PRX style panel header anchored to the top-left of the
    subplot's GridSpec cell (or axes box when `use_ax_top=True`) with a fixed
    point-based horizontal offset between the letter badge and the title text.
    """
    from matplotlib.transforms import offset_copy

    fig = ax.figure
    subspec = ax.get_subplotspec()
    pos_box = ax.get_position()
    if subspec is not None and not use_ax_top:
        spec_box = subspec.get_position(fig)
        base_x = spec_box.x0
        base_y = spec_box.y1
    else:
        spec_box = subspec.get_position(fig) if subspec is not None else pos_box
        base_x = spec_box.x0
        base_y = pos_box.y1

    trans_badge = offset_copy(fig.transFigure, fig=fig, x=0.0, y=y_offset_pt, units="points")
    trans_title = offset_copy(fig.transFigure, fig=fig, x=25.0, y=y_offset_pt + 1.0, units="points")

    fig.text(
        base_x,
        base_y,
        f" {badge_letter} ",
        transform=trans_badge,
        fontsize=10.0,
        fontweight="bold",
        color="#FFFFFF",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=badge_bg, edgecolor="none"),
        zorder=20,
    )
    fig.text(
        base_x,
        base_y,
        title_text,
        transform=trans_title,
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        va="bottom",
        ha="left",
        zorder=20,
    )


def add_figure_header_banner(
    fig: Any,
    sup_title: str,
    sub_title: str,
    y_top: float = 0.958,
) -> None:
    """
    Renders a clean top title and subtitle banner across the figure with Kaggle
    16:9 safe-frame top padding.
    """
    fig.text(
        0.5,
        y_top,
        sup_title,
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        y_top - 0.028,
        sub_title,
        ha="center",
        va="top",
        fontsize=9.6,
        color="#475569",
    )


def get_cached_fgns_model() -> Tuple[FinslerGaugeSheafNetwork, torch.device]:
    """
    Loads and caches the 6.57M-parameter FinslerGaugeSheafNetwork on CUDA/CPU.
    """
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is not None:
        return _MODEL_SINGLETON

    PoissonDirichletCavitySolver.clear_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FGNSConfig()
    model = FinslerGaugeSheafNetwork(config).to(device)
    ckpt = PROJECT_ROOT / "artifacts" / "checkpoints" / "model_final.pt"
    if ckpt.is_file():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "models" in state and "model" in state["models"]:
            state = state["models"]["model"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model.eval()
    _MODEL_SINGLETON = (model, device)
    return model, device


def get_arc2_eval_task(task_id: str) -> ARCTask:
    """
    Retrieves a specific ARC-AGI-2 Evaluation task with its ground-truth solution.
    """
    global _TASK_MAP_CACHE
    if _TASK_MAP_CACHE is None:
        chal_p = PROJECT_ROOT / "data" / "arc-agi_evaluation_challenges.json"
        sol_p = PROJECT_ROOT / "data" / "arc-agi_evaluation_solutions.json"
        tasks = load_arc_dataset(chal_p, sol_p)
        _TASK_MAP_CACHE = {t.task_id: t for t in tasks}
    return _TASK_MAP_CACHE[task_id]


def run_real_task_inference(task_id: str) -> Tuple[ARCTask, BatchData, ModelOutput]:
    """
    Executes a deterministic forward pass of FinslerGaugeSheafNetwork on the specified
    ARC-AGI-2 Evaluation task and returns (task, batch, output).
    """
    PoissonDirichletCavitySolver.clear_cache()
    model, device = get_cached_fgns_model()
    task = get_arc2_eval_task(task_id)
    support = list(task.train)
    query = [task.test[0]]
    static_k = max(1, min(6, len(support)))
    batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=static_k)
    with torch.no_grad():
        output = model(batch, tau=0.05, current_step_ratio=1.0)
    return task, batch, output


def load_master_ablation_json() -> Dict[str, Any]:
    """
    Loads the empirical ablation report JSON from artifacts/logs/ablations/master_ablation_report.json.
    """
    json_path = PROJECT_ROOT / "artifacts" / "logs" / "ablations" / "master_ablation_report.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
