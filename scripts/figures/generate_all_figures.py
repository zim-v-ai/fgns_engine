"""
Location: // fgns_engine/scripts/figures/generate_all_figures.py
Description: Master Orchestrator for Building & Running the Native Rust 3D GGX Manifold
             Ray-Tracer, Generating All Eight Publication-Grade Scientific Figures
             (PNG 350 DPI + Vector PDF), and Exporting the Standalone Figure Empirical
             Audit Report (`figures_empirical_audit.json` & `.md`) with Full 2D Grids & Matrices.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fgns.kernels.cuda_fgns_ops import is_cuda_kernel_image_usable
from scripts.figures.fig00_kaggle_card_thumbnail import generate_fig00
from scripts.figures.fig01_na3d_finsler_lie_3d import generate_fig01
from scripts.figures.fig02_architecture_3d_stack import generate_fig02
from scripts.figures.fig03_sheaf_cauchy_dynamics_3d import generate_fig03
from scripts.figures.fig04_arc2_visual_case_studies import generate_fig04
from scripts.figures.fig05_ablation_scientific_dashboard import generate_fig05
from scripts.figures.fig06_topological_operators_gallery import generate_fig06
from scripts.figures.fig07_spectral_mdl_ttt_dynamics import generate_fig07
from scripts.figures.style_utils import get_figure_audit_registry


def build_and_run_rust_manifold_engine() -> None:
    """
    Compiles and executes `tools/fgns_manifold_engine` via Cargo in release mode,
    then converts the resulting 1600x1200 PPM renders into lossless PNGs.
    """
    rust_crate_dir = PROJECT_ROOT / "tools" / "fgns_manifold_engine"
    renders_dir = PROJECT_ROOT / "artifacts" / "rust_renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    if (rust_crate_dir / "Cargo.toml").is_file():
        print(">>> [RUST 3D ENGINE] Compiling & executing Cook-Torrance GGX 3D ray-tracer...")
        try:
            subprocess.run(
                ["cargo", "run", "--release"],
                cwd=str(rust_crate_dir),
                check=True,
            )
        except Exception as exc:
            print(f"[WARN] Cargo execution note: {exc}")

    for ppm_path in sorted(renders_dir.glob("*.ppm")):
        png_path = ppm_path.with_suffix(".png")
        img = Image.open(ppm_path).convert("RGB")
        img.save(png_path)
        print(f"[OK] Converted 3D render: {ppm_path.name} -> {png_path.name} ({img.size[0]}x{img.size[1]})")


def _invoke_figure_generator(fn: Callable[..., Any], output_dir: Path, fallback_name: str) -> Path:
    """
    Invokes a figure generator with `output_dir` if supported and normalizes the returned Path.
    """
    sig = inspect.signature(fn)
    if len(sig.parameters) >= 1:
        res = fn(output_dir)
    else:
        res = fn()
    if isinstance(res, Path):
        return res
    return output_dir / fallback_name


def _is_2d_grid_or_matrix(val: Any) -> bool:
    """
    Returns True if `val` is a list of compact grid strings (multi-row 2D grid)
    or a 2D list of numeric/string rows.
    """
    if not isinstance(val, list) or len(val) < 2:
        return False
    first = val[0]
    if isinstance(first, str) and (" " in first or "." in first):
        return True
    if isinstance(first, list):
        return True
    return False


def _export_figures_empirical_audit(elapsed_total: float) -> Tuple[Path, Path]:
    """
    Writes the accumulated live telemetry (both scalar metrics and 2D grids/matrices)
    from all figure generators into `artifacts/logs/ablations/figures_empirical_audit.json` and `.md`.
    """
    audit_dir = PROJECT_ROOT / "artifacts" / "logs" / "ablations"
    audit_dir.mkdir(parents=True, exist_ok=True)

    json_path = audit_dir / "figures_empirical_audit.json"
    md_path = audit_dir / "figures_empirical_audit.md"

    cuda_ok = is_cuda_kernel_image_usable()
    device_str = "cuda" if cuda_ok else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
    registry = get_figure_audit_registry()

    payload: Dict[str, Any] = {
        "metadata": {
            "author": "Vyacheslav Zimovsky",
            "email": "v.zimovsky.ai@gmail.com",
            "license": "MIT-0",
            "fgns_version": "5.0.0",
            "checkpoint": "artifacts/checkpoints/model_final.pt",
            "device": device_str,
            "gpu_name": device_name,
            "elapsed_seconds": round(elapsed_total, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "figures_telemetry": registry,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_lines: List[str] = [
        "# FGNS-TRM v5.0.0: Standalone Empirical Telemetry & 2D Grid Audit of Publication Figures (`fig01`–`fig07`)",
        f"*Device: {device_str} ({device_name}) | Checkpoint: `{payload['metadata']['checkpoint']}` | Elapsed: {elapsed_total:.2f}s*",
        "",
    ]
    for fig_key, fig_data in registry.items():
        md_lines.append(f"## `{fig_key}`")
        md_lines.append("| Telemetry Metric | Measured Value |")
        md_lines.append("|---|---|")
        grid_blocks: List[Tuple[str, Any]] = []
        for k, v in fig_data.items():
            if _is_2d_grid_or_matrix(v):
                grid_blocks.append((k, v))
                rows_cnt = len(v)
                md_lines.append(f"| `{k}` | *(2D Grid/Matrix: {rows_cnt} rows — see block below)* |")
            elif isinstance(v, float):
                val_str = f"`{v:.3e}`" if (abs(v) > 0 and (abs(v) < 1e-3 or abs(v) >= 1e4)) else f"`{v:.4f}`"
                md_lines.append(f"| `{k}` | {val_str} |")
            else:
                md_lines.append(f"| `{k}` | `{v}` |")
        md_lines.append("")

        for gk, gv in grid_blocks:
            md_lines.append(f"### `{fig_key}` :: `{gk}`")
            md_lines.append("```text")
            for row_item in gv:
                if isinstance(row_item, str):
                    md_lines.append(f"  {row_item}")
                elif isinstance(row_item, list):
                    formatted_cells = [
                        f"{x:>9s}" if isinstance(x, str) else f"{float(x):6.3f}"
                        for x in row_item
                    ]
                    md_lines.append("  " + " ".join(formatted_cells))
            md_lines.append("```")
            md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path


def generate_all(output_dir: Optional[Path] = None) -> List[Path]:
    """
    Executes the Rust 3D manifold ray-tracer followed by all eight publication figure generators
    and writes the empirical audit reports.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_and_run_rust_manifold_engine()

    generators = [
        ("Figure 0 (2:1 Kaggle Card & Pipeline Cover)", generate_fig00, "fig00_kaggle_card_thumbnail.png"),
        ("Figure 1 (NA3D & Lemma 1 Subspace Split)", generate_fig01, "fig01_na3d_finsler_lie_3d.png"),
        ("Figure 2 (Two-Stage Structural Priors & 6.57M Architecture)", generate_fig02, "fig02_architecture_3d_stack.png"),
        ("Figure 3 (Stiefel Sheaf, OctoWave & Cauchy Dynamics)", generate_fig03, "fig03_sheaf_cauchy_dynamics_3d.png"),
        ("Figure 4 (4-Archetype Public ARC-AGI-2 Case Studies)", generate_fig04, "fig04_arc2_visual_case_studies.png"),
        ("Figure 5 (8-Subsystem Ablation Forest Plot & Scorecard)", generate_fig05, "fig05_ablation_scientific_dashboard.png"),
        ("Figure 6 (Unified Spectral-Green Operator Gallery)", generate_fig06, "fig06_topological_operators_gallery.png"),
        ("Figure 7 (Schatten-1 Nuclear MDL & Geodesic SO(3) TTT)", generate_fig07, "fig07_spectral_mdl_ttt_dynamics.png"),
    ]

    saved_paths: List[Path] = []
    t_start = time.perf_counter()
    for name, fn, fb_file in generators:
        t0 = time.perf_counter()
        out_path = _invoke_figure_generator(fn, output_dir, fb_file)
        dt = time.perf_counter() - t0
        print(f"[OK] {name} -> {out_path.name} ({dt:.2f}s)", flush=True)
        saved_paths.append(out_path)

    total_dt = time.perf_counter() - t_start
    json_audit, md_audit = _export_figures_empirical_audit(total_dt)
    print(f"All 8 figures generated in {total_dt:.2f}s at {output_dir}")
    print(f"  [AUDIT JSON] {json_audit}")
    print(f"  [AUDIT MD]   {md_audit}")
    return saved_paths


if __name__ == "__main__":
    generate_all()
