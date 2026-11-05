"""
Location: // fgns_engine/scripts/figures/generate_all_figures.py
Description: Master Orchestrator for Building & Running the Native Rust 3D GGX Manifold
             Ray-Tracer and Generating All Seven Publication-Grade Scientific Figures
             (PNG 350 DPI + Vector PDF) for the FGNS-TRM v5.0.0 ARC-AGI-2 Paper Track Writeup.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from typing import List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.figures.fig00_kaggle_card_thumbnail import generate_fig00
from scripts.figures.fig01_na3d_finsler_lie_3d import generate_fig01
from scripts.figures.fig02_architecture_3d_stack import generate_fig02
from scripts.figures.fig03_sheaf_cauchy_dynamics_3d import generate_fig03
from scripts.figures.fig04_arc2_visual_case_studies import generate_fig04
from scripts.figures.fig05_ablation_scientific_dashboard import generate_fig05
from scripts.figures.fig06_topological_operators_gallery import generate_fig06
from scripts.figures.fig07_spectral_mdl_ttt_dynamics import generate_fig07


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


def generate_all(output_dir: Path | None = None) -> List[Path]:
    """
    Executes the Rust 3D manifold ray-tracer followed by all seven publication figure generators.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_and_run_rust_manifold_engine()

    generators = [
        ("Figure 0 (2:1 Kaggle Card & Thumbnail Cover)", generate_fig00),
        ("Figure 1 (NA3D & Theorem I Split)", generate_fig01),
        ("Figure 2 (Two-Stage Conservation Priors & 6.57M Architecture)", generate_fig02),
        ("Figure 3 (Stiefel Sheaf, OctoWave & Cauchy Dynamics)", generate_fig03),
        ("Figure 4 (4-Archetype Blind ARC-AGI-2 Case Studies)", generate_fig04),
        ("Figure 5 (8-Subsystem Ablation Forest Plot & Scorecard)", generate_fig05),
        ("Figure 6 (Unified Spectral-Green Operator Gallery)", generate_fig06),
        ("Figure 7 (Schatten-1 Nuclear MDL & Geodesic SO(3) TTT)", generate_fig07),
    ]

    saved_paths: List[Path] = []
    t_start = time.perf_counter()
    for name, fn in generators:
        t0 = time.perf_counter()
        out_path = fn(output_dir)
        dt = time.perf_counter() - t0
        print(f"[OK] {name} -> {out_path.name} ({dt:.2f}s)")
        saved_paths.append(out_path)

    total_dt = time.perf_counter() - t_start
    print(f"All 7 figures generated in {total_dt:.2f}s at {output_dir}")
    return saved_paths


if __name__ == "__main__":
    generate_all()
