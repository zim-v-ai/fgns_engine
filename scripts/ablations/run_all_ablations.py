"""
Location: // fgns_engine/scripts/ablations/run_all_ablations.py
Description: Master Ablation Orchestrator and Paper Table Synthesizer for FGNS-TRM.
             Runs the complete 8-component ablation battery and exports publication-ready
             Markdown, JSON, and LaTeX tables for the ARC Prize Paper Track.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel

from scripts.ablations.common import (
    ensure_utf8_environment,
    get_ablation_device,
    load_benchmark_tasks,
)
from scripts.ablations.ablate_01_na3d_stiefel import run_na3d_ablation
from scripts.ablations.ablate_02_sheaf_laplacian import run_sheaf_laplacian_ablation
from scripts.ablations.ablate_03_octowave_ssm import run_octowave_ablation
from scripts.ablations.ablate_04_cauchy_dynamics import run_cauchy_dynamics_ablation
from scripts.ablations.ablate_05_spectral_mdl import run_spectral_mdl_ablation
from scripts.ablations.ablate_06_boundary_size import run_boundary_size_ablation
from scripts.ablations.ablate_07_palette_budget import run_palette_budget_ablation
from scripts.ablations.ablate_08_operator_adapter import run_operator_adapter_ablation


def export_markdown_tables(report: Dict[str, Any], path: Path) -> None:
    """Renders formatted Markdown tables of all 8 ablations for publication."""
    md: List[str] = [
        "# FGNS-TRM: Complete Architectural Ablation Battery",
        f"*Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)*",
        f"*Benchmark Timestamp: {report['metadata']['timestamp']} | Device: {report['metadata']['device']}*",
        "",
        "## Table 1: Representation & Orientation: NA3D Stiefel Frame Transport vs. Baselines",
        "| Architecture Variant | Frame Bundle | Query Loss (lower is better) | Angular Error (rad) |",
        "|---|---|---|---|",
    ]

    r1 = report["results"]["01_na3d_stiefel"]
    md.append(f"| Isotropic Scalar | Identity ($I$) | {r1['isotropic_scalar']['query_loss']:.4f} | {r1['isotropic_scalar']['angular_error_rad']:.4f} |")
    md.append(f"| Planar SO(2) | $SO(2)$ Circle | {r1['planar_so2']['query_loss']:.4f} | {r1['planar_so2']['angular_error_rad']:.4f} |")
    md.append(f"| **NA3D Stiefel (FGNS-TRM)** | **$\\text{{St}}(3, d)$ Lie Rotor** | **{r1['na3d_stiefel']['query_loss']:.4f}** | **{r1['na3d_stiefel']['angular_error_rad']:.4f}** |")

    md.extend([
        "",
        "## Table 2: Non-Abelian Cellular Sheaf Connection vs. Isotropic GCN vs. Attention",
        "| Graph Diffusion Mechanism | Restriction Maps | Query Loss | Dirichlet Energy $\\mathcal{E}_\\mathcal{F}$ (higher resists oversmoothing) |",
        "|---|---|---|---|",
    ])
    r2 = report["results"]["02_sheaf_laplacian"]
    md.append(f"| Standard GCN | $W = I$ (Isotropic) | {r2['isotropic_gcn']['query_loss']:.4f} | {r2['isotropic_gcn']['dirichlet_energy']:.4f} |")
    md.append(f"| Graph Attention (GAT) | $\\alpha \\in [0, 1]$ (Scalar) | {r2['scalar_gat']['query_loss']:.4f} | {r2['scalar_gat']['dirichlet_energy']:.4f} |")
    md.append(f"| **Cellular Sheaf (FGNS-TRM)** | **$W \\in O(d)$ (Orthogonal)** | **{r2['cellular_sheaf_od']['query_loss']:.4f}** | **{r2['cellular_sheaf_od']['dirichlet_energy']:.4f}** |")

    md.extend([
        "",
        "## Table 3: Context Aggregation: OctoWave 8-Axis SSM vs. Conv2D vs. 2D Attention",
        "| Mechanism | Scan Directions | Query Loss | Latency (ms) | Complexity |",
        "|---|---|---|---|---|",
    ])
    r3 = report["results"]["03_octowave_ssm"]
    md.append(f"| Dilated Conv2D | Local 3x3 | {r3['dilated_conv2d']['query_loss']:.4f} | {r3['dilated_conv2d']['latency_ms']:.1f} | $\\mathcal{{O}}(HW)$ |")
    md.append(f"| 2-Axis Mamba | 2 Cardinal | {r3['mamba_2axis_scan']['query_loss']:.4f} | {r3['mamba_2axis_scan']['latency_ms']:.1f} | $\\mathcal{{O}}(HW)$ |")
    md.append(f"| Full 2D Attention | Global Pairwise | {r3['full_2d_attention']['query_loss']:.4f} | {r3['full_2d_attention']['latency_ms']:.1f} | $\\mathcal{{O}}((HW)^2)$ |")
    md.append(f"| **OctoWave SSM (FGNS-TRM)** | **8 Directions (Triton)** | **{r3['octowave_8axis_ssm']['query_loss']:.4f}** | **{r3['octowave_8axis_ssm']['latency_ms']:.1f}** | **$\\mathcal{{O}}(HW)$** |")

    md.extend([
        "",
        "## Table 4: Iterative Dynamics: Cauchy Resolvent vs. Discrete ResNet vs. GRU",
        "| State Transition | Integration Type | Query Loss | Monotonic Descent (%) |",
        "|---|---|---|---|",
    ])
    r4 = report["results"]["04_cauchy_dynamics"]
    md.append(f"| Feedforward Single-Pass | None | {r4['single_pass_feedforward']['query_loss']:.4f} | {r4['single_pass_feedforward']['monotonic_descent_pct']:.1f}% |")
    md.append(f"| Discrete ResNet | $x_{{t+1}} = x_t + f(x_t)$ | {r4['discrete_resnet']['query_loss']:.4f} | {r4['discrete_resnet']['monotonic_descent_pct']:.1f}% |")
    md.append(f"| Recurrent GRU | Gated Discrete | {r4['recurrent_gru']['query_loss']:.4f} | {r4['recurrent_gru']['monotonic_descent_pct']:.1f}% |")
    md.append(f"| **Cauchy Resolvent (FGNS-TRM)** | **$(zI - A)^{{-1}}$ Complex Contour** | **{r4['cauchy_resolvent']['query_loss']:.4f}** | **{r4['cauchy_resolvent']['monotonic_descent_pct']:.1f}%** |")

    md.extend([
        "",
        "## Table 5: Test-Time Adaptation: Spectral MDL (Schatten-1) vs. Standard L2 Weight Decay",
        "| TTT Regularization Objective | Mathematical Form | Effective Rank $r_{{\\text{{eff}}}}$ | Query Generalization Loss |",
        "|---|---|---|---|",
    ])
    r5 = report["results"]["05_spectral_mdl"]
    md.append(f"| Zero TTT | No adaptation | {r5['zero_ttt']['effective_rank_r_eff']:.2f} | {r5['zero_ttt']['query_generalization_loss']:.4f} |")
    md.append(f"| Standard L2 | $\\lambda \\|\\Delta W\\|_2^2$ | {r5['standard_l2_ttt']['effective_rank_r_eff']:.2f} | {r5['standard_l2_ttt']['query_generalization_loss']:.4f} |")
    md.append(f"| Schatten-2 Frobenius | $\\|\\Delta W\\|_F^2$ | {r5['schatten2_frobenius']['effective_rank_r_eff']:.2f} | {r5['schatten2_frobenius']['query_generalization_loss']:.4f} |")
    md.append(f"| **Spectral MDL (FGNS-TRM)** | **Schatten-1 $\\|\\Delta W\\|_*$** | **{r5['spectral_mdl_schatten1']['effective_rank_r_eff']:.2f}** | **{r5['spectral_mdl_schatten1']['query_generalization_loss']:.4f}** |")

    md.extend([
        "",
        "## Table 6: Lattice Dimension Prediction: Darboux Boundary Flux vs. Global Pooling",
        "| Dimension Predictor | Mechanism | Exact Match (H, W) (%) | MAE ($|\\Delta H| + |\\Delta W|$) |",
        "|---|---|---|---|",
    ])
    r6 = report["results"]["06_boundary_size"]
    md.append(f"| Identity Baseline | Trivial ($H_{{in}}, W_{{in}}$) | {r6['identity_baseline']['exact_dimension_match_pct']:.1f}% | {r6['identity_baseline']['mean_absolute_error']:.2f} |")
    md.append(f"| Global Avg Pooling | Spatial Mean + Linear | {r6['global_avg_pooling']['exact_dimension_match_pct']:.1f}% | {r6['global_avg_pooling']['mean_absolute_error']:.2f} |")
    md.append(f"| Spatial Flattened MLP | Dense Projection | {r6['spatial_mlp']['exact_dimension_match_pct']:.1f}% | {r6['spatial_mlp']['mean_absolute_error']:.2f} |")
    md.append(f"| **Boundary Metric (FGNS-TRM)** | **Darboux Boundary Flux** | **{r6['boundary_metric_flux']['exact_dimension_match_pct']:.1f}%** | **{r6['boundary_metric_flux']['mean_absolute_error']:.2f}** |")

    md.extend([
        "",
        "## Table 7: Color Conservation: S_10 Haar LRQ vs. Unconstrained Softmax",
        "| Palette Estimator | Equivariance | Color Hallucination Rate (%) | Earth Mover's Distance |",
        "|---|---|---|---|",
    ])
    r7 = report["results"]["07_palette_budget"]
    md.append(f"| Unconstrained Softmax | None | {r7['unconstrained_softmax']['color_hallucination_pct']:.2f}% | {r7['unconstrained_softmax']['earth_movers_distance']:.4f} |")
    md.append(f"| Non-Equivariant MLP | None | {r7['non_equivariant_mlp']['color_hallucination_pct']:.2f}% | {r7['non_equivariant_mlp']['earth_movers_distance']:.4f} |")
    md.append(f"| **Haar LRQ (FGNS-TRM)** | **$S_{{10}}$ Exact Equivariant** | **{r7['equivariant_haar_lrq']['color_hallucination_pct']:.2f}%** | **{r7['equivariant_haar_lrq']['earth_movers_distance']:.4f}** |")

    md.extend([
        "",
        "## Table 8: Test-Time Optimizer: SO(3) Rodrigues Geodesic vs. Euclidean LoRA",
        "| Adaptation Method | Parameter Space | Steps to Convergence | Query Loss |",
        "|---|---|---|---|",
    ])
    r8 = report["results"]["08_operator_adapter"]
    md.append(f"| Full Fine-Tuning | Full Euclidean ($W$) | {r8['full_fine_tuning']['mean_steps_to_convergence']:.1f} | {r8['full_fine_tuning']['query_generalization_loss']:.4f} |")
    md.append(f"| Euclidean LoRA | Low-Rank Euclidean ($BA$) | {r8['euclidean_lora']['mean_steps_to_convergence']:.1f} | {r8['euclidean_lora']['query_generalization_loss']:.4f} |")
    md.append(f"| **Geodesic SO(3) (FGNS-TRM)** | **Lie Group Geodesic ($\\omega \\in \\mathfrak{{so}}(3)$)** | **{r8['geodesic_so3_adapter']['mean_steps_to_convergence']:.1f}** | **{r8['geodesic_so3_adapter']['query_generalization_loss']:.4f}** |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def export_latex_tables(report: Dict[str, Any], path: Path) -> None:
    """Renders formatted LaTeX tables for NeurIPS / ICLR submission."""
    tex = [
        "% FGNS-TRM Publication Tables (Generated automatically)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Comprehensive Architectural Ablations of FGNS-TRM Components on the ARC Benchmark.}",
        "\\label{tab:fgns_ablations}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Subsystem} & \\textbf{Variant} & \\textbf{Prior} & \\textbf{Query Loss $\\downarrow$} & \\textbf{Key Scientific Metric} \\\\",
        "\\midrule",
    ]

    r1 = report["results"]["01_na3d_stiefel"]
    tex.append(f"NA3D Transport & Isotropic Scalar & None & {r1['isotropic_scalar']['query_loss']:.4f} & $\\theta_{{\\text{{err}}}}= {r1['isotropic_scalar']['angular_error_rad']:.2f}$ rad \\\\")
    tex.append(f" & Planar $SO(2)$ & 2D Circle & {r1['planar_so2']['query_loss']:.4f} & $\\theta_{{\\text{{err}}}}= {r1['planar_so2']['angular_error_rad']:.2f}$ rad \\\\")
    tex.append(f" & \\textbf{{NA3D Stiefel}} & \\textbf{{$\\text{{St}}(3, d)$ Lie Rotor}} & \\textbf{{{r1['na3d_stiefel']['query_loss']:.4f}}} & \\textbf{{$\\theta_{{\\text{{err}}}}= {r1['na3d_stiefel']['angular_error_rad']:.2f}$ rad}} \\\\")
    tex.append("\\midrule")

    r2 = report["results"]["02_sheaf_laplacian"]
    tex.append(f"Sheaf Laplacian & Standard GCN & $W=I$ & {r2['isotropic_gcn']['query_loss']:.4f} & $\\mathcal{{E}}_\\mathcal{{F}}= {r2['isotropic_gcn']['dirichlet_energy']:.3f}$ \\\\")
    tex.append(f" & Graph Attention & Scalar & {r2['scalar_gat']['query_loss']:.4f} & $\\mathcal{{E}}_\\mathcal{{F}}= {r2['scalar_gat']['dirichlet_energy']:.3f}$ \\\\")
    tex.append(f" & \\textbf{{Cellular Sheaf}} & \\textbf{{$W \\in O(d)$}} & \\textbf{{{r2['cellular_sheaf_od']['query_loss']:.4f}}} & \\textbf{{$\\mathcal{{E}}_\\mathcal{{F}}= {r2['cellular_sheaf_od']['dirichlet_energy']:.3f}$}} \\\\")
    tex.append("\\midrule")

    r5 = report["results"]["05_spectral_mdl"]
    tex.append(f"Test-Time Training & Standard L2 & $\\lambda \\|W\\|_2^2$ & {r5['standard_l2_ttt']['query_generalization_loss']:.4f} & $r_{{\\text{{eff}}}}= {r5['standard_l2_ttt']['effective_rank_r_eff']:.2f}$ \\\\")
    tex.append(f" & \\textbf{{Spectral MDL}} & \\textbf{{Schatten-1 $\\|W\\|_*$}} & \\textbf{{{r5['spectral_mdl_schatten1']['query_generalization_loss']:.4f}}} & \\textbf{{$r_{{\\text{{eff}}}}= {r5['spectral_mdl_schatten1']['effective_rank_r_eff']:.2f}$}} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table*}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def main() -> None:
    ensure_utf8_environment()
    parser = argparse.ArgumentParser(description="Master FGNS Architectural Ablation Orchestrator")
    parser.add_argument("--tasks", type=int, default=5, help="Number of benchmark tasks per ablation")
    parser.add_argument("--out-dir", type=str, default="artifacts/logs/ablations", help="Output directory")
    args = parser.parse_args()

    console = Console()
    device = get_ablation_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold magenta]FGNS-TRM: Complete 8-Component Architectural Ablation Suite[/bold magenta]\n"
        f"Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)\n"
        f"Device: {device} | Tasks per experiment: {args.tasks}\n"
        f"Output Target: {out_dir.resolve()}",
        border_style="cyan"
    ))

    t_start = time.time()
    tasks = load_benchmark_tasks(limit=args.tasks)

    results: Dict[str, Any] = {}
    results["01_na3d_stiefel"] = run_na3d_ablation(tasks, device)
    results["02_sheaf_laplacian"] = run_sheaf_laplacian_ablation(tasks, device)
    results["03_octowave_ssm"] = run_octowave_ablation(tasks, device)
    results["04_cauchy_dynamics"] = run_cauchy_dynamics_ablation(tasks, device)
    results["05_spectral_mdl"] = run_spectral_mdl_ablation(tasks, device)
    results["06_boundary_size"] = run_boundary_size_ablation(tasks, device)
    results["07_palette_budget"] = run_palette_budget_ablation(tasks, device)
    results["08_operator_adapter"] = run_operator_adapter_ablation(tasks, device)

    elapsed = time.time() - t_start
    full_report = {
        "metadata": {
            "author": "Vyacheslav Zimovsky",
            "email": "v.zimovsky.ai@gmail.com",
            "license": "MIT-0",
            "device": str(device),
            "benchmark_tasks": len(tasks),
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }

    json_path = out_dir / "master_ablation_report.json"
    md_path = out_dir / "master_ablation_report.md"
    tex_path = out_dir / "master_ablation_report.tex"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    export_markdown_tables(full_report, md_path)
    export_latex_tables(full_report, tex_path)

    console.print(f"\n[bold green]✓ Complete ablation battery generated in {elapsed:.1f}s.[/bold green]")
    console.print(f"  JSON  : {json_path.resolve()}")
    console.print(f"  MD    : {md_path.resolve()}")
    console.print(f"  LaTeX : {tex_path.resolve()}")


if __name__ == "__main__":
    main()
