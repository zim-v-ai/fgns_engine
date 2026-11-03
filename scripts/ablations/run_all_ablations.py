"""
Location: // fgns_engine/scripts/ablations/run_all_ablations.py
Description: Master GPU-Accelerated Ablation Orchestrator, Single-Pass Compact Grid Visualizer,
             and Publication Table Synthesizer for FGNS-TRM v4.4.0 on Blind ARC-AGI-2 Evaluation.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+ / CUDA sm_120 Accelerated
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch

from scripts.ablations.ablate_01_na3d_stiefel import run_na3d_ablation
from scripts.ablations.ablate_02_sheaf_laplacian import run_sheaf_laplacian_ablation
from scripts.ablations.ablate_03_octowave_ssm import run_octowave_ablation
from scripts.ablations.ablate_04_cauchy_dynamics import run_cauchy_dynamics_ablation
from scripts.ablations.ablate_05_spectral_mdl import run_spectral_mdl_ablation
from scripts.ablations.ablate_06_boundary_size import run_boundary_size_ablation
from scripts.ablations.ablate_07_palette_budget import run_palette_budget_ablation
from scripts.ablations.ablate_08_operator_adapter import run_operator_adapter_ablation
from scripts.ablations.common import (
    ensure_utf8_environment,
    get_ablation_device,
    load_benchmark_tasks,
    synthesize_compact_task_visual_samples,
)


def export_markdown_tables(report: Dict[str, Any], path: Path) -> None:
    """Renders multi-metric publication Markdown tables with verdicts for all 8 ablations."""
    meta = report["metadata"]
    res = report["results"]

    md: List[str] = [
        "# FGNS-TRM v4.4.0: Complete Architectural Ablation Battery on Blind ARC-AGI-2 Evaluation",
        f"*Author: {meta['author']} ({meta['email']}) | License: {meta['license']}*",
        f"*Timestamp: {meta['timestamp']} | Device: {meta['device']} ({meta.get('gpu_name', 'CPU')}) | Corpus: {meta.get('corpus', 'ARC-AGI-2 Evaluation')} ({meta['benchmark_tasks']} tasks)*",
        "",
        "## Table 1: Representation & Orientation: NA3D-Finsler-Lie vs. Flat ViT, RoPE-SO(2) & Unsplit SO(3) (Theorem I)",
        "| Condition | Query Loss $\\downarrow$ | Pixel Acc (%) $\\uparrow$ | FG Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Color Subspace Leakage $\\downarrow$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in res["01_na3d_stiefel"].items():
        md.append(
            f"| `{k}` | {v['query_loss']:.4f} | {v['pixel_accuracy_pct']:.2f}% | {v['fg_accuracy_pct']:.2f}% | "
            f"{v['exact_grid_match_pct']:.1f}% | {v.get('color_subspace_leakage', 0.0):.2e} | {v.get('verdict', '')} |"
        )

    md.extend([
        "",
        "## Table 2: Cellular Sheaf Connection Laplacian vs. Isotropic GCN, Scalar GAT & Anti-Diffusion (Theorem II)",
        "| Condition | Query Loss $\\downarrow$ | Pixel Acc (%) $\\uparrow$ | FG Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Energy Contraction Ratio $\\downarrow$ | Stiefel Ortho Error $\\downarrow$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for k, v in res["02_sheaf_laplacian"].items():
        md.append(
            f"| `{k}` | {v['query_loss']:.4f} | {v['pixel_accuracy_pct']:.2f}% | {v['fg_accuracy_pct']:.2f}% | "
            f"{v['exact_grid_match_pct']:.1f}% | {v.get('energy_contraction_ratio', 1.0):.3f} | {v.get('stiefel_orthogonality_error', 0.0):.2e} | {v.get('verdict', '')} |"
        )

    md.extend([
        "",
        "## Table 3: Spatial Wavefront Propagation: OctoWave 8-Axis SSM vs. ConvNeXt/NCA, Mamba-2 & ViT 2D Attention",
        "| Condition | Query Loss $\\downarrow$ | Pixel Acc (%) $\\uparrow$ | FG Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Diagonal Wavefront Reach $\\uparrow$ | Latency (ms) $\\downarrow$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for k, v in res["03_octowave_ssm"].items():
        md.append(
            f"| `{k}` | {v['query_loss']:.4f} | {v['pixel_accuracy_pct']:.2f}% | {v['fg_accuracy_pct']:.2f}% | "
            f"{v['exact_grid_match_pct']:.1f}% | {v.get('diagonal_wavefront_reach', 0.0):.4f} | {v['latency_ms']:.1f} | {v.get('verdict', '')} |"
        )

    md.extend([
        "",
        "## Table 4: Recurrent Field Dynamics: Dual-State Cauchy Resolvent vs. Feedforward, Euler ODE & GRU",
        "| Condition | Query Loss $\\downarrow$ | Pixel Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Monotonic Step Descent (%) $\\uparrow$ | Trajectory Loss Drop $\\Delta\\mathcal{L}_{1\\to 4}$ $\\uparrow$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for k, v in res["04_cauchy_dynamics"].items():
        md.append(
            f"| `{k}` | {v['query_loss']:.4f} | {v['pixel_accuracy_pct']:.2f}% | {v['exact_grid_match_pct']:.1f}% | "
            f"{v['monotonic_descent_pct']:.1f}% | {v.get('trajectory_loss_drop', 0.0):+.4f} | {v.get('verdict', '')} |"
        )

    md.extend([
        "",
        "## Table 5: Zero-Leakage LOO Test-Time Adaptation: Spectral MDL (Schatten-1) vs. Baselines",
        "| Condition | Blind Test Loss $\\downarrow$ | Pixel Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Effective Rank $r_{\\text{eff}}$ $\\downarrow$ | Spectral Entropy $H_{\\text{vN}}$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for k, v in res["05_spectral_mdl"].items():
        md.append(
            f"| `{k}` | {v['query_generalization_loss']:.4f} | {v['pixel_accuracy_pct']:.2f}% | "
            f"{v['exact_grid_match_pct']:.1f}% | {v['effective_rank_r_eff']:.2f} | {v.get('spectral_entropy', 0.0):.3f} | {v.get('verdict', '')} |"
        )

    md.extend([
        "",
        "## Table 6: Coupled Boltzmann-Woodbury 1-Jet Stokes Dimension Predictor (Theorem III)",
        "| Condition | Exact (H, W) Match (%) $\\uparrow$ | Mean Abs Error (px) $\\downarrow$ | Query Loss $\\downarrow$ | Pixel Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for k, v in res["06_boundary_size"].items():
        md.append(
            f"| `{k}` | {v['exact_dimension_match_pct']:.1f}% | {v['mean_absolute_error']:.2f} | "
            f"{v['query_loss']:.4f} | {v['pixel_accuracy_pct']:.2f}% | {v['exact_grid_match_pct']:.1f}% | {v.get('verdict', '')} |"
        )

    md.extend([
        "",
        "## Table 7: Single-Equation S10-Equivariant Role-Woodbury, Diode Sink & Hare LRQ Palette Predictor",
        "| Condition | Color Hallucination (%) $\\downarrow$ | Palette EMD $\\downarrow$ | Exact Budget (%) $\\uparrow$ | Pixel Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for k, v in res["07_palette_budget"].items():
        md.append(
            f"| `{k}` | {v['color_hallucination_pct']:.2f}% | {v['earth_movers_distance']:.4f} | "
            f"{v.get('exact_budget_match_pct', 0.0):.1f}% | {v['pixel_accuracy_pct']:.2f}% | {v['exact_grid_match_pct']:.1f}% | {v.get('verdict', '')} |"
        )

    md.extend([
        "",
        "## Table 8: Transductive Geodesic SO(3)^C + D4xS9 Operator Adapter under OOD Automorphisms (Theorem IV)",
        "| Condition | Blind Test Loss $\\downarrow$ | D4xS9 Inv Acc (%) $\\uparrow$ | Exact Grid (%) $\\uparrow$ | Mean Steps to Converge $\\downarrow$ | Effective Rank $r_{\\text{eff}}$ | Scientific Verdict |",
        "|---|---|---|---|---|---|---|",
    ])
    for k, v in res["08_operator_adapter"].items():
        md.append(
            f"| `{k}` | {v['query_generalization_loss']:.4f} | {v.get('d4_s9_invariance_accuracy_pct', v['pixel_accuracy_pct']):.2f}% | "
            f"{v['exact_grid_match_pct']:.1f}% | {v['mean_steps_to_convergence']:.1f} | {v['effective_rank_r_eff']:.2f} | {v.get('verdict', '')} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def export_latex_tables(report: Dict[str, Any], path: Path) -> None:
    """Renders formatted NeurIPS / ICLR / ARC Prize Paper Track LaTeX summary table."""
    res = report["results"]
    tex = [
        "% FGNS-TRM v4.4.0 Publication Ablation Summary Table",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Controlled Architectural Ablations of FGNS-TRM v4.4.0 vs. Literature Analogues on Blind ARC-AGI-2 Evaluation Tasks.}",
        "\\label{tab:fgns_ablations}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Subsystem} & \\textbf{Condition / Analogue} & \\textbf{Test Loss $\\downarrow$} & \\textbf{Pixel Acc (\\%) $\\uparrow$} & \\textbf{Exact (\\%) $\\uparrow$} & \\textbf{Size Acc (\\%) $\\uparrow$} & \\textbf{Primary Domain Invariant} \\\\",
        "\\midrule",
    ]

    for k, v in res["01_na3d_stiefel"].items():
        leak = v.get("color_subspace_leakage", 0.0)
        tex.append(
            f"01. NA3D Frame & \\texttt{{{k.replace('_', '\\_')}}} & {v['query_loss']:.4f} & "
            f"{v['pixel_accuracy_pct']:.2f}\\% & {v['exact_grid_match_pct']:.1f}\\% & {v['size_accuracy_pct']:.1f}\\% & Leak $={leak:.2e}$ \\\\"
        )
    tex.append("\\midrule")

    for k, v in res["02_sheaf_laplacian"].items():
        rat = v.get("energy_contraction_ratio", 1.0)
        tex.append(
            f"02. Sheaf Laplacian & \\texttt{{{k.replace('_', '\\_')}}} & {v['query_loss']:.4f} & "
            f"{v['pixel_accuracy_pct']:.2f}\\% & {v['exact_grid_match_pct']:.1f}\\% & {v['size_accuracy_pct']:.1f}\\% & $\\mathcal{{E}}_t/\\mathcal{{E}}_0={rat:.3f}$ \\\\"
        )
    tex.append("\\midrule")

    for k, v in res["05_spectral_mdl"].items():
        rk = v.get("effective_rank_r_eff", 1.0)
        tex.append(
            f"05. Spectral MDL & \\texttt{{{k.replace('_', '\\_')}}} & {v['query_generalization_loss']:.4f} & "
            f"{v['pixel_accuracy_pct']:.2f}\\% & {v['exact_grid_match_pct']:.1f}\\% & {v['size_accuracy_pct']:.1f}\\% & $r_{{\\text{{eff}}}}={rk:.2f}$ \\\\"
        )
    tex.append("\\midrule")

    for k, v in res["06_boundary_size"].items():
        mae = v.get("mean_absolute_error", 0.0)
        tex.append(
            f"06. Boundary Size & \\texttt{{{k.replace('_', '\\_')}}} & {v['query_loss']:.4f} & "
            f"{v['pixel_accuracy_pct']:.2f}\\% & {v['exact_grid_match_pct']:.1f}\\% & {v['exact_dimension_match_pct']:.1f}\\% & MAE $={mae:.2f}$ px \\\\"
        )
    tex.append("\\midrule")

    for k, v in res["07_palette_budget"].items():
        emd = v.get("earth_movers_distance", 0.0)
        tex.append(
            f"07. Palette Budget & \\texttt{{{k.replace('_', '\\_')}}} & {v['query_loss']:.4f} & "
            f"{v['pixel_accuracy_pct']:.2f}\\% & {v['exact_grid_match_pct']:.1f}\\% & {v['size_accuracy_pct']:.1f}\\% & EMD $={emd:.4f}$ \\\\"
        )
    tex.append("\\midrule")

    for k, v in res["08_operator_adapter"].items():
        inv = v.get("d4_s9_invariance_accuracy_pct", v["pixel_accuracy_pct"])
        tex.append(
            f"08. Geodesic TTT & \\texttt{{{k.replace('_', '\\_')}}} & {v['query_generalization_loss']:.4f} & "
            f"{v['pixel_accuracy_pct']:.2f}\\% & {v['exact_grid_match_pct']:.1f}\\% & {v['size_accuracy_pct']:.1f}\\% & $D_4\\times S_9$ Acc $={inv:.1f}\\%$ \\\\"
        )

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def run_complete_ablation_suite(
    tasks_limit: int = 10,
    out_dir_str: str = "artifacts/logs/ablations",
    force_gpu: bool = False,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes all 8 ablations on GPU/CPU, generates the single-pass non-duplicated
    task_visual_samples dictionary, and writes JSON, Markdown, and LaTeX artifacts.
    """
    ensure_utf8_environment()
    console = Console()
    device = get_ablation_device(force_gpu=force_gpu, device_override=device_override)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_benchmark_tasks(limit=tasks_limit, include_arc2_eval=True)
    task_ids_str = ", ".join(t.task_id for t in tasks)

    console.print(
        Panel(
            f"[bold magenta]FGNS-TRM v4.4.0: Blind ARC-AGI-2 Evaluation 8-Component Scientific Ablation Suite[/bold magenta]\n"
            f"Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)\n"
            f"Device: {device} ({gpu_name}) | ARC-AGI-2 Eval Tasks ({len(tasks)}): {task_ids_str}\n"
            f"Output Target: {out_dir.resolve()}",
            border_style="cyan",
        )
    )

    t_start = time.time()
    results: Dict[str, Any] = {}
    results["01_na3d_stiefel"] = run_na3d_ablation(tasks, device)
    results["02_sheaf_laplacian"] = run_sheaf_laplacian_ablation(tasks, device)
    results["03_octowave_ssm"] = run_octowave_ablation(tasks, device)
    results["04_cauchy_dynamics"] = run_cauchy_dynamics_ablation(tasks, device)
    results["05_spectral_mdl"] = run_spectral_mdl_ablation(tasks, device)
    results["06_boundary_size"] = run_boundary_size_ablation(tasks, device)
    results["07_palette_budget"] = run_palette_budget_ablation(tasks, device)
    results["08_operator_adapter"] = run_operator_adapter_ablation(tasks, device)

    console.print("\n[bold cyan]>>> Synthesizing Single-Pass Compact Visual Samples (1 Train + 1 Test Response per Task)...[/bold cyan]")
    task_visual_samples = synthesize_compact_task_visual_samples(tasks, device)

    elapsed = time.time() - t_start
    full_report: Dict[str, Any] = {
        "metadata": {
            "author": "Vyacheslav Zimovsky",
            "email": "v.zimovsky.ai@gmail.com",
            "license": "MIT-0",
            "fgns_version": "4.4.0",
            "corpus": "ARC-AGI-2 Evaluation (Blind Out-of-Distribution)",
            "device": str(device),
            "gpu_name": gpu_name,
            "benchmark_tasks": len(tasks),
            "task_ids": [t.task_id for t in tasks],
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
        "task_visual_samples": task_visual_samples,
    }

    json_path = out_dir / "master_ablation_report.json"
    md_path = out_dir / "master_ablation_report.md"
    tex_path = out_dir / "master_ablation_report.tex"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    export_markdown_tables(full_report, md_path)
    export_latex_tables(full_report, tex_path)

    summary_table = Table(title="FGNS-TRM v4.4.0 SOTA Ablation Verdicts Summary (ARC-AGI-2 Eval)", header_style="bold cyan")
    summary_table.add_column("Ablation Study", style="bold white")
    summary_table.add_column("SOTA Variant", style="bold green")
    summary_table.add_column("Test Loss", justify="right")
    summary_table.add_column("Pixel Acc", justify="right")
    summary_table.add_column("Exact Match", justify="right")
    summary_table.add_column("Size Acc", justify="right")
    summary_table.add_column("Verdict", style="yellow")

    sota_keys = [
        ("01_na3d_stiefel", "na3d_stiefel"),
        ("02_sheaf_laplacian", "cellular_sheaf_od"),
        ("03_octowave_ssm", "octowave_8axis_ssm"),
        ("04_cauchy_dynamics", "cauchy_resolvent"),
        ("05_spectral_mdl", "spectral_mdl_schatten1"),
        ("06_boundary_size", "boundary_metric_flux"),
        ("07_palette_budget", "equivariant_haar_lrq"),
        ("08_operator_adapter", "geodesic_so3_adapter"),
    ]
    for study_name, sota_var in sota_keys:
        sv = results[study_name][sota_var]
        summary_table.add_row(
            study_name,
            sota_var,
            f"{sv['query_loss']:.4f}",
            f"{sv['pixel_accuracy_pct']:.2f}%",
            f"{sv['exact_grid_match_pct']:.1f}%",
            f"{sv['size_accuracy_pct']:.1f}%",
            str(sv.get("verdict", ""))[:60],
        )

    console.print("\n")
    console.print(summary_table)
    console.print(f"\n[bold green]✓ Complete 8-component ablation battery + {len(task_visual_samples)} visual task samples finished in {elapsed:.1f}s on {device} ({gpu_name}).[/bold green]")
    console.print(f"  JSON  : {json_path.resolve()}")
    console.print(f"  MD    : {md_path.resolve()}")
    console.print(f"  LaTeX : {tex_path.resolve()}")

    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Master FGNS v4.4.0 Architectural Ablation Orchestrator")
    parser.add_argument("--tasks", type=int, default=10, help="Number of blind ARC-AGI-2 Evaluation tasks per ablation")
    parser.add_argument("--out-dir", type=str, default="artifacts/logs/ablations", help="Output directory")
    parser.add_argument("--gpu", action="store_true", help="Force GPU CUDA execution on RTX 5070")
    parser.add_argument("--device", type=str, default=None, help="Target device override (e.g. cuda, cuda:0, cpu)")
    args = parser.parse_args()

    run_complete_ablation_suite(
        tasks_limit=args.tasks,
        out_dir_str=args.out_dir,
        force_gpu=args.gpu,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
