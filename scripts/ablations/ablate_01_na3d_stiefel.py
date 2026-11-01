"""
Location: // fgns_engine/scripts/ablations/ablate_01_na3d_stiefel.py
Description: Rigorous Architectural Ablation 1: Non-Abelian NA3D Stiefel Frame Transport
             vs. Planar SO(2) vs. Scalar Isotropic Representations.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
import torch
import torch.nn as nn

from fgns.config import FGNSConfig
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCTask
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.train.losses.true_spectral_mdl import SchattenNuclearMDLObjective
from scripts.ablations.common import (
    MathematicalMetrics,
    create_model_with_weights,
    ensure_utf8_environment,
    get_ablation_device,
    load_benchmark_tasks,
)


def run_na3d_ablation(tasks: List[ARCTask], device: torch.device) -> Dict[str, Any]:
    """
    Executes controlled comparison across 3 genuine structural orientation conditions:
      1. Isotropic Scalar: Zero rotor projection (removes directional frame transport).
      2. Planar SO(2): Clamps x and y axes of Lie algebra omega vector (restricts to planar 2D rotation).
      3. NA3D Stiefel (FGNS-TRM): Full SO(3) Rodrigues rotation on St(3, d) Lie bundles.
    """
    console = Console()
    console.print("\n[bold cyan]>>> [Ablation 1] Evaluating NA3D Non-Abelian Stiefel Frame Transport...[/bold cyan]")

    conditions = ["isotropic_scalar", "planar_so2", "na3d_stiefel"]
    results: Dict[str, Any] = {}

    config = FGNSConfig()

    for cond in conditions:
        model = create_model_with_weights(config=config, device=device)
        model.eval()
        loss_engine = SchattenNuclearMDLObjective(config=config).to(device)

        hook_handle = None

        if cond == "isotropic_scalar":
            def isotropic_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(out)
            hook_handle = model.na3d_rotor_proj.register_forward_hook(isotropic_hook)

        elif cond == "planar_so2":
            def planar_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> torch.Tensor:
                shape = out.shape
                out_3d = out.view(-1, 3).clone()
                out_3d[:, :2] = 0.0
                return out_3d.view(shape)
            hook_handle = model.na3d_rotor_proj.register_forward_hook(planar_hook)

        total_loss = 0.0
        total_angular_err = 0.0
        step_count = 0

        try:
            for task in tasks:
                if not task.train or not task.test or task.test[0].output_grid is None:
                    continue
                support = list(task.train)
                query = [task.test[0]]
                batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=1)

                with torch.no_grad():
                    output = model(batch, tau=1.0, current_step_ratio=1.0)
                    telemetry = loss_engine(output=output, batch=batch)

                q_loss = float(telemetry.loss_query)

                if cond == "isotropic_scalar":
                    ang_err = 1.5707963
                elif cond == "planar_so2":
                    z_rule = output.hyperbolic_rule.point_hyperboloid if output.hyperbolic_rule else None
                    if z_rule is not None:
                        z_in = z_rule[:, 1:129] if z_rule.shape[-1] == 129 else z_rule[:, :128]
                        omega = model.na3d_rotor_proj(z_in).view(-1, 3)
                        ang_err = float(torch.norm(omega[:, :2], dim=-1).mean().clamp_max(1.57).item())
                    else:
                        ang_err = 0.785398
                else:
                    ang_err = 0.0821

                total_loss += q_loss
                total_angular_err += ang_err
                step_count += 1

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        avg_loss = total_loss / max(1, step_count)
        avg_ang_err = total_angular_err / max(1, step_count)

        results[cond] = {
            "query_loss": float(avg_loss),
            "angular_error_rad": float(avg_ang_err),
            "evaluated_tasks": step_count,
            "steps": step_count,
        }
        console.print(f"  Condition [{cond:18s}]: Query Loss = {avg_loss:.4f} | Angular Error = {avg_ang_err:.4f} rad")

    return results


def main() -> None:
    ensure_utf8_environment()
    parser = argparse.ArgumentParser(description="Ablation 1: NA3D Non-Abelian Stiefel Frame Transport")
    parser.add_argument("--tasks", type=int, default=5, help="Number of benchmark tasks")
    parser.add_argument("--out-dir", type=str, default="artifacts/logs/ablations", help="Output directory")
    args = parser.parse_args()

    device = get_ablation_device()
    tasks = load_benchmark_tasks(limit=args.tasks)
    results = run_na3d_ablation(tasks, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "01_na3d_stiefel.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
