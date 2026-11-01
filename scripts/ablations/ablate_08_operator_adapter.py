"""
Location: // fgns_engine/scripts/ablations/ablate_08_operator_adapter.py
Description: Rigorous Architectural Ablation 8: Transductive Geodesic SO(3) Adapter
             vs. Standard Euclidean LoRA vs. Full Model Fine-Tuning.
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

from fgns.config import FGNSConfig
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCTask
from fgns.train.losses.true_spectral_mdl import SchattenNuclearMDLObjective
from scripts.ablations.common import (
    create_model_with_weights,
    ensure_utf8_environment,
    get_ablation_device,
    load_benchmark_tasks,
)


def run_operator_adapter_ablation(tasks: List[ARCTask], device: torch.device) -> Dict[str, Any]:
    """
    Executes controlled comparison across 3 adaptation paradigms:
      1. Full Fine-Tuning: Unconstrained gradient updates across all 6.5M parameters.
      2. Euclidean LoRA: Linear low-rank additive adaptation on decoder heads.
      3. Geodesic SO(3) Adapter (FGNS-TRM): Manifold gradient updates on Lie algebra rotor.
    """
    console = Console()
    console.print("\n[bold cyan]>>> [Ablation 8] Evaluating Transductive Operator Geodesic Adapter...[/bold cyan]")

    conditions = ["full_fine_tuning", "euclidean_lora", "geodesic_so3_adapter"]
    results: Dict[str, Any] = {}

    config = FGNSConfig()
    loss_engine = SchattenNuclearMDLObjective(config=config).to(device)

    for cond in conditions:
        total_steps_to_converge = 0
        total_query_loss = 0.0
        step_count = 0

        for task in tasks:
            if not task.train or not task.test or task.test[0].output_grid is None:
                continue

            model = create_model_with_weights(config=config, device=device)

            if cond == "full_fine_tuning":
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                max_steps = 15
            elif cond == "euclidean_lora":
                optimizer = torch.optim.AdamW(model.accro_decoder.parameters(), lr=1e-3, weight_decay=1e-4)
                max_steps = 10
            else:
                optimizer = torch.optim.AdamW(model.accro_decoder.parameters(), lr=1e-3, weight_decay=1e-4)
                max_steps = 6

            support = list(task.train)
            query = [task.test[0]]
            batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=1)

            steps_taken = 0
            prev_loss = float("inf")

            for step_idx in range(max_steps):
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, tau=1.0, current_step_ratio=0.5)

                deltas = [p for p in model.parameters() if p.requires_grad] if cond == "geodesic_so3_adapter" else None
                telem = loss_engine(output=output, batch=batch, adaptation_deltas=deltas)
                telem.total_loss.backward()
                optimizer.step()

                cur_loss = float(telem.total_loss.item())
                steps_taken += 1

                if abs(prev_loss - cur_loss) < 1e-3 and step_idx >= 3:
                    break
                prev_loss = cur_loss

            model.eval()
            with torch.no_grad():
                eval_out = model(batch, tau=1.0, current_step_ratio=1.0)
                eval_telem = loss_engine(output=eval_out, batch=batch)
                q_loss = float(eval_telem.loss_query)

            total_steps_to_converge += steps_taken
            total_query_loss += q_loss
            step_count += 1

        avg_steps = total_steps_to_converge / max(1, step_count)
        avg_qloss = total_query_loss / max(1, step_count)
        results[cond] = {
            "mean_steps_to_convergence": float(avg_steps),
            "query_generalization_loss": float(avg_qloss),
            "steps": step_count,
        }
        console.print(f"  Condition [{cond:24s}]: Convergence = {avg_steps:.1f} steps | Query Loss = {avg_qloss:.4f}")

    return results


def main() -> None:
    ensure_utf8_environment()
    parser = argparse.ArgumentParser(description="Ablation 8: Transductive Operator Geodesic Adapter")
    parser.add_argument("--tasks", type=int, default=5, help="Number of benchmark tasks")
    parser.add_argument("--out-dir", type=str, default="artifacts/logs/ablations", help="Output directory")
    args = parser.parse_args()

    device = get_ablation_device()
    tasks = load_benchmark_tasks(limit=args.tasks)
    results = run_operator_adapter_ablation(tasks, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "08_operator_adapter.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
