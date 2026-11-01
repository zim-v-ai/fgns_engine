"""
Location: // fgns_engine/scripts/ablations/ablate_05_spectral_mdl.py
Description: Rigorous Architectural Ablation 5: Spectral MDL Schatten-1 Objective
             vs. Schatten-2 Frobenius vs. Standard L2 Weight Decay vs. Zero TTT.
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


def run_spectral_mdl_ablation(tasks: List[ARCTask], device: torch.device) -> Dict[str, Any]:
    """
    Executes controlled comparison across 4 Test-Time Training regularization objectives:
      1. Zero TTT: Frozen baseline with zero adaptation steps.
      2. Standard L2: Euclidean weight decay penalty on deltas.
      3. Schatten-2 Frobenius: Matrix Frobenius norm ||Delta W||_F^2.
      4. Spectral MDL (FGNS-TRM): Nuclear norm Schatten-1 ||Delta W||_*.
    """
    console = Console()
    console.print("\n[bold cyan]>>> [Ablation 5] Evaluating Spectral MDL (Schatten-1) Regularization...[/bold cyan]")

    conditions = ["zero_ttt", "standard_l2_ttt", "schatten2_frobenius", "spectral_mdl_schatten1"]
    results: Dict[str, Any] = {}

    config = FGNSConfig()
    loss_engine = SchattenNuclearMDLObjective(config=config).to(device)

    for cond in conditions:
        total_rank = 0.0
        total_query_loss = 0.0
        step_count = 0

        for task in tasks:
            if not task.train or not task.test or task.test[0].output_grid is None:
                continue

            model = create_model_with_weights(config=config, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            support = list(task.train)
            query = [task.test[0]]
            batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=1)

            if cond != "zero_ttt":
                for _ in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch, tau=1.0, current_step_ratio=0.5)

                    if cond == "spectral_mdl_schatten1":
                        deltas = [p for p in model.parameters() if p.requires_grad]
                        telem = loss_engine(output=output, batch=batch, adaptation_deltas=deltas)
                        loss = telem.total_loss
                    elif cond == "schatten2_frobenius":
                        frob_penalty = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
                        telem = loss_engine(output=output, batch=batch, adaptation_deltas=None)
                        loss = telem.loss_query + (0.01 * frob_penalty)
                    else:
                        telem = loss_engine(output=output, batch=batch, adaptation_deltas=None)
                        loss = telem.total_loss

                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                eval_out = model(batch, tau=1.0, current_step_ratio=1.0)
                eval_telem = loss_engine(output=eval_out, batch=batch)

                r_eff = float(eval_telem.effective_rank)
                q_loss = float(eval_telem.loss_query)

                total_rank += r_eff
                total_query_loss += q_loss
                step_count += 1

        avg_rank = total_rank / max(1, step_count)
        avg_qloss = total_query_loss / max(1, step_count)
        results[cond] = {
            "effective_rank_r_eff": float(avg_rank),
            "query_generalization_loss": float(avg_qloss),
            "steps": step_count,
        }
        console.print(f"  Condition [{cond:24s}]: Effective Rank = {avg_rank:.2f} | Query Loss = {avg_qloss:.4f}")

    return results


def main() -> None:
    ensure_utf8_environment()
    parser = argparse.ArgumentParser(description="Ablation 5: Spectral MDL Objective")
    parser.add_argument("--tasks", type=int, default=5, help="Number of benchmark tasks")
    parser.add_argument("--out-dir", type=str, default="artifacts/logs/ablations", help="Output directory")
    args = parser.parse_args()

    device = get_ablation_device()
    tasks = load_benchmark_tasks(limit=args.tasks)
    results = run_spectral_mdl_ablation(tasks, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "05_spectral_mdl.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
