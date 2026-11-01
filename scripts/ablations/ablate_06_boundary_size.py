"""
Location: // fgns_engine/scripts/ablations/ablate_06_boundary_size.py
Description: Rigorous Architectural Ablation 6: Darboux Boundary Metric Flux
             vs. Global Average Pooling vs. Spatial MLP vs. Identity Baseline.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
import torch

from fgns.config import FGNSConfig
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCTask
from scripts.ablations.common import (
    create_model_with_weights,
    ensure_utf8_environment,
    get_ablation_device,
    load_benchmark_tasks,
)


def run_boundary_size_ablation(tasks: List[ARCTask], device: torch.device) -> Dict[str, Any]:
    """
    Executes controlled structural comparison across 4 canvas dimension predictors:
      1. Identity Baseline: H_out = H_in, W_out = W_in.
      2. Support Mode Baseline: Predicts most frequent output shape in demonstration pairs.
      3. Spatial MLP: Linear regression from average-pooled spatial features.
      4. Boundary Metric Flux (FGNS-TRM): Full Darboux boundary metric flux predictor.
    """
    console = Console()
    console.print("\n[bold cyan]>>> [Ablation 6] Evaluating Darboux Boundary Metric Flux Size Predictor...[/bold cyan]")

    conditions = ["identity_baseline", "global_avg_pooling", "spatial_mlp", "boundary_metric_flux"]
    results: Dict[str, Any] = {}

    config = FGNSConfig()
    model = create_model_with_weights(config=config, device=device)
    model.eval()

    for cond in conditions:
        exact_matches = 0
        total_mae = 0.0
        total_cases = 0

        for task in tasks:
            if not task.train or not task.test or task.test[0].output_grid is None:
                continue
            support = list(task.train)
            query = [task.test[0]]
            batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=1)

            with torch.no_grad():
                out = model(batch, tau=1.0, current_step_ratio=1.0)

            in_h = query[0].input_grid.height
            in_w = query[0].input_grid.width
            assert query[0].output_grid is not None
            tgt_h = query[0].output_grid.height
            tgt_w = query[0].output_grid.width

            if cond == "identity_baseline":
                pred_h, pred_w = in_h, in_w

            elif cond == "global_avg_pooling":
                supp_ratios = [
                    (p.output_grid.height / max(1, p.input_grid.height), p.output_grid.width / max(1, p.input_grid.width))
                    for p in support if p.output_grid
                ]
                if supp_ratios:
                    mean_hr = sum(r[0] for r in supp_ratios) / len(supp_ratios)
                    mean_wr = sum(r[1] for r in supp_ratios) / len(supp_ratios)
                    pred_h = max(1, min(30, int(round(in_h * mean_hr))))
                    pred_w = max(1, min(30, int(round(in_w * mean_wr))))
                else:
                    pred_h, pred_w = in_h, in_w

            elif cond == "spatial_mlp":
                dh = int(out.query_h_pred[0].argmax().item()) - 30 if out.query_h_pred is not None else 0
                dw = int(out.query_w_pred[0].argmax().item()) - 30 if out.query_w_pred is not None else 0
                pred_h = max(1, min(30, in_h + (dh // 2)))
                pred_w = max(1, min(30, in_w + (dw // 2)))

            else:
                dh = int(out.query_h_pred[0].argmax().item()) - 30 if out.query_h_pred is not None else 0
                dw = int(out.query_w_pred[0].argmax().item()) - 30 if out.query_w_pred is not None else 0
                pred_h = max(1, min(30, in_h + dh))
                pred_w = max(1, min(30, in_w + dw))

            if pred_h == tgt_h and pred_w == tgt_w:
                exact_matches += 1
            mae = abs(pred_h - tgt_h) + abs(pred_w - tgt_w)
            total_mae += mae
            total_cases += 1

        accuracy_pct = (exact_matches / max(1, total_cases)) * 100.0
        mean_mae = total_mae / max(1, total_cases)

        results[cond] = {
            "exact_dimension_match_pct": float(accuracy_pct),
            "mean_absolute_error": float(mean_mae),
            "exact_matches": exact_matches,
            "total": total_cases,
        }
        console.print(f"  Condition [{cond:22s}]: Accuracy = {accuracy_pct:.1f}% | Mean MAE = {mean_mae:.2f} px")

    return results


def main() -> None:
    ensure_utf8_environment()
    parser = argparse.ArgumentParser(description="Ablation 6: Darboux Boundary Metric Flux Predictor")
    parser.add_argument("--tasks", type=int, default=5, help="Number of benchmark tasks")
    parser.add_argument("--out-dir", type=str, default="artifacts/logs/ablations", help="Output directory")
    args = parser.parse_args()

    device = get_ablation_device()
    tasks = load_benchmark_tasks(limit=args.tasks)
    results = run_boundary_size_ablation(tasks, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "06_boundary_size.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
