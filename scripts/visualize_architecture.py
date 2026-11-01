"""
Location: // fgns_engine/scripts/visualize_architecture.py
Description: Architecture Inspector & Neural Inference Visualizer for FGNS-TRM.
             Provides comprehensive inspection for research papers and diagnostics:
               1. Hierarchical parameter, tensor shape, and memory footprint breakdown.
               2. Full end-to-end forward trace on real ARC tasks.
               3. 2D Colored ANSI grid rendering (Input -> Latent Dynamics -> Output).
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fgns.config import FGNSConfig
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCPair, ARCTask, Grid, load_arc_dataset
from fgns.models.fgns_network import FinslerGaugeSheafNetwork

ARC_COLOR_MAP: Dict[int, str] = {
    0: "grey19",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey50",
    6: "magenta",
    7: "dark_orange",
    8: "cyan",
    9: "deep_pink3",
}

ARC_CHAR_MAP: Dict[int, str] = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}


def render_arc_grid_rich(grid_matrix: List[List[int]], title: str = "") -> Panel:
    """
    Renders a 2D ARC integer grid using colored block text.
    """
    text = Text()
    h = len(grid_matrix)
    w = len(grid_matrix[0]) if h > 0 else 0

    for r_idx, row in enumerate(grid_matrix):
        for val in row:
            v_int = int(val) % 10
            color = ARC_COLOR_MAP.get(v_int, "white")
            text.append(f" {ARC_CHAR_MAP.get(v_int, '?')} ", style=f"bold white on {color}")
        if r_idx < h - 1:
            text.append("\n")

    return Panel(
        text,
        title=f"[bold cyan]{title}[/bold cyan] ({h}x{w})",
        border_style="dim white",
        expand=False,
    )


def inspect_module_hierarchy(model: nn.Module) -> Table:
    """
    Constructs a Rich table of parameter counts and memory footprint per component.
    """
    table = Table(
        title="FGNS-TRM: Architectural Subsystem Parameter Decomposition",
        title_style="bold magenta",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Subsystem / Module", style="bold white", width=38)
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Share (%)", justify="right", style="yellow")
    table.add_column("Dtype / Memory (MB)", justify="right", style="cyan")

    total_params = sum(p.numel() for p in model.parameters())
    subsystems: Dict[str, Tuple[int, float]] = {}

    for name, module in model.named_children():
        p_count = sum(p.numel() for p in module.parameters())
        mem_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024.0 * 1024.0)
        subsystems[name] = (p_count, mem_mb)

    for name, (p_count, mem_mb) in subsystems.items():
        share = (p_count / float(max(1, total_params))) * 100.0
        table.add_row(name, f"{p_count:,}", f"{share:.2f}%", f"{mem_mb:.2f} MB")

    table.add_section()
    total_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0)
    table.add_row("Total Model Graph", f"{total_params:,}", "100.00%", f"{total_mem:.2f} MB", style="bold green")

    return table


def run_visual_forward(
    model: FinslerGaugeSheafNetwork,
    task: ARCTask,
    device: torch.device,
    console: Console,
) -> None:
    """
    Runs forward pass on an ARC task and visualizes inputs, latent dynamics, and predictions.
    """
    console.print(Panel(f"[bold yellow]Task ID: {task.task_id}[/bold yellow] | Train Pairs: {len(task.train)} | Test Pairs: {len(task.test)}", style="bold blue"))

    support_pairs = list(task.train)
    query_pair = task.test[0]

    batch = FastVectorizedBatcher.assemble_batch(
        [support_pairs],
        [[query_pair]],
        device=device,
        static_k=1,
    )

    model.eval()
    with torch.no_grad():
        output = model(batch, tau=1.0, current_step_ratio=1.0)

    in_grid = query_pair.input_grid.to_2d_list()
    in_h = query_pair.input_grid.height
    in_w = query_pair.input_grid.width
    console.print(render_arc_grid_rich(in_grid, title=f"Test Input ({task.task_id})"))

    dh = int(output.query_h_pred[0].argmax().item()) - 30 if output.query_h_pred is not None else 0
    dw = int(output.query_w_pred[0].argmax().item()) - 30 if output.query_w_pred is not None else 0
    pred_h = max(1, min(30, in_h + dh))
    pred_w = max(1, min(30, in_w + dw))

    if query_pair.output_grid is not None:
        tgt_grid = query_pair.output_grid.to_2d_list()
        tgt_h = query_pair.output_grid.height
        tgt_w = query_pair.output_grid.width
        dim_match = (pred_h == tgt_h and pred_w == tgt_w)
        dim_style = "bold green" if dim_match else "bold red"
        console.print(f"Dimension Prediction: Predicted [bold]{pred_h}x{pred_w}[/bold] | Ground Truth: [bold]{tgt_h}x{tgt_w}[/bold] -> [{dim_style}]{'MATCH' if dim_match else 'MISMATCH'}[/{dim_style}]")
        console.print(render_arc_grid_rich(tgt_grid, title="Ground Truth Target"))
    else:
        console.print(f"Dimension Prediction: Predicted [bold]{pred_h}x{pred_w}[/bold] (Ground truth hidden)")

    logits = output.query_logits[0]
    pred_classes = torch.argmax(logits, dim=0).cpu().numpy()

    eff_h = max(1, min(30, pred_h if pred_h > 0 else in_grid.__len__()))
    eff_w = max(1, min(30, pred_w if pred_w > 0 else (len(in_grid[0]) if len(in_grid) > 0 else 1)))
    cropped_pred = [[int(pred_classes[r, c]) for c in range(eff_w)] for r in range(eff_h)]

    console.print(render_arc_grid_rich(cropped_pred, title="FGNS Predicted Canvas"))

    if output.query_budget is not None:
        p_dist = output.query_budget.simplex_proportions[0].cpu().numpy()
        p_table = Table(title="Predicted S_10 Palette Budget Distribution", header_style="bold magenta", border_style="dim")
        p_table.add_column("Color Index", justify="center")
        p_table.add_column("Color Swatch", justify="center")
        p_table.add_column("Density / Expected Mass", justify="right")

        for c_idx in range(10):
            swatch = f"[{ARC_COLOR_MAP[c_idx]}] ### [/{ARC_COLOR_MAP[c_idx]}]"
            p_table.add_row(str(c_idx), swatch, f"{p_dist[c_idx]:.4f}")
        console.print(p_table)


def main() -> None:
    reconfig = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig):
        try:
            reconfig(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="FGNS-TRM Architecture Inspector & Visualizer")
    parser.add_argument("--summary", action="store_true", help="Print subsystem parameter table")
    parser.add_argument("--data", type=str, default="data/arc-agi_training_challenges.json", help="Path to challenges JSON")
    parser.add_argument("--task", type=str, default=None, help="Specific task ID to visualize")
    parser.add_argument("--random", action="store_true", help="Select random task from dataset")
    args = parser.parse_args()

    console = Console()
    config = FGNSConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.print(f"[bold green]Initializing FinslerGaugeSheafNetwork on {device}...[/bold green]")
    model = FinslerGaugeSheafNetwork(config).to(device)

    ckpt_path = Path("artifacts/checkpoints/model_final.pt")
    if ckpt_path.is_file():
        try:
            payload = torch.load(ckpt_path, map_location=device)
            state_dict = payload["models"]["model"] if isinstance(payload, dict) and "models" in payload else payload
            model.load_state_dict(state_dict, strict=False)
            console.print(f"[cyan]Loaded checkpoint weights from {ckpt_path.resolve()}[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Could not load {ckpt_path} ({e}). Using initialized weights.[/yellow]")
    else:
        console.print("[dim]No checkpoint found. Visualizing with randomly initialized weights.[/dim]")

    if args.summary or (not args.task and not args.random):
        console.print(inspect_module_hierarchy(model))

    data_path = Path(args.data).resolve()
    if data_path.is_file() and (args.task or args.random or not args.summary):
        sol_path = data_path.parent / str(data_path.name).replace("challenges", "solutions")
        tasks = load_arc_dataset(data_path, sol_path if sol_path.is_file() else None)
        if tasks:
            if args.task:
                matched = [t for t in tasks if t.task_id == args.task]
                selected_task = matched[0] if matched else tasks[0]
            else:
                selected_task = random.choice(tasks)
            run_visual_forward(model, selected_task, device, console)


if __name__ == "__main__":
    main()
