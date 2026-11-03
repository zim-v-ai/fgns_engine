"""
Location: // fgns_engine/scripts/run_ablations.py
Description: Unified CLI Entry Point for the FGNS-TRM v4.2.0 8-Component Dual-Corpus
             Architectural Ablation Suite (ARC-AGI-2 Evaluation + ARC-AGI-1) with --gpu support.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse

from scripts.ablations.run_all_ablations import run_complete_ablation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="FGNS-TRM v4.2.0 Empirical Ablation Benchmark Suite")
    parser.add_argument("--tasks", type=int, default=10, help="Number of dual-corpus tasks to evaluate per ablation")
    parser.add_argument("--out-dir", type=str, default="artifacts/logs/ablations", help="Output directory for reports")
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