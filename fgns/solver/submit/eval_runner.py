"""
Location: // fgns_engine/fgns/solver/submit/eval_runner.py
Description: SOTA Master Combat Test-Time Solver Orchestrator with Stage-II Transductive LOO
             Test-Set Warmup and Dynamic Time-Bank Recirculation.
             Apex Competition Pipeline for FGNS v2.1.0 (Finsler-Gauge Sheaf):
               1. Priority Checkpoint Resolution: directly loads artifacts/checkpoints/model_final.pt
                  committed by MetaTrainingPipeline before probing external Kaggle mounts.
               2. Stage-II Transductive LOO Meta-Warmup (--warmup-epochs).
               3. Dynamic Time-Bank compute surplus recirculation (allocating up to 480s).
               4. High-throughput Coherent Multi-Fold Batched Geodesic NA3D-PTTSA & Newton-Schulz MDL TTT.
               5. Two-Scale Plackett-Luce Mode-2 modal hypothesis bifurcation (Attempt 1 & Attempt 2).
Language: Python 3.12+ / PyTorch 2.5+ / Tournament Orchestration sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
import collections
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Final, List, Optional, Sequence, Set, Tuple, Union

import torch

from fgns.config import FGNSConfig
from fgns.env.augment import AutomorphismEngine
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import (
    ARCPair,
    ARCTask,
    Grid,
    load_arc_dataset,
)
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.solver.evaluator.dual_evaluator import DualAttemptEvaluator
from fgns.solver.refiner.operator_adapter import TransductiveOperatorAdapter
from fgns.solver.submit.arc_scorer import (
    LeaderboardSummary,
    OfficialArcScorer,
)
from fgns.solver.submit.atomic_writer import (
    AtomicSubmissionWriter,
    KaggleSchemaAuditor,
)
from fgns.solver.submit.path_resolver import DatasetPathResolver
from fgns.train.losses.true_spectral_mdl import SchattenNuclearMDLObjective
from fgns.train.meta_train.logger import DualStreamSessionLogger
from fgns.types import AttemptPrediction, TaskSubmission

DEFAULT_TASK_TIME_BUDGET: Final[int] = 333
DEFAULT_MIN_TIME_BUDGET: Final[int] = 120
DEFAULT_MAX_BOOSTED_BUDGET: Final[int] = 480
MAX_GLOBAL_SESSION_SECS: Final[float] = 41400.0


class EvaluationPipelineRunner:
    """
    SOTA Master Combat Test-Time Optimization & Tournament Execution Pipeline.
    """

    def __init__(self, config: FGNSConfig, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.device = torch.device(
            config.hardware.device_name if torch.cuda.is_available() else "cpu"
        )

        log_path = getattr(args, "log", "solver.log")
        self.logger = DualStreamSessionLogger(log_path if log_path else "solver.log")

        self.model = FinslerGaugeSheafNetwork(config).to(self.device)
        self.load_pretrained_weights()

        self.time_bank_secs: int = 0
        self.submission_path = Path("submission.json")
        self.leaderboard_summary = LeaderboardSummary()

    def load_pretrained_weights(self) -> None:
        """Prioritizes artifacts/checkpoints/model_final.pt committed by MetaTrainingPipeline."""
        primary_ckpt = Path("artifacts/checkpoints/model_final.pt")
        if primary_ckpt.is_file():
            ckpt_candidate: Optional[Path] = primary_ckpt.resolve()
        else:
            ckpt_candidate = DatasetPathResolver.resolve_checkpoint("model_final.pt")

        if ckpt_candidate is not None and ckpt_candidate.is_file():
            try:
                state_dict = torch.load(ckpt_candidate, map_location=self.device)
                if isinstance(state_dict, dict) and "models" in state_dict and "model" in state_dict["models"]:
                    state_dict = state_dict["models"]["model"]
                self.model.load_state_dict(state_dict, strict=False)
                msg = f">>> [INIT] Successfully restored SOTA Meta-Weights from {ckpt_candidate}\n"
                self.logger.write(msg)
            except Exception as exc:
                msg = f">>> [WARN] Failed to load checkpoint ({exc}). Running with initialized cortex.\n"
                self.logger.write(msg)
        else:
            msg = ">>> [WARN] No checkpoint discovered. Running with freshly initialized cortex.\n"
            self.logger.write(msg)

    def run_transductive_loo_warmup(self, tasks: Sequence[ARCTask], warmup_epochs: int) -> None:
        if warmup_epochs <= 0 or len(tasks) == 0:
            return

        self.logger.write(
            f"\n>>> [STAGE-II TRANSDUCTIVE LOO WARMUP] Commencing {warmup_epochs} epochs across {len(tasks)} tasks...\n"
        )
        warmup_start = time.time()
        self.model.train()
        loss_engine = SchattenNuclearMDLObjective(self.config).to(self.device)

        use_fused = torch.cuda.is_available() and self.device.type == "cuda"
        opt = torch.optim.AdamW(self.model.parameters(), lr=1.5e-4, weight_decay=1e-4, fused=use_fused)
        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        amp_enabled = self.config.hardware.enable_bf16_amp and (self.device.type == "cuda")

        task_list = [t for t in tasks if len(t.train) >= 2]
        for ep in range(1, warmup_epochs + 1):
            random.shuffle(task_list)
            ep_loss = 0.0
            steps = 0
            for task in task_list:
                sym, color_perm = AutomorphismEngine.sample_automorphism(preserve_background=True)
                aug_train = [AutomorphismEngine.transform_pair(p, sym, color_perm) for p in task.train]
                q_idx = random.randint(0, len(aug_train) - 1)
                supp = [p for i, p in enumerate(aug_train) if i != q_idx]
                quer = [aug_train[q_idx]]

                batch = FastVectorizedBatcher.assemble_batch(
                    [supp], [quer], self.device, static_k=self.config.geometry.max_support_pairs
                )
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                    out = self.model(batch, tau=0.25)
                    proto_l = self.model.projector.palette.compute_orthogonality_loss()
                    telem = loss_engine(out, batch, prototype_ortho_loss=proto_l)
                telem.total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                ep_loss += float(telem.total_loss.detach().item())
                steps += 1

            avg_l = ep_loss / float(max(1, steps))
            self.logger.write(
                f"   -> [Transductive LOO Warmup Epoch {ep}/{warmup_epochs}] Mean Loss: {avg_l:.4f} ({time.time() - warmup_start:.1f}s)\n"
            )

        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def calculate_allocated_budget(self) -> int:
        base_budget = getattr(self.args, "time", DEFAULT_TASK_TIME_BUDGET)

        if self.time_bank_secs > 60:
            surplus_bonus = min(
                self.time_bank_secs // 4,
                DEFAULT_MAX_BOOSTED_BUDGET - base_budget,
            )
            allocated = base_budget + surplus_bonus
        else:
            allocated = max(base_budget, DEFAULT_MIN_TIME_BUDGET)

        return int(allocated)

    def execute(self) -> None:
        challenges_file, solutions_file = DatasetPathResolver.resolve_pair(self.args.data)
        has_solutions = solutions_file is not None and solutions_file.is_file()

        tasks = load_arc_dataset(
            challenges_path=challenges_file,
            solutions_path=solutions_file if has_solutions else None,
        )

        total_dataset_tasks = len(tasks)
        max_task_limit = (
            self.args.tasks if getattr(self.args, "tasks", None) is not None else total_dataset_tasks
        )

        submission_dict = AtomicSubmissionWriter.read_existing_submission(self.submission_path)
        completed_task_ids = set(submission_dict.keys())

        sol_status = str(solutions_file.resolve()) if (has_solutions and solutions_file is not None) else "NOT FOUND (Competition Mode)"
        sep_line = "=" * 60

        banner = (
            f"{sep_line}\n"
            f"   FGNS v2.1.0 EVALUATION PIPELINE: FINSLER-GAUGE SHEAF      \n"
            f"   Hardware Platform : {self.device} (NVIDIA RTX 5070 sm_120)\n"
            f"   Target Dataset    : {challenges_file.resolve()}\n"
            f"   Ground Truth Sol  : {sol_status}\n"
            f"   Base Task Budget  : {getattr(self.args, 'time', DEFAULT_TASK_TIME_BUDGET)}s (Dynamic Time-Bank Active)\n"
            f"   Tasks Available   : {total_dataset_tasks} (Limit: {max_task_limit})\n"
            f"   Already Completed : {len(completed_task_ids)} tasks restored\n"
            f"{sep_line}\n\n"
        )
        self.logger.write(banner)

        warmup_epochs = int(getattr(self.args, "warmup_epochs", 0))
        if warmup_epochs > 0:
            self.run_transductive_loo_warmup(tasks[:max_task_limit], warmup_epochs=warmup_epochs)

        global_start_time = time.time()
        processed_count = 0
        exact_match_score = 0

        for t_idx, task in enumerate(tasks):
            if task.task_id in completed_task_ids:
                continue

            if processed_count >= max_task_limit:
                msg = f">>> [LIMIT REACHED] Processed target limit of {max_task_limit} tasks. Halting.\n"
                self.logger.write(msg)
                break

            global_elapsed = time.time() - global_start_time
            if global_elapsed >= MAX_GLOBAL_SESSION_SECS:
                self.logger.write(
                    f"\n>>> [GLOBAL TIMEOUT IMMINENT] Elapsed {global_elapsed:.1f}s. Committing graceful shutdown.\n"
                )
                break

            processed_count += 1
            allocated_budget = self.calculate_allocated_budget()
            task_start_time = time.time()

            task_header = (
                f"\n[Task {t_idx + 1:>3}/{total_dataset_tasks}] ID: {task.task_id} | "
                f"Train Pairs: {len(task.train)} | Test Pairs: {len(task.test)} | "
                f"Budget: {allocated_budget}s (Bank: {self.time_bank_secs}s)\n"
            )
            self.logger.write(task_header)

            model_first, model_polished = TransductiveOperatorAdapter.refine_task_weights(
                base_model=self.model,
                task=task,
                config=self.config,
                time_budget_secs=allocated_budget,
                device=self.device,
            )

            predictions = DualAttemptEvaluator.evaluate_task(
                model_first=model_first,
                model_polished=model_polished,
                task=task,
                config=self.config,
                device=self.device,
            )

            task_elapsed_secs = int(time.time() - task_start_time)

            surplus_time = allocated_budget - task_elapsed_secs
            self.time_bank_secs += surplus_time

            if has_solutions:
                report = OfficialArcScorer.evaluate_task_detailed(predictions, task)
                for line in report.telemetry_lines:
                    self.logger.write(line + "\n")

                if report.task_solved:
                    exact_match_score += 1
                    current_pct = (float(exact_match_score) / float(processed_count)) * 100.0
                    win_msg = (
                        f">>> 🏆 [SCORE++] Task {task.task_id} PERFECTLY SOLVED! "
                        f"Current Leaderboard: {exact_match_score}/{processed_count} ({current_pct:.2f}%)\n"
                    )
                    self.logger.write(win_msg)

                self.leaderboard_summary.total_tasks += 1
                if report.task_solved:
                    self.leaderboard_summary.solved_tasks += 1

                self.leaderboard_summary.total_test_instances += report.num_tests
                self.leaderboard_summary.solved_test_instances += report.instances_solved

            try:
                AtomicSubmissionWriter.update_task_submission(
                    task_id=task.task_id,
                    predictions=predictions,
                    target_path=self.submission_path,
                )
                completed_task_ids.add(task.task_id)
            except Exception as err:
                err_msg = f">>> [ERROR] Submission writing error on task {task.task_id}: {err}\n"
                self.logger.write(err_msg)

            del model_first
            del model_polished
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vram_peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            )

            footer = (
                f"      [PROGRESS] Finished in {task_elapsed_secs}s. "
                f"Completed: {len(completed_task_ids)}/{total_dataset_tasks}. "
                f"VRAM Peak: {vram_peak_mb:.0f}MB\n"
            )
            self.logger.write(footer)

        total_time_elapsed = time.time() - global_start_time
        final_pct = (float(exact_match_score) / float(max(1, processed_count))) * 100.0

        summary_text = (
            f"\n{sep_line}\n"
            f"   FGNS v2.1.0 TOURNAMENT EXECUTION REPORT              \n"
            f"   Tasks Evaluated   : {processed_count}\n"
            f"   Tasks Solved 100% : {exact_match_score} / {processed_count} ({final_pct:.2f}%)\n"
            f"   Kaggle Score Pct  : {self.leaderboard_summary.kaggle_score_pct:.2f}%\n"
            f"   Time Bank Surplus : {self.time_bank_secs}s\n"
            f"   Total Time Elapsed: {total_time_elapsed:.1f}s\n"
            f"   Submission Path   : {self.submission_path.resolve()}\n"
            f"{sep_line}\n"
        )
        self.logger.write(summary_text)
        self.logger.close()


def verify_eval_runner_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for eval_runner.py.
    """
    class MockArgs(argparse.Namespace):
        data = "arc-agi_training_challenges.json"
        time = 333
        tasks = 1
        warmup_epochs = 0
        log = None

    config = FGNSConfig()
    orchestrator = EvaluationPipelineRunner(config=config, args=MockArgs())

    orchestrator.time_bank_secs = 240
    boosted_budget = orchestrator.calculate_allocated_budget()

    if boosted_budget != 393:
        return False

    orchestrator.time_bank_secs = 2000
    clamped_budget = orchestrator.calculate_allocated_budget()
    if clamped_budget != 480:
        return False

    orchestrator.logger.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="FGNS v2.1.0 SOTA Combat Tournament Solver")
    parser.add_argument(
        "--data", "-d", type=str, default="arc-agi_test_challenges.json",
        help="Path to challenges JSON file or dataset directory",
    )
    parser.add_argument(
        "--time", "-t", type=int, default=DEFAULT_TASK_TIME_BUDGET,
        help="Base task time budget in seconds (default 333s)",
    )
    parser.add_argument(
        "--tasks", "-n", type=int, default=None,
        help="Maximum number of tasks to process",
    )
    parser.add_argument(
        "--warmup-epochs", "-w", type=int, default=0,
        help="Stage-II transductive LOO warmup epochs on target test demonstrations (default 0, recommended 3 for Kaggle)",
    )
    parser.add_argument(
        "--log", "-l", type=str, default="solver.log",
        help="Path to session execution log file",
    )
    args = parser.parse_args()

    env_tasks = os.environ.get("FGNS_MAX_TASKS") or os.environ.get("AEON_MAX_TASKS")
    if env_tasks is not None:
        try:
            args.tasks = int(env_tasks)
        except ValueError:
            pass

    env_time = os.environ.get("FGNS_TIME_PER_TASK") or os.environ.get("AEON_TIME_PER_TASK")
    if env_time is not None:
        try:
            args.time = int(env_time)
        except ValueError:
            pass

    config = FGNSConfig()
    orchestrator = EvaluationPipelineRunner(config=config, args=args)
    orchestrator.execute()


if __name__ == "__main__":
    main()

