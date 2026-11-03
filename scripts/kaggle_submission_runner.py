"""
Location: // fgns_engine/scripts/kaggle_submission_runner.py
Description: Official ARC Prize 2026 (ARC-AGI-2) Zero-Failure Tournament Submission Runner
             for FGNS-TRM v4.4.0 (Finsler-Gauge Cellular Sheaf Network):
               1. Automatic detection of Kaggle Private Rerun (KAGGLE_IS_COMPETITION_RERUN)
                  vs. Interactive Save & Commit verification mode.
               2. Cheap-First Complexity Sorting: orders tasks by total pixel area
                  sum(H_in * W_in + H_out * W_out) to bank surplus time on small grids.
               3. Dynamic Time-Bank Allocation across 240 hidden test tasks within an
                  11.5-hour (41,400s) hard wall-clock safety envelope, with Multi-GPU (L4x4) support.
               4. Triple-Safety Zero-Failure Finalizer:
                    - Layer 1: Pre-loop baseline submission.json commit at t=0s.
                    - Layer 2: Per-task exception isolation & incremental atomic replacement.
                    - Layer 3: Finally-block schema audit guaranteeing 100% task coverage,
                      exact multi-test pair counts, integer range [0..9] (zero PAD=10),
                      and strict non-duplication (attempt_1 != attempt_2).
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
License: MIT-0 (MIT No Attribution)
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from fgns.config import FGNSConfig
from fgns.env.augment import AutomorphismEngine
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCPair, ARCTask, Grid, load_arc_dataset
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.models.size_predictor.boundary_size_predictor import BoundaryMetricSizePredictor
from fgns.solver.evaluator.dual_evaluator import DualAttemptEvaluator
from fgns.solver.evaluator.plackett_luce import PlackettLuceModalSynthesizer
from fgns.solver.refiner.operator_adapter import TransductiveOperatorAdapter
from fgns.solver.submit.arc_scorer import LeaderboardSummary, OfficialArcScorer
from fgns.solver.submit.atomic_writer import AtomicSubmissionWriter, KaggleSchemaAuditor
from fgns.solver.submit.path_resolver import DatasetPathResolver
from fgns.train.losses.true_spectral_mdl import SchattenNuclearMDLObjective
from fgns.types import AttemptPrediction, GridData

MAX_RERUN_SESSION_SECS: Final[float] = 41400.0
MAX_COMMIT_SESSION_SECS: Final[float] = 900.0
MIN_RERUN_TASK_BUDGET_SECS: Final[int] = 15
MAX_RERUN_TASK_BUDGET_SECS: Final[int] = 360
DEFAULT_COMMIT_TTT_TASKS: Final[int] = 2
DEFAULT_COMMIT_TTT_BUDGET_SECS: Final[int] = 12


def is_kaggle_competition_rerun(explicit_flag: bool = False) -> bool:
    """
    Detects whether Kaggle is executing the official private test set rerun.
    Checks KAGGLE_IS_COMPETITION_RERUN and FGNS_RERUN_MODE environment variables.
    """
    if explicit_flag:
        return True
    for env_key in ("KAGGLE_IS_COMPETITION_RERUN", "FGNS_RERUN_MODE"):
        val = os.environ.get(env_key, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    return False


def compute_task_pixel_complexity(task: ARCTask) -> int:
    """
    Computes total pixel area across all demonstration and test grids of a task
    for Cheap-First tournament scheduling.
    """
    total_area = 0
    for pair in task.train:
        total_area += pair.input_grid.height * pair.input_grid.width
        if pair.output_grid is not None:
            total_area += pair.output_grid.height * pair.output_grid.width
    for pair in task.test:
        total_area += pair.input_grid.height * pair.input_grid.width
    return int(total_area)


def _infer_fallback_dimensions(task: ARCTask, test_input: Grid) -> Tuple[int, int]:
    """
    Infers safe rectangular dimensions in [1, 30] from demonstration pairs when fallback is required.
    """
    in_h = max(1, min(30, int(test_input.height)))
    in_w = max(1, min(30, int(test_input.width)))

    if len(task.train) == 0:
        return in_h, in_w

    out_dims: List[Tuple[int, int]] = []
    same_as_input = True
    transposed_input = True

    for pair in task.train:
        if pair.output_grid is not None:
            oh = max(1, min(30, int(pair.output_grid.height)))
            ow = max(1, min(30, int(pair.output_grid.width)))
            out_dims.append((oh, ow))
            if (oh, ow) != (pair.input_grid.height, pair.input_grid.width):
                same_as_input = False
            if (oh, ow) != (pair.input_grid.width, pair.input_grid.height):
                transposed_input = False

    if not out_dims:
        return in_h, in_w

    if same_as_input:
        return in_h, in_w

    if len(set(out_dims)) == 1:
        return out_dims[0]

    if transposed_input:
        return in_w, in_h

    return in_h, in_w


def _sanitize_2d_grid(raw_grid: Any, fallback_h: int = 3, fallback_w: int = 3) -> List[List[int]]:
    """
    Guarantees a rectangular 2D list of Python ints strictly in [0, 9] with H, W in [1, 30].
    """
    safe_h = max(1, min(30, int(fallback_h)))
    safe_w = max(1, min(30, int(fallback_w)))

    if not isinstance(raw_grid, (list, tuple)) or len(raw_grid) == 0:
        return [[0 for _ in range(safe_w)] for _ in range(safe_h)]

    rows_raw = list(raw_grid)[:30]
    first_row = rows_raw[0]
    if not isinstance(first_row, (list, tuple)) or len(first_row) == 0:
        return [[0 for _ in range(safe_w)] for _ in range(safe_h)]

    target_h = max(1, min(30, len(rows_raw)))
    target_w = max(1, min(30, len(first_row)))

    clean_grid: List[List[int]] = []
    for r_idx in range(target_h):
        row_obj = rows_raw[r_idx] if r_idx < len(rows_raw) and isinstance(rows_raw[r_idx], (list, tuple)) else []
        clean_row: List[int] = []
        for c_idx in range(target_w):
            val = row_obj[c_idx] if c_idx < len(row_obj) else 0
            try:
                int_val = int(val)
            except (ValueError, TypeError):
                int_val = 0
            if int_val < 0 or int_val > 9:
                int_val = max(0, min(9, int_val % 10))
            clean_row.append(int_val)
        clean_grid.append(clean_row)

    return clean_grid


def _ensure_distinct_attempt_2(
    attempt_1: List[List[int]],
    attempt_2: List[List[int]],
    task: ARCTask,
) -> List[List[int]]:
    """
    Enforces the strict non-duplication invariant attempt_2 != attempt_1 while preserving [0, 9] bounds.
    """
    if attempt_1 != attempt_2:
        return attempt_2

    h = len(attempt_1)
    w = len(attempt_1[0])
    alt_grid = [list(row) for row in attempt_1]

    palette_colors: List[int] = []
    for pair in task.train:
        if pair.output_grid is not None:
            for val in pair.output_grid.buffer.ravel():
                c_val = int(val)
                if 0 <= c_val <= 9 and c_val not in palette_colors:
                    palette_colors.append(c_val)
    if not palette_colors:
        palette_colors = [0, 1]

    if h > 1 or w > 1:
        flipped_h = [list(reversed(row)) for row in alt_grid]
        if flipped_h != attempt_1:
            return flipped_h
        flipped_v = list(reversed([list(row) for row in alt_grid]))
        if flipped_v != attempt_1:
            return flipped_v

    orig_corner = alt_grid[0][0]
    replacement_color = (orig_corner + 1) % 10
    for cand in palette_colors:
        if cand != orig_corner:
            replacement_color = cand
            break
    alt_grid[0][0] = int(replacement_color)
    return alt_grid


def build_safe_fallback_attempts_for_task(task: ARCTask) -> List[Dict[str, List[List[int]]]]:
    """
    Constructs a schema-valid fallback prediction list covering every test instance in task.test.
    Guarantees rectangular [1..30] geometry, pure [0..9] colors, and attempt_1 != attempt_2.
    """
    entries: List[Dict[str, List[List[int]]]] = []
    for test_pair in task.test:
        in_grid_2d = test_pair.input_grid.to_2d_list()
        fb_h, fb_w = _infer_fallback_dimensions(task, test_pair.input_grid)

        if (fb_h, fb_w) == (test_pair.input_grid.height, test_pair.input_grid.width):
            att1 = _sanitize_2d_grid(in_grid_2d, fb_h, fb_w)
        else:
            att1 = [
                [
                    int(in_grid_2d[r % len(in_grid_2d)][c % len(in_grid_2d[0])]) % 10
                    for c in range(fb_w)
                ]
                for r in range(fb_h)
            ]
            att1 = _sanitize_2d_grid(att1, fb_h, fb_w)

        att2_raw = [list(reversed(row)) for row in att1]
        att2 = _ensure_distinct_attempt_2(att1, _sanitize_2d_grid(att2_raw, fb_h, fb_w), task)

        entries.append(
            {
                "attempt_1": att1,
                "attempt_2": att2,
            }
        )
    return entries


def sanitize_and_repair_task_predictions(
    task: ARCTask,
    predictions: Optional[Sequence[AttemptPrediction]],
) -> List[Dict[str, List[List[int]]]]:
    """
    Converts AttemptPrediction monads into Kaggle JSON dictionary entries and repairs any schema defect.
    """
    fallback_entries = build_safe_fallback_attempts_for_task(task)
    expected_count = len(task.test)

    if predictions is None or len(predictions) != expected_count:
        return fallback_entries

    repaired: List[Dict[str, List[List[int]]]] = []
    for idx in range(expected_count):
        pred = predictions[idx]
        fb_pair = fallback_entries[idx]
        try:
            raw_1 = pred.attempt_1.to_2d()
            raw_2 = pred.attempt_2.to_2d()
            clean_1 = _sanitize_2d_grid(raw_1, pred.attempt_1.height, pred.attempt_1.width)
            clean_2 = _sanitize_2d_grid(raw_2, pred.attempt_2.height, pred.attempt_2.width)
            clean_2 = _ensure_distinct_attempt_2(clean_1, clean_2, task)
            candidate_entry = {"attempt_1": clean_1, "attempt_2": clean_2}
            is_ok, _ = KaggleSchemaAuditor.validate_task_attempts(
                [candidate_entry], task_id=task.task_id, expected_test_count=1
            )
            repaired.append(candidate_entry if is_ok else fb_pair)
        except Exception:
            repaired.append(fb_pair)

    is_task_ok, _ = KaggleSchemaAuditor.validate_task_attempts(
        repaired, task_id=task.task_id, expected_test_count=expected_count
    )
    return repaired if is_task_ok else fallback_entries


def fast_zero_shot_evaluate_task(
    model: FinslerGaugeSheafNetwork,
    task: ARCTask,
    device: torch.device,
) -> List[AttemptPrediction]:
    """
    Executes rapid single-pass transductive neural inference for Commit-mode verification tasks.
    Uses FinslerGaugeSheafNetwork forward pass + PlackettLuceModalSynthesizer Mode-1 & Mode-2.
    """
    model.eval()
    predictions: List[AttemptPrediction] = []
    support_list = list(task.train)
    pttsa_delta = getattr(model, "pttsa_rule_delta", None)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = (
        (device.type == "cuda")
        and torch.cuda.is_available()
        and (torch.cuda.get_device_capability(device)[0] >= 8)
    )

    with torch.no_grad():
        for test_idx in range(len(task.test)):
            test_pair = task.test[test_idx]
            query_dummy = ARCPair(input_grid=test_pair.input_grid, output_grid=test_pair.input_grid)
            batch = FastVectorizedBatcher.assemble_batch([support_list], [[query_dummy]], device)

            with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(
                    batch,
                    tau=0.05,
                    current_step_ratio=1.0,
                    rule_perturbation=pttsa_delta,
                )

            h_in, w_in = BoundaryMetricSizePredictor.extract_input_dimensions(batch.query.in_masks)
            h_logits = output.query_h_pred[0].float()
            w_logits = output.query_w_pred[0].float()

            top2_h = torch.topk(h_logits, k=2, dim=-1).indices.to(torch.int64) - 30
            top2_w = torch.topk(w_logits, k=2, dim=-1).indices.to(torch.int64) - 30

            target_h_1 = int((h_in[0] + top2_h[0]).clamp(1, 30).item())
            target_w_1 = int((w_in[0] + top2_w[0]).clamp(1, 30).item())
            target_h_2 = int((h_in[0] + top2_h[1]).clamp(1, 30).item())
            target_w_2 = int((w_in[0] + top2_w[1]).clamp(1, 30).item())
            sec_dim: Optional[Tuple[int, int]] = (
                (target_h_2, target_w_2)
                if (target_h_2, target_w_2) != (target_h_1, target_w_1)
                else None
            )

            logits_10 = output.query_logits[0, :, :, :10].float()
            probs_10 = torch.softmax(logits_10 / 0.05, dim=-1)
            entropy_field = -torch.sum(probs_10 * torch.log(probs_10.clamp_min(1e-9)), dim=-1)
            p_color = output.p_color_matrix[0].float() if output.p_color_matrix is not None else torch.eye(10, device=device)

            attempt_1 = PlackettLuceModalSynthesizer.synthesize_candidate_grid(
                probs=probs_10,
                target_h=target_h_1,
                target_w=target_w_1,
            )
            attempt_2 = PlackettLuceModalSynthesizer.synthesize_orthogonal_mode2(
                primary_grid=attempt_1,
                mean_probs=probs_10,
                entropy_field=entropy_field,
                p_color_matrix=p_color,
                test_input=test_pair.input_grid,
                secondary_dimensions=sec_dim,
            )

            active_ent = float(entropy_field[:target_h_1, :target_w_1].mean().item())
            conf = max(0.0, min(1.0, 1.0 - (active_ent / 2.302585)))

            predictions.append(
                AttemptPrediction(
                    attempt_1=attempt_1,
                    attempt_2=attempt_2,
                    confidence=conf,
                    d4_vote_spread=(8, 8),
                )
            )

    return predictions


def discover_competition_challenges_and_solutions(
    requested_data: Optional[str] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Discovers the target ARC-AGI-2 challenge file across Kaggle 2026 mounts and local directories.
    """
    candidates: List[str] = []
    if requested_data:
        candidates.append(requested_data)
    candidates.extend(
        [
            "arc-agi_test_challenges.json",
            "arc-agi_test-challenges.json",
            "arc-agi_evaluation_challenges.json",
            "arc-agi_evaluation-challenges.json",
            "arc-agi_training_challenges.json",
        ]
    )

    for cand in candidates:
        chal_path, sol_path = DatasetPathResolver.resolve_pair(cand, use_cache=False)
        if chal_path.is_file() and chal_path.stat().st_size > 2:
            return chal_path, sol_path

    raise FileNotFoundError(
        "Unable to locate any valid ARC-AGI-2 challenges JSON file across DEFAULT_SEARCH_ROOTS."
    )


def load_pretrained_fgns_model(
    config: FGNSConfig,
    device: torch.device,
) -> Tuple[FinslerGaugeSheafNetwork, Optional[Path]]:
    """
    Instantiates FinslerGaugeSheafNetwork and loads model_final.pt if available.
    """
    model = FinslerGaugeSheafNetwork(config).to(device)
    ckpt_path = DatasetPathResolver.resolve_checkpoint("model_final.pt")

    if ckpt_path is not None and ckpt_path.is_file():
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            if isinstance(state_dict, dict) and "models" in state_dict and "model" in state_dict["models"]:
                state_dict = state_dict["models"]["model"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
        except Exception as err:
            print(f">>> [WARN] Checkpoint load warning ({ckpt_path}): {err}")
    model.eval()
    return model, ckpt_path


class KaggleTournamentOrchestrator:
    """
    Zero-Failure Master Competition Orchestrator for ARC Prize 2026 (ARC-AGI-2).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        force_rerun_mode: bool = False,
        max_tasks: Optional[int] = None,
        commit_ttt_tasks: int = DEFAULT_COMMIT_TTT_TASKS,
        commit_ttt_budget_secs: int = DEFAULT_COMMIT_TTT_BUDGET_SECS,
        rerun_task_budget_secs: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
    ) -> None:
        self.config = FGNSConfig()
        self.rerun_mode = is_kaggle_competition_rerun(force_rerun_mode)
        self.challenges_path, self.solutions_path = discover_competition_challenges_and_solutions(data_path)
        self.has_solutions = self.solutions_path is not None and self.solutions_path.is_file()

        if output_path is not None:
            self.output_path = Path(output_path).resolve()
        elif Path("/kaggle/working").is_dir():
            self.output_path = Path("/kaggle/working/submission.json").resolve()
        else:
            self.output_path = Path("submission.json").resolve()

        self.max_tasks = max_tasks
        self.commit_ttt_tasks = max(0, int(commit_ttt_tasks))
        self.commit_ttt_budget_secs = max(2, int(commit_ttt_budget_secs))
        self.rerun_task_budget_secs = rerun_task_budget_secs
        self.warmup_epochs = (
            int(warmup_epochs)
            if warmup_epochs is not None
            else (3 if self.rerun_mode else 0)
        )

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.primary_device = torch.device("cuda:0" if self.num_gpus > 0 else "cpu")
        self._submission_lock = threading.RLock()

    def run_transductive_loo_warmup(
        self,
        base_model: FinslerGaugeSheafNetwork,
        tasks: Sequence[ARCTask],
        warmup_epochs: int,
    ) -> None:
        """
        Executes Stage-II Transductive Leave-One-Out (LOO) Meta-Warmup across target task
        demonstrations (strictly never accessing test outputs) with NaN-guarded gradients,
        EMA smoothing, and a strict LOO validation improvement gate.
        """
        if warmup_epochs <= 0 or len(tasks) == 0:
            return

        import random

        eligible_tasks = [t for t in tasks if len(t.train) >= 2]
        if not eligible_tasks:
            return

        print(
            f"\n>>> [STAGE-II TRANSDUCTIVE LOO WARMUP] Commencing {warmup_epochs} epochs "
            f"across {len(eligible_tasks)} tasks on {self.primary_device}...",
            flush=True,
        )
        warmup_start = time.time()

        frozen_prefixes = (
            "projector.palette",
            "projector.lie_resonator",
            "projector.spectral_topology",
            "chordal_coord_proj",
            "query_canvas_initializer",
        )
        for name, param in base_model.named_parameters():
            if any(name.startswith(pref) for pref in frozen_prefixes):
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_params = [p for p in base_model.parameters() if p.requires_grad]
        loss_engine = SchattenNuclearMDLObjective(self.config).to(self.primary_device)
        use_fused = torch.cuda.is_available() and self.primary_device.type == "cuda"
        opt = torch.optim.AdamW(trainable_params, lr=5.0e-5, weight_decay=1e-4, fused=use_fused)
        amp_device = "cuda" if self.primary_device.type == "cuda" else "cpu"
        amp_enabled = (
            self.config.hardware.enable_bf16_amp
            and (self.primary_device.type == "cuda")
            and torch.cuda.is_available()
            and (torch.cuda.get_device_capability(self.primary_device)[0] >= 8)
        )

        def _eval_loo_loss(eval_subset: Sequence[ARCTask]) -> float:
            base_model.eval()
            total_l = 0.0
            cnt = 0
            with torch.no_grad():
                for t_eval in eval_subset:
                    supp_e = list(t_eval.train[:-1])
                    quer_e = [t_eval.train[-1]]
                    b_eval = FastVectorizedBatcher.assemble_batch(
                        [supp_e], [quer_e], self.primary_device, static_k=self.config.geometry.max_support_pairs
                    )
                    with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                        out_e = base_model(b_eval, tau=0.05, current_step_ratio=1.0)
                        tel_e = loss_engine(out_e, b_eval)
                    val_l = float(tel_e.loss_query) + 0.25 * float(tel_e.loss_size)
                    if not (val_l != val_l):
                        total_l += val_l
                        cnt += 1
            return total_l / float(max(1, cnt))

        audit_subset = eligible_tasks[: min(24, len(eligible_tasks))]
        best_loo_val = _eval_loo_loss(audit_subset)
        best_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        ema_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        ema_decay = 0.995

        for ep in range(1, warmup_epochs + 1):
            base_model.train()
            random.shuffle(eligible_tasks)
            ep_loss = 0.0
            steps = 0
            for task in eligible_tasks:
                sym, color_perm = AutomorphismEngine.sample_automorphism(preserve_background=True)
                aug_train = [AutomorphismEngine.transform_pair(p, sym, color_perm) for p in task.train]
                q_idx = random.randint(0, len(aug_train) - 1)
                supp = [p for i, p in enumerate(aug_train) if i != q_idx]
                quer = [aug_train[q_idx]]

                batch = FastVectorizedBatcher.assemble_batch(
                    [supp], [quer], self.primary_device, static_k=self.config.geometry.max_support_pairs
                )
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                    out = base_model(batch, tau=1.0, current_step_ratio=0.0)
                    proto_l = base_model.projector.palette.compute_orthogonality_loss()
                    telem = loss_engine(out, batch, prototype_ortho_loss=proto_l)
                if torch.isfinite(telem.total_loss):
                    telem.total_loss.backward()
                    for p in trainable_params:
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    opt.step()
                    with torch.no_grad():
                        for k, v in base_model.state_dict().items():
                            if torch.is_floating_point(v):
                                if not torch.isfinite(v).all():
                                    v.copy_(ema_state[k])
                                else:
                                    ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)
                    ep_loss += float(telem.total_loss.detach().item())
                    steps += 1

            with torch.no_grad():
                for k, v in base_model.state_dict().items():
                    if k in ema_state and torch.is_floating_point(v):
                        v.copy_(ema_state[k])

            ep_val_loss = _eval_loo_loss(audit_subset)
            if ep_val_loss <= best_loo_val:
                best_loo_val = ep_val_loss
                best_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
                status_tag = "COMMITTED"
            else:
                base_model.load_state_dict(best_state, strict=False)
                ema_state = {k: v.detach().clone() for k, v in best_state.items()}
                status_tag = "REVERTED TO PARETO BEST"

            avg_l = ep_loss / float(max(1, steps))
            print(
                f"   -> [Transductive LOO Warmup Epoch {ep}/{warmup_epochs}] "
                f"Train Loss: {avg_l:.4f} | LOO Val: {ep_val_loss:.4f} [{status_tag}] "
                f"({time.time() - warmup_start:.1f}s)",
                flush=True,
            )

        base_model.load_state_dict(best_state, strict=False)
        base_model.eval()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def execute(self) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """
        Executes the full competition submission pipeline with triple-safety finalization.
        """
        session_start = time.time()
        all_tasks = load_arc_dataset(
            challenges_path=self.challenges_path,
            solutions_path=self.solutions_path if self.has_solutions else None,
        )
        if self.max_tasks is not None and self.max_tasks > 0:
            all_tasks = all_tasks[: self.max_tasks]

        expected_task_ids = [t.task_id for t in all_tasks]
        task_map: Dict[str, ARCTask] = {t.task_id: t for t in all_tasks}

        """
        Safety Layer 1: Immediately construct and atomically persist a 100% schema-valid
        baseline submission.json covering every task_id and test instance at t=0s.
        """
        submission_dict: Dict[str, List[Dict[str, List[List[int]]]]] = {}
        existing_sub = AtomicSubmissionWriter.read_existing_submission(self.output_path)
        restored_task_ids: set[str] = set()

        for task in all_tasks:
            if task.task_id in existing_sub:
                valid_existing, _ = KaggleSchemaAuditor.validate_task_attempts(
                    existing_sub[task.task_id],
                    task_id=task.task_id,
                    expected_test_count=len(task.test),
                )
                if valid_existing:
                    submission_dict[task.task_id] = existing_sub[task.task_id]
                    restored_task_ids.add(task.task_id)
                    continue
            submission_dict[task.task_id] = build_safe_fallback_attempts_for_task(task)

        AtomicSubmissionWriter.write_atomic_submission(
            submission_dict=submission_dict,
            target_path=self.output_path,
            validate_schema=True,
            expected_task_ids=expected_task_ids,
        )

        sorted_tasks = sorted(all_tasks, key=compute_task_pixel_complexity)
        base_model, ckpt_path = load_pretrained_fgns_model(self.config, self.primary_device)

        mode_label = (
            "OFFICIAL PRIVATE RERUN (11.5h Full Transductive TTT)"
            if self.rerun_mode
            else f"INTERACTIVE COMMIT MODE (TTT on {self.commit_ttt_tasks} tasks + Fast Neural Inference)"
        )
        sep = "=" * 78
        print(
            f"{sep}\n"
            f"   FGNS-TRM v4.4.0 | ARC PRIZE 2026 (ARC-AGI-2) SUBMISSION RUNNER\n"
            f"   Execution Mode  : {mode_label}\n"
            f"   Hardware        : {self.primary_device} (GPUs detected: {self.num_gpus})\n"
            f"   Challenges File : {self.challenges_path}\n"
            f"   Checkpoint      : {ckpt_path if ckpt_path else 'Initialized Geometric Prior'}\n"
            f"   Total Tasks     : {len(all_tasks)} (Pre-initialized in {self.output_path})\n"
            f"{sep}",
            flush=True,
        )

        leaderboard = LeaderboardSummary()
        max_session_secs = MAX_RERUN_SESSION_SECS if self.rerun_mode else MAX_COMMIT_SESSION_SECS

        try:
            if self.warmup_epochs > 0:
                self.run_transductive_loo_warmup(
                    base_model=base_model,
                    tasks=sorted_tasks,
                    warmup_epochs=self.warmup_epochs,
                )

            pending_tasks = [t for t in sorted_tasks if t.task_id not in restored_task_ids]
            total_pending = len(pending_tasks)

            for idx, task in enumerate(pending_tasks):
                elapsed_total = time.time() - session_start
                if elapsed_total >= max_session_secs:
                    print(
                        f">>> [GLOBAL TIME GUARD] Reached {elapsed_total:.1f}s ceiling. "
                        f"Preserving pre-initialized predictions for remaining tasks.",
                        flush=True,
                    )
                    break

                remaining_tasks = max(1, total_pending - idx)
                remaining_secs = max(1.0, max_session_secs - elapsed_total)

                run_full_ttt = self.rerun_mode or (idx < self.commit_ttt_tasks)
                if self.rerun_mode:
                    if self.rerun_task_budget_secs is not None:
                        task_budget = int(self.rerun_task_budget_secs)
                    else:
                        dynamic_share = int(remaining_secs // float(remaining_tasks))
                        task_budget = max(
                            MIN_RERUN_TASK_BUDGET_SECS,
                            min(MAX_RERUN_TASK_BUDGET_SECS, dynamic_share),
                        )
                else:
                    task_budget = self.commit_ttt_budget_secs

                t0 = time.time()
                try:
                    if run_full_ttt:
                        model_first, model_polished = TransductiveOperatorAdapter.refine_task_weights(
                            base_model=base_model,
                            task=task,
                            config=self.config,
                            time_budget_secs=task_budget,
                            device=self.primary_device,
                        )
                        predictions = DualAttemptEvaluator.evaluate_task(
                            model_first=model_first,
                            model_polished=model_polished,
                            task=task,
                            config=self.config,
                            device=self.primary_device,
                        )
                        del model_first
                        del model_polished
                    else:
                        predictions = fast_zero_shot_evaluate_task(
                            model=base_model,
                            task=task,
                            device=self.primary_device,
                        )
                except Exception as task_err:
                    print(
                        f">>> [TASK RECOVERY] Task {task.task_id} encountered exception ({task_err}); "
                        f"using safe geometric fallback.",
                        flush=True,
                    )
                    predictions = None

                sanitized_entries = sanitize_and_repair_task_predictions(task, predictions)

                with self._submission_lock:
                    submission_dict[task.task_id] = sanitized_entries
                    if run_full_ttt or ((idx + 1) % 25 == 0) or (idx + 1 == total_pending):
                        AtomicSubmissionWriter.write_atomic_submission(
                            submission_dict=submission_dict,
                            target_path=self.output_path,
                            validate_schema=True,
                        )

                if self.has_solutions and predictions is not None:
                    report = OfficialArcScorer.evaluate_task_detailed(predictions, task)
                    leaderboard.total_tasks += 1
                    if report.task_solved:
                        leaderboard.solved_tasks += 1
                    leaderboard.total_test_instances += report.num_tests
                    leaderboard.solved_test_instances += report.instances_solved

                task_dt = time.time() - t0
                if run_full_ttt or ((idx + 1) % 20 == 0) or (idx + 1 == total_pending):
                    mode_tag = "TTT+D4" if run_full_ttt else "FAST-NS"
                    print(
                        f"[{idx + 1:>3}/{total_pending}] Task {task.task_id} ({mode_tag}) | "
                        f"Tests: {len(task.test)} | Area: {compute_task_pixel_complexity(task):>4}px | "
                        f"Time: {task_dt:.2f}s",
                        flush=True,
                    )

                if run_full_ttt:
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

        finally:
            """
            Safety Layer 3: Final mandatory audit & atomic persistence across all tasks.
            Guarantees zero missing task IDs, zero PAD=10 leaks, and attempt_1 != attempt_2.
            """
            for req_id in expected_task_ids:
                t_obj = task_map[req_id]
                if req_id not in submission_dict:
                    submission_dict[req_id] = build_safe_fallback_attempts_for_task(t_obj)
                else:
                    valid_t, _ = KaggleSchemaAuditor.validate_task_attempts(
                        submission_dict[req_id],
                        task_id=req_id,
                        expected_test_count=len(t_obj.test),
                    )
                    if not valid_t:
                        submission_dict[req_id] = build_safe_fallback_attempts_for_task(t_obj)

            is_valid, audit_errors = KaggleSchemaAuditor.validate_submission_dict(
                submission_dict=submission_dict,
                expected_task_ids=expected_task_ids,
            )
            if not is_valid:
                raise RuntimeError(f"Final submission audit failed: {audit_errors[:5]}")

            AtomicSubmissionWriter.write_atomic_submission(
                submission_dict=submission_dict,
                target_path=self.output_path,
                validate_schema=True,
                expected_task_ids=expected_task_ids,
            )

            local_mirror = Path("submission.json").resolve()
            if local_mirror != self.output_path:
                try:
                    AtomicSubmissionWriter.write_atomic_submission(
                        submission_dict=submission_dict,
                        target_path=local_mirror,
                        validate_schema=False,
                    )
                except OSError:
                    pass

            total_elapsed = time.time() - session_start
            print(
                f"{sep}\n"
                f"   FGNS-TRM v4.4.0 SUBMISSION FINALIZED SUCCESSFULLY\n"
                f"   Output File      : {self.output_path} ({self.output_path.stat().st_size:,} bytes)\n"
                f"   Tasks Verified   : {len(submission_dict)} / {len(expected_task_ids)}\n"
                f"   Schema Audit     : PASSED (0 errors, 0 PAD tokens, attempt_1 != attempt_2)\n"
                + (
                    f"   Local Eval Score : {leaderboard.solved_tasks}/{leaderboard.total_tasks} tasks "
                    f"(Kaggle Instance Score: {leaderboard.kaggle_score_pct:.2f}%)\n"
                    if self.has_solutions and leaderboard.total_tasks > 0
                    else ""
                )
                + f"   Total Wall Time  : {total_elapsed:.2f}s\n"
                f"{sep}",
                flush=True,
            )

        return submission_dict


def verify_kaggle_submission_runner_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for scripts/kaggle_submission_runner.py.
    """
    import tempfile

    g_in = Grid([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    g_out = Grid([[1, 2, 0], [2, 0, 1], [0, 1, 2]])
    mock_task = ARCTask(
        task_id="diag_task_001",
        train=(ARCPair(input_grid=g_in, output_grid=g_out),),
        test=(
            ARCPair(input_grid=g_in, output_grid=None),
            ARCPair(input_grid=g_out, output_grid=None),
        ),
    )

    fb = build_safe_fallback_attempts_for_task(mock_task)
    if len(fb) != 2:
        return False

    ok, errs = KaggleSchemaAuditor.validate_task_attempts(fb, task_id="diag_task_001", expected_test_count=2)
    if not ok or len(errs) > 0:
        return False

    for pair_entry in fb:
        if pair_entry["attempt_1"] == pair_entry["attempt_2"]:
            return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        chal_file = tmp_path / "mock_challenges.json"
        out_file = tmp_path / "submission.json"
        chal_payload = {
            "mock_a": {
                "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
                "test": [
                    {"input": [[0, 2], [2, 0]]},
                    {"input": [[3, 0], [0, 3]]},
                ],
            },
            "mock_b": {
                "train": [{"input": [[1, 1], [1, 1]], "output": [[2, 2], [2, 2]]}],
                "test": [{"input": [[4, 4], [4, 4]]}],
            },
        }
        chal_file.write_text(json.dumps(chal_payload), encoding="utf-8")

        runner = KaggleTournamentOrchestrator(
            data_path=str(chal_file),
            output_path=str(out_file),
            force_rerun_mode=False,
            commit_ttt_tasks=0,
        )
        sub = runner.execute()
        if not out_file.is_file():
            return False
        valid, errors = KaggleSchemaAuditor.validate_submission_dict(
            sub, expected_task_ids=["mock_a", "mock_b"]
        )
        if not valid or len(errors) > 0:
            return False
        if len(sub["mock_a"]) != 2 or len(sub["mock_b"]) != 1:
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FGNS-TRM v4.4.0 Official ARC Prize 2026 (ARC-AGI-2) Submission Runner"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Optional path or filename of ARC-AGI-2 challenges JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output submission.json path (defaults to /kaggle/working/submission.json or ./submission.json)",
    )
    parser.add_argument(
        "--rerun-mode",
        action="store_true",
        help="Force full competition rerun mode (equivalent to KAGGLE_IS_COMPETITION_RERUN=1)",
    )
    parser.add_argument(
        "--tasks",
        "-n",
        type=int,
        default=None,
        help="Optional limit on number of tasks to process",
    )
    parser.add_argument(
        "--commit-ttt-tasks",
        type=int,
        default=int(os.environ.get("FGNS_COMMIT_TASKS", str(DEFAULT_COMMIT_TTT_TASKS))),
        help="Number of tasks to run full TTT on during Commit mode (default: 2)",
    )
    parser.add_argument(
        "--commit-ttt-time",
        type=int,
        default=int(os.environ.get("FGNS_COMMIT_TTT_SECS", str(DEFAULT_COMMIT_TTT_BUDGET_SECS))),
        help="Per-task TTT time budget in seconds during Commit mode (default: 12s)",
    )
    parser.add_argument(
        "--rerun-task-time",
        type=int,
        default=None,
        help="Optional fixed per-task TTT time budget in seconds during Rerun mode",
    )
    parser.add_argument(
        "--warmup-epochs",
        "-w",
        type=int,
        default=None,
        help="Stage-II transductive LOO warmup epochs (defaults to 3 in Rerun mode, 0 in Commit mode)",
    )
    args = parser.parse_args()

    orchestrator = KaggleTournamentOrchestrator(
        data_path=args.data,
        output_path=args.output,
        force_rerun_mode=args.rerun_mode,
        max_tasks=args.tasks,
        commit_ttt_tasks=args.commit_ttt_tasks,
        commit_ttt_budget_secs=args.commit_ttt_time,
        rerun_task_budget_secs=args.rerun_task_time,
        warmup_epochs=args.warmup_epochs,
    )
    orchestrator.execute()


if __name__ == "__main__":
    main()
