"""
Location: // fgns_engine/scripts/ablations/common.py
Description: GPU-Accelerated Pure ARC-AGI-2 Evaluation Benchmark Loader, VRAM-Cached Deterministic
             Model Factory, Perturbation-Delta LOO Adaptation Harness, Compact Single-Pass Task
             Grid Visualizer, and 12-Metric Metrology Engine (FGNS-TRM v4.5.0):
               1. Pure Perturbation-Delta Spectral MDL in run_honest_loo_support_adaptation:
                  evaluates Schatten-1 nuclear norm and Von Neumann entropy strictly on
                  Delta_W = W - W_init, preserving pre-trained base representations!
               2. Explicit PoissonDirichletCavitySolver.clear_cache() at every task boundary.
Author: Vyacheslav Zimovsky (v.zimovsky.ai@gmail.com)
Language: Python 3.12+ / PyTorch 2.5+ / CUDA 12.8 sm_120 Accelerated
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.config import FGNSConfig
from fgns.env.augment import AutomorphismEngine, ColorPermutation, D4Group
from fgns.env.batcher import FastVectorizedBatcher
from fgns.env.task import ARCPair, ARCTask, load_arc_dataset
from fgns.kernels.cuda_fgns_ops import is_cuda_kernel_image_usable
from fgns.kernels.spectral_laplacian import PoissonDirichletCavitySolver
from fgns.models.fgns_network import FinslerGaugeSheafNetwork
from fgns.train.losses.true_spectral_mdl import SchattenNuclearMDLObjective

PRIORITY_ARC2_TASK_IDS: Final[Tuple[str, ...]] = (
    "0934a4d8",
    "135a2760",
    "136b0064",
    "13e47133",
    "142ca369",
)

_VRAM_BASE_STATE_CACHE: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}


def ensure_utf8_environment() -> None:
    """Configures system stdout and stderr to UTF-8 encoding with replace fallback."""
    reconfig_out = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig_out):
        try:
            reconfig_out(encoding="utf-8", errors="replace")
        except Exception:
            pass
    reconfig_err = getattr(sys.stderr, "reconfigure", None)
    if callable(reconfig_err):
        try:
            reconfig_err(encoding="utf-8", errors="replace")
        except Exception:
            pass


def set_deterministic_seed(seed: int = 42) -> None:
    """Locks random seeds across Python, NumPy, and PyTorch for reproducible comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda_kernel_image_usable():
        torch.cuda.manual_seed_all(seed)


def get_ablation_device(force_gpu: bool = False, device_override: Optional[str] = None) -> torch.device:
    """
    Returns preferred PyTorch execution device, enforcing IEEE-754 23-bit FP32 precision
    on explicit float32 matmuls while keeping BFloat16 Tensor Cores active for autocast.
    """
    cuda_ok = is_cuda_kernel_image_usable()

    if device_override is not None:
        req = device_override.lower().strip()
        if req.startswith("cuda") and cuda_ok:
            dev = torch.device(req)
        else:
            dev = torch.device("cuda" if cuda_ok else "cpu")
    else:
        dev = torch.device("cuda" if cuda_ok else "cpu")

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif force_gpu or torch.cuda.is_available():
        print(
            "[WARN] RTX 5070 Blackwell (sm_120) requires PyTorch built with CUDA 12.8 (cu128), "
            f"current build is torch.version.cuda={getattr(torch.version, 'cuda', None)}."
        )
    return dev


def load_benchmark_tasks(
    data_path: Optional[Path] = None,
    limit: int = 10,
    include_arc2_eval: bool = True,
) -> List[ARCTask]:
    """
    Loads a 100% pure ARC-AGI-2 Evaluation benchmark whenever arc-agi_evaluation_challenges.json
    is available on disk (prioritizing canonical geometric archetypes), falling back to
    data_path only if the evaluation manifest is absent.
    """
    selected_tasks: List[ARCTask] = []
    seen_ids: Set[str] = set()

    eval_chal = Path("data/arc-agi_evaluation_challenges.json").resolve()
    eval_sol = Path("data/arc-agi_evaluation_solutions.json").resolve()

    if include_arc2_eval and eval_chal.is_file() and eval_sol.is_file():
        eval_tasks = load_arc_dataset(eval_chal, eval_sol)
        eval_with_truth = [
            t for t in eval_tasks if len(t.test) > 0 and t.test[0].output_grid is not None
        ]
        id_map = {t.task_id: t for t in eval_with_truth}

        for pid in PRIORITY_ARC2_TASK_IDS:
            if pid in id_map and len(selected_tasks) < limit:
                selected_tasks.append(id_map[pid])
                seen_ids.add(pid)

        for t in eval_with_truth:
            if len(selected_tasks) >= limit:
                break
            if t.task_id not in seen_ids:
                selected_tasks.append(t)
                seen_ids.add(t.task_id)

    target_path = data_path or Path("data/arc-agi_training_challenges.json").resolve()
    if target_path.is_file() and len(selected_tasks) < limit:
        sol_path = target_path.parent / str(target_path.name).replace("challenges", "solutions")
        train_tasks = load_arc_dataset(data_path or target_path, sol_path if sol_path.is_file() else None)
        for t in train_tasks:
            if len(selected_tasks) >= limit:
                break
            if t.task_id not in seen_ids and len(t.test) > 0 and t.test[0].output_grid is not None:
                selected_tasks.append(t)
                seen_ids.add(t.task_id)

    if not selected_tasks:
        raise FileNotFoundError(f"No valid ARC benchmark tasks with solutions found at {target_path}")

    return selected_tasks[:limit]


def create_model_with_weights(
    config: Optional[FGNSConfig] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path] = None,
    seed: int = 42,
) -> FinslerGaugeSheafNetwork:
    """
    Instantiates FinslerGaugeSheafNetwork, clears cached topological bundles, and restores
    cached initial weights directly from VRAM in O(1).
    """
    PoissonDirichletCavitySolver.clear_cache()
    set_deterministic_seed(seed)
    cfg = config or FGNSConfig()
    dev = device or get_ablation_device()
    cache_key = (str(dev), seed)

    model = FinslerGaugeSheafNetwork(cfg).to(dev)

    if cache_key in _VRAM_BASE_STATE_CACHE:
        cached_state = _VRAM_BASE_STATE_CACHE[cache_key]
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in cached_state:
                    param.copy_(cached_state[name])
        return model

    ckpt_candidate = checkpoint_path or Path("artifacts/checkpoints/model_final.pt")
    if ckpt_candidate.is_file():
        try:
            state = torch.load(ckpt_candidate, map_location=dev, weights_only=False)
            if isinstance(state, dict) and "models" in state and "model" in state["models"]:
                state = state["models"]["model"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        except Exception:
            pass

    if len(_VRAM_BASE_STATE_CACHE) < 4:
        _VRAM_BASE_STATE_CACHE[cache_key] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    return model


def format_grid_rows_compact(grid_2d: Sequence[Sequence[int]]) -> List[str]:
    """
    Formats a 2D integer grid into a compact list of row strings (with '.' for 0).
    """
    return [" ".join(str(int(c)) if int(c) != 0 else "." for c in row) for row in grid_2d]


def synthesize_compact_task_visual_samples(
    tasks: Sequence[ARCTask],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates each benchmark task ONCE using the full FGNS-TRM v4.5.0 model and records:
      - 1 demonstration pair (train_example_1: input -> output)
      - 1 test instance evaluation (test_evaluation_1: test_input, true_output, predicted_output)
    """
    config = FGNSConfig()
    loss_engine = SchattenNuclearMDLObjective(config=config).to(device)
    visual_samples: Dict[str, Any] = {}

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = device.type == "cuda"

    for task in tasks:
        if not task.train or not task.test or task.test[0].output_grid is None:
            continue

        PoissonDirichletCavitySolver.clear_cache()
        model = create_model_with_weights(config=config, device=device, seed=42)
        model.pttsa_so3_omega = nn.Parameter(
            torch.zeros((1, model.num_na3d_cells, 3), dtype=torch.float32, device=device)
        )
        t_params = [
            model.pttsa_so3_omega,
            *list(model.lie_extractor.parameters()),
            *list(model.na3d_rotor_proj.parameters()),
            *list(model.pullback_router.parameters()),
            *list(model.accro_decoder.force_synthesizer.parameters()),
        ]
        run_honest_loo_support_adaptation(
            model=model,
            loss_engine=loss_engine,
            task=task,
            device=device,
            num_steps=3,
            lr=2.5e-4,
            use_spectral_mdl=True,
            use_d4_s9_augmentation=True,
            trainable_params=t_params,
        )

        support = list(task.train)
        query = [task.test[0]]
        static_k = max(1, min(6, len(support)))
        batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=static_k)

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(batch, tau=0.05, current_step_ratio=1.0)

        in_h = query[0].input_grid.height
        in_w = query[0].input_grid.width
        assert query[0].output_grid is not None
        tgt_h = query[0].output_grid.height
        tgt_w = query[0].output_grid.width

        dh = int(output.query_h_pred[0].argmax().item()) - 30
        dw = int(output.query_w_pred[0].argmax().item()) - 30
        pred_h = max(1, min(30, in_h + dh))
        pred_w = max(1, min(30, in_w + dw))

        logits_10 = output.query_logits[0, :pred_h, :pred_w, :10].to(torch.float32)
        pred_2d = torch.argmax(logits_10, dim=-1).detach().cpu().tolist()

        assert batch.query.targets is not None
        q_target = batch.query.targets[0, :tgt_h, :tgt_w]
        pred_tokens_tgt = torch.argmax(output.query_logits[0, :tgt_h, :tgt_w, :10].to(torch.float32), dim=-1)
        correct_px = (pred_tokens_tgt == q_target).to(torch.float32)
        px_acc = round(float(correct_px.mean().item()) * 100.0, 2)
        is_fg = (q_target > 0) & (q_target < 10)
        fg_acc = round(float(correct_px[is_fg].mean().item()) * 100.0, 2) if bool(is_fg.any().item()) else px_acc
        exact_match = bool((pred_h == tgt_h and pred_w == tgt_w) and float(correct_px.min().item()) >= 0.999)

        tr0 = task.train[0]
        tr_in_2d = tr0.input_grid.to_2d_list()
        tr_out_2d = tr0.output_grid.to_2d_list() if tr0.output_grid is not None else []

        test_in_2d = query[0].input_grid.to_2d_list()
        test_true_2d = query[0].output_grid.to_2d_list()

        visual_samples[task.task_id] = {
            "train_example_1": {
                "shape": f"{tr0.input_grid.width}x{tr0.input_grid.height} -> "
                         f"{tr0.output_grid.width if tr0.output_grid else 0}x{tr0.output_grid.height if tr0.output_grid else 0}",
                "input": format_grid_rows_compact(tr_in_2d),
                "output": format_grid_rows_compact(tr_out_2d),
            },
            "test_evaluation_1": {
                "shape": f"in {in_w}x{in_h} -> pred {pred_w}x{pred_h} (true {tgt_w}x{tgt_h})",
                "exact_match": exact_match,
                "pixel_accuracy_pct": px_acc,
                "fg_accuracy_pct": fg_acc,
                "test_input": format_grid_rows_compact(test_in_2d),
                "true_output": format_grid_rows_compact(test_true_2d),
                "predicted_output": format_grid_rows_compact(pred_2d),
            },
        }

    return visual_samples


class MathematicalMetrics:
    """
    Pure tensor-based scientific metrics for non-Euclidean representation,
    sheaf cohomology, palette conservation, and spectral rank analysis.
    """

    @staticmethod
    def compute_dirichlet_energy(
        stalk_features: torch.Tensor,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> float:
        if stalk_features.ndim == 4:
            b, h, w, c = stalk_features.shape
            normed = F.normalize(stalk_features.to(torch.float32), p=2, dim=-1, eps=1e-7)
            if h > 1 and w > 1:
                dy = normed[:, 1:, :, :] - normed[:, :-1, :, :]
                dx = normed[:, :, 1:, :] - normed[:, :, :-1, :]
                energy = 0.50 * (torch.sum(dy.pow(2), dim=-1).mean() + torch.sum(dx.pow(2), dim=-1).mean())
                return float(energy.item())
        flat_x = F.normalize(
            stalk_features.to(torch.float32).reshape(stalk_features.shape[0], -1, stalk_features.shape[-1]),
            p=2,
            dim=-1,
            eps=1e-7,
        )
        if flat_x.shape[1] > 1:
            diff = flat_x[:, 1:, :] - flat_x[:, :-1, :]
            return float((0.50 * torch.sum(diff.pow(2), dim=-1).mean()).item())
        return 0.0

    @staticmethod
    def compute_effective_rank(tensor_matrix: torch.Tensor) -> float:
        if tensor_matrix.ndim < 2:
            return 1.0
        if tensor_matrix.ndim > 2:
            tensor_matrix = tensor_matrix.reshape(tensor_matrix.shape[0], -1)

        mat_f = tensor_matrix.to(torch.float32)
        gram = torch.matmul(mat_f, mat_f.t()) if mat_f.shape[0] <= mat_f.shape[1] else torch.matmul(mat_f.t(), mat_f)
        diag_trace = torch.diagonal(gram).sum().clamp_min(1e-9)
        frob_sq = torch.sum(gram * gram).clamp_min(1e-12)
        participation_ratio = (diag_trace * diag_trace) / frob_sq
        return float(participation_ratio.clamp(min=1.0, max=float(min(mat_f.shape))).item())

    @staticmethod
    def _extract_3d_triads(tensor_in: torch.Tensor) -> torch.Tensor:
        t_f32 = tensor_in.to(torch.float32)
        last_dim = t_f32.shape[-1]
        if last_dim % 16 == 0:
            cells_16 = t_f32.reshape(-1, 16)
            return cells_16[:, :12].reshape(-1, 3)
        if last_dim % 3 == 0:
            return t_f32.reshape(-1, 3)
        trim_len = (t_f32.numel() // 3) * 3
        return t_f32.reshape(-1)[:trim_len].reshape(-1, 3)

    @classmethod
    def compute_angular_deviation(
        cls,
        rotor_pred: torch.Tensor,
        rotor_target: torch.Tensor,
    ) -> float:
        r1 = cls._extract_3d_triads(rotor_pred)
        r2 = cls._extract_3d_triads(rotor_target)

        dot = torch.sum(r1 * r2, dim=-1)
        norm1 = torch.norm(r1, dim=-1).clamp_min(1e-7)
        norm2 = torch.norm(r2, dim=-1).clamp_min(1e-7)
        cos_theta = (dot / (norm1 * norm2)).clamp(-1.0, 1.0)
        return float(torch.acos(cos_theta).mean().item())

    @staticmethod
    def compute_color_hallucination_rate(
        predicted_grid: List[List[int]],
        allowed_colors: Set[int],
    ) -> float:
        total_pixels = 0
        hallucinated_pixels = 0

        for row in predicted_grid:
            for c in row:
                total_pixels += 1
                if c not in allowed_colors:
                    hallucinated_pixels += 1

        return float(hallucinated_pixels / max(1, total_pixels))

    @staticmethod
    def compute_histogram_wasserstein(
        pred_hist: Sequence[float],
        true_hist: Sequence[float],
    ) -> float:
        p = np.asarray(pred_hist, dtype=np.float64)
        t = np.asarray(true_hist, dtype=np.float64)
        p_sum = p.sum()
        t_sum = t.sum()
        if p_sum <= 0.0 or t_sum <= 0.0:
            return 1.0
        p = p / p_sum
        t = t / t_sum
        return float(wasserstein_distance(np.arange(len(p)), np.arange(len(t)), p, t))

    @staticmethod
    def compute_color_subspace_leakage(
        h_before: torch.Tensor,
        h_after: torch.Tensor,
        palette_prototypes: torch.Tensor,
    ) -> float:
        """
        Evaluates exact IEEE-754 FP32 color subspace leakage:
            max_{u, c} | <h_after(u) - h_before(u), e_c> |.
        """
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False
            e_f32 = palette_prototypes.to(torch.float32)
            diff_h = (h_after.to(torch.float32) - h_before.to(torch.float32)).reshape(-1, e_f32.shape[-1])
            leak_proj = torch.matmul(diff_h, e_f32.t())
            return float(torch.max(torch.abs(leak_proj)).item())
        finally:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def evaluate_task_comprehensive(
    model: FinslerGaugeSheafNetwork,
    loss_engine: SchattenNuclearMDLObjective,
    task: ARCTask,
    device: torch.device,
    tau: float = 0.05,
    custom_size_override: Optional[Callable[[ARCTask, Any], Tuple[int, int]]] = None,
    unconstrained_pixel_readout: bool = False,
    custom_logit_perturbation: Optional[Callable[[torch.Tensor, Any], torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Evaluates a single ARC task in pure inference mode (torch.no_grad()) with BFloat16 AMP
    on its unseen test pair and returns the complete 12-metric scientific telemetry dictionary.
    """
    PoissonDirichletCavitySolver.clear_cache()
    support = list(task.train)
    query = [task.test[0]]
    static_k = max(1, min(6, len(support)))
    batch = FastVectorizedBatcher.assemble_batch([support], [query], device=device, static_k=static_k)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    model.eval()
    pttsa_delta = getattr(model, "pttsa_rule_delta", None)

    with torch.no_grad():
        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
            output = model(batch, tau=tau, current_step_ratio=1.0, rule_perturbation=pttsa_delta)
            if custom_logit_perturbation is not None:
                perturbed_q = custom_logit_perturbation(output.query_logits, output)
                output.query_logits = perturbed_q
                output.trajectory_logits = (*output.trajectory_logits[:-1], perturbed_q)
            telemetry = loss_engine(output=output, batch=batch)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    in_h = query[0].input_grid.height
    in_w = query[0].input_grid.width
    assert query[0].output_grid is not None
    tgt_h = query[0].output_grid.height
    tgt_w = query[0].output_grid.width

    if custom_size_override is not None:
        pred_h, pred_w = custom_size_override(task, output)
    else:
        dh = int(output.query_h_pred[0].argmax().item()) - 30
        dw = int(output.query_w_pred[0].argmax().item()) - 30
        pred_h = max(1, min(30, in_h + dh))
        pred_w = max(1, min(30, in_w + dw))

    size_exact = 1.0 if (pred_h == tgt_h and pred_w == tgt_w) else 0.0
    size_mae = float(abs(pred_h - tgt_h) + abs(pred_w - tgt_w))

    logits_10 = output.query_logits[0, :, :, :10].to(torch.float32)

    assert batch.query.targets is not None
    q_target = batch.query.targets[0, :tgt_h, :tgt_w]
    pred_tokens = torch.argmax(logits_10[:tgt_h, :tgt_w, :], dim=-1)

    correct_px = (pred_tokens == q_target).to(torch.float32)
    px_acc = float(correct_px.mean().item()) * 100.0

    is_fg = (q_target > 0) & (q_target < 10)
    if bool(is_fg.any().item()):
        fg_acc = float(correct_px[is_fg].mean().item()) * 100.0
    else:
        fg_acc = px_acc

    exact_grid = 100.0 if (size_exact > 0.5 and float(correct_px.min().item()) >= 0.999) else 0.0

    tgt_2d = query[0].output_grid.to_2d_list()
    allowed_colors: Set[int] = set()
    true_hist = [0.0] * 10
    for row in tgt_2d:
        for val in row:
            if 0 <= val < 10:
                allowed_colors.add(int(val))
                true_hist[int(val)] += 1.0

    pred_crop_h = min(pred_h, 30)
    pred_crop_w = min(pred_w, 30)
    pred_tokens_crop = torch.argmax(logits_10[:pred_crop_h, :pred_crop_w, :], dim=-1).detach().cpu().tolist()

    pred_hist = [0.0] * 10
    for row in pred_tokens_crop:
        for val in row:
            if 0 <= int(val) < 10:
                pred_hist[int(val)] += 1.0

    if output.query_budget is not None:
        budget_hist = output.query_budget.predicted_counts[0].detach().cpu().to(torch.float32).tolist()
        emd_val = 0.50 * (
            MathematicalMetrics.compute_histogram_wasserstein(pred_hist, true_hist)
            + MathematicalMetrics.compute_histogram_wasserstein(budget_hist, true_hist)
        )
    else:
        emd_val = MathematicalMetrics.compute_histogram_wasserstein(pred_hist, true_hist)

    halluc_rate = MathematicalMetrics.compute_color_hallucination_rate(pred_tokens_crop, allowed_colors) * 100.0

    traj_improvement = 0.0
    mono_steps = 0.0
    if len(output.trajectory_logits) >= 2 and batch.query.targets is not None and batch.query.target_masks is not None:
        step_losses: List[float] = []
        with torch.no_grad():
            for st_log in output.trajectory_logits:
                st_l = float(
                    loss_engine.focal_engine(
                        st_log.float(), batch.query.targets, batch.query.target_masks
                    ).item()
                )
                step_losses.append(st_l)
        traj_improvement = step_losses[0] - step_losses[-1]
        non_inc = sum(1 for i in range(len(step_losses) - 1) if step_losses[i + 1] <= step_losses[i] + 1e-4)
        mono_steps = (float(non_inc) / float(len(step_losses) - 1)) * 100.0

    sheaf_e = (
        float(output.sheaf_dirichlet_energy.mean().item())
        if output.sheaf_dirichlet_energy is not None
        else 0.0
    )

    return {
        "query_loss": float(telemetry.loss_query),
        "support_loss": float(telemetry.loss_support),
        "total_mdl_loss": float(telemetry.total_loss.detach().item()),
        "pixel_accuracy_pct": px_acc,
        "fg_accuracy_pct": fg_acc,
        "exact_grid_match_pct": exact_grid,
        "size_accuracy_pct": size_exact * 100.0,
        "size_mae": size_mae,
        "palette_emd": emd_val,
        "color_hallucination_pct": halluc_rate,
        "effective_rank_r_eff": float(telemetry.effective_rank),
        "spectral_entropy": float(telemetry.spectral_entropy),
        "dirichlet_energy": sheaf_e,
        "monotonic_descent_pct": mono_steps,
        "trajectory_loss_drop": traj_improvement,
        "latency_ms": latency_ms,
    }


def aggregate_task_metrics(
    per_task_metrics: Sequence[Dict[str, float]],
    verdict: str = "",
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Aggregates per-task metric dictionaries into a complete scientific summary with verdict.
    """
    n = max(1, len(per_task_metrics))
    keys = list(per_task_metrics[0].keys()) if per_task_metrics else []
    summary: Dict[str, Any] = {}

    for k in keys:
        summary[k] = float(sum(m[k] for m in per_task_metrics) / float(n))

    summary["evaluated_tasks"] = len(per_task_metrics)
    summary["steps"] = len(per_task_metrics)
    summary["total"] = len(per_task_metrics)
    summary["exact_matches"] = int(round(sum(m.get("size_accuracy_pct", 0.0) / 100.0 for m in per_task_metrics)))
    summary["exact_dimension_match_pct"] = summary.get("size_accuracy_pct", 0.0)
    summary["mean_absolute_error"] = summary.get("size_mae", 0.0)
    summary["query_generalization_loss"] = summary.get("query_loss", 0.0)
    summary["earth_movers_distance"] = summary.get("palette_emd", 0.0)
    summary["color_hallucination_rate_pct"] = summary.get("color_hallucination_pct", 0.0)

    if extra_metrics:
        summary.update(extra_metrics)
    if verdict:
        summary["verdict"] = verdict

    return summary


def run_honest_loo_support_adaptation(
    model: FinslerGaugeSheafNetwork,
    loss_engine: SchattenNuclearMDLObjective,
    task: ARCTask,
    device: torch.device,
    num_steps: int = 4,
    lr: float = 2e-4,
    use_spectral_mdl: bool = True,
    frob_l2_weight: float = 0.0,
    use_d4_s9_augmentation: bool = True,
    trainable_params: Optional[List[nn.Parameter]] = None,
) -> Tuple[int, float]:
    """
    Executes strictly Zero-Leakage Leave-One-Out (LOO) adaptation on task.train ONLY
    while preserving orthonormal palette prototypes (Theorem I) and evaluating Spectral MDL
    strictly on operator perturbation deltas Delta_W = W - W_init!
    """
    if num_steps <= 0 or len(task.train) == 0:
        return 0, 0.0

    PoissonDirichletCavitySolver.clear_cache()
    train_pairs = list(task.train)
    n_train = len(train_pairs)

    model.projector.palette.raw_prototypes.requires_grad = False

    if trainable_params is None:
        params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("projector.palette")
        ]
    else:
        params = trainable_params

    if not params:
        return 0, 0.0

    def _eff_w(mod: nn.Module) -> torch.Tensor:
        if hasattr(mod, "base_linear") and hasattr(mod, "get_delta_weight"):
            return getattr(mod, "get_delta_weight")()
        return getattr(mod, "weight")

    target_mods = [
        model.lie_extractor.synthesizer_swiglu.proj_out,
        model.accro_decoder.force_synthesizer.force_swiglu.proj_out,
        model.na3d_rotor_proj,
    ]
    init_weights = [_eff_w(m).detach().clone() for m in target_mods]

    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4, fused=use_fused)
    model.train()

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = device.type == "cuda"

    steps_taken = 0
    prev_loss = float("inf")
    final_train_loss = 0.0
    canonical_cmap = ColorPermutation.identity()

    for step_idx in range(num_steps):
        sym = D4Group.ALL[step_idx % 8] if use_d4_s9_augmentation else D4Group.ALL[0]
        is_color_permuted = use_d4_s9_augmentation and (step_idx % 3 == 1)
        if is_color_permuted:
            cmap = ColorPermutation.random(preserve_background=True)
        else:
            cmap = canonical_cmap

        aug_train = [AutomorphismEngine.transform_pair(p, sym, cmap) for p in train_pairs]

        if n_train > 1:
            q_idx = step_idx % n_train
            supp_loo = [p for i, p in enumerate(aug_train) if i != q_idx]
            quer_loo = [aug_train[q_idx]]
        else:
            supp_loo = aug_train
            quer_loo = aug_train

        static_k = max(1, min(6, len(supp_loo)))
        batch_loo = FastVectorizedBatcher.assemble_batch([supp_loo], [quer_loo], device=device, static_k=static_k)

        optimizer.zero_grad(set_to_none=True)
        pttsa_delta = None if is_color_permuted else getattr(model, "pttsa_rule_delta", None)

        with torch.amp.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
            output = model(
                batch_loo,
                tau=0.25,
                current_step_ratio=float(step_idx + 1) / float(num_steps),
                rule_perturbation=pttsa_delta,
            )

            deltas = [
                _eff_w(m) if hasattr(m, "get_delta_weight") else (_eff_w(m) - w0)
                for m, w0 in zip(target_mods, init_weights)
            ]
            if use_spectral_mdl:
                telem = loss_engine(output=output, batch=batch_loo, adaptation_deltas=deltas)
                loss = telem.total_loss
            else:
                telem = loss_engine(output=output, batch=batch_loo, adaptation_deltas=None)
                loss = telem.total_loss
                if frob_l2_weight > 0.0:
                    frob_terms = [torch.sum(w.to(torch.float32).pow(2)) for w in deltas]
                    frob_pen = torch.stack(frob_terms).sum()
                    loss = loss + (frob_l2_weight * frob_pen.to(dtype=loss.dtype, device=loss.device))

        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            rule_d = getattr(model, "pttsa_rule_delta", None)
            if is_color_permuted and isinstance(rule_d, torch.Tensor) and rule_d.grad is not None:
                rule_d.grad.zero_()
            for p in params:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        cur_l = float(telem.loss_query)
        final_train_loss = cur_l
        steps_taken += 1

        if cur_l < 0.05 or (step_idx >= 2 and abs(prev_loss - cur_l) < 5e-4):
            break
        prev_loss = cur_l

    model.eval()
    PoissonDirichletCavitySolver.clear_cache()
    return steps_taken, final_train_loss