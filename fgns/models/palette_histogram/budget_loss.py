"""
Location: // fgns_engine/fgns/models/palette_budget/budget_loss.py
Description: SOTA Non-Saturating Laplace-Smoothed Palette Budget Loss Engine (Query + Support Supervision).
             Core Mathematical Supervisor for FGNS v1.0.0 (Sovereign):
               1. Capacity-Normalized Integer Multiset Loss: Divides Smooth-L1 count discrepancies
                  by target area capacity K_target.
               2. Bin-Normalized 1D Wasserstein Earth Mover's Distance (EMD) in [0.0, 1.0].
               3. Laplace-Smoothed Simplex Kullback-Leibler Divergence: replaces hard clamp(min=1e-7)
                  with convex mixing (1 - eps)*p + eps/10, guaranteeing non-zero gradients everywhere!
               4. Scale-calibrated contrastive anti-collapse barrier protecting colored foreground entities.
               5. Joint Query + Masked Support Supervision.
Language: Python 3.12+ / PyTorch 2.5+ / Optimal Transport sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Final, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.safe_math import (
    safe_divide,
    safe_log,
)
from fgns.types import PaletteBudgetPrediction

DEFAULT_ANTI_COLLAPSE_MARGIN: Final[float] = 0.50
DEFAULT_ANTI_COLLAPSE_WEIGHT: Final[float] = 0.25
DEFAULT_WASSERSTEIN_WEIGHT: Final[float] = 0.50
DEFAULT_KL_SIMPLEX_WEIGHT: Final[float] = 0.50
DEFAULT_SMOOTH_L1_BETA: Final[float] = 1.00
DEFAULT_BUDGET_LOSS_EPS: Final[float] = 1e-4


def _evaluate_single_stream_budget_loss(
    pred_counts: torch.Tensor,
    pred_simplex: torch.Tensor,
    in_counts: torch.Tensor,
    true_counts: torch.Tensor,
    true_capacity: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    anti_collapse_margin: float,
    anti_collapse_weight: float,
    wasserstein_weight: float,
    kl_weight: float,
    smooth_l1_beta: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Internal vectorized evaluator for a batch of budget predictions [N, 10] with optional validity mask [N].
    Uses Laplace convex smoothing to guarantee non-vanishing KL and Wasserstein gradients.
    """
    N = true_counts.shape[0]
    cap_f32 = true_capacity.float().view(N, 1).clamp_min(1.0)
    true_f32 = true_counts.float()
    in_f32 = in_counts.float()

    raw_true_simplex = true_f32 / cap_f32
    true_sum = raw_true_simplex.sum(dim=-1, keepdim=True)
    has_true_mass = (true_sum.squeeze(-1) > 0.01)

    if valid_mask is not None:
        w_valid = (valid_mask & has_true_mass).float().view(N)
    else:
        w_valid = has_true_mass.float().view(N)

    valid_denom = w_valid.sum().clamp_min(1.0)
    true_simplex = raw_true_simplex / true_sum.clamp_min(1e-6)

    """1. Capacity-Normalized Smooth-L1 Count Discrepancy"""
    raw_count_error = F.smooth_l1_loss(
        pred_counts.float(),
        true_f32,
        beta=smooth_l1_beta,
        reduction="none",
    )
    norm_err_per_item = (raw_count_error.sum(dim=-1) / cap_f32.squeeze(-1)) * w_valid
    loss_smooth_l1 = norm_err_per_item.sum() / valid_denom

    """2. Bin-Normalized 1D Wasserstein Distance on Cumulative Simplex in [0, 1]"""
    pred_cdf = torch.cumsum(pred_simplex.float(), dim=-1)
    true_cdf = torch.cumsum(true_simplex, dim=-1)
    w1_per_item = (torch.abs(pred_cdf - true_cdf).mean(dim=-1)) * w_valid
    loss_wasserstein = w1_per_item.sum() / valid_denom

    """
    3. Laplace-Smoothed Simplex Kullback-Leibler Divergence:
    Convex mixture p_tilde = (1 - eps) * p + eps / 10 preserves exact derivative d(p_tilde)/dp = 1 - eps > 0!
    """
    laplace_eps = max(float(eps), 1e-4)
    safe_p = (1.0 - laplace_eps) * pred_simplex.float() + (laplace_eps * 0.10)
    safe_q = (1.0 - laplace_eps) * true_simplex + (laplace_eps * 0.10)
    kl_per_item = torch.sum(safe_q * (torch.log(safe_q) - torch.log(safe_p)), dim=-1).clamp_min(0.0) * w_valid
    loss_kl = kl_per_item.sum() / valid_denom

    """4. Contrastive Anti-Collapse Barrier"""
    k_in = torch.sum(in_f32, dim=-1, keepdim=True).clamp_min(1.0)
    fg_in_ratio = (torch.sum(in_f32[:, 1:], dim=-1, keepdim=True) / k_in).squeeze(-1)
    p0_pred = pred_simplex[:, 0].float()
    allowed_bg_threshold = (1.0 - (anti_collapse_margin * fg_in_ratio)).clamp(0.15, 0.99)
    collapse_violation = (F.relu(p0_pred - allowed_bg_threshold).pow(2)) * w_valid
    loss_anti_col = collapse_violation.sum() / valid_denom

    total_stream = (
        loss_smooth_l1
        + (wasserstein_weight * loss_wasserstein)
        + (kl_weight * loss_kl)
        + (anti_collapse_weight * loss_anti_col)
    )
    return total_stream, loss_smooth_l1, loss_wasserstein, loss_anti_col


def evaluate_budget_loss(
    prediction: PaletteBudgetPrediction,
    target_histograms: torch.Tensor,
    target_capacities: torch.Tensor,
    anti_collapse_margin: float = DEFAULT_ANTI_COLLAPSE_MARGIN,
    anti_collapse_weight: float = DEFAULT_ANTI_COLLAPSE_WEIGHT,
    wasserstein_weight: float = DEFAULT_WASSERSTEIN_WEIGHT,
    kl_weight: float = DEFAULT_KL_SIMPLEX_WEIGHT,
    smooth_l1_beta: float = DEFAULT_SMOOTH_L1_BETA,
    support_predictions: Optional[Sequence[PaletteBudgetPrediction]] = None,
    support_target_histograms: Optional[torch.Tensor] = None,
    support_target_capacities: Optional[torch.Tensor] = None,
    support_pair_valid_mask: Optional[torch.Tensor] = None,
    eps: float = DEFAULT_BUDGET_LOSS_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Evaluates scale-normalized quadripartite loss for the Neural Palette Capacity Oracle
    across both the Query target and all valid Support demonstrations.
    """
    (
        q_total,
        q_l1,
        q_w1,
        q_anti,
    ) = _evaluate_single_stream_budget_loss(
        pred_counts=prediction.predicted_counts,
        pred_simplex=prediction.simplex_proportions,
        in_counts=prediction.input_counts,
        true_counts=target_histograms,
        true_capacity=target_capacities,
        valid_mask=None,
        anti_collapse_margin=anti_collapse_margin,
        anti_collapse_weight=anti_collapse_weight,
        wasserstein_weight=wasserstein_weight,
        kl_weight=kl_weight,
        smooth_l1_beta=smooth_l1_beta,
        eps=eps,
    )

    if (
        support_predictions is not None
        and len(support_predictions) > 0
        and support_target_histograms is not None
        and support_target_capacities is not None
    ):
        supp_pred = support_predictions[0]
        s_hist_flat = support_target_histograms.view(-1, 10)
        s_cap_flat = support_target_capacities.view(-1)
        s_valid_flat = support_pair_valid_mask.view(-1) if support_pair_valid_mask is not None else None

        if supp_pred.predicted_counts.shape[0] == s_hist_flat.shape[0]:
            (
                s_total,
                s_l1,
                s_w1,
                s_anti,
            ) = _evaluate_single_stream_budget_loss(
                pred_counts=supp_pred.predicted_counts,
                pred_simplex=supp_pred.simplex_proportions,
                in_counts=supp_pred.input_counts,
                true_counts=s_hist_flat,
                true_capacity=s_cap_flat,
                valid_mask=s_valid_flat,
                anti_collapse_margin=anti_collapse_margin,
                anti_collapse_weight=anti_collapse_weight,
                wasserstein_weight=wasserstein_weight,
                kl_weight=kl_weight,
                smooth_l1_beta=smooth_l1_beta,
                eps=eps,
            )
            total_budget_loss = 0.50 * (q_total + s_total)
            loss_smooth_l1 = 0.50 * (q_l1 + s_l1)
            wasserstein_distance = 0.50 * (q_w1 + s_w1)
            loss_anti_col = 0.50 * (q_anti + s_anti)
        else:
            total_budget_loss = q_total
            loss_smooth_l1 = q_l1
            wasserstein_distance = q_w1
            loss_anti_col = q_anti
    else:
        total_budget_loss = q_total
        loss_smooth_l1 = q_l1
        wasserstein_distance = q_w1
        loss_anti_col = q_anti

    B = target_histograms.shape[0]
    true_cap_q = target_capacities.float().view(B, 1).clamp_min(1.0)
    true_simplex_q = target_histograms.float() / true_cap_q

    telemetry: Dict[str, float] = {
        "budget_loss_total": float(total_budget_loss.detach().item()),
        "budget_loss_smooth_l1": float(loss_smooth_l1.detach().item()),
        "budget_loss_wasserstein": float(wasserstein_distance.detach().item()),
        "budget_loss_anti_collapse": float(loss_anti_col.detach().item()),
        "mean_pred_background_prob": float(torch.mean(prediction.simplex_proportions[:, 0]).detach().item()),
        "mean_true_background_prob": float(torch.mean(true_simplex_q[:, 0]).detach().item()),
    }

    return total_budget_loss, loss_smooth_l1, wasserstein_distance, loss_anti_col, telemetry


class PaletteBudgetLossEngine(nn.Module):
    """
    SOTA PyTorch Module Container for the Laplace-Smoothed Palette Budget Loss.
    """

    def __init__(
        self,
        anti_collapse_margin: float = DEFAULT_ANTI_COLLAPSE_MARGIN,
        anti_collapse_weight: float = DEFAULT_ANTI_COLLAPSE_WEIGHT,
        wasserstein_weight: float = DEFAULT_WASSERSTEIN_WEIGHT,
        kl_weight: float = DEFAULT_KL_SIMPLEX_WEIGHT,
        smooth_l1_beta: float = DEFAULT_SMOOTH_L1_BETA,
        eps: float = DEFAULT_BUDGET_LOSS_EPS,
    ) -> None:
        super().__init__()
        self.anti_collapse_margin = float(anti_collapse_margin)
        self.anti_collapse_weight = float(anti_collapse_weight)
        self.wasserstein_weight = float(wasserstein_weight)
        self.kl_weight = float(kl_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.eps = eps

    def forward(
        self,
        prediction: PaletteBudgetPrediction,
        target_histograms: torch.Tensor,
        target_capacities: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Executes forward loss evaluation and returns composite scalar with telemetry.
        """
        (
            total_loss,
            _,
            _,
            _,
            telemetry,
        ) = evaluate_budget_loss(
            prediction=prediction,
            target_histograms=target_histograms,
            target_capacities=target_capacities,
            anti_collapse_margin=self.anti_collapse_margin,
            anti_collapse_weight=self.anti_collapse_weight,
            wasserstein_weight=self.wasserstein_weight,
            kl_weight=self.kl_weight,
            smooth_l1_beta=self.smooth_l1_beta,
            eps=self.eps,
        )
        return total_loss, telemetry


def verify_budget_loss_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for budget_loss.py.
    """
    torch.manual_seed(42)
    b_size = 2

    loss_engine = PaletteBudgetLossEngine()

    true_counts = torch.tensor(
        [[20.0, 5.0, 5.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
         [4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    target_caps = torch.tensor([36.0, 9.0], dtype=torch.float32)
    in_counts = true_counts.clone().to(torch.int64)
    perfect_simplex = true_counts / target_caps.view(b_size, 1)

    pred_perfect = PaletteBudgetPrediction(
        predicted_counts=true_counts.clone(),
        simplex_proportions=perfect_simplex.clone(),
        input_counts=in_counts,
        target_capacity=target_caps.to(torch.int64),
    )

    loss_perf, _ = loss_engine(pred_perfect, true_counts.to(torch.int64), target_caps)
    if float(loss_perf.item()) > 1e-3:
        return False

    var_counts = true_counts.clone().requires_grad_(True)
    var_simplex = perfect_simplex.clone().requires_grad_(True)

    pred_grad_test = PaletteBudgetPrediction(
        predicted_counts=var_counts,
        simplex_proportions=var_simplex,
        input_counts=in_counts,
        target_capacity=target_caps.to(torch.int64),
    )

    loss_grad, _ = loss_engine(pred_grad_test, true_counts.to(torch.int64), target_caps)
    loss_grad.backward()

    if var_counts.grad is None or var_simplex.grad is None:
        return False
    if torch.isnan(var_counts.grad).any() or torch.isnan(var_simplex.grad).any():
        return False

    return True