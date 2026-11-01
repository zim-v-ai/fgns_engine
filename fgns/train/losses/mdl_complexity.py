"""
Location: // fgns_engine/fgns/train/losses/mdl_complexity.py
Description: SOTA Pure Kolmogorov-Solomonoff Program Complexity Loss via Schatten-1 Nuclear Norm & Von Neumann Entropy.
             Core Regularization Engine for FGNS v1.2.0 (Finsler-Gauge Sheaf):
               1. Pure Algorithmic Minimum Description Length (MDL) regularization in operator space.
               2. Zero reference-weight anchoring (DEFAULT_LAMBDA_FROBENIUS = 0.0): allows the network
                  to freely discover novel task attractors while purging noise via spectral rank compression!
               3. SVD-Free Schatten-1 Nuclear Norm via regularized Gram matrix trace spectrum.
               4. Von Neumann Spectral Entropy H_vN and continuous Effective Rank R_eff evaluation.
Invariants: Strict non-negativity L_MDL >= 0, zero NaN generation on null matrices,
            C-infinity autograd smoothness across all singular modes, zero hash comment compliance.
Language: Python 3.12+ / PyTorch 2.5+ / Information Theory sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.schatten_norm import (
    schatten_gram_trace,
    von_neumann_spectral_entropy,
)

DEFAULT_LAMBDA_NUCLEAR: Final[float] = 1e-4
DEFAULT_LAMBDA_ENTROPY: Final[float] = 1e-3
DEFAULT_LAMBDA_FROBENIUS: Final[float] = 0.00
DEFAULT_SCHATTEN_P: Final[float] = 1.00
DEFAULT_MDL_EPS: Final[float] = 1e-6


class SpectralProgramComplexity(nn.Module):
    """
    SOTA Pure Kolmogorov-Solomonoff Algorithmic Program Complexity Loss Engine.
    Penalizes matrix rank via Schatten-1 Nuclear Norm and penalizes energy dispersion
    via Von Neumann Spectral Entropy, forcing the model to discover minimal monolithic programs
    without tethering weights to initial meta-priors.
    """

    def __init__(
        self,
        lambda_nuclear: float = DEFAULT_LAMBDA_NUCLEAR,
        lambda_entropy: float = DEFAULT_LAMBDA_ENTROPY,
        lambda_frobenius: float = DEFAULT_LAMBDA_FROBENIUS,
        schatten_p: float = DEFAULT_SCHATTEN_P,
        eps: float = DEFAULT_MDL_EPS,
    ) -> None:
        super().__init__()
        self.lambda_nuclear = float(lambda_nuclear)
        self.lambda_entropy = float(lambda_entropy)
        self.lambda_frobenius = float(lambda_frobenius)
        self.schatten_p = float(schatten_p)
        self.eps = eps

    def forward(
        self,
        adaptation_weights: Sequence[torch.Tensor],
        return_telemetry: bool = False,
    ) -> Union[Tuple[torch.Tensor, float, float], Tuple[torch.Tensor, float, float, Dict[str, float]]]:
        """
        Evaluates algorithmic program complexity across a sequence of operator matrices.
        """
        if not adaptation_weights:
            zero_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            if return_telemetry:
                empty_telemetry: Dict[str, float] = {
                    "loss_mdl_total": 0.0,
                    "mdl_nuclear_norm": 0.0,
                    "mdl_spectral_entropy": 0.0,
                    "mdl_effective_rank": 1.0,
                    "mdl_frobenius_drift": 0.0,
                    "mdl_active_matrices": 0.0,
                }
                return zero_loss, 0.0, 1.0, empty_telemetry
            return zero_loss, 0.0, 1.0

        device = adaptation_weights[0].device

        nuclear_accum = torch.tensor(0.0, dtype=torch.float32, device=device)
        entropy_accum = torch.tensor(0.0, dtype=torch.float32, device=device)
        frobenius_accum = torch.tensor(0.0, dtype=torch.float32, device=device)
        rank_accum = 0.0
        active_count = 0

        for w_delta in adaptation_weights:
            if w_delta.numel() < 4 or w_delta.ndim < 2:
                continue

            mat_2d = w_delta.reshape(-1, w_delta.shape[-2], w_delta.shape[-1]) if w_delta.ndim > 2 else w_delta

            """1. SVD-Free Schatten-1 Nuclear Norm Evaluation (Sparsity of Singular Values)"""
            schatten_val = schatten_gram_trace(mat_2d, p_order=self.schatten_p, eps=self.eps)
            nuclear_accum = nuclear_accum + torch.mean(schatten_val)

            """2. Von Neumann Spectral Entropy & Continuous Effective Rank Evaluation"""
            entropy_val, eff_rank = von_neumann_spectral_entropy(mat_2d, eps=self.eps)
            entropy_accum = entropy_accum + torch.mean(entropy_val)
            rank_accum += float(torch.mean(eff_rank).detach().item())

            """3. Optional Frobenius Term (0.0 by default in Pure Spectral MDL)"""
            if self.lambda_frobenius > 0.0:
                frob_sq = torch.sum(mat_2d.pow(2), dim=(-2, -1))
                frobenius_accum = frobenius_accum + torch.mean(frob_sq)

            active_count += 1

        if active_count == 0:
            zero_loss = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
            if return_telemetry:
                empty_telemetry = {
                    "loss_mdl_total": 0.0,
                    "mdl_nuclear_norm": 0.0,
                    "mdl_spectral_entropy": 0.0,
                    "mdl_effective_rank": 1.0,
                    "mdl_frobenius_drift": 0.0,
                    "mdl_active_matrices": 0.0,
                }
                return zero_loss, 0.0, 1.0, empty_telemetry
            return zero_loss, 0.0, 1.0

        inv_count = 1.0 / float(active_count)
        avg_nuclear = nuclear_accum * inv_count
        avg_entropy = entropy_accum * inv_count
        avg_frobenius = frobenius_accum * inv_count
        avg_eff_rank = rank_accum * inv_count

        total_mdl_penalty = (
            (self.lambda_nuclear * avg_nuclear)
            + (self.lambda_entropy * avg_entropy)
            + (self.lambda_frobenius * avg_frobenius)
        )

        mean_ent_scalar = float(avg_entropy.detach().item())

        if return_telemetry:
            telemetry: Dict[str, float] = {
                "loss_mdl_total": float(total_mdl_penalty.detach().item()),
                "mdl_nuclear_norm": float(avg_nuclear.detach().item()),
                "mdl_spectral_entropy": mean_ent_scalar,
                "mdl_effective_rank": avg_eff_rank,
                "mdl_frobenius_drift": float(avg_frobenius.detach().item()),
                "mdl_active_matrices": float(active_count),
            }
            return total_mdl_penalty, mean_ent_scalar, avg_eff_rank, telemetry

        return total_mdl_penalty, mean_ent_scalar, avg_eff_rank


SpectralRankMDLObjective = SpectralProgramComplexity


def verify_mdl_complexity_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for mdl_complexity.py.
    """
    torch.manual_seed(42)
    rows, cols, rank = 64, 64, 4

    mdl_engine = SpectralProgramComplexity(
        lambda_nuclear=1e-3,
        lambda_entropy=1e-3,
    )

    lora_b_zero = torch.zeros(rows, rank, dtype=torch.float32, requires_grad=True)
    lora_a_zero = torch.randn(rank, cols, dtype=torch.float32, requires_grad=True)
    delta_w_zero = torch.matmul(lora_b_zero, lora_a_zero)

    loss_zero, _, _ = mdl_engine([delta_w_zero])

    if float(loss_zero.item()) > 1e-5:
        return False
    if torch.isnan(loss_zero) or torch.isinf(loss_zero):
        return False

    loss_zero.backward()
    if lora_b_zero.grad is None or lora_a_zero.grad is None:
        return False
    if torch.isnan(lora_b_zero.grad).any() or torch.isnan(lora_a_zero.grad).any():
        return False

    u_vec = torch.randn(rows, 1, dtype=torch.float32)
    v_vec = torch.randn(1, cols, dtype=torch.float32)
    delta_w_rank1 = torch.matmul(u_vec, v_vec)

    delta_w_noise = torch.randn(rows, cols, dtype=torch.float32)

    _, ent_r1, r_eff_r1 = mdl_engine([delta_w_rank1])
    _, ent_noise, r_eff_noise = mdl_engine([delta_w_noise])

    if r_eff_r1 > 1.35:
        return False
    if ent_r1 >= ent_noise:
        return False
    if r_eff_r1 >= r_eff_noise:
        return False

    _, _, _, telem_dict = mdl_engine([delta_w_rank1], return_telemetry=True)
    if "loss_mdl_total" not in telem_dict or "mdl_effective_rank" not in telem_dict:
        return False

    return True

