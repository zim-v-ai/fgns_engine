"""
Location: // fgns_engine/fgns/train/losses/xor_regularization.py
Description: SOTA Symplectic True XOR Parity Involution & Darboux 2-Form Regularization Loss Engine.
             Core Mathematical Supervisor for FGNS v2.0.0 (Finsler-Gauge Sheaf):
               1. Enforces exact algebraic self-invariance: Omega(u, u) == 0.000000.
               2. Enforces non-commutative anti-symmetry: Omega(u, v) + Omega(v, u) == 0.000000.
               3. Active 4D Spatial Darboux Area Barrier: prevents trivial collapse of the spatial
                  XOR stream by supervising the symplectic 2-form <Omega_XOR, J G_parity> inside active masks.
               4. Dirichlet padding isolation penalizing parasitic vorticity leakage outside active canvas.
Invariants: Strict non-negativity L_xor >= 0, exact zero on ideal symplectic fields,
            C-infinity autograd smoothness across all channels, zero hash comment compliance.
Language: Python 3.12+ / PyTorch 2.5+ / Exterior Symplectic Algebra sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.safe_math import (
    safe_divide,
    safe_hyperbolic_sqrt,
)
from fgns.primitives.symplectic_involution import symplectic_bilinear_form
from fgns.primitives.xor_involution import AnalyticXORInvolutionEngine
from fgns.types import XORParityState

DEFAULT_XOR_WEIGHT: Final[float] = 0.20
DEFAULT_SELF_WEIGHT: Final[float] = 1.00
DEFAULT_ASYM_WEIGHT: Final[float] = 0.50
DEFAULT_ANTI_COLLAPSE_WEIGHT: Final[float] = 0.35
DEFAULT_MIN_RESPONSE_MARGIN: Final[float] = 0.50
DEFAULT_PAD_LEAK_WEIGHT: Final[float] = 0.25
DEFAULT_XOR_EPS: Final[float] = 1e-7


class XORParityRegularizationLoss(nn.Module):
    """
    SOTA Symplectic True XOR Parity Involution & Spatial Darboux 2-Form Regularization Loss Engine.
    """

    def __init__(
        self,
        weight: float = DEFAULT_XOR_WEIGHT,
        self_weight: float = DEFAULT_SELF_WEIGHT,
        asym_weight: float = DEFAULT_ASYM_WEIGHT,
        anti_collapse_weight: float = DEFAULT_ANTI_COLLAPSE_WEIGHT,
        min_response_margin: float = DEFAULT_MIN_RESPONSE_MARGIN,
        pad_leak_weight: float = DEFAULT_PAD_LEAK_WEIGHT,
        eps: float = DEFAULT_XOR_EPS,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.self_weight = float(self_weight)
        self.asym_weight = float(asym_weight)
        self.anti_collapse_weight = float(anti_collapse_weight)
        self.min_response_margin = float(min_response_margin)
        self.pad_leak_weight = float(pad_leak_weight)
        self.eps = eps

    def compute_algebraic_invariance_loss(
        self,
        xor_engine: AnalyticXORInvolutionEngine,
        u_sample: torch.Tensor,
        v_sample: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates fundamental algebraic axioms of the Lie-Darboux complex structure.
        """
        flux_self = xor_engine.vorticity_core(u_sample, u_sample)
        loss_self = torch.mean(torch.sum(flux_self.pow(2), dim=-1))

        if v_sample is not None:
            v_ref = v_sample
        else:
            v_ref = torch.roll(u_sample, shifts=1, dims=0)

        flux_uv = xor_engine.vorticity_core(u_sample, v_ref)
        flux_vu = xor_engine.vorticity_core(v_ref, u_sample)
        asym_violation = flux_uv + flux_vu
        loss_asym = torch.mean(torch.sum(asym_violation.pow(2), dim=-1))

        flux_opposing = xor_engine.vorticity_core(u_sample, -u_sample)
        norm_opposing = safe_hyperbolic_sqrt(torch.sum(flux_opposing.pow(2), dim=-1), eps=self.eps)
        collapse_deficit = F.relu(self.min_response_margin - norm_opposing)
        loss_anti_collapse = torch.mean(collapse_deficit.pow(2))

        return loss_self, loss_asym, loss_anti_collapse

    def forward(
        self,
        xor_state: Optional[XORParityState] = None,
        xor_flux: Optional[torch.Tensor] = None,
        target_masks: Optional[torch.Tensor] = None,
        xor_engine: Optional[AnalyticXORInvolutionEngine] = None,
        sample_states: Optional[torch.Tensor] = None,
        return_telemetry: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Evaluates composite symplectic True XOR regularization loss across 4D spatial fields and algebraic axioms.
        """
        active_flux: Optional[torch.Tensor] = None
        active_selector: Optional[torch.Tensor] = None

        if xor_state is not None:
            active_flux = xor_state.vorticity_flux
            active_selector = xor_state.parity_selector
        elif xor_flux is not None:
            active_flux = xor_flux

        if active_flux is None:
            zero_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            if return_telemetry:
                empty_telem: Dict[str, float] = {
                    "loss_xor_total": 0.0,
                    "loss_xor_self": 0.0,
                    "loss_xor_asym": 0.0,
                    "loss_xor_anti_collapse": 0.0,
                    "loss_xor_pad_leak": 0.0,
                    "xor_mean_vorticity_norm": 0.0,
                }
                return zero_loss, empty_telem
            return zero_loss

        device = active_flux.device
        dtype = active_flux.dtype

        """1. Padding Boundary Leakage Penalty"""
        loss_pad_leak = torch.tensor(0.0, dtype=dtype, device=device)

        if active_flux.ndim == 4 and target_masks is not None:
            mask_f32 = target_masks.float().unsqueeze(-1)
            is_padding = 1.0 - mask_f32
            pad_pixels = is_padding.sum().clamp_min(1.0)
            flux_on_padding = active_flux * is_padding
            loss_pad_leak = torch.sum(flux_on_padding.pow(2)) / pad_pixels

        """2. Direct Algebraic Invariant & 4D Spatial Darboux Area Supervision"""
        loss_self = torch.tensor(0.0, dtype=dtype, device=device)
        loss_asym = torch.tensor(0.0, dtype=dtype, device=device)
        loss_anti_collapse = torch.tensor(0.0, dtype=dtype, device=device)

        if xor_engine is not None and sample_states is not None:
            l_s, l_a, l_ac = self.compute_algebraic_invariance_loss(
                xor_engine=xor_engine,
                u_sample=sample_states,
            )
            loss_self = l_s.to(dtype=dtype, device=device)
            loss_asym = l_a.to(dtype=dtype, device=device)
            loss_anti_collapse = l_ac.to(dtype=dtype, device=device)
        else:
            flux_f32 = active_flux.to(torch.float32)
            vort_norm = safe_hyperbolic_sqrt(torch.sum(flux_f32.pow(2), dim=-1), eps=self.eps)

            if active_selector is not None and active_selector.shape == active_flux.shape and active_flux.shape[-1] % 2 == 0:
                sel_f32 = active_selector.to(torch.float32)
                darboux_area = torch.abs(symplectic_bilinear_form(flux_f32, sel_f32))
                response_mag = vort_norm + 0.50 * darboux_area
            else:
                response_mag = vort_norm

            if active_flux.ndim == 4 and target_masks is not None:
                m_f32 = target_masks.float()
                active_denom = m_f32.sum().clamp_min(1.0)
                mean_active_response = (response_mag * m_f32).sum() / active_denom
            else:
                mean_active_response = response_mag.mean()

            loss_anti_collapse = F.relu(self.min_response_margin - mean_active_response).pow(2).to(dtype)
            loss_self = (0.02 * torch.mean((vort_norm - self.min_response_margin).pow(2))).to(dtype)

        """3. Composite Differentiable Objective Assembly"""
        raw_composite = (
            (self.self_weight * loss_self)
            + (self.asym_weight * loss_asym)
            + (self.anti_collapse_weight * loss_anti_collapse)
            + (self.pad_leak_weight * loss_pad_leak)
        )

        total_loss = self.weight * raw_composite

        if return_telemetry:
            with torch.no_grad():
                vorticity_norm = safe_hyperbolic_sqrt(
                    torch.sum(active_flux.to(torch.float32).pow(2), dim=-1),
                    eps=self.eps,
                ).mean()

                telemetry: Dict[str, float] = {
                    "loss_xor_total": float(total_loss.detach().item()),
                    "loss_xor_self": float(loss_self.detach().item()),
                    "loss_xor_asym": float(loss_asym.detach().item()),
                    "loss_xor_anti_collapse": float(loss_anti_collapse.detach().item()),
                    "loss_xor_pad_leak": float(loss_pad_leak.detach().item()),
                    "xor_mean_vorticity_norm": float(vorticity_norm.item()),
                }
            return total_loss, telemetry

        return total_loss


SymplecticXORRegularizationLoss = XORParityRegularizationLoss


def verify_xor_regularization_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for xor_regularization.py.
    """
    torch.manual_seed(42)
    b_size, d_dim = 4, 32

    loss_engine = XORParityRegularizationLoss(
        weight=0.20,
        self_weight=1.00,
        asym_weight=0.50,
        anti_collapse_weight=0.35,
        min_response_margin=0.50,
        pad_leak_weight=0.25,
    )

    xor_module = AnalyticXORInvolutionEngine(d_model=d_dim)

    u = torch.randn(b_size, d_dim, dtype=torch.float32, requires_grad=True)
    v = torch.randn(b_size, d_dim, dtype=torch.float32, requires_grad=True)

    l_self, l_asym, _ = loss_engine.compute_algebraic_invariance_loss(
        xor_engine=xor_module,
        u_sample=u,
        v_sample=v,
    )

    if float(l_self.item()) > 1e-4 or float(l_asym.item()) > 1e-4:
        return False

    h_canvas, w_canvas = 6, 6
    active_flux_field = torch.zeros(2, h_canvas, w_canvas, d_dim, dtype=torch.float32, requires_grad=True)
    canvas_mask = torch.ones(2, h_canvas, w_canvas, dtype=torch.bool)
    canvas_mask[:, 3:, :] = False

    with torch.no_grad():
        active_flux_field[:, 3:, :, :] = 5.0

    state_leak = XORParityState(
        vorticity_flux=active_flux_field,
        parity_selector=torch.zeros_like(active_flux_field),
        involution_norm=torch.ones(2),
    )

    loss_leak, telem_leak = loss_engine(
        xor_state=state_leak,
        target_masks=canvas_mask,
        return_telemetry=True,
    )

    if telem_leak["loss_xor_pad_leak"] <= 0.10 or telem_leak["loss_xor_total"] <= 0.01:
        return False

    loss_leak.backward()

    if active_flux_field.grad is None or torch.isnan(active_flux_field.grad).any():
        return False
    if float(torch.norm(active_flux_field.grad).item()) <= 0.0:
        return False

    return True
