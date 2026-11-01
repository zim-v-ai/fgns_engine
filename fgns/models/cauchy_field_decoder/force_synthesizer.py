"""
Location: // fgns_engine/fgns/models/accro_field_decoder/force_synthesizer.py
Description: SOTA Sextuple Relational Reasoning Force Synthesizer with TRM Draft Error
             Self-Correction Residual for 4-Step Cellular TRM-ACCRO Flow.
Language: Python 3.12+ / PyTorch 2.5+ / Dynamical Field Reasoning sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.safe_math import (
    hyperbolic_saturating_norm,
    safe_divide,
    safe_hyperbolic_sqrt,
)
from fgns.primitives.swiglu import RMSNorm, SwiGLU
from fgns.primitives.wavelet_kan import FactoredMorletKANLinear
from fgns.types import TargetCoordinateBundle

DEFAULT_FORCE_MAX: Final[float] = 6.00
DEFAULT_FORCE_EPS: Final[float] = 1e-7


class RelationalReasoningForceSynthesizer(nn.Module):
    """
    SOTA Sextuple-Stream Relational Reasoning Force Synthesizer Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_wavelets: int = 4,
        f_max: float = DEFAULT_FORCE_MAX,
        eps: float = DEFAULT_FORCE_EPS,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"Dimension d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.f_max = float(f_max)
        self.eps = eps

        self.q_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.kan_modulator = FactoredMorletKANLinear(
            d_in=d_model,
            d_out=d_model,
            num_wavelets=num_wavelets,
            omega_0=5.0,
        )

        raw_inner = int(d_model * 2.5)
        d_inner = ((raw_inner + 31) // 32) * 32

        self.force_swiglu = SwiGLU(
            d_in=6 * d_model,
            d_out=d_model,
            d_inner=d_inner,
            alignment_multiple=32,
        )

        self.draft_residual_gate = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))

        self.norm_h = RMSNorm(d_model)
        self.norm_coords = RMSNorm(d_model)
        self.norm_slots = RMSNorm(d_model)
        self.norm_chem = RMSNorm(d_model)
        self.norm_xor = RMSNorm(d_model)
        self.norm_tape = RMSNorm(d_model)
        self.norm_output = RMSNorm(d_model)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        h_current: torch.Tensor,
        packed_slots: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None,
        chemical_biases: Optional[torch.Tensor] = None,
        palette_prototypes: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
        xor_stimulus: Optional[torch.Tensor] = None,
        h_tape: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        draft_error_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B_c, H, W, D = h_current.shape
        B_s, S, _ = packed_slots.shape
        hw = H * W
        device = h_current.device
        dtype = h_current.dtype

        if B_c != B_s:
            if B_c % B_s == 0:
                reps = B_c // B_s
                slots_aligned = packed_slots.repeat_interleave(reps, dim=0)
                mask_aligned = slot_mask.repeat_interleave(reps, dim=0) if slot_mask is not None else None
            else:
                slots_aligned = packed_slots[:B_c]
                mask_aligned = slot_mask[:B_c] if slot_mask is not None else None
        else:
            slots_aligned = packed_slots
            mask_aligned = slot_mask

        if target_coords is not None:
            coords_feed = self.norm_coords(target_coords)
        else:
            coords_feed = torch.zeros((B_c, H, W, D), dtype=dtype, device=device)

        h_effective = h_current
        if draft_error_residual is not None:
            g_err = torch.tanh(self.draft_residual_gate.to(dtype))
            h_effective = h_current + (g_err * draft_error_residual)

        h_normed = self.norm_h(h_effective)

        joint_query_input = torch.cat([h_normed, coords_feed], dim=-1).view(B_c, hw, 2 * D)
        q_heads = self.q_proj(joint_query_input).view(B_c, hw, self.num_heads, self.head_dim).transpose(1, 2)

        slots_normed = self.norm_slots(slots_aligned)
        k_heads = self.k_proj(slots_normed).view(B_c, S, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = self.v_proj(slots_normed).view(B_c, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * self.scale

        if mask_aligned is not None:
            mask_expanded = (mask_aligned > 0.05).view(B_c, 1, 1, S).expand(B_c, self.num_heads, hw, S)
            attn_logits = attn_logits.masked_fill(~mask_expanded, -10000.0)

        attn_weights = F.softmax(attn_logits, dim=-1)
        safe_attn = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        slot_context = torch.matmul(safe_attn, v_heads)
        slot_context = slot_context.transpose(1, 2).contiguous().view(B_c, H, W, D)
        slot_context = self.out_proj(slot_context)

        f_slots = self.kan_modulator(slot_context)

        if chemical_biases is not None and palette_prototypes is not None:
            proto_10 = palette_prototypes[:10, :].to(dtype=dtype, device=device)
            chem_biases_aligned = chemical_biases.to(dtype=dtype, device=device)

            chem_vector = torch.matmul(chem_biases_aligned, proto_10)
            f_chem = self.norm_chem(chem_vector.view(B_c, 1, 1, D).expand(B_c, H, W, D))
        else:
            f_chem = torch.zeros((B_c, H, W, D), dtype=dtype, device=device)

        if xor_stimulus is not None:
            f_xor = self.norm_xor(xor_stimulus)
        else:
            f_xor = torch.zeros((B_c, H, W, D), dtype=dtype, device=device)

        if h_tape is not None:
            f_tape = self.norm_tape(h_tape)
        else:
            f_tape = torch.zeros((B_c, H, W, D), dtype=dtype, device=device)

        joint_force_bundle = torch.cat(
            [h_effective, f_slots, f_chem, coords_feed, f_xor, f_tape],
            dim=-1,
        )
        f_raw = self.force_swiglu(joint_force_bundle)

        f_saturated = hyperbolic_saturating_norm(
            f_raw,
            v_max=self.f_max,
            eps=self.eps,
        )

        f_driving = self.norm_output(f_saturated)

        if target_mask is not None:
            mask_f32 = target_mask.float().unsqueeze(-1)
            f_driving = f_driving * mask_f32

        return f_driving


def verify_force_synthesizer_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for force_synthesizer.py.
    """
    torch.manual_seed(42)
    b_s, b_c = 2, 4
    h, w, d_model, num_slots = 4, 4, 32, 4

    synthesizer = RelationalReasoningForceSynthesizer(
        d_model=d_model,
        num_heads=4,
        num_wavelets=4,
        f_max=6.0,
    )

    h_curr = torch.randn(b_c, h, w, d_model, dtype=torch.float32, requires_grad=True)
    slots = torch.randn(b_s, num_slots, d_model, dtype=torch.float32, requires_grad=True)
    slot_m = torch.ones(b_s, num_slots, 1, dtype=torch.float32)
    target_c = torch.randn(b_c, h, w, d_model, dtype=torch.float32, requires_grad=True)
    chem_b = torch.randn(b_c, 10, dtype=torch.float32, requires_grad=True)
    prototypes = torch.randn(11, d_model, dtype=torch.float32)
    xor_feed = torch.randn(b_c, h, w, d_model, dtype=torch.float32, requires_grad=True)
    h_tape = torch.randn(b_c, h, w, d_model, dtype=torch.float32, requires_grad=True)
    draft_err = torch.randn(b_c, h, w, d_model, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(b_c, h, w, dtype=torch.bool)
    mask[:, :, 2:] = False

    f_driving = synthesizer(
        h_current=h_curr,
        packed_slots=slots,
        target_coords=target_c,
        chemical_biases=chem_b,
        palette_prototypes=prototypes,
        slot_mask=slot_m,
        xor_stimulus=xor_feed,
        h_tape=h_tape,
        target_mask=mask,
        draft_error_residual=draft_err,
    )

    if f_driving.shape != (b_c, h, w, d_model) or not torch.all(f_driving[:, :, 2:, :] == 0.0):
        return False

    loss = torch.sum(f_driving.pow(2))
    loss.backward()

    if h_curr.grad is None or slots.grad is None or h_tape.grad is None or draft_err.grad is None:
        return False

    return True
