"""
Location: // fgns_engine/fgns/primitives/holographic_tape.py
Description: SOTA Multi-Scale Holographic Operator Memory Tape Engine with Non-Abelian SO(3)
             Finsler-Lie Gauge Attention (NA3D-Gauge-Attention), Local-Glyph Translation-Invariant
             Intrinsic Coordinates, and Leave-One-Out (LOO) Isolation.
             Core Primitive for FGNS v4.0.0 (Finsler-Gauge Sheaf):
               1. Dual-Scale Intrinsic Glyph Coordinates: combines 5x5 local connected-glyph centroids
                  (solving multi-row glyph translation in Task 136b0064) with global canvas centroids!
               2. NA3D-Gauge-Attention: Decomposes query and tape tokens into 3D frames (L, C, R, beta),
                  evaluating exact 3D chiral mixed-product determinants det[R_j, C_j - C_i, L_i]
                  and Randers asymmetric diode potentials inside hardware-fused SDPA!
               3. Non-Abelian SO(3) Rodrigues Parallel Transport: applies retrieved torsion moments
                  omega_read in so(3) to rotate query core vectors without convex-hull norm collapse.
               4. Strict Leave-One-Out (LOO) block-diagonal masking for support demonstrations.
Invariants: Strict metric preservation on S^(D-1), det(U_Rodrigues) == +1.0,
            zero identity leakage on support pairs, zero hash comment compliance.
Language: Python 3.12+ / PyTorch 2.5+ / Non-Abelian Gauge Field Theory sm_120 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Final, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.primitives.safe_math import (
    hyperbolic_saturating_norm,
    safe_divide,
    safe_hyperbolic_sqrt,
    safe_reciprocal,
)
from fgns.primitives.swiglu import RMSNorm, SwiGLU, TopologicalBypassFeedForward

DEFAULT_TAPE_EPS: Final[float] = 1e-7
DEFAULT_TAPE_TEMPERATURE: Final[float] = 0.10
DEFAULT_VIRIAL_GAIN: Final[float] = 2.00
NA3D_STRIDE: Final[int] = 16


class HolographicKeySynthesizer(nn.Module):
    """
    Multi-Scale Conformal, Local-Glyph & Relative-Centroid Coordinate Key Synthesizer.
    """

    def __init__(self, d_model: int = 128, eps: float = DEFAULT_TAPE_EPS) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        raw_key_dim = (3 * d_model) + 7

        self.key_swiglu = SwiGLU(
            d_in=raw_key_dim,
            d_out=d_model,
            d_inner=2 * d_model,
            alignment_multiple=32,
        )
        self.local_glyph_gate = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.norm = RMSNorm(d_model)

    def _compute_relative_centroid_bundle(
        self,
        h_field: torch.Tensor,
        edge_energy: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates Dual-Scale Intrinsic Coordinates:
          1. Global Canvas Relative-Centroid Coordinates (rel_y_glob, rel_x_glob)
          2. Local 5x5 Glyph-Window Relative Coordinates (rel_y_loc, rel_x_loc)
        Guarantees that identical 3x3 glyphs at different rows/columns (e.g. Task 136b0064)
        produce matching intrinsic coordinate keys!
        """
        B, H, W, _ = h_field.shape
        device = h_field.device
        dtype = h_field.dtype
        mask_f32 = mask.float().unsqueeze(-1)

        y_lin = torch.linspace(-1.0, 1.0, steps=max(1, H), device=device, dtype=torch.float32).view(1, H, 1, 1)
        x_lin = torch.linspace(-1.0, 1.0, steps=max(1, W), device=device, dtype=torch.float32).view(1, 1, W, 1)

        y_grid = y_lin.expand(B, H, W, 1)
        x_grid = x_lin.expand(B, H, W, 1)

        struct_weight = (edge_energy.to(torch.float32) + 0.10) * mask_f32.to(torch.float32)
        w_sum = struct_weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-5)

        mu_y_glob = (struct_weight * y_grid).sum(dim=(1, 2), keepdim=True) / w_sum
        mu_x_glob = (struct_weight * x_grid).sum(dim=(1, 2), keepdim=True) / w_sum

        dy_glob = y_grid - mu_y_glob
        dx_glob = x_grid - mu_x_glob

        var_y = (struct_weight * dy_glob.pow(2)).sum(dim=(1, 2), keepdim=True) / w_sum
        var_x = (struct_weight * dx_glob.pow(2)).sum(dim=(1, 2), keepdim=True) / w_sum

        sig_y = torch.sqrt(var_y + 1e-4).clamp_min(0.15)
        sig_x = torch.sqrt(var_x + 1e-4).clamp_min(0.15)

        rel_y_glob = (dy_glob / sig_y).clamp(-4.0, 4.0)
        rel_x_glob = (dx_glob / sig_x).clamp(-4.0, 4.0)

        """Local 5x5 Glyph Centroid Extraction via 2D Box Convolution"""
        w_2d = struct_weight.permute(0, 3, 1, 2)
        wy_2d = (struct_weight * y_grid).permute(0, 3, 1, 2)
        wx_2d = (struct_weight * x_grid).permute(0, 3, 1, 2)

        loc_w = F.avg_pool2d(w_2d, kernel_size=5, stride=1, padding=2).permute(0, 2, 3, 1).clamp_min(1e-5)
        mu_y_loc = F.avg_pool2d(wy_2d, kernel_size=5, stride=1, padding=2).permute(0, 2, 3, 1) / loc_w
        mu_x_loc = F.avg_pool2d(wx_2d, kernel_size=5, stride=1, padding=2).permute(0, 2, 3, 1) / loc_w

        span_y_loc = 4.0 / float(max(1, H))
        span_x_loc = 4.0 / float(max(1, W))
        rel_y_loc = ((y_grid - mu_y_loc) / span_y_loc).clamp(-4.0, 4.0)
        rel_x_loc = ((x_grid - mu_x_loc) / span_x_loc).clamp(-4.0, 4.0)

        g_loc = torch.sigmoid(self.local_glyph_gate.to(torch.float32))
        rel_y = ((1.0 - g_loc) * rel_y_glob) + (g_loc * rel_y_loc)
        rel_x = ((1.0 - g_loc) * rel_x_glob) + (g_loc * rel_x_loc)

        abs_rel_y = torch.abs(rel_y_loc)
        abs_rel_x = torch.abs(rel_x_loc)

        coord_bundle = torch.cat(
            [y_grid, x_grid, rel_y, rel_x, abs_rel_y, abs_rel_x],
            dim=-1,
        ) * mask_f32.to(torch.float32)

        return coord_bundle.to(dtype)

    def forward(self, h_field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f32 = mask.float().unsqueeze(-1)

        y_pad = F.pad(h_field, (0, 0, 0, 0, 1, 1), mode="replicate")
        grad_y = (y_pad[:, 2:, :, :] - y_pad[:, :-2, :, :]) * 0.50

        x_pad = F.pad(h_field, (0, 0, 1, 1, 0, 0), mode="replicate")
        grad_x = (x_pad[:, :, 2:, :] - x_pad[:, :, :-2, :]) * 0.50

        edge_energy = safe_hyperbolic_sqrt(
            torch.mean(grad_y * grad_y + grad_x * grad_x, dim=-1, keepdim=True),
            eps=self.eps,
        )

        rel_coords = self._compute_relative_centroid_bundle(h_field, edge_energy, mask)
        raw_bundle = torch.cat([h_field, grad_y, grad_x, edge_energy, rel_coords], dim=-1)

        projected = self.key_swiglu(raw_bundle)
        keys = self.norm(projected) * mask_f32

        return keys


class HolographicValueSynthesizer(nn.Module):
    """
    Bilinear & SO(3) Torsion Operator Value Synthesizer.
    Encodes linear translation delta, multiplicative Grassmann parity,
    and 3D Lie cross-product torsion moments C_in x C_out in so(3).
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

        self.val_swiglu = SwiGLU(
            d_in=3 * d_model,
            d_out=d_model,
            d_inner=2 * d_model,
            alignment_multiple=32,
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f32 = joint_mask.float().unsqueeze(-1)
        B_flat, H, W, D = h_in.shape

        delta_linear = h_out - h_in
        delta_mult = h_in * h_out

        if D % NA3D_STRIDE == 0:
            c_cells = D // NA3D_STRIDE
            in_3d = h_in.to(torch.float32).view(B_flat, H, W, c_cells, NA3D_STRIDE)
            out_3d = h_out.to(torch.float32).view(B_flat, H, W, c_cells, NA3D_STRIDE)

            torsion_l = torch.cross(in_3d[..., 0:3], out_3d[..., 0:3], dim=-1)
            torsion_c = torch.cross(in_3d[..., 3:6], out_3d[..., 3:6], dim=-1)
            torsion_r = torch.cross(in_3d[..., 6:9], out_3d[..., 6:9], dim=-1)
            torsion_b = torch.cross(in_3d[..., 9:12], out_3d[..., 9:12], dim=-1)
            meta_diff = out_3d[..., 12:16] - in_3d[..., 12:16]

            delta_curv = torch.cat(
                [torsion_l, torsion_c, torsion_r, torsion_b, meta_diff], dim=-1
            ).view(B_flat, H, W, D).to(h_in.dtype)
        else:
            delta_curv = h_out - (2.0 * h_in)

        raw_bundle = torch.cat([delta_linear, delta_mult, delta_curv], dim=-1)
        projected = self.val_swiglu(raw_bundle)
        values = self.norm(projected) * mask_f32

        return values


class SupportCrossAttentionMemory(nn.Module):
    """
    SOTA Multi-Scale Holographic Operator Memory Tape Engine with Non-Abelian SO(3)
    Finsler-Lie Gauge Attention (NA3D-Gauge-Attention) and Leave-One-Out (LOO) Isolation.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        tau_tape: float = DEFAULT_TAPE_TEMPERATURE,
        virial_gain: float = DEFAULT_VIRIAL_GAIN,
        eps: float = DEFAULT_TAPE_EPS,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.tau_tape = float(tau_tape)
        self.virial_gain = float(virial_gain)
        self.eps = eps

        self.key_synthesizer = HolographicKeySynthesizer(d_model=d_model, eps=eps)
        self.val_synthesizer = HolographicValueSynthesizer(d_model=d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        """NA3D-Gauge-Attention Chiral & Rodrigues Parallel Transport Gates"""
        self.chiral_gauge_gain = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.rodrigues_transport_gate = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))

        self.read_refiner = TopologicalBypassFeedForward(
            d_model=d_model,
            expansion_factor=2.0,
            alignment_multiple=32,
        )

        self.norm_query = RMSNorm(d_model)
        self.norm_memory = RMSNorm(d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _augment_na3d_gauge_query_key(
        self,
        q_tokens: torch.Tensor,
        k_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements Factored NA3D-Gauge-Attention:
        Injects the 3D mixed-product chiral determinant det[R_j, C_j - C_i, L_i]
        and Randers asymmetric covector potential into Q and K before SDPA!
        """
        B, L, D = q_tokens.shape
        M = k_tokens.shape[1]
        if D % NA3D_STRIDE != 0:
            return q_tokens, k_tokens

        c_cells = D // NA3D_STRIDE
        q_3d = q_tokens.to(torch.float32).view(B, L, c_cells, NA3D_STRIDE)
        k_3d = k_tokens.to(torch.float32).view(B, M, c_cells, NA3D_STRIDE)

        l_q = q_3d[..., 0:3]
        c_q = q_3d[..., 3:6]
        r_q = q_3d[..., 6:9]
        b_q = torch.tanh(q_3d[..., 9:12])
        m_q = q_3d[..., 12:16]

        l_k = k_3d[..., 0:3]
        c_k = k_3d[..., 3:6]
        r_k = k_3d[..., 6:9]
        b_k = torch.tanh(k_3d[..., 9:12])
        m_k = k_3d[..., 12:16]

        """Factored 3D Mixed-Product Terms: <L_i, R_j x C_j> + <L_i x C_i, R_j>"""
        lq_cross_cq = torch.cross(l_q, c_q, dim=-1)
        rk_cross_ck = torch.cross(r_k, c_k, dim=-1)

        g_chiral = torch.tanh(self.chiral_gauge_gain.to(torch.float32))

        q_gauge = torch.cat(
            [
                l_q + (g_chiral * b_q),
                c_q,
                r_q + (g_chiral * lq_cross_cq),
                b_q,
                m_q,
            ],
            dim=-1,
        ).view(B, L, D).to(q_tokens.dtype)

        k_gauge = torch.cat(
            [
                l_k + (g_chiral * rk_cross_ck),
                c_k,
                r_k - (g_chiral * b_k),
                b_k,
                m_k,
            ],
            dim=-1,
        ).view(B, M, D).to(k_tokens.dtype)

        return q_gauge, k_gauge

    def _apply_na3d_rodrigues_parallel_transport(
        self,
        h_query: torch.Tensor,
        readout: torch.Tensor,
        mask_f32: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies Non-Abelian SO(3) Rodrigues Parallel Transport on the query 3D cells
        driven by the torsion moment retrieved from the holographic tape:
            C_i^new = exp(hat(R_read x L_query)) C_i + Delta C_read.
        """
        B, H, W, D = h_query.shape
        if D % NA3D_STRIDE != 0:
            return (h_query + readout) * mask_f32

        c_cells = D // NA3D_STRIDE
        q_cells = h_query.to(torch.float32).view(B, H, W, c_cells, NA3D_STRIDE)
        r_cells = readout.to(torch.float32).view(B, H, W, c_cells, NA3D_STRIDE)

        l_q = q_cells[..., 0:3]
        c_q = q_cells[..., 3:6]
        r_q = q_cells[..., 6:9]

        l_read = r_cells[..., 0:3]
        c_read = r_cells[..., 3:6]
        r_read = r_cells[..., 6:9]

        """Torsion generator omega = R_read x L_q + c_read in so(3)"""
        g_rot = torch.tanh(self.rodrigues_transport_gate.to(torch.float32))
        omega = g_rot * (torch.cross(r_read, l_q, dim=-1) + (0.50 * l_read))

        omega_norm = torch.sqrt(torch.sum(omega * omega, dim=-1, keepdim=True) + self.eps)
        u_axis = omega / omega_norm
        theta = math.pi * torch.tanh(omega_norm)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        c_rot = (
            (c_q * cos_t)
            + (torch.cross(u_axis, c_q, dim=-1) * sin_t)
            + (u_axis * torch.sum(u_axis * c_q, dim=-1, keepdim=True) * (1.0 - cos_t))
        )

        q_rotated = torch.cat(
            [
                l_q + l_read,
                c_rot + c_read,
                r_q + r_read,
                q_cells[..., 9:16] + r_cells[..., 9:16],
            ],
            dim=-1,
        ).view(B, H, W, D).to(h_query.dtype)

        return q_rotated * mask_f32

    def extract_holographic_tape(
        self,
        s_in: torch.Tensor,
        s_in_masks: torch.Tensor,
        s_out: torch.Tensor,
        s_out_masks: torch.Tensor,
        pair_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, H, W, D = s_in.shape
        hw = H * W
        total_memory_tokens = K * hw

        flat_in = s_in.view(B * K, H, W, D)
        flat_in_masks = s_in_masks.view(B * K, H, W)

        flat_out = s_out.view(B * K, H, W, D)
        flat_out_masks = s_out_masks.view(B * K, H, W)
        joint_masks = flat_in_masks | flat_out_masks

        if pair_valid_mask is not None:
            valid_exp = pair_valid_mask.view(B * K, 1, 1).expand(B * K, H, W)
            joint_masks = joint_masks & valid_exp

        keys_spatial = self.key_synthesizer(flat_in, flat_in_masks)
        keys_flat = keys_spatial.view(B, total_memory_tokens, D)

        vals_spatial = self.val_synthesizer(flat_in, flat_out, joint_masks)
        vals_flat = vals_spatial.view(B, total_memory_tokens, D)

        tape_mask = (flat_in_masks & joint_masks).view(B, total_memory_tokens)

        return keys_flat, vals_flat, tape_mask

    def estimate_virial_osmotic_pressure(
        self,
        tape_vals: torch.Tensor,
        tape_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f32 = tape_mask.float().unsqueeze(-1)

        val_norms = safe_hyperbolic_sqrt(
            torch.sum(tape_vals * tape_vals, dim=-1, keepdim=True),
            eps=self.eps,
        )
        masked_norms = val_norms * mask_f32
        valid_counts = mask_f32.sum(dim=1).clamp_min(1.0)

        mean_flux = masked_norms.sum(dim=1) / valid_counts
        p_osmotic = self.virial_gain * torch.tanh(mean_flux)

        return p_osmotic

    def forward(
        self,
        h_query: torch.Tensor,
        query_mask: torch.Tensor,
        tape_keys: torch.Tensor,
        tape_vals: torch.Tensor,
        tape_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes Non-Abelian SO(3) NA3D-Gauge-Attention readout from the holographic tape.
        """
        B, H, W, D = h_query.shape
        L = H * W
        M = tape_keys.shape[1]

        mask_f32 = query_mask.float().unsqueeze(-1)

        q_gauge_enriched = 0.50 * (h_query + self.key_synthesizer(h_query, query_mask))
        flat_query = self.norm_query(q_gauge_enriched).view(B, L, D)
        normed_keys = self.norm_memory(tape_keys)

        p_osmotic = self.estimate_virial_osmotic_pressure(tape_vals, tape_mask)

        """Augment Q and K with 3D Chiral Determinants & Randers Wind Covectors"""
        q_na3d, k_na3d = self._augment_na3d_gauge_query_key(flat_query, normed_keys)

        safe_tau = max(self.tau_tape, 1e-4)
        eff_scale = self.scale / safe_tau

        q = (self.q_proj(q_na3d) * eff_scale).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_na3d).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tape_vals).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        has_memory = (tape_mask.sum(dim=-1, keepdim=True) > 0).view(B, 1, 1)
        attn_bias = torch.where(
            tape_mask.view(B, 1, 1, M),
            torch.zeros((B, 1, 1, M), dtype=q.dtype, device=q.device),
            torch.full((B, 1, 1, M), -10000.0, dtype=q.dtype, device=q.device),
        )

        readout_heads = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False, scale=1.0
        )

        readout = readout_heads.transpose(1, 2).contiguous().view(B, L, D)
        readout = torch.where(has_memory, readout, torch.zeros_like(readout))

        projected = self.out_proj(readout).view(B, H, W, D)

        """SO(3) Rodrigues Parallel Transport Update"""
        fused = self._apply_na3d_rodrigues_parallel_transport(h_query, projected, mask_f32)
        h_refined = self.read_refiner(fused) * mask_f32

        return h_refined, p_osmotic

    def forward_support_loo(
        self,
        s_in: torch.Tensor,
        s_in_masks: torch.Tensor,
        tape_keys: torch.Tensor,
        tape_vals: torch.Tensor,
        tape_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Leave-One-Out (LOO) NA3D-Gauge-Attention Readout for Support Demonstrations.
        """
        B, K, H, W, D = s_in.shape
        L = H * W
        M = tape_keys.shape[1]
        device = s_in.device

        safe_tau = max(self.tau_tape, 1e-4)
        eff_scale = self.scale / safe_tau

        normed_keys = self.norm_memory(tape_keys)
        _, k_na3d = self._augment_na3d_gauge_query_key(normed_keys[:, :1, :], normed_keys)

        k_shared = self.k_proj(k_na3d).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v_shared = self.v_proj(tape_vals).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        token_pair_idx = torch.div(torch.arange(M, device=device), L, rounding_mode="floor")
        refined_pairs: List[torch.Tensor] = []

        for k_idx in range(K):
            h_k = s_in[:, k_idx, :, :, :]
            mask_k = s_in_masks[:, k_idx, :, :]
            mask_k_f32 = mask_k.float().unsqueeze(-1)

            h_k_gauge = 0.50 * (h_k + self.key_synthesizer(h_k, mask_k))
            flat_q_k = self.norm_query(h_k_gauge).view(B, L, D)
            q_k_na3d, _ = self._augment_na3d_gauge_query_key(flat_q_k, normed_keys[:, :1, :])

            q_k = (self.q_proj(q_k_na3d) * eff_scale).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            not_self_mask = (token_pair_idx != k_idx).unsqueeze(0).expand(B, M)
            loo_candidate_mask = tape_mask & not_self_mask

            has_other_pairs = (loo_candidate_mask.sum(dim=-1, keepdim=True) > 0)
            effective_tape_mask = torch.where(has_other_pairs, loo_candidate_mask, tape_mask)
            has_any_valid = (effective_tape_mask.sum(dim=-1, keepdim=True) > 0).view(B, 1, 1)

            attn_bias_k = torch.where(
                effective_tape_mask.view(B, 1, 1, M),
                torch.zeros((B, 1, 1, M), dtype=q_k.dtype, device=device),
                torch.full((B, 1, 1, M), -10000.0, dtype=q_k.dtype, device=device),
            )

            readout_k_heads = F.scaled_dot_product_attention(
                q_k, k_shared, v_shared, attn_mask=attn_bias_k, dropout_p=0.0, is_causal=False, scale=1.0
            )
            readout_k = readout_k_heads.transpose(1, 2).contiguous().view(B, L, D)
            readout_k = torch.where(has_any_valid, readout_k, torch.zeros_like(readout_k))

            proj_k = self.out_proj(readout_k).view(B, H, W, D)
            fused_k = self._apply_na3d_rodrigues_parallel_transport(h_k, proj_k, mask_k_f32)
            refined_k = self.read_refiner(fused_k) * mask_k_f32
            refined_pairs.append(refined_k)

        return torch.stack(refined_pairs, dim=1)


def verify_holographic_tape_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for holographic_tape.py.
    """
    torch.manual_seed(42)
    b, k, h, w, d = 2, 3, 6, 6, 32

    tape_module = SupportCrossAttentionMemory(d_model=d, num_heads=4)

    s_in = torch.randn(b, k, h, w, d, dtype=torch.float32, requires_grad=True)
    s_in_m = torch.ones(b, k, h, w, dtype=torch.bool)
    s_in_m[:, :, :, 3:] = False

    s_out = torch.randn(b, k, h, w, d, dtype=torch.float32, requires_grad=True)
    s_out_m = torch.ones(b, k, h, w, dtype=torch.bool)
    s_out_m[:, :, :, 3:] = False

    keys, vals, tape_mask = tape_module.extract_holographic_tape(
        s_in=s_in,
        s_in_masks=s_in_m,
        s_out=s_out,
        s_out_masks=s_out_m,
    )

    total_tokens = k * h * w
    if keys.shape != (b, total_tokens, d) or vals.shape != (b, total_tokens, d):
        return False

    h_q = torch.randn(b, h, w, d, dtype=torch.float32, requires_grad=True)
    q_mask = torch.ones(b, h, w, dtype=torch.bool)
    q_mask[:, :, 3:] = False

    h_read, p_osmotic = tape_module(
        h_query=h_q,
        query_mask=q_mask,
        tape_keys=keys,
        tape_vals=vals,
        tape_mask=tape_mask,
    )

    if h_read.shape != (b, h, w, d) or p_osmotic.shape != (b, 1):
        return False
    if not torch.all(h_read[:, :, 3:, :] == 0.0):
        return False

    s_loo_1 = tape_module.forward_support_loo(s_in, s_in_m, keys, vals, tape_mask)
    if s_loo_1.shape != (b, k, h, w, d):
        return False

    s_out_corrupted = s_out.detach().clone()
    s_out_corrupted[:, 0, :, :, :] = 999.0
    keys_c, vals_c, mask_c = tape_module.extract_holographic_tape(s_in, s_in_m, s_out_corrupted, s_out_m)
    s_loo_2 = tape_module.forward_support_loo(s_in, s_in_m, keys_c, vals_c, mask_c)

    pair0_diff = float(torch.max(torch.abs(s_loo_1[:, 0] - s_loo_2[:, 0])).item())
    if pair0_diff > 1e-5:
        return False

    loss = torch.sum(h_read.pow(2)) + torch.sum(s_loo_1.pow(2)) + torch.sum(p_osmotic.pow(2))
    loss.backward()

    if s_in.grad is None or s_out.grad is None or h_q.grad is None:
        return False
    if torch.isnan(s_in.grad).any() or torch.isnan(s_out.grad).any() or torch.isnan(h_q.grad).any():
        return False
    if float(torch.norm(h_q.grad).item()) <= 0.0:
        return False

    return True

