"""
Location: // fgns_engine/fgns/kernels/spectral_laplacian.py
Description: Matrix-Free Chebyshev Graph Laplacian, Reference-Retaining Zero-ABA Cached CUDA
             Finsler-Chordal Cavity Solver, Unified Differentiable Spectral-Green RKHS Operator
             (Explicit Analytical Woodbury Forward & Implicit Adjoint Backward Pass), and
             Multi-Scale Harmonic Lattice Operators (FGNS v5.0.0):
               1. SpectralGreenWoodburyFunction (torch.autograd.Function): solves the unified
                  Tikhonov-regularized Gram system G_lambda * W* = Psi_S^T M_S Y_S + lambda * W_0
                  in the forward pass and the exact implicit adjoint PDE G_lambda * Lambda = G_up
                  in the backward pass without unrolling iterations.
               2. DifferentiableSpectralGreenOperator (nn.Module): learnable D4-steerable radial-shell
                  wavelet filter bank, Gaussian-Bessel harmonic cavity basis, and Woodbury-Tikhonov
                  RKHS transport with end-to-end analytical gradients.
               3. Zero-ABA Lifecycle-Safe Bundle Cache: retains tensor references alongside version
                  counters and explicit clear_cache() boundaries.
Language: Python 3.12+ / PyTorch 2.5+ / Native CUDA C++17 sm_120 & sm_89 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.kernels.cuda_fgns_ops import cuda_finsler_chordal_cavity_op
from fgns.primitives.swiglu import RMSNorm, SwiGLU
from fgns.types import BravaisPeriodicityState, ColorRoleState, EikonalLevelSetState

DEFAULT_LAPLACIAN_EPS: Final[float] = 1e-7
NUM_CHORDAL_MODES: Final[int] = 3
NUM_TOPO_CHANNELS: Final[int] = 23


class SpectralGreenWoodburyFunction(torch.autograd.Function):
    """
    Unified Analytical Forward and Implicit-Adjoint Backward Operator for Support-to-Query
    Spectral-Green RKHS Functional Transport:

      Forward Pass:
        G_lambda = Psi_S^T (M_S (.) Psi_S) + diag(lambda)
        R        = Psi_S^T (M_S (.) Y_S)   + lambda (.) W_0
        W*       = G_lambda^{-1} R
        Z_q      = Psi_q W*

      Backward Pass (Exact Implicit Differentiation via Adjoint Green's System):
        G_lambda Lambda = Psi_q^T grad_Z_q + grad_W*
        grad_Psi_q      = grad_Z_q (W*)^T
        grad_Psi_S      = M_S (.) [ Y_S Lambda^T - Psi_S (W* Lambda^T + Lambda (W*)^T) ]
        grad_Y_S        = M_S (.) (Psi_S Lambda)
        grad_M_S        = sum_d Psi_S (.) [ (Y_S - Psi_S W*) Lambda^T ]
        grad_lambda     = sum_c Lambda (.) (W_0 - W*)
        grad_W_0        = lambda (.) Lambda
    """

    @staticmethod
    def forward(
        ctx: Any,
        psi_s: torch.Tensor,
        y_s: torch.Tensor,
        m_s: torch.Tensor,
        psi_q: torch.Tensor,
        lambda_vec: torch.Tensor,
        w_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        work_dtype = torch.float64 if psi_s.dtype == torch.float64 else torch.float32
        psi_s_f = psi_s.to(work_dtype)
        y_s_f = y_s.to(work_dtype)
        m_s_f = m_s.to(work_dtype)
        psi_q_f = psi_q.to(work_dtype)
        lam_f = lambda_vec.to(work_dtype).clamp_min(1e-5)
        w0_f = w_prior.to(work_dtype)

        w_psi_s = psi_s_f * m_s_f
        gram = torch.bmm(psi_s_f.transpose(1, 2), w_psi_s)
        diag_lam = torch.diag_embed(lam_f.squeeze(-1))
        g_lambda = gram + diag_lam

        rhs = torch.bmm(w_psi_s.transpose(1, 2), y_s_f) + (lam_f * w0_f)
        w_star = torch.linalg.solve(g_lambda, rhs)
        z_q = torch.bmm(psi_q_f, w_star)

        ctx.save_for_backward(psi_s_f, y_s_f, m_s_f, psi_q_f, lam_f, w0_f, g_lambda, w_star)
        return z_q.to(psi_q.dtype), w_star.to(psi_s.dtype)

    @staticmethod
    def backward(
        ctx: Any,
        grad_z_q: Optional[torch.Tensor],
        grad_w_star: Optional[torch.Tensor],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        psi_s_f, y_s_f, m_s_f, psi_q_f, lam_f, w0_f, g_lambda, w_star = ctx.saved_tensors
        work_dtype = psi_s_f.dtype

        gz_f = (
            grad_z_q.to(work_dtype)
            if grad_z_q is not None
            else torch.zeros((psi_q_f.shape[0], psi_q_f.shape[1], w_star.shape[-1]), dtype=work_dtype, device=psi_q_f.device)
        )
        gw_f = (
            grad_w_star.to(work_dtype)
            if grad_w_star is not None
            else torch.zeros_like(w_star)
        )

        g_up = torch.bmm(psi_q_f.transpose(1, 2), gz_f) + gw_f
        adjoint_lambda = torch.linalg.solve(g_lambda, g_up)

        grad_psi_q = torch.bmm(gz_f, w_star.transpose(1, 2))

        sym_w_adj = torch.bmm(w_star, adjoint_lambda.transpose(1, 2)) + torch.bmm(
            adjoint_lambda, w_star.transpose(1, 2)
        )
        grad_psi_s_raw = torch.bmm(y_s_f, adjoint_lambda.transpose(1, 2)) - torch.bmm(psi_s_f, sym_w_adj)
        grad_psi_s = m_s_f * grad_psi_s_raw

        grad_y_s = m_s_f * torch.bmm(psi_s_f, adjoint_lambda)

        residual_s = y_s_f - torch.bmm(psi_s_f, w_star)
        grad_m_s = torch.sum(
            psi_s_f * torch.bmm(residual_s, adjoint_lambda.transpose(1, 2)),
            dim=-1,
            keepdim=True,
        )

        grad_lam = torch.sum(adjoint_lambda * (w0_f - w_star), dim=-1, keepdim=True)
        grad_w0 = lam_f * adjoint_lambda

        return (
            grad_psi_s,
            grad_y_s,
            grad_m_s,
            grad_psi_q,
            grad_lam,
            grad_w0,
        )


def apply_spectral_green_woodbury(
    psi_s: torch.Tensor,
    y_s: torch.Tensor,
    m_s: torch.Tensor,
    psi_q: torch.Tensor,
    lambda_vec: torch.Tensor,
    w_prior: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Typed functional wrapper around SpectralGreenWoodburyFunction.apply.
    """
    res = SpectralGreenWoodburyFunction.apply(psi_s, y_s, m_s, psi_q, lambda_vec, w_prior)
    return cast(Tuple[torch.Tensor, torch.Tensor], res)


class StencilGraphLaplacianOps:
    """
    Fast F.pad slice-vectorized 2D stencil implementation of the normalized graph Laplacian.
    """

    DIRECTIONS: Tuple[Tuple[int, int, float], ...] = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 0.70710678),
        (-1, 1, 0.70710678),
        (1, -1, 0.70710678),
        (1, 1, 0.70710678),
    )

    @classmethod
    def shift_tensor_2d(cls, x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x_pad = F.pad(x.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)
        y0 = 1 - dy
        x0 = 1 - dx
        return x_pad[:, y0 : y0 + H, x0 : x0 + W, :]

    @classmethod
    def compute_edge_weights_and_degree(
        cls, grid: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = grid.shape
        device = grid.device
        mask_f32 = mask.float().unsqueeze(-1)
        g_f32 = grid.float().unsqueeze(-1)

        g_pad = F.pad(g_f32.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=-1.0).permute(0, 2, 3, 1)
        m_pad = F.pad(mask_f32.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)

        nb_g = torch.cat(
            [g_pad[:, (1 - dy) : (1 - dy + H), (1 - dx) : (1 - dx + W), :] for dy, dx, _ in cls.DIRECTIONS],
            dim=-1,
        )
        nb_m = torch.cat(
            [m_pad[:, (1 - dy) : (1 - dy + H), (1 - dx) : (1 - dx + W), :] for dy, dx, _ in cls.DIRECTIONS],
            dim=-1,
        )
        metric_w = torch.tensor(
            [m for _, _, m in cls.DIRECTIONS], dtype=torch.float32, device=device
        ).view(1, 1, 1, 8)

        edge_weights = (g_f32 == nb_g).float() * mask_f32 * nb_m * metric_w
        degree_accum = torch.sum(edge_weights, dim=-1, keepdim=True)

        is_positive_deg = degree_accum > 0.0
        inv_sqrt_deg = torch.where(
            is_positive_deg,
            torch.rsqrt(torch.clamp_min(degree_accum, DEFAULT_LAPLACIAN_EPS)),
            torch.zeros_like(degree_accum),
        ) * mask_f32

        return edge_weights, inv_sqrt_deg

    @classmethod
    def apply_normalized_laplacian(
        cls,
        v: torch.Tensor,
        edge_weights: torch.Tensor,
        inv_sqrt_deg: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W, C = v.shape
        mask_f32 = mask.float().unsqueeze(-1)
        v_scaled = v * inv_sqrt_deg

        v_pad = F.pad(v_scaled.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)
        nb_stack = torch.stack(
            [v_pad[:, (1 - dy) : (1 - dy + H), (1 - dx) : (1 - dx + W), :] for dy, dx, _ in cls.DIRECTIONS],
            dim=3,
        )

        adjacency_v = torch.sum(edge_weights.unsqueeze(-1) * nb_stack, dim=3)
        normalized_adjacency_v = adjacency_v * inv_sqrt_deg
        return (-normalized_adjacency_v) * mask_f32


class PoissonDirichletCavitySolver:
    """
    Reference-Retaining Zero-ABA Cached Native CUDA C++ Finsler-Chordal,
    Vectorized Color Structural Role, and Harmonic Lattice Basis Engine.
    """

    _BUNDLE_CACHE_DICT: Dict[
        Tuple[int, int, Tuple[int, ...], str, int, int],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
    ] = {}

    SHELL_1_DIRS: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    SHELL_2_DIRS: Tuple[Tuple[int, int], ...] = (
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (1, -2), (-1, 2), (1, 2),
    )

    @classmethod
    def clear_cache(cls) -> None:
        cls._BUNDLE_CACHE_DICT.clear()

    @classmethod
    def _compute_full_cached(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        num_iterations: int = 12,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            key = (
                int(grid.data_ptr()),
                int(mask.data_ptr()),
                tuple(grid.shape),
                str(grid.device),
                int(grid._version),
                int(mask._version),
            )
            if key in cls._BUNDLE_CACHE_DICT:
                cached_grid, cached_mask, cached_res = cls._BUNDLE_CACHE_DICT[key]
                if cached_grid is grid and cached_mask is mask:
                    return cached_res

            res = cuda_finsler_chordal_cavity_op(grid, mask, num_iterations=num_iterations)
            if len(cls._BUNDLE_CACHE_DICT) >= 6:
                cls._BUNDLE_CACHE_DICT.clear()
            cls._BUNDLE_CACHE_DICT[key] = (grid, mask, res)
            return res

    @classmethod
    def compute_steerable_d4_radial_wavelet(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        beta_sharpness: Optional[torch.Tensor] = None,
        card_threshold: Optional[torch.Tensor] = None,
        diag_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable D4-Steerable Radial-Shell Wavelet Operator:
        Evaluates continuous radial shell agreement on the r=1 cardinal ring S_1 and
        r=sqrt(2) diagonal ring S_2 via smooth Boltzmann-Sigmoid gates, propagating
        through a differentiable soft-dilation kernel so gradients flow w.r.t.
        (beta_sharpness, card_threshold, diag_threshold).
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)

        g_pad = F.pad(g_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=-1.0).squeeze(1).to(torch.int64)
        m_pad = F.pad(m_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=0.0).squeeze(1) > 0.5

        g_n, g_s = g_pad[:, :-2, 1:-1], g_pad[:, 2:, 1:-1]
        g_w, g_e = g_pad[:, 1:-1, :-2], g_pad[:, 1:-1, 2:]
        m_n, m_s = m_pad[:, :-2, 1:-1], m_pad[:, 2:, 1:-1]
        m_w, m_e = m_pad[:, 1:-1, :-2], m_pad[:, 1:-1, 2:]

        g_nw, g_ne = g_pad[:, :-2, :-2], g_pad[:, :-2, 2:]
        g_sw, g_se = g_pad[:, 2:, :-2], g_pad[:, 2:, 2:]
        m_nw, m_ne = m_pad[:, :-2, :-2], m_pad[:, :-2, 2:]
        m_sw, m_se = m_pad[:, 2:, :-2], m_pad[:, 2:, 2:]

        is_fg = ((g_flat > 0) & (g_flat < 10) & m_flat).to(torch.float32)

        k_n = ((g_n == g_flat) & m_n).to(torch.float32)
        k_s = ((g_s == g_flat) & m_s).to(torch.float32)
        k_w = ((g_w == g_flat) & m_w).to(torch.float32)
        k_e = ((g_e == g_flat) & m_e).to(torch.float32)
        s_card = 0.25 * (k_n + k_s + k_w + k_e)

        d_nw = ((g_nw != g_flat) & m_nw).to(torch.float32)
        d_ne = ((g_ne != g_flat) & m_ne).to(torch.float32)
        d_sw = ((g_sw != g_flat) & m_sw).to(torch.float32)
        d_se = ((g_se != g_flat) & m_se).to(torch.float32)
        d_diag = 0.25 * (d_nw + d_ne + d_sw + d_se)

        beta = (
            torch.clamp(F.softplus(beta_sharpness.to(torch.float32)), min=8.0, max=64.0)
            if beta_sharpness is not None
            else torch.tensor(36.0, dtype=torch.float32, device=device)
        )
        t_card = (
            torch.clamp(card_threshold.to(torch.float32), min=0.78, max=0.96)
            if card_threshold is not None
            else torch.tensor(0.875, dtype=torch.float32, device=device)
        )
        t_diag = (
            torch.clamp(diag_threshold.to(torch.float32), min=0.52, max=0.72)
            if diag_threshold is not None
            else torch.tensor(0.625, dtype=torch.float32, device=device)
        )

        gate_card = torch.sigmoid(beta * (s_card - t_card))
        gate_diag = torch.sigmoid(beta * (d_diag - t_diag))
        center_wavelet = is_fg * gate_card * gate_diag

        c_pad = F.pad(center_wavelet.unsqueeze(1), (1, 1, 1, 1), mode="constant", value=0.0).squeeze(1)
        arm_n = c_pad[:, :-2, 1:-1] * k_n
        arm_s = c_pad[:, 2:, 1:-1] * k_s
        arm_w = c_pad[:, 1:-1, :-2] * k_w
        arm_e = c_pad[:, 1:-1, 2:] * k_e

        full_wavelet = torch.maximum(
            center_wavelet,
            torch.maximum(torch.maximum(arm_n, arm_s), torch.maximum(arm_w, arm_e)),
        ) * is_fg

        return full_wavelet.view(*orig_batch, H, W)

    @classmethod
    def compute_morphological_plus_cross_mask(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates the D4-steerable radial-shell wavelet at canonical sharpness.
        """
        wavelet = cls.compute_steerable_d4_radial_wavelet(grid, mask)
        return (wavelet > 0.50).to(torch.float32) + (wavelet - wavelet.detach())

    @classmethod
    def _recover_cardinal_rays_and_eikonal(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            _,
            h_span,
            w_span,
            _,
            ey,
            ex,
            _,
            is_wall,
            is_seed,
        ) = cls._compute_full_cached(grid, mask, num_iterations=10)

        ell_n = torch.round(0.50 * (h_span * (1.0 + ey) + 1.0)).clamp_min(1.0)
        ell_s = torch.round(0.50 * (h_span * (1.0 - ey) + 1.0)).clamp_min(1.0)
        ell_w = torch.round(0.50 * (w_span * (1.0 + ex) + 1.0)).clamp_min(1.0)
        ell_e = torch.round(0.50 * (w_span * (1.0 - ex) + 1.0)).clamp_min(1.0)

        min_ray = torch.minimum(torch.minimum(ell_n, ell_s), torch.minimum(ell_w, ell_e))
        d_eikonal = F.relu(min_ray - 1.0) * (1.0 - is_wall) * mask.float().unsqueeze(-1)

        return (
            ell_n.squeeze(-1),
            ell_s.squeeze(-1),
            ell_w.squeeze(-1),
            ell_e.squeeze(-1),
            d_eikonal.squeeze(-1),
            is_wall.squeeze(-1),
            is_seed.squeeze(-1),
        )

    @classmethod
    def _extract_wall_and_seed_masks(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = cls._compute_full_cached(grid, mask, num_iterations=10)
        is_wall = res[7]
        is_seed = res[8]
        mask_f32 = mask.float().unsqueeze(-1)
        is_transparent = (1.0 - is_wall).clamp(0.0, 1.0) * mask_f32
        return is_transparent, is_wall, is_seed

    @classmethod
    def compute_finsler_chordal_bundle(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        num_iterations: int = 12,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            m_cav_bg,
            h_span,
            w_span,
            r_harm,
            ey,
            ex,
            seed_f,
            _,
            _,
        ) = cls._compute_full_cached(grid, mask, num_iterations=num_iterations)
        dipole_yx = torch.cat([ey, ex], dim=-1)
        return m_cav_bg, h_span, w_span, r_harm, dipole_yx, seed_f

    @classmethod
    def solve_cavity_potential(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        num_iterations: int = 12,
    ) -> torch.Tensor:
        mask_f32 = mask.float().unsqueeze(-1)
        m_cavity_bg, h_span, w_span, r_harmonic, _, _ = cls.compute_finsler_chordal_bundle(
            grid, mask, num_iterations=num_iterations
        )
        min_span = torch.minimum(h_span, w_span)
        span_depth = torch.tanh((min_span - 1.0) / 6.0) * m_cavity_bg
        harm_depth = torch.tanh(r_harmonic / 4.0) * m_cavity_bg
        return (0.50 * m_cavity_bg + 0.25 * span_depth + 0.25 * harm_depth) * mask_f32

    @classmethod
    def extract_color_structural_roles(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        support_inputs: Optional[torch.Tensor] = None,
        support_in_masks: Optional[torch.Tensor] = None,
        support_outputs: Optional[torch.Tensor] = None,
        support_out_masks: Optional[torch.Tensor] = None,
    ) -> ColorRoleState:
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        m_f32 = m_flat.float()
        active_h = m_f32.max(dim=-1)[0].sum(dim=-1).clamp_min(1.0)
        active_w = m_f32.max(dim=-2)[0].sum(dim=-1).clamp_min(1.0)
        active_area = m_f32.sum(dim=(-2, -1)).clamp_min(1.0)

        (
            m_cav_bg,
            _,
            _,
            _,
            _,
            _,
            _,
            is_wall,
            is_seed,
        ) = cls._compute_full_cached(g_flat, m_flat, num_iterations=10)

        y_idx = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1, 1)
        x_idx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W, 1)

        is_border_px = (
            (y_idx.squeeze(-1) == 0)
            | (y_idx.squeeze(-1) == (active_h.view(N, 1, 1) - 1.0))
            | (x_idx.squeeze(-1) == 0)
            | (x_idx.squeeze(-1) == (active_w.view(N, 1, 1) - 1.0))
        ) & m_flat
        border_total = is_border_px.float().sum(dim=(-2, -1)).clamp_min(1.0).unsqueeze(-1)

        is_c = (F.one_hot(g_flat.clamp(0, 9), num_classes=10).bool()) & m_flat.unsqueeze(-1)
        c_f32 = is_c.float()
        c_mass = c_f32.sum(dim=(1, 2))
        has_c = (c_mass > 0.5).float()

        r1_density = (c_mass / active_area.unsqueeze(-1)).clamp(0.0, 1.0)
        r2_border = ((c_f32 * is_border_px.float().unsqueeze(-1)).sum(dim=(1, 2)) / border_total).clamp(0.0, 1.0)

        row_fill = (c_f32.sum(dim=2) / active_w.view(N, 1, 1)).max(dim=1)[0]
        col_fill = (c_f32.sum(dim=1) / active_h.view(N, 1, 1)).max(dim=1)[0]
        r3_divider = torch.maximum(row_fill, col_fill).clamp(0.0, 1.0) * has_c

        ymin_c = torch.where(is_c, y_idx, torch.full_like(y_idx, float(H))).amin(dim=(1, 2))
        ymax_c = torch.where(is_c, y_idx, torch.full_like(y_idx, -1.0)).amax(dim=(1, 2))
        xmin_c = torch.where(is_c, x_idx, torch.full_like(x_idx, float(W))).amin(dim=(1, 2))
        xmax_c = torch.where(is_c, x_idx, torch.full_like(x_idx, -1.0)).amax(dim=(1, 2))

        h_box = (ymax_c - ymin_c + 1.0).clamp_min(1.0)
        w_box = (xmax_c - xmin_c + 1.0).clamp_min(1.0)
        box_area = h_box * w_box
        r4_rect_fill = ((c_mass / box_area) * has_c * (c_mass >= 4.0).float()).clamp(0.0, 1.0)

        seed_c = (c_f32 * is_seed).sum(dim=(1, 2)) / c_mass.clamp_min(1.0)
        r5_seed = (seed_c * has_c).clamp(0.0, 1.0)

        wall_c = (c_f32 * is_wall).sum(dim=(1, 2)) / c_mass.clamp_min(1.0)
        r6_wall = (wall_c * has_c).clamp(0.0, 1.0)

        cav_adj = (c_f32 * m_cav_bg).sum(dim=(1, 2)) / m_cav_bg.sum(dim=(1, 2)).clamp_min(1.0)
        r7_cavity = (cav_adj * has_c).clamp(0.0, 1.0)

        r8_aspect = (torch.minimum(h_box, w_box) / torch.maximum(h_box, w_box).clamp_min(1.0)) * has_c

        role_vectors = torch.stack(
            [r1_density, r2_border, r3_divider, r4_rect_fill, r5_seed, r6_wall, r7_cavity, r8_aspect],
            dim=-1,
        )

        act_h_exp = active_h.unsqueeze(-1)
        act_w_exp = active_w.unsqueeze(-1)
        is_interior_y = (ymin_c >= 1.0) & (ymax_c <= (act_h_exp - 2.0))
        is_interior_x = (xmin_c >= 1.0) & (xmax_c <= (act_w_exp - 2.0))
        is_thin_line = ((h_box == act_h_exp) & (w_box <= 2.0) & is_interior_x) | (
            (w_box == act_w_exp) & (h_box <= 2.0) & is_interior_y
        )

        div_mat = torch.where(
            (r3_divider > 0.95) & is_thin_line,
            r3_divider + (1.0 - r1_density),
            torch.full_like(r3_divider, -1e9),
        )
        occ_mat = torch.where(
            (r4_rect_fill > 0.98) & (h_box >= 2.0) & (w_box >= 2.0) & (r1_density < 0.50),
            r4_rect_fill + (1.0 / (c_mass + 1.0)),
            torch.full_like(r4_rect_fill, -1e9),
        )
        seed_mat = torch.where(
            (c_mass == 1.0) & (r5_seed > 0.95),
            r5_seed + 1.0,
            torch.full_like(r5_seed, -1e9),
        )

        div_mat[:, 0] = -1e9
        occ_mat[:, 0] = -1e9
        seed_mat[:, 0] = -1e9

        diode_sink = torch.zeros((N, 10), dtype=torch.float32, device=device)
        if (
            support_inputs is not None
            and support_in_masks is not None
            and support_outputs is not None
            and support_out_masks is not None
        ):
            B_s, K_s = support_inputs.shape[:2]
            s_in_oh = F.one_hot(support_inputs.clamp(0, 9), 10).bool() & support_in_masks.unsqueeze(-1)
            s_out_oh = F.one_hot(support_outputs.clamp(0, 9), 10).bool() & support_out_masks.unsqueeze(-1)
            s_in_has = s_in_oh.any(dim=(2, 3)).float()
            s_out_has = s_out_oh.any(dim=(2, 3)).float()
            vanished = ((s_in_has > 0.5) & (s_out_has < 0.5)).float().sum(dim=1) / s_in_has.sum(dim=1).clamp_min(1.0)
            if N == B_s * K_s:
                vanished = vanished.repeat_interleave(K_s, dim=0)
            diode_sink = (vanished > 0.80).float()
            diode_sink[:, 0] = 0.0

            occ_mat = occ_mat + (5.0 * diode_sink)
            div_mat = div_mat + (5.0 * diode_sink)

        max_div_val, max_div_col = torch.max(div_mat, dim=-1)
        divider_color = torch.where(max_div_val > 0.0, max_div_col, torch.full_like(max_div_col, -1))

        max_occ_val, max_occ_col = torch.max(occ_mat, dim=-1)
        occluder_color = torch.where(max_occ_val > 0.0, max_occ_col, torch.full_like(max_occ_col, -1))

        max_seed_val, max_seed_col = torch.max(seed_mat, dim=-1)
        seed_color = torch.where(max_seed_val > 0.0, max_seed_col, torch.full_like(max_seed_col, -1))

        return ColorRoleState(
            role_vectors=role_vectors.view(*orig_batch, 10, 8),
            diode_sink_mask=diode_sink.view(*orig_batch, 10),
            divider_color=divider_color.view(*orig_batch),
            occluder_color=occluder_color.view(*orig_batch),
            seed_color=seed_color.view(*orig_batch),
        )

    @classmethod
    def compute_bravais_lattice_periodicity(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
        max_period: int = 8,
    ) -> BravaisPeriodicityState:
        """
        Universal Two-Shell Punctured-Gibbs Bravais Lattice Resonator:
        Evaluates 1-shell and 2-shell punctured neighborhood inner products
        A_phase((y, x), (y + m*T_y, x + n*T_x)) via smooth Boltzmann-Gibbs weights.
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        g_pad1 = F.pad(g_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=-1.0).squeeze(1).to(torch.int64)
        m_pad1 = F.pad(m_flat.unsqueeze(1).float(), (1, 1, 1, 1), mode="constant", value=0.0).squeeze(1) > 0.5

        nb1_tokens = torch.stack(
            [g_pad1[:, (1 + dy) : (1 + dy + H), (1 + dx) : (1 + dx + W)] for dy, dx in cls.SHELL_1_DIRS],
            dim=-1,
        )
        nb1_valid = torch.stack(
            [m_pad1[:, (1 + dy) : (1 + dy + H), (1 + dx) : (1 + dx + W)] for dy, dx in cls.SHELL_1_DIRS],
            dim=-1,
        )

        g_pad2 = F.pad(g_flat.unsqueeze(1).float(), (2, 2, 2, 2), mode="constant", value=-1.0).squeeze(1).to(torch.int64)
        m_pad2 = F.pad(m_flat.unsqueeze(1).float(), (2, 2, 2, 2), mode="constant", value=0.0).squeeze(1) > 0.5

        nb2_tokens = torch.stack(
            [g_pad2[:, (2 + dy) : (2 + dy + H), (2 + dx) : (2 + dx + W)] for dy, dx in cls.SHELL_2_DIRS],
            dim=-1,
        )
        nb2_valid = torch.stack(
            [m_pad2[:, (2 + dy) : (2 + dy + H), (2 + dx) : (2 + dx + W)] for dy, dx in cls.SHELL_2_DIRS],
            dim=-1,
        )

        y_int = torch.arange(H, device=device, dtype=torch.int64).view(1, H, 1).expand(N, H, W)
        x_int = torch.arange(W, device=device, dtype=torch.int64).view(1, 1, W).expand(N, H, W)
        b_id = torch.arange(N, device=device, dtype=torch.int64).view(N, 1, 1).expand(N, H, W)

        active_h = m_flat.float().max(dim=-1)[0].sum(dim=-1).to(torch.int64).clamp_min(1).view(N, 1, 1)
        active_w = m_flat.float().max(dim=-2)[0].sum(dim=-1).to(torch.int64).clamp_min(1).view(N, 1, 1)
        is_interior = (
            (y_int >= 1)
            & (y_int < active_h - 1)
            & (x_int >= 1)
            & (x_int < active_w - 1)
            & m_flat
        )

        one_hot = F.one_hot(g_flat.clamp(0, 9), num_classes=10).to(torch.float32) * m_flat.float().unsqueeze(-1)

        mult_vec = torch.tensor([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64).view(1, 1, 1, 12)
        b_4d = b_id.unsqueeze(-1).expand(N, H, W, 12)
        y_4d = y_int.unsqueeze(-1).expand(N, H, W, 12)
        x_4d = x_int.unsqueeze(-1).expand(N, H, W, 12)

        best_score_x = torch.zeros((N, H, W), dtype=torch.float32, device=device)
        best_tx = torch.ones((N, H, W), dtype=torch.int64, device=device)
        best_coset_x = one_hot.clone()

        for tx in range(2, min(max_period + 1, W)):
            shifts_x = x_int.unsqueeze(-1) + (mult_vec * tx)
            in_bounds_x = (shifts_x >= 1) & (shifts_x < (active_w - 1).unsqueeze(-1)) & is_interior.unsqueeze(-1)
            xs_c = shifts_x.clamp(0, W - 1)

            s_nb1_x = nb1_tokens[b_4d, y_4d, xs_c]
            s_val1_x = nb1_valid[b_4d, y_4d, xs_c]
            j_val1_x = (nb1_valid.unsqueeze(-2) & s_val1_x & in_bounds_x.unsqueeze(-1)).float()
            a1_x = ((nb1_tokens.unsqueeze(-2) == s_nb1_x).float() * j_val1_x).sum(dim=-1) / j_val1_x.sum(dim=-1).clamp_min(1.0)

            s_nb2_x = nb2_tokens[b_4d, y_4d, xs_c]
            s_val2_x = nb2_valid[b_4d, y_4d, xs_c]
            j_val2_x = (nb2_valid.unsqueeze(-2) & s_val2_x & in_bounds_x.unsqueeze(-1)).float()
            a2_x = ((nb2_tokens.unsqueeze(-2) == s_nb2_x).float() * j_val2_x).sum(dim=-1) / j_val2_x.sum(dim=-1).clamp_min(1.0)

            a_phase_x = 0.65 * a1_x + 0.35 * a2_x
            w_gibbs_x = torch.sigmoid(14.0 * (a_phase_x - 0.58)) * in_bounds_x.float()

            w_sum_x = w_gibbs_x.sum(dim=-1)
            mean_a_x = (a_phase_x * w_gibbs_x).sum(dim=-1) / w_sum_x.clamp_min(1e-5)
            score_x = (mean_a_x * torch.tanh(w_sum_x * 2.5)) - (0.004 * float(tx))

            gathered_oh_x = one_hot[b_4d, y_4d, xs_c] * w_gibbs_x.unsqueeze(-1)
            vote_x = (one_hot * 0.12) + gathered_oh_x.sum(dim=-2)
            vote_x_norm = vote_x / vote_x.sum(dim=-1, keepdim=True).clamp_min(1e-5)

            better_x = score_x > (best_score_x + 1e-4)
            best_score_x = torch.where(better_x, score_x, best_score_x)
            best_tx = torch.where(better_x, torch.full_like(best_tx, tx), best_tx)
            best_coset_x = torch.where(better_x.unsqueeze(-1), vote_x_norm, best_coset_x)

        best_score_y = torch.zeros((N, H, W), dtype=torch.float32, device=device)
        best_ty = torch.ones((N, H, W), dtype=torch.int64, device=device)
        best_coset_y = one_hot.clone()

        for ty in range(2, min(max_period + 1, H)):
            shifts_y = y_int.unsqueeze(-1) + (mult_vec * ty)
            in_bounds_y = (shifts_y >= 1) & (shifts_y < (active_h - 1).unsqueeze(-1)) & is_interior.unsqueeze(-1)
            ys_c = shifts_y.clamp(0, H - 1)

            s_nb1_y = nb1_tokens[b_4d, ys_c, x_4d]
            s_val1_y = nb1_valid[b_4d, ys_c, x_4d]
            j_val1_y = (nb1_valid.unsqueeze(-2) & s_val1_y & in_bounds_y.unsqueeze(-1)).float()
            a1_y = ((nb1_tokens.unsqueeze(-2) == s_nb1_y).float() * j_val1_y).sum(dim=-1) / j_val1_y.sum(dim=-1).clamp_min(1.0)

            s_nb2_y = nb2_tokens[b_4d, ys_c, x_4d]
            s_val2_y = nb2_valid[b_4d, ys_c, x_4d]
            j_val2_y = (nb2_valid.unsqueeze(-2) & s_val2_y & in_bounds_y.unsqueeze(-1)).float()
            a2_y = ((nb2_tokens.unsqueeze(-2) == s_nb2_y).float() * j_val2_y).sum(dim=-1) / j_val2_y.sum(dim=-1).clamp_min(1.0)

            a_phase_y = 0.65 * a1_y + 0.35 * a2_y
            w_gibbs_y = torch.sigmoid(14.0 * (a_phase_y - 0.58)) * in_bounds_y.float()

            w_sum_y = w_gibbs_y.sum(dim=-1)
            mean_a_y = (a_phase_y * w_gibbs_y).sum(dim=-1) / w_sum_y.clamp_min(1e-5)
            score_y = (mean_a_y * torch.tanh(w_sum_y * 2.5)) - (0.004 * float(ty))

            gathered_oh_y = one_hot[b_4d, ys_c, x_4d] * w_gibbs_y.unsqueeze(-1)
            vote_y = (one_hot * 0.12) + gathered_oh_y.sum(dim=-2)
            vote_y_norm = vote_y / vote_y.sum(dim=-1, keepdim=True).clamp_min(1e-5)

            better_y = score_y > (best_score_y + 1e-4)
            best_score_y = torch.where(better_y, score_y, best_score_y)
            best_ty = torch.where(better_y, torch.full_like(best_ty, ty), best_ty)
            best_coset_y = torch.where(better_y.unsqueeze(-1), vote_y_norm, best_coset_y)

        choose_y = best_score_y >= best_score_x
        conf_max = torch.maximum(best_score_y, best_score_x) * is_interior.float()
        chosen_coset = torch.where(choose_y.unsqueeze(-1), best_coset_y, best_coset_x)

        heal_gate = (conf_max.pow(4)).clamp(0.0, 1.0).unsqueeze(-1)
        healed_probs = ((1.0 - heal_gate) * one_hot) + (heal_gate * chosen_coset)
        healed_probs = healed_probs / healed_probs.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        return BravaisPeriodicityState(
            period_y=best_ty.view(*orig_batch, H, W),
            period_x=best_tx.view(*orig_batch, H, W),
            period_confidence=conf_max.view(*orig_batch, H, W),
            healed_logits_10=healed_probs.view(*orig_batch, H, W, 10),
        )

    @classmethod
    def compute_eikonal_level_set_propagation(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> EikonalLevelSetState:
        """
        Pre-Eroded 3x3 Moore Chebyshev Eikonal Distance d_inf(y, x) & 48-Step 4-Connected
        Von Neumann Geodesic Propagator around non-convex rooms.
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        (
            ell_n,
            ell_s,
            ell_w,
            ell_e,
            _,
            is_wall,
            is_seed,
        ) = cls._recover_cardinal_rays_and_eikonal(g_flat, m_flat)

        alive_interior = ((is_wall < 0.5) & m_flat).float().unsqueeze(1)
        curr_shell = alive_interior
        d_eikonal_acc = torch.zeros_like(curr_shell)

        for _ in range(15):
            c_pad = F.pad(curr_shell, (1, 1, 1, 1), mode="constant", value=0.0)
            curr_shell = -F.max_pool2d(-c_pad, kernel_size=3, stride=1, padding=0) * alive_interior
            d_eikonal_acc = d_eikonal_acc + curr_shell
            if not bool((curr_shell > 0.5).any().item()):
                break

        d_eikonal = d_eikonal_acc.squeeze(1)

        one_hot = F.one_hot(g_flat.clamp(0, 9), num_classes=10).to(torch.float32) * m_flat.float().unsqueeze(-1)
        seed_mask = (is_seed > 0.5) & m_flat
        d_int = torch.round(d_eikonal).to(torch.int64).clamp(0, 15)

        lvl_indices = torch.arange(4, device=device, dtype=torch.int64).view(1, 1, 1, 4)
        is_lvl_seed = (seed_mask.unsqueeze(-1) & (d_int.unsqueeze(-1) == lvl_indices)).to(torch.float32)

        trans_mask = ((is_wall < 0.5) & m_flat).float().view(N, H, W, 1)
        field_40 = (is_lvl_seed.unsqueeze(-1) * one_hot.unsqueeze(-2)).reshape(N, H, W, 40) * trans_mask

        y_grid = torch.arange(H, device=device, dtype=torch.int64).view(1, H, 1).expand(N, H, W)
        x_grid = torch.arange(W, device=device, dtype=torch.int64).view(1, 1, W).expand(N, H, W)
        b_idx = torch.arange(N, device=device, dtype=torch.int64).view(N, 1, 1).expand(N, H, W)

        y0_cam = (y_grid - ell_n.to(torch.int64) + 1).clamp(0, H - 1)
        y1_cam = (y_grid + ell_s.to(torch.int64)).clamp(1, H)
        x0_cam = (x_grid - ell_w.to(torch.int64) + 1).clamp(0, W - 1)
        x1_cam = (x_grid + ell_e.to(torch.int64)).clamp(1, W)

        cx_pad = F.pad(field_40, (0, 0, 1, 0), mode="constant", value=0.0).cumsum(dim=2)
        field_40 = ((cx_pad[b_idx, y_grid, x1_cam, :] - cx_pad[b_idx, y_grid, x0_cam, :]) > 0.05).to(torch.float32) * trans_mask

        cy_pad = F.pad(field_40, (0, 0, 0, 0, 1, 0), mode="constant", value=0.0).cumsum(dim=1)
        field_40 = ((cy_pad[b_idx, y1_cam, x_grid, :] - cy_pad[b_idx, y0_cam, x_grid, :]) > 0.05).to(torch.float32) * trans_mask

        f_perm = field_40.permute(0, 3, 1, 2)
        t_perm = trans_mask.permute(0, 3, 1, 2)
        for _ in range(48):
            f_pad = F.pad(f_perm, (1, 1, 1, 1), mode="constant", value=0.0)
            f_card = torch.maximum(
                f_perm,
                torch.maximum(
                    torch.maximum(f_pad[:, :, :-2, 1:-1], f_pad[:, :, 2:, 1:-1]),
                    torch.maximum(f_pad[:, :, 1:-1, :-2], f_pad[:, :, 1:-1, 2:]),
                ),
            )
            f_perm = f_card * t_perm
        field_40 = f_perm.permute(0, 2, 3, 1)

        depth_color_table = field_40.view(N, H, W, 4, 10)
        depth_color_norm = depth_color_table / depth_color_table.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        depth_has_seed = (depth_color_table.sum(dim=-1) > 0.1).float()
        num_seeds_in_cam = depth_has_seed.sum(dim=-1).clamp_min(0.0)

        k_safe = torch.clamp_min(num_seeds_in_cam.to(torch.int64), 1)
        mod_idx = torch.remainder(d_int, k_safe).clamp(0, 3)

        propagated_col = depth_color_norm[b_idx, y_grid, x_grid, mod_idx, :]

        has_d0 = depth_has_seed[:, :, :, 0] > 0.5
        has_d1 = depth_has_seed[:, :, :, 1] > 0.5
        single_inner_seed = (num_seeds_in_cam == 1.0) & (~has_d0) & has_d1
        inner_col = depth_color_norm[:, :, :, 1, :]
        propagated_col = torch.where(
            (single_inner_seed & (d_int >= 1)).unsqueeze(-1),
            inner_col,
            torch.where(single_inner_seed.unsqueeze(-1), one_hot, propagated_col),
        )

        has_any_seed = (num_seeds_in_cam >= 1.0) & (is_wall < 0.5) & (propagated_col.sum(dim=-1) > 0.1)
        level_set_probs = torch.where(has_any_seed.unsqueeze(-1), propagated_col, one_hot)
        level_set_probs = level_set_probs / level_set_probs.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        return EikonalLevelSetState(
            wall_distance=d_eikonal.view(*orig_batch, H, W),
            level_set_probs_10=level_set_probs.view(*orig_batch, H, W, 10),
            chamber_seed_count=num_seeds_in_cam.view(*orig_batch, H, W),
        )


class DifferentiableSpectralGreenOperator(nn.Module):
    """
    Unified Trainable Spectral-Green RKHS Operator (Forward & Adjoint Backward Pass):
      1. Learnable D4-Steerable Radial-Shell Wavelet Filter Bank (beta_sharpness, card_thresh, diag_thresh).
      2. Learnable Gaussian-Bessel Dirichlet Cavity Harmonics (cavity_calibers, cavity_log_sigma).
      3. Closed-Form Woodbury-Tikhonov RKHS Resolvent solved via SpectralGreenWoodburyFunction
         with learnable Tikhonov regularizers (log_lambda_ridge) and prior anchor matrix (prior_anchor_logits).
    """

    def __init__(self) -> None:
        super().__init__()
        self.steerable_log_beta = nn.Parameter(torch.tensor(math.log(36.0), dtype=torch.float32))
        self.steerable_card_thresh = nn.Parameter(torch.tensor(0.875, dtype=torch.float32))
        self.steerable_diag_thresh = nn.Parameter(torch.tensor(0.625, dtype=torch.float32))

        self.cavity_calibers = nn.Parameter(torch.tensor([3.0, 5.0, 7.5], dtype=torch.float32))
        self.cavity_log_sigma = nn.Parameter(torch.tensor(math.log(0.55), dtype=torch.float32))

        self.log_lambda_ridge = nn.Parameter(torch.tensor([-1.3863, -0.6931, -1.3863], dtype=torch.float32))
        self.prior_anchor_logits = nn.Parameter(torch.zeros((NUM_TOPO_CHANNELS, 10), dtype=torch.float32))

    def compute_basis(
        self,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructs the 23-channel differentiable D4-steerable + Gaussian-Bessel cavity basis Psi_theta(X).
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        m_f32 = m_flat.float().unsqueeze(-1)

        (
            m_cav,
            h_span,
            w_span,
            _,
            dipole_yx,
            seed_f,
        ) = PoissonDirichletCavitySolver.compute_finsler_chordal_bundle(
            g_flat, m_flat, num_iterations=10
        )
        _, is_wall, is_seed = PoissonDirichletCavitySolver._extract_wall_and_seed_masks(g_flat, m_flat)

        is_zero_vacuum = (g_flat == 0).to(torch.float32).unsqueeze(-1)
        m_vac_cav = m_cav * is_zero_vacuum

        min_span = torch.minimum(h_span, w_span).clamp(1.0, 15.0)
        calibers = self.cavity_calibers.to(torch.float32).view(1, 1, 1, NUM_CHORDAL_MODES)
        sigma = torch.clamp(torch.exp(self.cavity_log_sigma.to(torch.float32)), min=0.25, max=2.50)
        gauss_logits = -0.50 * ((min_span - calibers) / sigma).pow(2)
        cheb_norm = F.softmax(gauss_logits, dim=-1)

        cavity_basis_3 = (m_vac_cav * cheb_norm) * m_f32
        open_weight = (1.0 - m_vac_cav) * m_f32

        chi_d4 = PoissonDirichletCavitySolver.compute_steerable_d4_radial_wavelet(
            g_flat,
            m_flat,
            beta_sharpness=self.steerable_log_beta,
            card_threshold=self.steerable_card_thresh,
            diag_threshold=self.steerable_diag_thresh,
        ).unsqueeze(-1)
        one_hot_10 = F.one_hot(g_flat.clamp(0, 9), num_classes=10).to(torch.float32) * open_weight

        standard_10 = one_hot_10 * (1.0 - chi_d4)
        steerable_10 = one_hot_10 * chi_d4

        basis_23 = torch.cat([standard_10, steerable_10, cavity_basis_3], dim=-1)
        cavity_mass_bins = cavity_basis_3.sum(dim=(-3, -2))

        chordal_6d = torch.cat(
            [
                m_cav,
                torch.log(h_span.clamp_min(1.0)) / 3.40,
                torch.log(w_span.clamp_min(1.0)) / 3.40,
                dipole_yx,
                seed_f,
            ],
            dim=-1,
        ) * m_f32

        skeleton_anchor = torch.clamp(is_wall + (is_seed * (seed_f > 0.5).float()), 0.0, 1.0)

        return (
            basis_23.view(*orig_batch, H, W, NUM_TOPO_CHANNELS),
            cavity_mass_bins.view(*orig_batch, NUM_CHORDAL_MODES),
            chordal_6d.view(*orig_batch, H, W, 6),
            skeleton_anchor.squeeze(-1).view(*orig_batch, H, W),
        )

    def _build_prior_and_lambda(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam_3 = F.softplus(self.log_lambda_ridge.to(torch.float32)).clamp(min=0.05, max=5.0)
        lam_std = lam_3[0].expand(10)
        lam_steer = lam_3[1].expand(10)
        lam_cav = lam_3[2].expand(NUM_CHORDAL_MODES)
        lam_23 = torch.cat([lam_std, lam_steer, lam_cav], dim=0).view(1, NUM_TOPO_CHANNELS, 1).expand(batch_size, -1, -1)

        eye_10 = torch.eye(10, dtype=torch.float32, device=device)
        cav_base = torch.zeros((NUM_CHORDAL_MODES, 10), dtype=torch.float32, device=device)
        cav_base[:, 0] = 1.0
        base_anchor = torch.cat([eye_10, eye_10, cav_base], dim=0)

        residual = 0.05 * torch.tanh(self.prior_anchor_logits.to(torch.float32))
        w_prior = F.relu(base_anchor + residual)
        w_prior = w_prior / w_prior.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        return lam_23.contiguous(), w_prior.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def solve_rkhs_transport(
        self,
        s_basis_23: torch.Tensor,
        s_in_masks: torch.Tensor,
        s_outputs: torch.Tensor,
        s_out_masks: torch.Tensor,
        alpha_q: torch.Tensor,
        alpha_s_loo: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves support-to-query and Leave-One-Out RKHS transport operators via
        SpectralGreenWoodburyFunction (exact analytical forward and implicit adjoint backward pass).
        """
        B, K, H, W, _ = s_basis_23.shape
        device = s_basis_23.device
        dev_type = "cuda" if s_basis_23.is_cuda else "cpu"

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            joint_m = (s_in_masks & s_out_masks).to(torch.float32).unsqueeze(-1)
            v_mask = (s_in_masks.any(dim=-1).any(dim=-1) & s_out_masks.any(dim=-1).any(dim=-1)).to(torch.float32)
            w_res = alpha_q.to(torch.float32).view(B, K, 1, 1, 1) * v_mask.view(B, K, 1, 1, 1)
            w_sum = w_res.sum(dim=1, keepdim=True).clamp_min(1e-6)
            w_norm = w_res / w_sum

            psi_s_q = s_basis_23.to(torch.float32).view(B, K * H * W, NUM_TOPO_CHANNELS)
            y_s_q = (F.one_hot(s_outputs.clamp(0, 9), num_classes=10).to(torch.float32) * joint_m).view(
                B, K * H * W, 10
            )
            m_s_q = (joint_m * (0.50 * w_norm + 0.50 * (v_mask / v_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)).view(B, K, 1, 1, 1))).view(
                B, K * H * W, 1
            )

            lam_b, w_prior_b = self._build_prior_and_lambda(B, device)
            eye_probe_b = torch.eye(NUM_TOPO_CHANNELS, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)

            _, w_star_q = apply_spectral_green_woodbury(
                psi_s=psi_s_q,
                y_s=y_s_q,
                m_s=m_s_q,
                psi_q=eye_probe_b,
                lambda_vec=lam_b,
                w_prior=w_prior_b,
            )
            w_star_q_pos = F.relu(w_star_q)
            t_rkhs_q = w_star_q_pos / w_star_q_pos.sum(dim=-1, keepdim=True).clamp_min(1e-5)

            std_anchor_q = t_rkhs_q[:, :10, :]
            steer_mass_q = (s_basis_23.to(torch.float32)[..., 10:20] * joint_m).sum(dim=(1, 2, 3)).unsqueeze(-1)
            steer_blend_q = torch.sigmoid(4.0 * (steer_mass_q - 0.50))
            t_steer_q = (steer_blend_q * t_rkhs_q[:, 10:20, :]) + ((1.0 - steer_blend_q) * std_anchor_q)
            t_rkhs_q = torch.cat([std_anchor_q, t_steer_q, t_rkhs_q[:, 20:, :]], dim=1)

            eye_k = torch.eye(K, dtype=torch.float32, device=device).unsqueeze(0)
            not_self = (1.0 - eye_k) * v_mask.unsqueeze(1)
            not_self_sum = not_self.sum(dim=-1, keepdim=True)
            w_loo = torch.where(
                not_self_sum > 1e-6,
                not_self / not_self_sum.clamp_min(1e-6),
                v_mask.unsqueeze(1) / v_mask.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1.0),
            ).to(torch.float32)

            alpha_loo_blend = 0.50 * alpha_s_loo.to(torch.float32) + 0.50 * w_loo
            m_s_loo = (
                alpha_loo_blend.view(B, K, K, 1, 1, 1) * joint_m.unsqueeze(1)
            ).reshape(B * K, K * H * W, 1)
            psi_s_loo = (
                s_basis_23.to(torch.float32).unsqueeze(1).expand(B, K, K, H, W, NUM_TOPO_CHANNELS)
            ).reshape(B * K, K * H * W, NUM_TOPO_CHANNELS)
            y_s_loo = (
                (F.one_hot(s_outputs.clamp(0, 9), num_classes=10).to(torch.float32) * joint_m)
                .unsqueeze(1)
                .expand(B, K, K, H, W, 10)
            ).reshape(B * K, K * H * W, 10)

            lam_bk, w_prior_bk = self._build_prior_and_lambda(B * K, device)
            eye_probe_bk = torch.eye(NUM_TOPO_CHANNELS, dtype=torch.float32, device=device).unsqueeze(0).expand(
                B * K, -1, -1
            )

            _, w_star_s = apply_spectral_green_woodbury(
                psi_s=psi_s_loo,
                y_s=y_s_loo,
                m_s=m_s_loo,
                psi_q=eye_probe_bk,
                lambda_vec=lam_bk,
                w_prior=w_prior_bk,
            )
            w_star_s_pos = F.relu(w_star_s)
            t_rkhs_s_loo = w_star_s_pos / w_star_s_pos.sum(dim=-1, keepdim=True).clamp_min(1e-5)

            std_anchor_s = t_rkhs_s_loo[:, :10, :]
            steer_mass_s = (
                psi_s_loo[:, :, 10:20] * m_s_loo
            ).sum(dim=1).unsqueeze(-1)
            steer_blend_s = torch.sigmoid(4.0 * (steer_mass_s - 0.50))
            t_steer_s = (steer_blend_s * t_rkhs_s_loo[:, 10:20, :]) + ((1.0 - steer_blend_s) * std_anchor_s)
            t_rkhs_s_loo = torch.cat([std_anchor_s, t_steer_s, t_rkhs_s_loo[:, 20:, :]], dim=1)

        return t_rkhs_q, t_rkhs_s_loo


class SpectralGraphTopology(nn.Module):
    """
    Comprehensive Differentiable Graph Laplacian, Eikonal Harmonic & Finsler-Chordal Vision Engine.
    """

    def __init__(
        self,
        num_eigenmodes: int = 8,
        chebyshev_order: int = 4,
        d_out: int = 128,
    ):
        super().__init__()
        self.num_eigenmodes = num_eigenmodes
        self.chebyshev_order = chebyshev_order
        self.d_out = d_out

        q_init, _ = torch.linalg.qr(torch.randn(num_eigenmodes, num_eigenmodes))
        self.register_buffer("probe_basis_q", q_init, persistent=True)

        raw_topo_channels = num_eigenmodes + 3

        self.swiglu_proj = SwiGLU(
            d_in=raw_topo_channels,
            d_out=d_out,
            alignment_multiple=32,
        )
        self.chordal_modulator = nn.Linear(6, d_out, bias=False)
        self.eikonal_harmonic_proj = nn.Linear(5, d_out, bias=False)
        self.norm = RMSNorm(d_out)

        self.latest_chordal_bundle: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

        nn.init.xavier_uniform_(self.chordal_modulator.weight, gain=0.25)
        nn.init.zeros_(self.eikonal_harmonic_proj.weight)

    def _generate_orthonormal_dct_probes(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        freq_pairs = (
            (0, 1), (1, 0), (1, 1), (0, 2),
            (2, 0), (1, 2), (2, 1), (2, 2),
        )

        y_idx = torch.arange(height, dtype=dtype, device=device).view(height, 1)
        x_idx = torch.arange(width, dtype=dtype, device=device).view(1, width)

        h_flt = float(max(1, height))
        w_flt = float(max(1, width))

        modes: List[torch.Tensor] = []
        for u, v in freq_pairs[: self.num_eigenmodes]:
            arg_y = (math.pi * float(u) * (2.0 * y_idx + 1.0)) / (2.0 * h_flt)
            arg_x = (math.pi * float(v) * (2.0 * x_idx + 1.0)) / (2.0 * w_flt)
            mode_2d = torch.cos(arg_y) * torch.cos(arg_x)
            modes.append(mode_2d)

        stacked_dct = torch.stack(modes, dim=-1)
        rotated_basis = torch.matmul(stacked_dct, cast(torch.Tensor, self.probe_basis_q))
        return rotated_basis.unsqueeze(0).expand(batch_size, height, width, self.num_eigenmodes)

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, H, W = grid.shape
        mask_f32 = mask.float().unsqueeze(-1)

        edge_weights, inv_sqrt_deg = StencilGraphLaplacianOps.compute_edge_weights_and_degree(
            grid, mask
        )

        v0 = self._generate_orthonormal_dct_probes(
            batch_size=B,
            height=H,
            width=W,
            device=grid.device,
            dtype=torch.float32,
        ) * mask_f32

        t_prev = v0
        t_curr = StencilGraphLaplacianOps.apply_normalized_laplacian(
            v0, edge_weights, inv_sqrt_deg, mask
        )

        chebyshev_accumulator = t_prev + t_curr

        for _ in range(2, self.chebyshev_order):
            l_t_curr = StencilGraphLaplacianOps.apply_normalized_laplacian(
                t_curr, edge_weights, inv_sqrt_deg, mask
            )
            t_next = (2.0 * l_t_curr - t_prev) * mask_f32
            chebyshev_accumulator = chebyshev_accumulator + t_next
            t_prev = t_curr
            t_curr = t_next

        norm_sq = torch.sum(chebyshev_accumulator * chebyshev_accumulator, dim=-1, keepdim=True)
        is_pos_norm = norm_sq > 0.0
        norm_factor = torch.where(
            is_pos_norm,
            torch.rsqrt(torch.clamp_min(norm_sq, DEFAULT_LAPLACIAN_EPS)),
            torch.zeros_like(norm_sq),
        )
        betti_0_modes = chebyshev_accumulator * norm_factor * mask_f32

        chordal_bundle = PoissonDirichletCavitySolver.compute_finsler_chordal_bundle(
            grid, mask, num_iterations=12
        )
        self.latest_chordal_bundle = chordal_bundle
        (
            m_cavity_bg,
            h_span,
            w_span,
            r_harmonic,
            dipole_yx,
            seed_field,
        ) = chordal_bundle

        _, _, _, _, d_eik, _, _ = PoissonDirichletCavitySolver._recover_cardinal_rays_and_eikonal(grid, mask)
        d_eik_4d = d_eik.unsqueeze(-1)
        eikonal_harmonics = torch.cat(
            [
                d_eik_4d / 15.0,
                torch.cos(math.pi * d_eik_4d),
                torch.sin(math.pi * d_eik_4d),
                torch.cos((2.0 * math.pi / 3.0) * d_eik_4d),
                torch.sin((2.0 * math.pi / 3.0) * d_eik_4d),
            ],
            dim=-1,
        ) * mask_f32

        min_span = torch.minimum(h_span, w_span)
        cavity_potential = (
            0.50 * m_cavity_bg
            + 0.25 * torch.tanh((min_span - 1.0) / 6.0) * m_cavity_bg
            + 0.25 * torch.tanh(r_harmonic / 4.0) * m_cavity_bg
        ) * mask_f32

        degree_field = torch.sum(edge_weights, dim=-1, keepdim=True)
        boundary_contrast = (degree_field < 4.0).float() * mask_f32
        euler_density = (1.0 - 0.25 * degree_field) * mask_f32

        raw_bundle = torch.cat(
            [betti_0_modes, cavity_potential, boundary_contrast, euler_density],
            dim=-1,
        )

        chordal_features = torch.cat(
            [
                m_cavity_bg,
                torch.log(h_span.clamp_min(1.0)) / 3.40,
                torch.log(w_span.clamp_min(1.0)) / 3.40,
                dipole_yx,
                seed_field,
            ],
            dim=-1,
        ) * mask_f32

        projected = self.swiglu_proj(raw_bundle)
        if projected.shape[-1] == self.chordal_modulator.out_features:
            projected = projected + self.chordal_modulator(chordal_features.to(projected.dtype))
        if projected.shape[-1] == self.eikonal_harmonic_proj.out_features:
            projected = projected + self.eikonal_harmonic_proj(eikonal_harmonics.to(projected.dtype))

        topo_embeddings = self.norm(projected) * mask_f32.to(projected.dtype)

        return topo_embeddings


def verify_spectral_laplacian_diagnostics() -> bool:
    """
    Diagnostic Verification Gate for spectral_laplacian.py (including finite-difference
    gradcheck verification of SpectralGreenWoodburyFunction forward & backward passes).
    """
    PoissonDirichletCavitySolver.clear_cache()
    torch.manual_seed(42)
    b, h, w, d = 2, 7, 7, 32

    psi_s_chk = torch.randn((1, 6, 4), dtype=torch.float64, requires_grad=True)
    y_s_chk = torch.rand((1, 6, 3), dtype=torch.float64, requires_grad=True)
    m_s_chk = torch.rand((1, 6, 1), dtype=torch.float64, requires_grad=True) + 0.2
    psi_q_chk = torch.randn((1, 5, 4), dtype=torch.float64, requires_grad=True)
    lam_chk = torch.rand((1, 4, 1), dtype=torch.float64, requires_grad=True) + 0.3
    w0_chk = torch.randn((1, 4, 3), dtype=torch.float64, requires_grad=True)

    grad_ok = torch.autograd.gradcheck(
        SpectralGreenWoodburyFunction.apply,
        (psi_s_chk, y_s_chk, m_s_chk, psi_q_chk, lam_chk, w0_chk),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    if not grad_ok:
        return False

    topo = SpectralGraphTopology(num_eigenmodes=8, chebyshev_order=4, d_out=d)

    grid = torch.zeros((b, h, w), dtype=torch.int64)
    grid[0, :, 3] = 4
    grid[0, 1:4, 0:2] = 8
    grid[0, 0, 5] = 5
    mask = torch.ones((b, h, w), dtype=torch.bool)

    out = topo(grid, mask)
    if out.shape != (b, h, w, d) or torch.isnan(out).any():
        return False

    roles = PoissonDirichletCavitySolver.extract_color_structural_roles(grid[:1], mask[:1])
    if int(roles.divider_color[0].item()) != 4:
        return False
    if int(roles.occluder_color[0].item()) != 8:
        return False
    if int(roles.seed_color[0].item()) != 5:
        return False

    cam_grid = torch.full((1, 15, 6), 3, dtype=torch.int64)
    for r in range(1, 14):
        cam_grid[0, r, 1:5] = 8
        if (r - 1) % 3 != 0:
            cam_grid[0, r, 2:4] = 2
    cam_grid[0, 12, 2] = 8
    cam_mask = torch.ones_like(cam_grid, dtype=torch.bool)
    bravais = PoissonDirichletCavitySolver.compute_bravais_lattice_periodicity(cam_grid, cam_mask, max_period=8)
    if int(torch.argmax(bravais.healed_logits_10[0, 12, 2]).item()) != 2:
        return False

    room_grid = torch.ones((1, 6, 9), dtype=torch.int64)
    room_grid[0, :, 4] = 2
    room_grid[0, 0, 0] = 7
    room_grid[0, 1, 1] = 8
    room_mask = torch.ones_like(room_grid, dtype=torch.bool)
    eik_state = PoissonDirichletCavitySolver.compute_eikonal_level_set_propagation(room_grid, room_mask)
    if float(eik_state.wall_distance[0, 2, 3].item()) != 0.0:
        return False
    pred_ring0 = int(torch.argmax(eik_state.level_set_probs_10[0, 2, 3]).item())
    pred_ring1 = int(torch.argmax(eik_state.level_set_probs_10[0, 2, 2]).item())
    if pred_ring0 != 7 or pred_ring1 != 8:
        return False

    plus_grid = torch.full((1, 5, 5), 2, dtype=torch.int64)
    plus_grid[0, 2, 1:4] = 4
    plus_grid[0, 1:4, 2] = 4
    plus_mask = torch.ones_like(plus_grid, dtype=torch.bool)
    green_op = DifferentiableSpectralGreenOperator()
    s_b23, _, _, _ = green_op.compute_basis(plus_grid.unsqueeze(1), plus_mask.unsqueeze(1))
    s_out_plus = plus_grid.clone()
    s_out_plus[plus_grid == 4] = 8
    alpha_q_test = torch.ones((1, 1), dtype=torch.float32)
    alpha_loo_test = torch.ones((1, 1, 1), dtype=torch.float32)
    t_q, _ = green_op.solve_rkhs_transport(
        s_basis_23=s_b23,
        s_in_masks=plus_mask.unsqueeze(1),
        s_outputs=s_out_plus.unsqueeze(1),
        s_out_masks=plus_mask.unsqueeze(1),
        alpha_q=alpha_q_test,
        alpha_s_loo=alpha_loo_test,
    )
    loss_green = (t_q[:, 14, 8] - 1.0).pow(2).sum() + (out ** 2).sum()
    loss_green.backward()
    if green_op.steerable_log_beta.grad is None or green_op.log_lambda_ridge.grad is None:
        return False
    if topo.swiglu_proj.proj_out.weight.grad is None:
        return False

    PoissonDirichletCavitySolver.clear_cache()
    return True
