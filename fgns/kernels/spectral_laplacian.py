"""
Location: // fgns_engine/fgns/kernels/spectral_laplacian.py
Description: Matrix-Free Chebyshev Graph Laplacian, Zero-Sync 8-Slot Memoized Native CUDA C++
             Finsler-Chordal Cavity Solver, Chamber-Local Bravais Lattice Coset Consensus (T_max=8),
             and Chebyshev Eikonal Wall-Distance Level-Set Propagator (FGNS v4.1.1).
Language: Python 3.12+ / PyTorch 2.5+ / Native CUDA C++17 sm_120 & sm_89 Aligned
Compliance: Zero-Hash Directives Strictly Enforced (Exclusively Docstrings for Comments)
"""

from __future__ import annotations

import math
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgns.kernels.cuda_fgns_ops import cuda_finsler_chordal_cavity_op
from fgns.primitives.safe_math import (
    safe_hyperbolic_sqrt,
    safe_reciprocal,
)
from fgns.primitives.swiglu import RMSNorm, SwiGLU
from fgns.types import BravaisPeriodicityState, ColorRoleState, EikonalLevelSetState

DEFAULT_LAPLACIAN_EPS: Final[float] = 1e-7


class StencilGraphLaplacianOps:
    """
    Matrix-free 2D stencil implementation of the normalized graph Laplacian.
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
        shifted = torch.zeros_like(x)

        src_y_start = max(0, -dy)
        src_y_end = min(H, H - dy)
        dst_y_start = max(0, dy)
        dst_y_end = min(H, H + dy)

        src_x_start = max(0, -dx)
        src_x_end = min(W, W - dx)
        dst_x_start = max(0, dx)
        dst_x_end = min(W, W + dx)

        if (src_y_end > src_y_start) and (src_x_end > src_x_start):
            shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = x[
                :, src_y_start:src_y_end, src_x_start:src_x_end, :
            ]

        return shifted

    @classmethod
    def compute_edge_weights_and_degree(
        cls, grid: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = grid.shape
        grid_expanded = grid.unsqueeze(-1)
        mask_f32 = mask.float().unsqueeze(-1)

        edge_weight_slices = []
        degree_accum = torch.zeros((B, H, W, 1), dtype=torch.float32, device=grid.device)

        for dy, dx, metric_weight in cls.DIRECTIONS:
            neighbor_grid = cls.shift_tensor_2d(grid_expanded, dy, dx)
            neighbor_mask = cls.shift_tensor_2d(mask_f32, dy, dx)

            same_color = (grid_expanded == neighbor_grid).float()
            valid_edge = mask_f32 * neighbor_mask

            weight = same_color * valid_edge * metric_weight
            edge_weight_slices.append(weight)
            degree_accum = degree_accum + weight

        edge_weights = torch.cat(edge_weight_slices, dim=-1)

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
        mask_f32 = mask.float().unsqueeze(-1)
        v_scaled = v * inv_sqrt_deg

        adjacency_v = torch.zeros_like(v)

        for idx, (dy, dx, _) in enumerate(cls.DIRECTIONS):
            w_dir = edge_weights[:, :, :, idx : idx + 1]
            shifted_v_scaled = cls.shift_tensor_2d(v_scaled, dy, dx)
            adjacency_v = adjacency_v + (w_dir * shifted_v_scaled)

        normalized_adjacency_v = adjacency_v * inv_sqrt_deg
        laplacian_v = (-normalized_adjacency_v) * mask_f32
        return laplacian_v


class PoissonDirichletCavitySolver:
    """
    Zero-Sync Multi-Slot Cached Native CUDA C++ Finsler-Chordal,
    Vectorized Color Structural Role, Chamber-Local Bravais Lattice (T_max=8),
    and Chebyshev Eikonal Level-Set Engine.
    """

    _BUNDLE_CACHE_DICT: Dict[
        Tuple[int, int, Tuple[int, ...], str, int, int],
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
    ] = {}

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
                return cls._BUNDLE_CACHE_DICT[key]

            res = cuda_finsler_chordal_cavity_op(grid, mask, num_iterations=num_iterations)
            if len(cls._BUNDLE_CACHE_DICT) >= 8:
                cls._BUNDLE_CACHE_DICT.clear()
            cls._BUNDLE_CACHE_DICT[key] = res
            return res

    @classmethod
    def _recover_cardinal_rays_and_eikonal(
        cls,
        grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recovers exact Beer-Lambert optical ray lengths (ell_N, ell_S, ell_W, ell_E)
        and the Chebyshev Eikonal Wall Distance:
            d_partial(y, x) = min(ell_N, ell_S, ell_W, ell_E) - 1
        in O(1) from the cached CUDA chordal bundle!
        """
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
        """
        Vectorized S10-Equivariant 8D Color Structural Role Extractor.
        """
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
            for c in range(1, 10):
                s_in_has = ((support_inputs == c) & support_in_masks).any(dim=-1).any(dim=-1).float()
                s_out_has = ((support_outputs == c) & support_out_masks).any(dim=-1).any(dim=-1).float()
                vanished = ((s_in_has > 0.5) & (s_out_has < 0.5)).float().sum(dim=-1) / (
                    s_in_has.sum(dim=-1).clamp_min(1.0)
                )
                if N == B_s * K_s:
                    vanished = vanished.repeat_interleave(K_s, dim=0)
                diode_sink[:, c] = (vanished > 0.80).float()

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
        Chamber-Local 2D Bravais Lattice Coset Consensus Engine (T in {2..8}).
        Uses exact Beer-Lambert optical ray lengths (ell_N, ell_S, ell_W, ell_E) so that
        periodic coset votes (y + m*T_y, x + n*T_x) stay strictly inside the same chamber
        [y_min_cam, y_max_cam] x [x_min_cam, x_max_cam] and never bleed across walls!
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
            _,
        ) = cls._recover_cardinal_rays_and_eikonal(g_flat, m_flat)

        y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(N, H, W)
        x_grid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(N, H, W)
        b_id = torch.arange(N, device=device, dtype=torch.int64).view(N, 1, 1).expand(N, H, W)
        y_int = y_grid.to(torch.int64)
        x_int = x_grid.to(torch.int64)

        y_min_cam = (y_grid - ell_n + 1.0).clamp(0.0, float(H - 1))
        y_max_cam = (y_grid + ell_s - 1.0).clamp(0.0, float(H - 1))
        x_min_cam = (x_grid - ell_w + 1.0).clamp(0.0, float(W - 1))
        x_max_cam = (x_grid + ell_e - 1.0).clamp(0.0, float(W - 1))

        is_interior = (is_wall < 0.5) & m_flat
        one_hot = F.one_hot(g_flat.clamp(0, 9), num_classes=10).to(torch.float32) * m_flat.float().unsqueeze(-1)

        best_score_x = torch.zeros((N, H, W), dtype=torch.float32, device=device)
        best_tx = torch.ones((N, H, W), dtype=torch.int64, device=device)
        best_coset_x = one_hot.clone()

        for tx in range(2, min(max_period + 1, W)):
            tx_f = float(tx)
            in_cam_r = (x_grid + tx_f <= x_max_cam) & is_interior
            xr_idx = (x_int + tx).clamp(0, W - 1)
            match_r = (g_flat == g_flat[b_id, y_int, xr_idx]) & in_cam_r & is_interior[b_id, y_int, xr_idx]

            cam_valid_x = in_cam_r.float().sum(dim=-1, keepdim=True)
            row_score = (match_r.float().sum(dim=-1, keepdim=True) / cam_valid_x.clamp_min(1.0)) * (cam_valid_x >= 3.0).float()
            row_score_3d = row_score.expand(N, H, W)

            vote_x = one_hot * 0.25
            for mult in (-4, -3, -2, -1, 1, 2, 3, 4):
                shift_x = mult * tx
                if abs(shift_x) < W:
                    xs_f = x_grid + float(shift_x)
                    valid_s = (xs_f >= x_min_cam) & (xs_f <= x_max_cam) & is_interior
                    xs_idx = (x_int + shift_x).clamp(0, W - 1)
                    valid_s = valid_s & is_interior[b_id, y_int, xs_idx]
                    vote_x = vote_x + (one_hot[b_id, y_int, xs_idx] * valid_s.float().unsqueeze(-1))

            vote_x_norm = vote_x / vote_x.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            better_x = row_score_3d > (best_score_x + 0.015)
            best_score_x = torch.where(better_x, row_score_3d, best_score_x)
            best_tx = torch.where(better_x, torch.full_like(best_tx, tx), best_tx)
            best_coset_x = torch.where(better_x.unsqueeze(-1), vote_x_norm, best_coset_x)

        best_score_y = torch.zeros((N, H, W), dtype=torch.float32, device=device)
        best_ty = torch.ones((N, H, W), dtype=torch.int64, device=device)
        best_coset_y = one_hot.clone()

        for ty in range(2, min(max_period + 1, H)):
            ty_f = float(ty)
            in_cam_d = (y_grid + ty_f <= y_max_cam) & is_interior
            yd_idx = (y_int + ty).clamp(0, H - 1)
            match_d = (g_flat == g_flat[b_id, yd_idx, x_int]) & in_cam_d & is_interior[b_id, yd_idx, x_int]

            cam_valid_y = in_cam_d.float().sum(dim=-2, keepdim=True)
            col_score = (match_d.float().sum(dim=-2, keepdim=True) / cam_valid_y.clamp_min(1.0)) * (cam_valid_y >= 3.0).float()
            col_score_3d = col_score.expand(N, H, W)

            vote_y = one_hot * 0.25
            for mult in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
                shift_y = mult * ty
                if abs(shift_y) < H:
                    ys_f = y_grid + float(shift_y)
                    valid_sy = (ys_f >= y_min_cam) & (ys_f <= y_max_cam) & is_interior
                    ys_idx = (y_int + shift_y).clamp(0, H - 1)
                    valid_sy = valid_sy & is_interior[b_id, ys_idx, x_int]
                    vote_y = vote_y + (one_hot[b_id, ys_idx, x_int] * valid_sy.float().unsqueeze(-1))

            vote_y_norm = vote_y / vote_y.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            better_y = col_score_3d > (best_score_y + 0.015)
            best_score_y = torch.where(better_y, col_score_3d, best_score_y)
            best_ty = torch.where(better_y, torch.full_like(best_ty, ty), best_ty)
            best_coset_y = torch.where(better_y.unsqueeze(-1), vote_y_norm, best_coset_y)

        choose_y = best_score_y >= best_score_x
        conf_max = torch.maximum(best_score_y, best_score_x) * is_interior.float()
        chosen_coset = torch.where(choose_y.unsqueeze(-1), best_coset_y, best_coset_x)

        heal_gate = F.relu((conf_max - 0.65) / 0.35).clamp(0.0, 1.0).unsqueeze(-1)
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
        Chebyshev Eikonal Wall-Distance Level-Set Propagator:
            d_partial(y, x) = min(ell_N, ell_S, ell_W, ell_E) - 1 in {0, 1, 2, ...}.
        Propagates isolated chamber seeds (is_seed == 1) along concentric depth contours
        d_partial(y, x) == d_partial(y_s, x_s) (mod K_room) inside each chamber!
        """
        orig_batch = grid.shape[:-2]
        H, W = grid.shape[-2], grid.shape[-1]
        device = grid.device

        g_flat = grid.reshape(-1, H, W)
        m_flat = mask.reshape(-1, H, W)
        N = g_flat.shape[0]

        (
            _,
            _,
            _,
            _,
            d_eikonal,
            is_wall,
            is_seed,
        ) = cls._recover_cardinal_rays_and_eikonal(g_flat, m_flat)

        one_hot = F.one_hot(g_flat.clamp(0, 9), num_classes=10).to(torch.float32) * m_flat.float().unsqueeze(-1)
        seed_mask = (is_seed > 0.5) & m_flat
        d_int = torch.round(d_eikonal).to(torch.int64).clamp(0, 15)

        depth_color_table = torch.zeros((N, H, W, 4, 10), dtype=torch.float32, device=device)
        depth_has_seed = torch.zeros((N, H, W, 4), dtype=torch.float32, device=device)

        for d_lvl in range(4):
            is_lvl_seed = (seed_mask & (d_int == d_lvl)).float().unsqueeze(-1)
            seed_col_lvl = one_hot * is_lvl_seed

            trans_mask = ((is_wall < 0.5) & m_flat).float().unsqueeze(-1)
            field_lvl = seed_col_lvl
            for _ in range(max(H, W)):
                f_pad = F.pad(field_lvl.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0)
                f_max = F.max_pool2d(f_pad, kernel_size=3, stride=1, padding=0).permute(0, 2, 3, 1)
                field_lvl = torch.maximum(field_lvl, f_max) * trans_mask

            depth_color_table[:, :, :, d_lvl, :] = field_lvl
            depth_has_seed[:, :, :, d_lvl] = (field_lvl.sum(dim=-1) > 0.1).float()

        num_seeds_in_cam = depth_has_seed.sum(dim=-1).clamp_min(0.0)

        k_safe = torch.clamp_min(num_seeds_in_cam.to(torch.int64), 1)
        mod_idx = torch.remainder(d_int, k_safe).clamp(0, 3)

        b_idx = torch.arange(N, device=device).view(N, 1, 1).expand(N, H, W)
        y_idx = torch.arange(H, device=device).view(1, H, 1).expand(N, H, W)
        x_idx = torch.arange(W, device=device).view(1, 1, W).expand(N, H, W)

        propagated_col = depth_color_table[b_idx, y_idx, x_idx, mod_idx, :]

        has_d0 = depth_has_seed[:, :, :, 0] > 0.5
        has_d1 = depth_has_seed[:, :, :, 1] > 0.5
        single_inner_seed = (num_seeds_in_cam == 1.0) & (~has_d0) & has_d1
        inner_col = depth_color_table[:, :, :, 1, :]
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
    Diagnostic Verification Gate for spectral_laplacian.py.
    """
    torch.manual_seed(42)
    b, h, w, d = 2, 7, 7, 32

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
        zig_x = 1 + ((r - 1) % 6 if ((r - 1) % 6) < 4 else 6 - ((r - 1) % 6))
        cam_grid[0, r, zig_x] = 1
    cam_grid[0, 7, 1] = 1
    cam_mask = torch.ones_like(cam_grid, dtype=torch.bool)
    bravais = PoissonDirichletCavitySolver.compute_bravais_lattice_periodicity(cam_grid, cam_mask, max_period=8)
    healed_cell = int(torch.argmax(bravais.healed_logits_10[0, 7, 1]).item())
    if healed_cell not in (1, 8):
        return False

    room_grid = torch.ones((1, 6, 6), dtype=torch.int64)
    room_grid[0, 0, 0] = 7
    room_grid[0, 1, 1] = 8
    room_mask = torch.ones_like(room_grid, dtype=torch.bool)
    eik_state = PoissonDirichletCavitySolver.compute_eikonal_level_set_propagation(room_grid, room_mask)
    pred_ring0 = int(torch.argmax(eik_state.level_set_probs_10[0, 0, 3]).item())
    pred_ring1 = int(torch.argmax(eik_state.level_set_probs_10[0, 1, 3]).item())
    pred_ring2 = int(torch.argmax(eik_state.level_set_probs_10[0, 2, 2]).item())
    if pred_ring0 != 7 or pred_ring1 != 8 or pred_ring2 != 7:
        return False

    loss = torch.sum(out ** 2)
    loss.backward()
    if topo.swiglu_proj.proj_out.weight.grad is None:
        return False

    return True
